import time
import datetime
from path import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchsnooper

import models

import custom_transforms
from utils import *
from datasets.sequence_folders import SequenceFolder

from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss, compute_errors
from loss_functions import compute_apc_mvs_e_loss
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
accumulation_steps = 1
best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)

class Net(object):
    def __init__(self, args):
        self.args = args
        self.prepare_info()
        self.prepare_data()
        self.BaseNet()
        self.prepare_optim()

    def prepare_info(self):
        timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
        save_path = Path(self.args.name)
        self.args.save_path = save_path/timestamp
        print('=> will save everything to {}'.format(self.args.save_path))
        self.args.save_path.makedirs_p()

        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = True

        self.training_writer = SummaryWriter(self.args.save_path)
        self.output_writers = []
        if self.args.log_output:
            for i in range(3):
                self.output_writers.append(SummaryWriter(self.args.save_path / 'valid' / str(i)))

    def prepare_data(self):
        # Data loading code
        normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                                std=[0.225, 0.225, 0.225])

        train_transform = custom_transforms.Compose([
            custom_transforms.RandomHorizontalFlip(),
            custom_transforms.RandomScaleCrop(),
            custom_transforms.ArrayToTensor(),
            normalize
        ])

        valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

        print("=> fetching scenes in '{}'".format(self.args.data))
        train_set = SequenceFolder(
            self.args.data,
            transform=train_transform,
            seed=self.args.seed,
            train=True,
            sequence_length=self.args.sequence_length
        )

        # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
        if self.args.with_gt:
            from datasets.validation_folders import ValidationSet
            val_set = ValidationSet(
                self.args.data,
                transform=valid_transform
            )
        else:
            val_set = SequenceFolder(
                self.args.data,
                transform=valid_transform,
                seed=self.args.seed,
                train=False,
                sequence_length=self.args.sequence_length
            )
        print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
        print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
        self.train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=self.args.batch_size, shuffle=True,
            num_workers=self.args.workers, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=self.args.batch_size, shuffle=False,
            num_workers=self.args.workers, pin_memory=True)

        if self.args.epoch_size == 0:
            self.args.epoch_size = len(self.train_loader)
    # TODO: BaseNet
    def BaseNet(self):
        global device
        args = self.args
        disp_net = models.DispResNet(args.resnet_layers, args.with_pretrain).to(device)
        pose_net = models.PoseResNet(18, args.with_pretrain).to(device)
        flow_net = models.PWCFlowNet().to(device)

        # load parameters
        if args.pretrained_disp:
            print("=> using pre-trained weights for DispResNet")
            weights = torch.load(args.pretrained_disp)
            disp_net.load_state_dict(weights['state_dict'], strict=False)

        if args.pretrained_pose:
            print("=> using pre-trained weights for PoseResNet")
            weights = torch.load(args.pretrained_pose)
            pose_net.load_state_dict(weights['state_dict'], strict=False)

        if args.pretrained_flow:
            print("=> using pre-trained weights for PWCFlowNet")
            weights = torch.load(args.pretrained_flow)
            flow_net.load_state_dict(weights['state_dict'], strict=False)

        self.disp_net = torch.nn.DataParallel(disp_net)
        self.pose_net = torch.nn.DataParallel(pose_net)
        self.flow_net = torch.nn.DataParallel(flow_net)

    def prepare_optim(self):
        print('=> setting adam solver')
        optim_params = [
            {'params': self.disp_net.parameters(), 'lr': self.args.lr},
            {'params': self.pose_net.parameters(), 'lr': self.args.lr}
        ]
        self.optimizer = torch.optim.Adam(optim_params,
                                     betas=(self.args.momentum, self.args.beta),
                                     weight_decay=self.args.weight_decay)


    def compute_loss(self):
        global n_iter, device, accumulation_steps
        end = time.time()
        self.logger.train_bar.update(0)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(precision=4)

        w1, w2, w3 = self.args.w1, self.args.w2, self.args.w3
        # switch to train mode
        self.disp_net.train()
        self.pose_net.train()
        self.flow_net.train()
        for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(self.train_loader):
            log_losses = i > 0 and n_iter % self.args.print_freq == 0

            # measure data loading time
            data_time.update(time.time() - end)
            tgt_img = tgt_img.to(device)
            ref_imgs = [img.to(device) for img in ref_imgs]
            intrinsics = intrinsics.to(device)
            # compute output
            tgt_depth, ref_depths = self.compute_depth(tgt_img, ref_imgs)
            # It 深度  Is 深度
            poses, poses_inv = self.compute_pose(tgt_img, ref_imgs)
            # It->Is   Is->It
            fwds, bwds = self.compute_flow(tgt_img, ref_imgs)

            loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                             poses, poses_inv, self.args.num_scales, self.args.with_ssim,
                                                             self.args.with_mask, self.args.with_auto_mask, self.args.padding_mode)

            loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

            loss_4, loss_5, loss_6 = compute_apc_mvs_e_loss(tgt_img, ref_imgs, intrinsics, fwds, bwds,
                                     tgt_depth, ref_depths, poses, poses_inv, self.args.num_scales)

            # loss1：L^M_P
            # Loss2：L_s
            # Loss3：L_GC
            # loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3
            loss = 1.0 * loss_4 + 0.1 * loss_5 + 0.5 * loss_6 + 0.1 * loss_2
            loss = loss / accumulation_steps

            if log_losses:
                self.training_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
                self.training_writer.add_scalar('disparity_smoothness_loss', loss_2.item(), n_iter)
                self.training_writer.add_scalar('geometry_consistency_loss', loss_3.item(), n_iter)
                self.training_writer.add_scalar('apc_loss', loss_4.item(), n_iter)
                self.training_writer.add_scalar('mvs_loss', loss_5.item(), n_iter)
                self.training_writer.add_scalar('e_loss', loss_6.item(), n_iter)
                self.training_writer.add_scalar('total_loss', loss.item(), n_iter)

            # record loss and EPE
            losses.update(loss.item(), self.args.batch_size)

            # compute gradient and do Adam step
            loss.backward()

            if (i+1)%accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.logger.train_bar.update(i + 1)
            if i % self.args.print_freq == 0:
                self.logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
            if i >= self.args.epoch_size - 1:
                break

            n_iter += 1

        return losses.avg[0]

    def train(self):
        global best_error, n_iter, device
        self.logger = TermLogger(n_epochs=self.args.epochs,
                                 train_size=min(len(self.train_loader),
                                                    self.args.epoch_size),
                                 valid_size=len(self.val_loader))
        self.logger.epoch_bar.start()
        for epoch in range(self.args.epochs):
            self.logger.epoch_bar.update(epoch)

            # train for one epoch
            self.logger.reset_train_bar()
            train_loss = self.compute_loss()
            self.logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))

            # evaluate on validation set
            self.logger.reset_valid_bar()
            if self.args.with_gt:
                errors, error_names = self.validate_with_gt(epoch)
            else:
                errors, error_names = self.validate_without_gt(epoch)
            error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
            self.logger.valid_writer.write(' * Avg {}'.format(error_string))

            for error, name in zip(errors, error_names):
                self.training_writer.add_scalar(name, error, epoch)

            # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
            decisive_error = errors[1]
            if best_error < 0:
                best_error = decisive_error

            # remember lowest error and save checkpoint
            is_best = decisive_error < best_error
            best_error = min(best_error, decisive_error)
            save_checkpoint(
                self.args.save_path, {
                    'epoch': epoch + 1,
                    'state_dict': self.disp_net.module.state_dict()
                }, {
                    'epoch': epoch + 1,
                    'state_dict': self.pose_net.module.state_dict()
                },
                is_best)

        self.logger.epoch_bar.finish()


    def compute_depth(self, tgt_img, ref_imgs):
        tgt_depth = [1/disp for disp in self.disp_net(tgt_img)]

        ref_depths = []
        for ref_img in ref_imgs:
            ref_depth = [1/disp for disp in self.disp_net(ref_img)]
            ref_depths.append(ref_depth)

        return tgt_depth, ref_depths

    def compute_pose(self, tgt_img, ref_imgs):
        poses = []
        poses_inv = []
        for ref_img in ref_imgs:
            poses.append(self.pose_net(tgt_img, ref_img))
            poses_inv.append(self.pose_net(ref_img, tgt_img))

        return poses, poses_inv

    def compute_flow(self, tgt_img, ref_imgs):
        fwd, bwd = [], []
        for ref_img in ref_imgs:
            fd, bd = self.flow_net(tgt_img, ref_img)
            fwd.append(fd)
            bwd.append(bd)
        return fwd, bwd

    @torch.no_grad()
    def validate_without_gt(self, epoch):
        global device
        batch_time = AverageMeter()
        losses = AverageMeter(i=4, precision=4)
        log_outputs = len(self.output_writers) > 0

        # switch to evaluate mode
        self.disp_net.eval()
        self.pose_net.eval()
        self.flow_net.eval()

        end = time.time()
        self.logger.valid_bar.update(0)
        for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(self.val_loader):
            tgt_img = tgt_img.to(device)
            ref_imgs = [img.to(device) for img in ref_imgs]
            intrinsics = intrinsics.to(device)
            intrinsics_inv = intrinsics_inv.to(device)

            # compute output
            tgt_depth = [1 / self.disp_net(tgt_img)]
            ref_depths = []
            for ref_img in ref_imgs:
                ref_depth = [1 / self.disp_net(ref_img)]
                ref_depths.append(ref_depth)

            if log_outputs and i < len(self.output_writers):
                if epoch == 0:
                    self.output_writers[i].add_image('val Input', tensor2array(tgt_img[0]), 0)

                self.output_writers[i].add_image('val Dispnet Output Normalized',
                                            tensor2array(1 / tgt_depth[0][0], max_value=None, colormap='magma'),
                                            epoch)
                self.output_writers[i].add_image('val Depth Output',
                                            tensor2array(tgt_depth[0][0], max_value=10),
                                            epoch)

            poses, poses_inv = self.compute_pose_with_inv(tgt_img, ref_imgs)

            loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                             poses, poses_inv, self.args.num_scales, self.args.with_ssim,
                                                             self.args.with_mask, False, self.args.padding_mode)

            loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

            loss_1 = loss_1.item()
            loss_2 = loss_2.item()
            loss_3 = loss_3.item()

            loss = loss_1
            losses.update([loss, loss_1, loss_2, loss_3])

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.logger.valid_bar.update(i + 1)
            if i % self.args.print_freq == 0:
                self.logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

        self.logger.valid_bar.update(len(self.val_loader))
        return losses.avg, ['Total loss', 'Photo loss', 'Smooth loss', 'Consistency loss']

    @torch.no_grad()
    def validate_with_gt(self, epoch):
        global device
        batch_time = AverageMeter()
        error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
        errors = AverageMeter(i=len(error_names))
        log_outputs = len(self.output_writers) > 0

        # switch to evaluate mode
        self.disp_net.eval()

        end = time.time()
        self.logger.valid_bar.update(0)
        for i, (tgt_img, depth) in enumerate(self.val_loader):
            tgt_img = tgt_img.to(device)
            depth = depth.to(device)

            # check gt
            if depth.nelement() == 0:
                continue

            # compute output
            output_disp = self.disp_net(tgt_img)
            output_depth = 1/output_disp[:, 0]

            if log_outputs and i < len(self.output_writers):
                if epoch == 0:
                    self.output_writers[i].add_image('Input', tgt_img[0], 0)
                    self.output_writers[i].add_image('val Input', tensor2array(tgt_img[0]), 0)
                    depth_to_show = depth[0]
                    self.output_writers[i].add_image('val target Depth',
                                                tensor2array(depth_to_show, max_value=10),
                                                epoch)
                    depth_to_show[depth_to_show == 0] = 1000
                    disp_to_show = (1/depth_to_show).clamp(0, 10)
                    self.output_writers[i].add_image('val target Disparity Normalized',
                                                tensor2array(disp_to_show, max_value=None, colormap='magma'),
                                                epoch)

                self.output_writers[i].add_image('val Dispnet Output Normalized',
                                            tensor2array(output_disp[0], max_value=None, colormap='magma'),
                                            epoch)
                self.output_writers[i].add_image('val Depth Output',
                                            tensor2array(output_depth[0], max_value=10),
                                            epoch)

            if depth.nelement() != output_depth.nelement():
                b, h, w = depth.size()
                output_depth = torch.nn.functional.interpolate(output_depth.unsqueeze(1), [h, w]).squeeze(1)

            errors.update(compute_errors(depth, output_depth, self.args.dataset))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            self.logger.valid_bar.update(i+1)
            if i % self.args.print_freq == 0:
                self.logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))
        self.logger.valid_bar.update(len(self.val_loader))
        return errors.avg, error_names

