from __future__ import division
import shutil
import numpy as np
import torch
import torch.nn.functional as F
import torchsnooper
from path import Path
import datetime
from collections import OrderedDict
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
# ----------------------------------------------------------------------
# -------global vars-----------------------


# ----------------------------------------------------------------------
# ---------------------------Utils Functions----------------------------
# -------real functions---------------------


# -------help functions---------------------
def compute_flow(coords, device=None):
    # coords:   [b, h, w, 2]
    # flow:     [b, 2, h, w]
    batch, height, width, _ = coords.size()
    coords = coords.permute(0, 3, 1, 2)
    base_coords = meshgrid(batch, height, width, is_homogenous=False, device=device)
    flow = coords - base_coords
    return flow

def compute_coords(flow, device=None):
    # flow:     [b, 2, h, w]
    # coords:   [b, h, w, 2]
    batch, _, height, width = flow.size()
    flow = flow.permute(0, 2, 3, 1)
    base_coords = meshgrid(batch, height, width, is_homogenous=False, device=device)
    coords = flow + base_coords
    return coords

def compute_occlusion_mask_from_flow(flow, tensor_size=None):
    # flow: [batch, 2, height, width]
    # mask: [batch, c, height, width]
    b, _, h, w = flow.size()
    if tensor_size:
        mask = torch.ones(tensor_size).to(flow.get_device())
    else:
        mask = torch.ones([b, 1, h, w]).to(flow.get_device())
    occ_mask = transformerFwd(mask.permute(0,2,3,1), flow.permute(0,2,3,1), out_size=[h,w]).permute(0,3,1,2)
    with torch.no_grad():
        occ_mask = torch.clamp(occ_mask, 0.0, 1.0)
    return occ_mask

def compute_simple_mask_from_coords(coords):
    # coords: [batch, height, width, 2]-> [0, width], [0, height]
    b, h, w, _ = coords.size()
    mask0 = coords[:, :, :, 0]>=0
    mask1 = coords[:, :, :, 0]<=w-1
    mask2 = coords[:, :, :, 1]>=0
    mask3 = coords[:, :, :, 1]<=h-1
    mask = mask0*mask1*mask2*mask3
    return mask.unsqueeze(1).float().to(coords.get_device())

def reproject(P, Point3D, is_trans=True, hw=None, expand=True):
    # P:        [b, 3, 4] Project Matrix
    # Point3D:  [b, n, 3/4] is_trans=True
    # Point3D:  [b, 3/4, n] is_trans=False
    # Point3D:  expand:True:3, False:4
    if is_trans:
        if expand:
            b, n, _ = Point3D.size()
            ones = torch.ones([b, n, 1]).float().to(Point3D.get_device())
            Point3D = torch.cat([Point3D, ones], dim=-1)
        Point2D = P.bmm(Point3D.transpose(1, 2))
    else:
        if expand:
            b, _, n = Point3D.size()
            ones = torch.ones([b, 1, n]).float().to(Point3D.get_device())
            Point3D = torch.cat([Point3D, ones], dim=1)
        Point2D = P.bmm(Point3D)
    # Point2D [b, 3, n]
    coords = (Point2D[:, :2, :] / (Point2D[:, 2, :].unsqueeze(1)+1e-12)).transpose(1, 2)
    depth = Point2D[:, 2, :].unsqueeze(1)
    # coords: [batch, n, 2]
    # depth:  [batch, 1, n]
    if hw:
        h, w = hw
        b, n, _ = coords.size()
        assert(n == h*w)
        coords = coords.reshape(b, h, w, 2)
        depth = depth.reshape(b, 1, h, w)
    return coords, depth

def compute_Kp(intrinsic, coords, is_homogenous=False, hw=None, is_plane=True):
    # K         [batch, 3, 3]
    # coords    [batch, height, width, 2]  is_homogenous=False
    # coords    [batch, height, width, 3]  is_homogenous=True
    # if  hw    [batch, n, 2/3]
    coords = coords.to(intrinsic.get_device())
    if hw:
        b, n, _ = coords.size()
        if not is_homogenous:
            ones = torch.ones([b, n, 1]).float().to(intrinsic.get_device())
            coords = torch.cat([coords, ones], dim=-1)
        coords = coords.permute(0, 2, 1)    # [b, 3, n]
    else:
        b, h, w, _ = coords.size()
        if not is_homogenous:
            ones = torch.ones([b, h, w, 1]).float().to(intrinsic.get_device())
            coords = torch.cat([coords, ones], dim=-1)
        coords = coords.permute(0, 3, 1, 2).reshape([b, 3, h*w])    # [b, 3, h*w]
    new_coords = torch.matmul(intrinsic.inverse(), coords)
    if is_plane:
        return new_coords
    else:
        if hw:
            h, w = hw
        return new_coords.reshape([b, 3, h, w])

def compute_DKp(depth, intrinsic, coords, is_plane=True):
    # depth     [b, 1, h, w]
    # K         [b, 3, 3]
    # coords    [b, h, w, 2]
    # return    [b, 3, h*w] -> Point3D
    b, h, w, _ = coords.size()
    new_coords = compute_Kp(intrinsic, coords) * depth.reshape([b, 1, h*w])
    if is_plane:
        return new_coords
    else:
        return new_coords.reshape([b, 3, h, w])

def trans2other(Point3D, pose, expand=True):
    # Point3D   [b, 3/4, n] expand:True:3, False:4
    # pose      [b, 6]
    posemat = pose_vec2mat(pose)    # [b, 3, 4]
    if expand:
        b, _, n = Point3D.size()
        ones = torch.ones([b, 1, n]).float().to(Point3D.get_device())
        Point3D = torch.cat([Point3D, ones], dim=1)
    # Point3D [b, 4, n]
    return posemat @ Point3D

def compute_RT(PoseMat):
    # input [batch, 3, 4]
    rot, tr = PoseMat[:, :, :3], PoseMat[:, :, -1]
    # rot [batch, 3, 3]
    # tr  [batch, 3]
    tx = tr[:, 0].unsqueeze(-1)
    ty = tr[:, 1].unsqueeze(-1)
    tz = tr[:, 2].unsqueeze(-1)
    zero = torch.zeros_like(tx).float().to(tx.get_device())
    Tx = torch.cat([zero, -tz, ty, tz, zero, -tx, -ty, tx, zero], dim=1).reshape([-1, 3, 3])
    RT = rot @ Tx
    return RT

# TODO: epipolar function from GLNet
def compute_Epipolar(Fcoords, intrinsic, pose):
    # Fcoords:  [b, h, w, 2]
    # K:        [b, 3, 3]
    # pose:     [b, 6]
    b, h, w, _ = Fcoords.size()
    Bcoords = meshgrid(b, h, w, device=Fcoords.get_device())
    BKp = compute_Kp(intrinsic, Bcoords, is_homogenous=False, is_plane=True)
    FKp = compute_Kp(intrinsic, Fcoords, is_homogenous=False, is_plane=True)
    posemat = pose_vec2mat(pose)
    RS = compute_RT(posemat)
    RSFKp = RS @ FKp        # [b, 3, h*w]
    E = torch.sum(BKp * RSFKp, dim=1).reshape([b, h, w])
    return E







# TODO: inverse_warp2 function from SC
def inverse_warp2(img, depth, ref_depth, pose, intrinsics, padding_mode='zeros'):
    """
        Inverse warp a source image to the target image plane.
        Args:
            img: the source image (where to sample pixels) -- [B, 3, H, W]
            depth: depth map of the target image -- [B, 1, H, W]
            ref_depth: the source depth map (where to sample depth) -- [B, 1, H, W]
            pose: 6DoF pose parameters from target to source -- [B, 6]
            intrinsics: camera intrinsic matrix -- [B, 3, 3]
        Returns:
            I'a:    projected_img:      Source image warped to the target image plane
            V:      valid_mask:         Float array indicating point validity
            D'b:    projected_depth:    sampled depth from source image
            Dab:    computed_depth:     computed depth of source image using the target depth
    """
    check_sizes(img, 'img', 'B3HW')
    check_sizes(depth, 'depth', 'B1HW')
    check_sizes(ref_depth, 'ref_depth', 'B1HW')
    check_sizes(pose, 'pose', 'B6')
    check_sizes(intrinsics, 'intrinsics', 'B33')
    b, _, h, w = img.size()
    base_coords = meshgrid(b, h, w) # [b, h, w, 2]
    points3d_a = compute_DKp(depth, intrinsics, base_coords, is_plane=True) # [b, 3, h*w]
    posemat = pose_vec2mat(pose)
    P = intrinsics @ posemat
    Pcoords, computed_depth = reproject(P, points3d_a, is_trans=False, expand=True, hw=[h, w])
    projected_img = grid_sample(img, Pcoords, padding_mode=padding_mode)
    valid_mask = compute_simple_mask_from_coords(Pcoords)
    projected_depth = grid_sample(ref_depth, Pcoords, padding_mode=padding_mode)
    return projected_img, valid_mask, projected_depth, computed_depth












# ----------------------------------------------------------------------
# ---------------------------tool Functions-----------------------------
# -------help functions----------------------
def euler2mat(angle):
    """Convert euler angles to rotation matrix.
     Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
    Args:
        angle: rotation angle along 3 axis (in radians) -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, 3, 3]
    """
    B = angle.size(0)
    x, y, z = angle[:, 0], angle[:, 1], angle[:, 2]

    cosz = torch.cos(z)
    sinz = torch.sin(z)

    zeros = z.detach()*0
    ones = zeros.detach()+1
    zmat = torch.stack([cosz, -sinz, zeros,
                        sinz,  cosz, zeros,
                        zeros, zeros,  ones], dim=1).reshape(B, 3, 3)

    cosy = torch.cos(y)
    siny = torch.sin(y)

    ymat = torch.stack([cosy, zeros,  siny,
                        zeros,  ones, zeros,
                        -siny, zeros,  cosy], dim=1).reshape(B, 3, 3)

    cosx = torch.cos(x)
    sinx = torch.sin(x)

    xmat = torch.stack([ones, zeros, zeros,
                        zeros,  cosx, -sinx,
                        zeros,  sinx,  cosx], dim=1).reshape(B, 3, 3)

    rotMat = xmat @ ymat @ zmat
    return rotMat

def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourht is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach()*0 + 1, quat], dim=1)
    norm_quat = norm_quat/norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:,
                                            1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
                          2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
                          2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(B, 3, 3)
    return rotMat

def pose_vec2mat(vec, rotation_mode='euler'):
    """
    Convert 6DoF parameters to transformation matrix.
    Args:s
        vec: 6DoF parameters in the order of tx, ty, tz, rx, ry, rz -- [B, 6]
    Returns:
        A transformation matrix -- [B, 3, 4]
    """
    translation = vec[:, :3].unsqueeze(-1)  # [B, 3, 1]
    rot = vec[:, 3:]
    if rotation_mode == 'euler':
        rot_mat = euler2mat(rot)  # [B, 3, 3]
    elif rotation_mode == 'quat':
        rot_mat = quat2mat(rot)  # [B, 3, 3]
    transform_mat = torch.cat([rot_mat, translation], dim=2)  # [B, 3, 4]
    return transform_mat


def check_sizes(input, input_name, expected):
    condition = [input.ndimension() == len(expected)]
    for i, size in enumerate(expected):
        if size.isdigit():
            condition.append(input.size(i) == int(size))
    assert (all(condition)), "wrong size for {}, expected {}, got  {}".format(
        input_name, 'x'.join(expected), list(input.size()))

def meshgrid(batch, height, width, is_homogenous=False, device=None):
    xx, yy = np.meshgrid(np.arange(0, width), np.arange(0, height))
    if is_homogenous:
        ones = np.ones([height, width])
        meshgrid = np.stack([xx, yy, ones], axis=-1)   # [h, w, 3]
    else:
        meshgrid = np.stack([xx, yy], axis=-1)         # [h, w, 2]
    meshgrid = torch.from_numpy(meshgrid)   # [h, w, 2/3]
    if device != None:
        meshgrid = meshgrid.float().to(device).unsqueeze(0).repeat(batch, 1, 1, 1)  # [b, h, w, 2/3]
    else:
        meshgrid = meshgrid.float().unsqueeze(0).repeat(batch, 1, 1, 1)             # [b, h, w, 2/3]
    return meshgrid

def grid_sample(img, coords, padding_mode='zeros', mode='bilinear'):
    # img       [b, c, h, w]
    # coords    [b, h, w, 2]
    b, _, h, w = img.size()
    coords_nor = torch.cat([2.0*coords[:, :, :, 0].unsqueeze(-1)/(w-1.0)-1.0,
                            2.0*coords[:, :, :, 1].unsqueeze(-1)/(h-1.0)-1.0], dim=-1)
    sample_img = F.grid_sample(img, coords_nor, mode=mode, padding_mode=padding_mode)
    return sample_img

# -------produce images----------------------
def high_res_colormap(low_res_cmap, resolution=1000, max_value=1):
    # Construct the list colormap, with interpolated values for higer resolution
    # For a linear segmented colormap, you can just specify the number of point in
    # cm.get_cmap(name, lutsize) with the parameter lutsize
    x = np.linspace(0, 1, low_res_cmap.N)
    low_res = low_res_cmap(x)
    new_x = np.linspace(0, max_value, resolution)
    high_res = np.stack([np.interp(new_x, x, low_res[:, i])
                         for i in range(low_res.shape[1])], axis=1)
    return ListedColormap(high_res)


def opencv_rainbow(resolution=1000):
    # Construct the opencv equivalent of Rainbow
    opencv_rainbow_data = (
        (0.000, (1.00, 0.00, 0.00)),
        (0.400, (1.00, 1.00, 0.00)),
        (0.600, (0.00, 1.00, 0.00)),
        (0.800, (0.00, 0.00, 1.00)),
        (1.000, (0.60, 0.00, 1.00))
    )

    return LinearSegmentedColormap.from_list('opencv_rainbow', opencv_rainbow_data, resolution)


COLORMAPS = {'rainbow': opencv_rainbow(),
             'magma': high_res_colormap(cm.get_cmap('magma')),
             'bone': cm.get_cmap('bone', 10000)}


def tensor2array(tensor, max_value=None, colormap='rainbow'):
    tensor = tensor.detach().cpu()
    if max_value is None:
        max_value = tensor.max().item()
    if tensor.ndimension() == 2 or tensor.size(0) == 1:
        norm_array = tensor.squeeze().numpy()/max_value
        array = COLORMAPS[colormap](norm_array).astype(np.float32)
        array = array.transpose(2, 0, 1)

    elif tensor.ndimension() == 3:
        assert(tensor.size(0) == 3)
        array = 0.5 + tensor.numpy()*0.5
    return array


def save_checkpoint(save_path, dispnet_state, exp_pose_state, is_best, filename='checkpoint.pth.tar'):
    file_prefixes = ['dispnet', 'exp_pose']
    states = [dispnet_state, exp_pose_state]
    for (prefix, state) in zip(file_prefixes, states):
        torch.save(state, save_path/'{}_{}'.format(prefix, filename))

    if is_best:
        for prefix in file_prefixes:
            shutil.copyfile(save_path/'{}_{}'.format(prefix, filename),
                            save_path/'{}_model_best.pth.tar'.format(prefix))

# -------from other codes--------------------

# get occlusion_mask from flow
def transformerFwd(U,
                   flo,
                   out_size,
                   name='SpatialTransformerFwd'):
    """Forward Warping Layer described in
    'Occlusion Aware Unsupervised Learning of Optical Flow by Yang Wang et al'

    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    flo: float
        The optical flow used for forward warping
        having the shape of [num_batch, height, width, 2].
    backprop: boolean
        Indicates whether to back-propagate through forward warping layer
    out_size: tuple of two ints
        The size of the output of the network (height, width)
    """

    def _repeat(x, n_repeats):
        rep = torch.ones(size=[n_repeats], dtype=torch.long).unsqueeze(1).transpose(1,0)
        x = x.view([-1,1]).mm(rep)
        return x.view([-1]).int()

    def _interpolate(im, x, y, out_size):
        # constants
        num_batch, height, width, channels = im.shape[0], im.shape[1], im.shape[2], im.shape[3]
        out_height = out_size[0]
        out_width = out_size[1]
        max_y = int(height - 1)
        max_x = int(width - 1)

        # scale indices from [-1, 1] to [0, width/height]
        x = (x + 1.0) * (width - 1.0) / 2.0
        y = (y + 1.0) * (height - 1.0) / 2.0

        # do sampling
        x0 = (torch.floor(x)).int()
        x1 = x0 + 1
        y0 = (torch.floor(y)).int()
        y1 = y0 + 1

        x0_c = torch.clamp(x0, 0, max_x)
        x1_c = torch.clamp(x1, 0, max_x)
        y0_c = torch.clamp(y0, 0, max_y)
        y1_c = torch.clamp(y1, 0, max_y)

        dim2 = width
        dim1 = width * height
        base = _repeat(torch.arange(0, num_batch) * dim1, out_height * out_width).to(im.get_device())

        base_y0 = base + y0_c * dim2
        base_y1 = base + y1_c * dim2
        idx_a = base_y0 + x0_c
        idx_b = base_y1 + x0_c
        idx_c = base_y0 + x1_c
        idx_d = base_y1 + x1_c

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = im.view([-1, channels])
        im_flat = im_flat.float()

        # and finally calculate interpolated values
        x0_f = x0.float()
        x1_f = x1.float()
        y0_f = y0.float()
        y1_f = y1.float()
        wa = ((x1_f - x) * (y1_f - y)).unsqueeze(1)
        wb = ((x1_f - x) * (y - y0_f)).unsqueeze(1)
        wc = ((x - x0_f) * (y1_f - y)).unsqueeze(1)
        wd = ((x - x0_f) * (y - y0_f)).unsqueeze(1)

        zerof = torch.zeros_like(wa)
        wa = torch.where(
            (torch.eq(x0_c, x0) & torch.eq(y0_c, y0)).unsqueeze(1), wa, zerof)
        wb = torch.where(
            (torch.eq(x0_c, x0) & torch.eq(y1_c, y1)).unsqueeze(1), wb, zerof)
        wc = torch.where(
            (torch.eq(x1_c, x1) & torch.eq(y0_c, y0)).unsqueeze(1), wc, zerof)
        wd = torch.where(
            (torch.eq(x1_c, x1) & torch.eq(y1_c, y1)).unsqueeze(1), wd, zerof)

        zeros = torch.zeros(
            size=[
                int(num_batch) * int(height) *
                int(width), int(channels)
            ],
            dtype=torch.float)
        output = zeros.to(im.get_device())
        output = output.scatter_add(dim=0, index=idx_a.long().unsqueeze(1).repeat(1,channels), src=im_flat * wa)
        output = output.scatter_add(dim=0, index=idx_b.long().unsqueeze(1).repeat(1,channels), src=im_flat * wb)
        output = output.scatter_add(dim=0, index=idx_c.long().unsqueeze(1).repeat(1,channels), src=im_flat * wc)
        output = output.scatter_add(dim=0, index=idx_d.long().unsqueeze(1).repeat(1,channels), src=im_flat * wd)

        return output

    def _meshgrid(height, width):
        # This should be equivalent to:
        x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
                                 np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        return torch.from_numpy(x_t).float(), torch.from_numpy(y_t).float()

    def _transform(flo, input_dim, out_size):
        num_batch, height, width, num_channels = input_dim.shape[0:4]

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        height_f = float(height)
        width_f = float(width)
        out_height = out_size[0]
        out_width = out_size[1]
        x_s, y_s = _meshgrid(out_height, out_width)
        x_s = x_s.to(flo.get_device()).unsqueeze(0)
        x_s = x_s.repeat([num_batch, 1, 1])

        y_s = y_s.to(flo.get_device()).unsqueeze(0)
        y_s =y_s.repeat([num_batch, 1, 1])

        x_t = x_s + flo[:, :, :, 0] / ((out_width - 1.0) / 2.0)
        y_t = y_s + flo[:, :, :, 1] / ((out_height - 1.0) / 2.0)

        x_t_flat = x_t.view([-1])
        y_t_flat = y_t.view([-1])

        input_transformed = _interpolate(input_dim, x_t_flat, y_t_flat,
                                            out_size)

        output = input_transformed.view([num_batch, out_height, out_width, num_channels])
        return output

    #out_size = int(out_size)
    output = _transform(flo, U, out_size)
    return output