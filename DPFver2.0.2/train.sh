TRAIN_SET=/disks/disk2/share/lijiaming/Dataset/monodepth/kitti_raw
LOG_ROOT=/disks/disk2/guohao/qjh_use/dpf
CUDA_VISIBLE_DEVICES=2,3 python train.py $TRAIN_SET \
--resnet-layers 18 \
--num-scales 1 \
-b4 --w1 1 --w2 0.1 --w3 0.5 --epoch-size 1000 --sequence-length 3 \
--with-ssim 1 \
--with-mask 1 \
--with-auto-mask 1 \
--with-pretrain 1 \
--log-output --with-gt \
--name $LOG_ROOT
--wd 0.005
