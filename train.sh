# multi-gpu
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=37686 tools/dist_train.sh projects/configs/CAPE-T/capet_r50_704x256_24ep_wocbgs_imagenet_pretrain.py 8 --work-dir work_dirs/capet_r50_704x256_24ep_wocbgs_imagenet_pretrain
