CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=37682 tools/dist_test.sh projects/configs/CAPE-T/capet_r50_704x256_24ep_wocbgs_imagenet_pretrain.py work_dirs/capet_r50_704x256_24ep_wocbgs_imagenet_pretrain/latest.pth 8 --eval bbox