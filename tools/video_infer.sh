#--bg ./tools/bg2.jpg \
#--net xception65 \

CUDA_VISIBLE_DEVICES="3" python tools/videoinfer.py \
    --use_cuda \
    --model hrnet \
    --checkpoint ./ckpts/hrnet_w18_small_v2.pth \
    --video ./seg_test1.mov \
    --output ./demob.avi
