
python tools/video_infer_optical.py \
    --use_cuda \
    --bg ./tools/bg.jpeg \
    --model deeplabv3_plus \
    --net resnet50 \
    --checkpoint ./ckpts/resnet50_bce_deeplab.pth \
    --video ./seg_test2.mov \
    --output ./zzzz.avi
