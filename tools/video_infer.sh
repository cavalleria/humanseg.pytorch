#--bg ./tools/bg2.jpg \
#--net xception65 \

CUDA_VISIBLE_DEVICES="3" python tools/videoinfer.py \
    --use_cuda \
    --model hrnet \
    --checkpoint ../models/HumanSeg/0603_072642/epoch78.pth \
    --video ./tools/seg_test.mov \
    --output ./demo.avi
