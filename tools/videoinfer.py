import cv2, torch, argparse
from time import time
import numpy as np
from torch.nn import functional as F
import sys,os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models import UNet
from models import DeepLabV3Plus
from models import HighResolutionNet

from utils import utils

def parse_args():
    parser = argparse.ArgumentParser(description="Arguments for the script")

    parser.add_argument('--use_cuda', action='store_true', default=False, help='Use GPU acceleration')
    parser.add_argument('--bg', type=str, default=None, help='Path to the background image file')
    parser.add_argument('--watch', action='store_true', default=False, help='Indicate show result live')
    parser.add_argument('--input_sz', type=int, default=320, help='Input size')
    parser.add_argument('--model', type=str, default='unet', help='model name')
    parser.add_argument('--net', type=str, default='resnet18', help='Path to the background image file')
    parser.add_argument('--checkpoint', type=str, default="", help='Path to the trained model file')
    parser.add_argument('--video', type=str, default="", help='Path to the input video')
    parser.add_argument('--output', type=str, default="", help='Path to the output video')

    return parser.parse_args()

def video_infer(args):
    cap = cv2.VideoCapture(args.video)
    _, frame = cap.read()
    H, W = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter(args.output, fourcc, 30, (W,H))
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Background
    if args.bg is not None:
        BACKGROUND = cv2.imread(args.bg)[...,::-1]
        BACKGROUND = cv2.resize(BACKGROUND, (W,H), interpolation=cv2.INTER_LINEAR)
        KERNEL_SZ = 25
        SIGMA = 0
    # Alpha transperency
    else:
        COLOR1 = [90, 140, 154]
        COLOR2 = [0, 0, 0]
    if args.model=='unet':
        model = UNet(backbone=args.net, num_classes=2, pretrained_backbone=None)
    elif args.model=='deeplabv3_plus':
        model = DeepLabV3Plus(backbone=args.net, num_classes=2, pretrained_backbone=None)
    elif args.model=='hrnet':
        model = HighResolutionNet(num_classes=2, pretrained_backbone=None)
    if args.use_cuda:
        model = model.cuda()
    trained_dict = torch.load(args.checkpoint, map_location="cpu")['state_dict']
    model.load_state_dict(trained_dict, strict=False)
    model.eval()

    while(cap.isOpened()):
        start_time = time()
        ret, frame = cap.read()
        if ret:
            image = frame[...,::-1]
            h, w = image.shape[:2]
            read_cam_time = time()

            # Predict mask
            X, pad_up, pad_left, h_new, w_new = utils.preprocessing(image, expected_size=args.input_sz, pad_value=0)
            preproc_time = time()
            with torch.no_grad():
                if args.use_cuda:
                    mask = model(X.cuda())
                    if mask.shape[1] != h_new:
                    mask = mask[..., pad_up: pad_up+h_new, pad_left: pad_left+w_new]
                    mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=True)
                    mask = F.softmax(mask, dim=1)
                    mask = mask[0,1,...].cpu().numpy()
                else:
                    mask = model(X)
                    mask = mask[..., pad_up: pad_up+h_new, pad_left: pad_left+w_new]
                    mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=True)
                    mask = F.softmax(mask, dim=1)
                    mask = mask[0,1,...].numpy()
            predict_time = time()

            # Draw result
            if args.bg is None:
                image_alpha = utils.draw_matting(image, mask)
                #image_alpha = utils.draw_transperency(image, mask, COLOR1, COLOR2)
            else:
                image_alpha = utils.draw_fore_to_back(image, mask, BACKGROUND, kernel_sz=KERNEL_SZ, sigma=SIGMA)
            draw_time = time()

            # Print runtime
            read = read_cam_time-start_time
            preproc = preproc_time-read_cam_time
            pred = predict_time-preproc_time
            draw = draw_time-predict_time
            total = read + preproc + pred + draw
            fps = 1 / pred
            print("read: %.3f [s]; preproc: %.3f [s]; pred: %.3f [s]; draw: %.3f [s]; total: %.3f [s]; fps: %.2f [Hz]" %
                (read, preproc, pred, draw, total, fps))
            # Wait for interupt
            cv2.putText(image_alpha, "%.2f [fps]" % (fps), (10, 50), font, 1.5, (0, 255, 0), 2, cv2.LINE_AA)
            out.write(image_alpha[..., ::-1])
            if args.watch:
                cv2.imshow('webcam', image_alpha[..., ::-1])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    out.release()

if __name__ == '__main__':
    args = parse_args()
    video_infer(args)
