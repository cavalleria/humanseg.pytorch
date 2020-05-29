import cv2, torch, argparse
from time import time
import numpy as np
from torch.nn import functional as F

from models import UNet
from models import DeepLabV3Plus

from utils import utils
from utils.postprocess import postprocess, threshold_mask

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

	fps = cap.get(cv2.CAP_PROP_FPS)
	out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (W, H))

	# Background
	if args.bg is not None:
		BACKGROUND = cv2.imread(args.bg)[...,::-1]
		BACKGROUND = cv2.resize(BACKGROUND, (W,H), interpolation=cv2.INTER_LINEAR)
		KERNEL_SZ = 25
		SIGMA = 0
	# Alpha transperency
	else:
		COLOR1 = [255, 0, 0]
		COLOR2 = [0, 0, 255]
	if args.model=='unet':
		model = UNet(backbone=args.net, num_classes=2, pretrained_backbone=None)
	elif args.model=='deeplabv3_plus':
		model = DeepLabV3Plus(backbone=args.net, num_classes=2,pretrained_backbone=None)
	if args.use_cuda:
		model = model.cuda()
	trained_dict = torch.load(args.checkpoint, map_location="cpu")['state_dict']
	model.load_state_dict(trained_dict, strict=False)
	model.eval()

	if W > H:
		w_new = int(args.input_sz)
		h_new = int(H * w_new / W)
	else:
		h_new = int(args.input_sz)
		w_new = int(W * h_new / H)
	disflow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
	prev_gray = np.zeros((h_new, w_new), np.uint8)
	prev_cfd = np.zeros((h_new, w_new), np.float32)
	is_init = True

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
					mask = mask[..., pad_up: pad_up+h_new, pad_left: pad_left+w_new]   
					#mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=True)
					mask = F.softmax(mask, dim=1)
					mask = mask[0,1,...].cpu().numpy()  #(213, 320)
				else:
					mask = model(X)
					mask = mask[..., pad_up: pad_up+h_new, pad_left: pad_left+w_new]
					#mask = F.interpolate(mask, size=(h,w), mode='bilinear', align_corners=True)
					mask = F.softmax(mask, dim=1)
					mask = mask[0,1,...].numpy()
			predict_time = time()

			# optical tracking
			cur_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
			cur_gray = cv2.resize(cur_gray, (w_new, h_new))
			scoremap = 255 * mask
			optflow_map = postprocess(cur_gray, scoremap, prev_gray, prev_cfd, disflow, is_init)
			optical_flow_track_time = time()
			prev_gray = cur_gray.copy()
			prev_cfd = optflow_map.copy()
			is_init = False
			optflow_map = cv2.GaussianBlur(optflow_map, (3, 3), 0)
			optflow_map = threshold_mask(optflow_map, thresh_bg=0.2, thresh_fg=0.8)
			img_matting = np.repeat(optflow_map[:, :, np.newaxis], 3, axis=2)
			bg_im = np.ones_like(img_matting) * 255
			re_image = cv2.resize(image, (w_new, h_new))
			comb = (img_matting * re_image + (1 - img_matting) * bg_im).astype(np.uint8)
			comb = cv2.resize(comb, (W, H))
			comb = comb[...,::-1]

			# Print runtime
			read = read_cam_time-start_time
			preproc = preproc_time-read_cam_time
			pred = predict_time-preproc_time
			optical = optical_flow_track_time-predict_time
			total = read + preproc + pred + optical
			print("read: %.3f [s]; preproc: %.3f [s]; pred: %.3f [s]; optical: %.3f [s]; total: %.3f [s]; fps: %.2f [Hz]" %
				(read, preproc, pred, optical, total, 1/pred))
			out.write(comb)
			if args.watch:
				cv2.imshow('webcam', comb[..., ::-1])
			if cv2.waitKey(1) & 0xFF == ord('q'):
				break
		else:
			break
	cap.release()
	out.release()

if __name__ == '__main__':
	args = parse_args()
	video_infer(args)