import os, cv2, torch
import numpy as np
from data_loader import transforms

mean = np.array([0.485, 0.456, 0.406])[None,None,:]
std = np.array([0.229, 0.224, 0.225])[None,None,:]

def preprocessing(image, expected_size=320, pad_value=0):
	image, pad_up, pad_left, h_new, w_new = transforms.resize_image(image, expected_size, pad_value, ret_params=True)
	image = image.astype(np.float32) / 255.0
	image = (image - mean) / std
	X = np.transpose(image, axes=(2, 0, 1))
	X = np.expand_dims(X, axis=0)
	X = torch.tensor(X, dtype=torch.float32)
	return X, pad_up, pad_left, h_new, w_new

def motion_blur(image, degree=18, angle=45):
    image = np.array(image)

    M = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, M, (degree, degree))

    motion_blur_kernel = motion_blur_kernel / degree
    blurred = cv2.filter2D(image, -1, motion_blur_kernel)
    # convert to uint8
    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred = np.array(blurred, dtype=np.uint8)
    return blurred

def draw_transperency(image, mask, color_f, color_b):
	"""
	image (np.uint8)
	mask  (np.float32) range from 0 to 1
	"""
	mask = mask.round()
	alpha = np.zeros_like(image, dtype=np.uint8)
	alpha[mask==1, :] = color_f
	alpha[mask==0, :] = color_b

	#blurred = motion_blur(image)
	blurred = image
	image_alpha = cv2.add(blurred, alpha)
	return image_alpha

def draw_matting(image, mask):
	"""
	image (np.uint8)
	mask  (np.float32) range from 0 to 1
	"""
	mask = 255*(1.0-mask)
	mask = np.expand_dims(mask, axis=2)
	mask = np.tile(mask, (1,1,3))
	mask = mask.astype(np.uint8)
	image_matting = cv2.add(image, mask)
	return image_matting

def draw_fore_to_back(image, mask, background, kernel_sz=13, sigma=0):
	"""
	image (np.uint8)
	mask  (np.float32) range from 0 to 1
	"""
	mask_filtered = cv2.GaussianBlur(mask, (kernel_sz, kernel_sz), sigma)
	mask_filtered = np.expand_dims(mask_filtered, axis=2)
	mask_filtered = np.tile(mask_filtered, (1,1,3))

	image_alpha = image*mask_filtered + background*(1-mask_filtered)
	return image_alpha.astype(np.uint8)
