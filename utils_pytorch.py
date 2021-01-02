import numpy as np
from skimage.measure import compare_psnr, compare_ssim
import torch


def compute_psnr_per_example(target, result):
	result = np.transpose(result.cpu().detach().numpy(), (1, 2, 0))
	target = np.transpose(target.cpu().detach().numpy(), (1, 2, 0))
	psnr = compare_psnr(target, result)
	return psnr


def compute_psnr_per_batch(target, result, batch_size):
	sum_psnr = 0.0
	for i in range(batch_size):
		psnr = compute_psnr_per_example(target[i, :, :, :], result[i, :, :, :])
		sum_psnr += psnr
	return (sum_psnr / (batch_size * 1.0)) 


def compute_ssim_per_example(target, result):
	result = np.transpose(result.cpu().detach().numpy(), (1, 2, 0))
	target = np.transpose(target.cpu().detach().numpy(), (1, 2, 0))
	print("result.shape", result.shape)
	print("target.shape", target.shape)
	ssim = compare_ssim(target[:, :, 0], result[:, :, 0])
	return ssim	


def compute_ssim_per_batch(target, result, batch_size):
	sum_ssim = 0.0
	for i in range(batch_size):
		ssim = compute_ssim_per_example(target[i, :, :, :], result[i, :, :, :])
		sum_ssim += ssim
	return (sum_ssim / (batch_size * 1.0)) 
