import model_pytorch
import Dataset
import utils_pytorch
from torch.utils.data import DataLoader
import torch
import random
import numpy as np
import time


class Object(object):
    pass

FLAGS = Object()
FLAGS.num_anchor = 16
FLAGS.inner_channel = 16 
FLAGS.deep_kernel = 3
FLAGS.deep_layer = 5
FLAGS.upscale = 4
FLAGS.original_SRCNN = True

model = "CARN" # "SRCNN" # "newCARN"
nr_epochs = 500

seed = 324
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

pth_path = 'Saved models/' + model + "/"

if model == "CARN" or model == "newCARN":
    file_name = model + "_" + str(nr_epochs) + "(epochs)_" +  str(FLAGS.num_anchor) + "(anch)_" +  str(FLAGS.inner_channel) + "(ich)_" + str(FLAGS.deep_layer) + "(nrBlocks).pth"
else:
    file_name = file_name = model + "_" +  str(nr_epochs) + "(epochs)_" + str(FLAGS.inner_channel) + "(ich).pth"
    if not FLAGS.original_SRCNN:
        file_name = model + "_" +  str(nr_epochs) + "(epochs)_" + str(FLAGS.inner_channel) + "(ich).pth"
    else:
        file_name = model + "_" +  str(nr_epochs) + "(epochs)_" + "original.pth"

path = pth_path + file_name
upscale_factor = 4
image_size = 320
num_channels = 1

dataset = Dataset.get_test_set(upscale_factor, image_size)
Dataset.save_Crops(dataset, "test")
dataloader = DataLoader(dataset, batch_size = 1)

if model == "CARN":
	net = model_pytorch.CARN(FLAGS, num_channels) 
elif model == "newCARN":
    net = model_pytorch.newCARN(FLAGS, num_channels) 
elif model == "SRCNN":
    net = model_pytorch.SRCNN(FLAGS)
else:
	raise ValueError("Wrong model name")

checkpoint = torch.load(path)
net.load_state_dict(checkpoint)

psnr_bicubic = 0.0
psnr_result = 0.0
ssim_bicubic = 0.0
ssim_result = 0.0
nr = 0
execution_times = []

for i, data in enumerate(dataloader, 0):
	input_lr, bic, target = data
	bic = bic
	target = target
	input_lr = input_lr
	
	# begin timer
	start = time.time()

	if model == "CARN" or model == "newCARN":
		outputs = net(input_lr, bic)
	elif model == "SRCNN":
		outputs = net(bic) # -> for SRCNN (only the bicubic interpolated LR image as input)
	else:
		raise ValueError("Wrong model name")

	# end timer
	time_taken = time.time() - start
	execution_times.append(time_taken)

	psnr_current_result = utils_pytorch.compute_psnr_per_batch(target, outputs, target.shape[0])
	psnr_current_bicubic = utils_pytorch.compute_psnr_per_batch(target, bic, target.shape[0])

	ssim_current_result = utils_pytorch.compute_ssim_per_batch(target, outputs, target.shape[0])
	ssim_current_bicubic = utils_pytorch.compute_ssim_per_batch(target, bic, target.shape[0])

	print("Batch: ", i + 1)
	print("PSNR result: ", psnr_current_result)
	print("PSNR bicubic: ", psnr_current_bicubic)
	print("SSIM result: ", ssim_current_result)
	print("SSIM bicubic: ", ssim_current_bicubic)	

	psnr_result += psnr_current_result
	psnr_bicubic += psnr_current_bicubic
	ssim_result += ssim_current_result
	ssim_bicubic += ssim_current_bicubic
	nr += 1.0

	if model == "CARN" or model == "newCARN":
		full_model_name =  model + "_" + str(nr_epochs) + "(epochs)_" +  str(FLAGS.num_anchor) + "(anch)_" +  str(FLAGS.inner_channel) + "(ich)_" + str(FLAGS.deep_layer) + "(nrBlocks).pth"
	else:
		if not FLAGS.original_SRCNN:
			full_model_name = model + "_" +  str(nr_epochs) + "(epochs)_" + str(FLAGS.inner_channel) + "(ich).pth"
		else:
			full_model_name = model + "_" +  str(nr_epochs) + "(epochs)_" + "original.pth"

	Dataset.saveImage(input_lr[0], '', i, "lr", full_model_name, "test")
	Dataset.saveImage(bic[0], '', i, "bic", full_model_name, "test")
	Dataset.saveImage(outputs[0], '', i, "res", full_model_name, "test")
	Dataset.saveImage(target[0], '', i, "gt", full_model_name, "test")

print('----------------------------------------------------------------------------------')
print("Average PSNR result: ", psnr_result / nr)
print("Average PSNR bicubic: ", psnr_bicubic / nr)
print("Average SSIM result: ", ssim_result / nr)
print("Average SSIM bicubic: ", ssim_bicubic / nr)

model_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)

print("Number of parameters", model_total_params)
print('Average execution time on CPU: ', np.mean(execution_times))
print('Model: ', model)
print('file_name', file_name)
