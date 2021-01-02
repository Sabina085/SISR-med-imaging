from skimage import measure 
import torch
import torch.nn.functional as F
from torch import optim
import os
import model_pytorch
from os import listdir
from os.path import isfile, join
import Dataset
from torch.utils.data import DataLoader
from torch.nn import MSELoss
import utils_pytorch
import shutil
from torch.utils.tensorboard import SummaryWriter
import datetime
import numpy as np
from torch import nn
import random


seed = 324
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


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


# He initialization
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data)
        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 32
upscale_factor = 4
patch_size = 96
image_size = 320

dataset = Dataset.get_training_set(upscale_factor, patch_size)
Dataset.save_Crops(dataset, "train")
dataloader = DataLoader(dataset, batch_size = batch_size)

num_channels = 1

if model == "CARN":
    net = model_pytorch.CARN(FLAGS, num_channels) 
elif model == "newCARN":
    net = model_pytorch.newCARN(FLAGS, num_channels) 
elif model == "SRCNN":
    net = model_pytorch.SRCNN(FLAGS)
else:
    raise ValueError("Wrong model name")

net = net.to(device)

if model == "CARN" or model == "newCARN":
    net.apply(weights_init) # He initialization

model_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Number of parameters", model_total_params)

criterion = MSELoss()

'''
if model == "CARN":
    optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay = 0.0001) # L2 regularization for CARN
else:
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
'''

optimizer = optim.Adam(net.parameters(), lr=1e-3)
nr_epochs = 500
nr_print = 5
iteration = 0

for epoch in range(nr_epochs): 
    running_loss = 0.0
    running_psnr = 0.0
    psnr_bicubic = 0.0

    for i, data in enumerate(dataloader, 0):
        iteration += batch_size
        input_lr, bic, target = data

        optimizer.zero_grad()

        bic = bic.to(device)
        target = target.to(device)
        input_lr = input_lr.to(device)

        if model == "CARN" or model == "newCARN":
            outputs = net(input_lr, bic)
        elif model == "SRCNN":
            outputs = net(bic) # -> for SRCNN (only the bicubic interpolated LR image as input)
        else:
            raise ValueError("Wrong model name")

        loss = torch.mean(criterion(outputs, target))
        loss.backward()

        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.)
        optimizer.step()

        running_psnr += utils_pytorch.compute_psnr_per_batch(target, outputs, target.shape[0])
        psnr_bicubic += utils_pytorch.compute_psnr_per_batch(target, bic, target.shape[0])

        running_loss += loss.item()

        if i % nr_print == (nr_print-1):
            if model == "CARN" or model == "newCARN":
                full_model_name =  model + "_" + str(nr_epochs) + "(epochs)_" +  str(FLAGS.num_anchor) + "(anch)_" +  str(FLAGS.inner_channel) + "(ich)_" + str(FLAGS.deep_layer) + "(nrBlocks).pth"
            else:
                if not FLAGS.original_SRCNN:
                    full_model_name = model + "_" +  str(nr_epochs) + "(epochs)_" + str(FLAGS.inner_channel) + "(ich).pth"
                else:
                    full_model_name = model + "_" +  str(nr_epochs) + "(epochs)_" + "original.pth"

            Dataset.saveImage(input_lr[0], epoch, i, "lr", full_model_name, "train")
            Dataset.saveImage(bic[0], epoch, i, "bic", full_model_name, "train")
            Dataset.saveImage(outputs[0], epoch, i, "res", full_model_name, "train")
            Dataset.saveImage(target[0], epoch, i, "gt", full_model_name, "train")

            print('[%d, %5d] PSNR: %.6f' %(epoch + 1, i + 1, running_psnr / nr_print))
            running_psnr = 0.0
            print('[%d, %5d] PSNR bicubic: %.6f' %(epoch + 1, i + 1, psnr_bicubic / nr_print))
            psnr_bicubic = 0.0


folder_path = "Saved models/" + model + "/"
if not os.path.exists(folder_path):
    os.makedirs(folder_path) 

if model == "CARN" or model == "newCARN":
    file_name = model + "_" + str(nr_epochs) + "(epochs)_" +  str(FLAGS.num_anchor) + "(anch)_" +  str(FLAGS.inner_channel) + "(ich)_" + str(FLAGS.deep_layer) + "(nrBlocks).pth"
else:
    file_name = file_name = model + "_" +  str(nr_epochs) + "(epochs)_" + str(FLAGS.inner_channel) + "(ich).pth"

    if not FLAGS.original_SRCNN:
        file_name = model + "_" +  str(nr_epochs) + "(epochs)_" + str(FLAGS.inner_channel) + "(ich).pth"
    else:
        file_name = model + "_" +  str(nr_epochs) + "(epochs)_" + "original.pth"

torch.save(net.state_dict(), folder_path + file_name)

print('Finished Training')
print("Number of parameters", model_total_params)

