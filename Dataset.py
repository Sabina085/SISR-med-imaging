import torch.utils.data as data
from os import listdir
from os.path import join
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Resize, RandomCrop, RandomHorizontalFlip, RandomVerticalFlip, Normalize
import PIL
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import matplotlib.cm as cm
from torchvision import transforms
import copy
import random


batch_size = 4
patch_size = 96
image_size = 320
upscale_factor = 4
save_on_disk_train = False
save_on_disk_test = False

seed = 324
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])


def load_img(filepath):
    img = Image.open(filepath).convert('L')
    return img


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, input_transform=None, target_transform=None, bic_transform = None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [join(image_dir, x) for x in listdir(image_dir) if is_image_file(x)]

        self.input_transform = input_transform
        self.target_transform = target_transform
        self.bic_transform = bic_transform

        self.std_transform = Compose([
            Normalize([0.15], [0.15]),
        ])
        

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])
        target = copy.deepcopy(input)

        if self.target_transform:
            target = self.target_transform(target)

        input_lr = copy.deepcopy(target)
        if self.input_transform:
            input_lr = transforms.ToPILImage()(input_lr).convert("L")
            input_lr = self.input_transform(input_lr)
 
        bic = copy.deepcopy(input_lr)
        if self.bic_transform:
            bic = transforms.ToPILImage()(bic).convert("L")
            bic = self.bic_transform(bic)

        return input_lr, bic, target


    def __len__(self):
        return len(self.image_filenames)


def calculate_valid_crop_size(crop_size, upscale_factor):
    return crop_size - (crop_size % upscale_factor)


def input_transform(crop_size, upscale_factor):
    return Compose([
        Resize(crop_size // upscale_factor, PIL.Image.BICUBIC),
        ToTensor(),
    ])


def target_transform(crop_size):
    return Compose([
        RandomCrop(crop_size),
        RandomHorizontalFlip(0.5),
        RandomVerticalFlip(0.5),
        ToTensor(),
    ])


def bic_transform(crop_size):
    return Compose([
        Resize(crop_size, PIL.Image.BICUBIC),
        ToTensor(),
    ])


def get_training_set(upscale_factor, patch_size):
    train_dir='Subset_images/train_HR/'
    crop_size = calculate_valid_crop_size(patch_size, upscale_factor)
    return DatasetFromFolder(train_dir,
                             input_transform = input_transform(crop_size, upscale_factor),
                             target_transform = target_transform(crop_size),
                             bic_transform  = bic_transform(crop_size))


def get_test_set(upscale_factor, image_size):
    test_dir = 'Subset_images/test_HR/'
    return DatasetFromFolder(test_dir,
                             input_transform = input_transform(image_size, upscale_factor),
                             target_transform = Compose([ToTensor(),]),
                             bic_transform  = bic_transform(image_size))


def saveImage(img, epoch, batch, img_type, model, type_result):
    folder_path = "Results/" + model + "/" + type_result + "/"

    if not os.path.exists(folder_path):
        os.makedirs(folder_path) 

    file_path = folder_path + str(epoch) + "_" + str(batch) + "_" + img_type + ".png"
    img = img.squeeze().cpu().detach().numpy()
    plt.imsave(file_path, img, cmap = plt.get_cmap('gray'))


def save_Crops(dataset, type_set):
    save_path = 'Subset_images/'
    lr_crops = type_set + '_LR_crops/'
    hr_crops = type_set + '_HR_crops/'
    bic_crops = type_set + '_bic_crops/'

    lr_folder_path = save_path + lr_crops
    hr_folder_path = save_path + hr_crops
    bic_folder_path = save_path + bic_crops

    if not os.path.exists(lr_folder_path):
        os.makedirs(lr_folder_path) 

    if not os.path.exists(hr_folder_path):
        os.makedirs(hr_folder_path) 

    if not os.path.exists(bic_folder_path):
        os.makedirs(bic_folder_path)

    num_images = 0

    for input_lr, bic, target in dataset:
        file = str(num_images) + '.png'
        path_lr = os.path.join(lr_folder_path, file)
        path_hr = os.path.join(hr_folder_path, file)
        path_bic = os.path.join(bic_folder_path, file)

        input_lr = np.transpose(input_lr.cpu().detach().numpy(), (1, 2, 0))
        target = np.transpose(target.cpu().detach().numpy(), (1, 2, 0))
        bic = np.transpose(bic.cpu().detach().numpy(), (1, 2, 0))

        plt.imsave(path_lr, input_lr[:, :, 0], cmap = plt.get_cmap('gray'))
        plt.imsave(path_hr, target[:, :, 0], cmap = plt.get_cmap('gray'))
        plt.imsave(path_bic, bic[:, :, 0], cmap = plt.get_cmap('gray'))
        num_images += 1


if save_on_disk_train:
    dataset_train = get_training_set(upscale_factor, patch_size)
    save_Crops(dataset_train, "train")


if save_on_disk_test:
    dataset_test = get_test_set(upscale_factor, image_size)
    save_Crops(dataset_test, "test")
