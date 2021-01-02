import numpy as np
import imageio
import h5py
import matplotlib.pyplot as plt
from skimage import img_as_ubyte
import os
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join


train_path = 'Training_data/' # 'Val_data/'
files_train = [f for f in sorted(listdir(train_path)) if isfile(join(train_path, f))]
print("Files: ", files_train)

num_images = 0

for file_train in files_train:
	f = h5py.File(train_path + file_train, 'r')
	dset = f['reconstruction_rss']

	save_path = "./Subset_images/train_HR"
	if not os.path.exists(save_path):
		os.makedirs(save_path) 

	print(dset)
	print(f['reconstruction_rss'].shape)
	data = np.transpose(np.array(dset[:,:,:]), (1, 2, 0))

	for i in range(data.shape[2]):
		file = str(num_images) + '.png'
		path = os.path.join(save_path, file)
		img = data[:, :, i]
		plt.imsave(path, img, cmap = 'gray')
		num_images += 1

