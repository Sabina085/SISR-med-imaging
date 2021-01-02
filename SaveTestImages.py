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


test_path = 'Test_data/'
files_test = [f for f in sorted(listdir(test_path)) if isfile(join(test_path, f))]
print("Files: ", files_test)

num_images = 0

for file_test in files_test:
	f = h5py.File(test_path + file_test, 'r')
	dset = f['reconstruction_rss']
	save_path = "./Subset_images/test_HR"
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

