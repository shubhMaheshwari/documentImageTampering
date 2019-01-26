# -*- coding: utf-8 -*-

# =============================================================================
#  @article{zhang2017beyond,
#    title={Beyond a {Gaussian} denoiser: Residual learning of deep {CNN} for image denoising},
#    author={Zhang, Kai and Zuo, Wangmeng and Chen, Yunjin and Meng, Deyu and Zhang, Lei},
#    journal={IEEE Transactions on Image Processing},
#    year={2017},
#    volume={26}, 
#    number={7}, 
#    pages={3142-3155}, 
#  }
# by Kai Zhang (08/2018)
# cskaizhang@gmail.com
# https://github.com/cszn
# modified on the code from https://github.com/SaoYan/DnCNN-PyTorch
# =============================================================================

# no need to run this code separately

import os
import glob
import cv2
import h5py
import numpy as np
# from multiprocessing import Pool
from torch.utils.data import Dataset
import torch

total_patches = 100
patch_size, stride = 48, 10
batch_size = 128


class DenoisingDataset(Dataset):
	"""Dataset wrapping tensors.
	Arguments:
		dataset (Tensor): dataset with h5py file for each camera model, 100 patches(for now) each
	"""
	def __init__(self, dataset_h5py):
		super(DenoisingDataset, self).__init__()

		self.dataset_h5py = h5py.File( dataset_h5py , "r")
		
		self.camera_list = [ cam.encode("utf-8") for cam in list(self.dataset_h5py.keys()) ]

		# for cam in self.camera_list:
		# 	print(type(cam))

		# self.dataset_h5py.close()
		# os._exit(0)	

		print("Total {} camera models".format(len(self.camera_list)))
	# From 25 patches get 100 different images
	# Create 200 patches of 25 groups (same camera and position) and 4 images per group


	# We want to make sure each patch is visited atleast once.
	# Hence looking at the index for 1st image, other 3 are taken random from the model
	def __getitem__(self, index):
		
		camera_model = index//100
		images = np.zeros((4,patch_size,patch_size,3))
		patch_index = index%100		
		rand_ind = np.sort(np.random.randint(0,11,4))
		# print(rand_ind,rand_patch)
		for i,rand in enumerate(rand_ind):
			try:
				images[i,...] = self.dataset_h5py[self.camera_list[camera_model]][rand,patch_index,...]
			except Exception as e:
				print(self.camera_list[camera_model],rand_ind)
				images[i,...] = self.dataset_h5py[self.camera_list[camera_model]][0,patch_index,...]

		# print(images.shape)		
		return torch.from_numpy(images.astype('float64')/255.0)

	def __len__(self):
		return 100*len(self.camera_list)


def collate(image_list):
	# We are given a list of camera model groups. Now using these we can create n^2 cases 
	X = torch.stack(image_list)
	X = X.transpose(2,4)
	# print(X.shape)
	X = X.view(-1,3,48,48)
	return X

if __name__ == '__main__': 

	data = datagenerator(data_dir='data/Train400')


#    print('Shape of result = ' + str(res.shape))
#    print('Saving data...')
#    if not os.path.exists(save_dir):
#            os.mkdir(save_dir)
#    np.save(save_dir+'clean_patches.npy', res)
#    print('Done.')       