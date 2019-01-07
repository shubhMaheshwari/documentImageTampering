import os
import numpy as np
import xml.etree.ElementTree as ET 
import torch
from torchvision.transforms import transforms
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import cv2

train_transforms =  transforms.Compose([
	# transforms.RandomHorizontalFlip(),
	# transforms.RandomVerticalFlip(),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406, 0.0, 0.0 ,0.0],std=[0.229, 0.224, 0.225, 1.0, 1.0, 1.0])
])

class DataSet(torch.utils.data.Dataset):
	"""
		Our dataset loader for each training, testing, val dataset

	"""
	def __init__(self,opt,image_dir,xml_file, fake_images):
		super().__init__()
		
		# get the ground truth of all images
		y_dict = {}
		tree = ET.parse(xml_file)
		root = tree.getroot()
		for elem in root:
			d = elem.attrib
			y_dict[d['id']] = int(d['modified'])

		# Calculate filename list
		image_list = []
		y_list = []
		for filename in os.listdir(image_dir):
			doc_id = filename.split('.')[0]
			if (int(y_dict[doc_id]) == 0 and np.random.random() <= 0.3) or int(y_dict[doc_id]) == 1:
				y_list.append(int(y_dict[doc_id]))
				image_list.append(os.path.join(image_dir,filename))	        

		for filename in os.listdir(fake_images):
			image_list.append(os.path.join(fake_images,filename))	        
			y_list.append(1)


		if len(y_list) != len(image_list):
			print("Error creating dataset")


		self.image_list = image_list
		self.y_list = y_list
		self.opt = opt

		print("Dataset size Images(0):{}".format(len(self.image_list)))
	
	def crop_center(self,img):
		y,x,c = img.shape
		cropx = self.opt.cropx
		cropy = self.opt.cropy

		startx = x//2-(cropx//2)
		starty = y//2-(cropy//2)

		if startx < 0:
			new_im = np.zeros((cropy,cropx,c),dtype='uint8') 
			if starty < 0:
				new_im[-starty:-starty+y,-startx:-startx+x,:] = img
			else:
				new_im[:,-startx:-startx+x,:] = img[starty:starty+cropy,:,:]
			return new_im

		elif starty < 0:
			new_im = np.zeros((cropy,cropx,c),dtype='uint8') 
			new_im[-starty:-starty+y,:,:] = img[:,startx:startx+cropx,:]
			return new_im	

		else:
			return img[starty:starty+cropy,startx:startx+cropx]

	def __getitem__(self,idx):
	
		"""
			Given an index it returns the image and its ground truth (fake: 1, true: 0)
		"""
		try:
			path = self.image_list[idx]
			im = cv2.imread(path)
			y = self.y_list[idx]
		except Exception as e:
			idx += 1
			path = self.image_list[idx]
			im = cv2.imread(path)
			y = self.y_list[idx]

		print(path)
		if(im is None):
			print("fjakjdla;ks" + str(path))
		im = self.crop_center(im)
		dct = np.zeros((im.shape))
		dct[:,:,0] = cv2.dct(np.float32(im[:,:,0])/255.0)
		dct[:,:,1] = cv2.dct(np.float32(im[:,:,1])/255.0)
		dct[:,:,2] = cv2.dct(np.float32(im[:,:,2])/255.0)

		im = np.concatenate((im,dct),axis=2)
		im = im.astype('float32')
		# print(im.)

		im = train_transforms(im)

		# plt.imshow(im.transpose(0,2))
		# plt.show()

		y =  torch.LongTensor([y])

		return im,y				



	def __len__(self):
		return len(self.image_list)



def create_samplers(length,split):
	"""
		To make a train and validation split 
		we must know out of which indices should
		the dataloader load for training images and validation images
	"""

	# Validation dataset size
	val_max_size = np.floor(length*split).astype('int')

	# List of Randomly sorted indices
	idx = np.arange(length)
	idx = np.random.permutation(idx)

	# Make a split
	train_idx = idx[0:val_max_size]
	validation_idx = idx[val_max_size:length]

	# Create the sampler required by dataloaders
	train_sampler = SubsetRandomSampler(train_idx)
	val_sampler = SubsetRandomSampler(validation_idx)

	return train_sampler,val_sampler


if __name__ == "__main__":
	pass