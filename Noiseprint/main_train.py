# -*- coding: utf-8 -*-

# PyTorch 0.4.1, https://pytorch.org/docs/stable/index.html

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

# run this to train the model

# =============================================================================
# For batch normalization layer, momentum should be a value from [0.1, 1] rather than the default 0.1. 
# The Gaussian noise output helps to stablize the batch normalization, thus a large momentum (e.g., 0.95) is preferred.
# =============================================================================

import argparse
import re
import os, glob, datetime, time
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.init as init
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import data_generator as dg
from data_generator import DenoisingDataset,collate


# Params
parser = argparse.ArgumentParser(description='PyTorch DnCNN')
parser.add_argument('--model', default='Noiseprint', type=str, help='choose a type of model')
parser.add_argument('--batch_size', default=1, type=int, help='batch size')
parser.add_argument('--train_data', default='./dataset/dataset.h5py', type=str, help='path of train data')
parser.add_argument('--epoch', default=180, type=int, help='number of train epoches')
parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate for Adam')
args = parser.parse_args()

batch_size = args.batch_size
cuda = torch.cuda.is_available()
n_epoch = args.epoch

save_dir = os.path.join('models', args.model)

if not os.path.exists(save_dir):
	os.mkdir(save_dir)


class DnCNN(nn.Module):
	def __init__(self, depth=17, n_channels=64, image_channels=3, use_bnorm=True, kernel_size=3):
		super(DnCNN, self).__init__()
		kernel_size = 3
		padding = 1
		layers = []

		layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
		layers.append(nn.ReLU(inplace=True))
		for _ in range(depth-2):
			layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
			layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
			layers.append(nn.ReLU(inplace=True))
		layers.append(nn.Conv2d(in_channels=n_channels, out_channels=1, kernel_size=kernel_size, padding=padding, bias=False))
		self.dncnn = nn.Sequential(*layers)
		print(self.dncnn)
		self._initialize_weights()

	def forward(self, x):
		out = self.dncnn(x)
		X = out.view(out.shape[0],-1)

		dists = (X**2).sum(1).unsqueeze(0) + (X**2).sum(1).unsqueeze(1) - 2 * torch.mm(X,X.t())
		# print(dists)

		# Normalizing because the value goes > 10000
		dists = dists/(48*48)

		# Finding the distance base logisitic loss
		dists = torch.exp(-dists)
		# print(dists)
		# In case dij is too big exp(-dij) would be 0. Hence adding 1e-6
		dists = dists + 1e-6
		# print(dists)
		sum_ind = np.where(np.eye(batch_size*4) == 0)
		dists = dists/(dists[sum_ind[0],sum_ind[1]].view(4,3).sum(1))	
		# print(dists)

		# Take only those elements whose are in the same group
		kron_ind =  np.where(np.kron(np.eye(batch_size),np.ones((4,4))))
		# print(dists)
		dists = dists[kron_ind[0],kron_ind[1]]
		# print(dists)
		dists = torch.sum(-torch.log(dists) )
		return dists,out

	def _initialize_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				init.orthogonal_(m.weight)
				print('init weight')
				if m.bias is not None:
					init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				init.constant_(m.weight, 1)
				init.constant_(m.bias, 0)


def findLastCheckpoint(save_dir):
	file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
	if file_list:
		epochs_exist = []
		for file_ in file_list:
			result = re.findall(".*model_(.*).pth.*", file_)
			epochs_exist.append(int(result[0]))
		initial_epoch = max(epochs_exist)
	else:
		initial_epoch = 0
	return initial_epoch


def log(*args, **kwargs):
	 print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
	# model selection
	print('===> Building model')
	model = DnCNN()
	# Convert to double for better precision
	model.double()

	initial_epoch = findLastCheckpoint(save_dir=save_dir)  # load the last model in matconvnet style
	if initial_epoch > 0:
		print('resuming by loading epoch %03d' % initial_epoch)
		# model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
		model = torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch))
	model.train()
	# criterion = nn.MSELoss(reduction = 'sum')  # PyTorch 0.4.1
	criterion = nn.CrossEntropyLoss()
	if cuda:
		model = model.cuda()
		 # device_ids = [0]
		 # model = nn.DataParallel(model, device_ids=device_ids).cuda()
		 # criterion = criterion.cuda()
	DDataset = DenoisingDataset(args.train_data)
	DLoader = DataLoader(dataset=DDataset, collate_fn=collate ,num_workers=10, drop_last=True, batch_size=batch_size, shuffle=True)    

	optimizer = optim.Adam(model.parameters(), lr=args.lr)
	scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90], gamma=0.2)  # learning rates
	for epoch in range(initial_epoch, n_epoch):

		scheduler.step(epoch)  # step to the learning rate in this epcoh

		epoch_loss = 0
		start_time = time.time()

		for n_count, batch_x in enumerate(DLoader):
				print(len(batch_x))
				optimizer.zero_grad()
				if cuda:
					batch_x= batch_x.cuda()
				
				loss,output = model(batch_x)
				epoch_loss += loss.item()
				loss.backward()
				optimizer.step()
				if n_count % 10 == 0:
					print('%4d %4d / %4d loss = %2.4f' % (epoch+1, n_count, batch_size, loss.item()/batch_size))
		elapsed_time = time.time() - start_time

		log('epoch = %4d , loss = %4.4f , time = %4.2f s' % (epoch+1, epoch_loss/n_count, elapsed_time))
		np.savetxt('train_result.txt', np.hstack((epoch+1, epoch_loss/n_count, elapsed_time)), fmt='%2.4f')
		# torch.save(model.state_dict(), os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))
		torch.save(model, os.path.join(save_dir, 'model_%03d.pth' % (epoch+1)))






