# Test  the model, load the data,
# followed by saving our model for later testing
from dataloader import DataSet
from model import Model
from options import TestOptions
import torch
from torchvision.transforms import *
import numpy as np 
import os
from sklearn.metrics import confusion_matrix,roc_curve
import cv2
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
# Get the Hyperparaeters 
opt = TestOptions().parse()

target_dataset = DataSet(opt,"./dataset/FindIt-Dataset-Test/T1-test/img/", "./dataset/FindIt-Dataset-Test/T1-Test-GT.xml","./dataset/FindIt-Dataset-Test/T2-test/img/" )
target_loader = torch.utils.data.DataLoader(target_dataset,batch_size=opt.val_batch_size,num_workers=opt.workers//5,shuffle=False)

# Load the model and send it to gpu
test_transforms =  transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0.,0.,0.,0. ],
						 std = [ 1/0.229, 1/0.224, 1/0.225,1.,1.,1. ]),
	transforms.Normalize(mean = [ -0.485, -0.456, -0.406,0.,0.,0. ],
						 std = [ 1., 1., 1.,1.,1.,1. ]) ])


device = torch.device("cuda" if (torch.cuda.is_available() and opt.use_gpu) else "cpu")
opt.device = device
model = Model(opt)
if opt.use_gpu:

	model = model.cuda()	
	model = torch.nn.DataParallel(model, device_ids=opt.gpus)

# Load the weights and make predictions
model.load_state_dict(torch.load('./checkpoints/' + 'model_{}.pt'.format(opt.load_epoch)))

# Print our model 
print('------------ Model -------------')
print(model)
print('-------------- End ----------------')

model.eval()
def get_accuracy(pred_sample, sample_labels):
	confidence = pred_sample[np.arange(len(sample_labels)),sample_labels[0,:]]
	# confidence = pred_sample[:,1]
	pred_sample = np.argmax(pred_sample.cpu().data.numpy(),axis=1 )

	return pred_sample,confidence

classified_list = []
confidence_list = []
label_list = []
print("Hello")
for i,(target_images, target_labels) in enumerate(target_loader):
	print(i)
	target_labels.squeeze(1)

	try:
		pred_target,_,loss_mmd = model(target_images.to(device),target_images.to(device))
	except RuntimeError  as r:
		print("Error:",r)
		torch.cuda.empty_cache()
		continue

	target_pred,confidence = get_accuracy(pred_target,target_labels)	
	classified_list.extend(target_pred)
	label_list.extend(target_labels)
	confidence_list.extend(confidence.cpu().data.numpy())

fpr, tpr, thresholds = roc_curve(classified_list,confidence_list)
conf_target = confusion_matrix(label_list,classified_list)

print("==============Best results======================")
print(conf_target)