from dataloaders import make_data_loader, make_id2class
from models.model import Model
from tqdm import tqdm
from utils.lr_scheduler import LR_Scheduler
from utils.metrics import *
from utils.saver import Saver
from utils.summaries import TensorboardSummary

import os
import torch
import torch.nn as nn
import torchvision.models as models

def bn(planes):
	return nn.BatchNorm2d(planes)

class Tester(object):
	def __init__(self, args):
		self.args = args
		
		# Define Dataloader
		kwargs = {'num_workers': args.workers, 'pin_memory': True}
		self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
		self.id2class = make_id2class(args)
	
		norm = bn

		# Define Network
		model = Model(args, self.nclass)
		
		train_params = [{'params': model.parameters(), 'lr': args.lr}]
		
		# Define Optimizer
		optimizer = torch.optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay)
		criterion = nn.CrossEntropyLoss()

		self.model, self.optimizer, self.criterion = model, optimizer, criterion

		# Define lr scheduler
		self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

		# Using CUDA
		if args.cuda:
			self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
			self.model = self.model.cuda()

		# Resuming checkpoint
		self.best_pred = 0.0
		if args.resume is None:
			raise RuntimeError("Checkpoint for test is required")
		else:
			if not os.path.isfile(args.resume):
				raise RuntimeError("{}: No such checkpoint exists".format(args.resume))
			checkpoint = torch.load(args.resume)
			args.start_epoch = checkpoint['epoch']

			if args.cuda:
				pretrained_dict = checkpoint['state_dict']
				model_dict = {}
				state_dict = self.model.module.state_dict()
				for k, v in pretrained_dict.items():
					if k in state_dict:
						model_dict[k] = v
				state_dict.update(model_dict)
				self.model.module.load_state_dict(state_dict)
			else:
				print("Please use CUDA")
				raise NotImplementedError

			if not args.ft:
				self.optimizer.load_state_dict(checkpoint['optimizer'])
			self.best_pred = checkpoint['best_pred']
			print("Loading {} (epoch {}) successfully done".format(args.resume, checkpoint['epoch']))

		if args.ft:
			args.start_epoch = 0

	def test(self):
		self.model.eval()
		
		tbar = tqdm(self.test_loader, desc='\r')
		for i, sample in enumerate(tbar):
			image = sample['image']
			name = sample['name']

			if self.args.cuda:
				image = image.cuda()
			with torch.no_grad():
				output = self.model(image)
			pred = output.data.cpu().numpy()
			print(pred.shape)
			#pred = np.argmax(pred, axis=1)
			tbar.set_description("{} : {}".format(name[0], self.id2class(output[0])))
