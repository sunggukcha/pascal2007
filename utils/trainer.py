from dataloaders import make_data_loader
from utils.metrics import Evaluator
from utils.saver import Saver

import torch
import torch.nn as nn

def bn(planes):
	return nn.BatchNorm2d(planes)

class Trainer(object):
	def __init__(self, args):
		self.args = args

		# Define saver
		self.saver = Saver(args)
		self.saver.save_experiment_config()

		# Define TensorBoard summary
		self.summary = TensorboardSummary(self.saver.experiment_dir)
		if not args.test:
			self.writer = self.summary.create_summary()

		# Define Dataloader
		kwargs = {'num_workers': args.workers, 'pin_memory': True}
		self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
	
		norm = bn

		# Define Network
		model = Model(self.args.model)
		
		train_params = [{'params': model.get_params(), 'lr': args.lr}]
		
		# Define Optimizer
		optimizer = torch.optim.SGD(train_params, momentum=args.momentum, weight_decay=args.weight_decay)
		criterion = nn.CrossEntropyLoss()

		self.model, self.optimizer, self.criterion = model, optimizer, criterion

		# Define Evaluator
		self.evaluator = Evaluator(self.args.dataset)

		# Define lr scheduler
		self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr, args.epochs, len(self.train_loader))

		# Using CUDA
		if args.cuda:
			self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
			self.model = self.model.cuda()


