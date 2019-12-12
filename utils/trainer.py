from dataloaders import make_data_loader
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
		if args.resume is not None:
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

		# layer wise freezing
		self.history = {}

	def train(self, epoch):
		train_loss = 0.0
		self.model.train()
		tbar = tqdm(self.train_loader)
		num_img_tr = len(self.train_loader)
		for i, sample in enumerate(tbar):
			image, target = sample['image'].cuda(), sample['label'].cuda()
			self.scheduler(self.optimizer, i, epoch, self.best_pred)
			self.optimizer.zero_grad()
			output = self.model(image)
			loss = self.criterion(output, target)
			loss.backward()
			self.optimizer.step()
			train_loss += loss.item()
			tbar.set_description('Train loss: %.3f' % (train_loss / (i+1)))
			self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
		self.writer.add_scalar('train/total_loss_epoch', train_loss, epoch)
		print("[epoch: %d, loss: %.3f]" % (epoch, train_loss))

		if self.args.freeze:
			for n, i in self.model.named_parameters():
				self.history[str(epoch)+n] = i.cpu().detach()

		if self.args.no_val:
			is_best = False
			self.saver.save_checkpoint(
				{'epoch' : epoch + 1,
				'state_dict': self.model.module.state_dict(),
				'optimizer': self.optimizer.state_dict(),
				'best_pred': self.best_pred,},
				is_best
				)

	def val(self, epoch):
		top1 = AverageMeter('Acc@1', ':6.2f')
		top5 = AverageMeter('Acc@5', ':6.2f')

		self.model.eval()
		
		tbar = tqdm(self.val_loader, desc='\r')
		test_loss = 0.0
		for i, sample in enumerate(tbar):
			image, target = sample['image'], sample['label']
			if self.args.cuda:
				image, target = image.cuda(), target.cuda()
			with torch.no_grad():
				output = self.model(image)
			loss = self.criterion(output, target)

			#
			test_loss += loss.item()

			# top accuracy record
			acc1, acc5 = accuracy(output, target, topk=(1, 5))
			top1.update(acc1[0], image.size(0))
			top5.update(acc5[0], image.size(0))

			tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
			pred = output.data.cpu().numpy()
			target = target.cpu().numpy()
			pred = np.argmax(pred, axis=1)
			# Add batch sample into evaluator
			#self.evaluator.add_batch(target, pred)

		# Fast test during the training
		_top1 = top1.avg
		_top5 = top5.avg
		self.writer.add_scalar('val/total_loss_epoch', test_loss, epoch)
		self.writer.add_scalar('val/top1', _top1, epoch)
		self.writer.add_scalar('val/top5', _top5, epoch)
		print('Validation:')
		print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
		print("Top-1: %.3f, Top-5: %.3f" % (_top1, _top5))
		print('Loss: %.3f' % test_loss)

		new_pred = _top1
		if new_pred > self.best_pred:
			is_best = True
			self.best_pred = float(new_pred)
			self.saver.save_checkpoint({
				'epoch': epoch + 1,
				'state_dict': self.model.module.state_dict(),
				'optimizer': self.optimizer.state_dict(),
				'best_pred': self.best_pred,
				}, is_best)
