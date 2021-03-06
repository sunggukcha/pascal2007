from utils.tester import Tester
from utils.trainer import Trainer

import argparse
import os
import time
import torch
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def get_args():
	parser = argparse.ArgumentParser()

	# external params
	parser.add_argument('--seed', type=int, default=1,
						help='Random seed')
	parser.add_argument('--checkname', type=str, default=None,
						help='Checkpoint name')
		# GPU
	parser.add_argument('--cuda', type=bool, default=True,
						help='CUDA usage')
	parser.add_argument('--gpu-ids', type=str, default='0',
						help='E.g., 0 | 0,1,2,3 ')
		# CPU
	parser.add_argument('--workers', type=int, default=0,
						help='Number of workers for dataloader')

	# training options
	parser.add_argument('--dataset', type=str, default='caltech101')
	parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
	parser.add_argument('--pretrained', default=True, type=bool,
						help='True if load pretrained model')
	parser.add_argument('--ft', type=bool, default=None,
						help='True if finetune')
	parser.add_argument('--resume', type=str, default=None)
	parser.add_argument('--test', default=False, action='store_true',
						help='True if test mode')
	parser.add_argument('--start-epoch', type=int, default=0)
	parser.add_argument('--no-val', type=bool, default=False,
						help='True if train without validation')
	parser.add_argument('--freeze', default=False, action='store_true',
						help='Layer wise freezing')
	parser.add_argument('--time', default=False, action='store_true')
	parser.add_argument('--decomp', default=False, action='store_true')
	
	# hyper params
	parser.add_argument('--lr', type=float, default=None,
						help='Initial learning rate')
	parser.add_argument('--lr-scheduler', type=str, default='poly',
						choices=['poly', 'step', 'cos'],
						help='lr scheduler mode: (default: poly)')
	parser.add_argument('--epochs', type=int, default=None,
						help='Training epochs')
	parser.add_argument('--batch-size', type=int, default=1,
						help='Batch size')
	parser.add_argument('--momentum', type=float, default=0.9,
						help='Momentum')
	parser.add_argument('--weight-decay', type=float, default=1e-4,
						help='Weight decay')

	return parser.parse_args()

if __name__ == "__main__":
	args = get_args()
	args.cuda = torch.cuda.is_available()
	if args.cuda:
		try:
			args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
		except ValueError:
			raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
	
	if args.checkname == None:
		args.checkname = args.arch

	print(args)
	torch.manual_seed(args.seed)
	if not args.test:
		trainer = Trainer(args)
		print("Starting epoch: {}".format(trainer.args.start_epoch))
		print("Total epochs: {}".format(trainer.args.epochs))
		if trainer.args.time: start = time.time()
		for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
			trainer.train(epoch)
			if not trainer.args.no_val and not trainer.args.time:
				trainer.val(epoch)
		if trainer.args.time:
			now = time.time()
			print(now - start)
			print(trainer.histories)
			trainer.val(epoch)
		trainer.writer.close()
	else:
		tester = Tester(args)
		tester.test()
