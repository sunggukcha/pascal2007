import argparse
import os
import torch


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
	parser.add_argument('--test', type=bool, default=False,
						help='True if test mode')
	
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

if __name__ == "__main__":
	args = get_args()
	args.cuda = torch.cuda.is_available()
	if args.cuda:
		try:
			args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
		except ValueError:
			raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
	
	print(args)
	torch.manual_seed(args.seed)
	trainer = Trainer(args)
	print("Starting epoch: {}".format(trainer.args.start_epoch))
	print("Total epochs: {}".format(trainer.args.epochs))

	for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
		trainer.train(epoch)
		if not trainer.args.no_val:
			trainer.val(epoch)

	trainer.writer.close()
