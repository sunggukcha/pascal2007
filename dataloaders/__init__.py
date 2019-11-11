from dataloaders.datasets import caltech
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

	if args.dataset == 'caltech101':
		train_set = caltech.caltech101Classification(args, split='train')
		val_set = caltech.caltech101Classification(args, split='val')
		test_set = caltech.caltech101Classification(args, split='test')
		num_classes = train_set.NUM_CLASSES

		train_loader = Dataloader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
		val_loader = Dataloader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
		test_loader = Dataloader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

		return train_loader, val_loader, test_loader, num_classes:

	else:
		print("Dataloader for {} is not implemented".format(args.dataset))
		raise NotImplementedError
