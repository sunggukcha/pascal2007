'''
	Author: Sungguk Cha
	eMail : navinad@naver.com

	Dataloader for caltech101 dataset image classification.
	It randomly divides the dataset into train/val/test set with 7:1:2 ratio with fixed random seed.
'''

import os

from PIL import Image
from mypath import Path
from torch.utils import data
from torchvision import transforms
from dataloaders import classification_transforms as tr

import random

class caltech101Classification(data.Dataset):
	'''
		caltech101 classification
	'''
	NUM_CLASSES = 101

	def __init__ (self, args, root=Path.db_root_dir('caltech101'), split='train'):

		self.labels = {'accordion': 0, 'airplanes': 1, 'anchor': 2, 'ant': 3, 'barrel': 4, 'bass': 5, 'beaver': 6, 'binocular': 7, 'bonsai': 8, 'brain': 9, 'brontosaurus': 10, 'buddha': 11, 'butterfly': 12, 'camera': 13, 'cannon': 14, 'car_side': 15, 'ceiling_fan': 16, 'cellphone': 17, 'chair': 18, 'chandelier': 19, 'cougar_body': 20, 'cougar_face': 21, 'crab': 22, 'crayfish': 23, 'crocodile': 24, 'crocodile_head': 25, 'cup': 26, 'dalmatian': 27, 'dollar_bill': 28, 'dolphin': 29, 'dragonfly': 30, 'electric_guitar': 31, 'elephant': 32, 'emu': 33, 'euphonium': 34, 'ewer': 35, 'Faces': 36, 'Faces_easy': 37, 'ferry': 38, 'flamingo': 39, 'flamingo_head': 40, 'garfield': 41, 'gerenuk': 42, 'gramophone': 43, 'grand_piano': 44, 'hawksbill': 45, 'headphone': 46, 'hedgehog': 47, 'helicopter': 48, 'ibis': 49, 'inline_skate': 50, 'joshua_tree': 51, 'kangaroo': 52, 'ketch': 53, 'lamp': 54, 'laptop': 55, 'Leopards': 56, 'llama': 57, 'lobster': 58, 'lotus': 59, 'mandolin': 60, 'mayfly': 61, 'menorah': 62, 'metronome': 63, 'minaret': 64, 'Motorbikes': 65, 'nautilus': 66, 'octopus': 67, 'okapi': 68, 'pagoda': 69, 'panda': 70, 'pigeon': 71, 'pizza': 72, 'platypus': 73, 'pyramid': 74, 'revolver': 75, 'rhino': 76, 'rooster': 77, 'saxophone': 78, 'schooner': 79, 'scissors': 80, 'scorpion': 81, 'sea_horse': 82, 'snoopy': 83, 'soccer_ball': 84, 'stapler': 85, 'starfish': 86, 'stegosaurus': 87, 'stop_sign': 88, 'strawberry': 89, 'sunflower': 90, 'tick': 91, 'trilobite': 92, 'umbrella': 93, 'watch': 94, 'water_lilly': 95, 'wheelchair': 96, 'wild_cat': 97, 'windsor_chair': 98, 'wrench': 99, 'yin_yang': 100}

		self.root	= root
		self.split	= split
		self.args	= args
		self.files	= {}

		random.seed(args.seed)
		files = self.recursive_glob(rootdir=self.root, suffix='.jpg')
		self.files['train'], self.files['val'], self.files['test'] = self.divide(files)
	
	def __len__(self):
		return len(self.files[self.split])

	def __getitem__(self, index):
		img_path = self.files[self.split][index].rstrip()
		lbl = self.getlabel(img_path)
		_img = Image.open(img_path).convert('RGB')
		name = os.path.basename(img_path)
		sample = {'image': _img, 'label': lbl, 'name': name}

		if self.split == 'train':
			return self.transform_tr(sample)
		elif self.split == 'val':
			return self.transform_val(sample)
		else:
			return self.transform_test(sample)

	def transform_tr(self, sample):
		composed_transforms = transform.Compose([
			tr.Normalize(mean=[0.54584709 0.52875816 0.50213723] ,std=[0.24912914 0.2460751  0.24773671]),
			tr.ToTensor()])
		return composed_transforms(sample)

	def transform_val(self, sample):
		return self.transform_tr(sample)

	def transform_test(self, sample):
		return self.transform_tr(sample)

	def getlabel(self, img_path):
		ss = img_path.split('/')
		return self.labels[ss[-2]]
	

	def divide(self, files):
		'''
			It divides the training images into train/val/set set with the random seed.
		'''
		train = []
		val = []
		test = []
		for image in files:
			k = random.random() # [0, 1)
			if k < 0.7:
				train.append(image)
			elif k < 0.8:
				val.append(image)
			else:
				test.append(image)
		print(len(train), len(val), len(test))
		return train, val, test

	def recursive_glob (self, rootdir='.', suffix=''):
		return [os.path.join(looproot, filename)
				for looproot, _, filenames in os.walk(rootdir)
				for filename in filenames if filename.endwith(suffix)]
