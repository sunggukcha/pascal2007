from models.classifier import Classifier
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

class Model(nn.Module):
	def __init__(self, args, nclass):
		super(Model, self).__init__()

		self.nclass = nclass
		if args.dataset == 'embedding':
			self.backbone = Classifier(self.nclass)
		else:
			self.backbone = models.__dict__[args.arch](pretrained=args.pretrained)
			final_feature = self.backbone.fc.in_features
			self.backbone.fc = nn.Linear(final_feature, self.nclass)

	def forward(self, input):
		x = self.backbone(input)
		return x
