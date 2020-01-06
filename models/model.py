from models.classifier import Classifier
import models.xor_resnet as XOR
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

XORS = {
    'xor_resnet18': XOR.resnet18,
    'xor_resnet34': XOR.resnet34,
    'xor_resnet50': XOR.resnet50,
    'xor_resnet101': XOR.resnet101,
    'xor_resnet152': XOR.resnet152
}

class Model(nn.Module):
	def __init__(self, args, nclass):
		super(Model, self).__init__()

		self.nclass = nclass
		if args.dataset == 'embedding':
			self.backbone = Classifier(self.nclass)
		elif 'xor_resnet' in args.arch:
			self.backbone = XORS[args.arch](pretrained=args.pretrained)
			final_feature = self.backbone.fc.in_features
			self.backbone.fc = nn.Linear(final_feature, self.nclass)
		else:
			self.backbone = models.__dict__[args.arch](pretrained=args.pretrained)
			final_feature = self.backbone.fc.in_features
			self.backbone.fc = nn.Linear(final_feature, self.nclass)

	def forward(self, input):
		x = self.backbone(input)
		return x
