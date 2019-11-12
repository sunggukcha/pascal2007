import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

model_names = sorted(name for name in models.__dict__
					if name.islower() and not name.startswitih("__")
					and callable(models.__dict__[name]))

class Model(nn.Module):
	def __init__(self, model):
		super(Model, self).__init__()

		self.backbone = build_backbone(model)

	def forward(self, input):
		x = self.backbone(input)
		
