import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.last_fc = nn.Linear(300, num_classes)


    def forward(self, x):
        return self.last_fc(x)
