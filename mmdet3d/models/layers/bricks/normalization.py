# Copyright (c) OpenMMLab. All rights reserved.
import torch
from torch import nn


class Offset(nn.Module):

    def __init__(self, init_value=0.):
        super(Offset, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input + self.bias
