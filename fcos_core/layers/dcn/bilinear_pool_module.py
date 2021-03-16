import torch
from torch import nn
from torch.nn.parameter import Parameter
from .bilinear_pool_func import bilinear_pooling


class BilinearPooling(nn.Module):

    def __init__(self, trans_std=0.005):
        super(BilinearPooling, self).__init__()
        self.trans_std = Parameter(torch.tensor(trans_std,dtype=torch.float32))

    def forward(self, data, offset):
        dimension, channel, height, width = data.shape
        self.trans_std.data = torch.clamp(self.trans_std.data,0.001,0.01)
        return bilinear_pooling(
            data, offset, dimension, channel, height,
            width, self.trans_std)
