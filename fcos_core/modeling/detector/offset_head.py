import torch
import torch.nn as nn
import math
import numpy as np
from fcos_core.modeling.make_layers import group_norm
from fcos_core.modeling.make_layers import make_fc
from fcos_core.layers import BilinearPooling

class Offset_Head(nn.Module):
    def __init__(self):
        super(Offset_Head, self).__init__()
        self.thresh = 1e-8
        self.is_vec = True
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = make_fc(256,256,False)
        self.fc2 = make_fc(256,256,False)
        self.fc3 = make_fc(256,2,False)
        self.align_pooling = BilinearPooling(trans_std=0.005)

    def forward(self, feature, offset_feature):
        x = self.avgpool(offset_feature)
        x = x.view(x.size(0),-1)
        x = self.fc1(x)
        x = self.fc2(x)
        offset = self.fc3(x)
        #print('offset head is')
        #print(offset)
        align_x = self.align_pooling(feature, offset)
        return align_x


def build_offset_head():
    return Offset_Head()




