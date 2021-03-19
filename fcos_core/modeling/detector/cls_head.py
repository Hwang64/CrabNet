import torch
import torch.nn as nn
import math
import numpy as np
from fcos_core.modeling.make_layers import group_norm
from fcos_core.modeling.make_layers import make_fc
from fcos_core.modeling.detector import MPNCOV
from torch.nn import functional as F
from torch.nn.parameter import Parameter
import numpy as np

class Cls_Head(nn.Module):
    def __init__(self,gap, gcp, mix):
        super(Cls_Head, self).__init__()
        self.gap = gap
        self.gcp = gcp
        self.mix = mix
        self.thresh = 1e-8
        self.is_vec = True
        self.gcp_mode = 0
        if self.gap == True:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
            if self.mix == True: self.fc_gap = make_fc(2048,1024,False)
            else: self.fc_gap = make_fc(512,80,False)
        if self.gcp == True:
            self.reduce_dim = 64
            self.layer_reduce = nn.Conv2d(2048, self.reduce_dim, kernel_size=1, stride=1, padding=0, bias=False)
            if self.mix == True :
                if self.gcp_mode == 0: self.fc_gcp = make_fc(self.reduce_dim*self.reduce_dim,1024,False)
                if self.gcp_mode == 1: self.fc_gcp = make_fc(int(self.reduce_dim*(self.reduce_dim+1)/2),1024,False)
            if self.mix == False:
                if self.gcp_mode == 0: self.fc_gcp = make_fc(self.reduce_dim*self.reduce_dim,80,False)
                if self.gcp_mode == 1: self.fc_gcp = make_fc(int(self.reduce_dim*(self.reduce_dim+1)/2),80,False)
        if self.mix == True:
            self.fc_mix = make_fc(1024,80,False)

        self.classification_loss_func = nn.BCEWithLogitsLoss()

    def _signed_sqrt(self, x):
         x = torch.mul(x.sign(), torch.sqrt(x.abs()+self.thresh))
         return x

    def _l2norm(self, x):
         x = nn.functional.normalize(x)
         return x

    def forward(self, cls_feature, targets, iteration):
        #cpu_device = torch.device("cpu")
        if not self.training: return None
        for l, target in enumerate(targets):
            cls_label = target.extra_fields['labels'] - 1
            cls_target = torch.zeros(80).to(cls_label.device)
            cls_target = cls_target.scatter(0,cls_label,1).unsqueeze(0)
            if l==0 :cls_targets = cls_target
            else: cls_targets = torch.cat((cls_targets, cls_target), dim=0)

        if self.gap == True:
            avg_x = self.avgpool(cls_feature)
            if self.mix == True:
                max_x = self.maxpool(cls_feature)
                gap_x = torch.add(avg_x,max_x)
                gap_x = gap_x.view(gap_x.size(0),-1)
                gap_x = F.relu(self.fc_gap(gap_x),inplace=True)
            else:
                gap_x = avg_x.view(avg_x.size(0),-1)
                x = self.fc_gap(gap_x)
        if self.gcp == True:
            batch_size, dim, h, w = cls_feature.data.shape
            cls_feature = self.layer_reduce(cls_feature)
            if self.gcp_mode == 0:
                cls_feature = cls_feature.reshape(batch_size, self.reduce_dim, h*w)
                gcp_feature = 1./(h*w)*cls_feature.bmm(cls_feature.transpose(1,2))
                gcp_feature = self._signed_sqrt(gcp_feature)
                gcp_feature = gcp_feature.view(gcp_feature.size(0),-1)
                gcp_feature = self._l2norm(gcp_feature)
                if self.mix == True : gcp_x = F.relu(self.fc_gcp(gcp_feature),inplace=True)
                if self.mix == False: x = self.fc_gcp(gcp_feature)
            if self.gcp_mode == 1:
                gcp_feature = MPNCOV.CovpoolLayer(cls_feature)
                gcp_feature = MPNCOV.SqrtmLayer(gcp_feature, 5)
                gcp_feature = MPNCOV.TriuvecLayer(gcp_feature)
                gcp_feature = gcp_feature.view(gcp_feature.size(0), -1)
                if self.mix == True : gcp_x = F.relu(self.fc_gcp(gcp_feature),inplace=True)
                if self.mix == False: x = self.fc_gcp(gcp_feature)

        if self.mix == True:
            x = self.fc_mix(gap_x)
        alpha = 0.5 * (1+np.cos(np.pi*(float(iteration)/float(360000))))
        classification_loss = alpha * self.classification_loss_func(x,cls_targets)

        losses = {"classification_loss": classification_loss}
        return losses

def build_cls_head(gap, gcp, mix):
    return Cls_Head(gap, gcp, mix)




