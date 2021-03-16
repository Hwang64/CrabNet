# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn

from fcos_core.structures.image_list import to_image_list
import torch.nn.functional as F
from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads
from fcos_core.modeling.make_layers import conv_with_kaiming_uniform
from fcos_core.layers import Conv2d
from fcos_core.layers import DFConv2d
from fcos_core.modeling.detector.cls_head import build_cls_head
from fcos_core.modeling.detector.offset_head import build_offset_head

class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNN, self).__init__()

        self.extra = True
        self.backbone = build_backbone(cfg)
        self.cls_backbone = self.backbone[0]
        self.reg_backbone = self.backbone[1]
        if self.extra == False: self.rpn = build_rpn(cfg, self.cls_backbone.out_channels, self.extra)
        if self.extra == True:  self.rpn, self.extra_rpn = build_rpn(cfg, self.cls_backbone.out_channels, self.extra)
        self.roi_heads = build_roi_heads(cfg, self.cls_backbone.out_channels)
        self.cfg = cfg
        self.deform_conv_cls = DFConv2d(256,256,with_modulated_dcn=False,kernel_size=3,stride=1,groups=1,dilation=1,deformable_groups=1,bias=False)
        self.deform_conv_reg = DFConv2d(256,256,with_modulated_dcn=False,kernel_size=3,stride=1,groups=1,dilation=1,deformable_groups=1,bias=False)
        self.cls_head = build_cls_head(True,False,False)
        conv_wo_relu   = conv_with_kaiming_uniform(False,True)
        self.cls_weight_conv = conv_wo_relu(256,1,1)
        self.reg_weight_conv = conv_wo_relu(256,1,1)

    def forward(self, arguments, images, targets=None):
        """
        Arguments:
            images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        outputs = []
        final_proposals = []
        if self.training: iteration = arguments["iteration"]
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        images = to_image_list(images)
        cls_features, last_backbone_cls = self.cls_backbone(images.tensors)
        reg_features = self.reg_backbone(images.tensors)

        for cls_feature, reg_feature in zip(cls_features, reg_features):

            cls_feature_align = self.deform_conv_cls(cls_feature)
            reg_feature_align = self.deform_conv_reg(reg_feature)

            cls_feature_weight =  F.sigmoid(self.cls_weight_conv(cls_feature_align))
            reg_feature_weight =  F.sigmoid(self.reg_weight_conv(reg_feature_align))

            cls_output = torch.add(torch.mul(reg_feature_align,reg_feature_weight),cls_feature_align)
            reg_output = torch.add(torch.mul(cls_feature_align,cls_feature_weight),reg_feature_align)
            output = torch.add(cls_output,reg_output)
            outputs.append(output)

        outputs = tuple(outputs)
        proposals, proposal_losses = self.rpn(images, outputs, outputs, targets)
        if self.extra == True:
            if self.training: extra_proposals, extra_proposal_losses = self.extra_rpn(images, None, reg_features, targets, iteration)
        if self.training: cls_head_losses = self.cls_head(last_backbone_cls, targets, iteration)
        if self.roi_heads:
            x_cls, x_reg, result, detector_losses = self.roi_heads(features, features, proposals, targets)
        else:
            # RPN-only models don't have roi_heads
            result = proposals
            detector_losses = {}

        if self.training:
            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            if self.extra == True: losses.update(extra_proposal_losses)
            losses.update(cls_head_losses)
            return losses

        return result
