# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from collections import OrderedDict

from torch import nn

from fcos_core.modeling import registry
from fcos_core.modeling.make_layers import conv_with_kaiming_uniform
from . import fpn as fpn_module
from . import resnet
from . import mobilenet
from . import hrfpn as hrfpn_module
from . import hrnet

@registry.BACKBONES.register("R-50-C4")
@registry.BACKBONES.register("R-50-C5")
@registry.BACKBONES.register("R-101-C4")
@registry.BACKBONES.register("R-101-C5")
def build_resnet_backbone(cfg):
    body = resnet.ResNet(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    return model


@registry.BACKBONES.register("R-50-FPN")
@registry.BACKBONES.register("R-101-FPN")
@registry.BACKBONES.register("R-152-FPN")
def build_resnet_fpn_backbone(cfg):
    body = resnet.ResNet(cfg)
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            in_channels_stage2,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelMaxPool(),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


@registry.BACKBONES.register("R-18-FPN-RETINANET")
@registry.BACKBONES.register("R-34-FPN-RETINANET")
@registry.BACKBONES.register("R-50-FPN-RETINANET")
@registry.BACKBONES.register("R-101-FPN-RETINANET")
def build_resnet_fpn_p3p7_backbone(cfg):
    in_channels_stage2 = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    cls_in_channels_stage2 = cfg.MODEL.RESNETS.CLS_RES2_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    in_channels_p6p7 = in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    cls_in_channels_p6p7 = cls_in_channels_stage2 * 8 if cfg.MODEL.RETINANET.USE_C5 \
        else out_channels
    cls_resnet = resnet.ResNet(cfg, 'CLS')
    cls_fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            cls_in_channels_stage2 * 2,
            cls_in_channels_stage2 * 4,
            cls_in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(cls_in_channels_p6p7, out_channels),
        last_backbone = True,
        last_fpn = False
    )
    reg_resnet = resnet.ResNet(cfg, 'REG')
    reg_fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(in_channels_p6p7, out_channels),
        last_backbone = False,
        last_fpn = False
    )
    #if cfg.MODEL.BACKBONE.REG_CONV_BODY == "HRNET-W18":
    #    width = 18
    #elif cfg.MODEL.BACKBONE.REG_CONV_BODY == "HRNET-W32":
    #    width = 32
    #elif cfg.MODEL.BACKBONE.REG_CONV_BODY == "HRNET-W40":
    #    width = 40
    #else:
    #    raise NotImplementedError
    #hrnet_args = dict(
    #    stage1=dict(
    #        num_modules=1,
    #        num_branches=1,
    #        block='BOTTLENECK',
    #        num_blocks=(4,),
    #        num_channels=(64,),
    #        fuse_method='SUM'),
    #    stage2=dict(
    #        num_modules=1,
    #        num_branches=2,
    #        block='BASIC',
    #        num_blocks=(4, 4),
    #        num_channels=(width, width * 2),
    #        fuse_method='SUM'),
    #    stage3=dict(
    #        num_modules=4,
    #        num_branches=3,
    #        block='BASIC',
    #        num_blocks=(4, 4, 4),
    #        num_channels=(width, width * 2, width * 4),
    #        fuse_method='SUM'),
    #    stage4=dict(
    #        num_modules=3,
    #        num_branches=4,
    #        block='BASIC',
    #        num_blocks=(4, 4, 4, 4),
    #        num_channels=(width, width * 2, width * 4, width * 8),
    #        fuse_method='SUM')
    #)
    #fpn_in_channels = [width, width * 2, width * 4, width * 8]

    #hrnet.BatchNorm2d = nn.SyncBatchNorm if cfg.MODEL.SYNCBN else nn.BatchNorm2d
    #reg_hrnet = hrnet.HighResolutionNet(extra=hrnet_args)
    #reg_fpn = getattr(hrfpn_module, cfg.MODEL.HRNET.FPN.TYPE)(
    #    in_channels=fpn_in_channels,
    #    out_channels=cfg.MODEL.HRNET.FPN.OUT_CHANNEL,
    #    conv_stride=cfg.MODEL.HRNET.FPN.CONV_STRIDE,
    #    num_level=len(cfg.MODEL.FCOS.FPN_STRIDES),
    #)
    cls_backbone = nn.Sequential(OrderedDict([("cls_backbone",cls_resnet),("cls_fpn",cls_fpn)]))
    cls_backbone.out_channels = out_channels
    reg_backbone = nn.Sequential(OrderedDict([("reg_backbone", reg_resnet), ("reg_fpn", reg_fpn)]))
    #reg_backbone = nn.Sequential(OrderedDict([("reg_backbone", reg_hrnet), ("reg_fpn", reg_fpn)]))
    reg_backbone.out_channels = out_channels
    return cls_backbone, reg_backbone


@registry.BACKBONES.register("MNV2-FPN-RETINANET")
def build_mnv2_fpn_backbone(cfg):
    body = mobilenet.MobileNetV2(cfg)
    in_channels_stage2 = body.return_features_num_channels
    out_channels = cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
    fpn = fpn_module.FPN(
        in_channels_list=[
            0,
            in_channels_stage2[1],
            in_channels_stage2[2],
            in_channels_stage2[3],
        ],
        out_channels=out_channels,
        conv_block=conv_with_kaiming_uniform(
            cfg.MODEL.FPN.USE_GN, cfg.MODEL.FPN.USE_RELU
        ),
        top_blocks=fpn_module.LastLevelP6P7(out_channels, out_channels),
    )
    model = nn.Sequential(OrderedDict([("body", body), ("fpn", fpn)]))
    model.out_channels = out_channels
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)
