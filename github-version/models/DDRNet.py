#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
DDRNet
jieli_cn@163.com
"""

import torch
import torch.nn as nn
from .projection_layer import Project2Dto3D
from .DDR import DDR_ASPP3d, DDR_CCP1_ASPP3d, DDR_CCP2_ASPP3d
from .DDR import BottleneckDDR2d, BottleneckDDR3d, DownsampleBlock3d
from .visual import voxel_complete_ply, voxel_complete_edge_ply, canny_edge_detector, _downsample_label
from torchvision.transforms import ToTensor
import numpy as np
import time
from .Disp_vgg_BN import Disp_vgg_BN  # depth estimation models
import torch.nn.functional as F


# DDRNet
# ----------------------------------------------------------------------
class SSC_RGBD_DDRNet(nn.Module):
    def __init__(self, num_classes=12):
        super(SSC_RGBD_DDRNet, self).__init__()
        print('SSC_RGBD_DDRNet: RGB and Depth streams with DDR blocks for Semantic Scene Completion')

        w, h, d = 240, 144, 240
        # --- define depth estimation model
        # self.disp_net = Disp_vgg_BN().cuda()
        # --- depth
        c_in, c, c_out, dilation, residual = 1, 4, 8, 1, True  # 1
        self.dep_feature2d = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, 1, 0),  # reduction
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
        )
        self.project_layer_dep = Project2Dto3D(w, h, d)  # w=240, h=144, d=240
        self.dep_feature3d = nn.Sequential(
            DownsampleBlock3d(8, 20),
            BottleneckDDR3d(c_in=20, c=4, c_out=20, dilation=1, residual=True),
            DownsampleBlock3d(20, 64),  # nn.MaxPool3d(kernel_size=2, stride=2)
            BottleneckDDR3d(c_in=64, c=16, c_out=64, dilation=1, residual=True),
        )

        # --- RGB
        self.to_tensor = ToTensor()
        c_in, c, c_out, dilation, residual = 3, 4, 8, 1, True
        self.rgb_feature2d = nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, 1, 0),  # reduction
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
            BottleneckDDR2d(c_out, c, c_out, dilation=dilation, residual=residual),
        )
        self.project_layer_rgb = Project2Dto3D(w, h, d)  # w=240, h=144, d=240
        self.project_layer_fuse = Project2Dto3D(60, 36, 60)

        self.rgb_feature3d = nn.Sequential(
            DownsampleBlock3d(8, 20),   # 8, 16
            BottleneckDDR3d(c_in=20, c=4, c_out=20, dilation=1, residual=True),
            DownsampleBlock3d(20, 64),  # nn.MaxPool3d(kernel_size=2, stride=2)
            BottleneckDDR3d(c_in=64, c=16, c_out=64, dilation=1, residual=True),
        )

        # -------------1/4

        # ck = 256
        # self.ds = DownsamplerBlock_3d(64, ck)
        ck = 64
        c = 16
        # c_in, c, c_out, kernel=3, stride=1, dilation=1, residual=True
        self.res3d_1d = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=2, residual=True)
        self.res3d_2d = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=3, residual=True)
        self.res3d_3d = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=5, residual=True)

        self.res3d_1r = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=2, residual=True)
        self.res3d_2r = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=3, residual=True)
        self.res3d_3r = BottleneckDDR3d(c_in=ck, c=c, c_out=ck, kernel=3, dilation=5, residual=True)

        self.aspp = DDR_ASPP3d(c_in=int(ck * 4), c=16, c_out=64)
        # self.aspp = DDR_ASPP3d(c_in=int(ck * 4), c=64, c_out=int(ck * 4))

        # 64 * 5 = 320
        self.conv_out = nn.Sequential(
            nn.Conv3d(320, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, num_classes, 1, 1, 0)
        )

        self.pre_conv4 = nn.Sequential(
            nn.Conv3d(in_channels=16, out_channels=num_classes,
                      kernel_size=1, stride=1))

        # self.edge_downsample = _downsample_label()

        # ----  weights init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight.data)  # gain=1
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0, std=0.1)

    def forward(self, x_depth=None, x_rgb=None, pred_depth=None, p=None, p4=None):
        H, W = x_depth.size()[2:]  # H, W: 480 640
        # depth estimation
        # disp0, disp1, disp2, disp3 = self.disp_net(x_rgb)
        # estimated_depth0 = 1 / disp0
        # estimated_depth1 = 1 / disp1
        # estimated_depth2 = 1 / disp2
        # estimated_depth3 = 1 / disp3
        # estimated_depth0_up = F.interpolate(estimated_depth0, (H, W), mode='bilinear', align_corners=True)
        # estimated_depth1_up = F.interpolate(estimated_depth1, (H, W), mode='bilinear', align_corners=True)
        # estimated_depth2_up = F.interpolate(estimated_depth2, (H, W), mode='bilinear', align_corners=True)
        # estimated_depth3_up = F.interpolate(estimated_depth3, (H, W), mode='bilinear', align_corners=True)

        # depth_concat = torch.cat(
        #     (estimated_depth0_up, estimated_depth1_up, estimated_depth2_up, estimated_depth3_up, x_depth), 1)
        pred_depth = torch.unsqueeze(pred_depth, 1)
        # y3d = self.project_layer_dep(pred_depth.expand(-1, 8, -1, -1), p)
        # y3d = self.dep_feature3d(y3d)

        x0_rgb = self.rgb_feature2d(x_rgb)  # torch.Size([1, 8, 480, 640])
        x0_rgb = self.project_layer_rgb(x0_rgb, p)
        x0_rgb = self.rgb_feature3d(x0_rgb)    # torch.Size([1, 64, 60, 36, 60])

        y = self.dep_feature2d(pred_depth)

        x0_depth = self.dep_feature2d(x_depth)  # x_depth
        x0_depth = self.project_layer_dep(x0_depth + y, p)
        x0_depth = self.dep_feature3d(x0_depth)

        f0 = torch.add(x0_depth, x0_rgb)  # torch.Size([1, 64, 60, 36, 60])
        # f0 = torch.add(f0, y3d)
        # f0 = torch.add(f0, edges_ds)

        x_4_d = self.res3d_1d(x0_depth)
        x_4_r = self.res3d_1r(x0_rgb)

        f1 = torch.add(x_4_d, x_4_r)     # torch.Size([1, 64, 60, 36, 60])

        x_5_d = self.res3d_2d(x_4_d)
        x_5_r = self.res3d_2r(x_4_r)

        f2 = torch.add(x_5_d, x_5_r)    # torch.Size([1, 64, 60, 36, 60])

        x_6_d = self.res3d_3d(x_5_d)
        x_6_r = self.res3d_3r(x_5_r)

        f3 = torch.add(x_6_d, x_6_r)   # torch.Size([1, 64, 60, 36, 60])

        x = torch.cat((f0, f1, f2, f3), dim=1)  # channels concatenate
        # print('SSC: channels concatenate x', x.size())  # (BS, 256L, 60L, 36L, 60L)

        x = self.aspp(x)
        out = self.conv_out(x)  # (BS, 12L, 60L, 36L, 60L)

        # projected depth
        voxel_size_lr = [60, 36, 60]
        px = p4 / (voxel_size_lr[1] * voxel_size_lr[2])  # x,y,z shape torch.Size([8, 480, 640])
        py = (p4 - px * voxel_size_lr[1] * voxel_size_lr[2]) / voxel_size_lr[2]
        pz = p4 - px * voxel_size_lr[1] * voxel_size_lr[2] - py * voxel_size_lr[2]
        bs = torch.zeros(out.shape[0])
        for i in range(out.shape[0]):
            bs[i] = i
        bs = torch.unsqueeze(bs, 1)
        bs = torch.unsqueeze(bs, 2)
        bs = bs.expand(-1, 480, 640)  # torch.Size([2, 480, 640])
        bs = bs.long().cuda()
        depth_scale = 4 * 0.02
        _, x_copy = torch.max(x, 1)
        x_copy[x_copy > 0] = 1
        x_copy = x_copy.T
        proj_depth = x_copy[px, py, pz, bs] * pz * depth_scale  # torch.Size([2, 480, 640])
        proj_depth = torch.unsqueeze(proj_depth, 1)
        fusion_depth = self.project_layer_fuse(proj_depth + x_depth + pred_depth, p4)
        # fusion_depth = fusion_depth.repeat(1, 16, 1, 1, 1)
        fusion_depth = fusion_depth.expand(-1, 16, -1, -1, -1)
        fusion_depth = self.pre_conv4(fusion_depth)
        aux1 = fusion_depth
        aux2 = out + fusion_depth
        return out, aux1, aux2

