#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
DDR
jieli_cn@163.com
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from .kk import Bottleneckkk3d


# ----------------------------------------------------------------------
class BasicDDR2d(nn.Module):
    def __init__(self, c, k=3, dilation=1, residual=True):
        super(BasicDDR2d, self).__init__()
        d = dilation
        p = k // 2 * d
        self.conv_1xk = nn.Conv2d(c, c, (1, k), stride=1, padding=(0, p), bias=True, dilation=(1, d))
        self.conv_kx1 = nn.Conv2d(c, c, (k, 1), stride=1, padding=(p, 0), bias=True, dilation=(d, 1))
        self.residual = residual

    def forward(self, x):
        y = self.conv_1xk(x)
        y = F.relu(y, inplace=True)
        y = self.conv_kx1(y)
        y = F.relu(y + x, inplace=True) if self.residual else F.relu(y, inplace=True)
        return y


# ----------------------------------------------------------------------
class BasicDDR3d(nn.Module):
    def __init__(self, c, k=3, dilation=1, stride=1, residual=True):
        super(BasicDDR3d, self).__init__()
        d = dilation
        p = k // 2 * d
        # p = (d * (k - 1) + 1) // 2
        s = stride
        # print("k:{}, d:{}, p:{}".format(k, d, p))
        self.conv_1x1xk = nn.Conv3d(c, c, (1, 1, k), stride=(1, 1, s), padding=(0, 0, p), bias=True, dilation=(1, 1, d))
        self.conv_1xkx1 = nn.Conv3d(c, c, (1, k, 1), stride=(1, s, 1), padding=(0, p, 0), bias=True, dilation=(1, d, 1))
        self.conv_kx1x1 = nn.Conv3d(c, c, (k, 1, 1), stride=(s, 1, 1), padding=(p, 0, 0), bias=True, dilation=(d, 1, 1))
        self.residual = residual

    def forward(self, x):
        y = self.conv_1x1xk(x)
        y = F.relu(y, inplace=True)
        y = self.conv_1xkx1(y)
        y = F.relu(y, inplace=True)
        y = self.conv_kx1x1(y)
        y = F.relu(y + x, inplace=True) if self.residual else F.relu(y, inplace=True)
        return y


class BottleneckDDR2d(nn.Module):
    def __init__(self, c_in, c, c_out, kernel=3, stride=1, dilation=1, residual=True):
        super(BottleneckDDR2d, self).__init__()
        s = stride
        k = kernel
        d = dilation
        p = k // 2 * d
        self.conv_in = nn.Conv2d(c_in, c, kernel_size=1, bias=False)
        self.conv_1xk = nn.Conv2d(c, c, (1, k), stride=s, padding=(0, p), bias=True, dilation=(1, d))
        self.conv_kx1 = nn.Conv2d(c, c, (k, 1), stride=s, padding=(p, 0), bias=True, dilation=(d, 1))
        self.conv_out = nn.Conv2d(c, c_out, kernel_size=1, bias=False)
        self.residual = residual

    def forward(self, x):
        y = self.conv_in(x)
        y = F.relu(y, inplace=True)
        y = self.conv_1xk(y)
        y = F.relu(y, inplace=True)
        y = self.conv_kx1(y)
        y = F.relu(y, inplace=True)
        y = self.conv_out(y)
        y = F.relu(y + x, inplace=True) if self.residual else F.relu(y, inplace=True)
        return y


class Bottleneckconv2d(nn.Module):
    def __init__(self, c_in, c, c_out, kernel=3, stride=1, dilation=1, residual=True):
        super(Bottleneckconv2d, self).__init__()
        s = stride
        k = kernel
        d = dilation
        p = k // 2 * d
        self.conv_in = nn.Conv2d(c_in, c, kernel_size=1, bias=False)
        self.conv = nn.Conv2d(c, c, kernel_size=k, stride=s, padding=p, bias=True, dilation=d)
        self.conv_out = nn.Conv2d(c, c_out, kernel_size=1, bias=False)
        self.residual = residual

    def forward(self, x):
        y = self.conv_in(x)
        y = F.relu(y, inplace=True)
        y = self.conv(y)
        y = F.relu(y, inplace=True)
        y = self.conv_out(y)
        y = F.relu(y + x, inplace=True) if self.residual else F.relu(y, inplace=True)
        return y


class BottleneckDDR3d(nn.Module):   # the i/o sizes are same
    def __init__(self, c_in, c, c_out, kernel=3, stride=1, dilation=1, residual=True):
        super(BottleneckDDR3d, self).__init__()
        s = stride
        k = kernel
        d = dilation
        p = k // 2 * d   # output size keeps the same as input size
        self.conv_in = nn.Conv3d(c_in, c, kernel_size=1, bias=False)
        self.conv1x1x3 = nn.Conv3d(c, c, (1, 1, k), stride=s, padding=(0, 0, p), bias=True, dilation=(1, 1, d))
        self.conv1x3x1 = nn.Conv3d(c, c, (1, k, 1), stride=s, padding=(0, p, 0), bias=True, dilation=(1, d, 1))
        self.conv3x1x1 = nn.Conv3d(c, c, (k, 1, 1), stride=s, padding=(p, 0, 0), bias=True, dilation=(d, 1, 1))
        self.conv_out = nn.Conv3d(c, c_out, kernel_size=1, bias=False)
        self.residual = residual

    def forward(self, x):
        y0 = self.conv_in(x)
        y0 = F.relu(y0, inplace=True)

        y1 = self.conv1x1x3(y0)
        y1 = F.relu(y1, inplace=True)

        y2 = self.conv1x3x1(y1) + y1
        y2 = F.relu(y2, inplace=True)

        y3 = self.conv3x1x1(y2) + y2 + y1
        y3 = F.relu(y3, inplace=True)

        y = self.conv_out(y3)

        y = F.relu(y + x, inplace=True) if self.residual else F.relu(y, inplace=True)
        return y


class DownsampleBlock3d(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=2, p=1):
        super(DownsampleBlock3d, self).__init__()
        self.conv = nn.Conv3d(c_in, c_out-c_in, kernel_size=k, stride=s, padding=p, bias=False)
        self.pool = nn.MaxPool3d(2, stride=2)
        # self.bn = nn.BatchNorm2d(c_out, eps=1e-3)

    def forward(self, x):
        y = torch.cat([self.conv(x), self.pool(x)], 1)
        # y = self.bn(y)
        y = F.relu(y, inplace=True)
        return y


class DDR_ASPP3d(nn.Module):
    def __init__(self, c_in, c, c_out, residual=False):
        super(DDR_ASPP3d, self).__init__()
        print('DDR_ASPP3d: c_in:{}, c:{}, c_out:{}'.format(c_in, c, c_out))

        self.aspp0 = nn.Conv3d(c_in, c_out, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.aspp1 = BottleneckDDR3d(c_in, c, c_out, dilation=3, residual=residual)  # 6

        self.aspp2 = BottleneckDDR3d(c_in, c, c_out, dilation=5, residual=residual)  # 12

        self.aspp3 = BottleneckDDR3d(c_in, c, c_out, dilation=7, residual=residual)  # 18

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),     # target output size of (1,1,1)
                                             nn.Conv3d(c_in, c_out, 1, stride=1, bias=False))

    def forward(self, x):   # (BS, 256L, 60L, 36L, 60L)
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x_ = self.global_avg_pool(x)

        # x_ = F.upsample(x_, size=x.size()[2:], mode='trilinear', align_corners=True)
        x_ = F.interpolate(x_, size=x.size()[2:], mode='trilinear', align_corners=True)
        x = x0 + x1 + x2 + x3 + x_
        # x = torch.cat((x0, x1, x2, x3, x_), dim=1)   # 64 * 5 = 320
        # print(x0.shape, x1.shape, x2.shape, x3.shape, x_.shape, x.shape)
        return x


# Residual block
class BasicResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class DDR_CCP1_ASPP3d(nn.Module):
    def __init__(self, c_in, c, c_out, residual=False):
        super(DDR_CCP1_ASPP3d, self).__init__()
        print('DDR__CCP_ASPP3d: c_in:{}, c:{}, c_out:{}'.format(c_in, c, c_out))

        self.aspp0 = nn.Conv3d(c_in, c_out, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.aspp1 = Bottleneckkk3d(c_in, c, c_out, dilation=3, residual=residual)

        self.aspp2 = Bottleneckkk3d(c_in, c, c_out, dilation=5, residual=residual)

        self.aspp3 = Bottleneckkk3d(c_in, c, c_out, dilation=7, residual=residual)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),     # target output size of (1,1,1)
                                             nn.Conv3d(c_in, c_out, 1, stride=1, bias=False))

        ### CCPNet
        self.aspp_conv = nn.Conv3d(c_out, c_out, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.basic_residual_block = BasicResidualBlock(c_out, c_out)

    def forward(self, x):   # (BS, 256L, 60L, 36L, 60L)
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x01 = torch.add(x0, x1)
        x01 = self.aspp_conv(x01)
        x01 = self.basic_residual_block(x01)

        x2 = self.aspp2(x)
        x012 = torch.add(x01, x2)
        x012 = self.aspp_conv(x012)
        x012 = self.basic_residual_block(x012)

        x3 = self.aspp3(x)
        x0123 = torch.add(x012, x3)
        x0123 = self.aspp_conv(x0123)
        x0123 = self.basic_residual_block(x0123)

        x_ = self.global_avg_pool(x)
        # x_ = F.upsample(x_, size=x.size()[2:], mode='trilinear', align_corners=True)
        x_ = F.interpolate(x_, size=x.size()[2:], mode='trilinear', align_corners=True)
        x01234 = torch.add(x0123, x_)
        x01234 = self.aspp_conv(x01234)
        x01234 = self.basic_residual_block(x01234)

        # x = torch.cat((x0, x1, x2, x3, x_), dim=1)   # 64 * 5 = 320
        # print(x0.shape, x1.shape, x2.shape, x3.shape, x_.shape, x.shape)
        return x01234  # 2, 64, 60, 36, 60

class DDR_CCP2_ASPP3d(nn.Module):
    def __init__(self, c_in, c, c_out, residual=False):
        super(DDR_CCP2_ASPP3d, self).__init__()
        print('DDR__CCP_ASPP3d: c_in:{}, c:{}, c_out:{}'.format(c_in, c, c_out))

        self.aspp0 = nn.Conv3d(c_in, c_out, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.aspp1 = BottleneckDDR3d(c_in, c, c_out, dilation=6, residual=residual)

        self.aspp2 = BottleneckDDR3d(c_in, c, c_out, dilation=12, residual=residual)

        self.aspp3 = BottleneckDDR3d(c_in, c, c_out, dilation=18, residual=residual)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),  # target output size of (1,1,1)
                                             nn.Conv3d(c_in, c_out, 1, stride=1, bias=False))

        ### CCPNet
        self.aspp_conv = nn.Conv3d(c_out, c_out, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.basic_residual_block = BasicResidualBlock(c_out, c_out)

    def forward(self, x):  # (BS, 256L, 60L, 36L, 60L)
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x01 = torch.add(x0, x1)
        x01 = self.aspp_conv(x01)
        x01 = self.basic_residual_block(x01)

        x2 = self.aspp2(x)
        x012 = torch.add(x01, x2)
        x012 = torch.add(x012, x0)
        x012 = self.aspp_conv(x012)
        x012 = self.basic_residual_block(x012)

        x3 = self.aspp3(x)
        x0123 = torch.add(x012, x3)
        x0123 = torch.add(x0123, x01)
        x0123 = self.aspp_conv(x0123)
        x0123 = self.basic_residual_block(x0123)

        x_ = self.global_avg_pool(x)
        # x_ = F.upsample(x_, size=x.size()[2:], mode='trilinear', align_corners=True)
        x_ = F.interpolate(x_, size=x.size()[2:], mode='trilinear', align_corners=True)
        x01234 = torch.add(x0123, x_)
        x01234 = torch.add(x01234, x012)
        x01234 = self.aspp_conv(x01234)
        x01234 = self.basic_residual_block(x01234)

        output = torch.cat(x0, x01, x012, x0123, x01234)  # 64 * 5 = 320
        # x = torch.cat((x0, x1, x2, x3, x_), dim=1)   # 64 * 5 = 320
        # print(x0.shape, x1.shape, x2.shape, x3.shape, x_.shape, x.shape)
        #return x01234  # 2, 64, 60, 36, 60
        return output

# for Seg2DNet
class BasicResidualBlock_2d(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicResidualBlock_2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class DDR_CCP2_ASPP2d(nn.Module):
    def __init__(self, c_in, c, c_out, residual=False):
        super(DDR_CCP2_ASPP2d, self).__init__()
        print('DDR__CCP_ASPP2d: c_in:{}, c:{}, c_out:{}'.format(c_in, c, c_out))

        self.aspp0 = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)

        self.aspp1 = BottleneckDDR2d(c_in, c, c_out, dilation=6, residual=residual)

        self.aspp2 = BottleneckDDR2d(c_in, c, c_out, dilation=12, residual=residual)

        self.aspp3 = BottleneckDDR2d(c_in, c, c_out, dilation=18, residual=residual)

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),  # target output size of (1,1,1)
                                             nn.Conv2d(c_in, c_out, 1, stride=1, bias=False))

        ### CCPNet
        self.aspp_conv = nn.Conv2d(c_out, c_out, kernel_size=1, stride=1, padding=0, dilation=1, bias=False)
        self.basic_residual_block = BasicResidualBlock_2d(c_out, c_out)

    def forward(self, x):  # (BS, 256L, 60L, 36L, 60L)
        x0 = self.aspp0(x)
        x1 = self.aspp1(x)
        x01 = torch.add(x0, x1)
        x01 = self.aspp_conv(x01)
        x01 = self.basic_residual_block(x01)

        x2 = self.aspp2(x)
        x012 = torch.add(x01, x2)
        x012 = self.aspp_conv(x012)
        x012 = self.basic_residual_block(x012)

        x3 = self.aspp3(x)
        x0123 = torch.add(x012, x3)
        x0123 = self.aspp_conv(x0123)
        x0123 = self.basic_residual_block(x0123)

        # x_ = self.global_avg_pool(x)
        # # x_ = F.upsample(x_, size=x.size()[2:], mode='trilinear', align_corners=True)
        # x_ = F.interpolate(x_, size=x.size()[2:], mode='bicubic', align_corners=True)
        # x01234 = torch.add(x0123, x_)
        x01234 = self.aspp_conv(x0123)
        # x01234 = self.basic_residual_ block(x01234)

        return x01234
        # ccp2
        # x0 = self.aspp0(x)
        # x1 = self.aspp1(x)
        # x01 = torch.add(x0, x1)
        # x01 = self.aspp_conv(x01)
        # x01 = self.basic_residual_block(x01)
        #
        # x2 = self.aspp2(x)
        # x012 = torch.add(x01, x2)
        # x012 = torch.add(x012, x0)
        # x012 = self.aspp_conv(x012)
        # x012 = self.basic_residual_block(x012)
        #
        # x3 = self.aspp3(x)
        # x0123 = torch.add(x012, x3)
        # x0123 = torch.add(x0123, x01)
        # x0123 = self.aspp_conv(x0123)
        # x0123 = self.basic_residual_block(x0123)
        #
        # x_ = self.global_avg_pool(x)
        # # x_ = F.upsample(x_, size=x.size()[2:], mode='trilinear', align_corners=True)
        # x_ = F.interpolate(x_, size=x.size()[2:], mode='bicubic', align_corners=True)  # 'trilinear'
        # x01234 = torch.add(x0123, x_)
        # x01234 = torch.add(x01234, x012)
        # x01234 = self.aspp_conv(x01234)
        # x01234 = self.basic_residual_block(x01234)
        #
        # output = torch.cat((x0, x01, x012, x0123, x01234))
        # # x = torch.cat((x0, x1, x2, x3, x_), dim=1)
        # #return x01234  # 2, 64, 60, 36, 60
        # return output