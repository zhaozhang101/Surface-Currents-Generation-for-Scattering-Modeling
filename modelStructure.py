import os
from math import exp
import torch
import torch.nn as nn
from model import EqualLinear,StyledConv,ToRGB,ConvLayer,ResBlock
import torch.nn.functional as F
import torch
import torch.onnx
from torchviz import make_dot


# 生成模型框图
class A_Generator(nn.Module):
    def __init__(self):
        super(A_Generator, self).__init__()
        self.part1 = nn.Sequential(
            ResBlock(1, 16, downsample=True),  # b 8 128 128
            ResBlock(16, 16 ,downsample=True)  # b 16 64 64
        )
        self.part2 = nn.Sequential(
            nn.Linear(10,16*70*70),
            nn.ReLU(),
            nn.BatchNorm1d(16*70*70)
        )
        self.combina_Ixrel = nn.Sequential(
            ResBlock(32,128,downsample=False),
            ResBlock(128, 256, downsample=False),
            StyledConv(256,64,style_dim=None,kernel_size=5,upsample=True,use_style=False),
            StyledConv(64,1,style_dim=None,kernel_size=5,upsample=True,use_style=False)
        )
        self.combina_Iyrel = nn.Sequential(
            ResBlock(32, 128, downsample=False),
            ResBlock(128, 256, downsample=False),
            StyledConv(256, 64, style_dim=None, kernel_size=5, upsample=True, use_style=False),
            StyledConv(64, 1, style_dim=None, kernel_size=5, upsample=True, use_style=False)
        )
        self.combina_Izrel = nn.Sequential(
            ResBlock(32, 128, downsample=False),
            ResBlock(128, 256, downsample=False),
            StyledConv(256, 64, style_dim=None, kernel_size=5, upsample=True, use_style=False),
            StyledConv(64, 1, style_dim=None, kernel_size=5, upsample=True, use_style=False)
        )
        self.combina_Iximg = nn.Sequential(
            ResBlock(32, 128, downsample=False),
            ResBlock(128, 256, downsample=False),
            StyledConv(256, 64, style_dim=None, kernel_size=5, upsample=True, use_style=False),
            StyledConv(64, 1, style_dim=None, kernel_size=5, upsample=True, use_style=False)
        )
        self.combina_Iyimg = nn.Sequential(
            ResBlock(32, 128, downsample=False),
            ResBlock(128, 256, downsample=False),
            StyledConv(256, 64, style_dim=None, kernel_size=5, upsample=True, use_style=False),
            StyledConv(64, 1, style_dim=None, kernel_size=5, upsample=True, use_style=False)
        )
        self.combina_Izimg = nn.Sequential(
            ResBlock(32, 128, downsample=False),
            ResBlock(128, 256, downsample=False),
            StyledConv(256, 64, style_dim=None, kernel_size=5, upsample=True, use_style=False),
            StyledConv(64, 1, style_dim=None, kernel_size=5, upsample=True, use_style=False)
        )
        self.tmp = nn.Sequential(
            ResBlock()
        )

    def forward(self,surf,inci):
        # b 1 280 280    # 1 10
        surf = self.part1(surf)
        inci = self.part2(inci).view(-1,16,70,70)
        out = torch.cat([surf,inci],dim=1)
        outIxrel = self.combina_Ixrel(out)
        outIyrel = self.combina_Iyrel(out)
        outIzrel = self.combina_Izrel(out)
        outIximg = self.combina_Iximg(out)
        outIyimg = self.combina_Iyimg(out)
        outIzimg = self.combina_Izimg(out)
        return outIxrel,outIyrel,outIzrel,outIximg,outIyimg,outIzimg

# model = A_Generator().eval()
#
#
# # 将PyTorch模型导出为ONNX格式
# surf = torch.randn(size=(1,1,280,280))
# inci = torch.rand(size=(1,10))
#
#
# y = model(surf,inci)  # 获取网络的预测值
#
# MyConvNetVis = make_dot(y, params=dict(list(model.named_parameters()) + [('surf', surf)]))
# MyConvNetVis.format = "png"
# # 指定文件生成的文件夹
# MyConvNetVis.directory = "data"
# # 生成文件
# MyConvNetVis.view()