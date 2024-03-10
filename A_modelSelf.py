import copy
import os
from math import exp
import math
import torch.nn as nn
from model import EqualLinear, StyledConv, ToRGB, ConvLayer, ResBlock
import torch.nn.functional as F
import torch
import torch.onnx
from torchviz import make_dot


# class Decoder(nn.Module):
#     def __init__(self, latent_dim=4 * 4, style_dim=512, n_mlp=4):
#         super().__init__()
#         lr_mlp = 0.01
#         mapping = [EqualLinear(latent_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu')]
#         for i in range(n_mlp - 1):
#             mapping.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'))
#         self.mapping = nn.Sequential(*mapping)
#         in_channel = 512
#         ch_deco = {
#             6: 265,
#             7: 128,
#             8: 64,
#             9: 32,
#             10: 1,
#         }
#         self.convs = nn.ModuleList()
#         blur_kernel = [1, 3, 3, 1]
#         for i in range(6, 11, 1):  # 6 7 8 9 10
#             out_channel = ch_deco[i]  # 256 128 64 32 1
#             self.convs.append(
#                 StyledConv(
#                     in_channel,
#                     out_channel,
#                     3,
#                     style_dim,
#                     upsample=True,
#                     blur_kernel=blur_kernel,
#                 )
#             )
#             self.convs.append(
#                 StyledConv(
#                     out_channel, out_channel, 3, style_dim, upsample=False,blur_kernel=blur_kernel
#                 )
#             )
#             self.convs.append(ToRGB(in_channel=out_channel, style_dim=style_dim, upsample=True))
#             in_channel = out_channel
#
#
#         # convsurf.append(StyledConv(8, 512, kernel_size=3, style_dim=512,use_style=False))
#         self.convsurf = nn.ModuleList()
#         self.convsurf.append(StyledConv(512, 512, kernel_size=3, style_dim=512, use_style=False))
#         self.to_rgb = ToRGB(in_channel=512, style_dim=512)
#         conv = []
#         conv.append(ConvLayer(1,8,kernel_size=3,downsample=False))
#         conv.append(ConvLayer(8,512,kernel_size=3,downsample=True))
#         self.conv = nn.Sequential(*conv)
#
#     def forward(self, code, surface):  # b 24
#         style = self.mapping(code)  # b 512
#         content = self.conv(surface) # b 512 128 128
#         out = self.convsurf[0](content, style)  # 512 256 256
#         skip = self.to_rgb(out, style)
#
#         for conv1, conv2, to_rgb in zip(self.convs[::3], self.convs[1::3], self.convs[2::3]):
#             out = conv1(out, style)
#             out = conv2(out, style)
#             skip = to_rgb(out, style, skip)
#         image = skip
#         return image

# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


# 创建高斯核，通过两个一维高斯分布向量进行矩阵乘法得到
# 可以设定channel参数拓展为3通道
def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


# Classes to re-use window
class SSIM(nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window.to(img1.device)
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return self.ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

    # 计算SSIM
    # 直接使用SSIM的公式，但是在计算均值时，不是直接求像素平均值，而是采用归一化的高斯核卷积来代替。
    # 在计算方差和协方差时用到了公式Var(X)=E[X^2]-E[X]^2, cov(X,Y)=E[XY]-E[X]E[Y].
    # 正如前面提到的，上面求期望的操作采用高斯核卷积代替。
    def ssim(self, img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
        # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
        if val_range is None:
            if torch.max(img1) > 128:
                max_val = 255
            else:
                max_val = 1

            if torch.min(img1) < -0.5:
                min_val = -1
            else:
                min_val = 0
            L = max_val - min_val
        else:
            L = val_range

        padd = 0
        (_, channel, height, width) = img1.size()
        if window is None:
            real_size = min(window_size, height, width)
            window = create_window(real_size, channel=channel).to(img1.device)

        mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
        mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

        C1 = (0.01 * L) ** 2
        C2 = (0.03 * L) ** 2

        v1 = 2.0 * sigma12 + C2
        v2 = sigma1_sq + sigma2_sq + C2
        cs = torch.mean(v1 / v2)  # contrast sensitivity

        ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

        if size_average:
            ret = ssim_map.mean()
        else:
            ret = ssim_map.mean(1).mean(1).mean(1)

        if full:
            return ret, cs
        return ret


class A_Decoder(nn.Module):
    def __init__(self):
        super(A_Decoder, self).__init__()
        self.IxModule = nn.ModuleList()
        self.IxModule.append(ConvLayer(1, 8, kernel_size=3, downsample=True))
        self.IxModule.append(StyledConv(8, 1, kernel_size=3, style_dim=30, upsample=True))

        self.IyModule = nn.ModuleList()
        self.IyModule.append(ConvLayer(1, 8, kernel_size=3, downsample=True))
        self.IyModule.append(StyledConv(8, 1, kernel_size=3, style_dim=30, upsample=True))

        self.IzModule = nn.ModuleList()
        self.IzModule.append(ConvLayer(1, 8, kernel_size=3, downsample=True))
        self.IzModule.append(StyledConv(8, 1, kernel_size=3, style_dim=30, upsample=True))

    def forward(self, input, style):
        IxOut = self.IxModule[0](input)
        IxOut = self.IxModule[1](IxOut, style)

        IyOut = self.IyModule[0](input)
        IyOut = self.IyModule[1](IyOut, style)

        IzOut = self.IzModule[0](input)
        IzOut = self.IzModule[1](IzOut, style)
        return IxOut, IyOut, IzOut


class A_Encoder(nn.Module):
    def __init__(self):
        super(A_Encoder, self).__init__()
        self.CodeModule = []
        self.CodeModule.append(ConvLayer(3, 32, kernel_size=3, downsample=True))
        self.CodeModule.append(ConvLayer(32, 64, kernel_size=3, downsample=True))
        self.CodeModule.append(ConvLayer(64, 64, kernel_size=3, downsample=True))
        self.CodeModule.append(ConvLayer(64, 128, kernel_size=3, downsample=True))
        self.CodeModule.append(ConvLayer(128, 256, kernel_size=3, downsample=True))
        self.CodeModule.append(ConvLayer(256, 512, kernel_size=3, downsample=True))
        self.CodeModule.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.CodeModule = nn.Sequential(*self.CodeModule)
        self.LinRMSheight = EqualLinear(512, 10)
        self.LinRelalength = EqualLinear(512, 10)
        self.Linincid = EqualLinear(512, 10)

    def forward(self, IxPart, IyPart, IzPart):
        Combine = torch.cat([IxPart, IyPart, IzPart], dim=1)
        code = self.CodeModule(Combine).squeeze()
        RMSheight, Relalength, incid = self.LinRMSheight(code), self.LinRelalength(code), self.Linincid(code)
        return RMSheight, Relalength, incid


class A_DiscriminatorInput(nn.Module):
    def __init__(self):
        super(A_DiscriminatorInput, self).__init__()
        # 关键是 code编码是否正确；
        list = []
        list.append(EqualLinear(30, 128))
        list.append(EqualLinear(128, 256))
        list.append(EqualLinear(256, 64))
        list.append(EqualLinear(64, 1))
        self.list = nn.Sequential(*list)

    def forward(self, RMSheight, Relalegth, Inci):
        code = torch.cat([RMSheight, Relalegth, Inci], dim=1)
        g_adv = self.list(code)
        return g_adv


class A_DiscriminatorOutput(nn.Module):
    def __init__(self):
        super(A_DiscriminatorOutput, self).__init__()
        # 关键是 code编码是否正确；
        list = []
        list.append(ConvLayer(3, 32, kernel_size=3, downsample=True))
        list.append(ConvLayer(32, 64, kernel_size=3, downsample=True))
        list.append(ConvLayer(64, 64, kernel_size=3, downsample=True))
        list.append(ConvLayer(64, 128, kernel_size=3, downsample=True))
        list.append(ConvLayer(128, 256, kernel_size=3, downsample=True))
        list.append(ConvLayer(256, 512, kernel_size=3, downsample=True))
        list.append(nn.MaxPool2d((1, 1)))
        self.list = nn.Sequential(*list)

    def forward(self, Ix, Iy, Iz):
        I_curt = torch.cat([Ix, Iy, Iz], dim=1)
        g_adv = self.list(I_curt).squeeze()
        return g_adv


class AtrousConv(nn.Module):
    def __init__(self, in_channel, out_channel, factor=2):
        super(AtrousConv, self).__init__()
        self.factor = factor
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel // 2, kernel_size=3, stride=2, padding=1,
                      dilation=1),
            nn.BatchNorm2d(out_channel // 2),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel // 4, kernel_size=3, stride=2, padding=2,
                      dilation=2),
            nn.BatchNorm2d(out_channel // 4),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel // 8, kernel_size=3, stride=2, padding=4,
                      dilation=4),
            nn.BatchNorm2d(out_channel // 8),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel // 8, kernel_size=3, stride=2, padding=8,
                      dilation=8),
            nn.BatchNorm2d(out_channel // 8),
            nn.ReLU(inplace=True))

    def forward(self, input):
        out1 = self.conv1(input)
        out2 = self.conv2(input)
        out3 = self.conv3(input)
        out4 = self.conv4(input)
        out = torch.concatenate([out1, out2, out3, out4], dim=1)
        if self.factor == 4:
            out = self.maxpool(out)
        return out


class A_Generator(nn.Module):
    def __init__(self):
        super(A_Generator, self).__init__()
        self.part_surf = nn.Sequential(  # 替换空洞卷积
            ResBlock(1, 4, downsample=True, factor=4),  # b 8 164 164
            ResBlock(4, 8, downsample=True, factor=2)  # b 16 82 82
        )
        self.part_Azi = nn.Sequential(
            nn.Linear(8, 4 * 41 * 41),
            nn.ReLU(),
            nn.BatchNorm1d(4 * 41 * 41)
        )
        self.part_mask = nn.Sequential(
            AtrousConv(1, 8, factor=2),
            AtrousConv(8, 8, factor=4)
        )
        self.part_Zen = nn.Sequential(
            nn.Linear(7, 4 * 41 * 41),
            nn.ReLU(),
            nn.BatchNorm1d(4 * 41 * 41)
        )
        self.style_azi = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.style_zen = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )
        self.combina = nn.ModuleList([
            ResBlock(24, 128, downsample=False),
            ResBlock(128, 256, downsample=False),
            StyledConv(256, 128, style_dim=64, kernel_size=3, upsample=True, use_style=True),
            StyledConv(128, 64, style_dim=64, kernel_size=3, upsample=True, use_style=True),
            StyledConv(64, 6, style_dim=64, kernel_size=3, upsample=True, use_style=True)]
        )
        self.combina_Ixrel = nn.ModuleList([
            ResBlock(6, 128, downsample=True),
            StyledConv(128, 64, style_dim=64, kernel_size=3, upsample=True, use_style=True),
            StyledConv(64, 1, style_dim=64, kernel_size=3, upsample=False, use_style=True)]
        )
        self.combina_Iyrel = nn.ModuleList([
            ResBlock(6, 128, downsample=True),
            StyledConv(128, 64, style_dim=64, kernel_size=3, upsample=True, use_style=True),
            StyledConv(64, 1, style_dim=64, kernel_size=3, upsample=False, use_style=True)]
        )
        self.combina_Izrel = nn.ModuleList([
            ResBlock(6, 128, downsample=True),
            StyledConv(128, 64, style_dim=64, kernel_size=3, upsample=True, use_style=True),
            StyledConv(64, 1, style_dim=64, kernel_size=3, upsample=False, use_style=True)]
        )
        self.combina_Iximg = nn.ModuleList([
            ResBlock(6, 128, downsample=True),
            StyledConv(128, 64, style_dim=64, kernel_size=3, upsample=True, use_style=True),
            StyledConv(64, 1, style_dim=64, kernel_size=3, upsample=False, use_style=True)]
        )
        self.combina_Iyimg = nn.ModuleList([
            ResBlock(6, 128, downsample=True),
            StyledConv(128, 64, style_dim=64, kernel_size=3, upsample=True, use_style=True),
            StyledConv(64, 1, style_dim=64, kernel_size=3, upsample=False, use_style=True)]
        )
        self.combina_Izimg = nn.ModuleList([
            ResBlock(6, 128, downsample=True),
            StyledConv(128, 64, style_dim=64, kernel_size=3, upsample=True, use_style=True),
            StyledConv(64, 1, style_dim=64, kernel_size=3, upsample=False, use_style=True)]
        )

    def mkMask(self, azi, zen):
        distance = 145
        zenAssist = torch.tensor([10, 20, 30, 40, 50, 60, 70],device=zen.device)
        aziAssist = torch.tensor([0, 45, 90, 135, 180, 225, 270, 315],device=azi.device)
        azi, zen = torch.sum(azi * aziAssist, dim=1, keepdim=True), torch.sum(zen * zenAssist, dim=1, keepdim=True)
        Sx = distance * torch.sin(zen / 180 * torch.pi) * torch.cos(azi / 180 * torch.pi)
        Sy = distance * torch.sin(zen / 180 * torch.pi) * torch.sin(azi / 180 * torch.pi)
        Sz = distance * torch.cos(zen / 180 * torch.pi)

        x = torch.linspace(-17.5, 17.5, 328)
        y = torch.linspace(-17.5, 17.5, 328)
        xx, yy = torch.meshgrid(x, y)
        zz = torch.zeros_like(xx)
        Surf_mat = torch.concatenate((xx.unsqueeze(0), yy.unsqueeze(0), zz.unsqueeze(0)), dim=0).to(azi.device)

        vec1 = torch.concatenate([Sx, Sy, Sz], dim=1).unsqueeze(2).unsqueeze(2)
        vec1 = torch.tile(vec1, (1, 1, 328, 328))
        vec2 = vec1 - Surf_mat

        dot_product = torch.sum(vec1 * vec2, dim=1, keepdim=True)
        norm_A = torch.norm(vec1, dim=1, keepdim=True)
        norm_B = torch.norm(vec2, dim=1, keepdim=True)
        cos_theta = dot_product / (norm_A * norm_B)
        cos_theta[cos_theta < torch.cos(2 / 180 * torch.tensor(torch.pi))] = 0
        cos_theta[cos_theta >= torch.cos(2 / 180 * torch.tensor(torch.pi))] = 1
        return cos_theta

    def forward(self, surf, azi, zen):
        # b 1 328 328    # 1 10
        surfed = self.part_surf(surf)  # b 16 64 64
        mask = self.mkMask(azi, zen)
        masked = self.part_mask(mask)
        inci_azi = self.part_Azi(azi).view(-1, 4, 41, 41)
        inci_zen = self.part_Zen(zen).view(-1, 4, 41, 41)
        out = torch.cat([surfed, masked, inci_azi, inci_zen], dim=1)
        sty_azi = self.style_azi(azi)
        sty_zen = self.style_zen(zen)
        style = torch.concatenate([sty_azi, sty_zen], dim=1)
        for j in range(len(self.combina)):
            if j > 1:
                out = self.combina[j](out, style)
            else:
                out = self.combina[j](out)

        outall = out
        outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg = outall, outall, outall, outall, outall, outall

        for i in range(len(self.combina_Iximg)):
            if i > 0:
                outIxrel = self.combina_Ixrel[i](outIxrel, style)
                outIyrel = self.combina_Iyrel[i](outIyrel, style)
                outIzrel = self.combina_Izrel[i](outIzrel, style)
                outIximg = self.combina_Iximg[i](outIximg, style)
                outIyimg = self.combina_Iyimg[i](outIyimg, style)
                outIzimg = self.combina_Izimg[i](outIzimg, style)
            else:
                outIxrel = self.combina_Ixrel[i](outIxrel)
                outIyrel = self.combina_Iyrel[i](outIyrel)
                outIzrel = self.combina_Izrel[i](outIzrel)
                outIximg = self.combina_Iximg[i](outIximg)
                outIyimg = self.combina_Iyimg[i](outIyimg)
                outIzimg = self.combina_Izimg[i](outIzimg)

        return outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg, outall

# model = A_Generator().eval()
# # 将PyTorch模型导出为ONNX格式
# surf = torch.randn(size=(2,1,328,328))
# azi = torch.tensor([[0,0,0,0,1,0,0,0],[0,0,0,0,1,0,0,0]],dtype=torch.float32)
# zen = torch.tensor([[0,0,0,1,0,0,0],[0,0,0,1,0,0,0]],dtype=torch.float32)
# mask = torch.randn(size=(2,1,328,328))
#
# y = model(surf,azi,zen,mask)  # 获取网络的预测值
#
# MyConvNetVis = make_dot(y, params=dict(list(model.named_parameters())))
# MyConvNetVis.format = "png"
# # 指定文件生成的文件夹
# MyConvNetVis.directory = "data"
# # 生成文件
# MyConvNetVis.view()
