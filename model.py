import copy
import torch.nn as nn
import torchvision
import math
import random
import functools
import operator
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

channels = {
    4: 512,
    8: 512,
    16: 512,
    32: 512,
    64: 256,
    128: 128,
    256: 64,
    512: 32,
    1024: 16,
}

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)

def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k

# 四维 上采样函数 不会受kernel影响 尺度固定 2
class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out

# 四维 下采样函数 不会受kernel影响 尺度固定 2
class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out

# 四维 卷积函数 会受kernel、padding影响
class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, dilation=1, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            dilation=self.dilation,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        bias = self.bias*self.lr_mul if self.bias is not None else None
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, bias)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=bias
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )

# 对结果额外 * sqrt（2）
class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)

# 类卷积 不改变长宽 只改变通道
class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        use_style=True,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample
        self.use_style = use_style

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        if use_style:
            self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        else:
            self.modulation = nn.Parameter(torch.Tensor(1, 1, in_channel, 1, 1).fill_(1))

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if self.use_style:
            style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
            weight = self.scale * self.weight * style
        else:
            weight = self.scale * self.weight.expand(batch,-1,-1,-1,-1) * self.modulation

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        factor=2,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        dilation = 1,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = factor
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = factor
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                dilation=dilation,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,  # 有 bias 并且 不激活 的情况才会添加
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))  # 加 bias

            else:
                layers.append(ScaledLeakyReLU(0.2))  # 额外 除：根号 2

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], downsample=True,factor=2):
        super().__init__()
        if downsample:
            if factor==2:
                self.conv1 = ConvLayer(in_channel, in_channel, 3)
                self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample, factor=factor)
            else:
                self.conv1 = ConvLayer(in_channel, in_channel, 3, downsample=downsample, factor=factor//2)
                self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample, factor=factor//2)
        else:
            self.conv1 = ConvLayer(in_channel, in_channel, 3, downsample=downsample)
            self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=downsample)

        if downsample or in_channel != out_channel:
            self.skip = ConvLayer(
                in_channel, out_channel, 1, downsample=downsample,factor=factor, activate=False, bias=False
            )
        else:
            self.skip = None

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        if self.skip is None:
            skip = input
        else:
            skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out

class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        use_style=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()
        self.use_style = use_style

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            use_style=use_style,
            upsample=upsample,
            downsample=downsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style=None):
        out = self.conv(input, style)
        out = self.activate(out)

        return out

class ToRGB(nn.Module):             # 只在解码器中运用了
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 1, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out

class Encoder(nn.Module):
    def __init__(self, latent_dim):
        # 256, 8 ,3 , 1, 1
        super().__init__()
        size,num_down,n_res = 256,3,1
        stem = [ConvLayer(3, channels[size], 1)]
        log_size = int(math.log(size, 2)) # 8
        in_channel = channels[size]

        for i in range(log_size, log_size-num_down, -1): # 8,5  # 128 ~ 32 # 128 256 512
            out_channel = channels[2 ** (i - 1)]
            stem.append(ResBlock(in_channel, out_channel, downsample=True))
            in_channel = out_channel
        stem += [ResBlock(in_channel, in_channel, downsample=False) for i in range(n_res)]
        self.stem = nn.Sequential(*stem)

        self.content = nn.Sequential(
                        ConvLayer(in_channel, in_channel, 1),
                        ConvLayer(in_channel, in_channel, 1)
                        )
        style  = []
        for i in range(log_size-num_down, 2, -1): # 5->2 # 16 8 4 # 512 512 512
            out_channel = channels[2 ** (i - 1)]
            style.append(ConvLayer(in_channel, out_channel, 3, downsample=True))
            in_channel = out_channel
        style += [
            nn.Flatten(),
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], latent_dim),
              ]
        self.style_l = nn.Sequential(*style)
        self.style_h = copy.deepcopy(self.style_l)
        self.style_theta = copy.deepcopy(self.style_l)


    def forward(self, input):
        act = self.stem(input) # (b,512,32,32)
        # print(act.shape)
        # content = self.content(act) # b,512,32,32
        style_l = self.style_l(act) # b,8
        style_h = self.style_h(act)  # b,8
        style_theta = self.style_theta(act)  # b,8
        style = torch.cat([style_l,style_h,style_theta],dim=1)
        return style

class Decoder(nn.Module):
    def __init__(self, latent_dim=4*4,style_dim=512,n_mlp=4):
        super().__init__()
        lr_mlp = 0.01
        mapping = [EqualLinear(latent_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu')]
        for i in range(n_mlp - 1):
            mapping.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'))
        self.mapping = nn.Sequential(*mapping)
        in_channel = 512
        ch_deco = {
            6: 265,
            7: 128,
            8: 64,
            9: 32,
            10:1,
        }
        self.convs = nn.ModuleList()
        blur_kernel = [1, 3, 3, 1]
        for i in range(6,11,1):  # 6 7 8 9 10
            out_channel = ch_deco[i]  # 256 128 64 32 1
            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )
            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel
                )
            )
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel

        convsurf = []
        convsurf.append(StyledConv(1,8,kernel_size=3,style_dim=512))
        convsurf.append(StyledConv(8,512, kernel_size=3, style_dim=512))
        self.convsurf = nn.Sequential(*convsurf)
        self.to_rgb = ToRGB(in_channel=512,style_dim=512)

    def forward(self,code,surface):  # b 24
        style = self.mapping(code) # b 512
        out = self.convsurf(surface,style)  # 512 256 256
        skip = self.to_rgb(out, style)

        for conv1,conv2,to_rgb in zip(self.convs[::3],self.convs[1::3],self.convs[2::3]):
            out = conv1(out,style)
            out = conv2(out,style)
            skip = to_rgb(out, style, skip)
        image = skip
        return image

class G_enco(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoding = Encoder(latent_dim= 8)

    def forward(self, input):
        return self.encoding(input)

class G_deco(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoding = Decoder()

    def forward(self, input):
        # 24
        return self.decoding(input)
        # 3 * 256 * 256

class Discriminator(nn.Module):
    def __init__(self, size=256, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()
        self.size = size  # 256
        l_branch = self.make_net_(32)
        l_branch += [ConvLayer(channels[32], 1, 1, activate=False)]
        self.l_branch = nn.Sequential(*l_branch)


        g_branch = self.make_net_(8)
        self.g_branch = nn.Sequential(*g_branch)
        self.g_adv = ConvLayer(channels[8], 1, 1, activate=False)

        self.g_std = nn.Sequential(ConvLayer(channels[8], channels[4], 3, downsample=True),
                      nn.Flatten(),
                      EqualLinear(channels[4] * 4 * 4, 128, activation='fused_lrelu'),
                      )
        self.g_final = EqualLinear(128, 1, activation=False)


    def make_net_(self, out_size):
        size = self.size
        convs = [ConvLayer(3, channels[size], 1)]
        log_size = int(math.log(size, 2))  # 256 8
        out_log_size = int(math.log(out_size, 2)) # 32 5
        in_channel = channels[size]  # 64

        for i in range(log_size, out_log_size, -1): # 8~5  # 8~3
            out_channel = channels[2 ** (i - 1)]  # make_net_32 # 128 64 32 # 128 256 512
            # make_net_8 128 64 32 16 8           # 128 256 512 512 512
            convs.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel

        return convs

    def forward(self, x):
        # 4 3 256 256
        l_adv = self.l_branch(x) # 4 1 32 32

        g_act = self.g_branch(x) # 4 512 8 8
        g_adv = self.g_adv(g_act) # 4 1 8 8

        # output = self.g_std(g_act)
        # # BATCH 不同个体之间的差异
        # g_stddev = torch.sqrt(output.var(0, keepdim=True, unbiased=False) + 1e-8).repeat(x.size(0),1)
        # g_std = self.g_final(g_stddev) # 4 1
        return [l_adv, g_adv] # （b 1 32 32）（b 1 8 8）（b 1）

class Discriminator_A(nn.Module):
    def __init__(self):
        super().__init__()
        layer,layer2 = [],[]
        layer.append(EqualLinear(24, 128, activation='fused_lrelu'))
        layer.append(EqualLinear(128, 256, activation='fused_lrelu'))
        layer.append(EqualLinear(256, 12, activation='fused_lrelu'))
        self.layer = nn.Sequential(*layer)

        layer2.append(EqualLinear(24, 128, activation='fused_lrelu'))
        layer2.append(EqualLinear(128, 256, activation='fused_lrelu'))
        layer2.append(EqualLinear(256, 12, activation='fused_lrelu'))
        self.layer2 = nn.Sequential(*layer2)

    def forward(self,input):
        res1 = self.layer(input)
        res2 = self.layer(input)
        return [res1,res2]

# indence1 = G_enco()
# indence2 = G_deco()
# a = torch.randn(size=(1,3,265,256))
# b = torch.randn(size=(1,24))
# c = indence1(a)
# d = indence2(b)
