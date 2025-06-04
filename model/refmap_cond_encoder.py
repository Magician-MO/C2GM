import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import xformers.ops
from einops import rearrange
from PIL import Image
from torch import nn

# from .dcn_v2 import DCN_sep as DCN
from .spade_rmap import SPADE

"""
    CNN-SR
"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()
        self.conv_in = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.acti = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        out = self.conv_in(x)
        out = self.acti(out)
        return out

class Resblock(nn.Module):
    def __init__(self, n_feat, kernel_size, stride, padding):
        super(Resblock, self).__init__()
        self.res_block = nn.Sequential(
            nn.Conv2d(
                in_channels=n_feat,
                out_channels=n_feat,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(
                in_channels=n_feat,
                out_channels=n_feat,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
        )

    def forward(self, x):
        identity = x
        out = self.res_block(x)
        return out + identity


class Downsample(nn.Module):
    def __init__(self, in_channels, scale):
        super(Downsample, self).__init__()
        self.downsample = nn.Sequential(
            nn.PixelUnshuffle(scale),
            nn.Conv2d(
                in_channels=in_channels * scale * scale,
                out_channels=in_channels,
                kernel_size=1,
            ),
        )

    def forward(self, x):
        return self.downsample(x)


class MapEncoder(pl.LightningModule):
    def __init__(self, out_channel=8):
        super(MapEncoder, self).__init__()

        self.first_layer_source = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # 256 256 32

        # (1) cnn encoder
        self.layer1_source = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),  # 256 256 64
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1
            ),  ##128 128 64
            nn.LeakyReLU(),
        )

        self.layer2_source = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),  # 128 128 128
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1
            ),  # 64 64 128
            nn.LeakyReLU(),
        )

        self.layer3_source = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # 64 64 256
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1
            ),  # 32 32 256
            nn.LeakyReLU(),
        )

        # (3)out
        self.last_linear = nn.Conv2d(256, out_channel, 3, bias=False, padding=1)

    def forward(self, source):
        # (1)cnn encoder
        # b 3 256 256 -> b 256 32 32
        source_cond = self.first_layer_source(source)
        source_cond = self.layer1_source(source_cond)
        source_cond = self.layer2_source(source_cond)
        source_cond = self.layer3_source(source_cond)
        out = self.last_linear(source_cond)
        # 4 32 32
        return out


class MapRefEncoder(pl.LightningModule):
    def __init__(self, out_channel=8):
        super(MapRefEncoder, self).__init__()

        self.first_layer_source = ConvBlock(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # 256 256 32

        self.first_layer_ref = ConvBlock(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # 256 256 32

        # (1) cnn encoder
        self.layer1_source = nn.Sequential(
            ConvBlock(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),  # 256 256 64
            Resblock(n_feat=64, kernel_size=3, stride=1, padding=1),  # 256 256 64
            Downsample(in_channels=64, scale=2),  # 128 128 64
        )

        self.layer2_source = nn.Sequential(
            ConvBlock(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),  # 128 128 128
            Resblock(n_feat=128, kernel_size=3, stride=1, padding=1),  # 128 128 128
            Downsample(in_channels=128, scale=2),  # 64 64 128
        )

        self.layer3_source = nn.Sequential(
            ConvBlock(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # 64 64 256
            Resblock(n_feat=256, kernel_size=3, stride=1, padding=1),  # 64 64 256
            Downsample(in_channels=256, scale=2),  # 32 32 256
        )

        self.layer1_ref = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                # feat_in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 256 256 64
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Downsample(in_channels=64, scale=2),  # 128 128 64
        )

        self.layer2_ref = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                # feat_in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 128 128 128
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Downsample(in_channels=128, scale=2),  # 64 64 128
        )

        self.layer3_ref = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                # feat_in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 64 64 256
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Downsample(in_channels=256, scale=2),  # 32 32 256
        )

        self.last_linear = nn.Conv2d(512, out_channel, 1, bias=False)


    def forward(self, sr, ref):
        # (1)cnn encoder
        # b 3 256 256 -> b 32 256 256
        source_cond = self.first_layer_source(sr)
        ref_cond = self.first_layer_ref(ref)

        ref_cond = self.layer1_ref(ref_cond)
        source_cond = self.layer1_source(source_cond)

        ref_cond = self.layer2_ref(ref_cond)
        source_cond = self.layer2_source(source_cond)

        ref_cond = self.layer3_ref(ref_cond)
        source_cond = self.layer3_source(source_cond)

        out = self.last_linear(torch.cat([ref_cond, source_cond], dim=1))

        return out


class MapDualSpadeRefEncoder(pl.LightningModule):
    def __init__(self, out_channel=8):
        super(MapDualSpadeRefEncoder, self).__init__()

        self.first_layer_source = ConvBlock(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # 256 256 32

        self.first_layer_ref_scale = ConvBlock(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # 256 256 32

        # (1) cnn encoder
        self.layer1_source = nn.Sequential(
            ConvBlock(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),  # 256 256 64
            Resblock(n_feat=64, kernel_size=3, stride=1, padding=1),  # 256 256 64
            Downsample(in_channels=64, scale=2),  # 128 128 64
        )

        self.layer2_source = nn.Sequential(
            ConvBlock(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),  # 128 128 128
            Resblock(n_feat=128, kernel_size=3, stride=1, padding=1),  # 128 128 128
            Downsample(in_channels=128, scale=2),  # 64 64 128
        )

        self.layer3_source = nn.Sequential(
            ConvBlock(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),  # 64 64 256
            Resblock(n_feat=256, kernel_size=3, stride=1, padding=1),  # 64 64 256
            Downsample(in_channels=256, scale=2),  # 32 32 256
        )

        self.layer1_ref_scale = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                # feat_in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 256 256 64
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Downsample(in_channels=64, scale=2),  # 128 128 64
        )

        self.layer2_ref_scale = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                # feat_in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 128 128 128
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Downsample(in_channels=128, scale=2),  # 64 64 128
        )

        self.layer3_ref_scale = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                # feat_in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 64 64 256
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Downsample(in_channels=256, scale=2),  # 32 32 256
        )

        self.spade1 = SPADE(64, 128)
        self.spade2 = SPADE(128, 256)
        self.spade3 = SPADE(256, 512)

        self.last_linear = nn.Conv2d(512, out_channel, 1, bias=False)

    def forward(self, source, ref_scale):
        # (1)cnn encoder
        # b 3 256 256 -> b 8 32 32
        source_cond = self.first_layer_source(source)
        ref_scale_cond = self.first_layer_ref_scale(ref_scale)

        source_cond = self.layer1_source(source_cond)
        ref_scale_cond = self.layer1_ref_scale(ref_scale_cond)
        ref_scale_cond = self.spade1(ref_scale_cond, torch.cat([ref_scale_cond, source_cond], dim=1))

        source_cond = self.layer2_source(source_cond)
        ref_scale_cond = self.layer2_ref_scale(ref_scale_cond)
        ref_scale_cond = self.spade2(ref_scale_cond, torch.cat([ref_scale_cond, source_cond], dim=1))

        source_cond = self.layer3_source(source_cond)
        ref_scale_cond = self.layer3_ref_scale(ref_scale_cond)
        ref_scale_cond = self.spade3(ref_scale_cond, torch.cat([ref_scale_cond, source_cond], dim=1))

        out = self.last_linear(torch.cat([ref_scale_cond, source_cond], dim=1))

        return out

#TODO: 使用预训练ViT/CLIP模型作为特征提取器（额外的segment_cond或者替换特征提取主干）
#TODO: 更新特征融合模块（SPADE->生成-特征提取方向）

# ###################hybrid-spade#################
class MapStyleEncoder(pl.LightningModule):
    def __init__(self):
        super(MapStyleEncoder, self).__init__()

        self.first_layer_source = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # 256 256 32

        # (1) cnn encoder
        self.layer1_source = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),  # 256 256 32
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1
            ),  ##128 128 32
            nn.LeakyReLU(),
        )

        self.layer2_source = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),  # 32 32 32
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1
            ),  # 64 64 32
            nn.LeakyReLU(),
        )

        self.layer3_source = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),  # 64 64 32
            nn.LeakyReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1
            ),  # 32 32 32
            nn.LeakyReLU(),
        )

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),  # 64 64 32
            nn.Tanh()
        )

    def forward(self, source):
        # (1)cnn encoder
        # b 3 256 256 -> b 256 32 32
        source_cond = self.first_layer_source(source)
        source_cond = self.layer1_source(source_cond)
        source_cond = self.layer2_source(source_cond)
        source_cond = self.layer3_source(source_cond)
        source_cond = self.last_layer(source_cond)
        # 256 32 32
        return source_cond

class SpadeRefEncoder(pl.LightningModule):
    def __init__(self, out_channel):
        super(SpadeRefEncoder, self).__init__()

        self.first_layer_source = ConvBlock(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # 256 256 32

        self.first_layer_ref = ConvBlock(
            in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1
        )  # 256 256 32

        # (1) cnn encoder
        self.layer1_source = nn.Sequential(
            ConvBlock(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),  # 256 256 64
            Resblock(n_feat=32, kernel_size=3, stride=1, padding=1),  # 256 256 64
            Downsample(in_channels=32, scale=2),  # 128 128 64
        )

        self.layer2_source = nn.Sequential(
            ConvBlock(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),  # 128 128 128
            Resblock(n_feat=32, kernel_size=3, stride=1, padding=1),  # 128 128 128
            Downsample(in_channels=32, scale=2),  # 64 64 128
        )

        self.layer3_source = nn.Sequential(
            ConvBlock(
                in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1
            ),  # 64 64 256
            Resblock(n_feat=32, kernel_size=3, stride=1, padding=1),  # 64 64 256
            Downsample(in_channels=32, scale=2),  # 32 32 256
        )

        self.layer1_ref = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                # feat_in_channels=64,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 256 256 64
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Downsample(in_channels=32, scale=2),  # 128 128 64
        )

        self.layer2_ref = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                # feat_in_channels=128,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 128 128 128
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Downsample(in_channels=32, scale=2),  # 64 64 128
        )

        self.layer3_ref = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                # feat_in_channels=256,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=1,
            ),  # 64 64 256
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            Downsample(in_channels=32, scale=2),  # 32 32 256
        )

        self.spade1 = SPADE(32, 64)
        self.spade2 = SPADE(32, 64)
        self.spade3 = SPADE(32, 64)

        self.last_linear = nn.Conv2d(32*3, out_channel, 1, bias=False)

    def forward(self, source, ref, map_latent, if_sample_style, style_mean, style_std):
        # (1)cnn encoder
        # b 3 256 256 -> b 8 32 32
        source_cond = self.first_layer_source(source)
        ref_cond = self.first_layer_ref(ref)

        ref_cond = self.layer1_ref(ref_cond)
        source_cond = self.layer1_source(source_cond)
        source_cond = self.spade1(source_cond, torch.cat([ref_cond, source_cond], dim=1))

        ref_cond = self.layer2_ref(ref_cond)
        source_cond = self.layer2_source(source_cond)
        source_cond = self.spade2(source_cond, torch.cat([ref_cond, source_cond], dim=1))

        ref_cond = self.layer3_ref(ref_cond)
        source_cond = self.layer3_source(source_cond)
        source_cond = self.spade3(source_cond, torch.cat([ref_cond, source_cond], dim=1))

        content_mean = source_cond.mean(dim=(2, 3), keepdim=True)
        content_std = source_cond.std(dim=(2, 3), keepdim=True) + 1e-6

        if not if_sample_style:
            style_mean = map_latent.mean(dim=(2, 3), keepdim=True)
            style_std = map_latent.std(dim=(2, 3), keepdim=True) + 1e-6
        else:
            style_mean = style_mean
            style_std = style_std

        source_cond_origin = source_cond
        source_cond_styled = ((source_cond - content_mean) / content_std) * style_std + style_mean
        out = self.last_linear(torch.cat([ref_cond, source_cond_origin, source_cond_styled], dim=1))

        return out

class AdaIN_Encoder(pl.LightningModule):
    def __init__(self, out_channel=320):
        super(AdaIN_Encoder, self).__init__()
        self.source_ref_encoder = SpadeRefEncoder(out_channel)
        self.target_style_encoder = MapStyleEncoder()

    def forward(self, source, ref, style=None, if_sample_style=False, style_mu=None, style_std=None):
        if style is not None:
            map_style_latent = self.target_style_encoder(style)
        else:
            map_style_latent = None
        sr_ref_latent = self.source_ref_encoder(source, ref, map_style_latent, if_sample_style, style_mu, style_std)
        return sr_ref_latent


# test
# device = "cuda:3"

# sr_path = "dataset/exps/data/sr/17_64069_40915.png"
# ref_path = "dataset/exps/data/ref/17_64326_40904.png"
# sr = np.asarray(Image.open(sr_path).convert("RGB"))
# ref = np.asarray(Image.open(ref_path).convert("RGB"))

# sr = torch.from_numpy(sr).float().to(device).unsqueeze(0).permute(0, 3, 1, 2)
# ref = torch.from_numpy(ref).float().to(device).unsqueeze(0).permute(0, 3, 1, 2)

# encoder = AdaIN_Encoder(device)
# encoder = encoder.to(device)
# encoder(sr, ref, sr)
