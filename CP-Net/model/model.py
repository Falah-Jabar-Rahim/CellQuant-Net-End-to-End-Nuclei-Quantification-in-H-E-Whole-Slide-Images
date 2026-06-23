# ============================================================
# CellPriorNet
# Inputs:
#   x_rgb: [B,3,H,W]
#   x_h:   [B,1,H,W]
# Output:
#   dict with:
#     nuclei_binary_map: [B,2,H,W]
#     nuclei_type_map:   [B,num_nuclei_classes,H,W]
# ============================================================
import os
import sys
import math
import torch
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch.nn as nn
from pathlib import Path
import torch.nn.functional as F
from monai.networks.blocks import UpSample
from monai.utils import InterpolateMode
from model.unirepLKnet import UniRepLKNet
from monai.networks.nets.basic_unet import UpCat, TwoConv
from monai.networks.layers.utils import get_act_layer
from typing import Optional, Sequence, Tuple, Union, List

# -------------------------
#  blocks (decoder + heads)
# -------------------------
encoder_feature_channel = {
    "unireplknet_n": (80, 160, 320, 640),
    "unireplknet_s": (96, 192, 384, 768),
}


def get_conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias):
    if type(kernel_size) is int:
        use_large_impl = kernel_size > 5
    else:
        assert len(kernel_size) == 2 and kernel_size[0] == kernel_size[1]
        use_large_impl = kernel_size[0] > 5
    has_large_impl = 'LARGE_KERNEL_CONV_IMPL' in os.environ
    if has_large_impl and in_channels == out_channels and out_channels == groups and use_large_impl and stride == 1 and padding == kernel_size // 2 and dilation == 1:
        sys.path.append(os.environ['LARGE_KERNEL_CONV_IMPL'])
        #   Please follow the instructions https://github.com/DingXiaoH/RepLKNet-pytorch/blob/main/README.md
        #   export LARGE_KERNEL_CONV_IMPL=absolute_path_to_where_you_cloned_the_example (i.e., depthwise_conv2d_implicit_gemm.py)
        # TODO more efficient PyTorch implementations of large-kernel convolutions. Pull requests are welcomed.
        # Or you may try MegEngine. We have integrated an efficient implementation into MegEngine and it will automatically use it.
        from depthwise_conv2d_implicit_gemm import DepthWiseConv2dImplicitGEMM
        return DepthWiseConv2dImplicitGEMM(in_channels, kernel_size, bias=bias)
    else:
        return nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, groups=groups, bias=bias)

def get_bn(channels, use_sync_bn = False):
    if use_sync_bn:
        return nn.SyncBatchNorm(channels)
    else:
        return nn.BatchNorm2d(channels)

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups, dilation=1):
    if padding is None:
        padding = kernel_size // 2
    result = nn.Sequential()
    result.add_module('conv', get_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False))
    result.add_module('bn', get_bn(out_channels))
    return result

class RingBank(nn.Module):
    def __init__(self, channels=1, radii=(5, 8, 12), thickness=2):
        super().__init__()
        kernels = []
        Kmax = 0
        for r in radii:
            k = 2 * (r + thickness) + 1
            yy, xx = torch.meshgrid(
                torch.arange(k) - k // 2, torch.arange(k) - k // 2, indexing="ij"
            )
            rr = (xx**2 + yy**2).float().sqrt()
            ring = ((rr >= r) & (rr <= r + thickness)).float()
            ring = ring / ring.sum().clamp_min(1.0)
            kernels.append(ring[None, None, ...])  # [1,1,k,k]
            Kmax = max(Kmax, k)

        pads = []
        for w in kernels:
            pad = (Kmax - w.shape[-1]) // 2
            pads.append(F.pad(w, (pad, pad, pad, pad)))

        W = torch.cat(pads, dim=0)  # [S,1,K,K]
        self.register_buffer("weight", W.repeat(channels, 1, 1, 1))
        self.groups = channels
        self.padding = Kmax // 2

    def forward(self, x):  # x: [B,1,H,W]
        y = F.conv2d(x, self.weight, padding=self.padding, groups=self.groups)  # [B,S,H,W]
        return y.max(dim=1, keepdim=True).values

def gaussian_1d(sigma, radius=None):
    if radius is None:
        radius = max(1, int(3 * sigma))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    return g

class DepthwiseSeparableGaussian(nn.Module):
    def __init__(self, channels, sigma):
        super().__init__()
        g = gaussian_1d(sigma)  # [K]
        k = g.numel()
        self.convx = nn.Conv2d(
            channels, channels, (1, k), padding=(0, k // 2), groups=channels, bias=False
        )
        self.convy = nn.Conv2d(
            channels, channels, (k, 1), padding=(k // 2, 0), groups=channels, bias=False
        )
        with torch.no_grad():
            self.convx.weight.zero_()
            self.convy.weight.zero_()
            for c in range(channels):
                self.convx.weight[c, 0, 0, :] = g
                self.convy.weight[c, 0, :, 0] = g
        for p in self.parameters():
            p.requires_grad = False  # fixed

    def forward(self, x):
        return self.convy(self.convx(x))

class DoGMultiScale(nn.Module):
    """Multi-scale DoG -> max abs across scales."""
    def __init__(self, channels=1, sigmas=(1.6, 2.4, 3.2)):
        super().__init__()
        self.g1 = nn.ModuleList([DepthwiseSeparableGaussian(channels, s) for s in sigmas])
        self.g2 = nn.ModuleList([DepthwiseSeparableGaussian(channels, s * math.sqrt(2)) for s in sigmas])

    def forward(self, x):  # [B,1,H,W]
        dogs = [(g1(x) - g2(x)).abs() for g1, g2 in zip(self.g1, self.g2)]
        return torch.stack(dogs, dim=1).max(dim=1).values  # [B,1,H,W]

def _get_encoder_channels_by_backbone(backbone: str, in_channels: int) -> tuple:
    enc = encoder_feature_channel[backbone]
    return tuple([in_channels] + list(enc))  # e.g. (4, 96, 192, 384, 768)

class RepLKDeocder(nn.Module):
    def __init__(
        self,
        encoder_channels: Sequence[int],
        spatial_dims: int,
        decoder_channels: Sequence[int],
        stage_lk_sizes,
        drop_path,
        upsample: str,
        pre_conv: Optional[str],
        interp_mode: str,
        align_corners: Optional[bool],
        small_kernel,
        stem_dim: int,  # <<< required
        dw_ratio: int = 1,
        small_kernel_merged: bool = False,
        norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.1}),
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        dropout: Union[float, tuple] = 0.0,
        bias: bool = False,
        is_pad: bool = True,
        ffn_ratio: int = 4,
    ):
        super().__init__()

        in_channels = [encoder_channels[-1]] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:-1][::-1]) + [0]
        halves = [True] * (len(skip_channels) - 1) + [False]

        self.blocks = nn.ModuleList([
            UpCat(
                spatial_dims=spatial_dims,
                in_chns=in_chn,
                cat_chns=skip_chn,
                out_chns=out_chn,
                act=act,
                norm=norm,
                dropout=dropout,
                bias=bias,
                upsample=upsample,
                pre_conv=pre_conv,
                interp_mode=interp_mode,
                align_corners=align_corners,
                halves=halve,
                is_pad=is_pad,
            )
            for in_chn, skip_chn, out_chn, halve in zip(in_channels, skip_channels, decoder_channels, halves)
        ])

        # extra refinement
        self.upsample1 = UpSample(spatial_dims, 256, 256, 2, mode=upsample,
                                  pre_conv=pre_conv, interp_mode=interp_mode, align_corners=align_corners)
        self.upsample2 = UpSample(spatial_dims, 128, 128, 2, mode=upsample,
                                  pre_conv=pre_conv, interp_mode=interp_mode, align_corners=align_corners)

        # dynamic skip sizes from encoder stem
        skip1_ch = stem_dim // 2  # 48 for unireplknet_s, 40 for unireplknet_n
        skip0_ch = stem_dim // 4  # 24 for s, 20 for n

        self.convs  = TwoConv(spatial_dims, 256 + skip1_ch, decoder_channels[-2], act, norm, bias, dropout)
        self.convs1 = TwoConv(spatial_dims, 128 + skip0_ch, decoder_channels[-1], act, norm, bias, dropout)

    def forward(self, features: List[torch.Tensor], input_feature: torch.Tensor, skip_connect: int = 3):
        skips = features[:-1][::-1]
        features = features[1:][::-1]
        x = features[0]

        for i, block in enumerate(self.blocks):
            if i < skip_connect:
                x = block(x, skips[i])
            else:
                # tail refinement (same logic you had)
                skip = input_feature[1]
                x = self.upsample1(x)
                x = torch.cat([skip, x], dim=1)
                x = self.convs(x)

                skip = input_feature[0]
                x = self.upsample2(x)
                x = torch.cat([skip, x], dim=1)
                x = self.convs1(x)

        return x

class SegmentationHead(nn.Sequential):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        act: Optional[Union[Tuple, str]] = None,
        scale_factor: float = 1.0,
    ):
        bn_layer = nn.BatchNorm2d(in_channels)
        conv_layer1 = conv_bn(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, groups=1)
        nonlinear_layer = nn.GELU()
        conv_layer2 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, groups=1)

        up_layer: nn.Module = nn.Identity()
        if scale_factor > 1.0:
            up_layer = UpSample(
                spatial_dims=spatial_dims,
                scale_factor=scale_factor,
                mode="nontrainable",
                pre_conv=None,
                interp_mode=InterpolateMode.LINEAR,
            )
        if act is not None:
            _ = get_act_layer(act)

        super().__init__(bn_layer, conv_layer1, nonlinear_layer, conv_layer2, up_layer)

# -------------------------
# Final model:  encoder/decoder + priors + 2 heads (binary + type)
# -------------------------
class CellPriorNet(nn.Module):
    def __init__(
        self,
        num_nuclei_classes: int,
        backbone: str = "unireplknet_s",
        pretrained_encoder_ckpt: Optional[Union[str, Path]] = None,  # optional: load encoder weights
        magnification: str = "20x",  # "20x" or "40x"
        use_dog: bool = True,
        use_ring: bool = True,
        use_raw_h: bool = False,
        fuse_mode: str = "mean",     # "mean" | "wsum" | "mul" | "max"
        fuse_alpha: float = 0.6,
        in_channels=4,
        decoder_channels: Tuple[int, ...] = (1024, 512, 256, 128, 64),
        spatial_dims: int = 2,
        norm: Union[str, tuple] = ("batch", {"eps": 1e-3, "momentum": 0.1}),
        act: Union[str, tuple] = ("relu", {"inplace": True}),
        dropout: Union[float, tuple] = 0.0,
        decoder_bias: bool = False,
        upsample: str = "nontrainable",
        interp_mode: str = "nearest",
        drop_path_rate: float = 0.1,
        large_kernel_sizes: Sequence[int] = (11, 21, 23, 25), # (11, 21, 23, 25) or (13, 27, 29, 31)
        small_kernel: int = 5,
    ):
        super().__init__()

        assert backbone in encoder_feature_channel, f"backbone must be one of {list(encoder_feature_channel.keys())}"

        # ---- prior config ----
        self.use_dog = use_dog
        self.use_ring = use_ring
        self.fuse_mode = fuse_mode
        self.fuse_alpha = fuse_alpha
        self.use_raw_h = use_raw_h

        mag = magnification.lower()
        if mag == "40x":
            sigmas = (2.5, 3.5, 4.5)
            ring_radii = (2, 5)
        else:
            sigmas = (1.6, 2.4, 3.2)
            ring_radii = (4, 8)

        self.h_dog = DoGMultiScale(channels=1, sigmas=sigmas)
        self.h_ring = RingBank(channels=1, radii=ring_radii, thickness=2)

        encoder_channels = _get_encoder_channels_by_backbone(backbone, in_channels=in_channels)
        # pick dims based on backbone
        if backbone == "unireplknet_s":
            dims = (96, 192, 384, 768)
        elif backbone == "unireplknet_n":
            dims = (80, 160, 320, 640)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self.encoder = UniRepLKNet(
            in_chans=in_channels,
            num_classes=None,
            #depths=(2, 2, 8, 2),
            depths=(3, 3, 3, 3),
            kernel_sizes=(
                (3, 3, 3),
                (13, 13, 13),
                (13, 13, 13),
                (13, 13, 13)),
            dims=dims,
            drop_path_rate=0.3,
            layer_scale_init_value=1e-6,
            head_init_scale=1.0,
            deploy=False,
            with_cp=False,
            init_cfg=None,
            attempt_use_lk_impl=True,
            use_sync_bn=False,
        )


        self.decoder = RepLKDeocder(
            encoder_channels=encoder_channels,
            stage_lk_sizes=large_kernel_sizes,
            small_kernel=small_kernel,
            drop_path=drop_path_rate,
            spatial_dims=spatial_dims,
            decoder_channels=decoder_channels,
            act=act,
            norm=norm,
            dropout=dropout,
            bias=decoder_bias,
            upsample=upsample,
            interp_mode=interp_mode,
            pre_conv=None,
            align_corners=None,
            stem_dim=dims[0],
        )

        # ---- heads ----
        self.nuclei_binary_head = SegmentationHead(
            spatial_dims=spatial_dims,
            in_channels=decoder_channels[-1],
            out_channels=2,  # binary
            kernel_size=1,
            act=None,
            scale_factor=1.0,
        )

        self.nuclei_type_head = SegmentationHead(
            spatial_dims=spatial_dims,
            in_channels=decoder_channels[-1],
            out_channels=num_nuclei_classes,
            kernel_size=1,
            act=None,
            scale_factor=1.0,
        )

        self.num_nuclei_classes = num_nuclei_classes


    def _fuse_prior(self, dog_map, ring_map):
        # invert ring map after min-max normalization
        if ring_map is not None:
            rmin = ring_map.amin(dim=(2, 3), keepdim=True)
            rmax = ring_map.amax(dim=(2, 3), keepdim=True)
            ring_map = 1.0 - (ring_map - rmin) / (rmax - rmin + 1e-6)

        if dog_map is None:
            fused = ring_map
        elif ring_map is None:
            fused = dog_map
        else:
            if self.fuse_mode == "mean":
                fused = 0.5 * (dog_map + ring_map)
            elif self.fuse_mode == "wsum":
                fused = self.fuse_alpha * dog_map + (1.0 - self.fuse_alpha) * ring_map
            elif self.fuse_mode == "mul":
                fused = torch.sqrt(torch.clamp(dog_map * ring_map, min=0.0) + 1e-6)
            else:  # "max"
                fused = torch.max(dog_map, ring_map)

        # per-tile standardize
        mu = fused.mean(dim=(2, 3), keepdim=True)
        sd = fused.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (fused - mu) / sd

    @torch.no_grad()
    def _adapt_first_conv_3ch_to_4ch(self, state_dict: dict) -> dict:
        """
        If checkpoint has a 3-channel first conv, adapt it to 4-channel by:
          new_w[:, 0:3] = old_w
          new_w[:, 3]   = mean(old_w[:, 0:3])
        Works for typical keys like 'downsample_layers.0.0.weight' etc.
        """
        # find likely first conv weight: 4D, in_ch==3, out_ch>0
        candidate_keys = []
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and v.ndim == 4 and v.shape[1] == 3:
                # prefer earlier/stem keys if present
                score = 0
                if "downsample_layers.0.0.weight" in k:
                    score += 100
                if "stem" in k or "patch_embed" in k:
                    score += 50
                if "downsample_layers.0" in k:
                    score += 20
                candidate_keys.append((score, k))

        if not candidate_keys:
            return state_dict

        candidate_keys.sort(reverse=True)
        k = candidate_keys[0][1]
        w = state_dict[k]  # [out, 3, kh, kw]
        out, in_ch, kh, kw = w.shape
        if in_ch != 3:
            return state_dict

        w4 = torch.zeros((out, 4, kh, kw), dtype=w.dtype, device=w.device)
        w4[:, :3] = w
        w4[:, 3:4] = w.mean(dim=1, keepdim=True)
        state_dict[k] = w4
        return state_dict

    def forward(self, x_rgb: torch.Tensor, x_h: torch.Tensor, pad: int = 10) -> dict:
        """
        x_rgb: [B,3,H,W]
        x_h:   [B,1,H,W]
        """

        if self.use_raw_h:
            # Directly use H channel as the 4th input channel
            x4 = torch.cat([x_rgb, x_h], dim=1)
        else:
            # priors on padded H
            x_h_padded = F.pad(x_h, (pad, pad, pad, pad), mode="reflect")
            dog_map = self.h_dog(x_h_padded) if self.use_dog else None
            ring_map = self.h_ring(x_h_padded) if self.use_ring else None

            # crop back
            B, C, H, W = x_h.shape
            if dog_map is not None:
                dog_map = dog_map[:, :, pad : pad + H, pad : pad + W]
            if ring_map is not None:
                ring_map = ring_map[:, :, pad : pad + H, pad : pad + W]

            prior = self._fuse_prior(dog_map, ring_map)  # [B,1,H,W]
            x4 = torch.cat([x_rgb, prior], dim=1)        # [B,4,H,W]

        # LK encoder -> decoder -> heads
        # Your UniRepLKNet returns (_, z, input_feature) in your LKCell code
        _, z, input_feature = self.encoder(x4)
        decoder_out = self.decoder(z, input_feature)

        out = {
            "nuclei_binary_map": self.nuclei_binary_head(decoder_out),  # [B,2,H,W]
            "nuclei_type_map": self.nuclei_type_head(decoder_out),      # [B,C,H,W]
        }
        return out