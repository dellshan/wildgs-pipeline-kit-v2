# -*- coding: utf-8 -*-
"""
Patched LSeg implementation (modules/models/lseg_net.py)

What changed vs. the original:
- Robust checkpoint loader: supports Lightning .ckpt / {'model': sd} / raw state_dict;
  strips common prefixes ('net.', 'module.') and loads with strict=False.
- logit_scale handling: store a learnable scalar in log-space (self.logit_log), then
  call .exp() during forward (matches CLIP practice). Avoids accidentally overwriting
  a nn.Parameter by device moves.
- dtype fix: force both image/text features to FP32 to avoid matmul float/half mismatch.
- channels_last fix: actually assign the contiguous() result back to x.
- text tokenization: on the same device as x; tiny cache to avoid re-tokenizing.
"""
import math
import types
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lseg_blocks import FeatureFusionBlock, Interpolate, _make_encoder, FeatureFusionBlock_custom, forward_vit
import clip
import numpy as np


# ----------------------------- optional blocks -----------------------------
class depthwise_clipseg_conv(nn.Module):
    def __init__(self):
        super(depthwise_clipseg_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def depthwise_clipseg(self, x, channels):
        x = torch.cat([self.depthwise(x[:, i].unsqueeze(1)) for i in range(channels)], dim=1)
        return x

    def forward(self, x):
        channels = x.shape[1]
        out = self.depthwise_clipseg(x, channels)
        return out


class depthwise_conv(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1):
        super(depthwise_conv, self).__init__()
        self.depthwise = nn.Conv2d(1, 1, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        # support for 4D tensor with NCHW
        C, H, W = x.shape[1:]
        x = x.reshape(-1, 1, H, W)
        x = self.depthwise(x)
        x = x.view(-1, C, H, W)
        return x


class depthwise_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(depthwise_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

    def forward(self, x, act=True):
        x = self.depthwise(x)
        if act:
            x = self.activation(x)
        return x


class bottleneck_block(nn.Module):
    def __init__(self, kernel_size=3, stride=1, padding=1, activation='relu'):
        super(bottleneck_block, self).__init__()
        self.depthwise = depthwise_conv(kernel_size=3, stride=1, padding=1)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

    def forward(self, x, act=True):
        sum_layer = x.max(dim=1, keepdim=True)[0]
        x = self.depthwise(x)
        x = x + sum_layer
        if act:
            x = self.activation(x)
        return x


# ----------------------------- base model -----------------------------
class BaseModel(torch.nn.Module):
    def load(self, path):
        """Load model from file. Compatible with Lightning .ckpt, dict({'model': sd}), or plain state_dict."""
        # PyTorch 2.6+ defaults to weights_only=True; Lightning ckpt needs full unpickling.
        try:
            ckpt = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=torch.device("cpu"))

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            raw = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            raw = ckpt["model"]
        else:
            raw = ckpt

        cleaned = {}
        target = set(self.state_dict().keys())
        for k, v in raw.items():
            k2 = k
            if k2.startswith("net."):
                k2 = k2[4:]
            if k2.startswith("module."):
                k2 = k2[7:]
            if k2 in target:
                cleaned[k2] = v

        missing, unexpected = self.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"[BaseModel.load] Missing keys ({len(missing)}): {missing[:8]}")
        if unexpected:
            print(f"[BaseModel.load] Unexpected keys ({len(unexpected)}): {unexpected[:8]}")


def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        activation=nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )


# ----------------------------- LSeg core -----------------------------
class LSeg(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="clip_vitl16_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs,
    ):
        super(LSeg, self).__init__()

        self.channels_last = channels_last

        hooks = {
            "clip_vitl16_384": [5, 11, 17, 23],
            "clipRN50x16_vitl16_384": [5, 11, 17, 23],
            "clip_vitb32_384": [2, 5, 8, 11],
        }

        # Instantiate backbone and reassemble blocks
        self.clip_pretrained, self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks[backbone],
            use_readout=readout,
        )

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        # Learnable CLIP-style logit scale (stored in log-space)
        self.logit_log = nn.Parameter(torch.tensor(float(np.log(1 / 0.07)), dtype=torch.float32))

        if backbone in ["clipRN50x16_vitl16_384"]:
            self.out_c = 768
        else:
            self.out_c = 512
        self.scratch.head1 = nn.Conv2d(features, self.out_c, kernel_size=1)

        self.arch_option = kwargs.get("arch_option", 0)
        self.block_depth = kwargs.get("block_depth", 0)
        activation = kwargs.get("activation", "relu")
        if self.arch_option == 1:
            self.scratch.head_block = bottleneck_block(activation=activation)
        elif self.arch_option == 2:
            self.scratch.head_block = depthwise_block(activation=activation)

        self.scratch.output_conv = head

        # Make CLIP run in FP32 to avoid half/float mismatch on CUDA
        try:
            self.clip_pretrained.float()
        except Exception:
            pass

        # labels are provided by subclass (LSegNet) before super().__init__
        self._cached_labels = None
        self._cached_tokens = None

    def _get_text_tokens(self, labels: List[str], device: torch.device):
        labels_tup = tuple(labels)
        if (self._cached_tokens is None) or (self._cached_labels != labels_tup):
            self._cached_labels = labels_tup
            self._cached_tokens = clip.tokenize(labels)
        return self._cached_tokens.to(device)

    def forward(self, x, labelset=''):
        # channels_last fix
        if self.channels_last is True:
            x = x.contiguous(memory_format=torch.channels_last)

        # labels / tokens
        if labelset == '' or labelset is None:
            labels = self.labels
        else:
            labels = list(labelset)
        text = self._get_text_tokens(labels, x.device)

        # Visual backbone forward
        layer_1, layer_2, layer_3, layer_4 = forward_vit(self.pretrained, x)

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        # Encode text → force FP32
        with torch.inference_mode():
            text_features = self.clip_pretrained.encode_text(text).float()

        # Image features → also FP32
        image_features = self.scratch.head1(path_1).float()

        imshape = image_features.shape
        image_features = image_features.permute(0, 2, 3, 1).reshape(-1, self.out_c)

        # normalized features (float32 for stability)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_log.exp()  # scalar
        logits_per_image = logit_scale * (image_features @ text_features.t())

        out = logits_per_image.float().view(imshape[0], imshape[2], imshape[3], -1).permute(0, 3, 1, 2)

        if self.arch_option in [1, 2] and self.block_depth > 0:
            for _ in range(self.block_depth - 1):
                out = self.scratch.head_block(out)
            out = self.scratch.head_block(out, False)

        out = self.scratch.output_conv(out)
        return out


class LSegNet(LSeg):
    """Network for semantic segmentation."""
    def __init__(self, labels, path=None, scale_factor=0.5, crop_size=480, **kwargs):

        features = kwargs.pop("features", 256)  # avoid double-passing to super
        kwargs["use_bn"] = True

        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.labels = list(labels)

        head = nn.Sequential(
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
        )

        super().__init__(head, features=features, **kwargs)

        if path is not None:
            self.load(path)
