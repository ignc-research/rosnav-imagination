#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tmodels

from rl.models.unet import (
    UNetEncoder,
    UNetDecoder,
    MiniUNetEncoder,
    LearnedRGBProjection,
    MergeMultimodal,
    ResNetRGBEncoder,
) 

from einops import rearrange

def softmax_2d(x):
    b, h, w = x.shape
    x_out = F.softmax(rearrange(x, "b h w -> b (h w)"), dim=1)
    x_out = rearrange(x_out, "b (h w) -> b h w", h=h)
    return x_out

def padded_resize(x, size):
    """For an image tensor of size (bs, c, h, w), resize it such that the
    larger dimension (h or w) is scaled to `size` and the other dimension is
    zero-padded on both sides to get `size`.
    """
    h, w = x.shape[2:]
    top_pad = 0
    bot_pad = 0
    left_pad = 0
    right_pad = 0
    if h > w:
        left_pad = (h - w) // 2
        right_pad = (h - w) - left_pad
    elif w > h:
        top_pad = (w - h) // 2
        bot_pad = (w - h) - top_pad
    x = F.pad(x, (left_pad, right_pad, top_pad, bot_pad))
    x = F.interpolate(x, size, mode="bilinear", align_corners=False)
    return x

# ================================ Anticipation base ==================================
class BaseModel(nn.Module):
    """The basic model for semantic anticipator
    """
    def __init__(self, num_import_layers, num_output_layers, nsf):
        super().__init__()
        self.normalize_channels = num_output_layers*[softmax_2d]

        self._create_gp_models(num_import_layers, num_output_layers, nsf)

    def forward(self, x):
        final_outputs = {}
        gp_outputs = self._do_gp_anticipation(x)
        final_outputs.update(gp_outputs)

        return final_outputs

    def _create_gp_models(self):
        raise NotImplementedError

    def _do_gp_anticipation(self, x):
        raise NotImplementedError

    def _normalize_decoder_output(self, x_dec):
        return torch.stack([normalize_channel(x_dec[:,i]) for i, normalize_channel in enumerate(self.normalize_channels)],dim=1)

# SemAnt Model
class SemAnt2D(BaseModel):
    """
    Anticipated using rgb and depth projection.
    """
    def _create_gp_models(self, num_import_layers, num_output_layers, nsf):
        nmodes = 2
        nsf = 32
        unet_encoder = UNetEncoder(num_import_layers, nsf=nsf)
        unet_decoder = UNetDecoder(num_output_layers, nsf=nsf)
        self.gp_encoder = unet_encoder

        # Decoder module
        self.gp_decoder = unet_decoder
        
    def _do_gp_anticipation(self, x):
        """
        Inputs:
            x is a dictionary containing the following keys:
                'rgb' - (bs, 3, H, W) RGB input
                'ego_map_gt' - (bs, 2, H, W) probabilities
        """
        x_enc = self.gp_encoder(x) 
        x_dec = self.gp_decoder(x_enc)  # (bs, 2, H, W)
        x_dec = self._normalize_decoder_output(x_dec)
        outputs = {"occ_estimate": x_dec}

        return outputs
