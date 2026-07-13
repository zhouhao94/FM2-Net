import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.layers import DropPath, to_2tuple, trunc_normal_
from ..DAM_module import *
from ..gaussian_lowpass import LPSystem
from ..gated_cross_attention import GatedCrossAttention2D
#from ..frequency_adjust import FrequencyAdjust
#from ..frequency_scale import FrequencyScale
from ..frequency_integration import FrequencyIntegration
from ..frequency_modulation import FrequencyModulation
from ..cross_frequency_scale import CrossFrequencyScale
import math
import time
from Models.encoders.vmamba import Backbone_VSSM, CrossMambaFusionBlock, ConcatMambaFusionBlock
from ..utils_vis import *


class RGBXTransformer(nn.Module):
    def __init__(self, 
                 num_classes=1000,
                 norm_layer=nn.LayerNorm,
                 depths=[2,2,27,2], # [2,2,27,2] for vmamba small
                 dims=96,
                 pretrained=None,
                 mlp_ratio=4.0,
                 downsample_version='v1',
                 ape=False,
                 img_size=[480, 640],
                 patch_size=4,
                 drop_path_rate=0.2,
                 **kwargs):
        super().__init__()
        
        self.ape = ape

        self.vssm = Backbone_VSSM(
            pretrained=pretrained,
            norm_layer=norm_layer,
            num_classes=num_classes,
            depths=depths,
            dims=dims,
            mlp_ratio=mlp_ratio,
            downsample_version=downsample_version,
            drop_path_rate=drop_path_rate,
        )

        # absolute position embedding
        if self.ape:
            self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
            self.absolute_pos_embed = []
            self.absolute_pos_embed_x = []
            for i_layer in range(len(depths)):
                input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                      self.patches_resolution[1] // (2 ** i_layer))
                dim=int(dims * (2 ** i_layer))
                absolute_pos_embed = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed, std=.02)
                absolute_pos_embed_x = nn.Parameter(torch.zeros(1, dim, input_resolution[0], input_resolution[1]))
                trunc_normal_(absolute_pos_embed_x, std=.02)
                
                self.absolute_pos_embed.append(absolute_pos_embed)
                self.absolute_pos_embed_x.append(absolute_pos_embed_x)

        self.lp_system = LPSystem()

        self.lows_cross_attn = nn.ModuleList([
            GatedCrossAttention2D(c_q=dims, num_heads=4, sr_ratio=1, attn_dim_ratio=1, gate_type="headwise"),
            GatedCrossAttention2D(c_q=dims*2, num_heads=4, sr_ratio=1, attn_dim_ratio=2, gate_type="headwise"),
            GatedCrossAttention2D(c_q=dims*4, num_heads=4, sr_ratio=1, attn_dim_ratio=4, gate_type="headwise"),
            GatedCrossAttention2D(c_q=dims*8, num_heads=4, sr_ratio=1, attn_dim_ratio=8, gate_type="headwise"),
        ])
        self.highs_cross_attn = nn.ModuleList([
            GatedCrossAttention2D(c_q=dims, num_heads=4, sr_ratio=1, attn_dim_ratio=1, gate_type="headwise"),
            GatedCrossAttention2D(c_q=dims*2, num_heads=4, sr_ratio=1, attn_dim_ratio=2, gate_type="headwise"),
            GatedCrossAttention2D(c_q=dims*4, num_heads=4, sr_ratio=1, attn_dim_ratio=4, gate_type="headwise"),
            GatedCrossAttention2D(c_q=dims*8, num_heads=4, sr_ratio=1, attn_dim_ratio=8, gate_type="headwise"),
        ])

        self.freq_integrations = nn.ModuleList([
            #FrequencyIntegration(dim=dims),
            #FrequencyIntegration(dim=dims*2),
            FrequencyIntegration(dim=dims*4),
            FrequencyIntegration(dim=dims*8),
        ])

        self.freq_modulations = nn.ModuleList([
            FrequencyModulation(dim=dims),
            FrequencyModulation(dim=dims*2),
            FrequencyModulation(dim=dims*4),
            FrequencyModulation(dim=dims*8),
        ])

    def forward_features(self, x_rgb, x_e):
        """
        x_rgb: B x C x H x W
        """
        B = x_rgb.shape[0]
        outs_fused = []
        
        outs_rgb, highs_rgb = self.vssm(x_rgb) # B x C x H x W
        outs_x, highs_x = self.vssm(x_e) # B x C x H x W


        for i in range(4):
            low_rgb = outs_rgb[i]
            low_x = outs_x[i]

            high_rgb = highs_rgb[i]
            high_x = highs_x[i]

            low_fuse = low_rgb + self.lows_cross_attn[i](low_rgb, low_x)
            high_fuse = high_rgb + self.highs_cross_attn[i](high_rgb, high_x)
            

            if i < 2:
                x_fuse = low_fuse + high_fuse
            else:
                x_fuse = self.freq_integrations[i-2](low_fuse, high_fuse)

            x_fuse = self.freq_modulations[i](x_fuse, low_fuse, high_fuse)
            
            outs_fused.append(x_fuse)
        return outs_fused

    def forward(self, x_rgb, x_e):
        out = self.forward_features(x_rgb, x_e)
        return out

class vssm_tiny(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_tiny, self).__init__(
            depths=[2, 2, 9, 2], 
            dims=96,
            pretrained='../pretrained_model/vmamba/vssmtiny_dp01_ckpt_epoch_292.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.2,
        )

class vssm_small(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_small, self).__init__(
            depths=[2, 2, 27, 2],
            dims=96,
            pretrained='../pretrained_model/vmamba/vssmsmall_dp03_ckpt_epoch_238.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.3,
        )

class vssm_base(RGBXTransformer):
    def __init__(self, fuse_cfg=None, **kwargs):
        super(vssm_base, self).__init__(
            depths=[2, 2, 27, 2],
            dims=128,
            pretrained='../pretrained_model/vmamba/vssmbase_dp06_ckpt_epoch_241.pth',
            mlp_ratio=0.0,
            downsample_version='v1',
            drop_path_rate=0.6, # VMamba-B with droppath 0.5 + no ema. VMamba-B* represents for VMamba-B with droppath 0.6 + ema
        )