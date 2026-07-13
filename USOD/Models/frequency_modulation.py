import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .frequency_integration import StarReLU
from timm.layers import trunc_normal_

class Mlp(nn.Module):
    """ MLP as used in MetaFormer models, eg Transformer, MLP-Mixer, PoolFormer, MetaFormer baslines and related networks.
    Mostly copied from timm.
    """

    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class FrequencyModulation(nn.Module):
    def __init__(self, dim, expansion_ratio=1, reweight_expansion_ratio=.125,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14, weight_resize=True, group=32, init_scale=1e-5,
                 **kwargs):
        super().__init__()
        
        self.size = size
        self.filter_size = size // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize

        self.reweight_1 = Mlp(dim, reweight_expansion_ratio, group * (num_filters//2), bias=False)
        self.reweight_2 = Mlp(dim, reweight_expansion_ratio, group * (num_filters//4), bias=False)
        self.reweight_3 = Mlp(dim, reweight_expansion_ratio, group * (num_filters//4), bias=False)
        self.complex_weights = nn.Parameter(
            torch.randn(num_filters, dim//group, self.size, self.filter_size,dtype=torch.float32) * init_scale)
        trunc_normal_(self.complex_weights, std=init_scale)
        self.act2 = act2_layer()

        self.norm_x = nn.LayerNorm(dim)
        self.norm_low = nn.LayerNorm(dim)
        self.norm_high = nn.LayerNorm(dim)

        self.scale = nn.Parameter(torch.ones(1, dim, 1, 1) * 1e-4)

    def init_reweight_bias(self, group, num_filters):
            # 创建一个 (group, num_filters) 的矩阵，对角线部分为单位矩阵，其余为 0
            bias_matrix = torch.zeros(group, num_filters)
            min_dim = min(group, num_filters)
            for i in range(min_dim):
                bias_matrix[i][i] = 1.0
            
            # 展开为一维向量
            bias_vector = bias_matrix.view(-1)
            bias_vector = bias_vector.repeat(group * num_filters // len(bias_vector))
            
            # 设置 fc2 的 bias
            self.reweight.fc2.bias.data = bias_vector

    def forward(self, x, low, high):
        B, C, H, W, = x.shape
        x_spatial_before = x
        x_rfft = torch.fft.rfft2(x.to(torch.float32), dim=(2, 3), norm='ortho')
        B, C, RH, RW, = x_rfft.shape
        x = x.permute(0, 2, 3, 1) 
        # reshape for added rgb and depth features
        low = low.permute(0, 2, 3, 1) 
        high = high.permute(0, 2, 3, 1)

        # routeing for the final fused feature, the fused low-frequency feature, and the fused high-frequency feature.
        routeing_1 = self.reweight_1(self.norm_x(x).mean(dim=(1, 2))).view(B, -1, self.num_filters//2).tanh_()
        routeing_2 = self.reweight_2(self.norm_low(low).mean(dim=(1, 2))).view(B, -1, self.num_filters//4).tanh_()
        routeing_3 = self.reweight_3(self.norm_high(high).mean(dim=(1, 2))).view(B, -1, self.num_filters//4).tanh_()
        routeing = torch.cat((routeing_1, routeing_2, routeing_3), dim=2)  # b, group, num_filters

        weight = self.complex_weights
        if not weight.shape[2:4] == x_rfft.shape[2:4]:
            weight = F.interpolate(weight, size=x_rfft.shape[2:4], mode='bicubic', align_corners=True)

        weight = torch.einsum('bgf,fchw->bgchw', routeing, weight)
        weight = weight.reshape(B, C, RH, RW)
        weight = 1 + self.scale * weight

        x_rfft = torch.view_as_complex(torch.stack([x_rfft.real * weight, x_rfft.imag * weight], dim=-1))
        x = torch.fft.irfft2(x_rfft, s=(H, W), dim=(2, 3), norm='ortho') # x -> x_mod


        return x