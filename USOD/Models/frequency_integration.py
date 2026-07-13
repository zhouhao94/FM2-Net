import torch
import torch.nn as nn
import torch.nn.functional as F


class StarReLU(nn.Module):
    """
    StarReLU: s * relu(x) ** 2 + b
    """

    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)

    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )


class FrequencyIntegration(nn.Module):
    def __init__(self, dim, qkv_bias=False, attn_drop=0., proj_drop=0., lf_dy_weight=True, hf_dy_weight=True):
        super().__init__()
        ### 高频、低频动态权重
        self.lf_dy_weight = lf_dy_weight
        self.hf_dy_weight = hf_dy_weight

        if self.lf_dy_weight:
            self.dy_freq_2 = nn.Linear(dim, dim, bias=True)  # 低频调节权重
            self.lf_gamma= nn.Parameter(1e-5 * torch.ones((dim)),requires_grad=True) # no decay

        if self.hf_dy_weight:
            self.dy_freq = nn.Linear(dim, dim, bias=True)  # 高频调节权重
            self.hf_gamma= nn.Parameter(1e-5 * torch.ones((dim)),requires_grad=True) # no decay

        self.dy_freq_starrelu = StarReLU()  # 自定义激活函数
        self.ignore_cls_token = 0  # 忽略的类标记数量

        # channel attention for fusion
        self.channel_pool_x1 = ChannelPool()
        self.channel_pool_x2 = ChannelPool()
        self.channel_embed = nn.Sequential(
                        nn.Conv2d(4, dim//4, kernel_size=1, bias=True),
                        nn.Conv2d(dim//4, dim//4, kernel_size=3, stride=1, padding=1, bias=True),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(dim//4, 2, kernel_size=1, bias=True),
                        nn.Sigmoid()
                        )

    def forward(self, low_fq, high_fq, H=None, W=None):
        B, C, H, W = low_fq.shape

        inp = low_fq + high_fq  # 融合
        inp = inp.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)  # B x (H*W) x C

        # 计算低频和高频动态频率调节
        dy_freq_feat = self.dy_freq_starrelu(inp)

        if hasattr(self, 'dy_freq_2'):
            dy_freq_lf = self.dy_freq_2(dy_freq_feat).tanh()  # 低频的动态调整 [B x (H*W) x C]
            dy_freq_lf = dy_freq_lf.reshape(B, H, W, C).permute(0, 3, 1, 2)  # B x C x H x W

        if hasattr(self, 'dy_freq'):
            dy_freq = F.softplus(self.dy_freq(dy_freq_feat))  # 高频的动态调整
            dy_freq2 = dy_freq ** 2
            dy_freq = 2 * dy_freq2 / (dy_freq2 + 0.3678)  # 使用 softplus 和归一化调节权重
            dy_freq = dy_freq.reshape(B, H, W, C).permute(0, 3, 1, 2)  # B x C x H x W

        if hasattr(self, 'dy_freq_2'):
            fused = low_fq + low_fq * dy_freq_lf * self.lf_gamma.view(1, -1, 1, 1)  # 低频部分加权调整
        if hasattr(self, 'dy_freq'):
            fused = fused + dy_freq * high_fq * self.hf_gamma.view(1, -1, 1, 1)  # 高频部分加权调整

        return fused