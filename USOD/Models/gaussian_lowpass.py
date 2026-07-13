import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_kernel2d(k: int, sigma: float, device=None, dtype=None):
    # k must be odd
    ax = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel  # (k, k)

class DepthwiseGaussianConv2d(nn.Module):
    """
    Depthwise Gaussian low-pass filter:
    y = conv2d(x, weight, groups=C), weight is fixed Gaussian.
    """
    def __init__(self, channels: int, k: int, sigma: float):
        super().__init__()
        assert k % 2 == 1, "k must be odd"
        self.channels = channels
        self.k = k
        self.sigma = sigma
        self.pad = k // 2

        # register gaussian kernel as buffer (not learnable)
        w = gaussian_kernel2d(k, sigma, device="cpu", dtype=torch.float32)  # (k,k)
        w = w.view(1, 1, k, k).repeat(channels, 1, 1, 1)                   # (C,1,k,k)
        self.register_buffer("weight", w, persistent=False)

    def forward(self, x):
        # x: (B,C,H,W)
        w = self.weight.to(device=x.device, dtype=x.dtype)
        return F.conv2d(x, w, bias=None, stride=1, padding=self.pad, groups=self.channels)

class LPSystem(nn.Module):
    """
    For 4-stage features: outs = [F1,F2,F3,F4], each (B,C,H,W)
    Return lows, highs.
    """
    def __init__(self, channels=(96, 192, 384, 768), sigmas=(1.2, 0.6, 0.3, 0.15)):
        super().__init__()
        ks = []
        for s in sigmas:
            k = 2 * math.ceil(3 * s) + 1
            k = max(3, int(k))
            if k % 2 == 0:
                k += 1
            ks.append(k)

        self.lps = nn.ModuleList([
            DepthwiseGaussianConv2d(c, k, s)
            for c, k, s in zip(channels, ks, sigmas)
        ])

    def forward(self, outs):
        lows, highs = [], []
        for feat, lp in zip(outs, self.lps):
            low = lp(feat)
            high = feat - low
            lows.append(low)
            highs.append(high)
        return lows, highs
