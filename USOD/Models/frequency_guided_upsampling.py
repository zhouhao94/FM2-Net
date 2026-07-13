import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1) Low-pass / High-pass (frequency-friendly)
# -----------------------------
class FixedGaussianBlur(nn.Module):
    """
    Fixed depthwise Gaussian blur as a low-pass filter.
    This is stable and differentiable (kernel fixed).
    """
    def __init__(self, channels: int, kernel_size: int = 5, sigma: float = 1.0):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma

        # Create 2D gaussian kernel
        ax = torch.arange(kernel_size) - kernel_size // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()

        # Depthwise conv weight: (C, 1, k, k)
        weight = kernel.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        self.register_buffer("weight", weight)

        self.padding = kernel_size // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,H,W)
        return F.conv2d(x, self.weight, bias=None, stride=1, padding=self.padding, groups=self.channels)


def high_pass(x: torch.Tensor, lp: nn.Module) -> torch.Tensor:
    """High-pass = x - LowPass(x)."""
    return x - lp(x)


# -----------------------------
# 2) Convolution-like Local Cross Attention (Neighborhood Attention)
# -----------------------------
class ConvLikeLocalCrossAttn(nn.Module):
    """
    Convolution-like local cross-attention:
      - Q from x_up (upsampled LR)
      - K/V from g_hp (high-pass HR guidance)
      - attention computed inside kxk neighborhood using unfold
      - softmax over neighborhood positions (k*k), i.e. conv-like

    Shapes:
      x_up: (B, Cq, H, W)
      g:    (B, Cg, H, W)

    Output:
      (B, Cout, H, W)
    """
    def __init__(self, dim_q: int, dim_kv: int, dim_out: int,
                 heads: int = 4, kernel_size: int = 5):
        super().__init__()
        assert dim_out % heads == 0, "dim_out must be divisible by heads"
        assert kernel_size % 2 == 1, "kernel_size must be odd"

        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.dim_out = dim_out
        self.heads = heads
        self.kernel_size = kernel_size
        self.d = dim_out // heads
        self.scale = self.d ** -0.5

        # 1x1 projections
        self.q_proj = nn.Conv2d(dim_q, dim_out, kernel_size=1, bias=False)
        self.k_proj = nn.Conv2d(dim_kv, dim_out, kernel_size=1, bias=False)
        self.v_proj = nn.Conv2d(dim_kv, dim_out, kernel_size=1, bias=False)
        self.out_proj = nn.Conv2d(dim_out, dim_out, kernel_size=1, bias=False)

        # Optional normalization (often helps)
        self.norm_q = nn.GroupNorm(num_groups=min(32, dim_out), num_channels=dim_out)
        self.norm_k = nn.GroupNorm(num_groups=min(32, dim_out), num_channels=dim_out)
        self.norm_v = nn.GroupNorm(num_groups=min(32, dim_out), num_channels=dim_out)

    def forward(self, x_up: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x_up.shape
        k = self.kernel_size
        hh = self.heads
        d = self.d

        # Project
        q = self.norm_q(self.q_proj(x_up))        # (B, dim_out, H, W)
        k_map = self.norm_k(self.k_proj(g))       # (B, dim_out, H, W)
        v_map = self.norm_v(self.v_proj(g))       # (B, dim_out, H, W)

        # Reshape q for attention computation
        # q: (B, hh, d, H, W) -> (B, hh, d, 1, H*W)
        q = q.view(B, hh, d, H, W).view(B, hh, d, 1, H * W)

        # Unfold local neighborhoods for K/V:
        # (B, dim_out*k*k, H*W) -> (B, hh, d, k*k, H*W)
        k_patch = F.unfold(k_map, kernel_size=k, padding=k // 2) \
                    .view(B, hh, d, k * k, H * W)
        v_patch = F.unfold(v_map, kernel_size=k, padding=k // 2) \
                    .view(B, hh, d, k * k, H * W)

        # Attention logits: dot(q, k) inside neighborhood
        # -> (B, hh, k*k, H*W)
        attn = (q * k_patch).sum(dim=2) * self.scale

        # Softmax over neighborhood positions (conv-like)
        attn = F.softmax(attn, dim=2)

        # Weighted sum of V in neighborhood
        # (B, hh, d, H*W)
        out = (attn.unsqueeze(2) * v_patch).sum(dim=3)

        # Restore shape (B, dim_out, H, W)
        out = out.view(B, hh * d, H, W)
        out = self.out_proj(out)
        return out


# -----------------------------
# 3) Guided Upsample Block (frequency-aware)
# -----------------------------
class FrequencyAwareGuidedUpsample(nn.Module):
    """
    High-res guidance (g_hr) guides low-res feature upsampling (x_lr) via:
      1) Upsample x_lr -> x_up
      2) Extract high-frequency residual g_hp = g_hr - LP(g_hr)
      3) Local cross-attn: Q from x_up, K/V from g_hp
      4) Residual injection: y = x_up + alpha * delta

    This is often better than directly mixing HR into LR,
    because it preserves LR low-frequency structure and only injects details.
    """
    def __init__(self,
                 dim_lr: int,
                 dim_hr: int,
                 dim_out: int,
                 heads: int = 4,
                 kernel_size: int = 5,
                 upsample_mode: str = "bilinear",
                 align_corners: bool = False,
                 lp_type: str = "gaussian",
                 lp_kernel: int = 5,
                 lp_sigma: float = 1.0):
        super().__init__()

        self.upsample_mode = upsample_mode
        self.align_corners = align_corners

        # Project LR/HR to a common working dimension (dim_out)
        self.lr_proj = nn.Conv2d(dim_lr, dim_out, kernel_size=1, bias=False) if dim_lr != dim_out else nn.Identity()
        self.hr_proj = nn.Conv2d(dim_hr, dim_out, kernel_size=1, bias=False) if dim_hr != dim_out else nn.Identity()

        # Low-pass module (fixed)
        if lp_type == "gaussian":
            self.lp = FixedGaussianBlur(channels=dim_out, kernel_size=lp_kernel, sigma=lp_sigma)
        elif lp_type == "avg":
            # avgpool as low-pass
            self.lp = nn.AvgPool2d(kernel_size=lp_kernel, stride=1, padding=lp_kernel // 2)
        else:
            raise ValueError("lp_type must be 'gaussian' or 'avg'")

        # Local cross-attn (conv-like)
        self.local_attn = ConvLikeLocalCrossAttn(
            dim_q=dim_out,
            dim_kv=dim_out,
            dim_out=dim_out,
            heads=heads,
            kernel_size=kernel_size
        )

        # Residual strength (start from 0 => does not harm baseline initially)
        self.alpha = nn.Parameter(torch.zeros(1))

        # Optional post-fusion refinement
        self.refine = nn.Sequential(
            nn.Conv2d(dim_out, dim_out, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim_out, dim_out, 3, padding=1, bias=False),
        )

        # Optional stability: keep injection bounded (comment out if you want freer learning)
        self.use_bounded_alpha = True

    def forward(self, x_up: torch.Tensor, g_hr: torch.Tensor) -> torch.Tensor:
        """
        x_lr: (B, dim_lr, H_lr, W_lr)
        g_hr: (B, dim_hr, H_hr, W_hr)
        return: (B, dim_out, H_hr, W_hr)
        """
        B, _, Hh, Wh = g_hr.shape

        # 1) Upsample LR to HR resolution
        #x_up = F.interpolate(x_lr, size=(Hh, Wh), mode=self.upsample_mode,
        #                     align_corners=self.align_corners if self.upsample_mode in ["bilinear", "bicubic"] else None)

        # 2) Project to working dim
        x_up = self.lr_proj(x_up)    # (B, dim_out, Hh, Wh)
        g = self.hr_proj(g_hr)       # (B, dim_out, Hh, Wh)

        # 3) High-pass extraction from guidance (frequency perspective)
        g_hp = high_pass(g, self.lp)

        # 4) Local cross-attn injects only "detail-like" information
        delta = self.local_attn(x_up, g_hp)

        # 5) Residual fusion (start harmless, then learn to inject)
        alpha = torch.tanh(self.alpha) if self.use_bounded_alpha else self.alpha
        y = x_up + alpha * delta

        # 6) Refinement
        y = y + self.refine(y)
        return y


# -----------------------------
# 4) Minimal usage demo
# -----------------------------
if __name__ == "__main__":
    torch.manual_seed(0)

    # Example shapes:
    # LR feature:  (B, C_lr,  H/2, W/2)
    # HR guidance: (B, C_hr,  H,   W)
    B = 2
    C_lr = 64
    C_hr = 64
    H, W = 56, 56

    x_lr = torch.randn(B, C_lr, H // 2, W // 2)
    g_hr = torch.randn(B, C_hr, H, W)

    block = FrequencyAwareGuidedUpsample(
        dim_lr=C_lr,
        dim_hr=C_hr,
        dim_out=64,
        heads=4,
        kernel_size=5,
        upsample_mode="bilinear",
        lp_type="gaussian",
        lp_kernel=5,
        lp_sigma=1.0
    )

    y = block(x_lr, g_hr)
    print("Output shape:", y.shape)  # (B, dim_out, H, W)
