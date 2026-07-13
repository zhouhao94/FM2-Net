import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedCrossAttention2D(nn.Module):
    """
    RGB-Depth Cross Attention with:
      1) Spatial reduction on K/V (Lite)
      2) Gated attention (G1): after SDPA output, multiply sigmoid gate from query (RGB)
    Inputs:
      q_feat : (B, Cq, Hq, Wq)  # RGB
      kv_feat: (B, Ckv, Hk, Wk) # Depth
    Output:
      out    : (B, Cq, Hq, Wq)
    """
    def __init__(self, c_q, c_kv=None, num_heads=4, sr_ratio=4, attn_dim_ratio=2,
                 qkv_bias=True, attn_drop=0., proj_drop=0.,
                 gate_type="headwise",  # "headwise" or "elementwise"
                 gate_scale=2.0,        # 让 sigmoid(0)=0.5 变成接近 1 的幅度（更像原模型）
                 kv_down="conv"):
        super().__init__()
        c_kv = c_kv if c_kv is not None else c_q
        assert c_q % num_heads == 0, "c_q must be divisible by num_heads"
        assert gate_type in ("headwise", "elementwise")

        self.num_heads = num_heads
        self.sr_ratio = sr_ratio
        self.gate_type = gate_type
        self.gate_scale = float(gate_scale)

        # attention 内部通道瓶颈
        d_attn = max(c_q // attn_dim_ratio, num_heads)
        d_attn = (d_attn // num_heads) * num_heads
        self.d_attn = d_attn
        self.head_dim = d_attn // num_heads
        self.scale = self.head_dim ** -0.5

        # 1x1 conv 投影
        self.q_proj = nn.Conv2d(c_q,  d_attn, 1, bias=qkv_bias)
        self.k_proj = nn.Conv2d(c_kv, d_attn, 1, bias=qkv_bias)
        self.v_proj = nn.Conv2d(c_kv, d_attn, 1, bias=qkv_bias)

        # K/V 降采样
        if sr_ratio > 1:
            if kv_down == "conv":
                self.kv_down = nn.Conv2d(d_attn, d_attn, kernel_size=sr_ratio, stride=sr_ratio,
                                         padding=0, groups=d_attn, bias=False)  # depthwise stride conv
            elif kv_down == "avgpool":
                self.kv_down = nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio)
            else:
                raise ValueError("kv_down must be 'conv' or 'avgpool'")
        else:
            self.kv_down = nn.Identity()

        # ---- gate（由 query/RGB 生成）----
        # 更稳：对 gate 的输入做 LN（在2D上用 GroupNorm(1, C) 等价于 LN）
        self.gate_norm = nn.GroupNorm(1, c_q)

        if gate_type == "headwise":
            # (B,C,H,W) -> (B,heads,H,W)
            self.gate_proj = nn.Conv2d(c_q, num_heads, 1, bias=True)
            nn.init.constant_(self.gate_proj.bias, 0.0)
        else:
            # (B,C,H,W) -> (B,d_attn,H,W)
            self.gate_proj = nn.Conv2d(c_q, d_attn, 1, bias=True)
            nn.init.constant_(self.gate_proj.bias, 0.0)

        self.attn_drop = nn.Dropout(attn_drop)
        self.out_proj = nn.Conv2d(d_attn, c_q, 1, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q_feat, kv_feat):
        B, _, Hq, Wq = q_feat.shape

        # Q
        q = self.q_proj(q_feat).flatten(2).transpose(1, 2)  # (B, Nq, d_attn)
        Nq = q.shape[1]
        q = q.view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2) * self.scale  # (B,h,Nq,hd)

        # K/V
        k = self.k_proj(kv_feat)
        v = self.v_proj(kv_feat)
        k = self.kv_down(k)  # (B,d_attn,H',W')
        v = self.kv_down(v)
        k = k.flatten(2).transpose(1, 2)  # (B, Nk, d_attn)
        v = v.flatten(2).transpose(1, 2)  # (B, Nk, d_attn)
        Nk = k.shape[1]
        k = k.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # (B,h,Nk,hd)
        v = v.view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)  # (B,h,Nk,hd)

        # SDPA
        attn = (q @ k.transpose(-2, -1)).softmax(dim=-1)  # (B,h,Nq,Nk)
        attn = self.attn_drop(attn)
        out = attn @ v  # (B,h,Nq,hd)

        # ---- G1 gate：由 RGB(query) 生成，乘在 SDPA 输出上 ----
        g_in = self.gate_norm(q_feat)
        if self.gate_type == "headwise":
            gate = torch.sigmoid(self.gate_proj(g_in)) * self.gate_scale  # (B,h,H,W)
            gate = gate.flatten(2).transpose(1, 2)                        # (B,Nq,h)
            gate = gate.transpose(1, 2).unsqueeze(-1)                     # (B,h,Nq,1)
        else:
            gate = torch.sigmoid(self.gate_proj(g_in)) * self.gate_scale  # (B,d_attn,H,W)
            gate = gate.flatten(2).transpose(1, 2)                        # (B,Nq,d_attn)
            gate = gate.view(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)  # (B,h,Nq,hd)

        out = out * gate

        # reshape back
        out = out.transpose(1, 2).contiguous().view(B, Nq, self.d_attn)   # (B,Nq,d_attn)
        out = out.transpose(1, 2).reshape(B, self.d_attn, Hq, Wq)
        out = self.out_proj(out)
        out = self.proj_drop(out)
        return out
