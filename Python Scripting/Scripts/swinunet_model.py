"""
SwinUNet - Swin Transformer Architecture for Image Denoising
=============================================================
Production-quality implementation inspired by SwinIR.

Uses window-based multi-head self-attention with shifted windows
and residual learning for high-fidelity denoising.

Architecture:
    Input -> Shallow Feature Extraction (Conv3x3)
          -> Deep Feature Extraction (4x RSTB blocks, 24 transformer layers)
          -> Feature Reconstruction (Conv3x3 + Feature Residual)
          -> Global Residual Learning (+ Input)
          -> Output

Handles arbitrary input sizes via automatic reflect-padding.
Pixel values expected in [0, 1] range.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# =====================================================================
#  Window Utilities
# =====================================================================

def window_partition(x, window_size):
    """Partition feature map into non-overlapping windows.
    (B, H, W, C) -> (B*nW, ws, ws, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(
        -1, window_size, window_size, C)


def window_reverse(windows, window_size, H, W):
    """Reverse window partition back to feature map.
    (B*nW, ws, ws, C) -> (B, H, W, C)
    """
    nW = (H // window_size) * (W // window_size)
    B = windows.shape[0] // nW
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


# =====================================================================
#  Building Blocks
# =====================================================================

class Mlp(nn.Module):
    """Feed-forward network: Linear -> GELU -> Drop -> Linear -> Drop."""

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, drop=0.0):
        super().__init__()
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class WindowAttention(nn.Module):
    """Window-based Multi-Head Self-Attention with learnable
    relative position bias.

    Args:
        dim:         Number of input channels.
        window_size: (Wh, Ww) tuple.
        num_heads:   Number of attention heads.
        qkv_bias:    Add bias to QKV projection.
        attn_drop:   Attention dropout rate.
        proj_drop:   Output projection dropout rate.
    """

    def __init__(self, dim, window_size, num_heads,
                 qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size          # (Wh, Ww)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        # --- Relative position bias table ---
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # --- Compute pair-wise relative position index ---
        coords_h = torch.arange(window_size[0])
        coords_w = torch.arange(window_size[1])
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing='ij'))   # (2, Wh, Ww)
        coords_flat = torch.flatten(coords, 1)                   # (2, N)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, N, N)
        rel = rel.permute(1, 2, 0).contiguous()                  # (N, N, 2)
        rel[:, :, 0] += window_size[0] - 1
        rel[:, :, 1] += window_size[1] - 1
        rel[:, :, 0] *= 2 * window_size[1] - 1
        self.register_buffer("relative_position_index", rel.sum(-1))

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mask=None):
        """
        Args:
            x:    (num_windows*B, N, C)  where N = Wh*Ww
            mask: (num_windows, N, N)    or None
        """
        B_, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv.unbind(0)   # each: (B_, nH, N, head_dim)

        attn = (q * self.scale) @ k.transpose(-2, -1)  # (B_, nH, N, N)

        # Add relative position bias
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(N, N, -1).permute(2, 0, 1).contiguous()  # (nH, N, N)
        attn = attn + bias.unsqueeze(0)

        # Apply shifted-window mask
        if mask is not None:
            nW = mask.shape[0]
            attn = (attn.view(B_ // nW, nW, self.num_heads, N, N)
                    + mask.unsqueeze(1).unsqueeze(0))
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class SwinTransformerLayer(nn.Module):
    """Single Swin Transformer layer - either W-MSA or SW-MSA.

    Args:
        dim:         Input feature channels.
        num_heads:   Attention heads.
        window_size: Local attention window size.
        shift_size:  Cyclic shift for SW-MSA (0 = W-MSA).
        mlp_ratio:   MLP expansion ratio.
    """

    def __init__(self, dim, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=2.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(
            dim, (window_size, window_size), num_heads,
            qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    # ------------------------------------------------------------------
    def _make_mask(self, H, W, device):
        """Build the attention mask for shifted-window MSA."""
        if self.shift_size == 0:
            return None
        mask = torch.zeros(1, H, W, 1, device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                mask[:, h, w, :] = cnt
                cnt += 1
        mw = window_partition(mask, self.window_size)       # (nW, ws, ws, 1)
        mw = mw.view(-1, self.window_size * self.window_size)
        am = mw.unsqueeze(1) - mw.unsqueeze(2)             # (nW, N, N)
        return am.masked_fill(am != 0, -100.0).masked_fill(am == 0, 0.0)

    # ------------------------------------------------------------------
    def forward(self, x, H, W):
        """
        Args:
            x: (B, H*W, C)
            H, W: spatial resolution
        Returns:
            (B, H*W, C)
        """
        B, L, C = x.shape
        shortcut = x

        x = self.norm1(x).view(B, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x,
                           shifts=(-self.shift_size, -self.shift_size),
                           dims=(1, 2))

        # Window partition -> attention -> reverse
        xw = window_partition(x, self.window_size)                      # (nW*B, ws, ws, C)
        xw = xw.view(-1, self.window_size * self.window_size, C)       # (nW*B, N, C)
        xw = self.attn(xw, mask=self._make_mask(H, W, x.device))       # (nW*B, N, C)
        xw = xw.view(-1, self.window_size, self.window_size, C)        # (nW*B, ws, ws, C)
        x = window_reverse(xw, self.window_size, H, W)                 # (B, H, W, C)

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x,
                           shifts=(self.shift_size, self.shift_size),
                           dims=(1, 2))

        # Residual + MLP
        x = shortcut + x.view(B, L, C)
        x = x + self.mlp(self.norm2(x))
        return x


class RSTB(nn.Module):
    """Residual Swin Transformer Block.

    N Swin Transformer layers with alternating W-MSA / SW-MSA,
    followed by a 3x3 convolution and an outer residual connection.
    """

    def __init__(self, dim, depth, num_heads, window_size=8,
                 mlp_ratio=2.0, qkv_bias=True, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.use_checkpoint = False  # toggled by training script
        self.layers = nn.ModuleList([
            SwinTransformerLayer(
                dim, num_heads, window_size,
                shift_size=0 if i % 2 == 0 else window_size // 2,
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop, attn_drop=attn_drop)
            for i in range(depth)
        ])
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

    def forward(self, x):
        """(B, C, H, W) -> (B, C, H, W)"""
        B, C, H, W = x.shape
        residual = x

        feat = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                feat = checkpoint(layer, feat, H, W,
                                  use_reentrant=False)
            else:
                feat = layer(feat, H, W)
        feat = feat.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        return self.conv(feat) + residual


# =====================================================================
#  Main Model
# =====================================================================

class SwinUnet(nn.Module):
    """SwinUNet for Image Denoising.

    A SwinIR-inspired architecture using Residual Swin Transformer
    Blocks with global residual learning.  Processes grayscale images
    in [0, 1] range and supports arbitrary spatial resolutions.

    Default config: ~2.3 M parameters (embed_dim=96, 4 RSTBs x 6 layers).

    Args:
        img_size:       Reference patch size for training (not a hard constraint).
        in_chans:       Number of input channels (1 for grayscale).
        embed_dim:      Base feature embedding dimension.
        depths:         Number of Swin TF layers inside each RSTB.
        num_heads:      Attention heads per RSTB.
        window_size:    Local window size for W-MSA / SW-MSA.
        mlp_ratio:      MLP hidden-dim expansion ratio.
        qkv_bias:       Bias in QKV projections.
        drop_rate:      Dropout rate.
        attn_drop_rate: Attention dropout rate.
    """

    def __init__(self, img_size=128, in_chans=1, embed_dim=96,
                 depths=(6, 6, 6, 6), num_heads=(6, 6, 6, 6),
                 window_size=8, mlp_ratio=2.0, qkv_bias=True,
                 drop_rate=0.0, attn_drop_rate=0.0):
        super().__init__()
        self.window_size = window_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # ---- Shallow Feature Extraction ----
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)

        # ---- Deep Feature Extraction (RSTB cascade) ----
        self.rstb_layers = nn.ModuleList([
            RSTB(embed_dim, depths[i], num_heads[i], window_size,
                 mlp_ratio, qkv_bias, drop_rate, attn_drop_rate)
            for i in range(len(depths))
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # ---- Image Reconstruction ----
        self.conv_last = nn.Conv2d(embed_dim, in_chans, 3, 1, 1)

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                    nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)  noisy image, pixel values in [0, 1]
        Returns:
            (B, C, H, W)  denoised image, clamped to [0, 1]
        """
        _, _, H, W = x.shape
        ws = self.window_size

        # Reflect-pad to nearest window_size multiple
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        x_in = x  # save padded input for global residual

        # Shallow feature extraction
        shallow = self.conv_first(x_in)

        # Deep feature extraction
        deep = shallow
        for rstb in self.rstb_layers:
            deep = rstb(deep)

        # Layer norm on deep features
        B, C, Hp, Wp = deep.shape
        deep = (deep.permute(0, 2, 3, 1)
                .reshape(B, Hp * Wp, C))
        deep = self.norm(deep)
        deep = (deep.view(B, Hp, Wp, C)
                .permute(0, 3, 1, 2).contiguous())

        # Feature residual + image reconstruction + global residual
        out = self.conv_last(self.conv_after_body(deep) + shallow) + x_in

        # Remove padding and clamp
        out = out[:, :, :H, :W]
        return torch.clamp(out, 0.0, 1.0)
