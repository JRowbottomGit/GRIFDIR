"""CNN/UNet score network in pixel space. Scatter mesh→image, UNet, gather back.

Default architecture (simple baseline):
  conv → norm → act (→ FiLM) per block, MaxPool down, bilinear up.

Optional enhancements (boolean flags):
  double_conv       — 2 convs per block instead of 1 (adds depth)
  residual          — residual skip connections in each ConvBlock
  bottleneck_attn   — self-attention at the bottleneck (4×4 = 16 tokens)
  strided_down      — learnable strided Conv2d instead of MaxPool
  dropout           — dropout rate (0 = disabled)

double_conv vs hidden_dim (more filters):
  hidden_dim widens each layer (more features per pixel, one transform).
  double_conv deepens each scale (two sequential nonlinear transforms).
  Both add capacity but differently — depth helps more for score models.
"""
import torch
import torch.nn as nn
from diffusion.embedding import SinusoidalTimeEmbedding, FourierTimeEmbedding


class FiLM(nn.Module):
    def __init__(self, cond_dim, feat_dim):
        super().__init__()
        self.proj = nn.Linear(cond_dim, 2 * feat_dim)

    def forward(self, x, cond):
        s, b = self.proj(cond).chunk(2, dim=-1)
        return x * (1 + s[:, :, None, None]) + b[:, :, None, None]


class ConvBlock(nn.Module):
    """
    Default (double_conv=False, residual=False):
        conv → norm → act → FiLM(cond)

    double_conv=True adds a second conv after FiLM:
        conv1 → norm1 → act → FiLM(cond) → conv2 → norm2 → act → dropout

    residual=True adds a skip connection around the whole block
    (1×1 conv shortcut if in_ch != out_ch).
    """
    def __init__(self, in_ch, out_ch, cond_dim=0, dropout=0.0,
                 double_conv=False, residual=False):
        super().__init__()
        self.double_conv = double_conv
        self.residual    = residual

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act   = nn.SiLU()
        self.film  = FiLM(cond_dim, out_ch) if cond_dim > 0 else None

        if double_conv:
            self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
            self.norm2 = nn.GroupNorm(min(8, out_ch), out_ch)
            self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if residual:
            self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, cond=None):
        h = self.act(self.norm1(self.conv1(x)))
        if self.film is not None and cond is not None:
            h = self.film(h, cond)
        if self.double_conv:
            h = self.drop(self.act(self.norm2(self.conv2(h))))
        if self.residual:
            h = h + self.skip(x)
        return h


class AttnBlock(nn.Module):
    """Self-attention on spatial tokens. At 4×4 bottleneck = 16 tokens, free global context."""
    def __init__(self, ch, n_heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(1, ch)
        self.attn = nn.MultiheadAttention(ch, n_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x).reshape(B, C, H * W).permute(0, 2, 1)
        h, _ = self.attn(h, h, h)
        return x + h.permute(0, 2, 1).reshape(B, C, H, W)


class DownBlock(nn.Module):
    def __init__(self, ch, cond_dim, dropout=0.0, double_conv=False,
                 residual=False, strided_down=False):
        super().__init__()
        _kw = dict(cond_dim=cond_dim, dropout=dropout,
                   double_conv=double_conv, residual=residual)
        self.block = ConvBlock(ch, ch, **_kw)
        if strided_down:
            self.pool = nn.Conv2d(ch, ch, 4, stride=2, padding=1)
        else:
            self.pool = nn.MaxPool2d(2)

    def forward(self, x, cond):
        x = self.block(x, cond)
        return self.pool(x), x  # (downsampled, skip)


class UpBlock(nn.Module):
    def __init__(self, ch, cond_dim, dropout=0.0, double_conv=False, residual=False):
        super().__init__()
        self.up    = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.block = ConvBlock(ch * 2, ch, cond_dim=cond_dim, dropout=dropout,
                               double_conv=double_conv, residual=residual)

    def forward(self, x, skip, cond):
        return self.block(torch.cat([self.up(x), skip], dim=1), cond)


class CNNScoreNetwork(nn.Module):
    """
    2-D UNet score network. Requires set_mesh_hierarchy() before forward().

    forward(inp, t, pos) -> score   [same interface as MultiscaleGNNScoreNetwork]

    Enhancement flags (all default False/0):
        double_conv     — 2 convs per block
        residual        — skip connections in each block
        bottleneck_attn — self-attention at bottleneck
        strided_down    — strided Conv2d instead of MaxPool
        dropout         — dropout probability
    """
    def __init__(self, input_dim=1, pos_dim=2, hidden_dim=64, time_dim=64,
                 n_levels=4, time_embedding='fourier',
                 grid_h=32, grid_w=32, tris_per_pixel=2,
                 double_conv=False, residual=False,
                 bottleneck_attn=False, strided_down=False,
                 dropout=0.0, attn_heads=4):
        super().__init__()
        self.input_dim       = input_dim
        self.pos_dim         = pos_dim
        self.grid_h          = grid_h
        self.grid_w          = grid_w
        self.tris_per_pixel  = tris_per_pixel
        self._in_ch          = input_dim * tris_per_pixel
        self._bottleneck_attn = bottleneck_attn

        self.time_embed = (SinusoidalTimeEmbedding(time_dim)
                           if time_embedding == 'sinusoidal'
                           else FourierTimeEmbedding(time_dim))

        H   = hidden_dim
        _kw = dict(dropout=dropout, double_conv=double_conv, residual=residual)

        self.in_conv = nn.Conv2d(self._in_ch, H, 3, padding=1)
        self.downs   = nn.ModuleList([
            DownBlock(H, time_dim, strided_down=strided_down, **_kw)
            for _ in range(n_levels - 1)
        ])
        self.bottle  = ConvBlock(H, H, cond_dim=time_dim, **_kw)
        if bottleneck_attn:
            self.bottle_attn = AttnBlock(H, n_heads=attn_heads)
            self.bottle2     = ConvBlock(H, H, cond_dim=time_dim, **_kw)
        self.ups     = nn.ModuleList([
            UpBlock(H, time_dim, **_kw) for _ in range(n_levels - 1)
        ])
        self.out_conv = nn.Conv2d(H, self._in_ch, 1)

        self.register_buffer('_node_to_pixel', None)
        self.register_buffer('_node_tri_idx',  None)

    # ------------------------------------------------------------------
    def set_mesh_hierarchy(self, edge_indices, n_nodes_list, pool_edges,
                           unpool_maps, coarse_coords=None, level_coords=None,
                           lap_pe=None):
        if level_coords is None:
            raise RuntimeError("CNNScoreNetwork requires level_coords in set_mesh_hierarchy")
        self._build_node_map(level_coords[0], n_nodes_list[0])

    def _build_node_map(self, coords, n_nodes):
        H, W = self.grid_h, self.grid_w
        c  = coords.float()
        ix = (c[:, 0] * W).floor().long().clamp(0, W - 1)
        iy = (c[:, 1] * H).floor().long().clamp(0, H - 1)
        pixel_id = iy * W + ix  # [N]

        # Assign local triangle index (0 or 1) within each pixel, sorted by x
        order    = torch.argsort(pixel_id.double() * n_nodes
                                 + torch.arange(n_nodes, dtype=torch.double))
        tri_idx  = torch.zeros(n_nodes, dtype=torch.long)
        seen: dict = {}
        for i in order.tolist():
            pid          = pixel_id[i].item()
            tri_idx[i]   = seen.get(pid, 0)
            seen[pid]    = seen.get(pid, 0) + 1

        device = next(self.parameters()).device
        self._node_to_pixel = pixel_id.to(device)
        self._node_tri_idx  = tri_idx.to(device)

    # ------------------------------------------------------------------
    def _to_image(self, signal):
        """[B, N, C] -> [B, C*T, H, W]"""
        B, N, C = signal.shape
        H, W, T = self.grid_h, self.grid_w, self.tris_per_pixel
        img = signal.new_zeros(B, C * T, H, W)
        iy  = self._node_to_pixel // W
        ix  = self._node_to_pixel %  W
        for c_t in range(C * T):
            c, t = c_t // T, c_t % T
            m    = self._node_tri_idx == t
            img[:, c_t, iy[m], ix[m]] = signal[:, m, c]
        return img

    def _from_image(self, img):
        """[B, C*T, H, W] -> [B, N, C]"""
        B  = img.shape[0]
        T  = self.tris_per_pixel
        C  = img.shape[1] // T
        iy = self._node_to_pixel // self.grid_w
        ix = self._node_to_pixel %  self.grid_w
        vals = img[:, :, iy, ix]  # [B, C*T, N]
        c_off = torch.arange(C, device=img.device) * T
        idx   = (c_off.unsqueeze(-1) + self._node_tri_idx.unsqueeze(0)
                 ).unsqueeze(0).expand(B, -1, -1)  # [B, C, N]
        return vals.gather(1, idx).permute(0, 2, 1)  # [B, N, C]

    # ------------------------------------------------------------------
    def forward(self, inp, t, pos, domain_onehot=None):
        if self._node_to_pixel is None:
            raise RuntimeError("Call set_mesh_hierarchy() before forward()")
        signal = inp[:, self.pos_dim:, :].permute(0, 2, 1)  # [B, N, C]
        t_emb  = self.time_embed(t)                          # [B, T_dim]

        x      = self.in_conv(self._to_image(signal))            # [B, H, h, w]
        skips  = []
        for d in self.downs:
            x, sk = d(x, t_emb)
            skips.append(sk)
        x = self.bottle(x, t_emb)
        if self._bottleneck_attn:
            x = self.bottle2(self.bottle_attn(x), t_emb)
        for u, sk in zip(self.ups, reversed(skips)):
            x = u(x, sk, t_emb)
        x = self.out_conv(x)                                 # [B, C*T, h, w]
        return self._from_image(x).permute(0, 2, 1)         # [B, C, N]
