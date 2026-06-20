"""FNO score network in pixel space — a thin wrapper over the upstream
neuraloperator FNO (https://github.com/neuraloperator/neuraloperator), which
must be installed separately (`pip install neuraloperator`). The operator is
NOT vendored here.

Same mesh↔grid scatter/gather interface as CNNScoreNetwork:
  set_mesh_hierarchy(...)  — builds node→pixel map from finest-level coords
  forward(inp, t, pos)     — [B, pos+1, N] → [B, 1, N]  (same as CNN/MultiscaleGNN)

Architecture:
  mesh → scatter → [B, _in_ch, H, W]
  time embedding → project → [B, T, H, W]  (broadcast)
  cat → [B, _in_ch + T, H, W]
  FNO → [B, _in_ch, H, W]
  gather → [B, 1, N]

FNO operates on a fixed regular grid (same nx×ny used for data generation).
This makes it a standard discrete baseline — it cannot generalise to
different mesh resolutions without retraining.
"""

import torch
import torch.nn as nn

from diffusion.embedding import SinusoidalTimeEmbedding, FourierTimeEmbedding


def _import_fno():
    """Import the FNO operator from the upstream ``neuraloperator`` package.

    The FNO baseline is NOT vendored. Install the upstream package with
    ``pip install neuraloperator`` (https://github.com/neuraloperator/neuraloperator).
    Imported lazily so this module loads even when neuraloperator is absent.
    """
    try:
        from neuralop.models import FNO
    except ImportError as exc:
        raise ImportError(
            "The FNO baseline requires the upstream neuraloperator package "
            "(`pip install neuraloperator`)."
        ) from exc
    return FNO


# ---------------------------------------------------------------------------
# FNO Score Network
# ---------------------------------------------------------------------------

class FNOScoreNetwork(nn.Module):
    """
    2-D FNO score network. Requires set_mesh_hierarchy() before forward().

    forward(inp, t, pos) -> score   [same interface as MultiscaleGNNScoreNetwork / CNNScoreNetwork]

    Args:
        input_dim:      signal channels (1 for scalar field)
        pos_dim:        position/conditioning channels prepended to inp
        hidden_dim:     FNO hidden_channels (width)
        time_dim:       time embedding dimension
        n_modes:        Fourier modes per spatial dim (default 16)
        n_fno_layers:   number of FNO Fourier blocks (default 4)
        grid_h, grid_w: regular grid resolution (must match data nx, ny)
        tris_per_pixel: mesh nodes per pixel cell (default 2, matches CNN)
        time_embedding: 'fourier' or 'sinusoidal'
        n_time_feats:   channels used to inject time into FNO input (default 8)
    """

    def __init__(self, input_dim=1, pos_dim=2, hidden_dim=64, time_dim=64,
                 n_modes=16, n_fno_layers=4, grid_h=32, grid_w=32,
                 tris_per_pixel=2, time_embedding='fourier', n_time_feats=8,
                 positional_embedding=None, domain_padding=0.0):
        super().__init__()
        self.input_dim      = input_dim
        self.pos_dim        = pos_dim
        self.grid_h         = grid_h
        self.grid_w         = grid_w
        self.tris_per_pixel = tris_per_pixel
        self._in_ch         = input_dim * tris_per_pixel

        self.time_embed = (SinusoidalTimeEmbedding(time_dim)
                           if time_embedding == 'sinusoidal'
                           else FourierTimeEmbedding(time_dim))

        # Project scalar time embedding → spatial channels to concat with signal
        self.time_proj = nn.Linear(time_dim, n_time_feats)

        # FNO: input has signal + time channels, output reconstructs signal
        _fno_kwargs = dict(
            n_modes=(n_modes, n_modes),
            in_channels=self._in_ch + n_time_feats,
            out_channels=self._in_ch,
            hidden_channels=hidden_dim,
            n_layers=n_fno_layers,
            positional_embedding=positional_embedding,
        )
        if domain_padding and domain_padding > 0.0:
            _fno_kwargs['domain_padding'] = float(domain_padding)
        FNO = _import_fno()
        self.fno = FNO(**_fno_kwargs)

        self.register_buffer('_node_to_pixel', None)
        self.register_buffer('_node_tri_idx',  None)

    # ------------------------------------------------------------------
    # Mesh setup (called once after model creation)
    # ------------------------------------------------------------------

    def set_mesh_hierarchy(self, edge_indices, n_nodes_list, pool_edges,
                           unpool_maps, coarse_coords=None, level_coords=None,
                           lap_pe=None):
        if level_coords is None:
            raise RuntimeError("FNOScoreNetwork requires level_coords in set_mesh_hierarchy")
        self._build_node_map(level_coords[0], n_nodes_list[0])

    def _build_node_map(self, coords, n_nodes):
        H, W = self.grid_h, self.grid_w
        c  = coords.float()
        ix = (c[:, 0] * W).floor().long().clamp(0, W - 1)
        iy = (c[:, 1] * H).floor().long().clamp(0, H - 1)
        pixel_id = iy * W + ix  # [N]

        order   = torch.argsort(pixel_id.double() * n_nodes
                                + torch.arange(n_nodes, dtype=torch.double))
        tri_idx = torch.zeros(n_nodes, dtype=torch.long)
        seen: dict = {}
        for i in order.tolist():
            pid         = pixel_id[i].item()
            tri_idx[i]  = seen.get(pid, 0)
            seen[pid]   = seen.get(pid, 0) + 1

        device = next(self.parameters()).device
        self._node_to_pixel = pixel_id.to(device)
        self._node_tri_idx  = tri_idx.to(device)

    # ------------------------------------------------------------------
    # Scatter / gather helpers (identical to CNNScoreNetwork)
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
        vals  = img[:, :, iy, ix]   # [B, C*T, N]
        c_off = torch.arange(C, device=img.device) * T
        idx   = (c_off.unsqueeze(-1) + self._node_tri_idx.unsqueeze(0)
                 ).unsqueeze(0).expand(B, -1, -1)  # [B, C, N]
        return vals.gather(1, idx).permute(0, 2, 1)  # [B, N, C]

    # ------------------------------------------------------------------

    def forward(self, inp, t, pos, domain_onehot=None):
        if self._node_to_pixel is None:
            raise RuntimeError("Call set_mesh_hierarchy() before forward()")

        # inp: [B, pos_dim + input_dim, N] — extract signal channels
        signal = inp[:, self.pos_dim:, :].permute(0, 2, 1)  # [B, N, C]
        t_emb  = self.time_embed(t)                          # [B, time_dim]

        # Scatter to grid
        x = self._to_image(signal)                           # [B, _in_ch, H, W]

        # Broadcast time embedding to spatial dims and concat
        t_spatial = self.time_proj(t_emb)                    # [B, n_time_feats]
        t_spatial = t_spatial[:, :, None, None].expand(
            -1, -1, self.grid_h, self.grid_w)                # [B, n_time_feats, H, W]
        x = torch.cat([x, t_spatial], dim=1)                 # [B, _in_ch + T, H, W]

        # FNO forward
        x = self.fno(x)                                      # [B, _in_ch, H, W]

        # Gather back to mesh nodes
        return self._from_image(x).permute(0, 2, 1)         # [B, C, N]
