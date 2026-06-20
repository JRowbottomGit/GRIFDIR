"""GAOT (Geometry-Aware Operator Transformer) score network.

Baseline from https://github.com/camlab-ethz/GAOT (Wen et al., 2025).

Architecture: MAGNO encoder (mesh -> regular latent grid) -> Vision Transformer
-> MAGNO decoder (latent grid -> mesh). Naturally supports arbitrary mesh
coordinates; we treat it as a geometry-aware baseline.

Interface matches MultiscaleGNNScoreNetwork / FNOScoreNetwork / CNNScoreNetwork:
  set_mesh_hierarchy(...)  — registers finest-level coords as xcoord
  forward(inp, t, pos)     — [B, pos_dim + 1, N] -> [B, 1, N]

Time conditioning: per GAOT's own docstring we concat a projected time
embedding to the per-node features (same pattern as FNO/CNN wrappers)
rather than the experimental `condition=` argument.

GAOT's own dependencies (`rotary-embedding-torch`, and optionally
`torch_scatter`/`torch_cluster`/`open3d`) must be installed in the
training env. We default to `neighbor_search_method='native'` and
`use_torch_scatter=False` so only `rotary-embedding-torch` is strictly
required.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn as nn

from diffusion.embedding import SinusoidalTimeEmbedding, FourierTimeEmbedding

# ---------------------------------------------------------------------------
# External GAOT repo wiring.
#
# GAOT (https://github.com/camlab-ethz/GAOT) is NOT vendored here and is not
# pip-installable. Clone it and set $GAOT_REPO to point at it. Resolution +
# backend shims run lazily from GAOTScoreNetwork.__init__ so importing this
# module has no side effects.
# ---------------------------------------------------------------------------

def _ensure_gaot_on_path():
    """Add the external GAOT repo to sys.path and install backend shims."""
    repo = os.environ.get('GAOT_REPO')
    if not repo or not os.path.isdir(repo):
        raise ImportError(
            "The GAOT baseline requires the upstream GAOT repository "
            "(https://github.com/camlab-ethz/GAOT). Clone it and point the "
            "GAOT_REPO environment variable at it."
        )
    if repo not in sys.path:
        sys.path.insert(0, repo)
    _install_torch_scatter_stub()


# ---------------------------------------------------------------------------
# torch_scatter stub (XPU/CPU fallback)
#
# GAOT's gemb.py imports `from torch_scatter import scatter_mean, scatter_sum,
# scatter_max` at module load time. There's no XPU wheel. We stub the module
# with scatter_reduce-based equivalents BEFORE GAOT imports happen. If a real
# torch_scatter is already installed (e.g. on CUDA), we leave it alone.
# ---------------------------------------------------------------------------

def _install_torch_scatter_stub():
    if 'torch_scatter' in sys.modules:
        return
    try:
        import torch_scatter  # noqa: F401
        return
    except ImportError:
        pass

    import types
    import torch as _torch

    def _gather_segment(src, index, dim, dim_size, reduce_op, include_self=False):
        if dim_size is None:
            dim_size = int(index.max().item()) + 1 if index.numel() > 0 else 0
        # Build output shape: same as src, but src.shape[dim] replaced by dim_size.
        out_shape = list(src.shape)
        out_shape[dim] = dim_size
        out = _torch.zeros(out_shape, device=src.device, dtype=src.dtype)
        # Broadcast index to src shape
        idx_shape = [1] * src.ndim
        idx_shape[dim] = index.shape[0]
        idx = index.view(idx_shape).expand_as(src)
        out.scatter_reduce_(dim, idx, src, reduce=reduce_op, include_self=include_self)
        return out

    def scatter_mean(src, index, dim=0, out=None, dim_size=None):
        return _gather_segment(src, index, dim, dim_size, 'mean')

    def scatter_sum(src, index, dim=0, out=None, dim_size=None):
        return _gather_segment(src, index, dim, dim_size, 'sum')

    def scatter_max(src, index, dim=0, out=None, dim_size=None):
        # torch_scatter.scatter_max returns (values, argmax); caller ignores argmax.
        values = _gather_segment(src, index, dim, dim_size, 'amax')
        # argmax stub (torch_scatter returns -1 for empty segments); not used by GAOT.
        return values, None

    mod = types.ModuleType('torch_scatter')
    mod.scatter_mean = scatter_mean
    mod.scatter_sum  = scatter_sum
    mod.scatter_max  = scatter_max
    # Some code paths access torch_scatter.segment_csr — provide it too.
    def _segment_csr_stub(src, indptr, reduce='sum'):
        return _segment_csr_scatter_reduce(src, indptr, reduce)
    mod.segment_csr  = _segment_csr_stub
    sys.modules['torch_scatter'] = mod


# Lazy imports happen in __init__ (via _ensure_gaot_on_path) so that simply
# importing this module does not require GAOT or its deps to be available.


# ---------------------------------------------------------------------------
# Backend-agnostic segment_csr replacement
#
# GAOT's bundled segment_csr fallback (used whenever torch_scatter is not
# installed) only supports reduce='mean'/'sum'. MAGNO's cosine attention
# softmax calls it with reduce='max' — which raises ValueError on CPU/XPU
# builds that have no torch_scatter wheel.
#
# We replace it with a scatter_reduce-based version that works identically
# on CUDA, XPU, and CPU. Activated once at module import — GAOT's code
# imports `segment_csr` from its utils module, so patching the module
# binding is sufficient.
# ---------------------------------------------------------------------------

def _segment_csr_scatter_reduce(src, indptr, reduce, use_scatter=True):
    """CSR-format reduction using torch.scatter_reduce_.

    Args:
        src: [N_edges, C] (or [B, N_edges, C] when batched)
        indptr: [N_queries + 1] (or [B, N_queries + 1])
        reduce: 'max' | 'sum' | 'mean'
    Returns:
        [N_queries, C] (or [B, N_queries, C])
    """
    import torch as _torch  # local to keep top-level imports tidy

    batched = src.ndim == 3
    if batched:
        indptr_ref = indptr[0] if indptr.ndim == 2 else indptr
    else:
        indptr_ref = indptr

    n_queries = indptr_ref.shape[0] - 1
    counts = indptr_ref[1:] - indptr_ref[:-1]           # [N_queries]
    # per-edge query-index: edge j belongs to query segment[j]
    segment = _torch.repeat_interleave(
        _torch.arange(n_queries, device=src.device), counts
    )                                                    # [N_edges]

    if reduce == 'sum':
        op, include_self = 'sum', False
    elif reduce == 'mean':
        op, include_self = 'mean', False
    elif reduce == 'max':
        op, include_self = 'amax', False
    else:
        raise ValueError(f"Unsupported reduce: {reduce}")

    if batched:
        B = src.shape[0]
        C = src.shape[2]
        out = _torch.zeros(B, n_queries, C, device=src.device, dtype=src.dtype)
        idx = segment.view(1, -1, 1).expand(B, -1, C)
        out.scatter_reduce_(1, idx, src, reduce=op, include_self=include_self)
    else:
        if src.ndim == 1:
            out = _torch.zeros(n_queries, device=src.device, dtype=src.dtype)
            out.scatter_reduce_(0, segment, src, reduce=op, include_self=include_self)
        else:
            C = src.shape[1]
            out = _torch.zeros(n_queries, C, device=src.device, dtype=src.dtype)
            idx = segment.unsqueeze(-1).expand(-1, C)
            out.scatter_reduce_(0, idx, src, reduce=op, include_self=include_self)
    return out


_SEGMENT_CSR_PATCHED = False


def _patch_gaot_segment_csr():
    """Replace GAOT's segment_csr with our scatter_reduce version (idempotent).

    Must be called after sys.path contains the GAOT repo.
    """
    global _SEGMENT_CSR_PATCHED
    if _SEGMENT_CSR_PATCHED:
        return
    # Patch in BOTH locations GAOT accesses it from:
    #   - the utils module (source of truth)
    #   - the agno module (already bound its own `segment_csr` via `from ... import`)
    import src.model.layers.utils.segment_csr as _sc_mod
    _sc_mod.segment_csr = _segment_csr_scatter_reduce
    try:
        import src.model.layers.agno as _agno_mod
        _agno_mod.segment_csr = _segment_csr_scatter_reduce
    except ImportError:
        pass
    _SEGMENT_CSR_PATCHED = True


# ---------------------------------------------------------------------------
# Minimal config container mimicking GAOT's expected nested dataclass layout
# (config.args.magno, config.args.transformer, config.latent_tokens_size)
# ---------------------------------------------------------------------------

@dataclass
class _GAOTArgs:
    magno: object = None
    transformer: object = None


@dataclass
class _GAOTConfig:
    latent_tokens_size: List[int] = field(default_factory=lambda: [32, 32])
    args: _GAOTArgs = field(default_factory=_GAOTArgs)


# ---------------------------------------------------------------------------
# GAOT Score Network
# ---------------------------------------------------------------------------

class GAOTScoreNetwork(nn.Module):
    """GAOT wrapped to match the GRIFDIR score-network interface.

    Args:
        input_dim:              signal channels (1 for scalar field)
        pos_dim:                position/conditioning channels prepended in `inp`
                                (kept for signature parity; not fed to GAOT)
        hidden_dim:             transformer hidden size (GAOT's `transformer.hidden_size`)
        time_dim:               time embedding dimension
        n_time_feats:           channels used to inject time into per-node features
        time_embedding:         'fourier' or 'sinusoidal'
        latent_tokens_size:     regular latent grid `[H, W]` (both divisible by patch_size)
        patch_size:             ViT patch size (must divide H and W)
        n_transformer_layers:   number of transformer blocks
        magno_radius:           MAGNO neighbor-search radius (coords in [0,1])
        magno_hidden:           MAGNO internal MLP hidden size
        magno_mlp_layers:       MAGNO MLP depth
        magno_lifting:          MAGNO lifting channels (per-node latent width)
        positional_embedding:   'absolute' (default) or 'rope'
        use_geoembed:           GAOT's statistical geometric embedding
    """

    def __init__(self,
                 input_dim: int = 1,
                 pos_dim: int = 2,
                 hidden_dim: int = 256,
                 time_dim: int = 128,
                 n_time_feats: int = 8,
                 time_embedding: str = 'fourier',
                 latent_tokens_size: Optional[List[int]] = None,
                 patch_size: int = 2,
                 n_transformer_layers: int = 3,
                 magno_radius: float = 0.05,
                 magno_hidden: int = 64,
                 magno_mlp_layers: int = 3,
                 magno_lifting: int = 64,
                 positional_embedding: str = 'absolute',
                 use_geoembed: bool = True,
                 use_attention: bool = True,
                 use_torch_scatter: bool = False,
                 n_domains: int = 0):
        # n_domains: if > 0, append a per-batch broadcast of the domain
        # one-hot ([B, n_domains] -> [B, N, n_domains]) onto the per-node
        # input features alongside the time embedding. Only used in
        # multidomain training; pass 0 to disable.
        super().__init__()
        self.input_dim    = input_dim
        self.pos_dim      = pos_dim
        self.n_time_feats = n_time_feats

        if latent_tokens_size is None:
            latent_tokens_size = [32, 32]
        assert len(latent_tokens_size) == 2, \
            f"latent_tokens_size must be 2D, got {latent_tokens_size}"
        H, W = latent_tokens_size
        assert H % patch_size == 0 and W % patch_size == 0, \
            f"latent_tokens_size {latent_tokens_size} must be divisible by patch_size {patch_size}"
        self.H, self.W = H, W

        # ── Time embedding (scalar per-sample -> broadcast to nodes) ──────
        self.time_embed = (SinusoidalTimeEmbedding(time_dim)
                           if time_embedding == 'sinusoidal'
                           else FourierTimeEmbedding(time_dim))
        self.time_proj = nn.Linear(time_dim, n_time_feats)

        # ── Build GAOT config using GAOT's own dataclasses ────────────────
        _ensure_gaot_on_path()   # add the external GAOT repo to sys.path (+ shims)
        from src.model.layers.magno import MAGNOConfig
        from src.model.layers.attn import TransformerConfig, AttentionConfig
        from src.model.gaot import GAOT
        # Patch segment_csr AFTER GAOT's agno/magno modules have been
        # imported (so our replacement fn overrides their bound reference).
        _patch_gaot_segment_csr()

        magno_cfg = MAGNOConfig(
            coord_dim=2,
            radius=magno_radius,
            hidden_size=magno_hidden,
            mlp_layers=magno_mlp_layers,
            lifting_channels=magno_lifting,
            use_attention=use_attention,
            attention_type='cosine',
            use_geoembed=use_geoembed,
            embedding_method='statistical',
            transform_type='linear',
            neighbor_search_method='native',
            use_torch_scatter=use_torch_scatter,
            precompute_edges=False,
        )
        transformer_cfg = TransformerConfig(
            patch_size=patch_size,
            hidden_size=hidden_dim,
            num_layers=n_transformer_layers,
            positional_embedding=positional_embedding,
            use_long_range_skip=True,
            ffn_multiplier=4,
            attn_config=AttentionConfig(num_heads=8, num_kv_heads=8),
        )
        gaot_top_cfg = _GAOTConfig(
            latent_tokens_size=list(latent_tokens_size),
            args=_GAOTArgs(magno=magno_cfg, transformer=transformer_cfg),
        )

        # Per-node input to GAOT = signal + time + (optional) domain one-hot
        self.n_domains = int(n_domains)
        gaot_in = input_dim + n_time_feats + self.n_domains
        self.gaot = GAOT(input_size=gaot_in, output_size=input_dim, config=gaot_top_cfg)

        # ── Coordinate buffers (populated in set_mesh_hierarchy) ──────────
        # _xcoord is a plain attribute, NOT a registered buffer: in multidomain
        # training the mesh shape changes per batch (different shapes have
        # different N_nodes), and EMA's copy_params_from_model_to_ema would
        # fail on buffers whose shape changes between calls. We .to(device)
        # it explicitly in set_mesh_hierarchy. _latent_tokens_coord is a
        # registered buffer because its shape (H*W, 2) is fixed at __init__.
        self._xcoord = None
        self.register_buffer('_latent_tokens_coord', self._build_latent_grid(H, W))

        # Exposed for swap_mesh_hierarchy (multiscale eval path).
        self.n_levels           = 1
        self.latent_transformer = None
        self.lap_pe_dim         = 0

    # ------------------------------------------------------------------
    # Latent grid construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_latent_grid(H: int, W: int) -> torch.Tensor:
        """Regular grid in [0,1]^2, row-major (iy*W + ix), shape [H*W, 2]."""
        ys = (torch.arange(H, dtype=torch.float32) + 0.5) / H
        xs = (torch.arange(W, dtype=torch.float32) + 0.5) / W
        gy, gx = torch.meshgrid(ys, xs, indexing='ij')
        return torch.stack([gx.reshape(-1), gy.reshape(-1)], dim=-1)  # [H*W, 2]

    # ------------------------------------------------------------------
    # Mesh setup (called once after model creation, same signature as
    # FNO/CNN/MultiscaleGNN score networks)
    # ------------------------------------------------------------------

    def set_mesh_hierarchy(self, edge_indices, n_nodes_list, pool_edges,
                           unpool_maps, coarse_coords=None, level_coords=None,
                           lap_pe=None, cells_list=None, xy_list=None):
        self.n_levels = len(n_nodes_list)
        coords = level_coords[0].float()
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"Expected [N, 2] coords, got {coords.shape}")
        # Move onto the same device as the model parameters.
        device = next(self.parameters()).device
        self._xcoord = coords.to(device)

    # ------------------------------------------------------------------
    # Forward: GRIFDIR signature -> GAOT call
    # ------------------------------------------------------------------

    def forward(self, inp, t, pos, domain_onehot=None):
        """Args:
            inp: [B, pos_dim + input_dim, N]
            t:   [B]        diffusion timestep / sigma
            pos: [B, pos_dim, N]   (unused here, mesh coords come from buffer)
        Returns:
            [B, input_dim, N]
        """
        if self._xcoord is None:
            raise RuntimeError("Call set_mesh_hierarchy() before forward()")

        B, _, N = inp.shape
        # Extract signal channels (strip pos/conditioning channels)
        signal = inp[:, self.pos_dim:, :].permute(0, 2, 1)  # [B, N, input_dim]

        # Time embedding -> broadcast to every node
        t_emb     = self.time_embed(t)                       # [B, time_dim]
        t_feats   = self.time_proj(t_emb)                    # [B, n_time_feats]
        t_feats_n = t_feats.unsqueeze(1).expand(B, N, -1)    # [B, N, n_time_feats]

        pndata = torch.cat([signal, t_feats_n], dim=-1)      # [B, N, input_dim + n_time_feats]

        # Multidomain: append per-node-broadcast domain one-hot to features.
        if self.n_domains > 0:
            if domain_onehot is None:
                # Trained with domain conditioning but called without it —
                # fall back to a zero one-hot so the layer doesn't crash.
                domain_onehot = inp.new_zeros(B, self.n_domains)
            d_feats_n = domain_onehot.unsqueeze(1).expand(B, N, -1)  # [B, N, n_domains]
            pndata = torch.cat([pndata, d_feats_n], dim=-1)          # [B, N, +n_domains]

        # GAOT forward: latent_tokens_coord [H*W, 2], xcoord [N, 2], pndata [B, N, C_in]
        out = self.gaot(
            latent_tokens_coord=self._latent_tokens_coord,
            xcoord=self._xcoord,
            pndata=pndata,
        )                                                    # [B, N, input_dim]

        return out.permute(0, 2, 1)                          # [B, input_dim, N]
