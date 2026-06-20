import math
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

from diffusion.embedding import SinusoidalTimeEmbedding, FourierTimeEmbedding
# Baseline conv layers (PNA / GPS / SplineCNN / MoNet / circular / periodic-MLP)
# are not part of this release — only the FEM-conv layer ships.
from models.fem_conv import (
    FEMConvLayer, _compute_khop_edges, _compute_knn_edges
)


def _compute_p1_lumped_areas(xy_vertices, cells_triangles):
    """P1 lumped-mass: dual-cell area per VERTEX.

        alpha_v = (1/3) * sum_{T containing v} |T|

    Standard mass-lumping rule for a P1 finite-element scheme on a
    triangular mesh: each triangle's area is split equally among its
    three vertices. Used when model nodes are mesh vertices (e.g. pinball,
    where mesh_pos = Pinball_mesh_coords.pt = vertex coords).

    Args:
        xy_vertices:     [V, 2]  vertex coordinates
        cells_triangles: [T, 3]  triangle vertex indices into xy_vertices

    Returns:
        vertex_areas:    [V]     dual-cell area per vertex
    """
    if not isinstance(xy_vertices, torch.Tensor):
        xy_vertices = torch.from_numpy(xy_vertices).float()
    xy_vertices = xy_vertices.float()
    if not isinstance(cells_triangles, torch.Tensor):
        cells_triangles = torch.from_numpy(cells_triangles).long()
    cells_triangles = cells_triangles.long()

    v0 = xy_vertices[cells_triangles[:, 0]]
    v1 = xy_vertices[cells_triangles[:, 1]]
    v2 = xy_vertices[cells_triangles[:, 2]]
    tri_areas = 0.5 * torch.abs(
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) -
        (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    )                                                # [T]
    third = tri_areas / 3.0
    n = int(xy_vertices.shape[0])
    vertex_areas = torch.zeros(n, dtype=tri_areas.dtype)
    vertex_areas.scatter_add_(0, cells_triangles[:, 0], third)
    vertex_areas.scatter_add_(0, cells_triangles[:, 1], third)
    vertex_areas.scatter_add_(0, cells_triangles[:, 2], third)
    return vertex_areas.clamp_min(1e-10)


def _compute_triangle_node_areas(xy_vertices, cells_triangles):
    """DG-0 lumped-mass: per-triangle area, in the order of `cells_triangles`.

    Used when model nodes are TRIANGLES (cell-centred): for every hierarchy
    level, `n_nodes_list[lvl] == cells_triangles.shape[0]` and
    `centers[lvl]` are the triangle centroids. The natural lumped-mass
    weight for a triangle node is the triangle's own area.

    Args:
        xy_vertices:     [V, 2]  vertex coordinates (the underlying mesh
                                 vertices, NOT the model nodes / centers)
        cells_triangles: [T, 3]  triangle vertex indices into xy_vertices

    Returns:
        triangle_areas:  [T]     area per triangle, ordered to match
                                 cells_triangles row order = the model's
                                 node order.
    """
    if not isinstance(xy_vertices, torch.Tensor):
        xy_vertices = torch.from_numpy(xy_vertices).float()
    xy_vertices = xy_vertices.float()
    if not isinstance(cells_triangles, torch.Tensor):
        cells_triangles = torch.from_numpy(cells_triangles).long()
    cells_triangles = cells_triangles.long()

    v0 = xy_vertices[cells_triangles[:, 0]]
    v1 = xy_vertices[cells_triangles[:, 1]]
    v2 = xy_vertices[cells_triangles[:, 2]]
    triangle_areas = 0.5 * torch.abs(
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1]) -
        (v1[:, 1] - v0[:, 1]) * (v2[:, 0] - v0[:, 0])
    )                                                # [T]
    return triangle_areas.clamp_min(1e-10)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class FiLMTimeConditioner(nn.Module):
    """FiLM timestep conditioning: scale + shift that survive LayerNorm.

    Zero-initialized so conditioning starts as identity (h unchanged).
    """

    def __init__(self, time_embed_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        nn.init.constant_(self.proj[-1].weight, 0)
        nn.init.constant_(self.proj[-1].bias, 0)

    def forward(self, t_emb, h):
        """h * (1 + scale) + shift.  t_emb: [B, D], h: [B, N, H]."""
        scale, shift = self.proj(t_emb).chunk(2, dim=-1)
        return h * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class ResolutionFiLM(nn.Module):
    """FiLM conditioning on mesh resolution via log(mean cell area).

    Allows shared GNN weights to adapt to different mesh scales.
    Input:  log_area scalar (per-level), h [B, N, H]
    Output: h * (1 + scale) + shift

    Zero-initialized so conditioning starts as identity.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        nn.init.constant_(self.proj[-1].weight, 0)
        nn.init.constant_(self.proj[-1].bias, 0)

    def forward(self, log_area, h):
        """log_area: scalar tensor, h: [B, N, H]."""
        params = self.proj(log_area.view(1, 1))          # [1, 2H]
        scale, shift = params.chunk(2, dim=-1)            # [1, H] each
        return h * (1 + scale.unsqueeze(0)) + shift.unsqueeze(0)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class DiTBlock(nn.Module):
    def __init__(self, hidden_dim, n_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim, elementwise_affine=False, eps=1e-6)
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden), nn.GELU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, 6 * hidden_dim, bias=True),
        )
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)
        shift_msa, scale_msa, gate_msa = [v.unsqueeze(1) for v in (shift_msa, scale_msa, gate_msa)]
        shift_mlp, scale_mlp, gate_mlp = [v.unsqueeze(1) for v in (shift_mlp, scale_mlp, gate_mlp)]

        x_norm = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa * attn_out

        x_norm = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = x + gate_mlp * self.mlp(x_norm)
        return x


class LatentTransformerProcessor(nn.Module):
    def __init__(self, hidden_dim, n_blocks=4, n_heads=4, mlp_ratio=4.0,
                 time_embed_dim=64, coarse_coords=None):
        super().__init__()
        self.time_proj = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.pos_enc_proj = None
        if coarse_coords is not None:
            self.register_buffer('coarse_coords', coarse_coords.float())
            self.pos_enc_proj = nn.Sequential(
                nn.Linear(coarse_coords.shape[-1], hidden_dim), nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, n_heads, mlp_ratio) for _ in range(n_blocks)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, t_emb):
        c = self.time_proj(t_emb)
        if self.pos_enc_proj is not None:
            x = x + self.pos_enc_proj(self.coarse_coords).unsqueeze(0)
        for block in self.blocks:
            x = block(x, c)
        return self.final_norm(x)


class LearnedPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.message_net = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_net = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h_fine, pool_edges, n_coarse, device):
        B, n_fine, H = h_fine.shape
        fine_idx = pool_edges[0].to(device)
        coarse_idx = pool_edges[1].to(device)
        h_flat = h_fine.reshape(-1, H)

        fine_off = torch.arange(B, device=device) * n_fine
        coarse_off = torch.arange(B, device=device) * n_coarse
        fi = (fine_idx.unsqueeze(0) + fine_off.unsqueeze(1)).reshape(-1)
        ci = (coarse_idx.unsqueeze(0) + coarse_off.unsqueeze(1)).reshape(-1)

        h_messages = self.message_net(h_flat[fi])
        h_coarse = torch.zeros(B * n_coarse, H, device=device)
        h_coarse.scatter_reduce_(0, ci.unsqueeze(1).expand(-1, H), h_messages, reduce="mean", include_self=False)
        return self.update_net(h_coarse).reshape(B, n_coarse, H)


class LearnedUnpool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.message_net = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.combine_net = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim), nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h_coarse, h_fine_skip, unpool_map, device):
        B, N_fine, H = h_fine_skip.shape
        N_coarse = h_coarse.shape[1]
        umap = unpool_map.to(device)

        h_flat = h_coarse.reshape(-1, H)
        h_messages = self.message_net(h_flat)
        off = torch.arange(B, device=device) * N_coarse
        idx = (umap.unsqueeze(0) + off.unsqueeze(1)).reshape(-1)
        h_bc = h_messages[idx].reshape(B, N_fine, H)

        return self.combine_net(torch.cat([h_bc, h_fine_skip], dim=-1))


class LearnedResUnpool(nn.Module):
    """Residual unpooling: broadcast coarse features + additive correction from skip."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.message_net = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.correction_net = nn.Sequential(
            nn.LayerNorm(2 * hidden_dim), nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h_coarse, h_fine_skip, unpool_map, device):
        B, N_fine, H = h_fine_skip.shape
        N_coarse = h_coarse.shape[1]
        umap = unpool_map.to(device)

        h_flat = self.message_net(h_coarse.reshape(-1, H))
        off = torch.arange(B, device=device) * N_coarse
        idx = (umap.unsqueeze(0) + off.unsqueeze(1)).reshape(-1)
        h_bc = h_flat[idx].reshape(B, N_fine, H)

        return h_bc + self.correction_net(torch.cat([h_bc, h_fine_skip], dim=-1))


class SimpleMP(MessagePassing):
    def __init__(self, hidden_dim, edge_dim=0):
        super().__init__(aggr="mean")
        self.edge_dim = edge_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + edge_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, edge_index, edge_attr=None):
        return self.propagate(edge_index, h=h, edge_attr=edge_attr)

    def message(self, h_i, h_j, edge_attr):
        parts = [h_i, h_j - h_i]
        if edge_attr is not None:
            parts.append(edge_attr)
        return self.message_mlp(torch.cat(parts, dim=-1))

    def update(self, agg, h):
        return self.update_mlp(torch.cat([h, agg], dim=-1))


class MPBlock(nn.Module):
    def __init__(self, hidden_dim, num_layers, edge_dim=0, layer_type='simple_mp',
                 n_gps_heads=4, avg_deg=6.0, mixing_type='vector', fem_basis_type='p1',
                 fem_k_hops=2, knn_radius_mult=3.0, fem_lumped_mass=False):
        super().__init__()
        self.layer_type = layer_type
        if layer_type in ('fem_conv', 'knn_conv'):
            # For knn_conv: patch_width = 2 * k_hops * h, so set k_hops = knn_radius_mult
            # so the patch covers all nodes within radius = knn_radius_mult * h
            _khops = fem_k_hops if layer_type == 'fem_conv' else knn_radius_mult
            self.layers = nn.ModuleList([
                FEMConvLayer(hidden_dim, edge_dim=edge_dim, mixing_type=mixing_type,
                             fem_basis_type=fem_basis_type, k_hops=_khops,
                             lumped_mass=fem_lumped_mass)
                for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                SimpleMP(hidden_dim, edge_dim) for _ in range(num_layers)
            ])

    def forward(self, h, edge_index, edge_attr=None, B=1, N=None,
                edges_preexpanded=False, patch_width=None):
        alpha = 1.0 / len(self.layers)
        for layer in self.layers:
            if self.layer_type in ('fem_conv', 'knn_conv'):
                h = layer(h, edge_index, edge_attr=edge_attr, B=B, N=N,
                          edges_preexpanded=edges_preexpanded,
                          patch_width_override=patch_width)
            else:  # simple_mp
                h = h + alpha * layer(h, edge_index, edge_attr)
        return h


class MultiscaleGNNScoreNetwork(nn.Module):
    def __init__(self, input_dim=1, pos_dim=2, hidden_dim=64, time_dim=64,
                 n_gnn_layers_per_level=2, n_levels=3, time_embedding='fourier',
                 use_latent_transformer=False, n_transformer_blocks=4,
                 n_transformer_heads=4, transformer_pos_enc=True,
                 pooling_type='false', use_pos_reinject=False, use_edge_geom=False,
                 layer_type='simple_mp', lap_pe_dim=0, n_gps_heads=4,
                 mixing_type='vector', fem_basis_type='p1', fem_k_hops=2,
                 fem_use_radius=False, fem_radius_mult=3.0,
                 fem_lumped_mass=False,
                 knn_radius_mult=3.0,
                 film_time_cond=False,
                 res_invariant=False,
                 use_domain_heads=False,
                 n_latent=100,
                 n_domains=8,
                 num_layers=None, conv_type=None, heads=None, dropout=0.0,
                 learned_pool=None):
        super().__init__()
        if learned_pool is not None:
            pooling_type = 'learned_pool' if learned_pool else 'false'
        if pooling_type not in ('false', 'learned_pool', 'learned_res'):
            raise ValueError(f"pooling_type must be 'false', 'learned_pool', or 'learned_res', got '{pooling_type}'")
        self.input_dim = input_dim
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim
        self.time_dim = time_dim
        self.n_levels = n_levels
        self.use_latent_transformer = use_latent_transformer
        self.pooling_type = pooling_type
        self.use_pos_reinject = use_pos_reinject
        self.use_edge_geom = use_edge_geom
        self.layer_type = layer_type
        self.lap_pe_dim = lap_pe_dim
        self.fem_k_hops = fem_k_hops
        self.fem_use_radius = fem_use_radius
        self.fem_radius_mult = fem_radius_mult
        self.fem_lumped_mass = fem_lumped_mass
        self.res_invariant = res_invariant
        # Lumped-mass mode appends source-node dual-cell area as an extra
        # edge_attr column, so edge_dim grows by 1 when active.
        _edge_dim = (3 if use_edge_geom else 0) + (1 if fem_lumped_mass else 0)

        if time_embedding == 'sinusoidal':
            self.time_embed = SinusoidalTimeEmbedding(time_dim)
        else:
            self.time_embed = FourierTimeEmbedding(time_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + pos_dim + time_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, input_dim),
        )

        _block_kwargs = dict(edge_dim=_edge_dim, layer_type=layer_type,
                             n_gps_heads=n_gps_heads,
                             mixing_type=mixing_type,
                             fem_basis_type=fem_basis_type,
                             fem_k_hops=fem_k_hops,
                             knn_radius_mult=knn_radius_mult,
                             fem_lumped_mass=fem_lumped_mass)
        if res_invariant:
            # Shared weights across all levels + per-level FiLM on resolution
            self.shared_down_gnn = MPBlock(hidden_dim, n_gnn_layers_per_level, **_block_kwargs)
            self.shared_up_gnn = MPBlock(hidden_dim, n_gnn_layers_per_level, **_block_kwargs)
            self.res_film_down = ResolutionFiLM(hidden_dim)
            self.res_film_up = ResolutionFiLM(hidden_dim)
            self.down_gnns = None  # use shared_down_gnn in forward
            self.up_gnns = None    # use shared_up_gnn in forward
        else:
            self.shared_down_gnn = None
            self.shared_up_gnn = None
            self.res_film_down = None
            self.res_film_up = None
            self.down_gnns = nn.ModuleList([
                MPBlock(hidden_dim, n_gnn_layers_per_level, **_block_kwargs)
                for _ in range(n_levels)
            ])
            self.up_gnns = nn.ModuleList([
                MPBlock(hidden_dim, n_gnn_layers_per_level, **_block_kwargs)
                for _ in range(n_levels - 1)
            ])

        self._lap_pe = None   # retained as a no-op (GPS/LapPE removed)

        # FiLM timestep conditioning at every V-cycle level
        self.film_pre = None
        self.film_solve = None
        self.film_post = None
        if film_time_cond:
            self.film_pre = nn.ModuleList([
                FiLMTimeConditioner(time_dim, hidden_dim)
                for _ in range(n_levels)
            ])
            self.film_solve = FiLMTimeConditioner(time_dim, hidden_dim)
            self.film_post = nn.ModuleList([
                FiLMTimeConditioner(time_dim, hidden_dim)
                for _ in range(n_levels - 1)
            ])

        if use_pos_reinject:
            # pos_proj reinjests raw spatial (xy) coords, which are always 2D.
            # pos_dim may be larger (e.g. pinball adds mu/time conditioning channels)
            # but _level_coords only stores xy, so use 2 here.
            self.pos_proj = nn.ModuleList([
                nn.Linear(2, hidden_dim) for _ in range(n_levels)
            ])

        if pooling_type in ('learned_pool', 'learned_res'):
            self.pool_layers = nn.ModuleList([
                LearnedPool(hidden_dim) for _ in range(n_levels - 1)
            ])
            UnpoolCls = LearnedUnpool if pooling_type == 'learned_pool' else LearnedResUnpool
            self.unpool_layers = nn.ModuleList([
                UnpoolCls(hidden_dim) for _ in range(n_levels - 1)
            ])
        else:
            self.pool_mlps = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), Swish(), nn.Linear(hidden_dim, hidden_dim))
                for _ in range(n_levels - 1)
            ])
            self.unpool_mlps = nn.ModuleList([
                nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), Swish(), nn.Linear(hidden_dim, hidden_dim))
                for _ in range(n_levels - 1)
            ])

        self.latent_transformer = None
        self._n_transformer_blocks = n_transformer_blocks
        self._n_transformer_heads = n_transformer_heads
        self._transformer_pos_enc = transformer_pos_enc

        # Domain-specific encoder/decoder heads (multi-domain training)
        self.domain_encoder_head = None
        self.domain_decoder_head = None
        if use_domain_heads:
            from models.domain_heads import DomainEncoderHead, DomainDecoderHead
            self.domain_encoder_head = DomainEncoderHead(
                hidden_dim, n_latent=n_latent, n_domains=n_domains,
                n_heads=n_transformer_heads,
            )
            self.domain_decoder_head = DomainDecoderHead(
                hidden_dim, n_domains=n_domains,
                n_heads=n_transformer_heads,
            )

        self._edge_indices = None
        self._n_nodes_list = None
        self._pool_edges = None
        self._unpool_maps = None
        self._level_coords = None
        self._precomp_edge_indices = None  # precomputed per level for fem_conv/knn_conv
        self._knn_radius_mult = knn_radius_mult
        self._level_log_areas = None  # [n_levels] log(mean cell area) per level

    def _all_mp_blocks(self):
        """Yield all MPBlock instances (shared or per-level)."""
        if self.res_invariant:
            yield self.shared_down_gnn
            yield self.shared_up_gnn
        else:
            for block in self.down_gnns:
                yield block
            for block in self.up_gnns:
                yield block

    def set_mesh(self, edge_index, num_points=None):
        n = num_points if num_points is not None else edge_index.max().item() + 1
        self._edge_indices = [edge_index]
        self._n_nodes_list = [n]

    def set_mesh_hierarchy(self, edge_indices, n_nodes_list, pool_edges, unpool_maps,
                           coarse_coords=None, level_coords=None, lap_pe=None,
                           cells_list=None, xy_list=None):
        assert len(edge_indices) == self.n_levels
        assert len(pool_edges) == self.n_levels - 1
        self._edge_indices = edge_indices
        self._n_nodes_list = n_nodes_list
        self._pool_edges = pool_edges
        self._unpool_maps = unpool_maps
        self._level_coords = level_coords
        self._lap_pe = lap_pe   # list of [N_l, k] tensors, one per level (or None)

        # Lumped-mass dual-cell areas per level (FE quadrature weights).
        # α is mesh geometry (per-triangle / per-vertex area) and MUST match
        # the current mesh. Recompute every set_mesh_hierarchy call —
        # important for multidomain training (mesh swaps per batch) and
        # cross-resolution eval (mesh swaps per spec). Do NOT freeze like
        # the radius buffer (which is a hyperparameter, not geometry).
        self._level_node_areas = None
        if self.fem_lumped_mass:
            if cells_list is None or xy_list is None:
                raise ValueError(
                    "fem_lumped_mass=True requires xy_list (vertex coords) and "
                    "cells_list (triangle vertex indices) per level. These must "
                    "come from the data's own mesh — load_hierarchy_pt populates "
                    "them from the hierarchy .pt or, for square grids, from "
                    "ConductivityDataset(nx, ny).get_mesh_info()."
                )
            for lvl in range(self.n_levels):
                if cells_list[lvl] is None or xy_list[lvl] is None:
                    raise ValueError(
                        f"fem_lumped_mass=True: xy_list[{lvl}] or cells_list[{lvl}] is None."
                    )
            # Dispatch by node-count match:
            #   n_nodes == n_triangles → DG-0 (conductivity, square): α per cell
            #   n_nodes == n_vertices  → P1   (pinball):              α per vertex
            _device = next(self.parameters()).device
            self._level_node_areas = []
            _stats = []
            for lvl in range(self.n_levels):
                xy_lvl    = xy_list[lvl]
                cells_lvl = cells_list[lvl]
                n_v = (xy_lvl.shape[0] if hasattr(xy_lvl, 'shape')
                       else len(xy_lvl))
                n_t = (cells_lvl.shape[0] if hasattr(cells_lvl, 'shape')
                       else len(cells_lvl))
                n_nodes = n_nodes_list[lvl]
                if n_nodes == n_t:
                    areas = _compute_triangle_node_areas(xy_lvl, cells_lvl)  # DG-0
                elif n_nodes == n_v:
                    areas = _compute_p1_lumped_areas(xy_lvl, cells_lvl)      # P1
                else:
                    raise ValueError(
                        f"fem_lumped_mass=True: level {lvl} n_nodes={n_nodes} "
                        f"matches neither vertex count ({n_v}) nor triangle "
                        f"count ({n_t}). xy/cells out of sync with model nodes."
                    )
                areas = areas.to(_device)
                self._level_node_areas.append(areas)
                _stats.append((float(areas.sum().item()), int(areas.shape[0])))
            print("FEMConv lumped-mass per-level node areas: " +
                  ", ".join(f"L{l}:N={n},Σα={s:.4f}"
                            for l, (s, n) in enumerate(_stats)))

        # Precompute expanded edge indices (fem_conv k-hop or knn_conv)
        if level_coords is not None and self.layer_type in ('fem_conv', 'knn_conv'):
            if self.layer_type == 'fem_conv' and self.fem_use_radius:
                # Per-level radius FEM conv: each level gets its own physical radius
                # scaled to that level's mesh spacing, computed once from the training
                # mesh and reused on eval mesh swaps for resolution invariance.
                _has_radii = (hasattr(self, '_fem_level_radii_buf')
                              and self._fem_level_radii_buf is not None)
                if not _has_radii:
                    level_radii = []
                    for lvl in range(self.n_levels):
                        ei_l = edge_indices[lvl]
                        c_l = level_coords[lvl].float()
                        med_edge_l = (c_l[ei_l[0]] - c_l[ei_l[1]]).norm(dim=-1).median().item()
                        level_radii.append(self.fem_radius_mult * med_edge_l)
                    device = next(self.parameters()).device
                    self.register_buffer('_fem_level_radii_buf',
                                         torch.tensor(level_radii, device=device))
                    print(f"FEMConv per-level radii (mult={self.fem_radius_mult}): "
                          f"{[f'{r:.4f}' for r in level_radii]} ", end="")
                else:
                    level_radii = self._fem_level_radii_buf.tolist()
                    print(f"FEMConv reusing training per-level radii: "
                          f"{[f'{r:.4f}' for r in level_radii]} ", end="")
                self._precomp_edge_indices = []
                for lvl in range(self.n_levels):
                    c = level_coords[lvl].float()
                    n = n_nodes_list[lvl]
                    eik, _ = _compute_knn_edges(c, n, level_radii[lvl])
                    self._precomp_edge_indices.append(eik)
                    print(f"L{lvl}:{eik.shape[1]} ", end="", flush=True)
                print("done")
            elif self.layer_type == 'fem_conv' and self.fem_k_hops >= 1:
                k = self.fem_k_hops
                print(f"FEMConv: precomputing {k}-hop edges per level ", end="")
                self._precomp_edge_indices = []
                for lvl in range(self.n_levels):
                    ei = edge_indices[lvl]
                    n  = n_nodes_list[lvl]
                    c  = level_coords[lvl].float()
                    eik, _ = _compute_khop_edges(ei, c, n, k=k)
                    self._precomp_edge_indices.append(eik)
                    print(f"L{lvl}:{eik.shape[1]} ", end="", flush=True)
                print("done")
                # Fix: set patch_width from training mesh, preserve on eval swaps
                _has_pw = (hasattr(self, '_fem_training_pw_buf')
                           and self._fem_training_pw_buf is not None)
                if not _has_pw:
                    c0 = level_coords[0].float()
                    ei0 = edge_indices[0]
                    med = (c0[ei0[0]] - c0[ei0[1]]).norm(dim=-1).median().item()
                    pw = med * 2.0 * k
                    self.register_buffer('_fem_training_pw_buf',
                                         torch.tensor(pw))
                else:
                    pw = self._fem_training_pw_buf.item()
                for block in self._all_mp_blocks():
                    for layer in block.layers:
                        if isinstance(layer, FEMConvLayer):
                            layer.register_buffer(
                                '_patch_width', torch.tensor(pw))
            elif self.layer_type == 'knn_conv':
                print(f"KNNConv: precomputing radius edges (mult={self._knn_radius_mult}) ", end="")
                self._precomp_edge_indices = []
                for lvl in range(self.n_levels):
                    c = level_coords[lvl].float()
                    n = n_nodes_list[lvl]
                    # radius = knn_radius_mult * median edge length at this level
                    ei_1hop = edge_indices[lvl]
                    s, d = ei_1hop[0], ei_1hop[1]
                    med_edge = (c[s] - c[d]).norm(dim=-1).median().item()
                    radius = self._knn_radius_mult * med_edge
                    eik, _ = _compute_knn_edges(c, n, radius)
                    self._precomp_edge_indices.append(eik)
                    print(f"L{lvl}:{eik.shape[1]} ", end="", flush=True)
                print("done")

        # ────────────────────────────────────────────────────────────────────
        # Cache per-level edge_attr [E_l, n_geo (+1 if lumped_mass)] on device.
        # Without this, `_edge_attr` is rebuilt every layer × every level ×
        # every forward — significant overhead, especially with
        # fem_lumped_mass=True (extra .to(device) on areas + extra torch.cat
        # per call). All inputs are mesh properties, not learned, so we cache
        # once per set_mesh_hierarchy call and reuse on the hot path.
        # ────────────────────────────────────────────────────────────────────
        self._cached_edge_attr = None
        if self.use_edge_geom and level_coords is not None:
            _device = next(self.parameters()).device
            self._cached_edge_attr = []
            for lvl in range(self.n_levels):
                # Pick the edge_index this level's layer will actually use:
                # radius/k-hop FEM and knn_conv use _precomp_edge_indices,
                # everything else uses the dual-graph edge_indices.
                if (self.layer_type in ('fem_conv', 'knn_conv')
                        and getattr(self, '_precomp_edge_indices', None) is not None):
                    ei_l = self._precomp_edge_indices[lvl].to(_device)
                else:
                    ei_l = edge_indices[lvl].to(_device)
                coords_l = level_coords[lvl].float().to(_device)
                dx = coords_l[ei_l[1]] - coords_l[ei_l[0]]      # [E_l, 2]
                dist = dx.norm(dim=-1, keepdim=True)            # [E_l, 1]
                ea = torch.cat([dx, dist], dim=-1)              # [E_l, 3]
                if self.fem_lumped_mass and self._level_node_areas is not None:
                    alpha_src = self._level_node_areas[lvl].to(_device)[ei_l[0]].unsqueeze(-1)
                    ea = torch.cat([ea, alpha_src], dim=-1)     # [E_l, 4]
                self._cached_edge_attr.append(ea)

        # Compute per-level log(mean cell area) for ResolutionFiLM
        if self.res_invariant and level_coords is not None:
            log_areas = []
            for lvl in range(self.n_levels):
                c = level_coords[lvl].float()
                ei = edge_indices[lvl]
                # Proxy for cell area: median_edge_length^2
                edge_len = (c[ei[0]] - c[ei[1]]).norm(dim=-1)
                area_proxy = edge_len.median().item() ** 2
                log_areas.append(math.log(area_proxy + 1e-12))
            device = next(self.parameters()).device
            self._level_log_areas = torch.tensor(log_areas, dtype=torch.float32, device=device)
            print(f"ResolutionFiLM: log_areas = {[f'{a:.3f}' for a in log_areas]}")

        if self.use_latent_transformer and self.latent_transformer is None:
            # Only create once — subsequent set_mesh_hierarchy calls must not
            # destroy learned weights. When domain heads are active the
            # transformer operates on the fixed M-token latent (not mesh
            # nodes), so mesh positional encoding is disabled.
            if self.domain_encoder_head is not None:
                coords = None   # abstract latent space, no mesh PE
            else:
                coords = coarse_coords if (self._transformer_pos_enc and coarse_coords is not None) else None
            self.latent_transformer = LatentTransformerProcessor(
                hidden_dim=self.hidden_dim,
                n_blocks=self._n_transformer_blocks,
                n_heads=self._n_transformer_heads,
                time_embed_dim=self.time_dim,
                coarse_coords=coords,
            )
            device = next(self.parameters()).device
            self.latent_transformer = self.latent_transformer.to(device)

    @staticmethod
    def _batch_ei(edge_index, n_nodes, B, device):
        if B == 1:
            return edge_index.to(device)
        offsets = torch.arange(B, device=device) * n_nodes
        ei = edge_index.to(device).unsqueeze(0).expand(B, -1, -1) + offsets.view(B, 1, 1)
        return ei.permute(1, 0, 2).reshape(2, -1)

    def _pool(self, h_fine, pool_edges, pool_mlp, n_coarse, device):
        B, N_fine, H = h_fine.shape
        fine_idx = pool_edges[0].to(device)
        coarse_idx = pool_edges[1].to(device)
        h_flat = h_fine.reshape(-1, H)

        fine_off = torch.arange(B, device=device) * N_fine
        coarse_off = torch.arange(B, device=device) * n_coarse
        fi = (fine_idx.unsqueeze(0) + fine_off.unsqueeze(1)).reshape(-1)
        ci = (coarse_idx.unsqueeze(0) + coarse_off.unsqueeze(1)).reshape(-1)

        h_coarse = torch.zeros(B * n_coarse, H, device=device)
        h_coarse.scatter_reduce_(0, ci.unsqueeze(1).expand(-1, H), h_flat[fi], reduce="mean", include_self=False)
        return pool_mlp(h_coarse).reshape(B, n_coarse, H)

    def _unpool(self, h_coarse, h_skip, unpool_map, unpool_mlp, device):
        B, N_fine, H = h_skip.shape
        N_coarse = h_coarse.shape[1]
        umap = unpool_map.to(device)

        h_flat = h_coarse.reshape(-1, H)
        off = torch.arange(B, device=device) * N_coarse
        idx = (umap.unsqueeze(0) + off.unsqueeze(1)).reshape(-1)
        h_bc = h_flat[idx].reshape(B, N_fine, H)

        return unpool_mlp(torch.cat([h_bc, h_skip], dim=-1))

    def _edge_attr(self, level, B, device, ei_override=None):
        if not self.use_edge_geom or self._level_coords is None:
            return None
        # Hot path: use the per-level cache built in set_mesh_hierarchy.
        # Skip caching only when an `ei_override` is passed (rare —
        # typically when the caller has its own edge subset).
        if (ei_override is None
                and getattr(self, '_cached_edge_attr', None) is not None):
            ea = self._cached_edge_attr[level]
            if ea.device != device:
                ea = ea.to(device)
                self._cached_edge_attr[level] = ea
            return ea.repeat(B, 1) if B > 1 else ea
        # Slow path: rebuild for an override edge index.
        coords = self._level_coords[level].to(device)          # [N_l, 2]
        ei = ei_override.to(device)
        dx = coords[ei[1]] - coords[ei[0]]                     # [E_l, 2]
        dist = dx.norm(dim=-1, keepdim=True)                   # [E_l, 1]
        ea = torch.cat([dx, dist], dim=-1)                     # [E_l, 3]
        if self.fem_lumped_mass and self._level_node_areas is not None:
            areas = self._level_node_areas[level].to(device)   # [N_l]
            alpha_src = areas[ei[0]].unsqueeze(-1)             # [E_l, 1]
            ea = torch.cat([ea, alpha_src], dim=-1)            # [E_l, 4]
        return ea.repeat(B, 1) if B > 1 else ea

    def _pos_reinject(self, h, level, B):
        if not self.use_pos_reinject or self._level_coords is None:
            return h
        coords = self._level_coords[level].to(h.device)  # [N_l, 2]
        pos_feat = self.pos_proj[level](coords)           # [N_l, H]
        return h + pos_feat.unsqueeze(0).expand(B, -1, -1)

    def forward(self, inp, t, pos, domain_onehot=None):
        if self._pool_edges is None or self._unpool_maps is None:
            raise RuntimeError("Call set_mesh_hierarchy() before forward.")

        B, _, N = inp.shape
        device = inp.device

        positions = pos.squeeze(1)
        signal = inp.permute(0, 2, 1)[..., self.pos_dim:]

        t_emb_raw = self.time_embed(t)
        t_emb = t_emb_raw.unsqueeze(1).expand(-1, N, -1)
        h = self.encoder(torch.cat([signal, positions, t_emb], dim=-1))

        use_precomp = (self._precomp_edge_indices is not None
                       and self.layer_type in ('fem_conv', 'knn_conv'))

        skips = []
        for level in range(self.n_levels):
            h = self._pos_reinject(h, level, B)
            n_l = self._n_nodes_list[level]
            if use_precomp:
                ei_base = self._precomp_edge_indices[level]
                ea = self._edge_attr(level, B, device, ei_override=ei_base)
            else:
                ei_base = self._edge_indices[level]
                ea = self._edge_attr(level, B, device)
            ei = self._batch_ei(ei_base, n_l, B, device)
            down_gnn = self.shared_down_gnn if self.res_invariant else self.down_gnns[level]
            pw = (self._fem_level_radii_buf[level].item() * 2.0
                  if (use_precomp and self.fem_use_radius
                      and hasattr(self, '_fem_level_radii_buf')
                      and self._fem_level_radii_buf is not None)
                  else None)
            h = down_gnn(h.reshape(-1, self.hidden_dim), ei, ea,
                         B=B, N=n_l, edges_preexpanded=use_precomp, patch_width=pw)
            h = h.reshape(B, n_l, self.hidden_dim)

            # ResolutionFiLM: modulate shared weights by log(cell area)
            if self.res_invariant and self._level_log_areas is not None:
                h = self.res_film_down(self._level_log_areas[level], h)

            # FiLM: re-inject timestep after down GNN
            if self.film_pre is not None:
                h = self.film_pre[level](t_emb_raw, h)

            if level < self.n_levels - 1:
                skips.append(h)
                if self.pooling_type in ('learned_pool', 'learned_res'):
                    h = self.pool_layers[level](h, self._pool_edges[level],
                                                self._n_nodes_list[level + 1], device)
                else:
                    h = self._pool(h, self._pool_edges[level], self.pool_mlps[level],
                                   self._n_nodes_list[level + 1], device)

        # FiLM: re-inject timestep at coarsest level
        if self.film_solve is not None:
            h = self.film_solve(t_emb_raw, h)

        # Domain heads: coarse → fixed M latent → coarse
        if self.domain_encoder_head is not None and domain_onehot is not None:
            h_coarse_skip = h                                          # [B, N_coarse, H]
            h = self.domain_encoder_head(h, domain_onehot)             # [B, M, H]

        if self.latent_transformer is not None:
            h = self.latent_transformer(h, t_emb_raw)

        if self.domain_decoder_head is not None and domain_onehot is not None:
            h = self.domain_decoder_head(h, h_coarse_skip, domain_onehot)  # [B, N_coarse, H]

        for level in range(self.n_levels - 2, -1, -1):
            if self.pooling_type in ('learned_pool', 'learned_res'):
                h = self.unpool_layers[level](h, skips[level], self._unpool_maps[level], device)
            else:
                h = self._unpool(h, skips[level], self._unpool_maps[level],
                                 self.unpool_mlps[level], device)
            h = self._pos_reinject(h, level, B)
            n_l = self._n_nodes_list[level]
            if use_precomp:
                ei_base = self._precomp_edge_indices[level]
                ea = self._edge_attr(level, B, device, ei_override=ei_base)
            else:
                ei_base = self._edge_indices[level]
                ea = self._edge_attr(level, B, device)
            ei = self._batch_ei(ei_base, n_l, B, device)
            up_gnn = self.shared_up_gnn if self.res_invariant else self.up_gnns[level]
            pw = (self._fem_level_radii_buf[level].item() * 2.0
                  if (use_precomp and self.fem_use_radius
                      and hasattr(self, '_fem_level_radii_buf')
                      and self._fem_level_radii_buf is not None)
                  else None)
            h = up_gnn(h.reshape(-1, self.hidden_dim), ei, ea,
                       B=B, N=n_l, edges_preexpanded=use_precomp, patch_width=pw)
            h = h.reshape(B, n_l, self.hidden_dim)

            # ResolutionFiLM: modulate shared weights by log(cell area)
            if self.res_invariant and self._level_log_areas is not None:
                h = self.res_film_up(self._level_log_areas[level], h)

            # FiLM: re-inject timestep after up GNN
            if self.film_post is not None:
                h = self.film_post[level](t_emb_raw, h)

        return self.decoder(h).permute(0, 2, 1)
