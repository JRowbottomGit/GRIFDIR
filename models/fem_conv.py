"""
FEMConvLayer — local FEM-style filter stack over a virtual physical patch.

  Conceptual design:
    This layer is the closest analogue of a free CNN filter in the
    infinite-dimensional / irregular-mesh setting.

    A virtual filter patch is defined in PHYSICAL / GEOMETRIC coordinates,
    centered at each destination node.  Domain mesh nodes (graph neighbors) are
    evaluated inside that virtual patch at their relative physical offsets.

    The graph edges provide CANDIDATE neighbors only.
    The virtual patch defines the actual filter coordinate system.
    The physical patch_width is an explicit geometric parameter — NOT implicitly
    defined by neighbor count.

    In a multiscale hierarchy, patch_width should be set per level, e.g.
    proportional to the level spacing h_l.  This makes the filter physically
    scale-aware: on a coarse level a wider patch captures more context; on a
    fine level a narrower patch gives a tighter stencil.

  Basis types:
    p0 — piecewise-constant cell indicators on a patch_resolution × patch_resolution grid
         n_geo = patch_resolution²
         Analogous to a free discrete CNN stencil: each cell has its own coefficient.
         One-hot indicator per edge; zero for edges outside the patch.

    p1 — bilinear nodal basis on a patch_resolution × patch_resolution node grid
         n_geo = patch_resolution²
         Analogous to a continuous FEM-style filter field: bilinear interpolation
         from the 4 surrounding grid nodes.  Only 4 entries nonzero per edge.
         Zero for edges outside the patch.

    This is intended as a closer infinite-dimensional analogue of an
    unconstrained CNN filter than spectral / circular parameterisations, because
    the filter coefficients live directly on a geometric reference patch.

  Channel mixing — three modes (no [E, n_geo, H] intermediate):
    scalar  — scalar edge weight × shared channel map
    vector  — channelwise gate × shared channel map
    lowrank — low-rank kernel: K(edge) = Σ_r a_r(edge) u_r v_r^T
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


# ---------------------------------------------------------------------------
# Spatial edge pre-computation
# ---------------------------------------------------------------------------

def _compute_khop_edges(edge_index: torch.Tensor,
                        coords: torch.Tensor,
                        n_nodes: int,
                        k: int = 2) -> tuple:
    """
    Precompute 1-hop ∪ … ∪ k-hop edge_index and edge_attr via adjacency powers.

    Uses dense matrix powers A^2 … A^k (O(N²) mem, fine for N ≤ 4096).
    Relative positions come directly from ``coords``.

    Args:
        edge_index: [2, E]  int64   — original 1-hop graph
        coords:     [N, 2]  float32 — node (x, y) positions
        n_nodes:    int
        k:          int             — max hop depth (1, 2, or 3)

    Returns:
        edge_index_khop: [2, E']  union of 1 … k-hop, no self-loops
        edge_attr_khop:  [E', 3]  (dx, dy, dist)
    """
    device = edge_index.device
    src, dst = edge_index[0], edge_index[1]

    A = torch.zeros(n_nodes, n_nodes, dtype=torch.bool, device=device)
    A[src, dst] = True

    A_f       = A.float()
    A_power   = A_f.clone()    # A^1
    A_combined = A.clone()     # accumulates union

    for _ in range(k - 1):
        A_power   = A_f @ A_power          # A^(p+1)
        A_combined = A_combined | (A_power > 0)

    A_combined.diagonal().fill_(False)     # no self-loops

    new_src, new_dst = A_combined.nonzero(as_tuple=True)
    new_ei  = torch.stack([new_src, new_dst], dim=0)     # [2, E']
    rel     = coords[new_src] - coords[new_dst]          # [E', 2]
    dist    = rel.norm(dim=-1, keepdim=True)             # [E', 1]
    new_ea  = torch.cat([rel, dist], dim=1)              # [E', 3]
    return new_ei, new_ea


def _compute_knn_edges(coords: torch.Tensor,
                       n_nodes: int,
                       radius: float) -> tuple:
    """
    Build edge_index from ALL node pairs within physical radius ``radius``.

    No graph topology required — purely spatial.  This avoids the hop-count
    vs physical-proximity mismatch.  O(N²) memory; precompute once.

    Args:
        coords: [N, 2]  float32 — node (x, y) positions
        n_nodes: int
        radius:  float  — include all j with ||coords_i - coords_j|| < radius

    Returns:
        edge_index: [2, E']  no self-loops
        edge_attr:  [E', 3]  (dx, dy, dist)
    """
    c    = coords.float()                              # [N, 2]
    diff = c.unsqueeze(0) - c.unsqueeze(1)             # [N, N, 2]
    dist = diff.norm(dim=-1)                           # [N, N]

    mask = (dist < radius) & (dist > 0)               # exclude self-loops
    src, dst = mask.nonzero(as_tuple=True)
    ei  = torch.stack([src, dst], dim=0)              # [2, E']
    rel = diff[src, dst]                              # [E', 2]
    d   = dist[src, dst].unsqueeze(-1)                # [E', 1]
    ea  = torch.cat([rel, d], dim=1)                  # [E', 3]
    return ei, ea


# keep old name as alias for backwards compat
_compute_2hop_edges = lambda ei, c, n: _compute_khop_edges(ei, c, n, k=2)


def _expand_to_2hops(edge_index: torch.Tensor,
                     edge_attr: torch.Tensor,
                     n_nodes: int) -> tuple:
    """
    Fallback: expand 1-hop edges to 2-hop via path decomposition.
    Requires only edge_attr (no node coordinates).  Slower than
    _compute_2hop_edges; use that whenever coordinates are available.
    """
    src = edge_index[0]
    dst = edge_index[1]
    rel = edge_attr[:, :2].float()

    new_src_list, new_dst_list, new_rel_list = [], [], []
    for j in range(n_nodes):
        in_mask  = (dst == j)
        out_mask = (src == j)
        if not (in_mask.any() and out_mask.any()):
            continue
        in_s  = src[in_mask];  in_r  = rel[in_mask]
        out_d = dst[out_mask]; out_r = rel[out_mask]
        n_in, n_out = in_s.shape[0], out_d.shape[0]
        rel_2 = in_r.unsqueeze(1) + out_r.unsqueeze(0)
        new_src_list.append(in_s.unsqueeze(1).expand(n_in, n_out).reshape(-1))
        new_dst_list.append(out_d.unsqueeze(0).expand(n_in, n_out).reshape(-1))
        new_rel_list.append(rel_2.reshape(-1, 2))

    if not new_src_list:
        return edge_index, edge_attr

    new_src  = torch.cat(new_src_list)
    new_dst  = torch.cat(new_dst_list)
    new_rel  = torch.cat(new_rel_list)
    new_dist = new_rel.norm(dim=-1, keepdim=True)
    new_attr = torch.cat([new_rel, new_dist], dim=1)

    orig_attr = edge_attr
    if orig_attr.shape[1] < 3:
        orig_attr = torch.cat([orig_attr, rel.norm(dim=-1, keepdim=True)], dim=1)

    comb_ei   = torch.cat([edge_index, torch.stack([new_src, new_dst], 0)], dim=1)
    comb_attr = torch.cat([orig_attr, new_attr], dim=0)

    keep = comb_ei[0] != comb_ei[1]
    comb_ei   = comb_ei[:, keep]
    comb_attr = comb_attr[keep]

    keys = comb_ei[0].long() * n_nodes + comb_ei[1].long()
    sorted_keys, sort_perm = keys.sort(stable=True)
    uniq_mask = torch.ones(len(sorted_keys), dtype=torch.bool, device=keys.device)
    uniq_mask[1:] = sorted_keys[1:] != sorted_keys[:-1]
    keep_idx = sort_perm[uniq_mask]
    return comb_ei[:, keep_idx], comb_attr[keep_idx]


# ---------------------------------------------------------------------------
# Basis helpers
# ---------------------------------------------------------------------------

def _eval_p0_basis(rel: torch.Tensor, patch_width: float,
                   patch_resolution: int) -> torch.Tensor:
    """
    P0 piecewise-constant basis on a patch_resolution × patch_resolution cell grid.

    The virtual patch covers [-patch_width/2, patch_width/2]² in physical space.
    It is divided into R×R cells.  For each edge the cell that contains the
    source-relative offset receives weight 1; all other cells receive weight 0.
    Edges whose offset falls outside the patch receive all-zero basis vectors.

    Args:
        rel:              [E, 2]  relative offsets in physical coordinates
        patch_width:      physical width of the virtual patch
        patch_resolution: R — number of cells per dimension

    Returns:
        [E, R²]  one-hot indicator per cell; zero for edges outside the patch
    """
    R = patch_resolution
    E = rel.shape[0]
    n_geo = R * R

    # Map [-patch_width/2, patch_width/2]² → [0, 1]²
    u = rel / patch_width + 0.5                              # [E, 2]

    # Explicit inside-patch mask (zero outside)
    inside = ((u >= 0.0) & (u <= 1.0)).all(dim=-1)          # [E]

    # Cell index in [0, R-1] for each dimension (clamp for safety on boundary)
    ix = (u[:, 0] * R).floor().long().clamp(0, R - 1)       # [E]
    iy = (u[:, 1] * R).floor().long().clamp(0, R - 1)       # [E]
    cell_idx = ix + iy * R                                   # [E]  linear index

    basis = rel.new_zeros(E, n_geo)
    if inside.any():
        basis[inside] = basis[inside].scatter(
            1, cell_idx[inside].unsqueeze(1), 1.0
        )
    return basis                                             # [E, R²]


def _eval_p1_basis(rel: torch.Tensor, patch_width: float,
                   patch_resolution: int) -> torch.Tensor:
    """
    P1 bilinear nodal basis on a patch_resolution × patch_resolution node grid.

    The virtual patch covers [-patch_width/2, patch_width/2]² in physical space.
    R×R nodes are placed at uniform positions across the patch.  For each edge
    the bilinear interpolation weights from the 4 surrounding nodes are returned.
    Only 4 entries per row are nonzero.  Edges outside the patch receive zeros.

    Args:
        rel:              [E, 2]  relative offsets in physical coordinates
        patch_width:      physical width of the virtual patch
        patch_resolution: R — number of nodes per dimension

    Returns:
        [E, R²]  bilinear weights; at most 4 nonzeros per row; zero outside patch
    """
    R = patch_resolution
    E = rel.shape[0]
    n_geo = R * R

    # Map [-patch_width/2, patch_width/2]² → [0, 1]²
    u = rel / patch_width + 0.5                              # [E, 2]

    # Explicit inside-patch mask (zero outside)
    inside = ((u >= 0.0) & (u <= 1.0)).all(dim=-1)          # [E]

    # Position in grid coordinates [0, R-1]
    g = u * (R - 1)                                          # [E, 2]

    # Bottom-left node index; clamp so ix+1 ≤ R-1
    ix = g[:, 0].floor().long().clamp(0, R - 2)             # [E]
    iy = g[:, 1].floor().long().clamp(0, R - 2)             # [E]

    # Fractional position within the cell
    fx = (g[:, 0] - ix.float()).clamp(0.0, 1.0)             # [E]
    fy = (g[:, 1] - iy.float()).clamp(0.0, 1.0)             # [E]

    # Linear indices of the 4 surrounding nodes
    n00 = ix       + iy       * R                            # (ix,   iy  )
    n10 = (ix + 1) + iy       * R                           # (ix+1, iy  )
    n01 = ix       + (iy + 1) * R                           # (ix,   iy+1)
    n11 = (ix + 1) + (iy + 1) * R                           # (ix+1, iy+1)

    # Bilinear weights (sum to 1 for points inside the patch)
    w00 = (1.0 - fx) * (1.0 - fy)
    w10 = fx         * (1.0 - fy)
    w01 = (1.0 - fx) * fy
    w11 = fx         * fy

    basis = rel.new_zeros(E, n_geo)
    if inside.any():
        ins = inside
        idx = torch.stack([n00[ins], n10[ins], n01[ins], n11[ins]], dim=1)  # [E_in, 4]
        wts = torch.stack([w00[ins], w10[ins], w01[ins], w11[ins]], dim=1)  # [E_in, 4]
        basis[ins] = basis[ins].scatter_add(1, idx, wts)
    return basis                                             # [E, R²]


# ---------------------------------------------------------------------------
# Layer
# ---------------------------------------------------------------------------

class FEMConvLayer(MessagePassing):
    """
    Local FEM-style filter over a virtual physical patch, with configurable
    channel mixing.

    The spatial filter is evaluated on a virtual physical patch of width
    patch_width centered at each destination node.  Domain mesh nodes (graph
    neighbors) are evaluated inside that virtual patch at their relative physical
    offsets.  Graph edges provide CANDIDATE neighbors only; the virtual patch
    defines the actual filter coordinate system.

    patch_width is a first-class geometric parameter:
      - set explicitly for a fixed physical scale
      - or lazily inferred from median edge length on first forward pass
      - intended to be set per hierarchy level in multiscale models (∝ h_l)

    Basis types:
      p0 — piecewise-constant cell indicators  (free discrete CNN stencil)
      p1 — bilinear nodal interpolation        (continuous FEM filter field)

    Mixing modes:
      scalar  — scalar edge weight × shared channel map
      vector  — channelwise gate × shared channel map
      lowrank — low-rank kernel K(edge) = Σ_r a_r(edge) u_r v_r^T

    Args:
        hidden_dim:       H — node feature width
        edge_dim:         edge feature count (≥2 for dx, dy)
        mixing_type:      "scalar" | "vector" | "lowrank"  (default "vector")
        lowrank_rank:     R — rank for lowrank mixing  (default 8)
        fem_basis_type:   "p0" | "p1"  (default "p1")
        patch_resolution: number of cells/nodes per dimension  (default 5)
        patch_width:      physical width of the virtual patch; if None, lazily
                          inferred as 2*k_hops*median_edge on first forward pass
        k_hops:           1 or 2 — expand graph to k-hop neighbours before
                          message passing.  Use k_hops=2 to include 2nd-ring
                          neighbours (~9 nodes, analogous to a 3×3 CNN kernel).
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 0,
                 mixing_type: str = "vector", lowrank_rank: int = 8,
                 fem_basis_type: str = "p1", patch_resolution: int = 5,
                 patch_width=None, k_hops=1, lumped_mass: bool = False):
        # Lumped-mass mode = sum aggregation + per-edge area weight (read from
        # the LAST column of edge_attr by the caller). This implements a
        # density-invariant lumped FE-Galerkin quadrature; otherwise we use
        # plain mean aggregation (uniform-weight Monte Carlo).
        super().__init__(aggr=('add' if lumped_mass else 'mean'))
        assert mixing_type in ("scalar", "vector", "lowrank"), \
            f"mixing_type must be 'scalar', 'vector', or 'lowrank', got '{mixing_type}'"
        assert fem_basis_type in ("p0", "p1"), \
            f"fem_basis_type must be 'p0' or 'p1', got '{fem_basis_type}'"
        assert (isinstance(k_hops, (int, float)) and k_hops >= 1), \
            f"k_hops must be a number >= 1, got {k_hops}"
        self.H = hidden_dim
        self.mixing_type = mixing_type
        self.lowrank_rank = lowrank_rank
        self.fem_basis_type = fem_basis_type
        self.patch_resolution = patch_resolution
        self.k_hops = k_hops
        self.lumped_mass = lumped_mass
        n_geo = patch_resolution ** 2
        self.n_geo = n_geo

        # patch_width: explicit physical scale of the virtual filter patch
        if patch_width is not None:
            self.register_buffer('_patch_width', torch.tensor(float(patch_width)))
        # else: lazily set from median edge length on first forward pass

        # Shared channel map applied to source features before modulation
        self.msg_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Mixing-mode-specific parameters
        if mixing_type == "scalar":
            self.geo_to_scalar = nn.Linear(n_geo, 1, bias=False)
        elif mixing_type == "vector":
            self.geo_to_gate = nn.Linear(n_geo, hidden_dim, bias=False)
        elif mixing_type == "lowrank":
            R = lowrank_rank
            self.geo_to_coeff = nn.Linear(n_geo, R, bias=False)
            self.lowrank_U = nn.Parameter(torch.empty(R, hidden_dim))
            self.lowrank_V = nn.Parameter(torch.empty(R, hidden_dim))
            nn.init.xavier_uniform_(self.lowrank_U)
            nn.init.xavier_uniform_(self.lowrank_V)

        self.root = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    # ------------------------------------------------------------------
    def forward(self, h, edge_index, edge_attr=None, B=1, N=None,
                edges_preexpanded=False, patch_width_override=None):
        # NOTE: when k_hops>=2 the caller (MultiscaleGNNScoreNetwork) is expected
        # to supply pre-expanded k-hop edges (computed once in set_mesh_hierarchy)
        # and set edges_preexpanded=True.
        # For standalone / test use with B=1 and 1-hop edges, fall back to
        # _expand_to_2hops and cache the result.
        if (self.k_hops == 2 and edge_attr is not None and B == 1
                and not edges_preexpanded):
            if not hasattr(self, '_2hop_ei'):
                ei2, ea2 = _expand_to_2hops(edge_index, edge_attr, h.shape[0])
                self.register_buffer('_2hop_ei', ei2)
                self.register_buffer('_2hop_ea', ea2)
            edge_index = self._2hop_ei
            edge_attr  = self._2hop_ea

        # patch_width priority: override (per-level from caller) > stored buffer > lazy inference
        if patch_width_override is not None:
            patch_width = float(patch_width_override)
        elif hasattr(self, '_patch_width'):
            patch_width = float(self._patch_width)
        else:
            # Lazily infer from median edge length (fallback for standalone use)
            if edge_attr is not None:
                if edge_attr.shape[-1] >= 3:
                    pw = edge_attr[:, 2].median().item() * 2.0 * self.k_hops
                else:
                    pw = edge_attr[:, :2].norm(dim=-1).median().item() * 2.0 * self.k_hops
                self.register_buffer('_patch_width',
                                     torch.tensor(max(pw, 1e-6),
                                                  dtype=h.dtype, device=h.device))
            patch_width = float(getattr(self, '_patch_width', 1.0))
        agg = self.propagate(edge_index, x=h, edge_attr=edge_attr,
                             patch_width=patch_width)
        out = self.root(h) + agg
        return h + self.norm(out)

    # ------------------------------------------------------------------
    def message(self, x_j, edge_attr, patch_width):
        E = x_j.shape[0]

        if edge_attr is not None and edge_attr.shape[-1] >= 2:
            rel = edge_attr[:, :2]                                      # [E, 2]
        else:
            rel = x_j.new_zeros(E, 2)

        # Lumped-mass: source-node dual-cell area is in the LAST column of
        # edge_attr (set by MultiscaleGNNScoreNetwork._edge_attr when active).
        # Multiplied into every message — combined with aggr='add' this gives
        # a density-invariant lumped FE-Galerkin quadrature.
        alpha_src = None
        if self.lumped_mass and edge_attr is not None and edge_attr.shape[-1] >= 4:
            alpha_src = edge_attr[:, -1:]                               # [E, 1]

        # ── FEM basis on the virtual physical patch ────────────────────
        if self.fem_basis_type == "p0":
            geo = _eval_p0_basis(rel, patch_width, self.patch_resolution)
        else:  # "p1"
            geo = _eval_p1_basis(rel, patch_width, self.patch_resolution)
        # geo: [E, n_geo]  (zero rows for edges outside the patch)

        # ── Channel mixing — no [E, n_geo, H] intermediate ────────────
        msg_shared = self.msg_linear(x_j)                               # [E, H]

        if self.mixing_type == "scalar":
            s_gate = self.geo_to_scalar(geo)                            # [E, 1]
            msg = s_gate * msg_shared
        elif self.mixing_type == "vector":
            g = self.geo_to_gate(geo)                                   # [E, H]
            msg = g * msg_shared
        elif self.mixing_type == "lowrank":
            coeff_geo = self.geo_to_coeff(geo)                          # [E, R]
            proj = x_j @ self.lowrank_V.t()                             # [E, R]
            coeff = coeff_geo * proj                                    # [E, R]
            msg = coeff @ self.lowrank_U                                # [E, H]
        else:
            raise ValueError(f"Unknown mixing_type: {self.mixing_type}")

        if alpha_src is not None:
            msg = msg * alpha_src                                       # [E, H]
        return msg
