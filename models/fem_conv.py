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

import math

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
    vs physical-proximity mismatch.

    Two implementations, same result:
      N <= 4096 — dense [N, N] pairwise distances (original behaviour).
      N >  4096 — scipy cKDTree ball query: never materializes the N x N
                  matrix, so large meshes fit in memory. O(N log N + E).
                  Edges come out in the dense path's order.

    Args:
        coords: [N, d]  float32 — node positions (d = 2 or 3)
        n_nodes: int
        radius:  float  — include all j with ||coords_i - coords_j|| < radius

    Returns:
        edge_index: [2, E']  no self-loops
        edge_attr:  [E', d+1]  (rel..., dist)
    """
    c = coords.float()                                 # [N, d]

    if n_nodes > 4096:
        from scipy.spatial import cKDTree
        import numpy as _np
        c_np = c.detach().cpu().numpy()
        pairs = cKDTree(c_np).query_pairs(float(radius), output_type='ndarray')  # [P, 2], i<j
        if len(pairs) == 0:
            return (torch.zeros(2, 0, dtype=torch.long, device=coords.device),
                    torch.zeros(0, c.shape[1] + 1, device=coords.device))
        src = _np.concatenate([pairs[:, 0], pairs[:, 1]])
        dst = _np.concatenate([pairs[:, 1], pairs[:, 0]])
        order = _np.lexsort((dst, src))   # the dense path's (src, dst) order: same float sums
        ei = torch.from_numpy(_np.stack([src[order], dst[order]])).long().to(coords.device)
        rel = c[ei[1]] - c[ei[0]]                      # c[dst] - c[src], matches dense
        d = rel.norm(dim=-1, keepdim=True)
        return ei, torch.cat([rel, d], dim=1)

    diff = c.unsqueeze(0) - c.unsqueeze(1)             # [N, N, d]
    dist = diff.norm(dim=-1)                           # [N, N]

    mask = (dist < radius) & (dist > 0)               # exclude self-loops
    src, dst = mask.nonzero(as_tuple=True)
    ei  = torch.stack([src, dst], dim=0)              # [2, E']
    rel = diff[src, dst]                              # [E', d]
    d   = dist[src, dst].unsqueeze(-1)                # [E', 1]
    ea  = torch.cat([rel, d], dim=1)                  # [E', d+1]
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
    P0 piecewise-constant basis on an R^d cell grid over the virtual patch
    [-patch_width/2, patch_width/2]^d, where d = rel.shape[1] (2D or 3D).

    The cell containing each source-relative offset receives weight 1; all
    other cells 0. Edges outside the patch receive all-zero rows. For d=2 this
    reproduces the original R×R behaviour exactly (linear index ix + iy*R).

    Args:
        rel:              [E, d]  relative offsets in physical coordinates
        patch_width:      physical width of the virtual patch
        patch_resolution: R — number of cells per dimension

    Returns:
        [E, R^d]  one-hot indicator per cell; zero for edges outside the patch
    """
    R = patch_resolution
    E, d = rel.shape
    n_geo = R ** d

    u = rel / patch_width + 0.5                              # [E, d] in [0,1]^d
    inside = ((u >= 0.0) & (u <= 1.0)).all(dim=-1)           # [E]

    cell = (u * R).floor().long().clamp(0, R - 1)            # [E, d]
    stride = torch.tensor([R ** k for k in range(d)], device=rel.device)
    cell_idx = (cell * stride).sum(-1)                       # [E]

    basis = rel.new_zeros(E, n_geo)
    if inside.any():
        basis[inside] = basis[inside].scatter(
            1, cell_idx[inside].unsqueeze(1), 1.0
        )
    return basis                                             # [E, R^d]


def _eval_p1_basis(rel: torch.Tensor, patch_width: float,
                   patch_resolution: int) -> torch.Tensor:
    """
    P1 multilinear nodal basis on an R^d node grid over the virtual patch
    [-patch_width/2, patch_width/2]^d, where d = rel.shape[1]:
        d=2 — bilinear, 4 surrounding nodes per edge (original behaviour)
        d=3 — trilinear, 8 surrounding nodes per edge (volumetric meshes)

    Weights are the 2^d multilinear interpolation coefficients (sum to 1 inside
    the patch); edges outside the patch receive all-zero rows.

    Args:
        rel:              [E, d]  relative offsets in physical coordinates
        patch_width:      physical width of the virtual patch
        patch_resolution: R — number of nodes per dimension

    Returns:
        [E, R^d]  multilinear weights; at most 2^d nonzeros per row
    """
    R = patch_resolution
    E, d = rel.shape
    n_geo = R ** d

    u = rel / patch_width + 0.5                              # [E, d] in [0,1]^d
    inside = ((u >= 0.0) & (u <= 1.0)).all(dim=-1)           # [E]

    g = u * (R - 1)                                          # [E, d] grid coords
    i0 = g.floor().long().clamp(0, R - 2)                    # [E, d] lower corner
    f = (g - i0.float()).clamp(0.0, 1.0)                     # [E, d] fractional

    stride = torch.tensor([R ** k for k in range(d)], device=rel.device)
    basis = rel.new_zeros(E, n_geo)
    if inside.any():
        ins = inside
        idx_c, wts_c = [], []
        for corner in range(2 ** d):
            bits = [(corner >> k) & 1 for k in range(d)]
            offs = torch.tensor(bits, device=rel.device)
            idx_c.append(((i0[ins] + offs) * stride).sum(-1))       # [E_in]
            w = torch.ones_like(f[ins, 0])
            for k in range(d):
                w = w * (f[ins, k] if bits[k] else (1.0 - f[ins, k]))
            wts_c.append(w)
        idx = torch.stack(idx_c, dim=1)                             # [E_in, 2^d]
        wts = torch.stack(wts_c, dim=1)                             # [E_in, 2^d]
        basis[ins] = basis[ins].scatter_add(1, idx, wts)
    return basis                                             # [E, R^d]

def _eval_q2_basis(rel: torch.Tensor, patch_width: float,
                   patch_resolution: int) -> torch.Tensor:
    """
    Q2 (tensor-product quadratic Lagrange) nodal basis on the same R^d node grid
    as _eval_p1_basis. Consecutive node triples form 2-cell elements, so R must be
    odd (R = 3 is a single element); 3^d nodes are active per edge (9 in 2D). The
    weights sum to 1, interpolate at the nodes and reproduce quadratics exactly;
    unlike the P1 hats they go negative between nodes.

    Returns:
        [E, R^d]  weights; zero rows for edges outside the patch
    """
    R = patch_resolution
    assert R >= 3 and R % 2 == 1, f"Q2 needs an odd patch_resolution >= 3, got {R}"
    E, d = rel.shape
    u = rel / patch_width + 0.5                              # [E, d] in [0,1]^d
    inside = ((u >= 0.0) & (u <= 1.0)).all(dim=-1)
    g = u * (R - 1)                                          # grid coords in [0, R-1]
    m = torch.div(g, 2, rounding_mode='floor').long().clamp(0, (R - 3) // 2)
    s = g - 2.0 * m.to(g.dtype)                              # local coord in [0, 2]
    L = torch.stack([(s - 1.0) * (s - 2.0) / 2.0,            # [E, d, 3] Lagrange
                     s * (2.0 - s),                          # weights at local
                     s * (s - 1.0) / 2.0], dim=-1)           # nodes 0, 1, 2
    stride = torch.tensor([R ** k for k in range(d)], device=rel.device)
    basis = rel.new_zeros(E, R ** d)
    if inside.any():
        ins = inside
        idx_c, wts_c = [], []
        for c in range(3 ** d):
            offs = [(c // 3 ** k) % 3 for k in range(d)]
            idx_c.append(((2 * m[ins] + torch.tensor(offs, device=rel.device)) * stride).sum(-1))
            w = torch.ones_like(s[ins, 0])
            for k in range(d):
                w = w * L[ins, k, offs[k]]
            wts_c.append(w)
        basis[ins] = basis[ins].scatter_add(1, torch.stack(idx_c, 1), torch.stack(wts_c, 1))
    return basis


_BASIS = {'p0': _eval_p0_basis, 'p1': _eval_p1_basis, 'q2': _eval_q2_basis}


# ---------------------------------------------------------------------------
# Cut-cell quadrature
# ---------------------------------------------------------------------------

def _subtriangle_barycentric(n: int) -> torch.Tensor:
    """[n^2, 3] barycentric coordinates of the centroids of the n^2 congruent
    sub-triangles of a triangle split n times along each edge: equal-area samples."""
    pts = []
    for i in range(n):
        for j in range(n - i):
            pts.append((i + 1 / 3, j + 1 / 3))
            if i + j < n - 1:
                pts.append((i + 2 / 3, j + 2 / 3))
    ab = torch.tensor(pts, dtype=torch.float64) / n
    return torch.cat([ab, 1 - ab.sum(1, keepdim=True)], dim=1)


def _compute_cut_cell_edges(xy, cells, radius: float, patch_width: float,
                            basis_type: str, patch_resolution: int,
                            n_sub: int = 16, chunk: int = 64, nodes: str = 'cells') -> tuple:
    """
    Cut-cell quadrature edges for a cell-centred triangle mesh (node i = triangle i);
    nodes='vertices' gives the vertex-centred version, _compute_vertex_cut_cell_edges.

    With features constant per triangle, the ball integral of the conv is

        ∫_{B(x_i, r)} k(x_i - y) h(y) dy = Σ_j h_j |T_j ∩ B_i| k̄_ij,

    k̄_ij the mean of k over the piece T_j ∩ B_i. The kernel is W φ(· / patch_width),
    so k̄_ij = W φ̄_ij with φ̄_ij the patch basis averaged over the piece. Pieces are
    measured with n_sub^2 equal-area samples per triangle. Every triangle meeting the
    ball gets an edge, the node's own included, so a triangle entering the ball does
    so with weight ~0 instead of its whole area.

    Returns:
        edge_index: [2, E]  row 0 source triangle j, row 1 target node i
        edge_attr:  [E, 2 + 1 + n_geo + 1]  (x_i - piece centroid, its norm, φ̄_ij,
                    |T_j ∩ B_i|); the area stays last, where the lumped layer reads it
    """
    if nodes == 'vertices':
        return _compute_vertex_cut_cell_edges(xy, cells, radius, patch_width, basis_type,
                                              patch_resolution, n_sub=n_sub, chunk=chunk)
    from scipy.spatial import cKDTree
    xy = torch.as_tensor(xy, dtype=torch.float64)[:, :2]
    cells = torch.as_tensor(cells, dtype=torch.long)
    tri = xy[cells]                                                   # [T, 3, 2]
    cen = tri.mean(1)                                                 # [T, 2] node positions
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).abs()    # [T]
    samples = torch.einsum('sk,tkd->tsd', _subtriangle_barycentric(n_sub), tri)   # [T, S, 2]
    n_samp = samples.shape[1]
    reach = radius + (tri - cen[:, None]).norm(dim=-1).max().item()
    near = cKDTree(cen.numpy()).query_ball_point(cen.numpy(), reach)
    basis_fn = _BASIS[basis_type]

    srcs, dsts, attrs = [], [], []
    for start in range(0, len(cen), chunk):
        rows = range(start, min(start + chunk, len(cen)))
        dst = torch.cat([torch.full((len(near[i]),), i, dtype=torch.long) for i in rows])
        src = torch.cat([torch.as_tensor(near[i], dtype=torch.long) for i in rows])
        off = cen[dst][:, None] - samples[src]                         # [P, S, 2]  x_i - y
        inside = off.norm(dim=-1) < radius
        cnt = inside.sum(1)
        keep = cnt > 0
        src, dst, off, inside, cnt = src[keep], dst[keep], off[keep], inside[keep], cnt[keep]
        w = area[src] * cnt / n_samp                                   # |T_j ∩ B_i|
        rel = (off * inside[..., None]).sum(1) / cnt[:, None]          # x_i - piece centroid
        phi = basis_fn(off.reshape(-1, 2).float(), patch_width, patch_resolution)
        phi = (phi.reshape(len(src), n_samp, -1) * inside[..., None]).sum(1) / cnt[:, None]
        attrs.append(torch.cat([rel.float(), rel.norm(dim=-1, keepdim=True).float(),
                                phi.float(), w[:, None].float()], dim=1))
        srcs.append(src)
        dsts.append(dst)
    return torch.stack([torch.cat(srcs), torch.cat(dsts)]), torch.cat(attrs)


def _compute_vertex_cut_cell_edges(xy, cells, radius: float, patch_width: float,
                                   basis_type: str, patch_resolution: int,
                                   n_sub: int = 16, chunk: int = 64) -> tuple:
    """
    Cut-cell quadrature edges for a vertex-centred triangle mesh (node i = vertex i, e.g.
    the pinball mesh).

    With features linear on each triangle, h(y) = Σ_j h_j φ_j(y) with φ_j the hat of
    vertex j (1 at j, 0 at its neighbours), the ball integral of the conv is

        ∫_{B(x_i, r)} k(x_i - y) h(y) dy = Σ_j h_j w_ij k̄_ij,   w_ij = ∫_{B_i} φ_j,

    k̄_ij the φ_j-weighted mean of k over B_i, = W φ̄_ij with φ̄_ij the patch basis averaged
    the same way. Every vertex whose hat meets the ball gets an edge, its own included;
    the weights sum to |B_i ∩ Ω| and reproduce the centroid of B_i ∩ Ω, so constant and
    linear fields are integrated exactly (up to the n_sub^2 samples per triangle).

    Returns edge_index / edge_attr in the layout of _compute_cut_cell_edges (row 0 source
    vertex j, row 1 target vertex i), the piece centroid being the φ_j-weighted centroid.
    """
    from scipy.spatial import cKDTree
    xy = torch.as_tensor(xy, dtype=torch.float64)[:, :2]
    cells = torch.as_tensor(cells, dtype=torch.long)
    n_v = len(xy)
    tri = xy[cells]                                                   # [T, 3, 2]
    cen = tri.mean(1)
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    area = 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).abs()    # [T]
    bc = _subtriangle_barycentric(n_sub)                              # [S, 3]: hat values at the samples
    samples = torch.einsum('sk,tkd->tsd', bc, tri)                    # [T, S, 2]
    n_samp = samples.shape[1]
    reach = radius + (tri - cen[:, None]).norm(dim=-1).max().item()
    near = cKDTree(cen.numpy()).query_ball_point(xy.numpy(), reach)   # triangles near each vertex
    basis_fn = _BASIS[basis_type]

    srcs, dsts, attrs = [], [], []
    for start in range(0, n_v, chunk):
        rows = range(start, min(start + chunk, n_v))
        dst = torch.cat([torch.full((len(near[i]),), i, dtype=torch.long) for i in rows])
        t_idx = torch.cat([torch.as_tensor(near[i], dtype=torch.long) for i in rows])
        off = xy[dst][:, None] - samples[t_idx]                       # [P, S, 2]  x_i - y
        inside = off.norm(dim=-1) < radius
        keep = inside.any(1)
        dst, t_idx, off, inside = dst[keep], t_idx[keep], off[keep], inside[keep]
        wq = inside[..., None] * bc * (area[t_idx] / n_samp)[:, None, None]   # [P, S, 3]
        phi = basis_fn(off.reshape(-1, 2).float(), patch_width, patch_resolution).double()
        phi = torch.einsum('psk,psg->pkg', wq, phi.reshape(len(dst), n_samp, -1))   # [P, 3, G]
        mom = torch.einsum('psk,psd->pkd', wq, off)                   # [P, 3, 2]
        # sum over the triangles each (target, source vertex) pair shares
        key, inv = torch.unique((dst[:, None] * n_v + cells[t_idx]).reshape(-1), return_inverse=True)
        w = torch.zeros(len(key), dtype=torch.float64).index_add_(0, inv, wq.sum(1).reshape(-1))
        mom = torch.zeros(len(key), 2, dtype=torch.float64).index_add_(0, inv, mom.reshape(-1, 2))
        phi = torch.zeros(len(key), phi.shape[-1], dtype=torch.float64).index_add_(
            0, inv, phi.reshape(-1, phi.shape[-1]))
        ok = w > 0
        key, w, rel, phi = key[ok], w[ok], mom[ok] / w[ok, None], phi[ok] / w[ok, None]
        attrs.append(torch.cat([rel.float(), rel.norm(dim=-1, keepdim=True).float(),
                                phi.float(), w[:, None].float()], dim=1))
        srcs.append(key % n_v)
        dsts.append(key // n_v)
    return torch.stack([torch.cat(srcs), torch.cat(dsts)]), torch.cat(attrs)


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
        cut_cell:         read the basis, precomputed as its average over each
                          cell piece inside the ball, from edge_attr instead of
                          evaluating it at the offset (_compute_cut_cell_edges)
        lumped_norm:      divide the lumped sum by the sum of its weights at each
                          node, the area of the ball inside the domain: the
                          area-weighted mean, which returns a constant field
                          unchanged at walls as in the interior
        wall_signal:      add a learned vector (zero at init) times the share of
                          the ball (radius patch_width/2) outside the domain: a
                          boundary signal kept apart from the field. Exact with
                          cut_cell; with plain lumping the own cell is missing
                          from the sum, so it reads about 0.14 in the interior
        ball_norm:        divide the lumped sum by the full ball's area (radius
                          patch_width/2), one constant per level: zero padding
                          at the scale of a mean. Alternative to lumped_norm
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 0,
                 mixing_type: str = "vector", lowrank_rank: int = 8,
                 fem_basis_type: str = "p1", patch_resolution: int = 5,
                 patch_width=None, k_hops=1, lumped_mass: bool = False,
                 spatial_dim: int = 2, cut_cell: bool = False,
                 lumped_norm: bool = False, wall_signal: bool = False,
                 ball_norm: bool = False):
        # Lumped-mass mode = sum aggregation + per-edge area weight (read from
        # the LAST column of edge_attr by the caller). This implements a
        # density-invariant lumped FE-Galerkin quadrature; otherwise we use
        # plain mean aggregation (uniform-weight Monte Carlo).
        super().__init__(aggr=('add' if lumped_mass else 'mean'))
        assert mixing_type in ("scalar", "vector", "lowrank"), \
            f"mixing_type must be 'scalar', 'vector', or 'lowrank', got '{mixing_type}'"
        assert fem_basis_type in ("p0", "p1", "q2"), \
            f"fem_basis_type must be 'p0', 'p1' or 'q2', got '{fem_basis_type}'"
        assert (isinstance(k_hops, (int, float)) and k_hops >= 1), \
            f"k_hops must be a number >= 1, got {k_hops}"
        self.H = hidden_dim
        self.mixing_type = mixing_type
        self.lowrank_rank = lowrank_rank
        self.fem_basis_type = fem_basis_type
        self.patch_resolution = patch_resolution
        self.k_hops = k_hops
        self.lumped_mass = lumped_mass
        # Cut-cell quadrature: edge_attr carries the patch basis averaged over each
        # cell piece in columns d+1 .. d+n_geo (see _compute_cut_cell_edges).
        self.cut_cell = cut_cell
        assert not lumped_norm or lumped_mass, "lumped_norm needs lumped_mass"
        self.lumped_norm = lumped_norm
        assert not wall_signal or lumped_mass, "wall_signal needs lumped_mass"
        self.wall_signal = wall_signal
        if wall_signal:
            self.wall_gate = nn.Parameter(torch.zeros(hidden_dim))
        assert not ball_norm or lumped_mass, "ball_norm needs lumped_mass"
        assert not (ball_norm and lumped_norm), "lumped_norm and ball_norm are alternatives"
        self.ball_norm = ball_norm
        self.spatial_dim = spatial_dim              # 2 (planar) or 3 (volumetric)
        n_geo = patch_resolution ** spatial_dim     # R^2 bilinear / R^3 trilinear grid
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
                _d = self.spatial_dim
                if edge_attr.shape[-1] >= _d + 1:      # (rel..., dist) layout
                    pw = edge_attr[:, _d].median().item() * 2.0 * self.k_hops
                else:
                    pw = edge_attr[:, :_d].norm(dim=-1).median().item() * 2.0 * self.k_hops
                self.register_buffer('_patch_width',
                                     torch.tensor(max(pw, 1e-6),
                                                  dtype=h.dtype, device=h.device))
            patch_width = float(getattr(self, '_patch_width', 1.0))
        agg = self.propagate(edge_index, x=h, edge_attr=edge_attr,
                             patch_width=patch_width)
        if ((self.lumped_norm or self.ball_norm or self.wall_signal) and edge_attr is not None
                and edge_attr.shape[-1] >= self.spatial_dim + 2):
            # weights are areas (last column); their sum at a node is the area of
            # its ball inside the domain
            wsum = agg.new_zeros(agg.shape[0], 1).index_add_(
                0, edge_index[1], edge_attr[:, -1:].to(agg.dtype))
            r = 0.5 * patch_width
            ball = math.pi * r ** 2 if self.spatial_dim == 2 else 4.0 / 3.0 * math.pi * r ** 3
            if self.lumped_norm:
                agg = agg / wsum.clamp_min(1e-12)
            elif self.ball_norm:
                agg = agg / ball                                  # zero padding at mean scale
            if self.wall_signal:
                outside = (1.0 - wsum / ball).clamp(0.0, 1.0)    # 0 inside, ~0.5 at a wall
                agg = agg + outside * self.wall_gate
        out = self.root(h) + agg
        return h + self.norm(out)

    # ------------------------------------------------------------------
    def message(self, x_j, edge_attr, patch_width):
        E = x_j.shape[0]
        d = self.spatial_dim

        if edge_attr is not None and edge_attr.shape[-1] >= d:
            rel = edge_attr[:, :d]                                      # [E, d]
        else:
            rel = x_j.new_zeros(E, d)

        # Lumped-mass: source-node dual-cell area is in the LAST column of
        # edge_attr (set by MultiscaleGNNScoreNetwork._edge_attr when active).
        # edge_attr layout is (rel[d], dist, alpha) -> width d+2 when active.
        alpha_src = None
        if self.lumped_mass and edge_attr is not None and edge_attr.shape[-1] >= d + 2:
            alpha_src = edge_attr[:, -1:]                               # [E, 1]

        # ── FEM basis on the virtual physical patch ────────────────────
        if self.cut_cell:
            geo = edge_attr[:, d + 1:d + 1 + self.n_geo]                # precomputed piece averages
        elif self.fem_basis_type == "p0":
            geo = _eval_p0_basis(rel, patch_width, self.patch_resolution)
        elif self.fem_basis_type == "q2":
            geo = _eval_q2_basis(rel, patch_width, self.patch_resolution)
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
