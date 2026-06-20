"""
Model factory and mesh utilities for GNN-based diffusion models.
"""

import os
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict

from models.gnn import MeshGNNScoreNetwork
from models.mp_pde import MPPDEScoreNetwork
from models.multiscale import MultiscaleGNNScoreNetwork
from models.cnn import CNNScoreNetwork
# FNO import is lazy (inside get_model) to avoid requiring tensorly/neuralop
# for non-FNO runs.


def get_model(cfg, edge_index, num_points, device):
    """
    Create a GNN score model based on configuration.
    
    Args:
        cfg: Configuration object with model settings
        edge_index: Mesh connectivity tensor (2, num_edges)
        num_points: Number of mesh nodes/cells
        device: Device to place the model on
        
    Returns:
        Initialized model on the specified device
    """
    # pos_dim: 2 for (x,y) coords; pinball adds mu + phys_time as conditioning
    _domain = getattr(cfg.data, 'domain', 'square')
    _n_cond = getattr(cfg.data, 'n_cond_channels', 0)  # set by train.py for pinball
    pos_dim = 2 + _n_cond

    if cfg.model.conv_type in ('gcn', 'gat'):
        model = MeshGNNScoreNetwork(
            input_dim=1, pos_dim=pos_dim,
            hidden_dim=cfg.model.hidden_dim, time_dim=cfg.model.time_dim,
            num_layers=cfg.model.num_layers, conv_type=cfg.model.conv_type,
            heads=cfg.model.heads, dropout=cfg.model.dropout,
            time_embedding=cfg.model.time_embedding,
        ).to(device)
        model.set_mesh(edge_index, num_points=num_points)
    elif cfg.model.conv_type == 'mp_pde':
        model = MPPDEScoreNetwork(
            input_dim=1, pos_dim=pos_dim,
            hidden_dim=cfg.model.hidden_dim, time_dim=cfg.model.time_dim,
            num_layers=cfg.model.num_layers,
        ).to(device)
        model.set_mesh(edge_index, num_points=num_points)
    elif cfg.model.conv_type == 'multiscale':
        _domain = getattr(cfg.data, 'domain', 'square')
        if _domain == 'pinball':
            ms_base_dir = getattr(cfg.data, 'ms_base_dir', '')
            ms_finest_dir = getattr(cfg.data, 'ms_finest_dir', '') or ms_base_dir
            ms_res_dirs = getattr(cfg.data, 'ms_res_dirs', '')
            if isinstance(ms_res_dirs, str):
                ms_res_dirs = [d.strip() for d in ms_res_dirs.split(',') if d.strip()]
            ms_all_res_dirs = getattr(cfg.data, 'ms_all_res_dirs', '')
            if isinstance(ms_all_res_dirs, str):
                ms_all_res_dirs = [d.strip() for d in ms_all_res_dirs.split(',') if d.strip()]
            if not ms_all_res_dirs:
                ms_all_res_dirs = ms_res_dirs  # fallback: no skipped levels
            hier = load_pinball_hierarchy(ms_finest_dir, ms_base_dir,
                                          ms_res_dirs, ms_all_res_dirs)
        else:
            _script_dir = os.path.dirname(os.path.abspath(__file__))
            _hier_names = {
                'square': 'mesh_hierarchy.pt',
                'l_shape': 'mesh_hierarchy_l_shape.pt',
                'circle': 'mesh_hierarchy_circle.pt',
                'circle_with_hole': 'mesh_hierarchy_circle_with_hole.pt',
                'e_shape': 'mesh_hierarchy_e_shape.pt',
                'plus': 'mesh_hierarchy_plus.pt',
                'square_with_hole': 'mesh_hierarchy_square_with_hole.pt',
                'x_shape': 'mesh_hierarchy_x_shape.pt',
                'sst': 'mesh_hierarchy_sst.pt',
            }
            # Allow explicit hierarchy path override via cfg.data.hierarchy_path
            _hier_override = getattr(cfg.data, 'hierarchy_path', None)
            if _hier_override and os.path.exists(_hier_override):
                _hier_path = _hier_override
            elif _domain in _hier_names:
                _hier_path = os.path.join(_script_dir, 'meshes', _hier_names[_domain])
            else:
                raise ValueError(f"No hierarchy for domain '{_domain}'. Known: {list(_hier_names)}")
            hier = torch.load(_hier_path, weights_only=False)
        lap_pe_dim      = getattr(cfg.model, 'lap_pe_dim', 0)
        layer_type      = getattr(cfg.model, 'layer_type', 'simple_mp')
        n_gps_heads     = getattr(cfg.model, 'n_gps_heads', 4)
        mixing_type     = getattr(cfg.model, 'mixing_type', 'vector')
        fem_basis_type  = getattr(cfg.model, 'fem_basis_type', 'p1')
        fem_k_hops      = getattr(cfg.model, 'fem_k_hops', 2)
        fem_use_radius  = getattr(cfg.model, 'fem_use_radius', False)
        fem_radius_mult = getattr(cfg.model, 'fem_radius_mult', 3.0)
        fem_lumped_mass = getattr(cfg.model, 'fem_lumped_mass', False)
        knn_radius_mult = getattr(cfg.model, 'knn_radius_mult', 3.0)

        coarse_coords = hier['centers'][-1] if 'centers' in hier else None
        level_coords  = hier['centers'] if 'centers' in hier else None
        # Hierarchies use either `xy/cells` (singular, multidomain) or
        # `xy_list/cells_list` (list, eval). Square hierarchies store only
        # `centers + resolutions` — repopulate from ConductivityDataset.
        cells_list    = hier.get('cells_list', hier.get('cells', None))
        xy_list       = hier.get('xy_list',    hier.get('xy',    None))
        if (fem_lumped_mass and (xy_list is None or cells_list is None)
                and 'resolutions' in hier):
            # Only fem_lumped_mass needs xy/cells per level (for α weights).
            # Square hierarchies store only centers + resolutions. Repopulate
            # xy/cells from the cached conductivity .pt for that resolution
            # (no dolfinx required — these files are pre-generated and stored
            # with mesh_pos/xy/cells already inside).
            xy_list_built, cells_list_built = [], []
            _data_dir = getattr(cfg.data, 'data_dir', './data')
            for nx, ny in hier['resolutions']:
                # Try a few cached-file naming patterns (varies by sample count)
                _candidates = [
                    f'conductivity_nx{nx}_ny{ny}_n10000.pt',
                    f'conductivity_nx{nx}_ny{ny}_n5000.pt',
                    f'conductivity_nx{nx}_ny{ny}_n1000.pt',
                ]
                _data = None
                for _name in _candidates:
                    _path = os.path.join(_data_dir, _name)
                    if os.path.exists(_path):
                        _data = torch.load(_path, weights_only=False)
                        break
                if _data is None:
                    raise FileNotFoundError(
                        f"No cached conductivity file found for ({nx}, {ny}) under "
                        f"{_data_dir}. Tried {_candidates}. Required for "
                        f"fem_lumped_mass=True on square hierarchies."
                    )
                _xy = _data['xy']
                _cells = _data['cells']
                xy_list_built.append(_xy.float() if isinstance(_xy, torch.Tensor)
                                     else torch.from_numpy(_xy).float())
                cells_list_built.append(_cells.long() if isinstance(_cells, torch.Tensor)
                                        else torch.from_numpy(_cells).long())
            xy_list = xy_list_built
            cells_list = cells_list_built

        # ── FNO baseline (fixed regular grid, discrete) ───────────────────
        if layer_type == 'fno':
            from models.fno import FNOScoreNetwork
            _fno_pos = getattr(cfg.model, 'fno_pos_enc', 'none')
            if _fno_pos == 'none':
                _fno_pos = None
            model = FNOScoreNetwork(
                input_dim=1, pos_dim=pos_dim,
                hidden_dim=cfg.model.hidden_dim,
                time_dim=cfg.model.time_dim,
                n_modes=getattr(cfg.model, 'fno_n_modes', 16),
                n_fno_layers=getattr(cfg.model, 'fno_n_layers', 4),
                grid_h=cfg.data.nx,
                grid_w=cfg.data.ny,
                time_embedding=cfg.model.time_embedding,
                positional_embedding=_fno_pos,
                domain_padding=getattr(cfg.model, 'fno_pad', 0.0),
            ).to(device)
            model.set_mesh_hierarchy(
                edge_indices=hier['edge_indices'],
                n_nodes_list=hier['n_nodes_list'],
                pool_edges=hier['pool_edges'],
                unpool_maps=hier['unpool_maps'],
                coarse_coords=coarse_coords,
                level_coords=level_coords,
            )
            return model

        # ── GAOT baseline (Geometry-Aware Operator Transformer) ──────────
        if layer_type == 'gaot':
            from models.gaot import GAOTScoreNetwork
            _lts = getattr(cfg.model, 'gaot_latent_tokens_size', [32, 32])
            if isinstance(_lts, str):
                _lts = [int(x) for x in _lts.split(',') if x.strip()]
            else:
                _lts = list(_lts)
            model = GAOTScoreNetwork(
                input_dim=1, pos_dim=pos_dim,
                hidden_dim=cfg.model.hidden_dim,
                time_dim=cfg.model.time_dim,
                time_embedding=cfg.model.time_embedding,
                latent_tokens_size=_lts,
                patch_size=getattr(cfg.model, 'gaot_patch_size', 2),
                n_transformer_layers=getattr(cfg.model, 'gaot_n_transformer_layers', 3),
                magno_radius=getattr(cfg.model, 'gaot_magno_radius', 0.05),
                magno_hidden=getattr(cfg.model, 'gaot_magno_hidden', 64),
                magno_mlp_layers=getattr(cfg.model, 'gaot_magno_mlp_layers', 3),
                magno_lifting=getattr(cfg.model, 'gaot_magno_lifting', 64),
                positional_embedding=getattr(cfg.model, 'gaot_positional_embedding', 'absolute'),
                use_geoembed=getattr(cfg.model, 'gaot_use_geoembed', True),
                use_attention=getattr(cfg.model, 'gaot_use_attention', False),
                use_torch_scatter=getattr(cfg.model, 'gaot_use_torch_scatter', False),
            ).to(device)
            model.set_mesh_hierarchy(
                edge_indices=hier['edge_indices'],
                n_nodes_list=hier['n_nodes_list'],
                pool_edges=hier['pool_edges'],
                unpool_maps=hier['unpool_maps'],
                coarse_coords=coarse_coords,
                level_coords=level_coords,
            )
            return model

        # ── CNN/UNet baseline ─────────────────────────────────────────────
        if layer_type == 'cnn':
            model = CNNScoreNetwork(
                input_dim=1, pos_dim=pos_dim,
                hidden_dim=cfg.model.hidden_dim,
                time_dim=cfg.model.time_dim,
                n_levels=len(hier['n_nodes_list']),
                time_embedding=cfg.model.time_embedding,
                grid_h=cfg.data.nx,
                grid_w=cfg.data.ny,
                double_conv=getattr(cfg.model, 'cnn_double_conv', False),
                residual=getattr(cfg.model, 'cnn_residual', False),
                bottleneck_attn=getattr(cfg.model, 'cnn_bottleneck_attn', False),
                strided_down=getattr(cfg.model, 'cnn_strided_down', False),
                dropout=getattr(cfg.model, 'cnn_dropout', 0.0),
            ).to(device)
            model.set_mesh_hierarchy(
                edge_indices=hier['edge_indices'],
                n_nodes_list=hier['n_nodes_list'],
                pool_edges=hier['pool_edges'],
                unpool_maps=hier['unpool_maps'],
                coarse_coords=coarse_coords,
                level_coords=level_coords,
            )
            return model

        # ── GNN-based models ──────────────────────────────────────────────
        film_time_cond = getattr(cfg.model, 'film_time_cond', False)
        res_invariant = getattr(cfg.model, 'res_invariant', False)
        use_domain_heads = getattr(cfg.model, 'use_domain_heads', False)
        n_latent = getattr(cfg.model, 'n_latent', 100)
        n_domains = getattr(cfg.model, 'n_domains', 8)
        model = MultiscaleGNNScoreNetwork(
            input_dim=1, pos_dim=pos_dim,
            hidden_dim=cfg.model.hidden_dim, time_dim=cfg.model.time_dim,
            n_gnn_layers_per_level=cfg.model.num_layers,
            n_levels=len(hier['n_nodes_list']),
            use_latent_transformer=cfg.model.use_latent_transformer,
            n_transformer_blocks=cfg.model.n_transformer_blocks,
            n_transformer_heads=cfg.model.n_transformer_heads,
            pooling_type=cfg.model.pooling_type,
            use_pos_reinject=cfg.model.use_pos_reinject,
            use_edge_geom=cfg.model.use_edge_geom,
            layer_type=layer_type,
            lap_pe_dim=lap_pe_dim,
            n_gps_heads=n_gps_heads,
            mixing_type=mixing_type,
            fem_basis_type=fem_basis_type,
            fem_k_hops=fem_k_hops,
            fem_use_radius=fem_use_radius,
            fem_radius_mult=fem_radius_mult,
            fem_lumped_mass=fem_lumped_mass,
            knn_radius_mult=knn_radius_mult,
            film_time_cond=film_time_cond,
            res_invariant=res_invariant,
            use_domain_heads=use_domain_heads,
            n_latent=n_latent,
            n_domains=n_domains,
        ).to(device)

        # LapPE (GPS) baseline removed in this release; lap_pe stays None.
        lap_pe = None

        model.set_mesh_hierarchy(
            edge_indices=hier['edge_indices'],
            n_nodes_list=hier['n_nodes_list'],
            pool_edges=hier['pool_edges'],
            unpool_maps=hier['unpool_maps'],
            coarse_coords=coarse_coords,
            level_coords=level_coords,
            lap_pe=lap_pe,
            cells_list=cells_list,
            xy_list=xy_list,
        )
    else:
        raise ValueError(f"Unknown conv_type: {cfg.model.conv_type}")
    return model


def _load_coords(res_path):
    """Load mesh coordinates from a resolution directory."""
    coords_path = res_path / 'Pinball_mesh_coords.pt'
    if not coords_path.exists():
        coords_path = res_path / 'mesh_coords.pt'
    coords = torch.load(coords_path, weights_only=False)
    if not isinstance(coords, torch.Tensor):
        coords = torch.from_numpy(np.asarray(coords)).float()
    return coords.float()


def _cells_from_edges(edge_index, n_nodes):
    """
    Recover triangle cell list from a (primal-mesh) edge_index by enumerating
    3-cycles in the undirected adjacency graph. Returns (n_tris, 3) torch
    LongTensor of vertex indices in node-order (i.e. consistent with
    edge_index, which for the Pinball dataset is the same as
    Pinball_mesh_coords.pt order).

    Verified on Pinball orig (7525, 14620 tris), lc_500_kNN, lc_4000_kNN to
    produce exactly the same triangle set as meshio.read(Pinball_mesh.xml)
    after the cKDTree remap to centers-order — without the meshio dep, the
    FEniCS-XML "Unknown entry" warnings, or any vertex permutation step.
    """
    ei = edge_index.numpy() if isinstance(edge_index, torch.Tensor) else np.asarray(edge_index)
    adj = [set() for _ in range(n_nodes)]
    for s, d in ei.T:
        if s != d:
            adj[int(s)].add(int(d))
            adj[int(d)].add(int(s))
    tris = []
    for i in range(n_nodes):
        nbrs_gt_i = sorted(j for j in adj[i] if j > i)
        for ji, j in enumerate(nbrs_gt_i):
            adj_j = adj[j]
            for k in nbrs_gt_i[ji + 1:]:
                if k in adj_j:
                    tris.append((i, j, k))
    return torch.tensor(tris, dtype=torch.long)


def _get_composed_operator(all_dirs, fine_idx, coarse_idx):
    """Get restriction operator from fine_idx to coarse_idx by composing operators.

    When all_dirs[0] is the true original mesh (name does not end in '_kNN')
    and fine_idx == 0, use R_from_orig stored in the target coarse level's
    metadata directly (single load, no composition).

    Otherwise compose R_to_coarser.T step-by-step through intermediate levels.
    R_to_coarser at level i+1 has shape (n_finer_neighbor, n_level_i+1)
    (prolongation); its transpose gives the restriction.
    """
    fine_is_orig = not Path(all_dirs[0]).name.endswith('_kNN')
    if fine_is_orig and fine_idx == 0:
        coarse_path = Path(all_dirs[coarse_idx])
        meta = torch.load(coarse_path / 'projection_metadata.pt', weights_only=False)
        if 'operators' not in meta or 'R_from_orig' not in meta['operators']:
            raise KeyError(f"R_from_orig not found in {coarse_path}/projection_metadata.pt")
        return meta['operators']['R_from_orig']
    else:
        R_composed = None
        for step_idx in range(fine_idx, coarse_idx):
            next_path = Path(all_dirs[step_idx + 1])
            next_meta = torch.load(next_path / 'projection_metadata.pt', weights_only=False)
            if 'operators' not in next_meta or 'R_to_coarser' not in next_meta['operators']:
                raise KeyError(
                    f"R_to_coarser not found in {next_path}/projection_metadata.pt. "
                    f"Regenerate this level's projection metadata with its adjacent finer "
                    f"level present (see generate_multi_res_data.py). "
                    f"Cannot compose operators for level {fine_idx}->{coarse_idx}."
                )
            # R_to_coarser: (n_finer_neighbor, n_this) — prolongation
            # Transpose: (n_this, n_finer_neighbor) — restriction
            R_step = next_meta['operators']['R_to_coarser'].T
            if R_composed is None:
                R_composed = R_step
            else:
                R_composed = R_step @ R_composed
        return R_composed


def load_pinball_hierarchy(data_dir, ms_base_dir, ms_res_dirs, ms_all_res_dirs):
    """
    Load multiscale hierarchy from HMSAE resolution directories.

    Mirrors HMSAE MultiscaleGNN_BSMS._load_inter_level_edges:
      - Level 0->1 (from orig): uses R_from_orig directly.
      - Level i->i+1 (i>0): composes R_to_coarser.T through intermediate dirs
        listed in ms_all_res_dirs.

    Args:
        data_dir:         Finest-level dir (e.g. .../Pinball/orig)
        ms_base_dir:      Parent of coarse dirs (e.g. .../Pinball)
        ms_res_dirs:      List of coarse dir names used as levels
                          (e.g. ['lc_500_kNN', 'lc_2000_kNN'])
        ms_all_res_dirs:  List of ALL coarse dir names for operator composition
                          (e.g. ['lc_250_kNN', 'lc_500_kNN', 'lc_1000_kNN', 'lc_2000_kNN'])

    Returns:
        dict with keys: edge_indices, n_nodes_list, centers, pool_edges, unpool_maps
    """
    from scipy import sparse as sp

    resolution_dirs = [data_dir] + [str(Path(ms_base_dir) / d) for d in ms_res_dirs]
    all_dirs = [data_dir] + [str(Path(ms_base_dir) / d) for d in ms_all_res_dirs]
    n_levels = len(resolution_dirs)

    # Map used dirs to indices in all_dirs
    use_indices = []
    for d in resolution_dirs:
        if d in all_dirs:
            use_indices.append(all_dirs.index(d))
        else:
            raise ValueError(f"resolution_dir '{d}' not found in all_dirs: {all_dirs}")

    # ---- Load coords, edges, and (optionally) per-level mesh for lumped-mass ----
    edge_indices = []
    n_nodes_list = []
    centers = []
    xy_list = []
    cells_list = []

    for res_dir in resolution_dirs:
        res_path = Path(res_dir)
        coords = _load_coords(res_path)
        centers.append(coords)
        n_nodes_list.append(coords.shape[0])

        ei_path = res_path / 'Pinball_edge_index.pt'
        if not ei_path.exists():
            ei_path = res_path / 'edge_index.pt'
        if ei_path.exists():
            ei = torch.load(ei_path, weights_only=True).long()
        else:
            raise FileNotFoundError(f"No edge_index found in {res_path}")
        edge_indices.append(ei)

        # Per-level triangulation: recover triangles as 3-cycles in the
        # already-loaded primal edge_index. Cells come out in centers-order
        # natively (same node ordering as edge_index, which matches
        # Pinball_mesh_coords.pt). No meshio, no file reads, no remap.
        cells_lvl = _cells_from_edges(ei, coords.shape[0])
        xy_lvl = coords.float()
        xy_list.append(xy_lvl)
        cells_list.append(cells_lvl)

    print(f"Pinball hierarchy: {n_levels} levels — "
          + ", ".join(f"{n}" for n in n_nodes_list) + " nodes")
    print(f"  Using levels {use_indices} from {len(all_dirs)} total dirs")

    # ---- Build pool_edges and unpool_maps via composed operators ----
    pool_edges = []
    unpool_maps = []

    for i in range(1, n_levels):
        fine_all_idx = use_indices[i - 1]
        coarse_all_idx = use_indices[i]
        n_fine = n_nodes_list[i - 1]
        n_coarse = n_nodes_list[i]

        n_steps = coarse_all_idx - fine_all_idx
        R = _get_composed_operator(all_dirs, fine_all_idx, coarse_all_idx)
        if not sp.issparse(R):
            R = sp.csr_matrix(R)

        if R.shape != (n_coarse, n_fine):
            raise ValueError(
                f"Composed operator shape mismatch for level {i-1}->{i}: "
                f"expected ({n_coarse}, {n_fine}), got {R.shape}"
            )

        # Convert sparse R to pool_edges [2, E] = [fine_nodes, coarse_nodes]
        R_coo = R.tocoo()
        fine_nodes = torch.from_numpy(R_coo.col.copy()).long()
        coarse_nodes = torch.from_numpy(R_coo.row.copy()).long()
        pool_edge = torch.stack([fine_nodes, coarse_nodes], dim=0)
        pool_edges.append(pool_edge)

        # Build unpool_map: for each fine node, its coarse parent (argmax of R column)
        R_csc = R.tocsc()
        unpool_map = torch.zeros(n_fine, dtype=torch.long)
        for j in range(n_fine):
            col_start = R_csc.indptr[j]
            col_end = R_csc.indptr[j + 1]
            if col_end > col_start:
                rows = R_csc.indices[col_start:col_end]
                vals = R_csc.data[col_start:col_end]
                unpool_map[j] = int(rows[np.argmax(np.abs(vals))])
        unpool_maps.append(unpool_map)

        label = f"composed {n_steps} ops" if n_steps > 1 else "direct"
        print(f"  Level {i-1}->{i}: {pool_edge.shape[1]} pool edges, "
              f"{n_fine}->{n_coarse} nodes ({label})")

    return {
        'edge_indices': edge_indices,
        'n_nodes_list': n_nodes_list,
        'centers': centers,
        'pool_edges': pool_edges,
        'unpool_maps': unpool_maps,
        'xy_list': xy_list,
        'cells_list': cells_list,
    }


def cells_to_dual_edge_index(cells: np.ndarray) -> torch.Tensor:
    """
    Convert mesh cells (triangles) to a dual graph edge_index.
    
    In the dual graph, each cell becomes a node and cells that share
    an edge in the original mesh are connected.
    
    Args:
        cells: Array of shape (num_cells, 3) with vertex indices for each triangle
        
    Returns:
        edge_index: Tensor of shape (2, num_edges) for the dual graph
    """
    edge_to_cells = defaultdict(list)
    
    for cell_idx, cell in enumerate(cells):
        edges = [
            tuple(sorted([cell[0], cell[1]])),
            tuple(sorted([cell[1], cell[2]])),
            tuple(sorted([cell[2], cell[0]])),
        ]
        for edge in edges:
            edge_to_cells[edge].append(cell_idx)
    
    src_list = []
    dst_list = []
    
    for edge, cell_list in edge_to_cells.items():
        if len(cell_list) == 2:
            c1, c2 = cell_list
            src_list.extend([c1, c2])
            dst_list.extend([c2, c1])
    
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    return edge_index


def create_mesh(nx: int, ny: int):
    """
    Create a unit square mesh and return mesh info.
    
    Args:
        nx: Number of cells in x direction
        ny: Number of cells in y direction
        
    Returns:
        mesh_pos: Cell center coordinates (num_cells, 2)
        xy: Vertex coordinates (num_vertices, 3)
        cells: Cell connectivity (num_cells, 3)
        edge_index: Dual graph edge index (2, num_edges)
    """
    from dolfinx import mesh as dolfin_mesh
    from dolfinx.fem import functionspace
    from mpi4py import MPI
    
    comm = MPI.COMM_WORLD
    domain = dolfin_mesh.create_unit_square(comm, nx, ny, dolfin_mesh.CellType.triangle)
    domain.topology.create_connectivity(1, 2)
    
    xy = domain.geometry.x
    cells = domain.geometry.dofmap.reshape((-1, domain.topology.dim + 1))
    
    V = functionspace(domain, ("DG", 0))
    mesh_pos = np.array(V.tabulate_dof_coordinates()[:, :2])
    
    edge_index = cells_to_dual_edge_index(cells)
    
    return mesh_pos, xy, cells, edge_index


# ---------------------------------------------------------------------------
# Multi-domain hierarchy utilities
# ---------------------------------------------------------------------------

_HIERARCHY_FILES = {
    'square':           'mesh_hierarchy.pt',
    'l_shape':          'mesh_hierarchy_l_shape.pt',
    'circle':           'mesh_hierarchy_circle.pt',
    'circle_with_hole': 'mesh_hierarchy_circle_with_hole.pt',
    'e_shape':          'mesh_hierarchy_e_shape.pt',
    'plus':             'mesh_hierarchy_plus.pt',
    'square_with_hole': 'mesh_hierarchy_square_with_hole.pt',
    'x_shape':          'mesh_hierarchy_x_shape.pt',
}


def load_domain_hierarchy(domain: str, multiscale_dir: str | None = None) -> dict:
    """Load a single domain's mesh hierarchy .pt file.

    Returns dict with keys: edge_indices, n_nodes_list, centers,
    pool_edges, unpool_maps.
    """
    if domain not in _HIERARCHY_FILES:
        raise ValueError(f"No hierarchy for domain '{domain}'. Known: {list(_HIERARCHY_FILES)}")
    if multiscale_dir is None:
        multiscale_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'meshes')
    path = os.path.join(multiscale_dir, _HIERARCHY_FILES[domain])
    return torch.load(path, weights_only=False)


def load_all_hierarchies(domains: list[str] | None = None,
                         multiscale_dir: str | None = None) -> dict[str, dict]:
    """Load mesh hierarchies for multiple domains.

    Parameters
    ----------
    domains : list of domain names (default: all 8).
    multiscale_dir : directory containing mesh_hierarchy_*.pt files.

    Returns
    -------
    dict mapping domain name → hierarchy dict.
    """
    if domains is None:
        domains = list(_HIERARCHY_FILES.keys())
    return {d: load_domain_hierarchy(d, multiscale_dir) for d in domains}
