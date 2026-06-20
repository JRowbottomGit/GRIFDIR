"""
Resolution invariance evaluation for trained diffusion models.

Tests a trained model at different mesh resolutions and/or domains without
retraining. Generates unconditional samples and (optionally) DPS reconstructions.

Usage
-----
    # Single experiment via CLI:
    python eval/resolution_invariance.py \
        --run_dir exp/conv=multiscale/abc123 \
        --eval_domain square \
        --eval_resolutions 64,64 32,32 16,16

    # Batch experiments via YAML:
    python eval/resolution_invariance.py --config eval_res_config.yaml

YAML config format
------------------
    output_dir: results/res_invariance
    n_samples: 8
    n_steps: 200
    device: cuda

    experiments:
      - run_dir: exp/conv=multiscale/abc123
        checkpoint: model_ema_latest.pt
        evals:
          - domain: square
            resolutions: [[64,64],[32,32],[16,16]]
          - domain: square
            resolutions: [[128,128],[64,64],[32,32]]
          - domain: l_shape
            maxh_levels: [0.05, 0.1, 0.2]

      - run_dir: exp/conv=multiscale/pinball_run
        checkpoint: model_ema_latest.pt
        evals:
          - domain: pinball
            data_dir: /path/to/Pinball
            ms_base_dir: /path/to/Pinball
            ms_res_dirs: [lc_250_kNN, lc_1000_kNN]
            ms_all_res_dirs: [lc_250_kNN, lc_500_kNN, lc_1000_kNN]
"""

import os
import sys
import copy
import json
import argparse
from pathlib import Path

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

import yaml
from omegaconf import OmegaConf

from config import Config, merge_config, get_device
from model_mesh_utils import get_model, load_pinball_hierarchy
from diffusion.precond import ScoreModel, EDMDenoiser
from diffusion.sde import OU, CosineOU
from diffusion.noise import RBFKernel
from diffusion.noise_accel import RBFKernel as RBFKernelAccel
from eval.sampling import (
    sample_unconditional, sample_ve_heun,
    plot_uncond_samples,
    run_dps_eval, run_dps_eval_ve, plot_cond_eval,
)


# ═══════════════════════════════════════════════════════════════════════════
# Hierarchy builders
# ═══════════════════════════════════════════════════════════════════════════

def build_square_hierarchy(resolutions):
    """Build hierarchy for square domain at given resolutions.

    Parameters
    ----------
    resolutions : list of [nx, ny] pairs, finest first

    Returns
    -------
    hier : dict  {edge_indices, n_nodes_list, centers, pool_edges, unpool_maps,
                  xy_list, cells_list}
    """
    from data_tools.build_hierarchy import build_hierarchy, save_hierarchy
    levels, pool_edges, unpool_maps = build_hierarchy(
        [(r[0], r[1]) for r in resolutions]
    )
    return _levels_to_hier(levels, pool_edges, unpool_maps)


def build_lshape_hierarchy(maxh_levels):
    """Build hierarchy for L-shape domain at given maxh levels.

    Parameters
    ----------
    maxh_levels : list of floats, finest (smallest) first
    """
    from data_tools.build_hierarchy import build_hierarchy_unstructured
    levels, pool_edges, unpool_maps = build_hierarchy_unstructured('l_shape', maxh_levels)
    return _levels_to_hier(levels, pool_edges, unpool_maps)


def build_generic_hierarchy(domain, maxh_levels):
    """Build hierarchy for any gmsh-based domain."""
    from data_tools.build_hierarchy import build_hierarchy_unstructured
    levels, pool_edges, unpool_maps = build_hierarchy_unstructured(domain, maxh_levels)
    return _levels_to_hier(levels, pool_edges, unpool_maps)


def _levels_to_hier(levels, pool_edges, unpool_maps):
    """Convert build_hierarchy output to the dict format expected by model."""
    return {
        'edge_indices': [l['edge_index'] for l in levels],
        'n_nodes_list': [l['n_nodes'] for l in levels],
        'centers': [torch.from_numpy(l['centers']).float() for l in levels],
        'xy_list': [l['xy'] for l in levels],
        'cells_list': [l['cells'] for l in levels],
        'pool_edges': pool_edges,
        'unpool_maps': unpool_maps,
    }


def load_hierarchy_pt(pt_path, n_levels=None, offset=0):
    """Load a pre-generated hierarchy .pt file.

    Some hierarchies (multidomain conductivity, pinball) store xy/cells per
    level directly. Others (square grid: mesh_hierarchy.pt and friends) store
    only `centers` + `resolutions` to keep file size down. For the latter we
    repopulate xy_list/cells_list on the fly using the same code path that
    data_tools/build_hierarchy.create_level uses to produce them at training
    time — namely ConductivityDataset(num_samples=1, nx=nx, ny=ny).get_mesh_info().
    No fresh meshing or random fallback: this returns the *exact* triangulation
    the model was trained against.

    Parameters
    ----------
    pt_path  : path to .pt file saved by build_hierarchy.save_hierarchy
    n_levels : if set, only take n_levels levels (after offset)
    offset   : skip the first `offset` levels (for coarser eval from
               an existing hierarchy, e.g. offset=1 skips the finest level)

    Returns
    -------
    hier : dict matching the format expected by swap_mesh_hierarchy
    """
    data = torch.load(pt_path, map_location='cpu', weights_only=False)
    total = len(data['n_nodes_list'])
    end = min(offset + n_levels, total) if n_levels else total
    sl = slice(offset, end)
    n = end - offset
    xy_data = data.get('xy', data.get('xy_list'))
    cells_data = data.get('cells', data.get('cells_list'))

    xy_list = xy_data[sl] if xy_data is not None else None
    cells_list = cells_data[sl] if cells_data is not None else None

    # Square-grid hierarchies (e.g. mesh_hierarchy.pt) store only centers +
    # resolutions. Repopulate xy/cells per level from the canonical dataset
    # mesh — same code build_hierarchy.create_level used to make them.
    if (xy_list is None or cells_list is None) and 'resolutions' in data:
        from data_utils import ConductivityDataset
        resolutions = data['resolutions'][sl] if isinstance(data['resolutions'], list) else data['resolutions']
        xy_list_built, cells_list_built = [], []
        for nx, ny in resolutions:
            ds = ConductivityDataset(num_samples=1, nx=nx, ny=ny)
            _, xy, cells, _ = ds.get_mesh_info()
            xy_list_built.append(torch.from_numpy(xy).float() if not isinstance(xy, torch.Tensor) else xy.float())
            cells_list_built.append(torch.from_numpy(cells).long() if not isinstance(cells, torch.Tensor) else cells.long())
        xy_list = xy_list_built
        cells_list = cells_list_built

    return {
        'edge_indices': data['edge_indices'][sl],
        'n_nodes_list': data['n_nodes_list'][sl],
        'centers':      data['centers'][sl],
        'xy_list':      xy_list,
        'cells_list':   cells_list,
        'pool_edges':   data['pool_edges'][offset:offset + n - 1],
        'unpool_maps':  data['unpool_maps'][offset:offset + n - 1],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Model loading  (checkpoint → model, with mesh swap)
# ═══════════════════════════════════════════════════════════════════════════

def load_trained_model(run_dir, checkpoint='model_ema_latest.pt', device='cuda'):
    """Load a trained model from its run directory.

    Returns
    -------
    model           : nn.Module  (eval mode, weights loaded)
    cfg             : Config
    train_mesh_info : dict with keys 'xy' and 'cells' (numpy arrays, or None)
                      Ground-truth vertex coords and triangle connectivity from
                      the training dataset — used as a plotting fallback when
                      a hierarchy .pt lacks xy_list/cells_list.
    """
    config_path = os.path.join(run_dir, 'config.yaml')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = OmegaConf.load(config_path)
    cfg = merge_config(cfg)

    # Need original mesh to build model (sets internal buffers)
    domain = getattr(cfg.data, 'domain', 'square')
    train_mesh_info = {'xy': None, 'cells': None}
    if domain == 'multidomain':
        # Mirror train.py:482-510 — direct MultiscaleGNNScoreNetwork construction
        # since get_model() doesn't know 'multidomain'. We seed the model with
        # any one of the training domains' hierarchies; swap_mesh_hierarchy()
        # later reassigns it per eval spec.
        from model_mesh_utils import load_domain_hierarchy
        from models.multiscale import MultiscaleGNNScoreNetwork
        _domains_str = getattr(cfg.data, 'domains', '')
        _domain_list = [d.strip() for d in _domains_str.split(',') if d.strip()]
        if not _domain_list:
            raise ValueError("cfg.data.domains is empty; can't rebuild multidomain model.")
        first_domain = _domain_list[0]
        first_hier = load_domain_hierarchy(first_domain)

        # Determine n_levels from the CHECKPOINT, not from the hierarchy. At
        # training time train.py trims hierarchies to min_levels across all 8
        # domains, so the checkpoint's level count can be smaller than any
        # domain's current hierarchy. down_gnns.N is absent when
        # res_invariant=True (replaced by a single shared_down_gnn), so fall
        # back to pool_layers.N (L-1 entries for L levels, always present).
        _ckpt_path = os.path.join(run_dir, checkpoint)
        _sd_for_levels = torch.load(_ckpt_path, map_location='cpu', weights_only=False)
        _down_idxs = [int(k.split('.')[1]) for k in _sd_for_levels.keys()
                      if k.startswith('down_gnns.')]
        _pool_idxs = [int(k.split('.')[1]) for k in _sd_for_levels.keys()
                      if k.startswith('pool_layers.')]
        if _down_idxs:
            _ckpt_n_levels = 1 + max(_down_idxs)
        elif _pool_idxs:
            _ckpt_n_levels = 2 + max(_pool_idxs)
        elif '_fem_level_radii_buf' in _sd_for_levels:
            _ckpt_n_levels = int(_sd_for_levels['_fem_level_radii_buf'].shape[0])
        else:
            _ckpt_n_levels = len(first_hier['n_nodes_list'])
        # Re-seed via load_hierarchy_pt so the hierarchy is sliced to the checkpoint's
        # level count AND carries xy_list/cells_list — fem_lumped_mass needs them, and the
        # raw load_domain_hierarchy dict stores 'xy'/'cells' (not the *_list keys) which the
        # manual trim above dropped.
        from model_mesh_utils import _HIERARCHY_FILES
        _seed_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'meshes', _HIERARCHY_FILES[first_domain])
        first_hier = load_hierarchy_pt(_seed_path, n_levels=_ckpt_n_levels)
        print(f"  Multidomain seed: {first_domain} at n_levels={_ckpt_n_levels}")

        model = MultiscaleGNNScoreNetwork(
            input_dim=1, pos_dim=2,
            hidden_dim=cfg.model.hidden_dim, time_dim=cfg.model.time_dim,
            n_gnn_layers_per_level=cfg.model.num_layers,
            n_levels=_ckpt_n_levels,
            use_latent_transformer=cfg.model.use_latent_transformer,
            n_transformer_blocks=cfg.model.n_transformer_blocks,
            n_transformer_heads=cfg.model.n_transformer_heads,
            pooling_type=cfg.model.pooling_type,
            use_pos_reinject=cfg.model.use_pos_reinject,
            use_edge_geom=cfg.model.use_edge_geom,
            layer_type=cfg.model.layer_type,
            mixing_type=getattr(cfg.model, 'mixing_type', 'vector'),
            fem_basis_type=getattr(cfg.model, 'fem_basis_type', 'p1'),
            fem_k_hops=getattr(cfg.model, 'fem_k_hops', 2),
            fem_use_radius=getattr(cfg.model, 'fem_use_radius', False),
            fem_radius_mult=getattr(cfg.model, 'fem_radius_mult', 3.0),
            fem_lumped_mass=getattr(cfg.model, 'fem_lumped_mass', False),
            knn_radius_mult=getattr(cfg.model, 'knn_radius_mult', 3.0),
            film_time_cond=getattr(cfg.model, 'film_time_cond', False),
            res_invariant=getattr(cfg.model, 'res_invariant', False),
            use_domain_heads=cfg.model.use_domain_heads,
            n_latent=cfg.model.n_latent,
            n_domains=cfg.model.n_domains,
        ).to(device)
        # Use swap_mesh_hierarchy to register the seed hierarchy (same API as
        # eval-time mesh swaps; sets edge_indices/centers/pool/unpool buffers).
        swap_mesh_hierarchy(model, first_hier, device)

        # Load checkpoint and return early — skip the default get_model path.
        ckpt_path = os.path.join(run_dir, checkpoint)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(state_dict, dict):
            state_dict.pop('_metadata', None)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"Loaded {ckpt_path} ({sum(p.numel() for p in model.parameters()):,} params) — multidomain seed={first_domain}")
        return model, cfg, train_mesh_info

    if domain == 'pinball':
        from data_utils import PinballDataset
        # Pinball data lives under ms_finest_dir / ms_base_dir, not data_dir.
        # data_dir holds conductivity .pt caches and defaults to './data'.
        _pinball_data_dir = (getattr(cfg.data, 'ms_finest_dir', None)
                             or getattr(cfg.data, 'ms_base_dir', None)
                             or cfg.data.data_dir)
        ds = PinballDataset(_pinball_data_dir, num_samples=1)
        mesh_pos, _, edge_index = ds.get_mesh_info()
        num_points = mesh_pos.shape[0]

        # Mirror train.py's runtime computation of n_cond_channels for pinball
        # (mu + phys_time channels prepended to inp via pos_dim). Without this,
        # get_model below rebuilds the first encoder linear with the wrong
        # input width and state_dict load fails with a size mismatch.
        _n_cond = 0
        if getattr(cfg.data, 'enc_use_mu', True):
            _n_cond += ds.n_params
        if getattr(cfg.data, 'enc_use_time', True):
            _n_cond += 1
        cfg.data.n_cond_channels = _n_cond
    else:
        import glob
        candidates = glob.glob(
            os.path.join(cfg.data.data_dir, f'conductivity_{domain}_*.pt')
        )
        if not candidates:
            candidates = glob.glob(
                os.path.join(cfg.data.data_dir, f'conductivity_nx{cfg.data.nx}_ny{cfg.data.ny}_*.pt')
            )
        if not candidates:
            raise FileNotFoundError(f"No dataset in {cfg.data.data_dir}")
        from data_utils import CachedDataset
        ds = CachedDataset(sorted(candidates)[-1])
        mesh_pos, ds_xy, ds_cells, edge_index = ds.get_mesh_info()
        num_points = len(mesh_pos)
        train_mesh_info = {'xy': ds_xy, 'cells': ds_cells}

    edge_index_dev = edge_index.to(device) if isinstance(edge_index, torch.Tensor) else torch.tensor(edge_index).to(device)
    model = get_model(cfg, edge_index_dev, num_points, device)

    # Load checkpoint
    ckpt_path = os.path.join(run_dir, checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    # weights_only=False: FNO checkpoints serialize the GELU activation as a
    # callable, which PyTorch 2.6+ rejects under weights_only=True. We trust
    # our own checkpoints, so explicit opt-in.
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    # FNO checkpoints have a '_metadata' key at the top level that strict
    # load_state_dict rejects as "unexpected". Pop it before load.
    # GAOT checkpoints from before the EMA-shape-mismatch fix have `_xcoord`
    # as a registered buffer; the current code holds it as a plain attribute
    # that gets reset by set_mesh_hierarchy, so it's safe to drop here.
    # Lumped-mass checkpoints from before the α-recompute fix (3b85de5) have
    # `_lumped_node_areas_lvl{l}` registered buffers; the current code stores
    # them as a plain Python list (recomputed per set_mesh_hierarchy call), so
    # those state_dict keys no longer exist in the model — drop them.
    if isinstance(state_dict, dict):
        state_dict.pop('_metadata', None)
        state_dict.pop('_xcoord', None)
        for k in list(state_dict.keys()):
            if k.startswith('_lumped_node_areas_lvl'):
                del state_dict[k]
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded {ckpt_path} ({sum(p.numel() for p in model.parameters()):,} params)")

    return model, cfg, train_mesh_info


def swap_mesh_hierarchy(model, hier, device):
    """Replace the model's mesh hierarchy in-place, preserving trained weights.

    For latent transformer: save weights, rebuild with new coarse_coords,
    reload weights (strict=False to handle buffer shape changes).
    """
    has_transformer = (model.latent_transformer is not None)
    transformer_sd = None
    if has_transformer:
        transformer_sd = copy.deepcopy(model.latent_transformer.state_dict())

    coarse_coords = hier['centers'][-1] if 'centers' in hier else None
    level_coords = hier['centers'] if 'centers' in hier else None

    lap_pe = None

    # Pass xy_list/cells_list when present so lumped-mass models can recompute
    # per-level α buffers on the swapped mesh (and so plotting can build the
    # actual triangulation rather than falling back to scatter).
    extra = {}
    if 'cells_list' in hier and hier['cells_list'] is not None:
        extra['cells_list'] = hier['cells_list']
    if 'xy_list' in hier and hier['xy_list'] is not None:
        extra['xy_list'] = hier['xy_list']
    model.set_mesh_hierarchy(
        edge_indices=hier['edge_indices'],
        n_nodes_list=hier['n_nodes_list'],
        pool_edges=hier['pool_edges'],
        unpool_maps=hier['unpool_maps'],
        coarse_coords=coarse_coords,
        level_coords=level_coords,
        lap_pe=lap_pe,
        **extra,
    )

    # Restore transformer weights. Strip coordinate buffers that changed shape
    # (coarse_coords, lap_pe buffers) — set_mesh_hierarchy already set them.
    if has_transformer and transformer_sd is not None and model.latent_transformer is not None:
        current_sd = model.latent_transformer.state_dict()
        filtered_sd = {k: v for k, v in transformer_sd.items()
                       if k in current_sd and v.shape == current_sd[k].shape}
        model.latent_transformer.load_state_dict(filtered_sd, strict=False)

        # Force the transformer's position-encoding coords to match the NEW
        # coarsest level.  Without this, cross-resolution evals pool h down to
        # len(new_coarse_coords) tokens but the transformer still holds the
        # training-time coarse_coords buffer, causing a shape mismatch in the
        # pos_enc broadcast add inside LatentTransformerProcessor.forward.
        lt = model.latent_transformer
        if (hasattr(lt, 'coarse_coords')
                and lt.coarse_coords is not None
                and coarse_coords is not None):
            _target_dev = lt.coarse_coords.device
            # Re-register the buffer so shape-changed coords stick (assigning
            # with module.buf = t works, but explicit re-register is clearer).
            if 'coarse_coords' in lt._buffers:
                del lt._buffers['coarse_coords']
            lt.register_buffer('coarse_coords', coarse_coords.float().to(_target_dev))

    model.to(device)
    print(f"  Mesh swapped: {hier['n_nodes_list']} nodes across {len(hier['n_nodes_list'])} levels")


# ═══════════════════════════════════════════════════════════════════════════
# Fixed-grid (CNN / FNO) swap + interpolation adapter
# ═══════════════════════════════════════════════════════════════════════════

def _infer_grid_dims(n_nodes, tris_per_pixel):
    """Infer square grid (H, W) from n_nodes and tris_per_pixel."""
    side = int(round(math.sqrt(n_nodes / tris_per_pixel)))
    assert side * side * tris_per_pixel == n_nodes, \
        f"n_nodes={n_nodes} with tris_per_pixel={tris_per_pixel} is not a square grid"
    return side, side


def _cnn_pool_depth(model):
    """Number of downsampling stages in the CNN UNet (each halves grid dims)."""
    return len(model.downs)


def swap_grid(model, hier, device, mode='native', train_grid=None):
    """Swap eval mesh for a fixed-grid model (CNN or FNO).

    Parameters
    ----------
    model : CNNScoreNetwork or FNOScoreNetwork
    hier  : hierarchy dict (same format as for swap_mesh_hierarchy; only the
            finest level is used)
    device : torch device
    mode : 'native' or 'interpolate'
        - 'native' : rewrite model.grid_h/grid_w to match the eval hierarchy's
          finest-level grid, then rebuild the node→pixel map.  For FNO this is
          natural (Fourier modes transfer).  For CNN the new grid must be
          divisible by 2^pool_depth.
        - 'interpolate' : keep grid_h/grid_w at train_grid values, build a node
          map at the eval grid resolution.  Only valid for CNN (FNO does not
          need interpolation).  Caller wraps the model in InterpolatingCNN.
    train_grid : (train_h, train_w) tuple — required for mode='interpolate'.

    Returns
    -------
    eval_grid : (eval_h, eval_w) tuple — the eval finest-level grid dims
    """
    is_fno = hasattr(model, 'fno')
    tris = model.tris_per_pixel
    n_finest = hier['n_nodes_list'][0]
    eval_h, eval_w = _infer_grid_dims(n_finest, tris)

    if mode == 'native':
        # Validate grid compatibility
        if is_fno:
            # FNO: need grid >= 2 * n_modes for non-aliased modes
            n_modes = model.fno.n_modes[0] if hasattr(model.fno, 'n_modes') else None
            if n_modes is not None and min(eval_h, eval_w) < n_modes:
                raise ValueError(f"FNO requires grid >= n_modes={n_modes}, "
                                 f"got {eval_h}x{eval_w}")
        else:
            pool_depth = _cnn_pool_depth(model)
            stride = 2 ** pool_depth
            if eval_h % stride != 0 or eval_w % stride != 0:
                raise ValueError(f"CNN requires grid divisible by {stride} "
                                 f"(pool depth {pool_depth}), got {eval_h}x{eval_w}")
        model.grid_h = eval_h
        model.grid_w = eval_w

    elif mode == 'interpolate':
        if is_fno:
            raise ValueError("mode='interpolate' is not supported for FNO")
        if train_grid is None:
            raise ValueError("mode='interpolate' requires train_grid=(h, w)")
        model.grid_h, model.grid_w = train_grid
        # node map is built at eval resolution below — but _to_image/_from_image
        # use self.grid_h/grid_w (training).  We override these temporarily for
        # the node-map construction so the scatter indices are built in the
        # eval-grid pixel space, then swap back.
        model.grid_h, model.grid_w = eval_h, eval_w
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    model.set_mesh_hierarchy(
        edge_indices=hier['edge_indices'],
        n_nodes_list=hier['n_nodes_list'],
        pool_edges=hier['pool_edges'],
        unpool_maps=hier['unpool_maps'],
        level_coords=hier['centers'],
    )

    if mode == 'interpolate':
        # Node map was built against eval_h × eval_w pixel coordinates.
        # Leave grid_h/grid_w at eval-grid values; InterpolatingCNN handles
        # resizing between eval grid and the UNet's training grid.
        model.grid_h, model.grid_w = eval_h, eval_w

    model.to(device)
    print(f"  Grid swapped: {eval_h}x{eval_w} (mode={mode}, finest={n_finest} nodes)")
    return eval_h, eval_w


class InterpolatingCNN(nn.Module):
    """Adapter wrapping a CNNScoreNetwork to evaluate on a finer/coarser mesh
    by running the UNet at training resolution and bilinear-resizing between.

    Pipeline for each forward call (eval grid = hier's finest-level grid;
    train grid = (train_h, train_w) from model.in_conv training):

        1. scatter nodes → image at (eval_h, eval_w)
        2. F.interpolate → (train_h, train_w)
        3. run UNet (in_conv → downs → bottle → ups → out_conv)
        4. F.interpolate → (eval_h, eval_w)
        5. gather → nodes
    """

    def __init__(self, model, train_grid):
        super().__init__()
        self.model = model
        self.train_h, self.train_w = train_grid
        # Expose these so ScoreModel / callers treating this as the inner model
        # still see the hierarchy-adjusted attributes.
        self.pos_dim = model.pos_dim
        self.input_dim = model.input_dim
        self.tris_per_pixel = model.tris_per_pixel

    @property
    def grid_h(self):
        return self.model.grid_h

    @property
    def grid_w(self):
        return self.model.grid_w

    def forward(self, inp, t, pos, domain_onehot=None):
        m = self.model
        if m._node_to_pixel is None:
            raise RuntimeError("Call swap_grid(..., mode='interpolate') before forward()")

        # 1. scatter nodes → eval-grid image
        signal = inp[:, m.pos_dim:, :].permute(0, 2, 1)  # [B, N, C]
        img_eval = m._to_image(signal)                    # [B, C*T, eval_h, eval_w]

        # 2. resize to training grid
        img_train = F.interpolate(img_eval,
                                  size=(self.train_h, self.train_w),
                                  mode='bilinear', align_corners=False)

        # 3. run UNet stack at training resolution
        t_emb = m.time_embed(t)
        x = m.in_conv(img_train)
        skips = []
        for d in m.downs:
            x, sk = d(x, t_emb)
            skips.append(sk)
        x = m.bottle(x, t_emb)
        if m._bottleneck_attn:
            x = m.bottle2(m.bottle_attn(x), t_emb)
        for u, sk in zip(m.ups, reversed(skips)):
            x = u(x, sk, t_emb)
        x = m.out_conv(x)  # [B, C*T, train_h, train_w]

        # 4. resize back to eval grid
        x_eval = F.interpolate(x,
                               size=(m.grid_h, m.grid_w),
                               mode='bilinear', align_corners=False)

        # 5. gather back to nodes
        return m._from_image(x_eval).permute(0, 2, 1)  # [B, C, N]


# ═══════════════════════════════════════════════════════════════════════════
# Single evaluation run
# ═══════════════════════════════════════════════════════════════════════════

def run_eval(model, cfg, hier, eval_name, output_dir,
             n_samples=8, n_steps=200, device='cuda',
             dps_gt_field=None, n_dps_sensors=10, dps_guidance=1.0,
             train_mesh_info=None, domain_onehot=None):
    """Run unconditional sampling (+ optional DPS) on a swapped mesh.

    Parameters
    ----------
    model      : nn.Module with mesh already swapped
    cfg        : Config from training
    hier       : hierarchy dict (must have centers, xy_list, cells_list)
    eval_name  : string label for this experiment
    output_dir : directory for saving plots and metrics
    dps_gt_field : optional [1, N] GT field for DPS eval
    train_mesh_info : dict {'xy': ndarray, 'cells': ndarray} from the training
                      dataset, used as fallback when hier lacks xy_list/cells_list
    domain_onehot : optional [B, n_domains] tensor passed to model forward for
                    multi-domain runs. Default None (single-domain path).
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {'name': eval_name, 'n_nodes_list': hier['n_nodes_list']}

    # Finest-level mesh info
    mesh_pos = hier['centers'][0].numpy()
    pos = hier['centers'][0].to(device)
    n_nodes = hier['n_nodes_list'][0]

    # Noise sampler on new mesh
    use_accel = getattr(cfg.training, 'use_accel_sampler', True)
    NoiseCls = RBFKernelAccel if use_accel else RBFKernel
    noise_sampler = NoiseCls(
        mesh_points=pos,
        scale=cfg.noise.scale,
        eps=cfg.noise.eps,
        device=device,
    )

    # SDE
    sde_type = cfg.sde.sde_type
    if sde_type == 'vp':
        sde = OU(beta_min=cfg.sde.beta_min, beta_max=cfg.sde.beta_max)
    elif sde_type == 'vp_cosine':
        sde = CosineOU()
    elif sde_type == 've':
        sde = None
    else:
        raise ValueError(f"Unknown sde_type: {sde_type}")

    # Wrap model
    if sde_type == 've':
        score_model = EDMDenoiser(model, noise_sampler, cfg)
    else:
        score_model = ScoreModel(model, sde, noise_sampler, cfg)

    # ── Build pos_batch (with conditioning for pinball) ────────────────
    domain = getattr(cfg.data, 'domain', 'square')
    n_cond = getattr(cfg.data, 'n_cond_channels', 0)
    pos_batch = None
    if domain == 'pinball' and n_cond > 0:
        # Pinball: pos_batch = [B, N, 2 + n_params + 1]
        # Use mid-range mu and mid-time for unconditional eval
        n_params = n_cond - 1  # last channel is phys_time
        pos_b = pos.unsqueeze(0).expand(n_samples, -1, -1)  # [B, N, 2]
        mu_dummy = torch.zeros(n_samples, n_nodes, n_params, device=device)
        t_dummy = 0.5 * torch.ones(n_samples, n_nodes, 1, device=device)
        pos_batch = torch.cat([pos_b, mu_dummy, t_dummy], dim=-1)
        print(f"  Pinball conditioning: n_cond={n_cond}, pos_batch shape={pos_batch.shape}")

    # ── Unconditional sampling ──────────────────────────────────────────
    print(f"  Sampling {n_samples} unconditional samples ({n_steps} steps, {sde_type})...")
    model.eval()
    with torch.no_grad():
        if sde_type == 've':
            samples = sample_ve_heun(score_model, pos, n_samples=n_samples,
                                     n_steps=n_steps, device=device,
                                     pos_batch=pos_batch,
                                     domain_onehot=domain_onehot)
        else:
            samples = sample_unconditional(score_model, pos, n_samples=n_samples,
                                           n_steps=n_steps, device=device,
                                           pos_batch=pos_batch,
                                           domain_onehot=domain_onehot)

    # Visualize.  Build a Triangulation only if the training-mesh xy/cells
    # match the eval sample length.  For cross-resolution specs (finer/
    # coarser) the hierarchy file has no xy_list/cells_list, the training
    # triangulation doesn't match, so we pass tri=None and plot_uncond_samples
    # falls through to its existing scatter path keyed on xy.
    xy_list = hier.get('xy_list') or [None]
    cells_list = hier.get('cells_list') or [None]
    xy    = xy_list[0]
    cells = cells_list[0]
    if xy is None and train_mesh_info is not None:
        xy = train_mesh_info.get('xy')
    if cells is None and train_mesh_info is not None:
        cells = train_mesh_info.get('cells')

    tri = None
    shading = 'flat'   # DG-0 (per-cell). Square / conductivity datasets.
    if xy is not None and cells is not None:
        n_points = len(xy)
        n_tris   = len(cells)
        if n_nodes == n_points:
            # P1 layout (per-vertex). Pinball, etc.
            tri = Triangulation(xy[:, 0], xy[:, 1], cells)
            shading = 'gouraud'
        elif n_nodes == n_tris:
            tri = Triangulation(xy[:, 0], xy[:, 1], cells)
            shading = 'flat'

    # Coords for the scatter fallback must align with sample length — mesh_pos
    # always has exactly n_nodes entries from hier['centers'][0].
    print("tri:" , tri, "shading:", shading)
    fig_uncond = plot_uncond_samples(samples, tri, shading=shading, xy=mesh_pos)
    fig_path = os.path.join(output_dir, f'{eval_name}_uncond_samples.png')
    fig_uncond.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig_uncond)
    print(f"  Saved: {fig_path}")

    # Basic sample statistics
    results['sample_mean'] = float(samples.mean())
    results['sample_std'] = float(samples.std())
    results['sample_min'] = float(samples.min())
    results['sample_max'] = float(samples.max())

    # ── DPS reconstruction (optional) ───────────────────────────────────
    if dps_gt_field is not None:
        from pde_operators.sensors import SparseSensorOperator
        gt = dps_gt_field.to(device)
        print(f"  Running DPS reconstruction ({n_dps_sensors} sensors)...")
        # Build observation operator + observed values.  Match the
        # train.py in-loop eval pattern at line ~811.
        forward_op = SparseSensorOperator(n_dofs=n_nodes,
                                          n_sensors=n_dps_sensors,
                                          device=device)
        y_obs = forward_op.forward(gt.unsqueeze(0) if gt.ndim == 2 else gt)
        if sde_type == 've':
            x_recon, metrics = run_dps_eval_ve(
                y_obs, 1, score_model, pos, mesh_pos, gt,
                forward_op=forward_op, n_steps=n_steps,
                guidance_weight=dps_guidance, device=device,
                pos_batch=pos_batch,
            )
        else:
            x_recon, metrics = run_dps_eval(
                y_obs, 1, score_model, pos, mesh_pos,
                forward_op=forward_op, gt_field=gt,
                n_sensors=n_dps_sensors,
                n_steps=n_steps, guidance_weight=dps_guidance,
                device=device, pos_batch=pos_batch,
            )
        # Scatter fallback coords MUST align with sample length. `mesh_pos`
        # is hier['centers'][0].numpy() and always has exactly n_nodes entries,
        # unlike `xy` (training-mesh vertices, length differs on cross-res specs).
        fig_dps = plot_cond_eval(
            gt[0].cpu().numpy(), x_recon[:, 0].cpu().numpy(),
            forward_op, tri, metrics, mesh_pos=mesh_pos,
            shading='flat', xy=mesh_pos,
        )
        fig_path = os.path.join(output_dir, f'{eval_name}_dps.png')
        fig_dps.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig_dps)
        results['dps_rel_l2'] = metrics['rel_l2']
        results['dps_mse'] = metrics['mse']
        print(f"  DPS rel-L2: {metrics['rel_l2']:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# Batch runner
# ═══════════════════════════════════════════════════════════════════════════

def build_eval_hierarchy(eval_spec, n_levels=None):
    """Build hierarchy from an eval specification dict.

    Parameters
    ----------
    eval_spec : dict with keys:
        hierarchy_pt : str  (path to pre-generated .pt file — preferred)
        OR domain-specific keys for on-the-fly building:
            domain : str
            For square:  resolutions: [[nx, ny], ...]
            For l_shape/gmsh: maxh_levels: [float, ...]
            For pinball: data_dir, ms_base_dir, ms_res_dirs, ms_all_res_dirs
    n_levels : int, optional — only use first n_levels from .pt file
    """
    # Preferred: load from pre-generated .pt file
    if 'hierarchy_pt' in eval_spec:
        offset = eval_spec.get('offset', 0)
        return load_hierarchy_pt(eval_spec['hierarchy_pt'],
                                 n_levels=n_levels, offset=offset)

    domain = eval_spec['domain']

    if domain == 'square':
        return build_square_hierarchy(eval_spec['resolutions'])
    elif domain == 'l_shape':
        return build_lshape_hierarchy(eval_spec['maxh_levels'])
    elif domain == 'pinball':
        ms_res = eval_spec['ms_res_dirs']
        ms_all = eval_spec.get('ms_all_res_dirs', ms_res)
        return load_pinball_hierarchy(
            eval_spec['data_dir'], eval_spec['ms_base_dir'],
            ms_res, ms_all,
        )
    else:
        # Generic gmsh domain
        return build_generic_hierarchy(domain, eval_spec['maxh_levels'])


def make_eval_name(run_dir, eval_spec, idx):
    """Generate a descriptive name for an eval experiment."""
    run_name = Path(run_dir).name
    domain = eval_spec['domain']
    if domain == 'square':
        res = eval_spec['resolutions']
        tag = 'x'.join(f"{r[0]}" for r in res)
    elif domain == 'pinball':
        tag = ','.join(eval_spec['ms_res_dirs'])
    else:
        tag = ','.join(f"{h}" for h in eval_spec.get('maxh_levels', []))
    return f"{run_name}_{domain}_{tag}"


def run_batch(config_path, device='cuda'):
    """Run all experiments from a YAML config file."""
    with open(config_path) as f:
        batch_cfg = yaml.safe_load(f)

    output_dir = batch_cfg.get('output_dir', 'results/res_invariance')
    n_samples = batch_cfg.get('n_samples', 8)
    n_steps = batch_cfg.get('n_steps', 200)
    device = batch_cfg.get('device', device)

    all_results = []

    for exp in batch_cfg['experiments']:
        run_dir = exp['run_dir']
        checkpoint = exp.get('checkpoint', 'model_ema_latest.pt')
        print(f"\n{'='*70}")
        print(f"Loading: {run_dir}")
        model, cfg, train_mesh_info = load_trained_model(run_dir, checkpoint, device)

        for i, eval_spec in enumerate(exp['evals']):
            eval_name = make_eval_name(run_dir, eval_spec, i)
            print(f"\n--- Eval: {eval_name} ---")

            hier = build_eval_hierarchy(eval_spec)
            is_multiscale = hasattr(model, 'n_levels')
            is_fixedgrid  = hasattr(model, 'grid_h') and not is_multiscale

            if is_multiscale:
                n_levels_model = model.n_levels
                n_levels_hier = len(hier['n_nodes_list'])
                if n_levels_model != n_levels_hier:
                    print(f"  SKIP: model has {n_levels_model} levels, "
                          f"hierarchy has {n_levels_hier}")
                    continue
                swap_mesh_hierarchy(model, hier, device)
            elif is_fixedgrid:
                try:
                    swap_grid(model, hier, device, mode='native')
                except ValueError as e:
                    print(f"  SKIP: {e}")
                    continue
            else:
                print(f"  SKIP: model type not supported (no n_levels / grid_h)")
                continue
            results = run_eval(
                model, cfg, hier, eval_name, output_dir,
                n_samples=n_samples, n_steps=n_steps, device=device,
                train_mesh_info=train_mesh_info,
            )
            all_results.append(results)

            # Reload original weights for next eval (mesh swap may have
            # altered latent transformer buffers)
            ckpt_path = os.path.join(run_dir, checkpoint)
            sd = torch.load(ckpt_path, map_location=device, weights_only=False)
            if isinstance(sd, dict):
                sd.pop('_metadata', None)
            model.load_state_dict(sd, strict=False)
            model.eval()

    # Save summary
    summary_path = os.path.join(output_dir, 'results_summary.json')
    os.makedirs(output_dir, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"All results saved to {summary_path}")
    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# Eval suite: auto-generate battery of eval specs from training config
# ═══════════════════════════════════════════════════════════════════════════

def _get_eval_suite(cfg, n_levels, hierarchy_dir, cross_domain=False):
    """Generate a standard battery of eval specs from training config.

    All specs reference pre-generated .pt hierarchy files in hierarchy_dir.
    Generate these locally with data_tools/build_hierarchy.py.
    Coarser evals use offset into the existing hierarchy (no separate file needed).

    cross_domain=False (default) -> resolution-invariance specs on the trained
    domain (same_res / finer / coarser). cross_domain=True -> the trained model
    applied to every *other* pre-generated conductivity shape (cross-domain
    generalisation), kept separate from the resolution battery.

    Returns list of (eval_name, eval_spec) tuples.
    """
    domain = getattr(cfg.data, 'domain', 'square')
    specs = []

    def _pt(name):
        return os.path.join(hierarchy_dir, name)

    # Cross-domain generalisation: trained model on every *other* pre-generated
    # shape. Separate suite (own output folder) from resolution invariance.
    if cross_domain:
        if domain == 'pinball':
            raise ValueError("--cross_domain is not supported for pinball models")
        _CROSS_SHAPES = ['square', 'l_shape', 'plus', 'e_shape', 'x_shape',
                         'circle', 'square_with_hole', 'circle_with_hole']
        for shape in _CROSS_SHAPES:
            if shape == domain:
                continue
            pt_name = 'mesh_hierarchy.pt' if shape == 'square' else f'mesh_hierarchy_{shape}.pt'
            specs.append((f'cross_{shape}', {'domain': shape, 'hierarchy_pt': _pt(pt_name)}))
        return specs

    if domain == 'square':
        # mesh_hierarchy.pt has levels [32x32, 16x16, 8x8, 4x4]
        specs.append(('same_res', {'domain': domain, 'hierarchy_pt': _pt('mesh_hierarchy.pt')}))
        specs.append(('finer',    {'domain': domain, 'hierarchy_pt': _pt('mesh_hierarchy_square_64.pt')}))
        specs.append(('coarser',  {'domain': domain, 'hierarchy_pt': _pt('mesh_hierarchy.pt'), 'offset': 1}))

    elif domain == 'l_shape':
        # mesh_hierarchy_l_shape.pt has levels [maxh=0.05, 0.1, 0.2, 0.4]
        specs.append(('same_res', {'domain': domain, 'hierarchy_pt': _pt('mesh_hierarchy_l_shape.pt')}))
        specs.append(('finer',    {'domain': domain, 'hierarchy_pt': _pt('mesh_hierarchy_l_shape_0025.pt')}))
        # Coarser: skip finest level
        specs.append(('coarser',  {'domain': domain, 'hierarchy_pt': _pt('mesh_hierarchy_l_shape.pt'), 'offset': 1}))

    elif domain == 'pinball':
        ms_base_dir = getattr(cfg.data, 'ms_base_dir', None)
        if not ms_base_dir:
            raise ValueError("cfg.data.ms_base_dir is required for pinball eval suite")

        # Node counts for reference:
        #   orig(7525) lc_250(7384) lc_300(5266) lc_500(1934)
        #   lc_1000(557) lc_2000(226) lc_4000(140)
        _PINBALL_ORDER = [
            'orig', 'lc_250_kNN', 'lc_300_kNN', 'lc_500_kNN',
            'lc_1000_kNN', 'lc_2000_kNN', 'lc_4000_kNN',
        ]
        # Standard eval windows: same set as multires_eval_windows in sweep YAML
        _PINBALL_WINDOWS = [
            "orig,lc_500_kNN,lc_2000_kNN",
            "lc_250_kNN,lc_500_kNN,lc_2000_kNN",
            "orig,lc_1000_kNN,lc_4000_kNN",
            "lc_500_kNN,lc_1000_kNN,lc_4000_kNN",
        ]
        for window in _PINBALL_WINDOWS:
            levels = [s.strip() for s in window.split(',')]
            finest, coarse = levels[0], levels[1:]
            data_dir = (ms_base_dir if finest == 'orig'
                        else os.path.join(ms_base_dir, finest))
            start_idx = _PINBALL_ORDER.index(coarse[0])
            end_idx   = _PINBALL_ORDER.index(coarse[-1])
            ms_all    = _PINBALL_ORDER[start_idx:end_idx + 1]
            tag = window.replace(',', '-')
            specs.append((f"pinball_{tag}", {
                'domain':        'pinball',
                'data_dir':      data_dir,
                'ms_base_dir':   ms_base_dir,
                'ms_res_dirs':   coarse,
                'ms_all_res_dirs': ms_all,
            }))

        # Cross-domain: conductivity shapes at 3 levels (offset=1 skips finest,
        # matching the 3-level pinball model).
        for shape in ['x_shape', 'circle_with_hole']:
            specs.append((f'cross_{shape}', {
                'domain':       shape,
                'hierarchy_pt': _pt(f'mesh_hierarchy_{shape}.pt'),
                'offset':       1,
            }))

    elif domain == 'multidomain':
        # 8 conductivity shapes × 2 resolutions (same_res training mesh, finer).
        # Finer meshes expected at mesh_hierarchy_<shape>_0025.pt; skipped with
        # a warning if missing on disk.
        _MULTIDOMAIN_SHAPES = [
            'circle', 'circle_with_hole', 'e_shape', 'l_shape',
            'plus', 'square', 'square_with_hole', 'x_shape',
        ]
        for shape in _MULTIDOMAIN_SHAPES:
            pt_std = 'mesh_hierarchy.pt' if shape == 'square' \
                     else f'mesh_hierarchy_{shape}.pt'
            pt_fine = 'mesh_hierarchy_square_64.pt' if shape == 'square' \
                      else f'mesh_hierarchy_{shape}_0025.pt'
            specs.append((f'{shape}_same_res',
                          {'domain': shape, 'hierarchy_pt': _pt(pt_std)}))
            if os.path.exists(_pt(pt_fine)):
                specs.append((f'{shape}_finer',
                              {'domain': shape, 'hierarchy_pt': _pt(pt_fine)}))
            else:
                print(f"  [eval suite] note: {pt_fine} not found — "
                      f"skipping '{shape}_finer'")
        return specs

    else:
        raise ValueError(f"eval_suite not implemented for domain '{domain}'")

    return specs


def _get_eval_suite_fixedgrid(cfg, hierarchy_dir):
    """Eval suite for CNN/FNO models — square domain only, same/finer/coarser.

    CNN and FNO operate on a fixed rectangular grid, so they can only be
    evaluated on the square domain (no l_shape / pinball / cross-domain).

    Returns list of (eval_name, eval_spec) tuples.  Empty list for non-square
    training domains.
    """
    domain = getattr(cfg.data, 'domain', 'square')
    if domain != 'square':
        return []

    def _pt(name):
        return os.path.join(hierarchy_dir, name)

    return [
        ('same_res', {'domain': 'square', 'hierarchy_pt': _pt('mesh_hierarchy.pt')}),
        ('finer',    {'domain': 'square', 'hierarchy_pt': _pt('mesh_hierarchy_square_64.pt')}),
        ('coarser',  {'domain': 'square', 'hierarchy_pt': _pt('mesh_hierarchy.pt'), 'offset': 1}),
    ]


def _run_eval_suite_fixedgrid(model, cfg, train_mesh_info,
                              run_dir, checkpoint,
                              n_samples, n_steps, output_dir,
                              device, hierarchy_dir, use_wandb,
                              test_dataset=None):
    """Eval loop for CNN / FNO: same/finer/coarser on square domain.

    For CNN runs two modes per spec (native, interpolate).
    For FNO runs only native.
    DPS is run only for `same_res` specs (test data is at training resolution).
    """
    # Co-locate outputs under the run/checkpoint dir by default; an explicit
    # --output_dir (e.g. a shared results dir) keeps a per-run subdir.
    run_name = Path(run_dir).name
    run_output = (os.path.join(run_dir, 'resolution_invariance') if output_dir is None
                  else os.path.join(output_dir, run_name))
    os.makedirs(run_output, exist_ok=True)

    is_fno = hasattr(model, 'fno')
    model_tag = 'fno' if is_fno else 'cnn'
    train_grid = (model.grid_h, model.grid_w)

    # Log hyperparams to wandb (match the multiscale branch)
    if use_wandb:
        import wandb
        # Re-attach if caller (e.g. train.py post-context) lost the active run
        if wandb.run is None:
            _resume_training_wandb_run(run_dir, cfg, resume_mode='allow')
        wandb.config.update({
            'run_dir': run_dir, 'checkpoint': checkpoint,
            'train_domain': getattr(cfg.data, 'domain', 'square'),
            'n_samples': n_samples, 'n_steps': n_steps,
            'res_invariant': cfg.model.res_invariant,
            'use_edge_geom': cfg.model.use_edge_geom,
            'use_latent_transformer': cfg.model.use_latent_transformer,
            'sde_type': cfg.sde.sde_type,
            'hidden_dim': cfg.model.hidden_dim,
            'num_layers': cfg.model.num_layers,
            'layer_type': getattr(cfg.model, 'layer_type', model_tag),
            'conv_type': getattr(cfg.model, 'conv_type', 'multiscale'),
            'batch_size': getattr(cfg.training, 'batch_size', None),
            'train_run_id': Path(run_dir).name,
            'train_grid': f"{train_grid[0]}x{train_grid[1]}",
        }, allow_val_change=True)

    specs = _get_eval_suite_fixedgrid(cfg, hierarchy_dir)
    if not specs:
        print(f"  SKIP: fixed-grid eval suite is square-only "
              f"(domain='{cfg.data.domain}')")
        return []

    # Modes per model type
    modes = ('native',) if is_fno else ('native', 'interpolate')

    all_results = []
    for eval_tag, eval_spec in specs:
        try:
            hier = build_eval_hierarchy(eval_spec)
        except Exception as e:
            print(f"\n--- Eval: {run_name}_{eval_tag} ---")
            print(f"  SKIP (hierarchy build failed): {e}")
            continue

        for mode in modes:
            tag_full = f"{eval_tag}_{mode}" if not is_fno else eval_tag
            eval_name = (f"{run_name}_square_{tag_full}_"
                         f"{hier['n_nodes_list'][0]}nodes")
            print(f"\n--- Eval: {eval_name} ---")

            # Reload weights between evals (swap_grid doesn't change weights,
            # but buffer shapes change, so restore cleanly before each run).
            ckpt_path = os.path.join(run_dir, checkpoint)
            sd = torch.load(ckpt_path, map_location=device, weights_only=False)
            if isinstance(sd, dict):
                sd.pop('_metadata', None)
            current_sd = model.state_dict()
            filtered_sd = {k: v for k, v in sd.items()
                           if k not in current_sd or current_sd[k].shape == v.shape}
            model.load_state_dict(filtered_sd, strict=False)
            # Restore training grid dims before each swap
            model.grid_h, model.grid_w = train_grid
            model.eval()

            try:
                swap_grid(model, hier, device, mode=mode,
                          train_grid=train_grid)
            except ValueError as e:
                print(f"  SKIP ({mode}): {e}")
                continue

            # For interpolate mode, wrap the CNN in an adapter.  The inner
            # model carries the eval-grid node map; the adapter resizes
            # images to training grid around the UNet stack.
            eval_model = (InterpolatingCNN(model, train_grid)
                          if mode == 'interpolate' else model)

            # DPS only on same_res (test data is at training resolution)
            dps_gt = _sample_dps_gt(test_dataset) if eval_tag == 'same_res' else None

            results = run_eval(
                eval_model, cfg, hier, eval_name, run_output,
                n_samples=n_samples, n_steps=n_steps, device=device,
                train_mesh_info=train_mesh_info,
                dps_gt_field=dps_gt,
            )
            results['eval_tag'] = tag_full
            results['eval_domain'] = 'square'
            results['mode'] = mode
            all_results.append(results)

            if use_wandb:
                import wandb
                log_dict = {f"{tag_full}/sample_mean": results['sample_mean'],
                            f"{tag_full}/sample_std": results['sample_std']}
                img_path = os.path.join(run_output, f'{eval_name}_uncond_samples.png')
                if os.path.exists(img_path):
                    log_dict[f"{tag_full}/samples"] = wandb.Image(img_path)
                if 'dps_rel_l2' in results:
                    log_dict[f"{tag_full}/dps_rel_l2"] = results['dps_rel_l2']
                wandb.log(log_dict)

    summary_path = os.path.join(run_output, 'results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Suite complete ({model_tag}). {len(all_results)} evals saved to {summary_path}")
    if use_wandb:
        import wandb
        wandb.save(summary_path)
    return all_results


def _extract_train_run_info(run_dir):
    """Parse wandb (project, run_id) from a checkpoint directory name.

    Convention: `{project}_{YYMMDD}_{run_id}`.  Examples:
        my_sweep_240101_a1b2c3d4
          → project='my_sweep', run_id='a1b2c3d4'
        pinball_sweep_v2_240101_e5f6h7i8
          → project='pinball_sweep_v2',    run_id='e5f6h7i8'

    The project from the dir name is the AUTHORITATIVE source — it reflects
    the actual wandb project the run was created in (which for sweeps is set
    by the sweep controller, not by the cfg defaults saved to config.yaml).

    Returns (None, None) if the pattern doesn't match (legacy dirs without date).
    """
    name = Path(run_dir).name
    parts = name.split('_')
    if len(parts) >= 3 and len(parts[-2]) == 6 and parts[-2].isdigit():
        project = '_'.join(parts[:-2])
        run_id  = parts[-1]
        return project, run_id
    return None, None


def _extract_train_run_id(run_dir):
    """Back-compat wrapper — returns run_id only."""
    _, run_id = _extract_train_run_info(run_dir)
    return run_id


def _load_cfg_only(run_dir):
    """Read config.yaml from a run_dir without loading the model."""
    cfg_path = os.path.join(run_dir, 'config.yaml')
    if not os.path.exists(cfg_path):
        return None
    return OmegaConf.load(cfg_path)


def _resume_training_wandb_run(run_dir, cfg, resume_mode='must'):
    """Resume the original training run in wandb so eval metrics land on
    the same row as training curves.

    Project name is derived from the run_dir folder name (authoritative)
    because cfg.training.wandb_project in config.yaml holds only the CLI default,
    not the real project set by the wandb sweep controller at runtime.  Entity
    comes from cfg as a fallback.

    If resume_mode='never' (or parsing fails), falls back to a standalone
    'res_invariance_eval' run.
    """
    import wandb
    project, run_id = _extract_train_run_info(run_dir)
    entity = getattr(cfg.training, 'wandb_entity', None) if cfg is not None else None
    entity = entity or None

    if resume_mode == 'never' or run_id is None or project is None:
        print(f"  [wandb] standalone eval run (project='res_invariance_eval')")
        return wandb.init(project='res_invariance_eval',
                          config={'train_run_id': run_id,
                                  'run_dir': run_dir},
                          reinit=True)

    print(f"  [wandb] resuming training run {entity}/{project}/{run_id} (mode={resume_mode})")
    return wandb.init(project=project, entity=entity,
                      id=run_id, resume=resume_mode, reinit=True)


def _sample_dps_gt(test_dataset):
    """Pull a single ground-truth field from the held-out test dataset.

    Returns a `[1, N]` tensor (on CPU) or None if no test_dataset provided.
    Handles the three dataset shapes in GRIFDIR:
      - Pinball: tuple (field[1,N], mu, time_idx)  → returns field
      - SST:     tuple (field[1,N], mu, time_idx)  → returns field
      - CachedDataset: field[1,N]                   → returns field
    """
    if test_dataset is None or len(test_dataset) == 0:
        return None
    idx = torch.randint(len(test_dataset), (1,)).item()
    item = test_dataset[idx]
    field = item[0] if isinstance(item, (tuple, list)) else item
    if field.ndim == 1:
        field = field.unsqueeze(0)
    return field


def _build_test_dataset_from_cfg(cfg):
    """Construct a held-out test dataset mirroring train.py's logic.

    Returns None if no test split is available for this domain (e.g. SST
    without the PhySense test .npy, or multidomain).
    """
    domain = getattr(cfg.data, 'domain', 'square')
    try:
        if domain == 'pinball':
            from data_utils import PinballDataset
            _dir = (getattr(cfg.data, 'ms_finest_dir', None)
                    or getattr(cfg.data, 'ms_base_dir', None)
                    or cfg.data.data_dir)
            return PinballDataset(_dir, split='test')
        if domain == 'sst':
            raise NotImplementedError(
                "The SST domain is not included in this GRIFDIR release.")
        # Conductivity (square / l_shape / etc.)
        import glob
        if domain == 'square':
            pat = f'conductivity_nx{cfg.data.nx}_ny{cfg.data.ny}_*.pt'
        else:
            pat = f'conductivity_{domain}_*.pt'
        cand = sorted(glob.glob(os.path.join(cfg.data.data_dir, pat)))
        if not cand:
            return None
        from data_utils import CachedDataset
        return CachedDataset(cand[-1], split='test')
    except Exception as e:
        print(f"  (test dataset unavailable: {e})")
        return None


def run_eval_suite(run_dir, checkpoint, n_samples, n_steps, output_dir,
                   device, hierarchy_dir=None, use_wandb=False,
                   test_dataset=None, dps=True, cross_domain=False):
    """Run full eval suite for one model checkpoint.

    If `test_dataset` is provided, DPS reconstruction is run on a random test
    field for every `same_res` eval spec (finer/coarser use unconditional only
    to avoid the cross-resolution projection question).

    If `test_dataset` is None AND `dps=True`, an attempt is made to build one
    automatically from the training config (see `_build_test_dataset_from_cfg`).
    Pass `dps=False` to skip DPS entirely (unconditional eval only).
    """
    # Default hierarchy_dir: <repo_root>/multiscale (resolved from this file, not CWD)
    if hierarchy_dir is None:
        hierarchy_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'meshes')
    print(f"\n{'='*70}")
    print(f"Loading: {run_dir}")
    print(f"Hierarchy dir: {hierarchy_dir}")
    model, cfg, train_mesh_info = load_trained_model(run_dir, checkpoint, device)

    # Auto-build test_dataset from cfg if not provided and DPS is requested.
    if test_dataset is None and dps:
        test_dataset = _build_test_dataset_from_cfg(cfg)
    if test_dataset is not None:
        print(f"Test dataset: {len(test_dataset)} samples (DPS on same_res only)")

    # Detect model type
    is_multiscale = hasattr(model, 'n_levels')
    is_fixedgrid  = hasattr(model, 'grid_h') and not is_multiscale

    if is_fixedgrid:
        return _run_eval_suite_fixedgrid(model, cfg, train_mesh_info,
                                         run_dir, checkpoint,
                                         n_samples, n_steps, output_dir,
                                         device, hierarchy_dir, use_wandb,
                                         test_dataset=test_dataset)

    if not is_multiscale:
        print(f"  SKIP: model has no n_levels and no grid_h — unsupported type")
        return []

    n_levels = model.n_levels
    domain = getattr(cfg.data, 'domain', 'square')

    # Co-locate outputs under the run/checkpoint dir by default; an explicit
    # --output_dir (e.g. a shared results dir) keeps a per-run subdir.
    run_name = Path(run_dir).name
    _suite_dir = 'cross_domain' if cross_domain else 'resolution_invariance'
    run_output = (os.path.join(run_dir, _suite_dir) if output_dir is None
                  else os.path.join(output_dir, run_name))
    os.makedirs(run_output, exist_ok=True)

    if use_wandb:
        import wandb
        # Guard against being called after the training wandb context closed
        # (e.g. train.py's post-training eval block sits outside `with wandb.init()`).
        # If no active run, re-attach to the training run in the usual way.
        if wandb.run is None:
            _resume_training_wandb_run(run_dir, cfg, resume_mode='allow')
        wandb.config.update({
            'run_dir': run_dir, 'checkpoint': checkpoint,
            'train_domain': domain, 'n_levels': n_levels,
            'n_samples': n_samples, 'n_steps': n_steps,
            # Training hyperparams — passed through for easy filtering in wandb
            'res_invariant': cfg.model.res_invariant,
            'use_edge_geom': cfg.model.use_edge_geom,
            'use_latent_transformer': cfg.model.use_latent_transformer,
            'sde_type': cfg.sde.sde_type,
            'hidden_dim': cfg.model.hidden_dim,
            'num_layers': cfg.model.num_layers,
            'layer_type': getattr(cfg.model, 'layer_type', 'simple_mp'),
            'mixing_type': getattr(cfg.model, 'mixing_type', 'vector'),
            'fem_k_hops': getattr(cfg.model, 'fem_k_hops', None),
            'fem_use_radius': getattr(cfg.model, 'fem_use_radius', None),
            'fem_radius_mult': getattr(cfg.model, 'fem_radius_mult', None),
            'conv_type': getattr(cfg.model, 'conv_type', 'multiscale'),
            'batch_size': getattr(cfg.training, 'batch_size', None),
            'train_run_id': Path(run_dir).name,
        }, allow_val_change=True)

    specs = _get_eval_suite(cfg, n_levels, hierarchy_dir, cross_domain=cross_domain)

    # For multi-domain runs, look up each shape's one-hot so the model's domain
    # heads fire correctly at eval time. Non-multidomain paths pass None.
    _is_multidomain = (domain == 'multidomain')
    if _is_multidomain:
        from data_utils import domain_onehot as _domain_onehot_lookup

    all_results = []

    for eval_tag, eval_spec in specs:
        try:
            hier = build_eval_hierarchy(eval_spec, n_levels=n_levels)
        except Exception as e:
            print(f"\n--- Eval: {run_name}_{eval_tag} ---")
            print(f"  SKIP (hierarchy build failed): {e}")
            continue

        eval_domain = eval_spec['domain']
        n_finest = hier['n_nodes_list'][0]
        eval_name = f"{run_name}_{eval_domain}_{eval_tag}_{n_finest}nodes"
        print(f"\n--- Eval: {eval_name} ---")

        n_levels_hier = len(hier['n_nodes_list'])
        if n_levels != n_levels_hier:
            print(f"  SKIP: model has {n_levels} levels, hierarchy has {n_levels_hier}")
            continue

        swap_mesh_hierarchy(model, hier, device)
        # Only use train_mesh_info triangulation for same-domain evals —
        # cross-domain evals have different node counts so it causes a mismatch
        same_domain = (eval_domain == domain)
        # DPS only on same_res same-domain (test fields live on training mesh)
        dps_gt = (_sample_dps_gt(test_dataset)
                  if eval_tag == 'same_res' and same_domain else None)

        eval_domain_onehot = None
        if _is_multidomain:
            try:
                oh = _domain_onehot_lookup(eval_domain).to(device)  # [n_domains]
                eval_domain_onehot = oh.unsqueeze(0).expand(n_samples, -1)
            except ValueError as e:
                print(f"  SKIP (domain onehot): {e}")
                continue

        results = run_eval(
            model, cfg, hier, eval_name, run_output,
            n_samples=n_samples, n_steps=n_steps, device=device,
            train_mesh_info=train_mesh_info if same_domain else None,
            dps_gt_field=dps_gt,
            domain_onehot=eval_domain_onehot,
        )
        results['eval_tag'] = eval_tag
        results['eval_domain'] = eval_spec['domain']
        all_results.append(results)

        # Log to wandb
        if use_wandb:
            import wandb
            log_dict = {f"{eval_tag}/sample_mean": results['sample_mean'],
                        f"{eval_tag}/sample_std": results['sample_std']}
            # Log sample image
            img_path = os.path.join(run_output, f'{eval_name}_uncond_samples.png')
            if os.path.exists(img_path):
                log_dict[f"{eval_tag}/samples"] = wandb.Image(img_path)
            if 'dps_rel_l2' in results:
                log_dict[f"{eval_tag}/dps_rel_l2"] = results['dps_rel_l2']
            wandb.log(log_dict)

        # Reload weights for next eval — filter shape-mismatched buffers
        # (coarse_coords etc. already set correctly by swap_mesh_hierarchy)
        ckpt_path = os.path.join(run_dir, checkpoint)
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        if isinstance(sd, dict):
            sd.pop('_metadata', None)
        current_sd = model.state_dict()
        filtered_sd = {k: v for k, v in sd.items()
                       if k not in current_sd or current_sd[k].shape == v.shape}
        model.load_state_dict(filtered_sd, strict=False)
        model.eval()

    # Save summary
    summary_path = os.path.join(run_output, 'results_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Suite complete. {len(all_results)} evals saved to {summary_path}")

    if use_wandb:
        import wandb
        wandb.save(summary_path)

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# In-training multi-resolution eval for Pinball
# ═══════════════════════════════════════════════════════════════════════════

def run_multires_eval_pinball(
    score_model,
    cfg,
    log_dir,
    ms_base_dir,
    eval_hierarchies,
    n_samples=4,
    n_steps=100,
    device='cuda',
    use_wandb=False,
    epoch=None,
    checkpoint_path=None,
):
    """Run multi-resolution evaluation on a list of Pinball hierarchy configs.

    Called at the end of training (or periodically) from train_pinball.py.

    Parameters
    ----------
    score_model : ScoreModel or EDMDenoiser wrapping the GNN
    cfg         : training Config (OmegaConf)
    log_dir     : run checkpoint dir (plots saved here under multires_eval/)
    ms_base_dir : parent of all coarse resolution dirs
    eval_hierarchies : list of dicts, each specifying one hierarchy:
        {
            'name':          str,                 # label, e.g. 'orig-500-2000'
            'data_dir':      str,                 # finest-level dir
            'ms_res_dirs':   list[str],           # coarse level dir names
            'ms_all_res_dirs': list[str],         # for operator composition
        }
        The number of levels (len(ms_res_dirs)+1) must equal n_levels of the model.
    n_samples   : unconditional samples to draw per hierarchy
    n_steps     : diffusion sampling steps
    device      : torch device string
    use_wandb   : whether to log images/metrics to wandb
    epoch       : current epoch number (for wandb step key)
    checkpoint_path : if provided, reload weights from here before each eval
                      (avoids state leakage across mesh swaps)

    Returns
    -------
    list of result dicts, one per hierarchy
    """
    import wandb as _wandb

    model = score_model.model
    n_levels = model.n_levels
    output_dir = os.path.join(log_dir, 'multires_eval')
    os.makedirs(output_dir, exist_ok=True)

    # Re-attach to the training wandb run if it was closed before we got here
    # (post-training eval in train.py is outside the `with wandb.init(...)`
    # context, so wandb.run is None and wandb.log() silently no-ops otherwise).
    if use_wandb and _wandb.run is None:
        _resume_training_wandb_run(log_dir, cfg, resume_mode='allow')

    all_results = []

    # Save current state so we can always restore to training mesh
    train_state = copy.deepcopy(model.state_dict())

    for spec in eval_hierarchies:
        eval_name    = spec['name']
        ms_finest    = spec.get('ms_finest_dir', spec.get('data_dir', ''))  # backward compat
        ms_res       = spec['ms_res_dirs']
        ms_all       = spec.get('ms_all_res_dirs', ms_res)

        print(f"\n  [multires_eval] {eval_name}: {ms_finest.split('/')[-1]} → {' → '.join(ms_res)}")

        # Build hierarchy
        try:
            hier = load_pinball_hierarchy(ms_finest, ms_base_dir, ms_res, ms_all)
        except Exception as e:
            print(f"    SKIP (hierarchy build failed): {e}")
            continue

        n_levels_hier = len(hier['n_nodes_list'])
        if n_levels_hier != n_levels:
            print(f"    SKIP: model has {n_levels} levels, hierarchy has {n_levels_hier}")
            continue

        # Swap mesh
        swap_mesh_hierarchy(model, hier, device)

        # Rebuild noise sampler on new finest mesh
        pos = hier['centers'][0].to(device)
        n_nodes = hier['n_nodes_list'][0]
        use_accel = getattr(cfg.training, 'use_accel_sampler', True)
        NoiseCls = RBFKernelAccel if use_accel else RBFKernel
        noise_sampler = NoiseCls(
            mesh_points=pos,
            scale=cfg.noise.scale,
            eps=cfg.noise.eps,
            device=device,
        )

        # Rebuild score wrapper with new noise sampler
        sde_type = cfg.sde.sde_type
        if sde_type == 've':
            eval_score_model = EDMDenoiser(model, noise_sampler, cfg)
        else:
            if sde_type == 'vp':
                sde = OU(beta_min=cfg.sde.beta_min, beta_max=cfg.sde.beta_max)
            else:
                sde = CosineOU()
            eval_score_model = ScoreModel(model, sde, noise_sampler, cfg)

        # Build pos_batch (conditioning) — use dummy values for unconditional viz
        n_cond = getattr(cfg.data, 'n_cond_channels', 0)
        pos_batch = None
        if n_cond > 0:
            n_params = n_cond - 1
            pos_b  = pos.unsqueeze(0).expand(n_samples, -1, -1)
            mu_dummy = torch.zeros(n_samples, n_nodes, n_params, device=device)
            t_dummy  = 0.5 * torch.ones(n_samples, n_nodes, 1, device=device)
            pos_batch = torch.cat([pos_b, mu_dummy, t_dummy], dim=-1)

        # Sample
        model.eval()
        with torch.no_grad():
            if sde_type == 've':
                samples = sample_ve_heun(
                    eval_score_model, pos, n_samples=n_samples,
                    n_steps=n_steps, device=device, pos_batch=pos_batch,
                )
            else:
                samples = sample_unconditional(
                    eval_score_model, pos, n_samples=n_samples,
                    n_steps=n_steps, device=device, pos_batch=pos_batch,
                )

        # Plot
        mesh_pos = hier['centers'][0].numpy()
        xy = mesh_pos  # scatter plot fallback
        fig_uncond = plot_uncond_samples(samples, tri=None, shading='flat', xy=xy)
        fig_path = os.path.join(output_dir, f'{eval_name}_uncond_samples.png')
        fig_uncond.savefig(fig_path, dpi=120, bbox_inches='tight')
        plt.close(fig_uncond)
        print(f"    Saved: {fig_path}")

        result = {
            'name': eval_name,
            'n_nodes_list': hier['n_nodes_list'],
            'sample_mean': float(samples.mean()),
            'sample_std':  float(samples.std()),
            'sample_min':  float(samples.min()),
            'sample_max':  float(samples.max()),
        }
        all_results.append(result)

        # Log to wandb
        if use_wandb:
            log_dict = {
                f"multires/{eval_name}/sample_mean": result['sample_mean'],
                f"multires/{eval_name}/sample_std":  result['sample_std'],
                f"multires/{eval_name}/samples": _wandb.Image(fig_path),
            }
            if epoch is not None:
                log_dict['epoch'] = epoch
            _wandb.log(log_dict)

        # Restore weights before next mesh swap to avoid any state leakage.
        # strict=False only allows missing/extra keys, NOT shape mismatches —
        # filter shape-changed params (e.g. latent_transformer.coarse_coords
        # when the mesh swap landed on a level with a different node count).
        current_sd = model.state_dict()
        if checkpoint_path and os.path.exists(checkpoint_path):
            sd = torch.load(checkpoint_path, map_location=device, weights_only=False)
            if isinstance(sd, dict):
                sd.pop('_metadata', None)
        else:
            sd = train_state
        filtered = {k: v for k, v in sd.items()
                    if k in current_sd and current_sd[k].shape == v.shape}
        model.load_state_dict(filtered, strict=False)
        model.eval()

    # Restore to original training-mesh state fully (same shape filter)
    current_sd = model.state_dict()
    filtered = {k: v for k, v in train_state.items()
                if k in current_sd and current_sd[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)

    # Save summary JSON
    summary_path = os.path.join(output_dir, 'multires_results.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  [multires_eval] done — {len(all_results)} evals, summary: {summary_path}")

    return all_results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def get_run_dirs_from_sweep(sweep_path):
    """Query wandb API for finished runs in a training sweep.

    sweep_path: 'SWEEP_ID' or 'entity/project/SWEEP_ID'
    Returns list of (run_id, run_dir) tuples.
    """
    import wandb
    api = wandb.Api()
    parts = sweep_path.split('/')
    if len(parts) == 1:
        entity, project, sweep_id = 'GRIFDIR', 'res_invariant_train', parts[0]
    elif len(parts) == 3:
        entity, project, sweep_id = parts
    else:
        raise ValueError(f"Expected 'SWEEP_ID' or 'entity/project/SWEEP_ID', got: {sweep_path}")

    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    results = []
    for run in sweep.runs:
        if run.state != 'finished':
            print(f"  SKIP run {run.id} (state={run.state})")
            continue
        save_dir = run.config.get('save_dir')
        conv_type = run.config.get('conv_type', 'multiscale')
        if save_dir is None:
            print(f"  SKIP run {run.id}: no save_dir in config")
            continue
        conv_dir = Path(save_dir) / f'conv={conv_type}'
        matches = list(conv_dir.glob(f'*_{run.id}')) + [conv_dir / run.id]
        run_dir = str(next((p for p in matches if p.exists()), conv_dir / run.id))
        results.append((run.id, run_dir))
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Resolution invariance evaluation for trained diffusion models'
    )
    # Mode selection
    parser.add_argument('--config', type=str, default=None,
                        help='YAML config for batch experiments')
    parser.add_argument('--eval_suite', action='store_true',
                        help='Auto-run the resolution battery on the trained domain (same-res, finer, coarser).')
    parser.add_argument('--cross_domain', action='store_true',
                        help='With --eval_suite: run the trained model on every other pre-generated '
                             'shape (cross-domain generalisation) instead, into <run>/cross_domain/.')
    # Model
    parser.add_argument('--run_dir', type=str, default=None,
                        help='Run directory containing checkpoint + config.yaml')
    parser.add_argument('--checkpoint', type=str, default='model_ema_latest.pt')
    # Single eval params
    parser.add_argument('--eval_domain', type=str, default='square',
                        choices=['square', 'l_shape', 'pinball',
                                 'plus', 'e_shape', 'x_shape',
                                 'circle', 'square_with_hole', 'circle_with_hole'])
    parser.add_argument('--eval_resolutions', type=str, nargs='+', default=None,
                        help='Square: "64,64" "32,32" "16,16". Others: "0.05" "0.1" "0.2"')
    # Shared
    parser.add_argument('--n_samples', type=int, default=8)
    parser.add_argument('--n_steps', type=int, default=200)
    parser.add_argument('--output_dir', type=str, default=None,
                        help='default: <run_dir>/resolution_invariance')
    parser.add_argument('--hierarchy_dir', type=str, default='meshes',
                        help='Directory containing pre-generated mesh_hierarchy_*.pt files')
    parser.add_argument('--device', type=str, default=None)
    # Wandb
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_sweep', action='store_true',
                        help='Set when launched by wandb agent (reads wandb.config)')
    parser.add_argument('--from_sweep', type=str, default=None,
                        help='Evaluate all finished runs from a training sweep. '
                             'Accepts "SWEEP_ID" (defaults to GRIFDIR/res_invariant_train) '
                             'or "entity/project/SWEEP_ID".')
    parser.add_argument('--resume_mode', type=str, default='must',
                        choices=['must', 'allow', 'never'],
                        help="How to attach to the original training run:\n"
                             "  'must'  — fail if run_id not found in wandb (default, safest)\n"
                             "  'allow' — create new run if original missing\n"
                             "  'never' — always log to separate 'res_invariance_eval' project")
    parser.add_argument('--no_dps', action='store_true',
                        help='Skip DPS reconstruction (unconditional eval only).')
    args = parser.parse_args()

    device = get_device(args.device)

    # ── From-sweep batch mode ─────────────────────────────────────────
    if args.from_sweep:
        run_dirs = get_run_dirs_from_sweep(args.from_sweep)
        print(f"Found {len(run_dirs)} finished runs in sweep '{args.from_sweep}'")
        for run_id, run_dir in run_dirs:
            print(f"\n{'='*70}\nEvaluating run: {run_id}  ({run_dir})")
            if not os.path.isdir(run_dir):
                print(f"  SKIP: directory not found: {run_dir}")
                continue
            if args.use_wandb:
                _cfg = _load_cfg_only(run_dir)
                _resume_training_wandb_run(run_dir, _cfg,
                                           resume_mode=args.resume_mode)
            run_eval_suite(
                run_dir, args.checkpoint,
                n_samples=args.n_samples, n_steps=args.n_steps,
                output_dir=args.output_dir, device=device,
                hierarchy_dir=args.hierarchy_dir,
                use_wandb=args.use_wandb,
                dps=not args.no_dps,
                cross_domain=args.cross_domain,
            )
            if args.use_wandb:
                import wandb
                wandb.finish()
        return

    # ── Wandb sweep mode ──────────────────────────────────────────────
    if args.use_wandb:
        import wandb
        if args.wandb_sweep:
            wandb.init()
            # Override CLI args with sweep config
            sc = wandb.config
            args.run_dir = getattr(sc, 'run_dir', args.run_dir)
            args.n_samples = getattr(sc, 'n_samples', args.n_samples)
            args.n_steps = getattr(sc, 'n_steps', args.n_steps)
            args.checkpoint = getattr(sc, 'checkpoint', args.checkpoint)
            args.hierarchy_dir = getattr(sc, 'hierarchy_dir', args.hierarchy_dir)
            args.eval_suite = True  # sweep always runs full suite
        elif args.run_dir is not None:
            # Resume the original training run so eval metrics land on the
            # same wandb row as training curves/hyperparams.
            _cfg = _load_cfg_only(args.run_dir)
            _resume_training_wandb_run(args.run_dir, _cfg,
                                       resume_mode=args.resume_mode)
        else:
            wandb.init(project='res_invariance_eval',
                       config=vars(args))

    # ── Batch YAML mode ───────────────────────────────────────────────
    if args.config:
        run_batch(args.config, device=device)
        return

    if args.run_dir is None:
        parser.error("Provide --config, --eval_suite + --run_dir, or --run_dir + --eval_domain")

    # ── Eval suite mode ───────────────────────────────────────────────
    if args.eval_suite:
        run_eval_suite(
            args.run_dir, args.checkpoint,
            n_samples=args.n_samples, n_steps=args.n_steps,
            output_dir=args.output_dir, device=device,
            hierarchy_dir=args.hierarchy_dir,
            use_wandb=args.use_wandb,
            dps=not args.no_dps,
            cross_domain=args.cross_domain,
        )
        if args.use_wandb:
            import wandb
            wandb.finish()
        return

    # ── Single experiment via CLI ─────────────────────────────────────
    print(f"Device: {device}")
    model, cfg, train_mesh_info = load_trained_model(args.run_dir, args.checkpoint, device)

    # Parse resolutions
    if args.eval_domain == 'square':
        if args.eval_resolutions is None:
            nx = cfg.data.nx
            resolutions = [
                [nx * 2, nx * 2], [nx, nx],
                [nx // 2, nx // 2], [nx // 4, nx // 4],
            ]
        else:
            resolutions = [[int(x) for x in r.split(',')] for r in args.eval_resolutions]
        hier = build_square_hierarchy(resolutions)
        eval_name = f"square_{'x'.join(str(r[0]) for r in resolutions)}"

    elif args.eval_domain == 'l_shape':
        if args.eval_resolutions is None:
            maxh_levels = [0.03, 0.06, 0.12, 0.24]
        else:
            maxh_levels = [float(x) for x in args.eval_resolutions]
        hier = build_lshape_hierarchy(maxh_levels)
        eval_name = f"lshape_{'_'.join(f'{h:.2f}' for h in maxh_levels)}"

    elif args.eval_domain == 'pinball':
        parser.error("For pinball, use --config with YAML specifying data dirs")

    else:
        # Generic gmsh domain (plus, e_shape, x_shape, circle, *_with_hole)
        if args.eval_resolutions is None:
            maxh_levels = [0.05, 0.1, 0.2, 0.4]
        else:
            maxh_levels = [float(x) for x in args.eval_resolutions]
        hier = build_generic_hierarchy(args.eval_domain, maxh_levels)
        eval_name = f"{args.eval_domain}_{'_'.join(f'{h:.2f}' for h in maxh_levels)}"

    n_levels_model = model.n_levels
    n_levels_hier = len(hier['n_nodes_list'])
    if n_levels_model != n_levels_hier:
        print(f"ERROR: model has {n_levels_model} levels but hierarchy has {n_levels_hier}. "
              f"Provide {n_levels_model} resolution levels.")
        sys.exit(1)

    swap_mesh_hierarchy(model, hier, device)
    results = run_eval(
        model, cfg, hier, eval_name, args.output_dir,
        n_samples=args.n_samples, n_steps=args.n_steps, device=device,
        train_mesh_info=train_mesh_info,
    )
    print(f"\nResults: {json.dumps(results, indent=2)}")


if __name__ == '__main__':
    main()
