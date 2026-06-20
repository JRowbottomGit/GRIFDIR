"""
Unified GRIFDIR training script.

Handles all domain types via ``--domain``:
  - conductivity domains: square, l_shape, circle, x_shape, etc.
  - pinball: Navier-Stokes flow with mu/time conditioning
  - multidomain: joint training across multiple conductivity domains
                 with domain-specific cross-attention heads

Also supports: VE/EDM + VP SDE, AMP, resume from checkpoint.

Replaces the previous train.py, train_amp.py, train_pinball.py,
train_resume.py, and train_multidomain.py.
"""

import contextlib
import itertools
import time
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

import os
import glob
import socket
from datetime import datetime
from tqdm import tqdm

import torch
import numpy as np
import wandb
import argparse

from omegaconf import OmegaConf
from ema_pytorch import EMA

from torch.utils.data import DataLoader
from data_utils import (get_conductivity_dataloader, CachedDataset, PinballDataset,
                        DOMAIN_REGISTRY, N_DOMAINS, DOMAIN_FILES)
from diffusion.sde import OU, CosineOU
from diffusion.precond import ScoreModel, EDMDenoiser
from diffusion.noise import RBFKernel, RFFNoiseSampler
from diffusion.noise_accel import RBFKernel as RBFKernelAccel
from eval.sampling import (sample_unconditional, run_dps_eval, plot_uncond_samples,
                        plot_cond_eval, sample_ve_heun, run_dps_eval_ve)
from eval.resolution_invariance import run_multires_eval_pinball, run_eval_suite

from config import Config, merge_config, get_device
from model_mesh_utils import get_model, load_all_hierarchies

from pde_operators.sensors import SparseSensorOperator


def t_or_f(tf_str):
    """argparse type: parse the string forms of True/False/None to Python values."""
    if tf_str == "True" or tf_str == "true" or (type(tf_str) == bool and tf_str):
        return True
    elif tf_str == "False" or tf_str == "false" or (type(tf_str) == bool and not tf_str):
        return False
    elif tf_str == "None" or tf_str == "null" or tf_str is None:
        return None
    else:
        return tf_str


def _build_noise_sampler(mesh_pos, cfg, device):
    """Instantiate the correct noise sampler from config.

    sampler_type='rbf'  — dense RBF Cholesky (default, original behaviour)
    sampler_type='rff'  — Random Fourier Features (O(N*D), resolution-invariant,
                          default for SST-scale meshes where RBF is infeasible)
    """
    sampler_type = cfg.noise.sampler_type.lower()
    N = mesh_pos.shape[0]

    if sampler_type == 'rff':
        sampler = RFFNoiseSampler(mesh_pos, scale=cfg.noise.scale,
                                  n_features=cfg.noise.n_rff_features,
                                  eps=cfg.noise.eps, device=device)
        print(f"Noise sampler: RFF D={cfg.noise.n_rff_features} "
              f"(scale={cfg.noise.scale}, N={N})")
        return sampler

    # Default: dense RBF Cholesky
    NoiseCls = RBFKernelAccel if cfg.training.use_accel_sampler else RBFKernel
    sampler = NoiseCls(mesh_points=mesh_pos, scale=cfg.noise.scale,
                       eps=cfg.noise.eps, device=device)
    print(f"Noise sampler: {NoiseCls.__name__} "
          f"(scale={cfg.noise.scale}, N={N})")
    return sampler


# ═══════════════════════════════════════════════════════════════════════════
# Pinball hierarchy helpers (from train_pinball.py)
# ═══════════════════════════════════════════════════════════════════════════

PINBALL_HIERARCHY_ORDER = [
    'orig', 'lc_250_kNN', 'lc_300_kNN', 'lc_500_kNN',
    'lc_1000_kNN', 'lc_2000_kNN', 'lc_4000_kNN',
]


def resolve_window(window_str, ms_base_dir):
    """Parse a window string into (ms_finest_dir, ms_res_dirs, ms_all_res_dirs).

    ms_finest_dir : path to the finest-level data directory
        - ``ms_base_dir`` itself when the finest level is ``orig``
        - ``ms_base_dir/<level>`` when the finest level is a kNN coarsening
    ms_res_dirs : list of coarse-level dir names used as hierarchy levels
    ms_all_res_dirs : full operator chain for composition
    """
    levels = [s.strip() for s in window_str.split(',') if s.strip()]
    if len(levels) < 2:
        raise ValueError(f"train_window must have at least 2 levels, got: '{window_str}'")
    finest_name = levels[0]
    coarse_names = levels[1:]
    if finest_name.endswith('_kNN'):
        ms_finest_dir = os.path.join(ms_base_dir, finest_name)
    else:
        ms_finest_dir = ms_base_dir
    try:
        start_idx = PINBALL_HIERARCHY_ORDER.index(coarse_names[0])
        end_idx = PINBALL_HIERARCHY_ORDER.index(coarse_names[-1])
    except ValueError as e:
        raise ValueError(f"Level not found in PINBALL_HIERARCHY_ORDER: {e}")
    ms_all_res_dirs = PINBALL_HIERARCHY_ORDER[start_idx:end_idx + 1]
    return ms_finest_dir, coarse_names, ms_all_res_dirs


def parse_multires_eval_windows(windows_str, ms_base_dir):
    """Parse semicolon-separated eval windows → list of spec dicts."""
    specs = []
    for w in windows_str.split(';'):
        w = w.strip()
        if not w:
            continue
        ms_finest_dir, ms_res_dirs, ms_all = resolve_window(w, ms_base_dir)
        specs.append({
            'name': w.replace(',', '-'),
            'ms_finest_dir': ms_finest_dir,
            'ms_res_dirs': ms_res_dirs,
            'ms_all_res_dirs': ms_all,
        })
    return specs


# ═══════════════════════════════════════════════════════════════════════════
# Multi-domain helpers
# ═══════════════════════════════════════════════════════════════════════════


def _domain_onehot(domain_name, batch_size, device):
    idx = DOMAIN_REGISTRY[domain_name]
    oh = torch.zeros(batch_size, N_DOMAINS, device=device)
    oh[:, idx] = 1.0
    return oh


def _trim_hierarchy(hier, data_n_nodes):
    """Trim hierarchy to start from the level matching data_n_nodes."""
    n_nodes = hier['n_nodes_list']
    if n_nodes[0] == data_n_nodes:
        return hier
    try:
        offset = n_nodes.index(data_n_nodes)
    except ValueError:
        raise ValueError(
            f"Data has {data_n_nodes} nodes but hierarchy levels are "
            f"{n_nodes} — no level matches."
        )
    # Look up xy/cells under either naming convention (load_domain_hierarchy
    # uses singular `xy`/`cells`; load_hierarchy_pt uses `xy_list`/`cells_list`).
    _xy = hier.get('xy_list', hier.get('xy', None))
    _cells = hier.get('cells_list', hier.get('cells', None))
    return {
        'edge_indices': hier['edge_indices'][offset:],
        'n_nodes_list': hier['n_nodes_list'][offset:],
        'pool_edges': hier['pool_edges'][offset:],
        'unpool_maps': hier['unpool_maps'][offset:],
        'centers': hier['centers'][offset:] if 'centers' in hier else None,
        'xy_list':    _xy[offset:]    if _xy    is not None else None,
        'cells_list': _cells[offset:] if _cells is not None else None,
    }


def _setup_hierarchy(model, hier, cfg):
    """Call set_mesh_hierarchy with optional LapPE and per-level triangulation."""
    centers = hier.get('centers')
    coarse_coords = centers[-1] if centers else None
    level_coords = centers if centers else None
    cells_list = hier.get('cells_list', None)
    xy_list = hier.get('xy_list', None)
    lap_pe = None
    model.set_mesh_hierarchy(
        edge_indices=hier['edge_indices'], n_nodes_list=hier['n_nodes_list'],
        pool_edges=hier['pool_edges'], unpool_maps=hier['unpool_maps'],
        coarse_coords=coarse_coords, level_coords=level_coords, lap_pe=lap_pe,
        cells_list=cells_list, xy_list=xy_list,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════

def _sync(device):
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'xpu':
        torch.xpu.synchronize()


def _save_ckpt(state_dict, path):
    try:
        torch.save(state_dict, path)
    except Exception as e:
        print(f"\n  WARNING: checkpoint save failed ({path}): {e}")


def _build_pinball_pos(pos, batch_size, num_points, cfg, dataset,
                       mu_batch=None, tidx_batch=None):
    """Build pos_batch with optional mu + time conditioning for pinball."""
    pos_batch = pos.unsqueeze(0).expand(batch_size, -1, -1)
    cond_parts = [pos_batch]
    if cfg.data.enc_use_mu and mu_batch is not None:
        cond_parts.append(mu_batch.unsqueeze(1).expand(-1, num_points, -1))
    if cfg.data.enc_use_time and tidx_batch is not None:
        t_phys = (tidx_batch / (dataset.n_times - 1)).unsqueeze(1).unsqueeze(2)
        cond_parts.append(t_phys.expand(-1, num_points, 1))
    if len(cond_parts) > 1:
        return torch.cat(cond_parts, dim=-1)
    return pos_batch


# ═══════════════════════════════════════════════════════════════════════════
# Main training function
# ═══════════════════════════════════════════════════════════════════════════

def train(cfg: Config, resume_run_id: str = None):
    # ---- Device ---- resolve once, store back in cfg as the single source of truth
    cfg.training.device = get_device(cfg.training.device)
    device = cfg.training.device
    print(f"Training on device: {device}")
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.structured(cfg)
    print(OmegaConf.to_yaml(cfg))

    # ---- Mode flags ----
    _domain = cfg.data.domain
    _is_pinball = (_domain == 'pinball')
    _is_sst = (_domain == 'sst')
    _is_multidomain = (_domain == 'multidomain')
    _is_single = not _is_pinball and not _is_multidomain and not _is_sst

    # ═══════════════════════════════════════════════════════════════════
    # Data loading (mode-specific)
    # ═══════════════════════════════════════════════════════════════════

    if _is_multidomain:
        domains_str = getattr(cfg.data, 'domains', '')
        if domains_str:
            domains = [d.strip() for d in domains_str.split(',') if d.strip()]
        else:
            domains = list(DOMAIN_REGISTRY.keys())
        print(f"Domains ({len(domains)}): {domains}")

        # Per-domain datasets
        domain_datasets, domain_dataloaders = {}, {}
        domain_mesh_pos, domain_tris, domain_plot_batches = {}, {}, {}
        for dname in domains:
            path = DOMAIN_FILES[dname]
            if not path.exists():
                raise FileNotFoundError(f"Dataset missing: {path}")
            ds = CachedDataset(str(path))
            # Cap samples per domain for balanced training (e.g. square=10k vs others=1k)
            _max = getattr(cfg.data, 'max_samples_per_domain', 0)
            if _max > 0 and len(ds) > _max:
                ds.samples = ds.samples[:_max]
                print(f"  {dname}: subsampled to {_max} (was {len(ds) + _max - _max})")
            domain_datasets[dname] = ds
            domain_dataloaders[dname] = DataLoader(
                ds, batch_size=cfg.training.batch_size, shuffle=True,
                num_workers=0, drop_last=True)
            mp, xy, cells, ei = ds.get_mesh_info()
            domain_mesh_pos[dname] = torch.from_numpy(mp).float().to(device)
            if xy is not None and cells is not None:
                domain_tris[dname] = Triangulation(xy[:, 0], xy[:, 1], cells)
            n_plot = min(4, len(ds))
            domain_plot_batches[dname] = torch.stack(
                [ds[i] for i in range(n_plot)]).to(device)

        # Hierarchies (trimmed to data resolution, then aligned to same n_levels)
        print("Loading mesh hierarchies...")
        domain_hiers = load_all_hierarchies(domains)
        for dname in domains:
            data_n = domain_datasets[dname].samples.shape[-1]
            domain_hiers[dname] = _trim_hierarchy(domain_hiers[dname], data_n)
        # All domains must have the same n_levels — truncate to the minimum
        min_levels = min(len(h['n_nodes_list']) for h in domain_hiers.values())
        for dname in domains:
            h = domain_hiers[dname]
            if len(h['n_nodes_list']) > min_levels:
                _xy_h = h.get('xy_list', h.get('xy', None))
                _cells_h = h.get('cells_list', h.get('cells', None))
                domain_hiers[dname] = {
                    'edge_indices': h['edge_indices'][:min_levels],
                    'n_nodes_list': h['n_nodes_list'][:min_levels],
                    'pool_edges':   h['pool_edges'][:min_levels - 1],
                    'unpool_maps':  h['unpool_maps'][:min_levels - 1],
                    'centers':      h['centers'][:min_levels] if h.get('centers') else None,
                    'xy_list':      _xy_h[:min_levels]    if _xy_h    is not None else None,
                    'cells_list':   _cells_h[:min_levels] if _cells_h is not None else None,
                }
            print(f"  {dname}: {domain_hiers[dname]['n_nodes_list']}")

        # Per-domain noise samplers
        domain_noise = {d: _build_noise_sampler(domain_mesh_pos[d], cfg, device)
                        for d in domains}
        # Defaults for shared code below
        dataset = domain_datasets[domains[0]]
        pos = domain_mesh_pos[domains[0]]
        edge_index = domain_hiers[domains[0]]['edge_indices'][0].to(device)
        num_points = domain_hiers[domains[0]]['n_nodes_list'][0]
        tri = domain_tris.get(domains[0])

    elif _is_pinball:
        # ms_finest_dir: resolved from train_window, or fall back to ms_base_dir
        _pinball_data_dir = (cfg.data.ms_finest_dir
                             or cfg.data.ms_base_dir
                             or cfg.data.data_dir)
        dataset = PinballDataset(_pinball_data_dir, num_samples=cfg.data.num_samples)
        _pinball_dir_for_splits = _pinball_data_dir
        mesh_pos, mesh_coords_t, edge_index = dataset.get_mesh_info()
        xy = mesh_pos
        num_points = len(mesh_pos)
        tri = None
        pos = torch.from_numpy(mesh_pos).float().to(device)
        edge_index = edge_index.to(device)
        print(f"Mesh: {num_points} vertices, {edge_index.shape[1]} edges")

    elif _is_sst:
        raise NotImplementedError(
            "The SST domain is not included in this GRIFDIR release."
        )

    else:  # single conductivity domain
        if _domain == 'square':
            pat = f'conductivity_nx{cfg.data.nx}_ny{cfg.data.ny}_*.pt'
        else:
            pat = f'conductivity_{_domain}_*.pt'
        candidates = glob.glob(os.path.join(cfg.data.data_dir, pat))
        data_path = sorted(candidates)[-1] if candidates else None
        if data_path is not None:
            print(f"Loading cached data: {data_path}")
            dataset = CachedDataset(data_path, split='train')
            if len(dataset) > cfg.data.num_samples:
                dataset.samples = dataset.samples[:cfg.data.num_samples]
            _conductivity_data_path = data_path
        else:
            _conductivity_data_path = None
            print("Generating data on-the-fly (requires dolfinx)...")
            dataset = get_conductivity_dataloader(
                num_samples=cfg.data.num_samples, batch_size=cfg.training.batch_size,
                nx=cfg.data.nx, ny=cfg.data.ny,
                max_numInc=cfg.data.max_numInc, backCond=cfg.data.backCond,
                shuffle=True, num_workers=cfg.training.num_workers,
            )[1]
        mesh_pos, xy, cells, edge_index = dataset.get_mesh_info()
        num_points = len(mesh_pos)
        tri = Triangulation(xy[:, 0], xy[:, 1], cells)
        pos = torch.from_numpy(mesh_pos).float().to(device)
        edge_index = edge_index.to(device)
        print(f"Mesh: {num_points} cells, {edge_index.shape[1]} edges")

    # ──────────────────────────────────────────────────────────────────────
    # Held-out val/test datasets for DPS evaluation.
    # Unconditional samples remain on the training dataset (no target, so no
    # leakage).  DPS has a ground-truth target, so it MUST be on held-out data.
    # Multidomain currently doesn't build val/test — flagged as follow-up.
    # ──────────────────────────────────────────────────────────────────────
    val_dataset = None
    test_dataset = None
    try:
        if _is_pinball:
            val_dataset  = PinballDataset(_pinball_dir_for_splits, split='val')
            test_dataset = PinballDataset(_pinball_dir_for_splits, split='test')
        elif _is_sst:
            # Match PhySense convention: only train/test on disk.  Training-time
            # DPS eval falls back to train data (val_dataset=None); post-training
            # DPS uses the held-out test file.
            test_dataset = SSTDataset(_sst_data_dir, land_mask_path=_land_mask_path,
                                      split='test', knn_k=_knn_k)
        elif _is_single and _conductivity_data_path is not None:
            # No val split: runtime 90/10 train/test slice.  Training-time DPS
            # falls back to train data; post-training DPS uses test.
            test_dataset = CachedDataset(_conductivity_data_path, split='test')
    except Exception as _e:
        print(f"  Warning: could not build val/test datasets ({_e}); "
              f"DPS val/test will fall back to training data.")
        val_dataset = None
        test_dataset = None

    # Fall back to training dataset if val/test construction failed or not
    # implemented for this branch (multidomain, on-the-fly conductivity).
    _dps_val_dataset  = val_dataset  if val_dataset  is not None else dataset
    _dps_test_dataset = test_dataset if test_dataset is not None else dataset
    if val_dataset is not None:
        print(f"  Val dataset: {len(val_dataset)} samples")
    if test_dataset is not None:
        print(f"  Test dataset: {len(test_dataset)} samples")

    # DataLoader for single-domain modes
    if not _is_multidomain:
        dataloader = DataLoader(dataset, batch_size=cfg.training.batch_size,
                                shuffle=True, num_workers=cfg.training.num_workers,
                                pin_memory=cfg.training.pin_memory,
                                persistent_workers=(cfg.training.persistent_workers
                                                    and cfg.training.num_workers > 0))

    # Area-weighted loss (single conductivity domains only).
    # If fem_lumped_mass=True we override this below (after model construction)
    # with model._level_node_areas[0] so the loss inner product is the
    # discrete L²(α) inner product, consistent with the lumped FE-Galerkin
    # aggregation used inside the convolution layer.
    node_areas = None
    if _is_single and cfg.training.area_weighted_loss:
        xy_t = torch.from_numpy(xy).float()
        cells_np = dataset.get_mesh_info()[2]
        if cells_np is not None:
            v0, v1, v2 = xy_t[cells_np[:, 0]], xy_t[cells_np[:, 1]], xy_t[cells_np[:, 2]]
            tri_areas = 0.5 * torch.abs(
                (v1 - v0)[:, 0] * (v2 - v0)[:, 1] -
                (v1 - v0)[:, 1] * (v2 - v0)[:, 0])
            node_areas = tri_areas / tri_areas.sum()

    # ═══════════════════════════════════════════════════════════════════
    # Wandb
    # ═══════════════════════════════════════════════════════════════════

    flat_config = OmegaConf.to_container(cfg, resolve=True)
    for section in ('model', 'sde', 'noise', 'data', 'training'):
        if section in flat_config and isinstance(flat_config[section], dict):
            for k, v in flat_config[section].items():
                if k not in flat_config:
                    flat_config[k] = v
    flat_config['slurm_job_id'] = os.environ.get('SLURM_JOB_ID', None)
    flat_config['slurm_agent_num'] = os.environ.get('SLURM_SWEEP_AGENT_NUM', None)
    flat_config['hostname'] = socket.gethostname()

    wandb_kwargs = {
        "project": cfg.training.wandb_project,
        "entity": cfg.training.wandb_entity,
        "config": flat_config,
        "mode": "online" if cfg.training.use_wandb else "disabled",
    }
    # Only touch disk for wandb when it's actually enabled (no empty exp/.../wandb otherwise).
    if cfg.training.use_wandb:
        wandb_log_dir = os.path.join(cfg.training.save_dir, "wandb")
        os.makedirs(wandb_log_dir, exist_ok=True)
        wandb_kwargs["settings"] = wandb.Settings(code_dir=wandb_log_dir)
        wandb_kwargs["dir"] = wandb_log_dir
    if resume_run_id is not None:
        wandb_kwargs["id"] = resume_run_id
        wandb_kwargs["resume"] = "allow"

    with wandb.init(**wandb_kwargs) as run:
        # ---- Log dir ----
        name = f"{cfg.training.wandb_project}_{datetime.now().strftime('%y%m%d')}_{run.id}"
        log_dir = os.path.join(cfg.training.save_dir, f"conv={cfg.model.conv_type}", name)
        os.makedirs(log_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(log_dir, "config.yaml"))
        print(f"Saving model to: {log_dir}")

        # ═══════════════════════════════════════════════════════════════
        # Model
        # ═══════════════════════════════════════════════════════════════

        if _is_pinball:
            _n_cond = 0
            if cfg.data.enc_use_mu:
                _n_cond += dataset.n_params
            if cfg.data.enc_use_time:
                _n_cond += 1
            cfg.data.n_cond_channels = _n_cond

        if _is_multidomain:
            first_hier = domain_hiers[domains[0]]
            _layer_type = cfg.model.layer_type
            if _layer_type == 'gaot':
                # Standalone GAOT — domain identity threaded via domain_onehot kwarg
                # in forward (concat into per-node features alongside time embedding).
                from models.gaot import GAOTScoreNetwork
                _lts = getattr(cfg.model, 'gaot_latent_tokens_size', [32, 32])
                if isinstance(_lts, str):
                    _lts = [int(x) for x in _lts.split(',') if x.strip()]
                else:
                    _lts = list(_lts)
                model = GAOTScoreNetwork(
                    input_dim=1, pos_dim=2,
                    hidden_dim=cfg.model.hidden_dim, time_dim=cfg.model.time_dim,
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
                    n_domains=cfg.model.n_domains,
                ).to(device)
            else:
                # Default: MultiscaleGNNScoreNetwork (fem_conv, simple_mp, ...).
                from models.multiscale import MultiscaleGNNScoreNetwork
                model = MultiscaleGNNScoreNetwork(
                    input_dim=1, pos_dim=2,
                    hidden_dim=cfg.model.hidden_dim, time_dim=cfg.model.time_dim,
                    n_gnn_layers_per_level=cfg.model.num_layers,
                    n_levels=len(first_hier['n_nodes_list']),
                    use_latent_transformer=cfg.model.use_latent_transformer,
                    n_transformer_blocks=cfg.model.n_transformer_blocks,
                    n_transformer_heads=cfg.model.n_transformer_heads,
                    pooling_type=cfg.model.pooling_type,
                    use_pos_reinject=cfg.model.use_pos_reinject,
                    use_edge_geom=cfg.model.use_edge_geom,
                    layer_type=_layer_type,
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
            _setup_hierarchy(model, first_hier, cfg)
        else:
            model = get_model(cfg, edge_index, num_points, device)

        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        if hasattr(model, 'domain_encoder_head') and model.domain_encoder_head is not None:
            print(f"  Domain encoder head: {sum(p.numel() for p in model.domain_encoder_head.parameters()):,}")
            print(f"  Domain decoder head: {sum(p.numel() for p in model.domain_decoder_head.parameters()):,}")

        # Override node_areas with the model's lumped α when fem_lumped_mass=True,
        # so the area-weighted loss matches the convolution's quadrature
        # (DG-0 per-triangle for square / P1 per-vertex for pinball — handled
        # by the model's dispatch). Pre-move to device once to avoid per-step
        # `.to(device)` overhead in the loss.
        if (cfg.training.area_weighted_loss
                and getattr(cfg.model, 'fem_lumped_mass', False)
                and hasattr(model, '_level_node_areas')
                and model._level_node_areas):
            _alpha = model._level_node_areas[0].detach().clone()
            node_areas = (_alpha / _alpha.sum()).to(device)
            print(f"area_weighted_loss: using lumped-mass α from model "
                  f"(N={node_areas.shape[0]}, sum=1.0)")
        elif node_areas is not None:
            # Pre-move the per-triangle (DG-0) node_areas to device once.
            node_areas = node_areas.to(device)

        # ═══════════════════════════════════════════════════════════════
        # SDE + noise + score model
        # ═══════════════════════════════════════════════════════════════

        sde_type = cfg.sde.sde_type
        if sde_type == "vp":
            sde = OU(beta_min=cfg.sde.beta_min, beta_max=cfg.sde.beta_max)
        elif sde_type == "vp_cosine":
            sde = CosineOU()
        elif sde_type == "ve":
            sde = None
        else:
            raise ValueError(f"Unknown sde_type '{sde_type}'")
        print(f"SDE type: {sde_type}")

        if not _is_multidomain:
            noise_sampler = _build_noise_sampler(pos, cfg, device)
        else:
            noise_sampler = domain_noise[domains[0]]

        if sde_type == "ve":
            score_model = EDMDenoiser(model, noise_sampler, cfg)
        else:
            score_model = ScoreModel(model, sde, noise_sampler, cfg)

        # ═══════════════════════════════════════════════════════════════
        # Optimizer + EMA
        # ═══════════════════════════════════════════════════════════════

        ema = EMA(score_model.model, beta=0.999, update_after_step=500, update_every=10)
        optimizer = torch.optim.AdamW(score_model.model.parameters(), lr=cfg.training.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.training.num_epochs, eta_min=1e-6)

        # ═══════════════════════════════════════════════════════════════
        # Resume from checkpoint
        # ═══════════════════════════════════════════════════════════════

        start_epoch = 0
        if resume_run_id is not None:
            candidates = [
                d for d in glob.glob(
                    os.path.join(cfg.training.save_dir, "conv=*", f"*_{resume_run_id}"))
                if os.path.exists(os.path.join(d, "training_state_latest.pt"))
            ]
            if candidates:
                log_dir = candidates[0]
            resume_path = os.path.join(log_dir, "training_state_latest.pt")
            if os.path.exists(resume_path):
                ckpt = torch.load(resume_path, map_location=device)
                score_model.model.load_state_dict(ckpt["model"])
                ema.ema_model.load_state_dict(ckpt["ema"])
                optimizer.load_state_dict(ckpt["optimizer"])
                lr_scheduler.load_state_dict(ckpt["scheduler"])
                ema.step = ckpt.get("ema_step", ema.step)
                start_epoch = ckpt["epoch"] + 1
                print(f"Resumed from epoch {start_epoch}")
            else:
                raise FileNotFoundError(f"Cannot resume: no checkpoint at {resume_path}")

        # AMP
        if cfg.training.use_amp and device != 'cpu':
            amp_ctx = torch.amp.autocast(device_type=device, dtype=torch.bfloat16)
        else:
            amp_ctx = contextlib.nullcontext()

        # Plot batch
        if _is_pinball or _is_sst:
            _pi = [dataset[i] for i in range(min(6, len(dataset)))]
            plot_batch = torch.stack([it[0] for it in _pi]).to(device)
            plot_mu = torch.stack([it[1] for it in _pi]).to(device)
            plot_tidx = torch.stack([it[2] for it in _pi]).to(device)
        elif _is_multidomain:
            pass  # per-domain plot batches already prepared
        else:
            plot_batch = torch.stack(
                [dataset[i] for i in range(min(6, len(dataset)))]).to(device)

        # ═══════════════════════════════════════════════════════════════
        # Training loop
        # ═══════════════════════════════════════════════════════════════

        if _is_multidomain:
            domain_iters = {d: itertools.cycle(dl) for d, dl in domain_dataloaders.items()}
            total_samples = sum(len(ds) for ds in domain_datasets.values())
            batches_per_epoch = total_samples // cfg.training.batch_size
            domain_cycle = itertools.cycle(domains)

        print(f"Starting training for {cfg.training.num_epochs} epochs (from {start_epoch})...")

        for epoch in range(start_epoch, cfg.training.num_epochs):
            score_model.model.train()
            epoch_losses = []
            domain_losses = {d: [] for d in domains} if _is_multidomain else {}

            if _is_multidomain:
                pbar = tqdm(range(batches_per_epoch),
                            desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}")
            else:
                pbar = tqdm(dataloader,
                            desc=f"Epoch {epoch+1}/{cfg.training.num_epochs}")

            for step_or_batch in pbar:
                # ---- Unpack batch (mode-dependent) ----
                if _is_multidomain:
                    dname = next(domain_cycle)
                    x0 = next(domain_iters[dname]).to(device)
                    batch_size = x0.shape[0]
                    _setup_hierarchy(model, domain_hiers[dname], cfg)
                    cur_noise = domain_noise[dname]
                    score_model.noise_sampler = cur_noise
                    cur_pos = domain_mesh_pos[dname]
                    pos_batch = cur_pos.unsqueeze(0).expand(batch_size, -1, -1)
                    domain_oh = _domain_onehot(dname, batch_size, device)
                elif _is_pinball:
                    x0, mu_batch, tidx_batch = step_or_batch
                    x0 = x0.to(device)
                    mu_batch = mu_batch.to(device)
                    tidx_batch = tidx_batch.to(device)
                    batch_size = x0.shape[0]
                    cur_noise = noise_sampler
                    pos_batch = _build_pinball_pos(pos, batch_size, num_points,
                                                  cfg, dataset, mu_batch, tidx_batch)
                    domain_oh = None
                elif _is_sst:
                    x0, _mu, _tidx = step_or_batch
                    x0 = x0.to(device)
                    batch_size = x0.shape[0]
                    cur_noise = noise_sampler
                    pos_batch = pos.unsqueeze(0).expand(batch_size, -1, -1)
                    domain_oh = None
                else:
                    x0 = step_or_batch.to(device)
                    batch_size = x0.shape[0]
                    cur_noise = noise_sampler
                    pos_batch = pos.unsqueeze(0).expand(batch_size, -1, -1)
                    domain_oh = None

                # ---- Training step (shared) ----
                optimizer.zero_grad()
                z = cur_noise.sample(batch_size).unsqueeze(1)

                with amp_ctx:
                    if sde_type == "ve":
                        rnd = torch.randn([batch_size, 1, 1], device=device)
                        sigma = (rnd * cfg.sde.P_std + cfg.sde.P_mean).exp()
                        x_noisy = x0 + sigma * z
                        x0_pred = score_model(x_noisy, sigma, pos_batch,
                                              domain_onehot=domain_oh)
                        sd = cfg.sde.sigma_data
                        weight = (sigma ** 2 + sd ** 2) / (sigma * sd) ** 2
                        per_node = weight * (x0_pred - x0) ** 2
                        if node_areas is not None:
                            loss = (per_node.squeeze(1) * node_areas).sum(-1).mean()
                        else:
                            loss = torch.mean(per_node)
                    else:
                        t = torch.rand(batch_size, device=device) * 0.999 + 0.001
                        mean_t = sde.mean_t(t, x0)
                        std_t = sde.std_t_scaling(t, x0)
                        xt = mean_t + std_t * z
                        score_pred, _ = score_model(xt, t, pos_batch,
                                                    domain_onehot=domain_oh)
                        residual = score_pred + z / std_t
                        residual_w = cur_noise.apply_L_inv(residual.squeeze(1))
                        if node_areas is not None:
                            loss = (residual_w ** 2 * node_areas).sum(-1).mean()
                        else:
                            loss = torch.mean(residual_w ** 2)

                if torch.isnan(loss):
                    raise ValueError("NaN loss")

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                ema.update()

                l = loss.item()
                epoch_losses.append(l)
                if _is_multidomain:
                    domain_losses[dname].append(l)
                    pbar.set_postfix(loss=f"{l:.4f}", domain=dname[:8])
                else:
                    pbar.set_postfix(loss=l)
                wandb.log({"train/loss": l})

            mean_loss = np.mean(epoch_losses)
            log_dict = {"train/epoch_loss": mean_loss,
                        "train/lr": lr_scheduler.get_last_lr()[0], "epoch": epoch}
            if _is_multidomain:
                for d in domains:
                    if domain_losses[d]:
                        log_dict[f"train/epoch_loss_{d}"] = np.mean(domain_losses[d])
            wandb.log(log_dict)
            lr_scheduler.step()

            # ════════════════════════════════════════════════════════════
            # Visualization
            # ════════════════════════════════════════════════════════════

            vis_every = cfg.training.vis_every
            if vis_every > 0 and (epoch + 1) % vis_every == 0:
                score_model.model.eval()
                with torch.no_grad():
                    if _is_multidomain:
                        for dname in domains:
                            if dname not in domain_tris:
                                continue
                            _setup_hierarchy(model, domain_hiers[dname], cfg)
                            score_model.noise_sampler = domain_noise[dname]
                            _tri = domain_tris[dname]
                            pb = domain_plot_batches[dname]
                            _pos = domain_mesh_pos[dname]
                            n_p = pb.shape[0]
                            d_oh = _domain_onehot(dname, n_p, device)
                            z_p = domain_noise[dname].sample(n_p).unsqueeze(1)
                            pp = _pos.unsqueeze(0).expand(n_p, -1, -1)

                            if sde_type == "ve":
                                sig = (torch.randn([n_p,1,1], device=device) * cfg.sde.P_std + cfg.sde.P_mean).exp()
                                noisy = pb + sig * z_p
                                pred = score_model(noisy, sig, pp, domain_onehot=d_oh)
                            else:
                                times = torch.rand(n_p, device=device) * 0.999 + 0.001
                                noisy = sde.mean_t(times, pb) + sde.std_t_scaling(times, pb) * z_p
                                _, pred = score_model(noisy, times, pp, domain_onehot=d_oh)

                            fig, axes = plt.subplots(3, n_p, figsize=(4*n_p, 9), squeeze=False)
                            for i in range(n_p):
                                for row, (v, title) in enumerate([
                                    (pb[i,0], "GT"), (noisy[i,0], "Noisy"), (pred[i,0], "Denoised")]):
                                    im = axes[row][i].tripcolor(_tri, v.cpu().numpy(), cmap='jet', shading='flat')
                                    axes[row][i].set_aspect('equal'); axes[row][i].axis('off')
                                    axes[row][i].set_title(title)
                                    fig.colorbar(im, ax=axes[row][i], fraction=0.046, pad=0.04)
                            fig.suptitle(dname)
                            plt.tight_layout()
                            wandb.log({f"val/denoised_{dname}": wandb.Image(fig)})
                            plt.close(fig)
                    else:
                        n_plot = plot_batch.shape[0]
                        z_plot = noise_sampler.sample(n_plot).unsqueeze(1)

                        if _is_pinball:
                            pos_plot = _build_pinball_pos(pos, n_plot, num_points,
                                                         cfg, dataset, plot_mu, plot_tidx)
                        else:
                            pos_plot = pos.unsqueeze(0).expand(n_plot, -1, -1)

                        if sde_type == "ve":
                            sig = (torch.randn([n_plot,1,1], device=device) * cfg.sde.P_std + cfg.sde.P_mean).exp()
                            noisy_batch = plot_batch + sig * z_plot
                            x0_pred = score_model(noisy_batch, sig, pos_plot)
                            noise_label = [f"σ={sig[i].item():.3f}" for i in range(n_plot)]
                        else:
                            times = torch.rand(n_plot, device=device) * 0.999 + 0.001
                            noisy_batch = sde.mean_t(times, plot_batch) + sde.std_t_scaling(times, plot_batch) * z_plot
                            _, x0_pred = score_model(noisy_batch, times, pos_plot)
                            noise_label = [f"t={times[i]:.3f}" for i in range(n_plot)]

                        fig, axes = plt.subplots(3, n_plot, figsize=(16, 6))
                        for idx in range(n_plot):
                            v_gt = plot_batch[idx, 0].cpu().numpy()
                            v_noisy = noisy_batch[idx, 0].cpu().numpy()
                            v_pred = x0_pred[idx, 0].cpu().numpy()
                            if _is_pinball or _is_sst:
                                _sx, _sy = (xy[:, 1], xy[:, 0]) if _is_sst else (xy[:, 0], xy[:, 1])
                                axes[0, idx].scatter(_sx, _sy, c=v_gt, cmap='jet', s=1)
                                axes[1, idx].scatter(_sx, _sy, c=v_noisy, cmap='jet', s=1)
                                axes[2, idx].scatter(_sx, _sy, c=v_pred, cmap='jet', s=1)
                            else:
                                axes[0, idx].tripcolor(tri, v_gt, cmap='Blues', shading='flat')
                                axes[1, idx].tripcolor(tri, v_noisy, cmap='Blues', shading='flat')
                                axes[2, idx].tripcolor(tri, v_pred, cmap='Blues', shading='flat')
                            axes[0, idx].set_aspect('equal'); axes[0, idx].set_title("GT"); axes[0, idx].axis("off")
                            axes[1, idx].set_aspect('equal'); axes[1, idx].set_title(f"Noisy ({noise_label[idx]})"); axes[1, idx].axis("off")
                            axes[2, idx].set_aspect('equal'); axes[2, idx].set_title("Denoised"); axes[2, idx].axis("off")
                        plt.tight_layout()
                        wandb.log({"val/denoised_examples": wandb.Image(fig)})
                        plt.close()

            # ════════════════════════════════════════════════════════════
            # Generative eval (unconditional + DPS) — single domain only
            # ════════════════════════════════════════════════════════════

            gen_every = cfg.training.gen_eval_every
            if gen_every > 0 and (epoch + 1) % gen_every == 0 and not _is_multidomain:
                score_model.model.eval()
                print(f"  Running generative eval (epoch {epoch+1})...")

                n_gen = 4
                gen_pos_batch = None
                if _is_pinball:
                    idxs = torch.randint(len(dataset), (n_gen,))
                    items = [dataset[i] for i in idxs]
                    gen_pos_batch = _build_pinball_pos(
                        pos, n_gen, num_points, cfg, dataset,
                        torch.stack([it[1] for it in items]).to(device),
                        torch.stack([it[2] for it in items]).to(device))

                if sde_type == "ve":
                    samples = sample_ve_heun(score_model, pos, n_samples=n_gen,
                                             n_steps=100, device=device,
                                             pos_batch=gen_pos_batch)
                else:
                    samples = sample_unconditional(score_model, pos, n_samples=n_gen,
                                                   n_steps=100, device=device,
                                                   pos_batch=gen_pos_batch)
                _shading = 'gouraud' if (_is_pinball or _is_sst) else 'flat'
                fig_uncond = plot_uncond_samples(samples, tri, shading=_shading,
                                                 xy=xy if (_is_pinball or _is_sst) else None)
                wandb.log({"val/uncond_samples": wandb.Image(fig_uncond), "epoch": epoch})
                plt.close(fig_uncond)

                # DPS target must be from held-out data (unconditional sampling
                # above can use training data — no target there).
                gt_idx = torch.randint(len(_dps_val_dataset), (1,)).item()
                if _is_pinball:
                    gt_item = _dps_val_dataset[gt_idx]
                    gt_field = gt_item[0].to(device)
                    dps_pos = _build_pinball_pos(
                        pos, 1, num_points, cfg, _dps_val_dataset,
                        gt_item[1].unsqueeze(0).to(device),
                        gt_item[2].unsqueeze(0).to(device))
                elif _is_sst:
                    gt_field = _dps_val_dataset[gt_idx][0].to(device)
                    dps_pos = None
                else:
                    gt_field = _dps_val_dataset[gt_idx].to(device)
                    dps_pos = None

                mesh_pos_np = pos.cpu().numpy() if (_is_pinball or _is_sst) else dataset.get_mesh_info()[0]
                n_dofs = pos.shape[0]
                forward_op = SparseSensorOperator(n_dofs=n_dofs, n_sensors=10, device=device)
                y_obs = forward_op.forward(gt_field.unsqueeze(0))
                if sde_type == "ve":
                    x_recon, metrics = run_dps_eval_ve(y_obs, 1, 
                        score_model, pos, mesh_pos_np, gt_field,
                        forward_op=forward_op, n_steps=100, guidance_weight=1.0,
                        device=device, pos_batch=dps_pos)
                else:
                    x_recon, metrics = run_dps_eval(
                        y_obs, 1, score_model, pos, mesh_pos_np,
                        forward_op=forward_op, gt_field=gt_field,
                        n_sensors=10, n_steps=100, guidance_weight=1.0,
                        device=device, pos_batch=dps_pos)
                fig_dps = plot_cond_eval(
                    gt_field[0].cpu().numpy(), x_recon[0, 0].numpy(),
                    forward_op, tri, metrics, mesh_pos=mesh_pos_np,
                    shading=_shading, xy=xy if (_is_pinball or _is_sst) else None)
                wandb.log({"val/dps_rel_l2": metrics["rel_l2"],
                           "val/dps_mse": metrics["mse"],
                           "val/dps_reconstruction": wandb.Image(fig_dps),
                           "epoch": epoch})
                plt.close(fig_dps)
                print(f"  Generative eval done — DPS rel-L2: {metrics['rel_l2']:.4f}")

            # ════════════════════════════════════════════════════════════
            # Checkpoints
            # ════════════════════════════════════════════════════════════

            ckpt_every = getattr(cfg.training, 'ckpt_every', 50)
            if (epoch + 1) % 50 == 0 or epoch == cfg.training.num_epochs - 1:
                _save_ckpt(model.state_dict(), os.path.join(log_dir, f"model_epoch_{epoch+1}.pt"))
                _save_ckpt(ema.ema_model.state_dict(), os.path.join(log_dir, f"model_ema_epoch_{epoch+1}.pt"))

            _save_ckpt(model.state_dict(), os.path.join(log_dir, "model_latest.pt"))
            _save_ckpt(ema.ema_model.state_dict(), os.path.join(log_dir, "model_ema_latest.pt"))
            _save_ckpt({
                "epoch": epoch,
                "model": model.state_dict(),
                "ema": ema.ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": lr_scheduler.state_dict(),
                "ema_step": ema.step,
            }, os.path.join(log_dir, "training_state_latest.pt"))

            print(f"Epoch {epoch+1}: mean_loss = {mean_loss:.6f}")

    # ════════════════════════════════════════════════════════════════════
    # Post-training multi-resolution eval
    # ════════════════════════════════════════════════════════════════════

    # Post-training pinball multires eval (needs explicit specs from --multires_eval_windows)
    _pinball_specs = getattr(cfg.training, 'multires_eval_specs', None)
    if args.eval_res_invar and _is_pinball and _pinball_specs:
        print("\n" + "=" * 60 + "\nPost-training pinball multi-resolution evaluation\n" + "=" * 60)
        ema_path = os.path.join(log_dir, "model_ema_latest.pt")
        if os.path.exists(ema_path):
            sd = torch.load(ema_path, map_location=device, weights_only=True)
            model.load_state_dict(sd, strict=False)
        model.eval()
        run_multires_eval_pinball(
            score_model=score_model, cfg=cfg, log_dir=log_dir,
            ms_base_dir=cfg.data.ms_base_dir,
            eval_hierarchies=_pinball_specs,
            n_samples=4, n_steps=100, device=device,
            use_wandb=cfg.training.use_wandb,
            epoch=cfg.training.num_epochs - 1, checkpoint_path=ema_path)

    # Post-training conductivity multires eval (auto-builds specs from domain)
    # run_eval_suite branches on model type: multiscale GNN uses hierarchy swap,
    # CNN/FNO use fixed-grid swap (square domain, same/finer/coarser).
    if args.eval_res_invar and not _is_pinball and not _is_multidomain and not _is_sst:
        print("\n" + "=" * 60 + "\nPost-training resolution invariance evaluation\n" + "=" * 60)
        run_eval_suite(
            run_dir=log_dir,
            checkpoint='model_ema_latest.pt',
            n_samples=4, n_steps=100,
            output_dir=os.path.join(log_dir, 'multires_eval'),
            device=device,
            use_wandb=cfg.training.use_wandb,
            test_dataset=_dps_test_dataset,
        )

    print("Training complete!")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified GRIFDIR training")
    parser.add_argument("--config", type=str, default=None)

    # Domain
    parser.add_argument('--domain', type=str, default='square',
                        help='square | l_shape | circle | x_shape | pinball | multidomain | ...')
    parser.add_argument('--land_mask_path', type=str, default='',
                        help='Path to land_mask_sealarge.pt (SST only)')
    parser.add_argument('--knn_k', type=int, default=0,
                        help='k for kNN graph construction (SST only, 0=use mesh edges)')
    parser.add_argument('--hierarchy_path', type=str, default='',
                        help='Explicit path to mesh hierarchy .pt (overrides domain lookup)')
    parser.add_argument('--domains', type=str, default='',
                        help='Multidomain: comma-separated domain names (empty=all)')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--time_dim', type=int, default=0,
                        help='Time embedding dim (0 = mirror hidden_dim)')
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--conv_type', type=str, default='multiscale',
                        choices=['gcn', 'gat', 'mp_pde', 'multiscale'])
    parser.add_argument('--use_latent_transformer', type=t_or_f, default=False)
    parser.add_argument('--n_transformer_blocks', type=int, default=4)
    parser.add_argument('--n_transformer_heads', type=int, default=4)
    parser.add_argument('--pooling_type', type=str, default='false',
                        choices=['false', 'learned_pool', 'learned_res'])
    parser.add_argument('--use_pos_reinject', type=t_or_f, default=False)
    parser.add_argument('--use_edge_geom', type=t_or_f, default=False)
    parser.add_argument('--layer_type', type=str, default='simple_mp',
                        choices=['simple_mp', 'fem_conv', 'knn_conv', 'cnn', 'fno', 'gaot'])
    parser.add_argument('--mixing_type', type=str, default='vector',
                        choices=['scalar', 'vector', 'lowrank'])
    parser.add_argument('--fem_basis_type', type=str, default='p1', choices=['p0', 'p1'])
    parser.add_argument('--fem_k_hops', type=int, default=2, choices=[1, 2, 3])
    parser.add_argument('--fem_use_radius', type=t_or_f, default=False)
    parser.add_argument('--fem_radius_mult', type=float, default=3.0)
    parser.add_argument('--fem_lumped_mass', type=t_or_f, default=False)
    parser.add_argument('--knn_radius_mult', type=float, default=3.0)
    parser.add_argument('--film_time_cond', type=t_or_f, default=False)
    parser.add_argument('--res_invariant', type=t_or_f, default=False)
    # FNO
    parser.add_argument('--fno_n_modes', type=int, default=16)
    parser.add_argument('--fno_n_layers', type=int, default=4)
    parser.add_argument('--fno_pos_enc', type=str, default='none',
                        choices=['none', 'grid'],
                        help="'grid' adds (x,y) coord channels — breaks translation invariance, helps non-periodic domains")
    parser.add_argument('--fno_pad', type=float, default=0.0,
                        help='Domain padding fraction before FFT to break periodicity (e.g. 0.125). 0.0 = no padding.')
    # CNN
    parser.add_argument('--cnn_double_conv', type=t_or_f, default=False)
    parser.add_argument('--cnn_residual', type=t_or_f, default=False)
    parser.add_argument('--cnn_bottleneck_attn', type=t_or_f, default=False)
    parser.add_argument('--cnn_strided_down', type=t_or_f, default=False)
    parser.add_argument('--cnn_dropout', type=float, default=0.0)
    # GAOT
    parser.add_argument('--gaot_latent_tokens_size', type=str, default='32,32',
                        help='Comma-separated latent grid [H,W] for GAOT (must divide by patch_size)')
    parser.add_argument('--gaot_patch_size', type=int, default=2)
    parser.add_argument('--gaot_n_transformer_layers', type=int, default=3)
    parser.add_argument('--gaot_magno_radius', type=float, default=0.05)
    parser.add_argument('--gaot_magno_hidden', type=int, default=64)
    parser.add_argument('--gaot_magno_mlp_layers', type=int, default=3)
    parser.add_argument('--gaot_magno_lifting', type=int, default=64)
    parser.add_argument('--gaot_positional_embedding', type=str, default='absolute',
                        choices=['absolute', 'rope'])
    parser.add_argument('--gaot_use_geoembed', type=t_or_f, default=True)
    parser.add_argument('--gaot_use_attention', type=t_or_f, default=True,
                        help='MAGNO cosine attention (paper default). XPU/CPU: uses our scatter_reduce segment_csr patch.')
    parser.add_argument('--gaot_use_torch_scatter', type=t_or_f, default=False)
    # Domain heads
    parser.add_argument('--use_domain_heads', type=t_or_f, default=False)
    parser.add_argument('--n_latent', type=int, default=100)
    parser.add_argument('--n_domains', type=int, default=N_DOMAINS)

    # SDE
    parser.add_argument('--sde_type', type=str, default='ve',
                        choices=['vp', 'vp_cosine', 've'])
    parser.add_argument('--model_type', type=str, default='C', choices=['RAW', 'C_SQRT', 'C'])
    parser.add_argument('--beta_max', type=float, default=15.0)
    parser.add_argument('--sigma_min', type=float, default=0.001)
    parser.add_argument('--sigma_max', type=float, default=40.0)
    parser.add_argument('--P_mean', type=float, default=-1.2)
    parser.add_argument('--P_std', type=float, default=1.2)
    parser.add_argument('--sigma_data', type=float, default=0.5)

    # Noise
    parser.add_argument('--noise_scale', type=float, default=0.1)
    parser.add_argument('--noise_eps', type=float, default=0.01)
    parser.add_argument('--noise_sampler_type', type=str, default='rbf',
                        choices=['rbf', 'rff'],
                        help="Noise sampler: 'rbf' (dense Cholesky, default) or "
                             "'rff' (Random Fourier Features, O(N*D), recommended for large meshes)")
    parser.add_argument('--n_rff_features', type=int, default=512,
                        help='Number of Random Fourier Features (only used with --noise_sampler_type=rff)')

    # Data
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--nx', type=int, default=32)
    parser.add_argument('--ny', type=int, default=32)
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--max_samples_per_domain', type=int, default=0,
                        help='Multidomain: cap each domain to N samples (0=no cap, e.g. 1000)')

    # Pinball
    parser.add_argument('--ms_base_dir', type=str, default='')
    parser.add_argument('--train_window', type=str, default='')
    parser.add_argument('--ms_res_dirs', type=str, default='')
    parser.add_argument('--ms_all_res_dirs', type=str, default='')
    parser.add_argument('--enc_use_mu', type=t_or_f, default=True)
    parser.add_argument('--enc_use_time', type=t_or_f, default=True)
    parser.add_argument('--multires_eval_windows', type=str, default='')

    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=2000)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save_dir', type=str, default='exp')
    parser.add_argument('--device', type=str, default='',
                        help="cuda / mps / xpu / cpu; empty = autodetect")
    parser.add_argument('--vis_every', type=int, default=50)
    parser.add_argument('--gen_eval_every', type=int, default=100)
    parser.add_argument('--area_weighted_loss', type=t_or_f, default=False)
    parser.add_argument('--use_amp', type=t_or_f, default=False)
    parser.add_argument('--use_accel_sampler', type=t_or_f, default=True)
    parser.add_argument('--ckpt_every', type=int, default=50)
    parser.add_argument('--eval_res_invar', type=t_or_f, default=False,
                        help='Run the heavy post-training resolution-invariance eval suite '
                             '(off by default; use evaluate.py to reproduce the res-invariance numbers)')

    # Wandb
    parser.add_argument('--use_wandb', action='store_true')
    parser.add_argument('--wandb_entity', type=str, default='')
    parser.add_argument('--wandb_project', type=str, default='grifdir')
    parser.add_argument('--wandb_sweep', action='store_true')

    # Resume
    parser.add_argument('--resume_run_id', type=str, default=None)

    args = parser.parse_args()

    # ---- Resume mode: load config from checkpoint ----
    if args.resume_run_id is not None and args.config is None:
        candidates = [
            d for d in glob.glob(
                os.path.join(args.save_dir, "conv=*", f"*_{args.resume_run_id}"))
            if os.path.exists(os.path.join(d, "training_state_latest.pt"))
        ]
        if candidates:
            args.config = os.path.join(candidates[0], "config.yaml")
            print(f"Resume: auto-loaded config from {args.config}")

    # ---- Build config ----
    if args.config is not None:
        cfg = OmegaConf.load(args.config)
        cfg = merge_config(cfg)
    else:
        cfg = OmegaConf.structured(Config())

    # ---- Override from CLI (only for non-resume, or selective resume overrides) ----
    if args.resume_run_id is None or args.config is None:
        cfg.model.hidden_dim = args.hidden_dim
        cfg.model.num_layers = args.num_layers
        cfg.model.conv_type = args.conv_type
        cfg.model.use_latent_transformer = args.use_latent_transformer
        cfg.model.n_transformer_blocks = args.n_transformer_blocks
        cfg.model.n_transformer_heads = args.n_transformer_heads
        cfg.model.pooling_type = args.pooling_type
        cfg.model.use_pos_reinject = args.use_pos_reinject
        cfg.model.use_edge_geom = args.use_edge_geom
        cfg.model.layer_type = args.layer_type
        cfg.model.mixing_type = args.mixing_type
        cfg.model.fem_basis_type = args.fem_basis_type
        cfg.model.fem_k_hops = args.fem_k_hops
        cfg.model.fem_use_radius = args.fem_use_radius
        cfg.model.fem_radius_mult = args.fem_radius_mult
        cfg.model.fem_lumped_mass = args.fem_lumped_mass
        cfg.model.knn_radius_mult = args.knn_radius_mult
        cfg.model.film_time_cond = args.film_time_cond
        cfg.model.res_invariant = args.res_invariant
        cfg.model.fno_n_modes = args.fno_n_modes
        cfg.model.fno_n_layers = args.fno_n_layers
        cfg.model.fno_pos_enc = args.fno_pos_enc
        cfg.model.fno_pad = args.fno_pad
        cfg.model.cnn_double_conv = args.cnn_double_conv
        cfg.model.cnn_residual = args.cnn_residual
        cfg.model.cnn_bottleneck_attn = args.cnn_bottleneck_attn
        cfg.model.cnn_strided_down = args.cnn_strided_down
        cfg.model.cnn_dropout = args.cnn_dropout
        # GAOT
        _gaot_lts = [int(x) for x in args.gaot_latent_tokens_size.split(',') if x.strip()]
        cfg.model.gaot_latent_tokens_size = _gaot_lts
        cfg.model.gaot_patch_size = args.gaot_patch_size
        cfg.model.gaot_n_transformer_layers = args.gaot_n_transformer_layers
        cfg.model.gaot_magno_radius = args.gaot_magno_radius
        cfg.model.gaot_magno_hidden = args.gaot_magno_hidden
        cfg.model.gaot_magno_mlp_layers = args.gaot_magno_mlp_layers
        cfg.model.gaot_magno_lifting = args.gaot_magno_lifting
        cfg.model.gaot_positional_embedding = args.gaot_positional_embedding
        cfg.model.gaot_use_geoembed = args.gaot_use_geoembed
        cfg.model.gaot_use_attention = args.gaot_use_attention
        cfg.model.gaot_use_torch_scatter = args.gaot_use_torch_scatter
        cfg.model.model_type = args.model_type
        cfg.model.use_domain_heads = args.use_domain_heads
        cfg.model.n_latent = args.n_latent
        cfg.model.n_domains = args.n_domains
        cfg.sde.sde_type = args.sde_type
        cfg.sde.beta_max = args.beta_max
        cfg.sde.sigma_min = args.sigma_min
        cfg.sde.sigma_max = args.sigma_max
        cfg.sde.P_mean = args.P_mean
        cfg.sde.P_std = args.P_std
        cfg.sde.sigma_data = args.sigma_data
        cfg.noise.scale = args.noise_scale
        cfg.noise.eps = args.noise_eps
        cfg.noise.sampler_type = args.noise_sampler_type
        cfg.noise.n_rff_features = args.n_rff_features
        cfg.data.nx = args.nx
        cfg.data.ny = args.ny
        cfg.data.num_samples = args.num_samples
        cfg.data.max_samples_per_domain = args.max_samples_per_domain
        cfg.data.domain = args.domain
        cfg.data.domains = args.domains
        cfg.data.enc_use_mu = args.enc_use_mu
        cfg.data.enc_use_time = args.enc_use_time
        cfg.data.land_mask_path = args.land_mask_path
        cfg.data.knn_k = args.knn_k
        cfg.data.hierarchy_path = args.hierarchy_path
        cfg.data.ms_base_dir = args.ms_base_dir
        cfg.data.ms_res_dirs = args.ms_res_dirs
        cfg.data.ms_all_res_dirs = args.ms_all_res_dirs
        cfg.training.batch_size = args.batch_size
        cfg.training.num_epochs = args.num_epochs
        cfg.training.lr = args.lr
        cfg.training.area_weighted_loss = args.area_weighted_loss
        cfg.training.vis_every = args.vis_every
        cfg.training.gen_eval_every = args.gen_eval_every
        cfg.training.use_amp = args.use_amp
        cfg.training.use_accel_sampler = args.use_accel_sampler
        cfg.training.ckpt_every = args.ckpt_every

    # Always allow path / wandb overrides (even on resume)
    cfg.data.data_dir = args.data_dir
    cfg.training.save_dir = args.save_dir
    cfg.training.device = args.device
    cfg.training.use_wandb = args.use_wandb
    cfg.training.wandb_entity = args.wandb_entity
    cfg.training.wandb_project = args.wandb_project

    # Pinball: resolve train_window → ms_finest_dir + operator chain
    if args.train_window.strip():
        _finest, _mr, _ma = resolve_window(args.train_window, cfg.data.ms_base_dir)
        cfg.data.ms_finest_dir = _finest
        cfg.data.ms_res_dirs = ','.join(_mr)
        cfg.data.ms_all_res_dirs = ','.join(_ma)

    # Multires eval specs (pinball)
    if args.multires_eval_windows.strip():
        cfg.training.multires_eval_specs = parse_multires_eval_windows(
            args.multires_eval_windows, cfg.data.ms_base_dir)
    elif not hasattr(cfg.training, 'multires_eval_specs'):
        cfg.training.multires_eval_specs = []

    # time_dim: explicit if --time_dim>0, else mirror hidden_dim
    if getattr(args, 'time_dim', 0) and args.time_dim > 0:
        cfg.model.time_dim = args.time_dim
    else:
        cfg.model.time_dim = cfg.model.hidden_dim

    train(cfg, resume_run_id=args.resume_run_id)
