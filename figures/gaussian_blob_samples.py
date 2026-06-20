"""

python sample_square.py --run_dir exp/res_invariant_train_260402_vrta2msl/
"""

import os
import copy
import argparse

import torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from matplotlib.tri import Triangulation
import numpy as np

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config, merge_config, get_device
from model_mesh_utils import get_model
from diffusion.precond import EDMDenoiser, ScoreModel
from diffusion.noise_accel import RBFKernel as RBFKernelAccel
from diffusion.noise import RBFKernel as RBFKernelRef
from diffusion.sde import OU, CosineOU
from data_utils import CachedDataset
from eval.sampling import sample_ve_heun
from eval.resolution_invariance import load_hierarchy_pt, swap_mesh_hierarchy, swap_grid
from data_tools.gaussian_blob_utils import gen_conductivity_on_mesh

def plot_uncond_samples(samples, tri, n_cols=4, shading='flat', xy=None):
    """
    Plot unconditional samples on the mesh.

    Args:
        samples: [B, 1, N] tensor (CPU)
        tri:     matplotlib.tri.Triangulation (None for scatter-based domains)
        n_cols:  columns in the grid
        shading: 'flat' (cell-based) or 'gouraud' (vertex-based)
        xy:      [N, 2] numpy coords (required when tri is None)

    Returns:
        fig
    """
    import math
    B = samples.shape[0]
    # Use square layout when B is a perfect square, else fall back to n_cols
    sqrt_B = int(math.isqrt(B))
    if sqrt_B * sqrt_B == B:
        n_cols = sqrt_B
    n_rows = math.ceil(B / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)
    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols, idx % n_cols]
        if idx < B:
            s = samples[idx, 0].numpy()
            if tri is not None:
                ax.tripcolor(tri, s, cmap='jet', shading=shading, vmin=0, vmax=1)
            else:
                ax.scatter(xy[:, 0], xy[:, 1], c=s, cmap='jet', s=2, alpha=0.9,
                           vmin=0, vmax=1)
            ax.set_aspect('equal')
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    return fig



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="exp/res_invariant_train_260402_vrta2msl/")
    parser.add_argument("--checkpoint", type=str, default="model_ema_latest.pt")
    parser.add_argument("--n_samples", type=int, default=8)
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = get_device(args.device)
    output_dir = args.output_dir or os.path.join(args.run_dir, "samples")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Device: {device}  |  Output: {output_dir}")

    # ── Load config ──────────────────────────────────────────────────────
    cfg = OmegaConf.load(os.path.join(args.run_dir, "config.yaml"))
    cfg = merge_config(cfg)

    # ── Mesh / dataset (load cached .pt; no dolfinx needed) ───────────────
    import glob as _glob
    _cached = _glob.glob(os.path.join(
        cfg.data.data_dir, f"conductivity_nx{cfg.data.nx}_ny{cfg.data.ny}_n*.pt"))
    if not _cached:
        raise FileNotFoundError(
            f"No cached conductivity data in {cfg.data.data_dir} for "
            f"nx={cfg.data.nx}, ny={cfg.data.ny} — download it (data_tools/download_data.py) "
            f"or regenerate with data_tools/generate_training_data.py")
    dataset = CachedDataset(sorted(_cached)[-1])
    mesh_pos, xy, cells, edge_index = dataset.get_mesh_info()

    print("mesh_pos:", mesh_pos.shape, "xy:", xy.shape, "cells:", cells.shape, "edge_index:", edge_index.shape)

    num_points = len(mesh_pos)
    pos = torch.from_numpy(mesh_pos).float().to(device)
    edge_index = edge_index.to(device)
    tri = Triangulation(xy[:, 0], xy[:, 1], cells)
    print(f"Mesh: {num_points} nodes, {edge_index.shape[1]} edges")

    # ── Model ────────────────────────────────────────────────────────────
    model = get_model(cfg, edge_index, num_points, device)

    ckpt_path = os.path.join(args.run_dir, args.checkpoint)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Some checkpoints (e.g. FNO) include a top-level '_metadata' key which
    # strict load_state_dict rejects as unexpected. Remove it when present.
    if isinstance(state_dict, dict):
        state_dict.pop('_metadata', None)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded {ckpt_path}  ({sum(p.numel() for p in model.parameters()):,} params)")

    # ── Noise sampler & EDM wrapper ──────────────────────────────────────
    noise_sampler = RBFKernelAccel(
        mesh_points=pos,
        scale=cfg.noise.scale,
        eps=cfg.noise.eps,
        device=device,
    )
    score_model = EDMDenoiser(model, noise_sampler, cfg)

    # ── Helper: sample (with sub-batching to avoid OOM), print stats and save
    def run_and_save(score_model, pos, tri, tag, batch_size=None):
        n = args.n_samples
        bs = batch_size or n
        print(f"\n── {tag}: sampling {n} samples ({args.n_steps} steps, sub-batch={bs})...")
        chunks = []
        with torch.no_grad():
            for start in range(0, n, bs):
                chunk_n = min(bs, n - start)
                torch.manual_seed(start)  # for reproducibility across runs
                chunk = sample_ve_heun(
                    score_model, pos,
                    n_samples=chunk_n,
                    n_steps=args.n_steps,
                    device=device,
                )
                chunks.append(chunk)
        samples = torch.cat(chunks, dim=0)
        print(f"   shape={samples.shape}  mean={samples.mean():.4f}  std={samples.std():.4f}")

        pt_path = os.path.join(output_dir, f"samples_{tag}.pt")
        torch.save(samples.cpu(), pt_path)
        print(f"   Saved raw samples → {pt_path}")

        fig = plot_uncond_samples(samples, tri, shading="flat")
        for ext in ("png", "pdf"):
            fig_path = os.path.join(output_dir, f"samples_{tag}.{ext}")
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"   Saved plot        → {fig_path}")
        plt.close(fig)

    # ── Training resolution (32×32) ───────────────────────────────────────
    run_and_save(score_model, pos, tri, tag="32x32")

    # Also plot a few ground-truth samples at the training resolution
    def _save_gt_samples_for_mesh(mesh_pos_np, tri_obj, tag):
        n_gt = 4
        gt_list = []
        for i in range(n_gt):
            np.random.seed(160 + i)
            arr = gen_conductivity_on_mesh(mesh_pos_np, max_numInc=cfg.data.max_numInc, backCond=cfg.data.backCond)
            # Ensure shape [1, 1, N] to match samples format [B, 1, N]
            gt_list.append(torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0))
        gt_samples = torch.cat(gt_list, dim=0)  # [B, 1, N]

        pt_path = os.path.join(output_dir, f"gt_samples_{tag}.pt")
        torch.save(gt_samples, pt_path)
        print(f"   Saved ground-truth samples → {pt_path}")

        fig = plot_uncond_samples(gt_samples, tri_obj, shading='flat')
        for ext in ("png", "pdf"):
            fig_path = os.path.join(output_dir, f"gt_samples_{tag}.{ext}")
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"   Saved GT plot      → {fig_path}")
        plt.close(fig)

    # mesh_pos is numpy array from dataset.get_mesh_info()
    _save_gt_samples_for_mesh(mesh_pos, tri, tag="32x32")

    # ── Double resolution (64×64) ─────────────────────────────────--------
    # Save model state so we can restore afterwards
    train_state = copy.deepcopy(model.state_dict())
    # Save training grid dims for fixed-grid models (CNN/FNO/GAOT)
    train_grid = (getattr(model, 'grid_h', None), getattr(model, 'grid_w', None))

    # Finer (64×64 = 8192-elt) mesh for resolution generalisation. Building meshes
    # needs dolfinx, so load the committed hierarchy instead — the same .pt the eval
    # suite uses for its "finer" spec.
    n_levels = getattr(model, 'n_levels', None) or 4
    nx_hi, ny_hi = cfg.data.nx * 2, cfg.data.ny * 2   # 64×64 — used in the saved-sample tag
    _repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"\nLoading high-res hierarchy (64×64, {n_levels} levels) from meshes/mesh_hierarchy_square_64.pt")
    hier_hi = load_hierarchy_pt(
        os.path.join(_repo, "meshes", "mesh_hierarchy_square_64.pt"), n_levels=n_levels)

    # Use appropriate swap function depending on model type
    if hasattr(model, 'n_levels'):
        swap_mesh_hierarchy(model, hier_hi, device)

        # New pos and noise sampler for the fine mesh
        pos_hi = hier_hi['centers'][0].to(device)
        noise_sampler_hi = RBFKernelAccel(
            mesh_points=pos_hi,
            scale=cfg.noise.scale,
            eps=cfg.noise.eps,
            device=device,
        )
        score_model_hi = EDMDenoiser(model, noise_sampler_hi, cfg)

        # Triangulation for the high-res mesh
        xy_hi    = hier_hi['xy_list'][0]
        cells_hi = hier_hi['cells_list'][0]
        tri_hi   = Triangulation(xy_hi[:, 0], xy_hi[:, 1], cells_hi)

    else:
        # Fixed-grid models (CNN/FNO): use grid swap helper which calls
        # model.set_mesh_hierarchy appropriately and adjusts grid params.
        print("Model appears fixed-grid (CNN/FNO). Swapping grid via swap_grid()")

        # Detect GAOT and align behavior with eval.resolution_invariance for GAOT only
        is_gaot = getattr(cfg.model, 'layer_type', '') == 'gaot' or hasattr(model, 'gaot')

        if is_gaot:
            # Reload filtered checkpoint before swapping to avoid stale/shape-mismatched buffers
            if os.path.exists(ckpt_path):
                sd = torch.load(ckpt_path, map_location=device, weights_only=False)
                if isinstance(sd, dict):
                    sd.pop('_metadata', None)
                current_sd = model.state_dict()
                filtered_sd = {k: v for k, v in sd.items() if k in current_sd and current_sd[k].shape == v.shape}
                model.load_state_dict(filtered_sd, strict=False)
                model.eval()

            # Restore train grid dims (helpful for GAOT/CNN adapters)
            if train_grid[0] is not None:
                model.grid_h, model.grid_w = train_grid

            swap_grid(model, hier_hi, device, mode='native')

            # New pos and noise sampler for the fine mesh — choose accel/ref per cfg
            pos_hi = hier_hi['centers'][0].to(device)
            use_accel = getattr(cfg.training, 'use_accel_sampler', True)
            NoiseCls = RBFKernelAccel if use_accel else RBFKernelRef
            noise_sampler_hi = NoiseCls(
                mesh_points=pos_hi,
                scale=cfg.noise.scale,
                eps=cfg.noise.eps,
                device=device,
            )

            # Wrap model depending on SDE type (EDM/VE uses EDMDenoiser; VP types use ScoreModel)
            sde_type = getattr(cfg.sde, 'sde_type', 've')
            if sde_type == 've':
                score_model_hi = EDMDenoiser(model, noise_sampler_hi, cfg)
            else:
                if sde_type == 'vp':
                    sde = OU(beta_min=cfg.sde.beta_min, beta_max=cfg.sde.beta_max)
                elif sde_type == 'vp_cosine':
                    sde = CosineOU()
                else:
                    sde = None
                score_model_hi = ScoreModel(model, sde, noise_sampler_hi, cfg)

            xy_hi    = hier_hi.get('xy_list')[0]
            cells_hi = hier_hi.get('cells_list')[0]
            tri_hi   = Triangulation(xy_hi[:, 0], xy_hi[:, 1], cells_hi)

        else:
            # Non-GAOT fixed-grid behavior unchanged (keep accel sampler + EDMDenoiser)
            swap_grid(model, hier_hi, device, mode='native')

            pos_hi = hier_hi['centers'][0].to(device)
            noise_sampler_hi = RBFKernelAccel(
                mesh_points=pos_hi,
                scale=cfg.noise.scale,
                eps=cfg.noise.eps,
                device=device,
            )
            score_model_hi = EDMDenoiser(model, noise_sampler_hi, cfg)

            xy_hi    = hier_hi.get('xy_list')[0]
            cells_hi = hier_hi.get('cells_list')[0]
            tri_hi   = Triangulation(xy_hi[:, 0], xy_hi[:, 1], cells_hi)

    # Also generate/save ground-truth samples on the high-res mesh
    mesh_pos_hi_np = hier_hi['centers'][0].cpu().numpy() if isinstance(hier_hi['centers'][0], torch.Tensor) else np.asarray(hier_hi['centers'][0])
    _save_gt_samples_for_mesh(mesh_pos_hi_np, tri_hi, tag=f"{nx_hi}x{ny_hi}")

    run_and_save(score_model_hi, pos_hi, tri_hi, tag=f"{nx_hi}x{ny_hi}", batch_size=1)

    # Restore model to training mesh (good practice if script is extended)
    # model.load_state_dict(train_state, strict=False)


if __name__ == "__main__":
    main()
