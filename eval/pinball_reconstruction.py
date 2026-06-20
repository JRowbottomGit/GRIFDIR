"""
Evaluation script for trained diffusion models on the Pinball dataset.

Loads a model checkpoint and config from a training run directory,
then performs sparse-sensor conditional (DPS/DAPS) reconstruction on
Pinball flow snapshots.

Usage
-----
python eval/pinball_reconstruction.py \
    --run_dir exp/conv=multiscale/<run_name> \
    --data_dir data/Pinball \
    --n_sensors 20 \
    --n_samples 10 \
    --sampler dps
"""

import os
import json
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from omegaconf import OmegaConf

from data_utils import PinballDataset
from diffusion.sde import OU, CosineOU
from diffusion.precond import ScoreModel, EDMDenoiser
from diffusion.noise import RBFKernel
from diffusion.noise_accel import RBFKernel as RBFKernelAccel
from eval.sampling import (
    sample_ve_heun,
    run_dps_eval_ve,
    run_daps_eval_ve,
    plot_cond_eval,
    sample_unconditional,
)
from config import Config, merge_config, get_device
from model_mesh_utils import get_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_pinball_pos(pos, batch_size, num_points, cfg, dataset,
                       mu_batch=None, tidx_batch=None):
    """Build pos_batch with optional mu + time conditioning for pinball."""
    pos_batch = pos.unsqueeze(0).expand(batch_size, -1, -1)
    cond_parts = [pos_batch]
    if getattr(cfg.data, 'enc_use_mu', False) and mu_batch is not None:
        cond_parts.append(mu_batch.unsqueeze(1).expand(-1, num_points, -1))
    if getattr(cfg.data, 'enc_use_time', False) and tidx_batch is not None:
        t_phys = (tidx_batch / (dataset.n_times - 1)).unsqueeze(1).unsqueeze(2)
        cond_parts.append(t_phys.expand(-1, num_points, 1))
    if len(cond_parts) > 1:
        return torch.cat(cond_parts, dim=-1)
    return pos_batch


def load_triangulation(mesh_coords, edge_index) -> Triangulation:
    """
    Build the mesh triangulation from the cached primal edge_index — pure torch,
    no meshio / FEniCS-XML. Triangles are recovered by enumerating 3-cycles in the
    vertex adjacency graph (model_mesh_utils._cells_from_edges), which reproduces the
    Pinball_mesh.xml triangle set exactly in Pinball_mesh_coords.pt order.
    """
    from model_mesh_utils import _cells_from_edges
    coords = np.asarray(mesh_coords)
    cells = _cells_from_edges(edge_index.detach().cpu(), coords.shape[0]).numpy()
    return Triangulation(coords[:, 0], coords[:, 1], cells)




def compute_posterior_mean_rmse(x_recon: torch.Tensor, gt_field: torch.Tensor) -> float:
    """
    RMSE between the posterior mean (average over samples) and the ground truth.

    Parameters
    ----------
    x_recon  : [B, 1, N]  - posterior samples
    gt_field : [1, N]     - ground truth field

    Returns
    -------
    float : RMSE value
    """
    posterior_mean = x_recon.mean(dim=0)          # [1, N]
    rmse = torch.sqrt(((posterior_mean - gt_field) ** 2).mean())
    return rmse.item()


def compute_energy_score(x_recon: torch.Tensor, gt_field: torch.Tensor) -> float:
    """
    Energy Score:  ES = E||X_gt - X|| - 0.5 * E||X - X'||
    where X, X' are independent draws from the posterior.

    Parameters
    ----------
    x_recon  : [B, 1, N]  - posterior samples
    gt_field : [1, N]     - ground truth field

    Returns
    -------
    float : Energy Score (lower is better)
    """
    B = x_recon.shape[0]
    x = x_recon.squeeze(1)      # [B, N]
    y = gt_field.squeeze(0)     # [N]

    # E||X - y||
    term1 = torch.norm(x - y.unsqueeze(0), dim=-1).mean()

    # E||X - X'||  (all pairs, i != j)
    diff = x.unsqueeze(0) - x.unsqueeze(1)          # [B, B, N]
    pair_norms = torch.norm(diff, dim=-1)            # [B, B]
    # exclude diagonal
    mask = ~torch.eye(B, dtype=torch.bool, device=x.device)
    term2 = pair_norms[mask].mean()

    return (term1 - 0.5 * term2).item()


def compute_mean_data_consistency(y_pred_batch: torch.Tensor, y_obs_batch: torch.Tensor) -> float:
    """
    Mean data consistency: average MSE between predicted and observed measurements
    across posterior samples.

    Parameters
    ----------
    y_pred_batch : [B, 1, n_sensors]  - forward-operator applied to x_recon
    y_obs_batch  : [B, 1, n_sensors]  - (repeated) observations

    Returns
    -------
    float : mean data-consistency MSE
    """
    mse_per_sample = ((y_pred_batch - y_obs_batch) ** 2).mean(dim=-1).mean(dim=-1)  # [B]
    return mse_per_sample.mean().item()




# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(run_dir: str, checkpoint: str = "model_ema_latest.pt", device: str = "cuda",
               data_dir: str = None, split: str = "test"):
    """
    Load a trained model from a run directory.

    Returns
    -------
    score_model, cfg, dataset, noise_sampler, sde, pos, tri
    """
    config_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")

    cfg = OmegaConf.load(config_path)
    cfg = merge_config(cfg)

    # ---- Dataset ----
    _data_dir = data_dir or getattr(cfg.data, 'data_dir', 'data/Pinball')
    dataset = PinballDataset(_data_dir, split=split)

    mesh_pos = dataset.mesh_pos                      # numpy [N, 2]
    mesh_coords = dataset.mesh_coords.numpy()        # numpy [N, 2]
    edge_index = dataset.edge_index

    num_points = mesh_pos.shape[0]
    pos = torch.from_numpy(mesh_pos).float().to(device)
    edge_index = edge_index.to(device)

    print(f"Mesh: {num_points} nodes, {edge_index.shape[1]} edges")

    # ---- SDE ----
    sde_type = cfg.sde.sde_type
    if sde_type == "vp":
        sde = OU(beta_min=cfg.sde.beta_min, beta_max=cfg.sde.beta_max)
    elif sde_type == "vp_cosine":
        sde = CosineOU()
    elif sde_type == "ve":
        sde = None
    else:
        raise ValueError(f"Unknown sde_type '{sde_type}'")

    # ---- Noise sampler ----
    use_accel = getattr(cfg.training, "use_accel_sampler", True)
    NoiseCls = RBFKernelAccel if use_accel else RBFKernel
    noise_sampler = NoiseCls(
        mesh_points=pos,
        scale=cfg.noise.scale,
        eps=cfg.noise.eps,
        device=device,
    )

    # ---- Model ----
    # Patch cfg so get_model knows about pinball conditioning dims
    _n_cond = 0
    if getattr(cfg.data, 'enc_use_mu', False):
        _n_cond += dataset.n_params
    if getattr(cfg.data, 'enc_use_time', False):
        _n_cond += 1
    cfg.data.n_cond_channels = _n_cond

    model = get_model(cfg, edge_index, num_points, device)

    if sde_type == "ve":
        score_model = EDMDenoiser(model, noise_sampler, cfg)
    else:
        score_model = ScoreModel(model, sde, noise_sampler, cfg)

    # ---- Load checkpoint ----
    ckpt_path = os.path.join(run_dir, checkpoint)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    score_model.model.load_state_dict(state_dict)
    score_model.model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ---- Triangulation (pure torch, from the cached edge_index) ----
    tri = load_triangulation(mesh_coords, edge_index)

    return score_model, cfg, dataset, noise_sampler, sde, pos, tri


# ---------------------------------------------------------------------------
# Unconditional sampling
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_unconditional(
    score_model,
    cfg,
    dataset,
    pos,
    tri,
    n_samples: int = 16,
    n_steps: int = 200,
    output_dir: str = "eval_pinball_results",
    device: str = "cuda",
):
    """Generate and visualise unconditional samples."""
    os.makedirs(output_dir, exist_ok=True)
    sde_type = cfg.sde.sde_type

    num_points = pos.shape[0]
    # Build a neutral pos_batch (no mu/time conditioning for unconditional)
    pos_batch = pos.unsqueeze(0).expand(n_samples, -1, -1)

    print(f"Generating {n_samples} unconditional samples ({sde_type}, {n_steps} steps)...")
    if sde_type == "ve":
        samples = sample_ve_heun(
            score_model, pos, n_samples=n_samples, n_steps=n_steps,
            device=device, pos_batch=pos_batch,
        )
    else:
        samples = sample_unconditional(
            score_model, pos, n_samples=n_samples, n_steps=n_steps,
            device=device, pos_batch=pos_batch,
        )

    # Publication-ready 3x2 grid (ICML half-page, ~3.5 in wide)
    if tri is not None:
        vmin, vmax = 0.0, 3.0
        fig_uncond, axes = plt.subplots(2, 3, figsize=(3.5, 2.7))
        for ax, s in zip(axes.ravel(), samples[:6]):
            vals = s[0].cpu().numpy() if s.dim() == 2 else s.cpu().numpy()
            ax.tripcolor(tri, vals, shading="gouraud", cmap="jet",
                         vmin=vmin, vmax=vmax)
            ax.set_aspect("equal")
            ax.axis("off")
        fig_uncond.subplots_adjust(wspace=0.04, hspace=0.04,
                                   left=0.02, right=0.98,
                                   top=0.98, bottom=0.02)
        save_path = os.path.join(output_dir, "unconditional_samples.png")
        fig_uncond.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        save_path_pdf = os.path.join(output_dir, "unconditional_samples.pdf")
        fig_uncond.savefig(save_path_pdf, dpi=300, bbox_inches="tight", pad_inches=0.02)
        print(f"Saved unconditional samples to {save_path}")
        plt.close(fig_uncond)

    # Save raw samples
    samples_path = os.path.join(output_dir, "unconditional_samples.pt")
    torch.save(samples.cpu(), samples_path)
    print(f"Saved raw samples to {samples_path}")

    return samples


# ---------------------------------------------------------------------------
# Conditional (sparse-sensor) evaluation
# ---------------------------------------------------------------------------

def eval_conditional(
    score_model,
    cfg,
    dataset,
    pos,
    mesh_pos,
    tri,
    forward_op,
    batch_size: int = 10,
    n_samples: int = 10,
    n_steps: int = 200,
    guidance_weight: float = 1.0,
    sampler: str = "dps",
    daps_langevin_steps: int = 20,
    daps_langevin_alpha: float = 0.5,
    sigma_y: float = 0.01,
    output_dir: str = "eval_pinball_results",
    device: str = "cuda",
):
    """Run DPS/DAPS conditional reconstruction for Pinball sparse-sensor task."""
    os.makedirs(output_dir, exist_ok=True)

    num_points = pos.shape[0]
    all_metrics = {"posterior_rmse": [], "energy_score": [], "data_consistency": []}

    print(f"Running {sampler.upper()} on {n_samples} Pinball test samples, {n_steps} steps...")

    N_test = len(dataset)
    # Shuffle indices with a fixed seed so evaluation covers diverse
    # trajectories and time steps rather than a consecutive block.
    rng = np.random.RandomState(seed=42)
    shuffled_indices = rng.permutation(N_test).tolist()
    eval_indices = shuffled_indices[:n_samples]

    for i, idx in enumerate(eval_indices):
        field, mu, t_idx = dataset[idx]   # field: [1, N], mu: [n_params], t_idx: scalar

        gt_field = field.to(device)       # [1, N]
        mu_single = mu.to(device)         # [n_params]
        t_single = t_idx.to(device)       # scalar

        if sampler == "daps":
            max_gpu_batch = 50 
        else:
            max_gpu_batch = 5  # Adjust based on your GPU memory

        # Observations
        y_obs = forward_op.forward(gt_field)                            # [1, n_sensors]
        y_obs += sigma_y * torch.randn_like(y_obs)

        # Sample in chunks of max_gpu_batch if batch_size exceeds GPU memory
        x_recon_chunks = []
        remaining = batch_size
        while remaining > 0:
            chunk = min(remaining, max_gpu_batch)

            mu_batch = mu_single.unsqueeze(0).expand(chunk, -1)        # [chunk, n_params]
            tidx_batch = t_single.unsqueeze(0).expand(chunk)           # [chunk]
            pos_batch = _build_pinball_pos(
                pos, chunk, num_points, cfg, dataset,
                mu_batch=mu_batch, tidx_batch=tidx_batch,
            )

            y_obs_chunk = y_obs.repeat(chunk, 1).unsqueeze(1)          # [chunk, 1, n_sensors]

            if sampler == "daps":
                x_chunk, metrics = run_daps_eval_ve(
                    y_obs=y_obs_chunk,
                    batch_size=chunk,
                    edm_model=score_model,
                    pos=pos,
                    mesh_pos=mesh_pos,
                    gt_field=gt_field,
                    forward_op=forward_op,
                    n_steps=n_steps,
                    beta_y=sigma_y,
                    langevin_steps=daps_langevin_steps,
                    alpha=daps_langevin_alpha,
                    device=device,
                    pos_batch=pos_batch,
                )
            else:  # dps
                x_chunk, metrics = run_dps_eval_ve(
                    y_obs=y_obs_chunk,
                    batch_size=chunk,
                    edm_model=score_model,
                    pos=pos,
                    mesh_pos=mesh_pos,
                    gt_field=gt_field,
                    forward_op=forward_op,
                    n_steps=n_steps,
                    guidance_weight=guidance_weight,
                    device=device,
                    pos_batch=pos_batch,
                    guidance_schedule="constant"
                )

            x_recon_chunks.append(x_chunk.detach().cpu())
            remaining -= chunk

        x_recon = torch.cat(x_recon_chunks, dim=0).to(gt_field.device)  # [B, 1, N]
        y_obs_batch = y_obs.repeat(batch_size, 1).unsqueeze(1)           # [B, 1, n_sensors]
        y_pred_batch = forward_op.forward(x_recon)                       # [B, 1, n_sensors]


        posterior_rmse    = compute_posterior_mean_rmse(x_recon, gt_field)
        energy_score      = compute_energy_score(x_recon, gt_field)
        data_consistency  = compute_mean_data_consistency(y_pred_batch, y_obs_batch)


        all_metrics["posterior_rmse"].append(posterior_rmse)
        all_metrics["energy_score"].append(energy_score)
        all_metrics["data_consistency"].append(data_consistency)

        print(f"  Sample {i+1}/{n_samples} [dataset idx={idx}]: posterior_rmse={posterior_rmse:.4f}, "
              f"energy_score={energy_score:.6f}, data_consistency={data_consistency:.6f}  [traj t={int(t_idx.item())}]")

        # Save reconstruction tensor + ground truth
        recon_path = os.path.join(output_dir, f"{sampler}_reconstruction_{i:03d}.pt")
        torch.save(x_recon.cpu(), recon_path)
        gt_path = os.path.join(output_dir, f"{sampler}_gt_{i:03d}.pt")
        torch.save(gt_field.cpu(), gt_path)

        # Plot
        if tri is not None:
            fig = plot_cond_eval(
                gt_field[0].cpu().numpy(),
                x_recon.squeeze(1).cpu().numpy(),
                forward_op,
                tri,
                metrics,
                mesh_pos,
            )
            fig.savefig(
                os.path.join(output_dir, f"{sampler}_reconstruction_{i:03d}.png"),
                dpi=150, bbox_inches="tight",
            )
            plt.close(fig)

    # Summary
    mean_posterior_rmse = float(np.mean(all_metrics["posterior_rmse"]))
    std_posterior_rmse = float(np.std(all_metrics["posterior_rmse"]))
    mean_energy_score = float(np.mean(all_metrics["energy_score"]))
    std_energy_score = float(np.std(all_metrics["energy_score"]))
    mean_data_consistency = float(np.mean(all_metrics["data_consistency"]))
    std_data_consistency = float(np.std(all_metrics["data_consistency"]))


    summary = {
        "n_samples": n_samples,
        "n_steps": n_steps,
        "guidance_weight": guidance_weight,
        "sampler": sampler,
        "sigma_y": sigma_y,
        "posterior_rmse_mean": mean_posterior_rmse,
        "posterior_rmse_std":  std_posterior_rmse,
        "energy_score_mean": mean_energy_score,
        "energy_score_std":  std_energy_score,
        "data_consistency_mean": mean_data_consistency,
        "data_consistency_std": std_data_consistency,
    }

    print("\n" + "=" * 60)
    print("Sparse-Sensor Pinball Reconstruction Summary")
    print("=" * 60)
    print(f"  Samples evaluated:  {n_samples}")
    print(f"  Diffusion steps:    {n_steps}")
    print(f"  Posterior RMSE:     {mean_posterior_rmse:.4f} ± {std_posterior_rmse:.4f}")
    print(f"  Energy Score:      {mean_energy_score:.6f} ± {std_energy_score:.6f}")
    print(f"  Data Consistency:   {mean_data_consistency:.6f} ± {std_data_consistency:.6f}")
    print("=" * 60)

    summary_path = os.path.join(output_dir, f"{sampler}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    return all_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained diffusion model on Pinball (sparse sensors)")

    parser.add_argument("--run_dir", type=str, required=True,
                        help="Path to training run directory (contains config.yaml and checkpoints)")
    parser.add_argument("--data_dir", type=str, default="data/Pinball",
                        help="Path to Pinball data directory (default: Pinball)")
    parser.add_argument("--checkpoint", type=str, default="model_ema_latest.pt",
                        help="Checkpoint filename to load")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results (default: <run_dir>/eval.pinball)")

    parser.add_argument("--n_sensors", type=int, default=20,
                        help="Number of sparse sensor locations")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of test samples to reconstruct")
    parser.add_argument("--n_steps", type=int, default=200,
                        help="Number of diffusion steps")
    parser.add_argument("--batch_size", type=int, default=5,
                        help="Number of posterior samples per observation")
    parser.add_argument("--sigma_y", type=float, default=0.01,
                        help="Observation noise std (default: 0.01)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sensor placement")

    parser.add_argument("--sampler", type=str, default="dps", choices=["dps", "daps"],
                        help="Sampling method: dps or daps")
    parser.add_argument("--dps_guidance_weight", type=float, default=1.0,
                        help="DPS guidance weight")
    
    parser.add_argument("--daps_langevin_steps", type=int, default=20,
                        help="DAPS inner Langevin steps per sigma level")
    parser.add_argument("--daps_langevin_alpha", type=float, default=0.5,
                        help="DAPS inner Langevin step size decay exponent (eta_t = eta / (1 + iter^alpha))")

    parser.add_argument("--skip_unconditional", action="store_true",
                        help="Skip unconditional sampling")
    parser.add_argument("--skip_conditional", action="store_true",
                        help="Skip conditional evaluation")
    parser.add_argument("--n_uncond_samples", type=int, default=4,
                        help="Number of unconditional samples to generate")

    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Dataset split to evaluate on (default: test)")
    parser.add_argument("--device", type=str, default="",
                        help="cuda / mps / xpu / cpu; empty = autodetect")

    args = parser.parse_args()

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = get_device(args.device)
    print(f"Evaluating on device: {device}")

    # ---- Load model ----
    score_model, cfg, dataset, noise_sampler, sde, pos, tri = load_model(
        args.run_dir,
        checkpoint=args.checkpoint,
        device=device,
        data_dir=args.data_dir,
        split=args.split,
    )

    mesh_pos = dataset.mesh_pos  # numpy [N, 2]
    num_points = pos.shape[0]

    # ---- Output directory ----
    if args.output_dir:
        output_dir = args.output_dir
    else:
        task_part = f"sparse_sensors_n{args.n_sensors}"
        if args.sampler == "daps":
            hparams = (
                f"steps{args.n_steps}"
                f"_ls{args.daps_langevin_steps}"
                f"_alpha{args.daps_langevin_alpha}"
                f"_sigma_y{args.sigma_y}"
            )
        else:
            hparams = (
                f"steps{args.n_steps}"
                f"_gw{args.dps_guidance_weight}"
                f"_sigma_y{args.sigma_y}"
            )
        output_dir = os.path.join(args.run_dir, task_part, args.split, args.sampler, hparams)
    os.makedirs(output_dir, exist_ok=True)

    print(f"SDE type:         {cfg.sde.sde_type}")
    print(f"Output directory: {output_dir}")

    # ---- Forward operator ----
    from pde_operators.sensors import SparseSensorOperator
    forward_op = SparseSensorOperator(
        n_dofs=num_points,
        n_sensors=args.n_sensors,
        seed=args.seed,
        device=device,
    )
    print(f"SparseSensorOperator: {args.n_sensors} sensors / {num_points} nodes "
          f"({100 * args.n_sensors / num_points:.1f}%)")

    # Save sensor indices so visualise_pinball.py can reload them without
    # needing to know n_sensors / seed.
    sensor_idx_path = os.path.join(output_dir, "sensor_indices.pt")
    torch.save(forward_op.sensor_indices.cpu(), sensor_idx_path)
    print(f"Saved sensor indices to {sensor_idx_path}")

    # ---- Unconditional sampling ----
    if not args.skip_unconditional:
        eval_unconditional(
            score_model, cfg, dataset, pos, tri,
            n_samples=args.n_uncond_samples,
            n_steps=args.n_steps,
            output_dir=output_dir,
            device=device,
        )

    # ---- Conditional evaluation ----
    if not args.skip_conditional:
        eval_conditional(
            score_model, cfg, dataset, pos, mesh_pos, tri,
            forward_op=forward_op,
            batch_size=args.batch_size,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            guidance_weight=args.dps_guidance_weight,
            sampler=args.sampler,
            daps_langevin_steps=args.daps_langevin_steps,
            daps_langevin_alpha=args.daps_langevin_alpha,   
            sigma_y=args.sigma_y,
            output_dir=output_dir,
            device=device,
        )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

