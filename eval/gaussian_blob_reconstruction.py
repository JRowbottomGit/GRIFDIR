"""
Gaussian-blob sparse-sensor reconstruction (square domain) — compute only.

Runs Fun-DPS / Fun-DAPS posterior sampling for a trained model and writes, into
<run_dir>/reconstruction/:
    {sampler}_reconstruction_NNN.pt   posterior samples [B, 1, N]
    {sampler}_gt_NNN.pt               ground-truth field [1, N]
    sensor_indices.pt                 sensor node indices
    {sampler}_summary.json            metrics (posterior RMSE, energy score, data consistency)

Render the paper figure from these with figures/gaussian_blob_reconstruction.py — the same
two-stage (compute → render) pattern used for the pinball reconstruction.
"""

import os
import json
import argparse

import torch
import numpy as np
from omegaconf import OmegaConf

from data_utils import CachedDataset
from diffusion.sde import OU, CosineOU
from diffusion.precond import ScoreModel, EDMDenoiser
from diffusion.noise import RBFKernel
from diffusion.noise_accel import RBFKernel as RBFKernelAccel
from eval.sampling import (
    run_dps_eval_ve,
    run_daps_eval_ve,
    compute_energy_score,
    compute_mean_data_consistency,
    compute_posterior_mean_rmse,
)
from config import merge_config, get_device
from model_mesh_utils import get_model
from pde_operators.poisson import PoissonOperator


def load_model(run_dir: str, checkpoint: str = "model_ema_latest.pt", device: str = "cuda"):
    """
    Load a trained model from a run directory.

    Parameters
    ----------
    run_dir : str
        Path to the training run directory (must contain config.yaml and checkpoint).
    checkpoint : str
        Checkpoint filename to load.
    device : str
        Device to load the model onto.

    Returns
    -------
    score_model : ScoreModel or EDMDenoiser
    cfg : Config
    dataset : CachedDataset
    noise_sampler : RBFKernel or RBFKernelAccel
    sde : OU or CosineOU or None
    """
    config_path = os.path.join(run_dir, "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found at {config_path}")

    cfg = OmegaConf.load(config_path)
    cfg = merge_config(cfg)

    # ---- Load dataset for mesh info (cached .pt; no dolfinx) ----
    import glob as _glob
    _cached = _glob.glob(os.path.join(
        cfg.data.data_dir, f"conductivity_nx{cfg.data.nx}_ny{cfg.data.ny}_n*.pt"))
    if not _cached:
        raise FileNotFoundError(
            f"No cached conductivity data in {cfg.data.data_dir} for "
            f"nx={cfg.data.nx}, ny={cfg.data.ny}")
    dataset = CachedDataset(sorted(_cached)[-1])

    mesh_pos, xy, cells, edge_index = dataset.get_mesh_info()
    num_points = len(mesh_pos)
    pos = torch.from_numpy(mesh_pos).float().to(device)
    edge_index = edge_index.to(device)

    print(f"Mesh: {num_points} cells, {edge_index.shape[1]} edges")

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

    return score_model, cfg, dataset, noise_sampler, sde


def run_reconstruction(
    score_model, cfg, dataset, pos, mesh_pos, forward_op, *,
    sampler="dps", n_samples=10, batch_size=10, n_steps=200,
    guidance_weight=1.0, sigma_y=0.01,
    daps_langevin_steps=20, daps_beta_y=1.0, daps_alpha=0.5,
    output_dir, device="cuda",
):
    """Reconstruct n_samples test fields from sparse observations; save samples + GT + metrics."""
    os.makedirs(output_dir, exist_ok=True)
    metrics = {"posterior_rmse": [], "energy_score": [], "data_consistency": []}
    max_gpu_batch = 25

    print(f"Running {sampler.upper()} reconstruction: {n_samples} fields, "
          f"{batch_size} posterior samples each, {n_steps} steps...")
    for i in range(n_samples):
        idx = len(dataset) - 1 - i        # use fields from the end of the dataset
        if idx < 0:
            idx = i
        gt_field = dataset[idx].to(device)                       # [1, N]
        y_obs = forward_op.forward(gt_field)                     # [1, n_sensors]
        y_obs = y_obs + sigma_y * torch.randn_like(y_obs)        # observation noise

        chunks = []
        remaining = batch_size
        while remaining > 0:
            chunk = min(remaining, max_gpu_batch)
            y_chunk = y_obs.repeat(chunk, 1).unsqueeze(1)        # [chunk, 1, n_sensors]
            if sampler == "daps":
                x_chunk, _ = run_daps_eval_ve(
                    y_obs=y_chunk, batch_size=chunk, edm_model=score_model,
                    pos=pos, mesh_pos=mesh_pos, gt_field=gt_field, forward_op=forward_op,
                    n_steps=n_steps, beta_y=daps_beta_y, alpha=daps_alpha,
                    langevin_steps=daps_langevin_steps, device=device,
                )
            else:
                x_chunk, _ = run_dps_eval_ve(
                    y_obs=y_chunk, batch_size=chunk, edm_model=score_model,
                    pos=pos, mesh_pos=mesh_pos, gt_field=gt_field, forward_op=forward_op,
                    n_steps=n_steps, guidance_weight=guidance_weight, device=device,
                    scale_grad=isinstance(forward_op, PoissonOperator),
                )
            chunks.append(x_chunk.detach().cpu())
            remaining -= chunk
        x_recon = torch.cat(chunks, dim=0).to(gt_field.device)   # [B, 1, N]

        y_obs_batch = y_obs.repeat(batch_size, 1).unsqueeze(1)
        y_pred_batch = forward_op.forward(x_recon)

        # ---- save (flat; same schema the pinball figure expects) ----
        torch.save(x_recon.cpu(), os.path.join(output_dir, f"{sampler}_reconstruction_{i:03d}.pt"))
        torch.save(gt_field.cpu(), os.path.join(output_dir, f"{sampler}_gt_{i:03d}.pt"))

        rmse = compute_posterior_mean_rmse(x_recon, gt_field)
        es   = compute_energy_score(x_recon, gt_field)
        dc   = compute_mean_data_consistency(y_pred_batch, y_obs_batch)
        metrics["posterior_rmse"].append(rmse)
        metrics["energy_score"].append(es)
        metrics["data_consistency"].append(dc)
        print(f"  [{i+1}/{n_samples}] idx={idx}  rmse={rmse:.4f}  es={es:.6f}  dc={dc:.6f}")

    # sensor indices, so the figure can overlay them without re-deriving
    if hasattr(forward_op, "sensor_indices"):
        torch.save(forward_op.sensor_indices.cpu(), os.path.join(output_dir, "sensor_indices.pt"))

    summary = {
        "sampler": sampler, "n_samples": n_samples, "n_steps": n_steps,
        "guidance_weight": guidance_weight, "sigma_y": sigma_y,
        "n_sensors": getattr(forward_op, "n_sensors", None),
        "posterior_rmse_mean": float(np.mean(metrics["posterior_rmse"])),
        "posterior_rmse_std":  float(np.std(metrics["posterior_rmse"])),
        "energy_score_mean":   float(np.mean(metrics["energy_score"])),
        "energy_score_std":    float(np.std(metrics["energy_score"])),
        "data_consistency_mean": float(np.mean(metrics["data_consistency"])),
        "data_consistency_std":  float(np.std(metrics["data_consistency"])),
    }
    with open(os.path.join(output_dir, f"{sampler}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved posterior samples + {sampler}_summary.json → {output_dir}")
    print(f"  posterior_rmse = {summary['posterior_rmse_mean']:.4f} "
          f"± {summary['posterior_rmse_std']:.4f}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Gaussian-blob sparse-sensor reconstruction (compute only; render with "
                    "figures/gaussian_blob_reconstruction.py).")
    parser.add_argument("--run_dir", type=str, required=True,
                        help="Training run / checkpoint dir (config.yaml + checkpoint).")
    parser.add_argument("--checkpoint", type=str, default="model_ema_latest.pt")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default: <run_dir>/reconstruction_n<n_sensors>")
    parser.add_argument("--task", type=str, default="sparse_sensors",
                        choices=["sparse_sensors", "poisson"])
    parser.add_argument("--sampler", type=str, default="dps", choices=["dps", "daps"])
    parser.add_argument("--n_samples", type=int, default=10,
                        help="Number of test fields to reconstruct")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Posterior samples per field")
    parser.add_argument("--n_steps", type=int, default=200)
    parser.add_argument("--n_sensors", type=int, default=20)
    parser.add_argument("--sigma_y", type=float, default=0.01, help="Observation noise")
    parser.add_argument("--dps_guidance_weight", type=float, default=1.0)
    parser.add_argument("--daps_langevin_steps", type=int, default=20)
    parser.add_argument("--daps_beta_y", type=float, default=1.0)
    parser.add_argument("--daps_alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="",
                        help="cuda / mps / xpu / cpu; empty = autodetect")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = get_device(args.device)
    print(f"Device: {device}")

    score_model, cfg, dataset, noise_sampler, sde = load_model(
        args.run_dir, checkpoint=args.checkpoint, device=device)

    mesh_pos, xy, cells, edge_index_np = dataset.get_mesh_info()
    pos = torch.from_numpy(mesh_pos).float().to(device)

    output_dir = args.output_dir or os.path.join(args.run_dir, f"reconstruction_n{args.n_sensors}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"SDE: {cfg.sde.sde_type}   Output: {output_dir}")

    if args.task == "sparse_sensors":
        from pde_operators.sensors import SparseSensorOperator
        forward_op = SparseSensorOperator(
            n_dofs=pos.shape[0], n_sensors=args.n_sensors, seed=args.seed, device=device)
        print(f"SparseSensorOperator: {args.n_sensors} sensors")
    elif args.task == "poisson":
        forward_op = PoissonOperator(mesh=dataset.domain, device=device)
        print("PoissonOperator")
    else:
        raise ValueError(f"Unknown task '{args.task}'")

    run_reconstruction(
        score_model, cfg, dataset, pos, mesh_pos, forward_op,
        sampler=args.sampler, n_samples=args.n_samples, batch_size=args.batch_size,
        n_steps=args.n_steps, guidance_weight=args.dps_guidance_weight, sigma_y=args.sigma_y,
        daps_langevin_steps=args.daps_langevin_steps, daps_beta_y=args.daps_beta_y,
        daps_alpha=args.daps_alpha, output_dir=output_dir, device=device,
    )
    print("\nRender the figure with:")
    print(f"  python figures/gaussian_blob_reconstruction.py {output_dir}")


if __name__ == "__main__":
    main()
