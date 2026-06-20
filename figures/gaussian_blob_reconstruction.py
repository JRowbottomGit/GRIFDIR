"""
Render the Gaussian-blob sparse-sensor reconstruction figure (paper Figs 5–6) from the
posterior samples written by eval/gaussian_blob_reconstruction.py.

Usage
-----
python figures/gaussian_blob_reconstruction.py <run_dir>/reconstruction [options]

Reuses the publication-ready single-row renderer shared with the pinball figure:
    Ground Truth (+ white sensors) | Posterior Mean | Posterior Std | Sample 1 … n
fields in jet, std in magma. One figure per reconstructed observation, written into the
same folder.
"""

import os
import sys
import argparse

import numpy as np
import torch
from matplotlib.tri import Triangulation
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import merge_config
from data_utils import CachedDataset
from pinball_reconstruction import load_all_samples, plot_per_observation


def square_triangulation(run_dir):
    """Build the training-mesh triangulation + node coords from the run's cached dataset."""
    cfg = merge_config(OmegaConf.load(os.path.join(run_dir, "config.yaml")))
    import glob
    cached = sorted(glob.glob(os.path.join(
        cfg.data.data_dir, f"conductivity_nx{cfg.data.nx}_ny{cfg.data.ny}_n*.pt")))
    if not cached:
        raise FileNotFoundError(
            f"No cached conductivity data in {cfg.data.data_dir} for "
            f"nx={cfg.data.nx}, ny={cfg.data.ny}")
    mesh_pos, xy, cells, _ = CachedDataset(cached[-1]).get_mesh_info()
    return Triangulation(xy[:, 0], xy[:, 1], cells), mesh_pos


def main():
    parser = argparse.ArgumentParser(
        description="Render Gaussian-blob reconstruction figures from saved posterior samples.")
    parser.add_argument("folder", type=str,
                        help="Reconstruction folder from eval/gaussian_blob_reconstruction.py "
                             "(e.g. checkpoints/gaussian_blob/reconstruction)")
    parser.add_argument("--run_dir", type=str, default=None,
                        help="Run dir holding config.yaml (default: the folder's parent)")
    parser.add_argument("--sampler", type=str, default=None,
                        help="Filter to 'dps' or 'daps' (default: all found)")
    parser.add_argument("--n_observations", type=int, default=None,
                        help="Max observations to render (default: all saved)")
    parser.add_argument("--n_show", type=int, default=3,
                        help="Max individual posterior samples to show per observation")
    parser.add_argument("--vmin", type=float, default=0.0, help="Field colormap min")
    parser.add_argument("--vmax", type=float, default=1.0, help="Field colormap max")
    args = parser.parse_args()

    folder = args.folder.rstrip("/")
    run_dir = args.run_dir or os.path.dirname(os.path.abspath(folder))
    tri, mesh_pos = square_triangulation(run_dir)

    # ---- Sensor locations (sensor indices are into the DOF/cell space = mesh_pos) ----
    sensor_idx_path = os.path.join(folder, "sensor_indices.pt")
    if os.path.exists(sensor_idx_path):
        sensor_indices = torch.load(sensor_idx_path, map_location="cpu", weights_only=True).numpy()
        sensor_coords = mesh_pos[sensor_indices]
        print(f"Loaded {len(sensor_indices)} sensor locations.")
    else:
        print("sensor_indices.pt not found — sensors will not be overlaid.")
        sensor_coords = np.empty((0, 2))

    # ---- Load posterior samples + render (square field is per-cell → flat shading) ----
    tensors_list, gt_list, paths = load_all_samples(folder, sampler=args.sampler)
    if args.n_observations is not None:
        tensors_list = tensors_list[:args.n_observations]
        paths = paths[:args.n_observations]
        if gt_list is not None:
            gt_list = gt_list[:args.n_observations]
    plot_per_observation(
        tensors_list, gt_list, paths, tri, sensor_coords,
        output_dir=folder, n_show=args.n_show, vmin=args.vmin, vmax=args.vmax,
        shading="flat",
    )


if __name__ == "__main__":
    main()
