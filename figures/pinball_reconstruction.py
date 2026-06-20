"""
Visualise saved posterior samples from eval/pinball_reconstruction.py.

Usage
-----
python figures/pinball_reconstruction.py <results_folder> [options]

The script scans <results_folder> for files matching the pattern
    {sampler}_reconstruction_???.pt
loads them, and produces:
  - mean field + pixel-wise std/variance maps
  - sensor location overlay
  - per-sample gallery
  - a summary JSON with basic statistics

All figures are written back into <results_folder>.

Example
-------
python figures/pinball_reconstruction.py \
    exp/conv=multiscale/my_run/sparse_sensors_n20/test/dps/steps200_gw1.0_sigma_y0.01 \
    --data_dir data/Pinball \
    --n_sensors 20 \
    --seed 42
"""

import argparse
import glob
import json
import os

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import torch
from matplotlib.colors import Normalize

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_utils import PinballDataset
from pde_operators.sensors import SparseSensorOperator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_triangulation(mesh_coords, edge_index):
    """Build the mesh triangulation from the cached primal edge_index — pure torch,
    no meshio / FEniCS-XML. Triangles are recovered with
    model_mesh_utils._cells_from_edges (reproduces the Pinball_mesh.xml triangle set)."""
    from model_mesh_utils import _cells_from_edges
    coords = np.asarray(mesh_coords)
    cells = _cells_from_edges(edge_index.detach().cpu(), coords.shape[0]).numpy()
    return mtri.Triangulation(coords[:, 0], coords[:, 1], cells)


def load_all_samples(folder: str, sampler: str = None):
    """
    Load all *_reconstruction_???.pt tensors from *folder*, together with
    matching *_gt_???.pt ground-truth tensors (if present).

    Returns
    -------
    tensors_list : list of Tensor [B, 1, N]  – posterior samples
    gt_list      : list of Tensor [1, N] | None  – ground truths (or None)
    paths        : list of str
    """
    if sampler:
        pattern = os.path.join(folder, f"{sampler}_reconstruction_???.pt")
    else:
        pattern = os.path.join(folder, "*_reconstruction_???.pt")

    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(
            f"No reconstruction .pt files found in {folder} "
            f"(pattern: {os.path.basename(pattern)})"
        )

    print(f"Found {len(paths)} reconstruction files.")
    tensors = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]

    gt_list = []
    for p in paths:
        gt_path = p.replace("_reconstruction_", "_gt_")
        if os.path.exists(gt_path):
            gt_list.append(torch.load(gt_path, map_location="cpu", weights_only=True))
        else:
            gt_list.append(None)

    has_gt = any(g is not None for g in gt_list)
    if not has_gt:
        gt_list = None
        print("No ground-truth files found; GT column will be skipped.")

    return tensors, gt_list, paths


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------

CMAP_FIELD = "jet" #"jet"
CMAP_STD   = "magma" #"plasma"
CMAP_ERR   = "RdBu_r"


def _tripcolor(ax, tri, vals, shading="gouraud", **kw):
    return ax.tripcolor(tri, vals, shading=shading, **kw)


def plot_mean_and_std(
    all_samples: torch.Tensor,        # [total_B, N]
    tri,
    sensor_coords: np.ndarray,        # [n_sensors, 2]
    output_dir: str,
    vmin: float = 0.0,
    vmax: float = 3.0,
):
    """
    Plot the posterior mean, pixel-wise std, and (optionally) variance
    across ALL loaded posterior samples.
    """
    mean_field = all_samples.mean(dim=0).numpy()   # [N]
    std_field  = all_samples.std(dim=0).numpy()    # [N]
    var_field  = std_field ** 2

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # ---- Mean ----
    ax = axes[0]
    pc = _tripcolor(ax, tri, mean_field, cmap=CMAP_FIELD, vmin=vmin, vmax=vmax)
    ax.scatter(sensor_coords[:, 0], sensor_coords[:, 1],
               c="white", s=20, edgecolors="black", linewidths=0.5, zorder=5,
               label="sensors")
    ax.set_title("Posterior Mean", fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)

    # ---- Std ----
    ax = axes[1]
    pc = _tripcolor(ax, tri, std_field, cmap=CMAP_STD)
    ax.scatter(sensor_coords[:, 0], sensor_coords[:, 1],
               c="white", s=20, edgecolors="black", linewidths=0.5, zorder=5)
    ax.set_title("Pixel-wise Std", fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)

    # ---- Variance ----
    ax = axes[2]
    pc = _tripcolor(ax, tri, var_field, cmap=CMAP_STD)
    ax.scatter(sensor_coords[:, 0], sensor_coords[:, 1],
               c="white", s=20, edgecolors="black", linewidths=0.5, zorder=5)
    ax.set_title("Pixel-wise Variance", fontsize=11)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Posterior statistics (all test samples)", fontsize=12, y=1.01)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = os.path.join(output_dir, f"posterior_statistics.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"Saved posterior_statistics.png/pdf to {output_dir}")
    plt.close(fig)

    return mean_field, std_field


def plot_per_observation(
    tensors_list,        # list of Tensor [B, 1, N]
    gt_list,             # list of Tensor [1, N] | None, or None
    paths,               # list of str
    tri,
    sensor_coords: np.ndarray,
    output_dir: str,
    n_show: int = 3,
    vmin: float = 0.0,
    vmax: float = 3.0,
    shading: str = "gouraud",
):
    """
    Publication-ready full-page figure (ICML two-column, 7 in wide).

    One row per observation, columns (left → right):
        Ground Truth | Posterior Mean | Posterior Std | Sample 1 | … | Sample n_show

    - GT, Mean and individual samples share one jet colorbar (fixed [vmin, vmax]).
    - Posterior Std uses a separate plasma colorbar.
    - Both colorbars run as vertical bars on the far right.
    - Sensor locations overlaid on every panel.
    - Column headers printed only on the first row.
    """
    n_obs   = len(tensors_list)
    has_gt  = gt_list is not None and any(g is not None for g in gt_list)

    # column layout: [GT?] | Mean | Std | Sample 1 … n_show
    col_labels = []
    if has_gt:
        col_labels.append("Ground Truth")
    col_labels += ["Posterior Mean", "Posterior Std"]
    col_labels += [f"Sample {k+1}" for k in range(n_show)]
    n_cols = len(col_labels)

    # ICML two-column width = 7.0 in; height per row scales with panel width
    fig_width  = 7.0
    #ax_w       = fig_width / n_cols
    #ax_h       = ax_w * 0.55                 # aspect ratio of the Pinball domain

    norm_field = Normalize(vmin=vmin, vmax=vmax)

    per_obs_dir = os.path.join(output_dir, "per_observation")
    os.makedirs(per_obs_dir, exist_ok=True)

    def _annotate(ax, vals, norm=None, cmap=CMAP_FIELD, plot_sensors=False):
        kw = dict(cmap=cmap, norm=norm) if norm is not None else dict(cmap=cmap)
        pc = _tripcolor(ax, tri, vals, shading=shading, **kw)
        if plot_sensors:
            ax.scatter(sensor_coords[:, 0], sensor_coords[:, 1],
                       c="white", s=5, edgecolors="black", linewidths=0.25, zorder=5)
        ax.set_aspect("equal")
        ax.axis("off")
        return pc

    def _add_cbar(fig, pc, ax, label=""):
        cb = fig.colorbar(pc, ax=ax, fraction=0.046, pad=0.03, shrink=0.6)
        cb.set_label(label, fontsize=5, labelpad=2)
        cb.ax.tick_params(labelsize=5)

    for row, (tensor, path) in enumerate(zip(tensors_list, paths)):
        tag     = os.path.splitext(os.path.basename(path))[0]
        samples = tensor.squeeze(1)          # [B, N]
        B       = samples.shape[0]
        show    = min(n_show, B)

        gt     = gt_list[row].squeeze(0).numpy() if (has_gt and gt_list[row] is not None) else None
        mean_f = samples.mean(dim=0).numpy()
        std_f  = samples.std(dim=0).numpy()

        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(fig_width, 3.75),
            squeeze=False,
        )
        axes = axes[0]   # shape [n_cols]

        col = 0

        # Ground truth
        if has_gt:
            pc = _annotate(axes[col], gt if gt is not None else mean_f,
                           norm=norm_field, plot_sensors=True)
            _add_cbar(fig, pc, axes[col])
            col += 1

        # Posterior mean
        pc = _annotate(axes[col], mean_f, norm=norm_field)
        _add_cbar(fig, pc, axes[col])
        col += 1

        # Posterior std
        pc = _annotate(axes[col], std_f, cmap=CMAP_STD)
        _add_cbar(fig, pc, axes[col])
        col += 1

        # Individual samples
        for k in range(show):
            vals = samples[k].numpy() if k < B else mean_f
            pc = _annotate(axes[col], vals, norm=norm_field)
            _add_cbar(fig, pc, axes[col])
            col += 1

        # Blank unused columns
        for c in range(col, n_cols):
            axes[c].axis("off")

        # Column headers
        for c, label in enumerate(col_labels):
            axes[c].set_title(label, fontsize=6, pad=3)

        fig.subplots_adjust(wspace=0.35, left=0.01, right=0.99, top=0.88, bottom=0.01)

        for ext in ("png", "pdf"):
            out = os.path.join(per_obs_dir, f"{tag}.{ext}")
            fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        print(f"Saved {tag}.png/pdf")

    print(f"Saved {n_obs} per-observation figures to {per_obs_dir}/")


def plot_sensor_locations(
    mesh_coords: np.ndarray,
    sensor_coords: np.ndarray,
    tri,
    output_dir: str,
):
    """Plot sensor locations on the mesh domain."""
    fig, ax = plt.subplots(figsize=(5, 4))
    if tri is not None:
        ax.triplot(tri, color="lightgray", linewidth=0.3, alpha=0.6)
    ax.scatter(mesh_coords[:, 0], mesh_coords[:, 1],
               c="steelblue", s=2, alpha=0.4, label="mesh nodes")
    ax.scatter(sensor_coords[:, 0], sensor_coords[:, 1],
               c="crimson", s=60, edgecolors="black", linewidths=0.6,
               zorder=5, label=f"sensors (n={len(sensor_coords)})")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Sensor locations", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(output_dir, f"sensor_locations.{ext}")
        fig.savefig(p, dpi=200, bbox_inches="tight")
    print(f"Saved sensor_locations.png/pdf to {output_dir}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualise saved Pinball posterior samples."
    )
    parser.add_argument("folder", type=str,
                        help="Results folder produced by eval/pinball_reconstruction.py")
    parser.add_argument("--data_dir", type=str, default="data/Pinball",
                        help="Path to Pinball data directory (for mesh)")
    parser.add_argument("--split", type=str, default="test",
                        choices=["train", "val", "test"])
    parser.add_argument("--n_sensors", type=int, default=20,
                        help="Number of sensors (must match eval run)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sensor placement (must match eval run)")
    parser.add_argument("--sampler", type=str, default=None,
                        help="Sampler prefix to filter files, e.g. 'dps' or 'daps'")
    parser.add_argument("--n_show", type=int, default=3,
                        help="Max individual samples to show per observation")
    parser.add_argument("--vmin", type=float, default=0.0,
                        help="Colormap minimum for field plots")
    parser.add_argument("--vmax", type=float, default=3.0,
                        help="Colormap maximum for field plots")
    args = parser.parse_args()

    output_dir = args.folder
    os.makedirs(output_dir, exist_ok=True)

    # ---- Dataset (for mesh coords) ----
    dataset = PinballDataset(args.data_dir, split=args.split)
    mesh_coords = dataset.mesh_coords.numpy()   # [N, 2]
    num_points = mesh_coords.shape[0]

    # ---- Triangulation (pure torch, from the cached edge_index) ----
    tri = load_triangulation(mesh_coords, dataset.edge_index)
    if tri is None:
        print("No triangulation available – cannot produce tripcolor plots.")
        return

    # ---- Sensor locations ----
    sensor_idx_path = os.path.join(output_dir, "sensor_indices.pt")
    if os.path.exists(sensor_idx_path):
        sensor_indices = torch.load(sensor_idx_path, map_location="cpu", weights_only=True).numpy()
        print(f"Loaded sensor indices from {sensor_idx_path} ({len(sensor_indices)} sensors)")
    else:
        print("sensor_indices.pt not found – reconstructing from n_sensors/seed arguments.")
        forward_op = SparseSensorOperator(
            n_dofs=num_points,
            n_sensors=args.n_sensors,
            seed=args.seed,
            device="cpu",
        )
        sensor_indices = forward_op.sensor_indices.numpy()
    sensor_coords = mesh_coords[sensor_indices]            # [n_sensors, 2]

    # ---- Sensor location figure ----
    #plot_sensor_locations(mesh_coords, sensor_coords, tri, output_dir)

    # ---- Load posterior samples ----
    tensors_list, gt_list, paths = load_all_samples(output_dir, sampler=args.sampler)

    # # Stack all samples across observations: [total_B, N]
    # all_samples = torch.cat(
    #     [t.squeeze(1) for t in tensors_list], dim=0
    # )  # [total_B, N]
    # print(f"Total posterior samples: {all_samples.shape[0]}, nodes: {all_samples.shape[1]}")

    # # ---- Global posterior statistics ----
    # mean_field, std_field = plot_mean_and_std(
    #     all_samples, tri, sensor_coords,
    #     output_dir=output_dir,
    #     vmin=args.vmin, vmax=args.vmax,
    # )

    # ---- Per-observation figures ----
    plot_per_observation(
        tensors_list, gt_list, paths, tri, sensor_coords,
        output_dir=output_dir,
        n_show=args.n_show,
        vmin=args.vmin, vmax=args.vmax,
    )


if __name__ == "__main__":
    main()
