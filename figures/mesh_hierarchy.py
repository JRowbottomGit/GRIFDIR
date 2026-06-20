"""
Visualise samples from one or more mesh datasets.

Layout: one row per dataset, three columns (three random samples).

Usage
-----
  # From GRIFDIR root:
  python multiscale/visualise_mesh_datasets.py \
      data/conductivity_x_shape_maxh0.100_n1000.pt \
      data/conductivity_circle_with_hole_maxh0.100_n1000.pt \
      data/conductivity_l_shape_maxh0.100_n1000.pt

  # All .pt files in data/ (one row each):
  python multiscale/visualise_mesh_datasets.py --all

  # All files in a custom directory:
  python multiscale/visualise_mesh_datasets.py --all --data_dir /path/to/data

  # Save to file instead of showing:
  python multiscale/visualise_mesh_datasets.py --all --out figures/datasets.png
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


N_COLS = 3   # samples per row


def load_pt(path):
    data = torch.load(path, weights_only=False)
    samples    = data["samples"].numpy()          # [N, 1, num_cells]
    mesh_pos   = data["mesh_pos"].numpy()         # [num_cells, 2]
    xy         = data["xy"].numpy()               # [num_verts, 2]
    cells      = data["cells"].numpy()            # [num_cells, 3]
    domain     = data.get("domain", os.path.basename(path))
    mesh_tag   = data.get("mesh_tag", "")
    return samples, mesh_pos, xy, cells, domain, mesh_tag


def plot_sample(ax, tri, values, title=None):
    im = ax.tripcolor(
        tri, values.flatten(),
        cmap="RdBu_r", shading="flat",
        vmin=values.min(), vmax=values.max(),
    )
    ax.set_aspect("equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=8, pad=3)
    return im


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(description="Visualise mesh dataset samples")
    parser.add_argument("paths", nargs="*", help=".pt dataset files")
    parser.add_argument("--all", action="store_true",
                        help="Use all conductivity_*.pt files found in --data_dir")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(repo_root, "data"),
                        help="Directory to search when --all is set (default: data/)")
    parser.add_argument("--out",  type=str, default=None,
                        help="Save path (default: figures/mesh_datasets.png)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for sample selection")
    args = parser.parse_args()

    if args.all:
        import glob
        found = sorted(glob.glob(os.path.join(args.data_dir, "conductivity_*.pt")))
        if not found:
            print(f"No conductivity_*.pt files found in {args.data_dir}")
            sys.exit(1)
        args.paths = found
        print(f"Found {len(found)} datasets:")
        for p in found:
            print(f"  {p}")

    if not args.paths:
        parser.error("Provide .pt paths or use --all")

    rng = np.random.default_rng(args.seed)

    n_rows = len(args.paths)
    fig, axes = plt.subplots(
        n_rows, N_COLS,
        figsize=(N_COLS * 3.5, n_rows * 3.2),
        squeeze=False,
    )

    for row, path in enumerate(args.paths):
        samples, mesh_pos, xy, cells, domain, mesh_tag = load_pt(path)
        tri = Triangulation(xy[:, 0], xy[:, 1], cells)

        # Pick N_COLS random samples
        indices = rng.choice(len(samples), size=N_COLS, replace=False)

        for col, idx in enumerate(indices):
            ax = axes[row][col]
            values = samples[idx, 0]          # [num_cells]
            title = f"sample {idx}" if col > 0 else None
            im = plot_sample(ax, tri, values, title=title)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Row label on leftmost axis
        axes[row][0].set_title(f"sample {indices[0]}", fontsize=8, pad=3)
        axes[row][0].set_ylabel(
            f"{domain}\n({mesh_tag})\nn={len(samples)}",
            fontsize=9, labelpad=6,
        )

    fig.suptitle("Conductivity samples per domain", fontsize=11, y=1.01)
    plt.tight_layout()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = args.out or os.path.join(script_dir, "mesh_datasets.png")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
