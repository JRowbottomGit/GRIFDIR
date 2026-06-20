"""
Generate a GRIFDIR dataset on an L-shaped domain.

Usage
-----
  # From the GRIFDIR root:
  python multiscale/create_lshape_dataset.py --maxh 0.1 --n 5000

  # Or from inside multiscale/:
  python create_lshape_dataset.py --maxh 0.1 --n 5000

Output
------
  data/conductivity_l_shape_maxh0.10_n5000.pt

  Keys:
    samples    : [N, 1, num_cells]  float32
    mesh_pos   : [num_cells, 2]     float32  (DG0 cell centres)
    xy         : [num_verts, 2]     float32  (vertex coords)
    cells      : [num_cells, 3]     int64
    edge_index : [2, num_edges]     int64    (dual graph)
    domain     : "l_shape"
    mesh_tag   : "maxh0.10"
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from mesh_factory import build_lshape_mesh, mesh_tag_for
from gaussian_blob_utils import gen_conductivity


def _create_sample(mesh_pos, max_numInc: int, backCond: float):
    sigma = gen_conductivity(
        mesh_pos[:, 0], mesh_pos[:, 1],
        max_numInc=max_numInc,
        backCond=backCond,
    )
    return torch.from_numpy(sigma).float().unsqueeze(0)  # [1, N]


def main():
    parser = argparse.ArgumentParser(description="Generate L-shape GRIFDIR dataset")
    parser.add_argument("--maxh", type=float, default=0.1,
                        help="Max element size (controls mesh density)")
    parser.add_argument("--n", type=int, default=5000,
                        help="Number of samples to generate")
    parser.add_argument("--max_numInc", type=int, default=3,
                        help="Max number of Gaussian inclusions per sample")
    parser.add_argument("--backCond", type=float, default=1.0,
                        help="Background conductivity value")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Output directory (relative to GRIFDIR root)")
    args = parser.parse_args()

    tag = mesh_tag_for("l_shape", maxh=args.maxh)
    print(f"Building L-shape mesh: maxh={args.maxh}  (tag={tag})")
    mesh_pos, xy, cells, edge_index = build_lshape_mesh(maxh=args.maxh)
    print(f"  Cells: {len(mesh_pos)}  |  Vertices: {len(xy)}  |  Dual edges: {edge_index.shape[1]}")

    samples = []
    for _ in tqdm(range(args.n), desc="Generating samples"):
        samples.append(_create_sample(mesh_pos, args.max_numInc, args.backCond))
    samples = torch.stack(samples)  # [N, 1, num_cells]

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.data_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"conductivity_l_shape_{tag}_n{args.n}.pt")
    torch.save(
        {
            "samples":    samples,
            "mesh_pos":   torch.from_numpy(mesh_pos).float(),
            "xy":         torch.from_numpy(xy).float(),
            "cells":      torch.from_numpy(cells).long(),
            "edge_index": edge_index,
            "domain":     "l_shape",
            "mesh_tag":   tag,
        },
        out_path,
    )
    print(f"Saved → {out_path}")
    print(f"  samples : {samples.shape}")
    print(f"  mesh_pos: {mesh_pos.shape}")


if __name__ == "__main__":
    main()
