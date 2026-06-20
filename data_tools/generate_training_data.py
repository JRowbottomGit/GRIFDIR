#!/usr/bin/env python
"""Generate a uniform-mesh Gaussian-blob conductivity dataset (requires dolfinx)."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import get_conductivity_dataloader

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--nx", type=int, default=32)
p.add_argument("--ny", type=int, default=32)
p.add_argument("--num_samples", type=int, default=10000)
p.add_argument("--seed", type=int, default=0,
               help="np.random seed (gen_conductivity is the RNG source) — set for reproducibility")
p.add_argument("--data_dir", type=str, default="data")
args = p.parse_args()

np.random.seed(args.seed)  # gen_conductivity draws from np.random; seed once
print(f"Generating {args.num_samples} samples on {args.nx}x{args.ny} mesh (seed={args.seed})...")
_, dataset = get_conductivity_dataloader(
    num_samples=args.num_samples, batch_size=32, nx=args.nx, ny=args.ny,
)
mesh_pos, xy, cells, edge_index = dataset.get_mesh_info()
data = {
    "samples": torch.stack([dataset[i] for i in range(len(dataset))]),
    "mesh_pos": torch.from_numpy(mesh_pos).float(),
    "xy": torch.from_numpy(xy).float(),
    "cells": torch.from_numpy(cells).long(),
    "edge_index": edge_index,
    "nx": args.nx, "ny": args.ny,
}
os.makedirs(args.data_dir, exist_ok=True)
out_path = os.path.join(args.data_dir, f"conductivity_nx{args.nx}_ny{args.ny}_n{args.num_samples}.pt")
torch.save(data, out_path)
print(f"Saved to {out_path}  (samples {tuple(data['samples'].shape)})")
