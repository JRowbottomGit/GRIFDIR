"""
Generate GRIFDIR conductivity datasets on arbitrary mesh domains.

Blob centres are sampled from mesh cell positions (always inside the domain)
and scaled relative to the bounding-box diagonal — so the distribution is
consistent across square, L-shape, x-shape, annulus, etc.

Usage
-----
  # From GRIFDIR root, single domain:
  python multiscale/create_mesh_dataset.py --domain x_shape --n 1000

  # Multiple domains in one go:
  python multiscale/create_mesh_dataset.py --domain x_shape circle_with_hole --n 1000

  # All domains known to mesh_factory:
  python multiscale/create_mesh_dataset.py --all --n 1000

  # Custom mesh parameters:
  python multiscale/create_mesh_dataset.py --domain l_shape --n 1000 --maxh 0.05

Output
------
  data/conductivity_<domain>_maxh<h>_n<N>.pt  (one file per domain)

  Keys in each .pt file  (same schema as create_lshape_dataset.py):
    samples    : [N, 1, num_cells]  float32   conductivity fields
    mesh_pos   : [num_cells, 2]     float32   DG0 cell-centre coords
    xy         : [num_verts, 2]     float32   vertex coords
    cells      : [num_cells, 3]     int64     triangle vertex indices
    edge_index : [2, num_edges]     int64     dual-graph (cell-adjacency)
    domain     : str                          domain name
    mesh_tag   : str                          e.g. "x_shape_maxh0.100"
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from tqdm import tqdm

from mesh_factory import build_mesh, mesh_tag_for, _DOMAIN_BUILDERS
from gaussian_blob_utils import gen_conductivity_on_mesh, boundary_distance_mask


# Domains supported by mesh_factory (no special kwargs beyond maxh)
_ALL_DOMAINS = sorted(_DOMAIN_BUILDERS.keys())


def _create_sample(
    mesh_pos: np.ndarray,
    max_numInc: int,
    backCond: float,
    centre_mask: np.ndarray,
) -> torch.Tensor:
    sigma = gen_conductivity_on_mesh(
        mesh_pos,
        max_numInc=max_numInc,
        backCond=backCond,
        centre_mask=centre_mask,
    )
    return torch.from_numpy(sigma).float().unsqueeze(0)   # [1, N_cells]


def generate_domain_dataset(
    domain: str,
    n: int,
    maxh: float = 0.1,
    max_numInc: int = 3,
    backCond: float = 1.0,
    data_dir: str = "data",
    domain_kwargs: dict | None = None,
    margin: float = 0.12,
):
    """Build mesh, generate N samples, save to data_dir."""
    kwargs = {"maxh": maxh}
    if domain_kwargs:
        kwargs.update(domain_kwargs)

    tag = mesh_tag_for(domain, **kwargs)
    print(f"\n>>> Domain: {domain}  (tag={tag})")
    mesh_pos, xy, cells, edge_index = build_mesh(domain, **kwargs)
    print(f"    Cells: {len(mesh_pos)}  |  Vertices: {len(xy)}  |  Dual edges: {edge_index.shape[1]}")

    # Pre-compute interior mask once — cells at least margin·L from boundary
    centre_mask = boundary_distance_mask(mesh_pos, xy, cells, margin=margin)
    n_admissible = centre_mask.sum()
    print(f"    Admissible blob centres: {n_admissible} / {len(mesh_pos)} "
          f"(margin={margin}, {100*n_admissible/len(mesh_pos):.1f}%)")

    samples = []
    for _ in tqdm(range(n), desc=f"  Generating {domain}"):
        samples.append(_create_sample(mesh_pos, max_numInc, backCond, centre_mask))
    samples = torch.stack(samples)   # [N, 1, num_cells]

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        data_dir,
    )
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"conductivity_{tag}_n{n}.pt")

    torch.save(
        {
            "samples":    samples,
            "mesh_pos":   torch.from_numpy(mesh_pos).float(),
            "xy":         torch.from_numpy(xy).float(),
            "cells":      torch.from_numpy(cells).long(),
            "edge_index": edge_index,
            "domain":     domain,
            "mesh_tag":   tag,
        },
        out_path,
    )
    print(f"    Saved → {out_path}  |  samples: {samples.shape}")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate conductivity datasets on arbitrary mesh domains",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Supported domains: {', '.join(_ALL_DOMAINS)}",
    )
    parser.add_argument(
        "--domain", nargs="+",
        help="One or more domain names (use --all for every domain)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help=f"Generate for all domains: {', '.join(_ALL_DOMAINS)}",
    )
    parser.add_argument("--n",          type=int,   default=1000,  help="Samples per domain")
    parser.add_argument("--maxh",       type=float, default=0.1,   help="gmsh max element size")
    parser.add_argument("--max_numInc", type=int,   default=3,     help="Max blobs per sample")
    parser.add_argument("--backCond",   type=float, default=1.0,   help="Background conductivity")
    parser.add_argument("--margin",     type=float, default=0.12,
                        help="Min distance from boundary as fraction of domain diagonal (default 0.12)")
    parser.add_argument("--data_dir",   type=str,   default="data", help="Output directory")
    parser.add_argument("--seed",       type=int,   default=0,
                        help="np.random seed (gen_conductivity is the RNG source) — set for reproducibility")
    args = parser.parse_args()

    if args.all:
        domains = _ALL_DOMAINS
    elif args.domain:
        domains = args.domain
    else:
        parser.error("Specify --domain <name> [<name> ...] or --all")

    unknown = [d for d in domains if d not in _ALL_DOMAINS]
    if unknown:
        parser.error(f"Unknown domain(s): {unknown}.  Supported: {_ALL_DOMAINS}")

    print(f"=== create_mesh_dataset  n={args.n}  maxh={args.maxh}  seed={args.seed} ===")
    print(f"    Domains: {domains}")

    np.random.seed(args.seed)  # gen_conductivity draws from np.random; seed once
    for domain in domains:
        generate_domain_dataset(
            domain=domain,
            n=args.n,
            maxh=args.maxh,
            max_numInc=args.max_numInc,
            backCond=args.backCond,
            data_dir=args.data_dir,
            margin=args.margin,
        )

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
