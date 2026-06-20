import torch
import numpy as np
import matplotlib
if torch.cuda.is_available():
    matplotlib.use('Agg')
else:
    matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.spatial import cKDTree

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mesh_factory import build_mesh


def create_level(nx, ny):
    from data_utils import ConductivityDataset
    ds = ConductivityDataset(num_samples=1, nx=nx, ny=ny)
    mesh_pos, xy, cells, edge_index = ds.get_mesh_info()
    return mesh_pos, xy, cells, edge_index


def build_unpool_map(fine_centers, coarse_centers):
    tree = cKDTree(coarse_centers)
    _, indices = tree.query(fine_centers, k=1)
    return torch.from_numpy(indices).long()


def build_pool_edges(fine_centers, coarse_centers, k=4):
    tree = cKDTree(coarse_centers)
    _, coarse_idx = tree.query(fine_centers, k=1)
    fine_idx = np.arange(len(fine_centers))
    return torch.stack([
        torch.from_numpy(fine_idx).long(),
        torch.from_numpy(coarse_idx).long(),
    ])


def build_hierarchy(resolutions):
    levels = []
    for nx, ny in resolutions:
        centers, xy, cells, ei = create_level(nx, ny)
        levels.append({
            'nx': nx, 'ny': ny,
            'centers': centers, 'xy': xy, 'cells': cells,
            'edge_index': ei, 'n_nodes': len(centers),
        })
        print(f"  Level {len(levels)-1}: {nx}x{ny} -> {len(centers)} cells, {ei.shape[1]} dual edges")

    pool_edges = []
    unpool_maps = []
    for i in range(len(levels) - 1):
        fine_c = levels[i]['centers']
        coarse_c = levels[i+1]['centers']
        pe = build_pool_edges(fine_c, coarse_c)
        um = build_unpool_map(fine_c, coarse_c)
        pool_edges.append(pe)
        unpool_maps.append(um)
        print(f"  Pool {i}->{i+1}: {fine_c.shape[0]} -> {coarse_c.shape[0]}")

    return levels, pool_edges, unpool_maps


def visualize_hierarchy(levels, pool_edges, save_path='hierarchy.png'):
    n = len(levels)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for i, level in enumerate(levels):
        ax = axes[i]
        tri = Triangulation(level['xy'][:, 0], level['xy'][:, 1], level['cells'])
        ax.triplot(tri, 'k-', linewidth=0.3)
        ax.plot(level['centers'][:, 0], level['centers'][:, 1], 'r.', markersize=2)
        if 'nx' in level:
            label = f"{level['nx']}x{level['ny']}"
        else:
            label = f"maxh={level['maxh']}"
        ax.set_title(f"Level {i}: {label}\n{level['n_nodes']} cells")
        ax.set_aspect('equal')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved hierarchy visualization to {save_path}")
    plt.show()


def save_hierarchy(levels, pool_edges, unpool_maps, save_path='mesh_hierarchy.pt'):
    data = {
        'edge_indices': [l['edge_index'] for l in levels],
        'n_nodes_list': [l['n_nodes'] for l in levels],
        'centers':      [torch.from_numpy(l['centers']).float() for l in levels],
        'xy':           [torch.from_numpy(l['xy']).float() for l in levels],
        'cells':        [torch.from_numpy(l['cells']).long() for l in levels],
        'pool_edges':   pool_edges,
        'unpool_maps':  unpool_maps,
    }
    if 'nx' in levels[0]:
        data['resolutions'] = [(l['nx'], l['ny']) for l in levels]
    torch.save(data, save_path)
    print(f"Saved hierarchy data to {save_path}")


# ---------------------------------------------------------------------------
# Unstructured domain hierarchy (l_shape, etc.)
# ---------------------------------------------------------------------------

def build_hierarchy_unstructured(domain: str, maxh_levels: list):
    """
    Build multiscale hierarchy for an unstructured domain.

    Parameters
    ----------
    domain     : domain key passed to build_mesh, e.g. 'l_shape'
    maxh_levels: list of max element sizes from finest to coarsest,
                 e.g. [0.05, 0.1, 0.2, 0.4]

    Returns
    -------
    levels, pool_edges, unpool_maps  (same format as build_hierarchy)
    """
    levels = []
    for maxh in maxh_levels:
        centers, xy, cells, ei = build_mesh(domain, maxh=maxh)
        levels.append({
            'maxh': maxh,
            'centers': centers, 'xy': xy, 'cells': cells,
            'edge_index': ei, 'n_nodes': len(centers),
        })
        print(f"  Level {len(levels)-1}: maxh={maxh} -> {len(centers)} cells, {ei.shape[1]} dual edges")

    pool_edges = []
    unpool_maps = []
    for i in range(len(levels) - 1):
        fine_c = levels[i]['centers']
        coarse_c = levels[i + 1]['centers']
        pe = build_pool_edges(fine_c, coarse_c)
        um = build_unpool_map(fine_c, coarse_c)
        pool_edges.append(pe)
        unpool_maps.append(um)
        print(f"  Pool {i}->{i+1}: {fine_c.shape[0]} -> {coarse_c.shape[0]}")

    return levels, pool_edges, unpool_maps


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default='square',
                        choices=['square', 'l_shape', 'plus', 'e_shape', 'x_shape',
                                 'circle', 'square_with_hole', 'circle_with_hole'])
    parser.add_argument('--nx', type=int, default=32)
    parser.add_argument('--ny', type=int, default=32)
    parser.add_argument('--maxh_levels', type=float, nargs='+',
                        default=[0.05, 0.1, 0.2, 0.4],
                        help='Unstructured: maxh values finest->coarsest')
    parser.add_argument('--out', type=str, default=None,
                        help='Output .pt path (auto-named if omitted)')
    args = parser.parse_args()

    if args.domain == 'square':
        resolutions = [(args.nx, args.ny), (args.nx // 2, args.ny // 2),
                       (args.nx // 4, args.ny // 4), (args.nx // 8, args.ny // 8)]
        print(f"Building square hierarchy: {resolutions}")
        levels, pool_edges, unpool_maps = build_hierarchy(resolutions)
        out = args.out or 'mesh_hierarchy.pt'
    else:
        print(f"Building {args.domain} hierarchy: maxh={args.maxh_levels}")
        levels, pool_edges, unpool_maps = build_hierarchy_unstructured(args.domain, args.maxh_levels)
        out = args.out or f'mesh_hierarchy_{args.domain}.pt'

    visualize_hierarchy(levels, pool_edges, save_path=out.replace('.pt', '.png'))
    save_hierarchy(levels, pool_edges, unpool_maps, save_path=out)
