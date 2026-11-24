"""
Tests for the mesh-transfer flags on vertex-centred meshes with holes (e.g. pinball):
fem_cut_cell with the model's nodes at the mesh vertices (hat-weighted cut-cell
quadrature) and unpool_interp restricted to the domain.

Run with:
    python -m pytest tests/test_vertex_mesh_flags.py -v
"""
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from scipy.spatial import Delaunay, cKDTree

from models.fem_conv import _compute_cut_cell_edges, _eval_p1_basis
from models.multiscale import (
    MultiscaleGNNScoreNetwork, _interp_matrix, _interp_matrix_in_domain,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINBALL = os.path.join(REPO, 'data', 'Pinball')
HOLE = (0.5, 0.5, 0.2)                                   # centre x, centre y, radius


def pinball_hierarchy():
    """The shipped pinball hierarchy (vertex-centred, three holes), as in training."""
    if not os.path.exists(os.path.join(PINBALL, 'lc_4000_kNN')):
        raise unittest.SkipTest(f"{PINBALL} not found")
    from model_mesh_utils import load_pinball_hierarchy
    return load_pinball_hierarchy(PINBALL, PINBALL, ['lc_1000_kNN', 'lc_4000_kNN'],
                                  ['lc_1000_kNN', 'lc_2000_kNN', 'lc_4000_kNN'])


def holed_mesh(n, ring=True):
    """Vertices of an n x n grid on the unit square outside a disc (plus points on the
    disc's edge if ring); Delaunay triangles, those inside the disc removed."""
    cx, cy, rh = HOLE
    g = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(g, g)
    pts = np.c_[X.ravel(), Y.ravel()]
    pts = pts[np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) > rh + 0.3 / n]
    if ring:
        m = max(8, int(round(2 * np.pi * rh * n)))
        th = 2 * np.pi * np.arange(m) / m
        pts = np.vstack([pts, np.c_[cx + rh * np.cos(th), cy + rh * np.sin(th)]])
    t = Delaunay(pts).simplices
    c = pts[t].mean(1)
    return pts, t[np.hypot(c[:, 0] - cx, c[:, 1] - cy) > rh]


class TestVertexCutCell(unittest.TestCase):
    """Hat-weighted cut-cell edges integrate constant, linear and smooth-kernel fields
    over the ball at every interior vertex."""

    def test_exact(self):
        xy, cells = holed_mesh(32)
        e = np.unique(np.sort(np.r_[cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [0, 2]]], 1), axis=0)
        r = 2.0 * np.median(np.linalg.norm(xy[e[:, 0]] - xy[e[:, 1]], axis=1))
        ei, ea = _compute_cut_cell_edges(xy, cells, r, 2 * r, 'p1', 5, nodes='vertices')
        n = len(xy)
        clear = np.hypot(xy[:, 0] - HOLE[0], xy[:, 1] - HOLE[1]) - HOLE[2] > r
        interior = np.all((xy > r) & (xy < 1 - r), 1) & clear
        self.assertGreater(interior.sum(), 100)

        # each vertex's own hat is in its sum
        self.assertTrue(((ei[0] == ei[1]).sum() == n))

        # weights cover the ball, and their centroid is its centre
        w = ea[:, -1].double()
        cover = torch.zeros(n, dtype=torch.float64).index_add_(0, ei[1], w).numpy()
        np.testing.assert_allclose(cover[interior] / (np.pi * r * r), 1.0, atol=0.01)
        mom = torch.zeros(n, 2, dtype=torch.float64).index_add_(0, ei[1], ea[:, :2].double() * w[:, None])
        shift = np.linalg.norm(mom.numpy()[interior], axis=1) / cover[interior] / r
        self.assertLess(shift.max(), 0.01)

        # smooth kernels (nodal values on the 5x5 patch) integrate to their dense ball integral
        ux, uy = np.meshgrid(np.linspace(-0.5, 0.5, 5), np.linspace(-0.5, 0.5, 5), indexing='xy')
        ux, uy = ux.ravel(), uy.ravel()
        W = torch.tensor(np.stack([np.ones(25), 1 + ux, np.exp(-(ux ** 2 + uy ** 2) / 0.1),
                                   np.cos(np.pi * ux) * np.cos(np.pi * uy) + 0.5 * uy,
                                   1 - (ux ** 2 + uy ** 2), 0.3 + ux * uy]), dtype=torch.float32)
        g = np.linspace(-r, r, 301)
        gx, gy = np.meshgrid(g, g)
        pts = np.stack([gx.ravel(), gy.ravel()], 1)
        pts = torch.from_numpy(pts[np.hypot(*pts.T) < r]).float()
        exact = (_eval_p1_basis(pts, 2 * r, 5) @ W.t()).sum(0).numpy() * (g[1] - g[0]) ** 2
        q = torch.zeros(n, len(W)).index_add_(0, ei[1], (ea[:, 3:28] @ W.t()) * ea[:, -1:])
        err = np.abs(q.numpy()[interior] - exact).mean(1) / np.abs(exact).mean()
        self.assertLess(err.max(), 0.015)


class TestDomainProlongation(unittest.TestCase):
    """_interp_matrix_in_domain never interpolates across a hole, and is otherwise linear."""

    @staticmethod
    def _rows_across_hole(P, coarse):
        idx = P.coalesce().indices().numpy()
        cnt = np.bincount(idx[0], minlength=P.shape[0])
        cen = np.zeros((P.shape[0], 2))
        np.add.at(cen, idx[0], coarse[idx[1]])
        tri = cnt == 3
        cen[tri] /= 3
        inside = np.hypot(cen[:, 0] - HOLE[0], cen[:, 1] - HOLE[1]) < HOLE[2]
        return int((tri & inside).sum()), cnt

    def test_no_mixing_across_hole(self):
        fine, cells = holed_mesh(32)
        coarse, _ = holed_mesh(8, ring=False)          # coarse nodes do not trace the hole
        parent = cKDTree(coarse).query(fine)[1]
        old, _ = self._rows_across_hole(_interp_matrix(coarse, fine), coarse)
        P = _interp_matrix_in_domain(coarse, fine, (fine, cells), parent)
        new, cnt = self._rows_across_hole(P, coarse)
        self.assertGreater(old, 0)
        self.assertEqual(new, 0)
        np.testing.assert_allclose(torch.sparse.sum(P, 1).to_dense().numpy(), 1.0, atol=1e-5)
        f = lambda x: 0.3 + 1.7 * x[:, 0] - 0.9 * x[:, 1]
        pf = torch.sparse.mm(P, torch.from_numpy(f(coarse)).float()[:, None]).numpy()[:, 0]
        linear = cnt == 3
        np.testing.assert_allclose(pf[linear], f(fine)[linear], atol=1e-5)

    def test_convex_domain_unchanged(self):
        """Without holes nothing is dropped: identical to _interp_matrix."""
        g = np.linspace(0, 1, 17)
        fine = np.stack(np.meshgrid(g, g), -1).reshape(-1, 2)
        cells = Delaunay(fine).simplices
        coarse = np.stack(np.meshgrid(g[::4], g[::4]), -1).reshape(-1, 2) + 0.01 * np.sin(np.arange(25))[:, None]
        coarse = np.clip(coarse, 0, 1)
        a = _interp_matrix(coarse, fine).to_dense()
        b = _interp_matrix_in_domain(coarse, fine, (fine, cells), cKDTree(coarse).query(fine)[1]).to_dense()
        torch.testing.assert_close(a, b)


class TestVertexNetwork(unittest.TestCase):

    def test_pinball_forward_backward(self):
        h = pinball_hierarchy()
        torch.manual_seed(0)
        net = MultiscaleGNNScoreNetwork(
            input_dim=1, pos_dim=2, hidden_dim=16, time_dim=16, n_gnn_layers_per_level=1,
            n_levels=len(h['n_nodes_list']), layer_type='fem_conv', fem_use_radius=True,
            fem_radius_mult=2.0, fem_lumped_mass=True, use_edge_geom=True, res_invariant=True,
            pooling_type='learned_res', fem_cut_cell=True, unpool_interp=True)
        net.set_mesh_hierarchy(edge_indices=h['edge_indices'], n_nodes_list=h['n_nodes_list'],
                               pool_edges=h['pool_edges'], unpool_maps=h['unpool_maps'],
                               coarse_coords=h['centers'][-1], level_coords=h['centers'],
                               cells_list=h['cells_list'], xy_list=h['xy_list'])
        self.assertTrue(all(P.is_sparse for P in net._unpool_P))
        self.assertEqual(len(net._cut_cell_attr), len(h['n_nodes_list']))
        pos = torch.as_tensor(h['centers'][0]).float()
        inp = torch.cat([pos.t().unsqueeze(0), torch.randn(1, 1, len(pos))], dim=1).repeat(2, 1, 1)
        out = net(inp, torch.tensor([0.5, 2.0]), pos.unsqueeze(0).repeat(2, 1, 1))
        self.assertEqual(tuple(out.shape), (2, 1, len(pos)))
        self.assertTrue(torch.isfinite(out).all())
        out.pow(2).mean().backward()
        grad = net.shared_down_gnn.layers[0].geo_to_gate.weight.grad
        self.assertTrue(grad is not None and grad.abs().sum() > 0)


if __name__ == '__main__':
    unittest.main()
