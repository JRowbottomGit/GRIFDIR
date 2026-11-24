"""
Tests for the mesh-transfer flags of MultiscaleGNNScoreNetwork:
fem_cut_cell (cut-cell quadrature) and unpool_interp (linear prolongation).

Run with:
    python -m pytest tests/test_mesh_transfer_flags.py -v
or standalone:
    python tests/test_mesh_transfer_flags.py
"""
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from models.fem_conv import _compute_cut_cell_edges, _eval_p1_basis
from models.multiscale import MultiscaleGNNScoreNetwork, _interp_matrix

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HIERARCHIES = {
    'uniform': os.path.join(REPO, 'meshes', 'mesh_hierarchy.pt'),
    # irregular (netgen) triangulations of non-convex domains
    'l_shape': os.path.join(REPO, 'meshes', 'mesh_hierarchy_l_shape.pt'),
    'square_with_hole': os.path.join(REPO, 'meshes', 'mesh_hierarchy_square_with_hole.pt'),
}


def _load(name):
    path = HIERARCHIES[name]
    if not os.path.exists(path):
        raise unittest.SkipTest(f"{path} not found")
    return torch.load(path, weights_only=False)


def _boundary_distance(points, xy, cells):
    """Distance from each point to the boundary of the triangulated domain (xy, cells):
    the edges that belong to exactly one triangle."""
    e = np.sort(np.r_[cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [0, 2]]], 1)
    e, cnt = np.unique(e, axis=0, return_counts=True)
    b = e[cnt == 1]
    a, d = xy[b[:, 0]], xy[b[:, 1]] - xy[b[:, 0]]
    t = np.clip(((points[:, None] - a) * d).sum(-1) / (d * d).sum(-1), 0.0, 1.0)
    return np.linalg.norm(points[:, None] - (a + t[..., None] * d), axis=-1).min(1)


class TestCutCellQuadrature(unittest.TestCase):
    """The cut-cell edge data integrate a P1 kernel over the ball on any mesh."""

    def _check(self, name):
        h = _load(name)
        xy, cells = np.asarray(h['xy'][0])[:, :2], np.asarray(h['cells'][0])
        c = h['centers'][0].numpy()
        e = np.asarray(h['edge_indices'][0])
        r = 2.0 * np.median(np.linalg.norm(c[e[0]] - c[e[1]], axis=1))   # the model's radius
        ei, ea = _compute_cut_cell_edges(xy, cells, r, 2 * r, 'p1', 5)
        interior = _boundary_distance(c, xy, cells) > 2.5 * r

        # weights cover the ball
        cover = torch.zeros(len(c)).index_add_(0, ei[1], ea[:, -1]).numpy()[interior]
        np.testing.assert_allclose(cover / (np.pi * r * r), 1.0, atol=0.015)

        # smooth kernels (nodal values on the 5x5 patch, index ix + 5 iy) integrate to
        # their dense ball integral
        ux, uy = np.meshgrid(np.linspace(-0.5, 0.5, 5), np.linspace(-0.5, 0.5, 5), indexing='xy')
        ux, uy = ux.ravel(), uy.ravel()
        W = torch.tensor(np.stack([np.ones(25), 1 + ux, np.exp(-(ux ** 2 + uy ** 2) / 0.1),
                                   np.cos(np.pi * ux) * np.cos(np.pi * uy) + 0.5 * uy,
                                   1 - (ux ** 2 + uy ** 2), 0.3 + ux * uy,
                                   np.sin(np.pi * ux) + 1.2, np.exp(uy)]), dtype=torch.float32)
        g = np.linspace(-r, r, 301)
        gx, gy = np.meshgrid(g, g)
        pts = np.stack([gx.ravel(), gy.ravel()], 1)
        pts = torch.from_numpy(pts[np.hypot(*pts.T) < r]).float()
        exact = (_eval_p1_basis(pts, 2 * r, 5) @ W.t()).sum(0).numpy() * (g[1] - g[0]) ** 2
        q = torch.zeros(len(c), 8).index_add_(0, ei[1], (ea[:, 3:28] @ W.t()) * ea[:, -1:])
        err = np.abs(q.numpy()[interior] - exact).mean(1) / np.abs(exact).mean()
        self.assertLess(err.max(), 0.015)

    def test_uniform(self):
        self._check('uniform')

    def test_irregular(self):
        self._check('l_shape')
        self._check('square_with_hole')


class TestLinearProlongation(unittest.TestCase):
    """_interp_matrix reproduces linear fields on every fine node, boundary included."""

    def _check(self, name):
        h = _load(name)
        f = lambda x: 0.3 + 1.7 * x[:, 0] - 0.9 * x[:, 1]
        for l in range(len(h['centers']) - 1):
            cc, cf = h['centers'][l + 1].numpy(), h['centers'][l].numpy()
            P = _interp_matrix(cc, cf)
            np.testing.assert_allclose(torch.sparse.sum(P, 1).to_dense().numpy(), 1.0, atol=1e-5)
            pf = torch.sparse.mm(P, torch.from_numpy(f(cc)).float()[:, None]).numpy()[:, 0]
            np.testing.assert_allclose(pf, f(cf), atol=1e-5)

    def test_uniform(self):
        self._check('uniform')

    def test_irregular(self):
        self._check('l_shape')
        self._check('square_with_hole')

    def test_finite_on_lattice_points(self):
        """Lattice-like nodes give flat Delaunay simplices (no barycentric transform):
        the prolongation weights must stay finite and each row must still sum to 1."""
        g = np.stack(np.meshgrid(*[np.arange(8)] * 3, indexing='ij'), -1).reshape(-1, 3)
        g = g.astype(np.float32) / 7
        src = g[np.random.default_rng(0).choice(len(g), 200, replace=False)]
        P = _interp_matrix(torch.as_tensor(src), torch.as_tensor(g)).coalesce()
        self.assertTrue(torch.isfinite(P.values()).all())
        torch.testing.assert_close(torch.sparse.sum(P, 1).to_dense(), torch.ones(len(g)),
                                   atol=1e-4, rtol=0)


class TestNetworkFlags(unittest.TestCase):

    def _net(self, **flags):
        torch.manual_seed(0)
        return MultiscaleGNNScoreNetwork(
            input_dim=1, pos_dim=2, hidden_dim=16, time_dim=16, n_gnn_layers_per_level=1,
            n_levels=4, layer_type='fem_conv', fem_use_radius=True, fem_radius_mult=2.0,
            fem_lumped_mass=True, use_edge_geom=True, res_invariant=True,
            pooling_type='learned_res', **flags)

    def test_cut_cell_needs_lumped_radius(self):
        with self.assertRaises(ValueError):
            MultiscaleGNNScoreNetwork(layer_type='fem_conv', fem_use_radius=True,
                                      fem_lumped_mass=False, fem_cut_cell=True)

    def test_forward_backward(self):
        h = _load('uniform')
        net = self._net(fem_cut_cell=True, unpool_interp=True)
        net.set_mesh_hierarchy(edge_indices=h['edge_indices'], n_nodes_list=h['n_nodes_list'],
                               pool_edges=h['pool_edges'], unpool_maps=h['unpool_maps'],
                               coarse_coords=h['centers'][-1], level_coords=h['centers'],
                               cells_list=h['cells'], xy_list=h['xy'])
        self.assertTrue(all(P.is_sparse for P in net._unpool_P))
        self.assertEqual(len(net._cut_cell_attr), 4)
        pos = h['centers'][0].float()
        inp = torch.cat([pos.t().unsqueeze(0), torch.randn(1, 1, len(pos))], dim=1).repeat(2, 1, 1)
        out = net(inp, torch.tensor([0.5, 2.0]), pos.unsqueeze(0).repeat(2, 1, 1))
        self.assertEqual(tuple(out.shape), (2, 1, len(pos)))
        self.assertTrue(torch.isfinite(out).all())
        out.pow(2).mean().backward()
        grad = net.shared_down_gnn.layers[0].geo_to_gate.weight.grad
        self.assertTrue(grad is not None and grad.abs().sum() > 0)


if __name__ == '__main__':
    unittest.main()
