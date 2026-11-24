"""
Tests for the lumped FEM conv's boundary options.

fem_lumped_norm: divide by the area of each node's ball inside the domain (the
area-weighted mean, partial-convolution padding), so a constant field returns itself at
walls as in the interior.
fem_lumped_ball_norm: divide by the full ball's area, one constant per level (zero
padding at the scale of a mean). Alternative to fem_lumped_norm.
fem_wall_signal: a learned vector (zero at init) times the share of the ball outside
the domain, a boundary signal kept apart from the field.
With none of them, the plain lumped sum drops with the covered area (zero padding).

Run with:
    python -m pytest tests/test_lumped_norm.py -v
"""
import sys, os, math, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch import nn

from models.fem_conv import FEMConvLayer
from models.multiscale import MultiscaleGNNScoreNetwork

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PINBALL = os.path.join(REPO, 'data', 'Pinball')

# Three target nodes (0, 1, 2), each fed by its own source nodes, with weights (areas)
# summing to 1.0 (interior), 0.5 (half the ball outside the domain) and 0.7 (graded).
AREAS = {0: [0.25, 0.25, 0.25, 0.25], 1: [0.25, 0.25], 2: [0.5, 0.1, 0.1]}
UNIT_BALL_PW = 2.0 / math.sqrt(math.pi)          # patch width whose ball has area 1


def toy_graph():
    src, dst, attr, n = [], [], [], 3
    for i, areas in AREAS.items():
        for k, a in enumerate(areas):
            off = torch.tensor([0.1 * (k + 1), -0.05 * k])
            src.append(n); dst.append(i); n += 1
            attr.append(torch.cat([off, off.norm().view(1), torch.tensor([a])]))
    return torch.tensor([src, dst]), torch.stack(attr), n


def make_layer(norm=False, signal=False, ball=False):
    torch.manual_seed(0)
    return FEMConvLayer(4, edge_dim=4, lumped_mass=True, lumped_norm=norm,
                        wall_signal=signal, ball_norm=ball)


def neighbour_term(layer, pw=1.0):
    """The layer's neighbour term for a constant field, with the root map and LayerNorm
    removed and the gate fixed to 1 (P1 weights sum to 1 inside the patch)."""
    with torch.no_grad():
        layer.root.weight.zero_(); layer.root.bias.zero_()
        layer.geo_to_gate.weight.fill_(1.0)
    layer.norm = nn.Identity()
    ei, ea, n = toy_graph()
    h = torch.ones(n, 4)
    out = layer(h, ei, edge_attr=ea, edges_preexpanded=True, patch_width_override=pw)
    return out - h, layer.msg_linear.weight.sum(1)


class TestLumpedNorm(unittest.TestCase):

    def test_constant_field_returns_itself(self):
        agg, v = neighbour_term(make_layer(norm=True))
        for i in AREAS:
            torch.testing.assert_close(agg[i], v)
        self.assertTrue((agg[3:] == 0).all())           # no in-edges: nothing to divide

    def test_plain_lumped_drops_with_covered_area(self):
        agg, v = neighbour_term(make_layer())
        for i, areas in AREAS.items():
            torch.testing.assert_close(agg[i], v * sum(areas))

    def test_ball_norm_is_zero_padding_at_mean_scale(self):
        agg, v = neighbour_term(make_layer(ball=True), pw=1.0)
        for i, areas in AREAS.items():
            torch.testing.assert_close(agg[i], v * sum(areas) / (math.pi * 0.25))

    def test_alternatives(self):
        with self.assertRaises(AssertionError):
            make_layer(norm=True, ball=True)
        with self.assertRaises(ValueError):
            MultiscaleGNNScoreNetwork(input_dim=1, pos_dim=2, hidden_dim=16, time_dim=16,
                                      layer_type='fem_conv', fem_use_radius=True,
                                      fem_lumped_mass=True, fem_lumped_norm=True,
                                      fem_lumped_ball_norm=True)

    def test_needs_lumped_mass(self):
        for flag in ('fem_lumped_norm', 'fem_wall_signal', 'fem_lumped_ball_norm'):
            with self.assertRaises(ValueError):
                MultiscaleGNNScoreNetwork(input_dim=1, pos_dim=2, hidden_dim=16, time_dim=16,
                                          layer_type='fem_conv', fem_use_radius=True,
                                          **{flag: True})


class TestWallSignal(unittest.TestCase):

    def test_reads_share_outside(self):
        layer = make_layer(norm=True, signal=True)
        with torch.no_grad():
            layer.wall_gate.fill_(1.0)
        agg, v = neighbour_term(layer, pw=UNIT_BALL_PW)
        for i, areas in AREAS.items():
            torch.testing.assert_close(agg[i], v + (1.0 - sum(areas)))

    def test_starts_off(self):
        ei, ea, n = toy_graph()
        h = torch.randn(n, 4)
        a = make_layer(norm=True, signal=True)(h, ei, edge_attr=ea, edges_preexpanded=True,
                                               patch_width_override=UNIT_BALL_PW)
        b = make_layer(norm=True)(h, ei, edge_attr=ea, edges_preexpanded=True,
                                  patch_width_override=UNIT_BALL_PW)
        torch.testing.assert_close(a, b)


class TestPinballNetwork(unittest.TestCase):
    """The boundary options in a full network on the shipped pinball hierarchy
    (vertex-centred, with cut-cell quadrature and in-domain prolongation)."""

    def run_net(self, **flags):
        if not os.path.exists(os.path.join(PINBALL, 'lc_4000_kNN')):
            raise unittest.SkipTest(f"{PINBALL} not found")
        from model_mesh_utils import load_pinball_hierarchy
        h = load_pinball_hierarchy(PINBALL, PINBALL, ['lc_1000_kNN', 'lc_4000_kNN'],
                                   ['lc_1000_kNN', 'lc_2000_kNN', 'lc_4000_kNN'])
        torch.manual_seed(0)
        net = MultiscaleGNNScoreNetwork(
            input_dim=1, pos_dim=2, hidden_dim=16, time_dim=16, n_gnn_layers_per_level=1,
            n_levels=len(h['n_nodes_list']), layer_type='fem_conv', fem_use_radius=True,
            fem_radius_mult=2.0, fem_lumped_mass=True, use_edge_geom=True, res_invariant=True,
            pooling_type='learned_res', fem_cut_cell=True, unpool_interp=True, **flags)
        net.set_mesh_hierarchy(edge_indices=h['edge_indices'], n_nodes_list=h['n_nodes_list'],
                               pool_edges=h['pool_edges'], unpool_maps=h['unpool_maps'],
                               coarse_coords=h['centers'][-1], level_coords=h['centers'],
                               cells_list=h['cells_list'], xy_list=h['xy_list'])
        pos = torch.as_tensor(h['centers'][0]).float()
        inp = torch.cat([pos.t().unsqueeze(0), torch.randn(1, 1, len(pos))], dim=1).repeat(2, 1, 1)
        out = net(inp, torch.tensor([0.5, 2.0]), pos.unsqueeze(0).repeat(2, 1, 1))
        self.assertEqual(tuple(out.shape), (2, 1, len(pos)))
        self.assertTrue(torch.isfinite(out).all())
        out.pow(2).mean().backward()
        return net.shared_down_gnn.layers[0]

    def test_norm_and_wall_signal(self):
        layer = self.run_net(fem_lumped_norm=True, fem_wall_signal=True)
        self.assertTrue(layer.lumped_norm and layer.wall_signal)
        for p in (layer.geo_to_gate.weight, layer.wall_gate):
            self.assertTrue(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)

    def test_ball_norm(self):
        layer = self.run_net(fem_lumped_ball_norm=True)
        self.assertTrue(layer.ball_norm and not layer.lumped_norm)
        g = layer.geo_to_gate.weight.grad
        self.assertTrue(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0)


if __name__ == '__main__':
    unittest.main()
