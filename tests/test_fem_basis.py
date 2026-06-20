"""
Tests for FEM basis functions and 2-hop edge expansion in fem_conv_layer.py.

Run with:
    python -m pytest tests/test_fem_basis.py -v
or standalone:
    python tests/test_fem_basis.py
"""
import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.fem_conv import (
    _eval_p0_basis,
    _eval_p1_basis,
    _compute_2hop_edges,
    _expand_to_2hops,
    FEMConvLayer,
)


# ---------------------------------------------------------------------------
# P1 bilinear basis tests
# ---------------------------------------------------------------------------

class TestP1Basis(unittest.TestCase):

    def _basis(self, rel, patch_width=1.0, R=5):
        return _eval_p1_basis(rel, patch_width, R)

    def test_exact_node_positions_give_unit_basis(self):
        """At each of the R×R grid nodes the corresponding basis value = 1."""
        R = 5
        pw = 1.0
        # Grid node positions in [-pw/2, pw/2]²
        xs = torch.linspace(-pw/2, pw/2, R)
        ys = torch.linspace(-pw/2, pw/2, R)
        # Iterate iy (slow) then ix (fast) so that row index = ix + iy*R
        # matches the basis linear index: n = ix + iy * R
        rels = []
        for iy in range(R):
            for ix in range(R):
                rels.append([xs[ix].item(), ys[iy].item()])
        rel = torch.tensor(rels)          # [R², 2]
        basis = self._basis(rel, patch_width=pw, R=R)  # [R², R²]

        # Each row k = ix + iy*R should be a one-hot at basis index k
        row_max, row_argmax = basis.max(dim=1)
        assert (row_max - 1.0).abs().max().item() < 1e-5, \
            "Basis value at exact node position should be 1"
        expected_idx = torch.arange(R * R)
        assert (row_argmax == expected_idx).all(), \
            "Argmax of each row should match that node's linear index"

    def test_partition_of_unity_inside_patch(self):
        """For any interior point the 4 bilinear weights sum to exactly 1."""
        torch.manual_seed(0)
        pw = 1.0
        R  = 5
        # Random interior points (strictly inside patch)
        rel = (torch.rand(200, 2) - 0.5) * pw * 0.98   # [200, 2]
        basis = self._basis(rel, patch_width=pw, R=R)   # [200, R²]
        row_sums = basis.sum(dim=1)                     # [200]
        assert (row_sums - 1.0).abs().max().item() < 1e-5, \
            f"Partition of unity violated: max deviation = {(row_sums - 1.0).abs().max().item()}"

    def test_at_most_4_nonzero_per_row(self):
        """Each interior point activates exactly 4 basis functions (bilinear corners)."""
        torch.manual_seed(1)
        pw = 1.0
        R  = 5
        rel = (torch.rand(100, 2) - 0.5) * pw * 0.98
        basis = self._basis(rel, patch_width=pw, R=R)
        nnz = (basis > 0).sum(dim=1)
        assert (nnz <= 4).all(), "Each interior point should activate at most 4 basis fns"

    def test_centre_of_4_nodes_equal_weights(self):
        """The centre of a grid cell should get weight 0.25 from each of the 4 corners."""
        pw = 1.0
        R  = 5
        # Cell spacing = pw / (R-1)
        h = pw / (R - 1)
        # Centre of cell (0, 0): between nodes (0,0), (1,0), (0,1), (1,1) in grid coords
        # Physical position: x = -pw/2 + 0.5*h, y = -pw/2 + 0.5*h
        cx = -pw/2 + 0.5 * h
        cy = -pw/2 + 0.5 * h
        rel = torch.tensor([[cx, cy]])
        basis = self._basis(rel, patch_width=pw, R=R)   # [1, R²]
        nonzero = basis[0][basis[0] > 0]
        assert len(nonzero) == 4, f"Expected 4 nonzero weights, got {len(nonzero)}"
        assert (nonzero - 0.25).abs().max().item() < 1e-5, \
            f"Centred weights should all be 0.25, got {nonzero.tolist()}"

    def test_outside_patch_all_zeros(self):
        """Points outside [-pw/2, pw/2]² must give all-zero basis vectors."""
        pw = 1.0
        R  = 5
        outside = torch.tensor([
            [ 0.6,  0.0],   # x > pw/2
            [-0.6,  0.0],   # x < -pw/2
            [ 0.0,  0.6],   # y > pw/2
            [ 0.0, -0.6],   # y < -pw/2
            [ 1.0,  1.0],   # corner
        ])
        basis = self._basis(outside, patch_width=pw, R=R)
        assert basis.sum().item() == 0.0, \
            "Points outside patch should produce all-zero basis vectors"

    def test_nonnegative_weights(self):
        """All bilinear weights are nonnegative (convex combination property)."""
        torch.manual_seed(2)
        pw = 2.0
        R  = 7
        rel = (torch.rand(500, 2) - 0.5) * pw
        basis = self._basis(rel, patch_width=pw, R=R)
        assert (basis >= 0).all(), "All bilinear weights must be nonnegative"

    def test_gradient_flows(self):
        """Gradients must flow through the basis evaluation."""
        pw = 1.0
        R  = 5
        rel = torch.randn(20, 2) * 0.3
        rel.requires_grad_(True)
        basis = self._basis(rel, patch_width=pw, R=R)
        # A non-zero loss
        loss = basis.pow(2).sum()
        loss.backward()
        assert rel.grad is not None, "rel.grad should not be None"
        assert rel.grad.abs().sum().item() > 0, "Gradient through basis should be nonzero"

    def test_different_resolutions(self):
        """Partition of unity holds for R = 3, 5, 7, 9."""
        torch.manual_seed(3)
        pw = 1.0
        for R in [3, 5, 7, 9]:
            rel = (torch.rand(50, 2) - 0.5) * pw * 0.96
            basis = _eval_p1_basis(rel, pw, R)
            row_sums = basis.sum(dim=1)
            assert (row_sums - 1.0).abs().max().item() < 1e-5, \
                f"Partition of unity failed for R={R}"


# ---------------------------------------------------------------------------
# P0 basis tests
# ---------------------------------------------------------------------------

class TestP0Basis(unittest.TestCase):

    def test_one_hot_per_row(self):
        """Each interior edge activates exactly one cell (piecewise constant)."""
        torch.manual_seed(0)
        pw = 1.0
        R  = 5
        rel = (torch.rand(100, 2) - 0.5) * pw * 0.98
        basis = _eval_p0_basis(rel, pw, R)
        row_sums = basis.sum(dim=1)
        assert (row_sums == 1.0).all(), "P0 basis row should sum to exactly 1 for interior points"
        nnz = (basis > 0).sum(dim=1)
        assert (nnz == 1).all(), "P0 basis should have exactly 1 nonzero per interior row"

    def test_outside_patch_all_zeros(self):
        pw = 1.0
        R  = 5
        outside = torch.tensor([[0.6, 0.0], [-0.6, 0.0], [0.0, 0.7]])
        basis = _eval_p0_basis(outside, pw, R)
        assert basis.sum().item() == 0.0


# ---------------------------------------------------------------------------
# 2-hop edge expansion tests
# ---------------------------------------------------------------------------

class TestTwoHopExpansion(unittest.TestCase):

    def _simple_graph(self):
        """Line graph: 0-1-2-3-4 (directed, symmetric)."""
        N = 5
        edges = [[0,1],[1,0],[1,2],[2,1],[2,3],[3,2],[3,4],[4,3]]
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        coords = torch.stack([torch.linspace(0, 1, N),
                              torch.zeros(N)], dim=1)
        return ei, coords, N

    def _triangle(self):
        """Complete triangle: 0-1, 1-2, 0-2 (directed both ways)."""
        N = 3
        edges = [[0,1],[1,0],[1,2],[2,1],[0,2],[2,0]]
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        coords = torch.tensor([[0.,0.],[1.,0.],[0.5,1.]])
        return ei, coords, N

    def test_2hop_includes_1hop(self):
        """All original 1-hop edges must appear in the 2-hop expansion."""
        ei, coords, N = self._simple_graph()
        ei2, ea2 = _compute_2hop_edges(ei, coords, N)
        orig_keys = set(zip(ei[0].tolist(), ei[1].tolist()))
        new_keys  = set(zip(ei2[0].tolist(), ei2[1].tolist()))
        assert orig_keys.issubset(new_keys), "2-hop edges must include all 1-hop edges"

    def test_2hop_no_self_loops(self):
        """No self-loops in expanded graph."""
        ei, coords, N = self._simple_graph()
        ei2, _ = _compute_2hop_edges(ei, coords, N)
        assert (ei2[0] != ei2[1]).all(), "2-hop expansion must not create self-loops"

    def test_2hop_line_graph_degree(self):
        """On a line graph 0-1-2-3-4, node 2 (centre) should see 0,1,3,4 via 2-hop."""
        ei, coords, N = self._simple_graph()
        ei2, _ = _compute_2hop_edges(ei, coords, N)
        nbrs_2 = set(ei2[0][ei2[1] == 2].tolist())  # nodes sending TO node 2
        assert 0 in nbrs_2, "Node 0 should be 2-hop neighbour of node 2 via node 1"
        assert 4 in nbrs_2, "Node 4 should be 2-hop neighbour of node 2 via node 3"

    def test_relative_positions_correct(self):
        """Edge attributes should match coords[src] - coords[dst]."""
        ei, coords, N = self._simple_graph()
        ei2, ea2 = _compute_2hop_edges(ei, coords, N)
        expected_rel = coords[ei2[0]] - coords[ei2[1]]
        assert (ea2[:, :2] - expected_rel).abs().max().item() < 1e-5, \
            "Relative positions must equal coords[src] - coords[dst]"

    def test_distance_column_correct(self):
        """Third column of edge_attr should be Euclidean distance."""
        ei, coords, N = self._simple_graph()
        ei2, ea2 = _compute_2hop_edges(ei, coords, N)
        computed_dist = ea2[:, :2].norm(dim=1)
        assert (ea2[:, 2] - computed_dist).abs().max().item() < 1e-5, \
            "Distance column must be norm of (dx, dy)"

    def test_triangle_all_2hop(self):
        """On a complete triangle every node is already a 1-hop neighbour;
        2-hop expansion should not add any new edges."""
        ei, coords, N = self._triangle()
        ei2, _ = _compute_2hop_edges(ei, coords, N)
        # Triangle is already a complete directed graph (minus self-loops)
        # So E' == E
        assert ei2.shape[1] == ei.shape[1], \
            "Complete triangle: 2-hop expansion should not add edges"

    def test_expand_to_2hops_fallback_matches_compute(self):
        """_expand_to_2hops (path-based) gives the same connectivity as _compute_2hop_edges."""
        ei, coords, N = self._simple_graph()
        src, dst = ei[0], ei[1]
        rel = coords[src] - coords[dst]
        dist = rel.norm(dim=-1, keepdim=True)
        ea = torch.cat([rel, dist], dim=1)

        ei_a2,  _  = _compute_2hop_edges(ei, coords, N)
        ei_fb,  _  = _expand_to_2hops(ei, ea, N)

        keys_a2 = set(zip(ei_a2[0].tolist(), ei_a2[1].tolist()))
        keys_fb = set(zip(ei_fb[0].tolist(), ei_fb[1].tolist()))
        assert keys_a2 == keys_fb, \
            "A²-based and path-based 2-hop expansion must give the same edge set"


# ---------------------------------------------------------------------------
# FEMConvLayer integration tests
# ---------------------------------------------------------------------------

class TestFEMConvLayerIntegration(unittest.TestCase):

    def _make_simple_graph(self, N=16, H=32):
        """Regular 4×4 grid graph with random node features."""
        torch.manual_seed(42)
        # Build grid edges
        edges = []
        for i in range(4):
            for j in range(4):
                node = i * 4 + j
                for di, dj in [(0,1),(1,0),(0,-1),(-1,0)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < 4 and 0 <= nj < 4:
                        edges.append([node, ni*4+nj])
        ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
        coords = torch.stack([
            torch.arange(N).float() % 4 / 3.0,
            torch.arange(N).float() // 4 / 3.0
        ], dim=1)
        src, dst = ei[0], ei[1]
        rel  = coords[src] - coords[dst]
        dist = rel.norm(dim=-1, keepdim=True)
        ea   = torch.cat([rel, dist], dim=1)
        h    = torch.randn(N, H)
        return h, ei, ea, N, H

    def test_forward_shape(self):
        """Output shape matches input shape."""
        h, ei, ea, N, H = self._make_simple_graph()
        layer = FEMConvLayer(H, edge_dim=3, mixing_type='vector', k_hops=1)
        out = layer(h, ei, edge_attr=ea)
        assert out.shape == h.shape

    def test_forward_k2_shape(self):
        h, ei, ea, N, H = self._make_simple_graph()
        layer = FEMConvLayer(H, edge_dim=3, mixing_type='vector', k_hops=2)
        out = layer(h, ei, edge_attr=ea)
        assert out.shape == h.shape

    def test_gradient_k1(self):
        """Gradient must flow through FEMConvLayer with k_hops=1."""
        h, ei, ea, N, H = self._make_simple_graph()
        layer = FEMConvLayer(H, edge_dim=3, mixing_type='vector', k_hops=1)
        out = layer(h, ei, edge_attr=ea)
        out.pow(2).mean().backward()
        assert layer.geo_to_gate.weight.grad is not None
        assert layer.geo_to_gate.weight.grad.abs().sum().item() > 0

    def test_gradient_k2(self):
        """Gradient must flow through FEMConvLayer with k_hops=2."""
        h, ei, ea, N, H = self._make_simple_graph()
        layer = FEMConvLayer(H, edge_dim=3, mixing_type='vector', k_hops=2)
        out = layer(h, ei, edge_attr=ea)
        out.pow(2).mean().backward()
        assert layer.geo_to_gate.weight.grad is not None
        assert layer.geo_to_gate.weight.grad.abs().sum().item() > 0

    def test_all_mixing_types(self):
        """All three mixing types produce correct output shape."""
        h, ei, ea, N, H = self._make_simple_graph()
        for mt in ('scalar', 'vector', 'lowrank'):
            layer = FEMConvLayer(H, edge_dim=3, mixing_type=mt, k_hops=1)
            out = layer(h, ei, edge_attr=ea)
            assert out.shape == h.shape, f"Shape mismatch for mixing_type={mt}"

    def test_p0_and_p1_basis_types(self):
        """Both P0 and P1 basis types work end-to-end."""
        h, ei, ea, N, H = self._make_simple_graph()
        for bt in ('p0', 'p1'):
            layer = FEMConvLayer(H, edge_dim=3, fem_basis_type=bt, k_hops=1)
            out = layer(h, ei, edge_attr=ea)
            assert out.shape == h.shape, f"Shape mismatch for fem_basis_type={bt}"

    def test_2hop_cache_reused(self):
        """The 2-hop cache should be built once; second forward pass uses it."""
        h, ei, ea, N, H = self._make_simple_graph()
        layer = FEMConvLayer(H, edge_dim=3, k_hops=2)
        _ = layer(h, ei, edge_attr=ea)          # first call: builds cache
        assert hasattr(layer, '_2hop_ei'), "Cache should be populated after first forward"
        cached_ei = layer._2hop_ei
        _ = layer(h, ei, edge_attr=ea)          # second call: reuses cache
        assert layer._2hop_ei is cached_ei, "Same cache object should be reused"


if __name__ == '__main__':
    import unittest
    # Run all tests
    suites = [
        unittest.TestLoader().loadTestsFromTestCase(c)
        for c in [TestP1Basis, TestP0Basis, TestTwoHopExpansion, TestFEMConvLayerIntegration]
    ]
    runner = unittest.TextTestRunner(verbosity=2)
    for s in suites:
        runner.run(s)
