"""
Package smoke test: every core module imports cleanly and the FEM-conv layer
(the paper's core contribution) runs a forward pass.

Run:
    python -m pytest tests/test_imports.py -v
"""
import os
import sys
import importlib

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CORE_MODULES = [
    "config", "model_mesh_utils", "data_utils",
    "diffusion.precond", "diffusion.sde", "diffusion.noise", "diffusion.embedding",
    "models.multiscale", "models.fem_conv", "models.fno", "models.cnn", "models.gaot",
    "pde_operators.poisson", "pde_operators.sensors",
    "eval.sampling",
]


@pytest.mark.parametrize("module", CORE_MODULES)
def test_module_imports(module):
    importlib.import_module(module)


def _make_coords(n):
    import torch
    return torch.rand(n, 2)


def test_femconv_forward_pass():
    import torch
    from models.fem_conv import FEMConvLayer

    # 4x4 grid graph (matches tests/test_fem_basis.py construction).
    edges = []
    for i in range(4):
        for j in range(4):
            n = i * 4 + j
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < 4 and 0 <= nj < 4:
                    edges.append([n, ni * 4 + nj])
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    coords = torch.stack([torch.arange(16).float() % 4 / 3.0,
                          torch.arange(16).float() // 4 / 3.0], dim=1)
    rel = coords[ei[0]] - coords[ei[1]]
    ea = torch.cat([rel, rel.norm(dim=-1, keepdim=True)], dim=1)

    h = torch.randn(16, 32)
    out = FEMConvLayer(32, edge_dim=3, mixing_type="vector", k_hops=1)(h, ei, edge_attr=ea)
    assert out.shape == h.shape


def test_cnn_forward_pass():
    import torch
    from models.cnn import CNNScoreNetwork
    N = 32
    coords = _make_coords(N)
    model = CNNScoreNetwork(input_dim=1, pos_dim=2, hidden_dim=16, time_dim=16,
                            n_levels=2, grid_h=4, grid_w=4)
    model.set_mesh_hierarchy(None, [N], None, None, level_coords=[coords])
    inp = torch.randn(2, 3, N)
    out = model(inp, torch.rand(2), pos=None)
    assert out.shape == (2, 1, N)


def test_fno_forward_pass():
    pytest.importorskip("neuralop", reason="neuraloperator not installed")
    import torch
    from models.fno import FNOScoreNetwork
    N = 32
    coords = _make_coords(N)
    model = FNOScoreNetwork(input_dim=1, pos_dim=2, hidden_dim=16, time_dim=16,
                            n_modes=4, n_fno_layers=2, grid_h=4, grid_w=4)
    model.set_mesh_hierarchy(None, [N], None, None, level_coords=[coords])
    inp = torch.randn(2, 3, N)
    out = model(inp, torch.rand(2), pos=None)
    assert out.shape == (2, 1, N)


def test_gaot_forward_pass():
    if not os.environ.get("GAOT_REPO"):
        pytest.skip("GAOT_REPO not set")
    import torch
    from models.gaot import GAOTScoreNetwork
    N = 32
    coords = _make_coords(N)
    model = GAOTScoreNetwork(input_dim=1, pos_dim=2, hidden_dim=64, time_dim=32,
                             latent_tokens_size=[4, 4], patch_size=2,
                             n_transformer_layers=1, magno_radius=0.5)
    model.set_mesh_hierarchy(None, [N], None, None, level_coords=[coords])
    inp = torch.randn(2, 3, N)
    out = model(inp, torch.rand(2), pos=None)
    assert out.shape == (2, 1, N)
