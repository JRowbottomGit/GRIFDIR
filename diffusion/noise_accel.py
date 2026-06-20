"""
Accelerated drop-in replacement for the RBFKernel in noise.py (original kept untouched for compatibility).

Adapted from: https://github.com/neuraloperator/FunDPS/blob/main/training/noise_samplers.py

Three speedups over the original RBFKernel on Intel XPU (benchmarked ~2.5x epoch time):

1. Cholesky on CPU, once at init.
   torch.linalg.cholesky falls back from XPU to CPU silently. We do it explicitly on CPU
   and move the result to device, so there is no fallback warning and no hidden round-trip
   on every call.

2. L_inv precomputed at init (triangular solve -> matmul at runtime).
   apply_L_inv() was calling torch.linalg.solve_triangular every batch, which also falls
   back to CPU on XPU. We instead solve L @ L_inv = I once on CPU at init, store L_inv on
   device, and replace the runtime solve with a plain matmul — which runs natively on XPU.
   The gradient through matmul is also faster than through solve_triangular.

3. Vectorised sample().
   The original sample() loops over N in Python and calls a separate matmul per sample.
   Here we draw all N noise vectors at once (shape s x N) and do a single batched matmul:
   L @ z -> (s, N) -> transpose -> (N, s). No Python loop, one kernel launch.
"""

import torch 
import numpy as np 

class NoiseSampler(object):
    def sample(self, N):
        raise NotImplementedError

class RBFKernel(NoiseSampler):
    """
    This sampler generates noise for a fixed discretization grid using an RBF kernel.
    
    """
    @torch.no_grad()
    def __init__(self, mesh_points, scale=1, eps=0.01, device=None):
        """
        mesh_points: (s, 2) tensor of mesh coordinates with s being the number of spatial points
        
        """

        self.num_points = mesh_points.shape[0]
        self.device = device
        self.scale = scale

        # (s^2, 2)
        # (s^2, s^2)
        # Build covariance on CPU (cholesky not supported on XPU)
        mesh_cpu = mesh_points.cpu()
        C = torch.exp(-torch.cdist(mesh_cpu, mesh_cpu) / (2 * scale**2))
        C.add_(torch.eye(C.size(-1)) * eps**2)

        L_cpu = torch.linalg.cholesky(C)  # C = L L^T, computed on CPU
        # Precompute L^{-1} once so apply_L_inv is a matmul (no triangular solve at runtime)
        L_inv_cpu = torch.linalg.solve_triangular(L_cpu, torch.eye(L_cpu.size(0)), upper=False)

        self.L = L_cpu.to(device)      # (s, s) on device
        self.L_inv = L_inv_cpu.to(device)  # (s, s) on device
        # C = L L^T

    @torch.no_grad()
    def sample(self, N):
        # Vectorised: draw all N samples in one batched matmul (no Python loop)
        z = torch.randn(self.num_points, N, device=self.device)
        return torch.matmul(self.L, z).T  # (N, s)
    
    def apply_L_inv(self, x):
        # Pure matmul — no triangular solve, runs natively on XPU
        return torch.matmul(self.L_inv, x.T).T
    
    def apply_C(self, x):
        # x: (N, s)
        return torch.matmul(self.L, torch.matmul(self.L.T, x.T)).T

    def apply_Csqrt(self, x):
        # x: (N, s)
        return torch.matmul(self.L, x.T).T  
        
if __name__ == "__main__":
    # create random mesh points in a [0,1]^2 domain
    mesh_points = torch.rand(2000, 2)

    sampler2 = RBFKernel(mesh_points, scale=0.1, eps=0.01, device='cpu')
    samples2 = sampler2.sample(4)

    print(samples2.shape)  # should be (4, 2000)      
    assert samples2.shape == (4, 2000), "Shape mismatch for RBF kernel noise sampler"

    y = sampler2.apply_L_inv(samples2)
    print(y.shape)

    y = sampler2.apply_C(samples2)
    print("apply C: ", y.shape)

    y = sampler2.apply_Csqrt(samples2)
    print("apply C sqrt: ", y.shape)
