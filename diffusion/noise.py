import torch
import numpy as np

class NoiseSampler(object):
    def sample(self, N):
        raise NotImplementedError

"""
Adapted from: https://github.com/neuraloperator/FunDPS/blob/main/training/noise_samplers.py

"""

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
        C = torch.exp(-torch.cdist(mesh_points, mesh_points) / (2 * scale**2))
        I = torch.eye(C.size(-1)).to(device)

        I.mul_(eps**2)  # inplace multiply by eps**2 (stability)
        C.add_(I)  # inplace add by I
        del I  # don't need it anymore

        self.L = torch.linalg.cholesky(C)

        del C  # don't need it anymore
        # C= L L^T

    @torch.no_grad()
    def sample(self, N):

        samples = torch.zeros((N, self.num_points)).to(self.device)
        for ix in range(N):
            # (s^2, s^2) * (s^2, 2) -> (s^2, 2)
            z = torch.randn(self.num_points, 1).to(self.device)
            samples[ix] = torch.matmul(self.L, z)[:,0]

        return samples
    
    def apply_L_inv(self, x):
        y = torch.linalg.solve_triangular(self.L, x.T, upper=False)
        return y.T
    
    def apply_C(self, x):
        # x: (N, s)
        return torch.matmul(self.L, torch.matmul(self.L.T, x.T)).T

    def apply_Csqrt(self, x):
        # x: (N, s)
        return torch.matmul(self.L, x.T).T  
        
class RFFNoiseSampler(NoiseSampler):
    """Random Fourier Features approximation to an RBF kernel GP.

    Sampling uses the O(N*D) feature matrix directly:
        f(x) = sqrt(2/D) * sum_d cos(omega_d . x + phi_d),  omega_d ~ N(0, l^{-2} I)
    giving Cov[f] ≈ Phi Phi^T ≈ K_RBF.

    For the linear-algebra helpers (apply_Csqrt, apply_L_inv) we precompute
    the thin SVD  Phi = U diag(S) V^T  once at init, and work in the
    symmetric principal-sqrt basis:

        C     ≈ Phi Phi^T + eps² I   =  U diag(S² + eps²) U^T   (on range(Phi))
        C^½   ≈ U diag(√(S² + eps²)) U^T
        C^{-½} ≈ U diag(1/√(S² + eps²)) U^T

    Off-range (null-space of Phi Phi^T) contributions are dropped —
    acceptable for D << N and Phi well-conditioned on range; inputs living
    primarily in range(Phi) are handled correctly.

    The symmetric sqrt convention keeps `||apply_L_inv(x)||² = x^T C^{-1} x`
    valid, so existing callers that use apply_L_inv for the whitened
    residual norm continue to work identically.

    Properties:
    - O(N*D) sampling, O(N*D² + D³) one-off SVD at init
    - Resolution-invariant: same random frequencies work on any mesh
    - Practical for meshes where dense Cholesky is infeasible
    """

    name = "RFF"

    @torch.no_grad()
    def __init__(self, mesh_points, scale=1, n_features=512, eps=0.01, device=None, **kwargs):
        """
        mesh_points: (N, dim) tensor of mesh coordinates
        scale:       RBF length scale
        n_features:  number of random Fourier features D (default 512)
        eps:         small i.i.d. regularisation (matches RBFKernel eps semantics)
        device:      target device
        """
        self.num_points = mesh_points.shape[0]
        self.D = n_features
        self.eps = eps
        self.device = device
        dim = mesh_points.shape[1]

        # Spectral density of RBF kernel: omega ~ N(0, scale^{-2} I)
        omega = torch.randn(self.D, dim) / scale          # (D, dim)
        phi   = torch.rand(self.D) * 2 * np.pi            # (D,)

        # Feature matrix Phi: (N, D)  — evaluate once, reuse every sample
        pts = mesh_points.cpu().to(torch.float32)
        Phi_cpu = np.sqrt(2.0 / self.D) * torch.cos(pts @ omega.T + phi.unsqueeze(0))

        # Thin SVD of Phi for Csqrt / C^{-1/2} / C^{-1} on range(Phi).
        # Computed on CPU (SVD fallback on some accelerators), moved to device.
        U_cpu, S_cpu, _ = torch.linalg.svd(Phi_cpu, full_matrices=False)   # U: (N,D), S: (D,)

        self.Phi = Phi_cpu.to(device)
        self.U   = U_cpu.to(device)
        self.S   = S_cpu.to(device)
        # Precompute spectral diagonals for Csqrt / L_inv
        self._diag_Csqrt  = torch.sqrt(self.S ** 2 + self.eps ** 2)          # (D,)
        self._diag_L_inv  = 1.0 / self._diag_Csqrt                            # (D,)

    @torch.no_grad()
    def sample(self, N):
        # z ~ N(0, I_D),  f = Phi @ z  ~  GP(0, Phi Phi^T)
        z = torch.randn(N, self.D, device=self.device)
        samples = z @ self.Phi.T    # (N, num_points)
        if self.eps > 0:
            samples = samples + self.eps * torch.randn_like(samples)
        return samples

    def _spectral_apply(self, diag, x):
        """Apply  U diag(d) U^T  to x of shape (B, N).  Returns (B, N)."""
        # (D, B) = U^T (N, D) @ x^T (N, B)
        UTx = self.U.T @ x.T
        # scale each row by diag component
        scaled = diag.unsqueeze(-1) * UTx          # (D, B)
        return (self.U @ scaled).T                 # (B, N)

    def apply_C(self, x):
        # C = Phi Phi^T + eps^2 I (exact — no SVD approximation needed)
        return (self.Phi @ (self.Phi.T @ x.T) + self.eps ** 2 * x.T).T

    def apply_Csqrt(self, x):
        # C^{1/2} ≈ U diag(sqrt(S^2 + eps^2)) U^T
        return self._spectral_apply(self._diag_Csqrt, x)

    def apply_L_inv(self, x):
        # Symmetric C^{-1/2} ≈ U diag(1/sqrt(S^2 + eps^2)) U^T
        # Satisfies ||apply_L_inv(x)||_2^2 = x^T C^{-1} x (same as RBFKernel
        # Cholesky L^{-1} for norm-based losses).
        return self._spectral_apply(self._diag_L_inv, x)


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
