"""
Shared sampling utilities for GRIFDIR score models.

Used by:
  - train.py          (periodic generative eval during training)
  - uncond_sampling_gnn.py (standalone unconditional sampling)
  - dps_reconstruction.py  (DPS sensor-guided reconstruction)

Two entry points:
  sample_unconditional(score_model, pos, n_samples, n_steps, device)
      -> samples [n_samples, 1, N], fig

  run_dps_eval(score_model, pos, mesh_pos, gt_field, n_sensors,
               n_steps, guidance_weight, sde, noise_sampler, device)
      -> x_recon [1, N], metrics dict, fig
"""

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from pde_operators.sensors import SparseSensorOperator

# PoissonOperator requires dolfinx/mpi4py/petsc — import lazily to avoid
# crashing on clusters where only torch+pyg are installed.
# Usage: from pde_operators.poisson import PoissonOperator  (at call site)

# ---------------------------------------------------------------------------
# Unconditional reverse-diffusion sampler (exponential integrator)
# ---------------------------------------------------------------------------

def _reverse_step_uncond(score_model, xt, t_scalar, pos_batch, sde, noise_sampler):
    """Single exponential-integrator reverse step, no gradients."""
    batch_size = xt.shape[0]
    device = xt.device
    t = torch.ones(batch_size, device=device) * t_scalar

    with torch.no_grad():
        score, _ = score_model(xt, t, pos_batch)
        beta_t = sde.beta_t(t).view(-1, 1, 1)
        noise = noise_sampler.sample(batch_size).unsqueeze(1)
        phi_t = torch.exp(0.5 * beta_t)  # delta_t = 1/n_steps folded in below
        return score, beta_t, noise, phi_t


def sample_unconditional(score_model, pos, n_samples: int, n_steps: int, device,
                         pos_batch=None, domain_onehot=None):
    """
    Run unconditional reverse diffusion via exponential integrator.

    Args:
        score_model: ScoreModel wrapper
        pos:         [N, 2] mesh positions
        n_samples:   number of samples to generate
        n_steps:     number of reverse-diffusion steps
        device:      torch device
        pos_batch:   [B, N, D] pre-built position+conditioning tensor (optional)

    Returns:
        samples: [n_samples, 1, N] tensor (on CPU)
    """
    sde = score_model.sde
    noise_sampler = score_model.noise_sampler

    ts = torch.linspace(1e-3, 1.0, n_steps, device=device)
    delta_t = ts[1] - ts[0]

    if pos_batch is None:
        pos_batch = pos.unsqueeze(0).expand(n_samples, -1, -1)  # [B, N, 2]
    xt = noise_sampler.sample(n_samples).unsqueeze(1)        # [B, 1, N]

    for ti in tqdm(reversed(ts), total=n_steps, desc="Uncond sampling", leave=False):
        t = torch.ones(n_samples, device=device) * ti
        with torch.no_grad():
            score, _ = score_model(xt, t, pos_batch, domain_onehot=domain_onehot)
            beta_t = sde.beta_t(t).view(-1, 1, 1)
            noise = noise_sampler.sample(n_samples).unsqueeze(1)
            phi_t = torch.exp(0.5 * beta_t * delta_t)
            xt = phi_t * xt + 2 * (phi_t - 1) * score + (phi_t**2 - 1).sqrt() * noise

    return xt.detach().cpu()


# ---------------------------------------------------------------------------
# VE / EDM unconditional sampler  (Heun ODE)
# ---------------------------------------------------------------------------

def _ve_sigma_schedule(sigma_max: float, sigma_min: float, n_steps: int, device):
    """Geometric (log-spaced) sigma schedule, sigma_max → sigma_min, with 0 appended."""
    sigmas = torch.logspace(
        torch.log10(torch.tensor(sigma_max)).item(),
        torch.log10(torch.tensor(sigma_min)).item(),
        n_steps,
        device=device,
    )
    return torch.cat([sigmas, torch.zeros(1, device=device)])


def sample_ve_heun(edm_model, pos, n_samples: int, n_steps: int, device,
                   pos_batch=None, domain_onehot=None):
    """
    Unconditional sampling for VE/EDM models via Heun ODE.

    Args:
        edm_model:  EDMDenoiser wrapper (has .cfg.sde.sigma_max/min, .noise_sampler)
        pos:        [N, 2] mesh positions
        n_samples:  number of samples
        n_steps:    number of sigma levels
        pos_batch:  [B, N, D] pre-built position+conditioning tensor (optional)

    Returns:
        samples: [n_samples, 1, N] tensor (CPU)
    """
    noise_sampler = edm_model.noise_sampler
    sigma_max = edm_model.cfg.sde.sigma_max
    sigma_min = edm_model.cfg.sde.sigma_min

    sigmas = _ve_sigma_schedule(sigma_max, sigma_min, n_steps, device)
    if pos_batch is None:
        pos_batch = pos.unsqueeze(0).expand(n_samples, -1, -1)

    xt = sigmas[0] * noise_sampler.sample(n_samples).unsqueeze(1)

    for i in tqdm(range(len(sigmas) - 1), desc="VE Heun sampling", leave=False):
        sigma_cur  = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_t = torch.full((n_samples, 1, 1), sigma_cur.item(), device=device)

        with torch.no_grad():
            x0_pred = edm_model(xt, sigma_t, pos_batch, domain_onehot=domain_onehot)

        if sigma_cur > 1e-8:
            score = (xt - x0_pred) / (sigma_cur + 1e-8)
            dt = sigma_next - sigma_cur
            xt_next = xt + dt * score

            if sigma_next > 1e-8:
                sigma_t_next = torch.full((n_samples, 1, 1), sigma_next.item(), device=device)
                with torch.no_grad():
                    x0_pred_next = edm_model(xt_next, sigma_t_next, pos_batch,
                                             domain_onehot=domain_onehot)
                score_next = (xt_next - x0_pred_next) / (sigma_next + 1e-8)
                xt_next = xt + dt * (0.5 * score + 0.5 * score_next)
        else:
            xt_next = x0_pred

        xt = xt_next

    return xt.detach().cpu()


# ---------------------------------------------------------------------------
# VE / EDM DPS sensor-guided sampler  (Heun ODE + guidance)
# ---------------------------------------------------------------------------

def run_dps_eval_ve(y_obs, 
                    batch_size, 
                    edm_model, 
                    pos, 
                    mesh_pos, 
                    gt_field,
                    forward_op, 
                    n_steps: int, 
                    guidance_weight: float, 
                    device,
                    guidance_schedule: str = "linear", # not used
                    tri=None, 
                    pos_batch=None, 
                    scale_grad: bool = False): # whether to scale the guidance gradient by the measurement residual (as in VP-DPS) or not
    """
    DPS reconstruction for VE/EDM models via Heun ODE with guidance correction.

    This should hopefully the algorithm in FunDPS: 

        Yao et al. "Guided Diffusion Sampling on Function Spaces with Applications to PDEs" 2026

    Algorithm 2 in https://arxiv.org/pdf/2505.17004

    At each sigma level we:
        1. Predict x0 = D(x_t, σ)  with grad w.r.t. x_t
        2. Compute measurement residual and gradient
        3. Apply first-order ODE step + guidance, then Heun correction

    Returns:
        x_recon:       [1, N] reconstruction (CPU)
        metrics:       dict(mse, rel_l2, meas_mse)
        sensor_coords: [K, 2] numpy array
        
    """
    noise_sampler = edm_model.noise_sampler
    sigma_max = edm_model.cfg.sde.sigma_max
    sigma_min = edm_model.cfg.sde.sigma_min
    N = pos.shape[0]


    sigmas = _ve_sigma_schedule(sigma_max, sigma_min, n_steps, device)
    if pos_batch is None:
        pos_batch = pos.unsqueeze(0)
        pos_batch = pos_batch.repeat(batch_size, 1, 1)  # [batch_size, N, 2]

    xt = sigmas[0] * noise_sampler.sample(batch_size).unsqueeze(1)

    for i in tqdm(range(len(sigmas) - 1), desc="VE DPS eval", leave=False):
        sigma_cur  = sigmas[i]
        sigma_next = sigmas[i + 1]
        #sigma_t = torch.full((1, 1, 1), sigma_cur.item(), device=device)
        sigma_t = torch.full((batch_size, 1, 1), sigma_cur.item(), device=device)

        # --- guidance step (needs grad) ---
        xt = xt.detach().requires_grad_(True)
        x0_pred = edm_model(xt, sigma_t, pos_batch)

        #residual_norm = meas_loss.sqrt().clamp(min=1e-8).detach()
        #grad_precond = grad_precond / residual_norm

        # normalise ti to [0,1] using log-sigma position in schedule
        #ti_norm = float(i) / max(len(sigmas) - 2, 1)
        #w_t = _scheduled_w(guidance_weight, 1.0 - ti_norm, guidance_schedule)
        w_t = guidance_weight

        # --- Unconditional Heun step ---
        if sigma_cur > 1e-8:
            score = (xt - x0_pred) / (sigma_cur + 1e-8)
            dt = sigma_next - sigma_cur
            xt_next = xt + dt * score  # Euler predictor (unconditional)

            # Heun corrector (unconditional)
            if sigma_next > 1e-8:
                sigma_t_next = torch.full((batch_size, 1, 1), sigma_next.item(), device=device)
                x0_pred = edm_model(xt_next, sigma_t_next, pos_batch)
                score_next = (xt_next - x0_pred) / (sigma_next + 1e-8)
                xt_next = xt + dt * (0.5 * score + 0.5 * score_next)

            # compute data consistency loss
            y_pred = forward_op.forward(x0_pred)
            meas_loss_per_sample = ((y_pred - y_obs) ** 2).reshape(batch_size, -1).sum(dim=-1)
            meas_loss = meas_loss_per_sample.sum()   # scalar for autograd; per-sample grads still correct

            grad = torch.autograd.grad(meas_loss, xt)[0]

            grad_precond = noise_sampler.apply_C(grad.squeeze(1)).unsqueeze(1)
            if scale_grad:
                residual_norm = meas_loss_per_sample.sqrt().clamp(min=1e-8).detach().view(-1, 1, 1)
                grad_precond = grad_precond / residual_norm
            # --- Apply guidance correction at the end ---
            xt_next = xt_next - w_t * grad_precond

        else:
            xt_next = x0_pred



        xt = xt_next.detach()

    x_recon = xt.detach().cpu()
    with torch.no_grad():
        mse = torch.mean((x_recon - gt_field.cpu()) ** 2).item()
        rel_l2 = (torch.norm(x_recon - gt_field.cpu()) / torch.norm(gt_field.cpu())).item()
        y_recon = forward_op.forward(x_recon.to(device))
        meas_mse = torch.mean((y_recon - y_obs) ** 2).item()

    metrics = {"mse": mse, "rel_l2": rel_l2, "meas_mse": meas_mse}
    return x_recon, metrics


# ---------------------------------------------------------------------------
# VE / EDM DAPS sensor-guided sampler  (DDIS-DAPS)
# ---------------------------------------------------------------------------

def run_daps_eval_ve(y_obs, batch_size, edm_model, pos, mesh_pos, gt_field,
                     forward_op, n_steps: int, device,
                     beta_y: float = 0.1,
                     langevin_steps: int = 10,
                     alpha: float = 0.5, # langevin step size decay exponent (eta_t = eta / (1 + iter^alpha))
                     tri=None, pos_batch=None):
    """
    DAPS (DDIS-DAPS) reconstruction for VE/EDM models.

    Prior: N(0, sigma_max^2 * C).  Initialisation follows the VE convention:
        a_N ~ sigma_max * N(0, C)   i.e.  sigmas[0] * noise_sampler.sample(1)

    At each outer sigma level (sigma_cur → sigma_next):
        1. x0^(0) = D(a_i, sigma_cur)                           [denoiser]
        2. For j = 0 .. N_L-1:                                  [Langevin]
               g_prior = -(1/sigma_cur^2) * (x0^(j) - x0^(0))
               g_like  = -(1/beta_y^2) * nabla_{x0} 0.5*||M(x0^(j)) - y_obs||^2
               x0^(j+1) = x0^(j) + eta * C(g_prior + g_like) + sqrt(2*eta) * eps_j
        3. a_{i-1} = x0^(N_L) + sigma_next * xi_i              [re-noising]

    Args:
        edm_model:          EDMDenoiser wrapper (has .cfg.sde.sigma_max/min, .noise_sampler).
        pos:                [N, 2] mesh positions (on device).
        mesh_pos:           [N, 2] numpy positions (unused, kept for API symmetry).
        gt_field:           [1, 1, N] ground-truth tensor (on device).
        forward_op:         Measurement operator with .forward().
        n_steps:            Number of sigma levels.
        device:             torch device.
        beta_y:             Likelihood noise scale.
        langevin_steps:     N_L inner Langevin iterations per outer step.
        tri:                Optional Triangulation for per-step debug plots.
        pos_batch:          [1, N, D] pre-built position tensor (optional).

    Returns:
        x_recon:  [1, 1, N] reconstruction (CPU).
        metrics:  dict(mse, rel_l2, meas_mse).
    """
    noise_sampler = edm_model.noise_sampler
    sigma_max = edm_model.cfg.sde.sigma_max
    sigma_min = edm_model.cfg.sde.sigma_min

    sigmas = _ve_sigma_schedule(sigma_max, sigma_min, n_steps, device)
    if pos_batch is None:
        pos_batch = pos.unsqueeze(0)
        pos_batch = pos_batch.repeat(batch_size, 1, 1)  # [batch_size, N, 2]
    # Initialise from N(0, sigma_max^2 * C)
    ai = sigmas[0] * noise_sampler.sample(batch_size).unsqueeze(1)  # [1, 1, N]

    
    # power iteration to estimate norm of forward operator A for step size bound (optional, can just use eta = 1e-3 or so)
    with torch.no_grad():
        x = noise_sampler.sample(1).unsqueeze(1)  # [1, 1, N]
        for _ in range(50):            

            x = forward_op.adjoint(forward_op.forward(x))
            x = x / x.norm()
        norm_A = x.norm().item()


    with torch.no_grad():
        v = noise_sampler.sample(1).unsqueeze(1)
        for _ in range(40):
            v = noise_sampler.apply_C(v.squeeze(1)).unsqueeze(1)
            norm_C = v.norm().item()
            v = v / norm_C
    for i in tqdm(range(len(sigmas) - 1), desc="DAPS eval", leave=False):
        sigma_cur  = sigmas[i]
        sigma_next = sigmas[i + 1]
        sigma_t = torch.full((batch_size, 1, 1), sigma_cur.item(), device=device)
        sigma_cur_sq = sigma_cur.item() ** 2

        # we can actually compute a good step size bound
        #norm_A = 1.0 # sparse observation has norm 1
        #norm_C = noise_sampler.scale  # scalar, largest eigenvalue of C
        #print("norm_C: ", norm_C)
        eta = 0.5 / (1 / sigma_cur_sq + norm_C * norm_A ** 2 / beta_y ** 2)
        # ── Step 1: denoiser prediction ──────────────────────────────────────
        with torch.no_grad():
            x0_0 = edm_model(ai, sigma_t, pos_batch)  # [1, 1, N]

        # ── Step 2: inner Langevin loop ───────────────────────────────────────
        x0_j = x0_0.detach().clone()

        
        for langevin_iter in range(langevin_steps):
            eta_t = eta / (1  + langevin_iter ** alpha) 
            #print("eta_t: ", eta_t)

            eps_j = noise_sampler.sample(batch_size).unsqueeze(1)  # [1, 1, N], ~ N(0, C)

            x0_j = x0_j.detach().requires_grad_(True)
            y_pred    = forward_op.forward(x0_j)
            meas_loss = torch.sum((y_pred - y_obs) ** 2) / (2.0 * beta_y ** 2)

            # the forward operator is linear, and as .adjoint, so we do not have to use backprop through it; we can directly apply the adjoint to the residual to get the gradient w.r.t. x0_j
            # g_like = -forward_op.adjoint((y_pred - y_obs).squeeze()) / (beta_y ** 2)  # [1, N]
            
            g_like    = -torch.autograd.grad(meas_loss, x0_j)[0]  # [1, 1, N]
            x0_j    = x0_j.detach()
            g_prior = -(x0_j - x0_0.detach()) / sigma_cur_sq      # [1, 1, N]
            # I have p(x0 | xt ) = N(x0; x0_0, sigma_cur^2 * C), so the prior gradient is ∇_{x0} log p(x0 | xt) = -C^{-1}(x0 - x0_0) / sigma_cur^2, where the 1/sigma_cur^2 comes from the covariance of the Gaussian. We can apply C^{-1} to this gradient to get a preconditioned gradient, which is what we use for the Langevin update. The likelihood gradient is already in the right form since it is derived from the measurement loss.
            # but using the preconditioner C, this goes away
            
            # print("g_like shape: ", g_like.shape, "  g_prior shape: ", g_prior.shape)

            g_like = noise_sampler.apply_C(g_like.squeeze(1)).unsqueeze(1)  # [1, 1, N]

            g_total = g_prior + g_like
            C_g     = g_total #noise_sampler.apply_C(g_total.squeeze(1)).unsqueeze(1)  # [1, 1, N]

            x0_j = x0_j + eta_t * C_g + (2.0 * eta_t) ** 0.5 * eps_j
            #print("is nan:", torch.isnan(x0_j).any().item(), "  grad norm:", g_total.norm().item())
            #print("sigma_cur_sq: ", sigma_cur_sq)

        #print("is nan:", torch.isnan(x0_j).any().item(), "  grad norm:", g_total.norm().item())
        # ── Step 3: re-noise to sigma_next ───────────────────────────────────
        xi_i = noise_sampler.sample(batch_size).unsqueeze(1)  # [1, 1, N], ~ N(0, C)
        ai   = x0_j.detach() + sigma_next * xi_i

    x_recon = ai.detach().cpu()

    with torch.no_grad():
        mse      = torch.mean((x_recon - gt_field.cpu()) ** 2).item()
        rel_l2   = (torch.norm(x_recon - gt_field.cpu()) / torch.norm(gt_field.cpu())).item()
        y_recon  = forward_op.forward(x_recon.to(device))
        meas_mse = torch.mean((y_recon - y_obs) ** 2).item()

    metrics = {"mse": mse, "rel_l2": rel_l2, "meas_mse": meas_mse}
    return x_recon, metrics


# ---------------------------------------------------------------------------
# Guidance weight scheduler
# ---------------------------------------------------------------------------

_VALID_SCHEDULES = ("constant", "linear", "cosine", "warmup_decay")


def _scheduled_w(w: float, ti: float, schedule: str) -> float:
    """
    Return the scheduled guidance weight at timestep ti in [0, 1].

    Schedules (ti goes from 1.0 → 0.0 during reverse diffusion):
      constant      w(t) = w
      linear        w(t) = w * t          — decays to 0 as sample forms; removes late-step spikes
      cosine        w(t) = w * cos²(πt/2) — smooth cosine decay to 0
      warmup_decay  w(t) = w * sin(πt)    — zero at both ends, peaks at t ≈ 0.5
    """
    if schedule == "constant":
        return w
    elif schedule == "linear":
        return w * float(ti)
    elif schedule == "cosine":
        import math
        return w * math.cos(math.pi * float(ti) / 2) ** 2
    elif schedule == "warmup_decay":
        import math
        return w * math.sin(math.pi * float(ti))
    else:
        raise ValueError(f"Unknown guidance_schedule '{schedule}'. Choose from {_VALID_SCHEDULES}")


# ---------------------------------------------------------------------------
# DPS sensor-guided sampler
# ---------------------------------------------------------------------------


def run_dps_eval(y_obs, batch_size, score_model, pos, mesh_pos, forward_op, gt_field,
                 n_sensors: int, n_steps: int, guidance_weight: float, device,
                 guidance_schedule: str = "linear", tri=None, pos_batch=None,
                 cov_dps: bool = True):
    """
    Run a single DPS reconstruction for one ground-truth field.

    Args:
        score_model:      ScoreModel wrapper
        pos:              [N, 2] mesh positions (tensor, on device)
        mesh_pos:         [N, 2] numpy array (same positions, for sensor selection)
        gt_field:         [1, N] ground-truth tensor (on device)
        n_sensors:        number of sparse sensors
        n_steps:          reverse-diffusion steps
        guidance_weight:  DPS gradient step scale
        device:           torch device

    Returns:
        x_recon:  [1, N] reconstruction (on CPU)
        metrics:  dict(mse, rel_l2, meas_mse)
        sensor_coords: [K, 2] numpy array
    """
    sde = score_model.sde
    noise_sampler = score_model.noise_sampler

    ts = torch.linspace(1e-3, 1.0, n_steps, device=device)
    delta_t = ts[1] - ts[0]

    if pos_batch is None:
        pos_batch = pos.unsqueeze(0)                         # [1, N, D]
    xt = noise_sampler.sample(batch_size).unsqueeze(1)               # [1, 1, N]

    for ti in tqdm(reversed(ts), total=n_steps, desc="DPS eval", leave=False):
        t = torch.ones(1, device=device) * ti
        xt = xt.detach().requires_grad_(True)

        score, x0_pred = score_model(xt, t, pos_batch)

        y_pred = forward_op.forward(x0_pred)
        meas_loss = torch.sum((y_pred - y_obs) ** 2)
        grad = torch.autograd.grad(meas_loss, xt)[0]

        with torch.no_grad():
            # Principled VP-DPS: ∇_{x_t} log p(y|x_t) ≈ σ_t² · C · ∇_{x̂_0} loss
            # where σ_t² = 1 − exp(−α_t) is the marginal variance of the VP process.
            grad = noise_sampler.apply_C(grad.squeeze(1)).unsqueeze(1)
            if cov_dps:
                sigma_t_sq = sde.std_t_scaling(t, t).view(-1, 1, 1) ** 2
                grad = grad * sigma_t_sq
            # Normalize by measurement residual
            residual_norm = meas_loss.sqrt().clamp(min=1e-8).detach()
            grad = grad / residual_norm

            beta_t = sde.beta_t(t).view(-1, 1, 1)
            noise = noise_sampler.sample(batch_size).unsqueeze(1)
            phi_t = torch.exp(0.5 * beta_t * delta_t)
            xt_uncond = phi_t * xt + 2 * (phi_t - 1) * score + (phi_t**2 - 1).sqrt() * noise

            w_t = _scheduled_w(guidance_weight, ti.item(), guidance_schedule)
            xt = xt_uncond - w_t * grad

        if tri is not None:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            im1 = ax1.tripcolor(tri, x0_pred.squeeze().cpu().numpy(), cmap='Blues', shading='flat')
            ax1.set_title(f"t={ti:.3f}  w={w_t:.3f}")
            ax1.set_aspect('equal'); ax1.axis('off')
            fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            im2 = ax2.tripcolor(tri, grad.squeeze().cpu().numpy(), cmap='Reds', shading='flat')
            ax2.set_title("Guidance gradient")
            ax2.set_aspect('equal'); ax2.axis('off')
            fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            plt.tight_layout()
            plt.show()
            

    x_recon = xt.detach().cpu()

    with torch.no_grad():
        mse = torch.mean((x_recon - gt_field.cpu()) ** 2).item()
        rel_l2 = (torch.norm(x_recon - gt_field.cpu()) / torch.norm(gt_field.cpu())).item()
        y_recon = forward_op.forward(x_recon.to(device))
        meas_mse = torch.mean((y_recon - y_obs) ** 2).item()

    metrics = {"mse": mse, "rel_l2": rel_l2, "meas_mse": meas_mse}
    return x_recon, metrics


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_uncond_samples(samples, tri, n_cols=4, shading='flat', xy=None):
    """
    Plot unconditional samples on the mesh.

    Args:
        samples: [B, 1, N] tensor (CPU)
        tri:     matplotlib.tri.Triangulation (None for scatter-based domains)
        n_cols:  columns in the grid
        shading: 'flat' (cell-based) or 'gouraud' (vertex-based)
        xy:      [N, 2] numpy coords (required when tri is None)

    Returns:
        fig
    """
    import math
    B = samples.shape[0]
    n_rows = math.ceil(B / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3.5 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)
    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols, idx % n_cols]
        if idx < B:
            s = samples[idx, 0].numpy()
            if tri is not None:
                im = ax.tripcolor(tri, s, cmap='jet', shading=shading)
            else:
                im = ax.scatter(xy[:, 0], xy[:, 1], c=s, cmap='jet', s=2, alpha=0.9)
            ax.set_aspect('equal')
            ax.axis('off')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        else:
            ax.axis('off')
    fig.suptitle("Unconditional samples", fontsize=10)
    plt.tight_layout()
    return fig


def plot_cond_eval(x_gt, x_recon_all, forward_op, tri, metrics, mesh_pos,
                   shading='flat', xy=None, n_sample_cols=3):
    """Posterior reconstruction figure in the paper's Fig 6/7 layout — a single row:

        Ground Truth (+ sensors) | Posterior Mean | Posterior Std | Sample 1 .. Sample k

    Field panels (GT / mean / samples) use 'jet' on a shared colour range; the std
    map uses 'magma'; sensors are white dots on the ground truth.

    Args:
        x_gt:        [N] ground-truth field.
        x_recon_all: [B, N] posterior samples (a single [N] array is also accepted).
                     Mean/std are taken over B; up to n_sample_cols samples are shown.
                     With B == 1 the (degenerate) std panel is omitted.
        forward_op:  measurement operator (sensor coords read from it for the overlay).
        tri:         matplotlib Triangulation (None → scatter via xy).
        metrics:     dict with 'rel_l2' (shown as a suptitle).
        mesh_pos:    [N, 2] mesh positions.
    """
    x_gt = np.asarray(x_gt).flatten()
    x_recon_all = np.atleast_2d(x_recon_all)          # [B, N]
    B = x_recon_all.shape[0]
    x_mean = x_recon_all.mean(axis=0)
    x_std = x_recon_all.std(axis=0)

    def _plot(ax, vals, cmap, vmin=None, vmax=None):
        if tri is not None:
            return ax.tripcolor(tri, vals, cmap=cmap, shading=shading, vmin=vmin, vmax=vmax)
        return ax.scatter(xy[:, 0], xy[:, 1], c=vals, cmap=cmap, s=1, vmin=vmin, vmax=vmax)

    # shared colour range across the field panels (GT, mean, samples)
    vmin = min(x_gt.min(), x_recon_all.min())
    vmax = max(x_gt.max(), x_recon_all.max())

    # GT, Mean, [Std only when >1 sample], then up to n_sample_cols samples
    n_samp = min(n_sample_cols, B)
    panels = ['gt', 'mean'] + (['std'] if B >= 2 else []) + [f'sample:{i}' for i in range(n_samp)]

    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.4), squeeze=False)
    for ax, panel in zip(axes[0], panels):
        if panel == 'gt':
            im = _plot(ax, x_gt, 'jet', vmin, vmax)
            if isinstance(forward_op, SparseSensorOperator):
                sc = mesh_pos[forward_op.sensor_indices.cpu().numpy()]
                ax.scatter(sc[:, 0], sc[:, 1], c='white', s=16,
                           edgecolors='black', linewidths=0.4, zorder=5)
            ax.set_title("Ground Truth")
        elif panel == 'mean':
            im = _plot(ax, x_mean, 'jet', vmin, vmax)
            ax.set_title("Posterior Mean")
        elif panel == 'std':
            im = _plot(ax, x_std, 'magma')
            ax.set_title("Posterior Std")
        else:                                          # 'sample:i'
            i = int(panel.split(':')[1])
            im = _plot(ax, x_recon_all[i], 'jet', vmin, vmax)
            ax.set_title(f"Sample {i + 1}")
        ax.set_aspect('equal'); ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if isinstance(metrics, dict) and metrics.get('rel_l2') is not None:
        fig.suptitle(f"rel-L2 = {metrics['rel_l2']:.4f}", y=1.02)
    plt.tight_layout()
    return fig


def compute_posterior_mean_rmse(x_recon: torch.Tensor, gt_field: torch.Tensor) -> float:
    """
    RMSE between the posterior mean (average over samples) and the ground truth.

    Parameters
    ----------
    x_recon  : [B, 1, N]  - posterior samples
    gt_field : [1, N]     - ground truth field

    Returns
    -------
    float : RMSE value
    """
    posterior_mean = x_recon.mean(dim=0)          # [1, N]
    rmse = torch.sqrt(((posterior_mean - gt_field) ** 2).mean())
    return rmse.item()


def compute_energy_score(x_recon: torch.Tensor, gt_field: torch.Tensor) -> float:
    """
    Energy Score:  ES = E||X_gt - X|| - 0.5 * E||X - X'||
    where X, X' are independent draws from the posterior.

    Parameters
    ----------
    x_recon  : [B, 1, N]  - posterior samples
    gt_field : [1, N]     - ground truth field

    Returns
    -------
    float : Energy Score (lower is better)
    """
    B = x_recon.shape[0]
    x = x_recon.squeeze(1)      # [B, N]
    y = gt_field.squeeze(0)     # [N]

    # E||X - y||
    term1 = torch.norm(x - y.unsqueeze(0), dim=-1).mean()

    # E||X - X'||  (all pairs, i != j)
    diff = x.unsqueeze(0) - x.unsqueeze(1)          # [B, B, N]
    pair_norms = torch.norm(diff, dim=-1)            # [B, B]
    # exclude diagonal
    mask = ~torch.eye(B, dtype=torch.bool, device=x.device)
    term2 = pair_norms[mask].mean()

    return (term1 - 0.5 * term2).item()


def compute_mean_data_consistency(y_pred_batch: torch.Tensor, y_obs_batch: torch.Tensor) -> float:
    """
    Mean data consistency: average MSE between predicted and observed measurements
    across posterior samples.

    Parameters
    ----------
    y_pred_batch : [B, 1, n_sensors]  - forward-operator applied to x_recon
    y_obs_batch  : [B, 1, n_sensors]  - (repeated) observations

    Returns
    -------
    float : mean data-consistency MSE
    """
    mse_per_sample = ((y_pred_batch - y_obs_batch) ** 2).mean(dim=-1).mean(dim=-1)  # [B]
    return mse_per_sample.mean().item()

