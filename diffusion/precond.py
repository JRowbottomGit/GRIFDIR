"""
Wrapper for the time-dependent neural operator.

"""


import torch
import torch.nn as nn 
from omegaconf import DictConfig

from .sde import OU, CosineOU

class ScoreModel():
    def __init__(self, model: nn.Module, sde: OU, noise_sampler, cfg: DictConfig):
        self.model = model
        self.sde = sde
        self.noise_sampler = noise_sampler
        self.cfg = cfg

    def __call__(self, xt, t, pos, domain_onehot=None):
        # Returns both the score function and the predicted x0.

        inp = torch.cat([pos.permute(0, 2, 1), xt], dim=1)
        std_t = self.sde.std_t_scaling(t, xt)

        if self.cfg.model.precond_last_layer:
            pred = self.model(inp, t, pos.unsqueeze(1), domain_onehot=domain_onehot) / std_t
        
        # Get model_type as uppercase string (handles both enum and string values)
        model_type = str(self.cfg.model.model_type)
        if hasattr(self.cfg.model.model_type, 'value'):
            model_type = self.cfg.model.model_type.value
        model_type = model_type.upper()
        
        if model_type == "RAW":  # the network output is nabla log p
            score = self.noise_sampler.apply_C(pred.squeeze()).unsqueeze(1)
        elif model_type == "C_SQRT": # the network output is C^{1/2} nabla log p 
            score = self.noise_sampler.apply_Csqrt(pred.squeeze()).unsqueeze(1)
        elif model_type == "C": # the network output is C nabla log p
            score = pred
        else:
            raise NotImplementedError(f"Unknown model_type: {self.cfg.model.model_type}")

        mean_t_scaling = self.sde.mean_t_scaling(t, xt)
        x0_pred = (xt + score * std_t**2) / mean_t_scaling

        return score, x0_pred


class EDMDenoiser:
    """
    VE/EDM-style wrapper (Karras et al. 2022 "Elucidating the Design Space").

    Forward process: x_noisy = x0 + σ · z   (no mean shrinkage)

    EDM preconditioning:
        c_in   = 1 / √(σ_data² + σ²)
        c_skip = σ_data² / (σ² + σ_data²)
        c_out  = σ · σ_data / √(σ² + σ_data²)

    __call__(x_noisy, sigma, pos) → x0_pred

    sigma shape: [B] or [B, 1, 1]
    """
    def __init__(self, model: nn.Module, noise_sampler, cfg: DictConfig):
        self.model = model
        self.noise_sampler = noise_sampler
        self.cfg = cfg
        self.sigma_data = cfg.sde.sigma_data

    def __call__(self, x_noisy, sigma, pos, domain_onehot=None):
        if sigma.dim() == 1:
            sigma = sigma.view(-1, 1, 1)

        sd = self.sigma_data
        c_skip = sd ** 2 / (sigma ** 2 + sd ** 2)
        c_out  = sigma * sd / (sigma ** 2 + sd ** 2).sqrt()
        c_in   = 1.0 / (sd ** 2 + sigma ** 2).sqrt()

        inp = torch.cat([pos.permute(0, 2, 1), c_in * x_noisy], dim=1)
        F_x = self.model(inp, sigma.squeeze(-1).squeeze(-1), pos,
                         domain_onehot=domain_onehot)
        x0_pred = c_skip * x_noisy + c_out * F_x
        return x0_pred