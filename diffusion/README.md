# diffusion/

Function-space score-based diffusion. The forward process corrupts a field with
**Gaussian-process noise sampled in function space** (resolution-invariant) instead of
i.i.d. pixel noise; a score network from `models/` is wrapped with a preconditioner and
trained to denoise. The reverse-time Heun sampler lives in `eval/sampling.py`.

| File | Purpose |
|------|---------|
| `precond.py` | Score-model wrappers around a raw `models/` network: `EDMDenoiser` (VE / EDM preconditioning — the paper default) and `ScoreModel` (VP). |
| `noise.py` | Function-space noise samplers — `RBFKernel` (RBF Gaussian-process noise), `RFFNoiseSampler`, and the `NoiseSampler` base. This is what makes the noise resolution-invariant. |
| `noise_accel.py` | Accelerated drop-in `RBFKernel` (training default; same interface as `noise.py`). |
| `sde.py` | VP forward processes `OU` / `CosineOU`. (VE is preconditioning-only and needs no SDE object.) |
| `embedding.py` | Diffusion-timestep embeddings (`SinusoidalTimeEmbedding`, `FourierTimeEmbedding`) used by every score network in `models/`. |

`sde_type` in `config.py` selects the process: `ve` (paper default), `vp`, or `vp_cosine`.
