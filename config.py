"""
Configuration classes for GNN-based diffusion models.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from omegaconf import OmegaConf


class ModelType(str, Enum):
    RAW = "RAW"
    C_SQRT = "C_SQRT"
    C = "C"


@dataclass
class ModelConfig:
    model_type: ModelType = ModelType.C
    precond_last_layer: bool = True  # divide network output by std_t
    hidden_dim: int = 256
    time_dim: int = 128
    num_layers: int = 16
    conv_type: str = "multiscale"  # "multiscale" (paper); "gcn"/"gat"/"mp_pde" are legacy
    heads: int = 4  # for GAT
    dropout: float = 0.0
    time_embedding: str = "fourier"  # "fourier" or "sinusoidal"
    time_embedding_scale: float = 16.0  # scale for Fourier time embedding
    use_latent_transformer: bool = False
    n_transformer_blocks: int = 4
    n_transformer_heads: int = 4
    pooling_type: str = "false"
    use_pos_reinject: bool = False
    use_edge_geom: bool = False
    layer_type: str = "simple_mp"
    mixing_type: str = "vector"      # scalar | vector | lowrank
    fem_basis_type: str = "p1"       # p0 | p1  (fem_conv only)
    fem_k_hops: int = 2              # 1=1-hop (3 nbrs), 2=2-hop (~9 nbrs), 3=3-hop
    fem_use_radius: bool = False     # fem_conv: use physical radius instead of k-hop for resolution invariance
    fem_radius_mult: float = 3.0     # fem_conv radius mode: radius = mult * median_edge_length (from training mesh)
    fem_lumped_mass: bool = False    # fem_conv: aggr='add' + multiply by source-node dual-cell area (lumped-mass FE quadrature, density-invariant)
    knn_radius_mult: float = 3.0     # knn_conv: radius = mult * median_edge_length
    fno_n_modes: int = 16            # FNO: Fourier modes per spatial dim
    fno_n_layers: int = 4           # FNO: number of Fourier blocks
    fno_pos_enc: str = "none"         # FNO: "none" | "grid" — 'grid' adds (x,y) channels
    fno_pad: float = 0.0              # FNO: domain padding, 0.0 = pure periodic FFT, 0.125 = std non-periodic
    cnn_double_conv: bool = False    # CNN: 2 convs per block (adds depth)
    cnn_residual: bool = False       # CNN: residual skip connections
    cnn_bottleneck_attn: bool = False # CNN: self-attn at bottleneck
    cnn_strided_down: bool = False   # CNN: strided conv instead of MaxPool
    cnn_dropout: float = 0.0         # CNN: dropout rate
    # GAOT: Geometry-Aware Operator Transformer baseline
    gaot_latent_tokens_size: Any = field(default_factory=lambda: [32, 32])  # latent regular grid [H, W]
    gaot_patch_size: int = 2          # ViT patch size (must divide H, W)
    gaot_n_transformer_layers: int = 3
    gaot_magno_radius: float = 0.05   # MAGNO neighbor-search radius (coords in [0,1])
    gaot_magno_hidden: int = 64       # MAGNO internal MLP hidden size
    gaot_magno_mlp_layers: int = 3
    gaot_magno_lifting: int = 64      # MAGNO lifting channels (per-node latent width)
    gaot_positional_embedding: str = 'absolute'  # 'absolute' | 'rope'
    gaot_use_geoembed: bool = True    # GAOT statistical geometric embedding
    gaot_use_attention: bool = True   # MAGNO cosine attention (paper default); works on XPU via scatter_reduce monkey-patch
    gaot_use_torch_scatter: bool = False  # use torch_scatter if installed; otherwise wrapper's scatter_reduce fallback
    film_time_cond: bool = False     # FiLM scale+shift timestep conditioning at every V-cycle level
    res_invariant: bool = False      # shared GNN weights + ResolutionFiLM conditioning on log(cell area)
    use_domain_heads: bool = False   # cross-attention domain encoder/decoder at bottleneck (multi-domain)
    n_latent: int = 100              # M — fixed number of latent tokens when use_domain_heads=True
    n_domains: int = 8               # dimension of domain one-hot vector


@dataclass
class SDEConfig:
    # --- forward process type ---
    sde_type: str = "ve"            # "ve" = VE/EDM (paper default); "vp" = VP-OU (linear β), "vp_cosine" = VP-cosine
    # --- VP parameters (sde_type = vp / vp_cosine) ---
    beta_min: float = 0.001
    beta_max: float = 15.0
    # --- VE/EDM parameters (sde_type = ve) ---
    sigma_min: float = 0.001        # lower end of σ schedule
    sigma_max: float = 40.0         # upper end of σ schedule (also sets log-normal range)
    P_mean: float = -1.2            # log-normal σ sampling mean  (Karras et al. 2022 EDM)
    P_std: float = 1.2              # log-normal σ sampling std
    sigma_data: float = 0.5         # assumed data std (EDM preconditioning)


@dataclass
class NoiseConfig:
    scale: float = 0.3       # RBF kernel length scale
    eps: float = 0.01        # regularization for numerical stability
    sampler_type: str = 'rbf'   # 'rbf' | 'rff' — 'rbf' is default for backwards compatibility
    n_rff_features: int = 512   # number of Random Fourier Features (only used when sampler_type='rff')


@dataclass
class DataConfig:
    nx: int = 64  # mesh resolution in x
    ny: int = 64  # mesh resolution in y
    num_samples: int = 10000
    max_numInc: int = 3
    backCond: float = 1.0
    data_dir: str = 'data'
    domain: str = 'square'          # 'square' | 'l_shape' | 'pinball' | 'multidomain'
    domains: str = ''               # comma-separated domain names for multi-domain (e.g. "circle,x_shape,l_shape")
    max_samples_per_domain: int = 0 # cap each domain to N samples for balanced training (0=no cap)
    ms_base_dir: str = ''           # pinball: parent of coarse resolution dirs
    ms_finest_dir: str = ''        # pinball: finest-level data dir (resolved from train_window or ms_base_dir)
    ms_res_dirs: str = ''           # pinball: comma-separated coarse dir names (used levels)
    ms_all_res_dirs: str = ''      # pinball: comma-separated ALL coarse dir names (for operator composition)
    n_cond_channels: int = 0        # set at runtime: extra conditioning dims (mu + time)
    enc_use_mu: bool = True          # pinball: include mu parameter vector in position encoding
    enc_use_time: bool = True        # pinball: include physical time t in position encoding
    land_mask_path: str = ''         # SST: path to land mask tensor
    knn_k: int = 0                   # SST: k for kNN graph construction (0=use mesh edges)
    hierarchy_path: str = ''         # explicit path to mesh_hierarchy .pt file (overrides domain lookup)


@dataclass
class TrainingConfig:
    num_epochs: int = 2000
    lr: float = 1e-3
    batch_size: int = 64
    device: str = ""   # "" / "auto" = autodetect (cuda→mps→xpu→cpu); or "cuda" / "mps" / "cpu"
    save_dir: str = "exp"
    log_wandb: bool = True
    use_wandb: bool = False
    wandb_entity: str = ""  # empty = your default wandb entity
    wandb_project: str = "grifdir"
    num_workers: int = 4
    pin_memory: bool = False
    persistent_workers: bool = False
    use_amp: bool = False
    use_accel_sampler: bool = True
    data_config: str = "baseline"
    vis_every: int = 50
    ckpt_every: int = 50
    gen_eval_every: int = 100  # run unconditional+DPS eval every N epochs (0=disabled)
    area_weighted_loss: bool = False  # weight loss by dual-cell area (for non-uniform meshes)
    multires_eval_specs: Any = field(default_factory=list)  # list of pinball hierarchy dicts for post-training eval


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    sde: SDEConfig = field(default_factory=SDEConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


def merge_config(overrides):
    """Layer a loaded (or CLI-built) OmegaConf node over the Config() defaults.

    The defaults are flattened to an untyped container first: a structured
    (dataclass-typed) node rejects any key absent from the schema, so a
    config.yaml saved by an older checkpoint — carrying keys since removed —
    would raise. Untyped, those extra keys merge through harmlessly.
    """
    defaults = OmegaConf.to_container(OmegaConf.structured(Config()), enum_to_str=True)
    return OmegaConf.merge(OmegaConf.create(defaults), overrides)


def get_device(preferred=None):
    """Resolve a torch device string.

    Honour an explicit `preferred` choice (from --device / config) when that
    backend is actually available; otherwise autodetect cuda → mps → xpu and
    fall back to cpu — so the same code runs on CUDA, Apple-silicon (mps) and
    CPU without per-call-site branching.
    """
    import torch

    def _available(dev):
        base = dev.split(":")[0]
        if base == "cuda":
            return torch.cuda.is_available()
        if base == "mps":
            return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if base == "xpu":
            return hasattr(torch, "xpu") and torch.xpu.is_available()
        return base == "cpu"

    if preferred and preferred != "auto":
        if _available(preferred):
            return preferred
        print(f"[device] requested '{preferred}' unavailable — autodetecting")
    for dev in ("cuda", "mps", "xpu"):
        if _available(dev):
            return dev
    return "cpu"
