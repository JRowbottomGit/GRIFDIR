"""
Domain-specific encoder/decoder heads for multi-domain training.

Maps between variable-size coarse graph features and a fixed-M latent space
via cross-attention, conditioned on a one-hot domain vector.

Architecture
------------
  DomainEncoderHead:  [B, N_coarse, H]  →  [B, M, H]
      M learned query tokens cross-attend to coarse GNN node features.
      Domain one-hot is projected and added to queries.

  DomainDecoderHead:  [B, M, H]  →  [B, N_coarse, H]
      Coarse GNN skip features (queries) cross-attend to latent (keys/values).
      Domain one-hot is projected and added to queries.

Both heads use zero-initialized output projections for stable training
start (identity at init), matching the DiTBlock/FiLM pattern used elsewhere.
"""

import torch
import torch.nn as nn


class DomainEncoderHead(nn.Module):
    """Cross-attention from M learned queries into variable-size GNN features.

    Parameters
    ----------
    hidden_dim : int
        Feature dimension (must match GNN hidden_dim).
    n_latent : int
        Number of latent tokens M (default 100).
    n_domains : int
        Dimension of the one-hot domain vector (default 8).
    n_heads : int
        Number of attention heads.
    """

    def __init__(self, hidden_dim: int, n_latent: int = 100,
                 n_domains: int = 8, n_heads: int = 4):
        super().__init__()
        self.n_latent = n_latent

        # Learnable latent queries
        self.queries = nn.Parameter(torch.randn(n_latent, hidden_dim) * 0.02)

        # One-hot domain → additive query bias
        self.domain_proj = nn.Sequential(
            nn.Linear(n_domains, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Cross-attention: Q = queries, K/V = coarse GNN features
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True,
        )
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Feed-forward after attention
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)

        # Zero-init output projections → identity at start
        nn.init.constant_(self.domain_proj[-1].weight, 0)
        nn.init.constant_(self.domain_proj[-1].bias, 0)
        nn.init.constant_(self.ff[-1].weight, 0)
        nn.init.constant_(self.ff[-1].bias, 0)

    def forward(self, h, domain_onehot):
        """
        Parameters
        ----------
        h : [B, N_coarse, H]  coarse-level GNN features.
        domain_onehot : [B, n_domains]  one-hot domain vector.

        Returns
        -------
        [B, M, H]  fixed-size latent representation.
        """
        B = h.shape[0]

        # Build queries: learned tokens + domain bias
        q = self.queries.unsqueeze(0).expand(B, -1, -1)       # [B, M, H]
        d = self.domain_proj(domain_onehot).unsqueeze(1)       # [B, 1, H]
        q = self.norm_q(q + d)

        # Cross-attention
        out, _ = self.cross_attn(q, h, h)                     # [B, M, H]
        out = self.norm_out(q + out)                           # residual + norm

        # Feed-forward
        out = out + self.ff(out)
        out = self.norm_ff(out)

        return out


class DomainDecoderHead(nn.Module):
    """Cross-attention from coarse GNN skip features into fixed-size latent.

    Symmetric to DomainEncoderHead: the queries are the coarse-level
    features (from the encoder skip), and keys/values are the latent.

    Parameters
    ----------
    hidden_dim : int
        Feature dimension (must match GNN hidden_dim).
    n_domains : int
        Dimension of the one-hot domain vector (default 8).
    n_heads : int
        Number of attention heads.
    """

    def __init__(self, hidden_dim: int, n_domains: int = 8, n_heads: int = 4):
        super().__init__()

        # One-hot domain → additive query bias
        self.domain_proj = nn.Sequential(
            nn.Linear(n_domains, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Cross-attention: Q = coarse skip, K/V = latent
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, batch_first=True,
        )
        self.norm_q = nn.LayerNorm(hidden_dim)
        self.norm_out = nn.LayerNorm(hidden_dim)

        # Feed-forward after attention
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm_ff = nn.LayerNorm(hidden_dim)

        # Zero-init output projections → identity at start
        nn.init.constant_(self.domain_proj[-1].weight, 0)
        nn.init.constant_(self.domain_proj[-1].bias, 0)
        nn.init.constant_(self.ff[-1].weight, 0)
        nn.init.constant_(self.ff[-1].bias, 0)

    def forward(self, latent, h_coarse_skip, domain_onehot):
        """
        Parameters
        ----------
        latent : [B, M, H]  transformer output.
        h_coarse_skip : [B, N_coarse, H]  coarse-level skip from encoder.
        domain_onehot : [B, n_domains]  one-hot domain vector.

        Returns
        -------
        [B, N_coarse, H]  features ready for up-cycle unpooling.
        """
        # Queries: coarse skip + domain bias
        d = self.domain_proj(domain_onehot).unsqueeze(1)       # [B, 1, H]
        q = self.norm_q(h_coarse_skip + d)                     # [B, N_coarse, H]

        # Cross-attention into latent
        out, _ = self.cross_attn(q, latent, latent)            # [B, N_coarse, H]
        out = self.norm_out(q + out)                           # residual + norm

        # Feed-forward
        out = out + self.ff(out)
        out = self.norm_ff(out)

        return out
