
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal time embedding inspired by Transformer positional encoding.
    Maps scalar time t to a higher-dimensional representation.
    """
    def __init__(self, embed_dim: int, max_period: float = 10000.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        
        # Learnable projection after sinusoidal encoding
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time tensor of shape (batch_size,) or (batch_size, 1)
            
        Returns:
            Time embedding of shape (batch_size, embed_dim)
        """
        if t.dim() == 2:
            t = t.squeeze(-1)
            
        half_dim = self.embed_dim // 2
        freqs = torch.exp(
            -np.log(self.max_period) * torch.arange(half_dim, device=t.device, dtype=t.dtype) / half_dim
        )
        
        # (batch_size, half_dim)
        args = t[:, None] * freqs[None, :]
        
        # (batch_size, embed_dim)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return self.proj(embedding)


class FourierTimeEmbedding(nn.Module):
    """
    Random Fourier features for time embedding.
    Generally provides better frequency coverage than fixed sinusoidal.
    """
    def __init__(self, embed_dim: int, scale: float = 16.0):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Random frequencies (fixed after initialization)
        self.register_buffer('freqs', torch.randn(embed_dim // 2) * scale)
        
        # Learnable projection
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 2:
            t = t.squeeze(-1)
            
        # (batch_size, embed_dim // 2)
        args = 2 * np.pi * t[:, None] * self.freqs[None, :]
        
        # (batch_size, embed_dim)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return self.proj(embedding)

if __name__ == "__main__":
    # Simple test of time embeddings
    t = torch.linspace(0, 1, steps=100)
    print(t.shape)
    sinusoidal_embed = SinusoidalTimeEmbedding(embed_dim=16)
    fourier_embed = FourierTimeEmbedding(embed_dim=16)
    
    print("Sinusoidal embedding shape:", sinusoidal_embed(t).shape)
    print("Fourier embedding shape:", fourier_embed(t).shape)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2,2, figsize=(12, 6))

    ax1 = axes[0,0]
    ax1.imshow(sinusoidal_embed(t).detach().cpu().numpy().T, aspect='auto')
    ax1.set_title("Sinusoidal Time Embedding")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Embedding Dimension")

    ax2 = axes[0,1]
    ax2.imshow(fourier_embed(t).detach().cpu().numpy().T, aspect='auto')
    ax2.set_title("Fourier Time Embedding")     
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Embedding Dimension")

    ax3 = axes[1,0]
    for emb_idx in [0, 4, 12]:
        ax3.plot(t, sinusoidal_embed(t).detach().cpu().numpy()[:, emb_idx], label=f"emb_idx={emb_idx}")

    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Embedding Value")
    ax3.legend()

    ax4 = axes[1,1]
    for emb_idx in [0, 4, 12]:
        ax4.plot(t, fourier_embed(t).detach().cpu().numpy()[:, emb_idx], label=f"emb_idx={emb_idx}")

    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Embedding Value")
    ax4.legend()

    plt.tight_layout()
    plt.savefig("time_embeddings.png", dpi=150)
    plt.close()