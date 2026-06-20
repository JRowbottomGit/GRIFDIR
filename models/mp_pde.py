import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, InstanceNorm

from diffusion.embedding import SinusoidalTimeEmbedding, FourierTimeEmbedding


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class MPPDELayer(MessagePassing):
    def __init__(self, hidden_dim, input_channels, pos_dim):
        super().__init__(node_dim=0, aggr="mean")

        self.message_net = nn.Sequential(
            nn.Linear(2 * hidden_dim + input_channels + pos_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
        )
        self.update_net = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
        )
        self.norm = InstanceNorm(hidden_dim)

    def forward(self, h, u, pos, edge_index, batch):
        h = self.propagate(edge_index, h=h, u=u, pos=pos)
        h = self.norm(h, batch)
        return h

    def message(self, h_i, h_j, u_i, u_j, pos_i, pos_j):
        return self.message_net(torch.cat([h_i, h_j, u_i - u_j, pos_i - pos_j], dim=-1))

    def update(self, agg, h):
        return h + self.update_net(torch.cat([h, agg], dim=-1))


class MPPDEScoreNetwork(nn.Module):
    def __init__(self, input_dim=1, pos_dim=2, hidden_dim=128, time_dim=64,
                 num_layers=6, time_embedding='fourier', time_embedding_scale=16.0,
                 conv_type=None, heads=None, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim

        if time_embedding == 'sinusoidal':
            self.time_embed = SinusoidalTimeEmbedding(time_dim)
        else:
            self.time_embed = FourierTimeEmbedding(time_dim, scale=time_embedding_scale)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim + pos_dim + time_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, hidden_dim), Swish(),
        )

        self.gnn_layers = nn.ModuleList([
            MPPDELayer(hidden_dim, input_dim, pos_dim) for _ in range(num_layers)
        ])

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), Swish(),
            nn.Linear(hidden_dim, input_dim),
        )

        self.register_buffer('_cached_edge_index', None, persistent=False)
        self._cached_num_points = None

    def set_mesh(self, edge_index, num_points=None):
        self._cached_edge_index = edge_index
        self._cached_num_points = num_points if num_points is not None else edge_index.max().item() + 1

    def forward(self, inp, t, pos):
        if self._cached_edge_index is None:
            raise RuntimeError("Call set_mesh() before forward.")

        B, _, N = inp.shape
        device = inp.device
        base_ei = self._cached_edge_index.to(device)

        positions = pos.squeeze(1)
        signal = inp.permute(0, 2, 1)[..., self.pos_dim:]

        t_emb = self.time_embed(t)

        offsets = torch.arange(B, device=device) * N
        ei = base_ei.unsqueeze(0).expand(B, -1, -1) + offsets.view(B, 1, 1)
        ei = ei.permute(1, 0, 2).reshape(2, -1)

        pos_flat = positions.reshape(B * N, self.pos_dim)
        u_flat = signal.reshape(B * N, self.input_dim)
        batch_idx = torch.arange(B, device=device).repeat_interleave(N)
        t_flat = t_emb[batch_idx]

        h = self.encoder(torch.cat([u_flat, pos_flat, t_flat], dim=-1))

        for layer in self.gnn_layers:
            h = layer(h, u_flat, pos_flat, ei, batch_idx)

        out = self.decoder(h)
        return out.view(B, N, self.input_dim).permute(0, 2, 1)
