"""
GNN-based Score Model using PyTorch Geometric.

This model takes noisy inputs at mesh positions along with time (noise level)
and predicts the score function for diffusion models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, GATConv, MessagePassing
from torch_geometric.data import Data, Batch
from diffusion.embedding import SinusoidalTimeEmbedding, FourierTimeEmbedding


class TimeConditionedLinear(nn.Module):
    """Linear layer with time-dependent scale and shift (FiLM conditioning)."""
    def __init__(self, in_features: int, out_features: int, time_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.scale = nn.Linear(time_dim, out_features)
        self.shift = nn.Linear(time_dim, out_features)
        
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, in_features)
            t_emb: Time embedding (num_nodes, time_dim) - expanded per node
        """
        h = self.linear(x)
        scale = self.scale(t_emb)
        shift = self.shift(t_emb)
        return h * (1 + scale) + shift


class GNNBlock(nn.Module):
    """
    A single GNN block with time conditioning.
    Uses message passing with time-dependent transformations.
    """
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        time_dim: int,
        conv_type: str = 'gcn',
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.conv_type = conv_type
        
        if conv_type == 'gcn':
            self.conv = GCNConv(in_channels, out_channels)
        elif conv_type == 'gat':
            self.conv = GATConv(in_channels, out_channels // heads, heads=heads, concat=True, dropout=dropout)
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")
            
        self.norm = nn.LayerNorm(out_channels)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels * 2),
        )
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection if dimensions match
        self.skip = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, in_channels)
            edge_index: Edge indices (2, num_edges)
            t_emb: Time embedding per node (num_nodes, time_dim)
        """
        h = self.conv(x, edge_index)
        h = self.norm(h)
        
        # Time conditioning via scale and shift
        time_out = self.time_mlp(t_emb)
        scale, shift = time_out.chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        
        h = F.silu(h)
        h = self.dropout(h)
        
        return h + self.skip(x)




class PDEMessagePassing(MessagePassing):
    r"""
    Implements:

        m_ij = \phi(f_i, f_j, x_i - x_j)
        f_i^{m+1} = \psi(f_i, mean_j m_ij)

    Aggregation: mean
    """

    def __init__(
        self,
        node_dim: int,
        hist_dim: int,
        pos_dim: int,
        hidden_dim: int,
        aggr: str = "mean",
    ):
        super().__init__(aggr=aggr)

        # phi : edge message MLP
        phi_in_dim = (
            2 * node_dim +   # f_i, f_j
            hist_dim +       # u_i - u_j
            pos_dim          # x_i - x_j
        )

        self.phi = nn.Sequential(
            nn.Linear(phi_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # psi : node update MLP
        psi_in_dim = node_dim + hidden_dim

        self.psi = nn.Sequential(
            nn.Linear(psi_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )

    def forward(
        self,
        f,               # (N, node_dim)
        edge_index,      # (2, E)
        pos              # (N, pos_dim)
    ):
        return self.propagate(
            edge_index,
            f=f,
            pos=pos,
        )

    def message(self, f_i, f_j, pos_i, pos_j):
        """
        Computes m_ij
        """

        x_diff = pos_i - pos_j

        edge_input = torch.cat(
            [f_i, f_j, x_diff],
            dim=-1
        )

        return self.phi(edge_input)

    def update(self, aggr_out, f):
        """
        Computes psi(f_i, mean_j m_ij)
        """

        node_input = torch.cat([f, aggr_out], dim=-1)
        return self.psi(node_input)



class PDEGNNBlock(nn.Module):
    """
    A GNN block using PDEMessagePassing with time conditioning.
    Uses position-aware message passing: m_ij = phi(f_i, f_j, x_i - x_j)
    """
    def __init__(
        self,
        node_dim: int,
        pos_dim: int,
        hidden_dim: int,
        time_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        self.conv = PDEMessagePassing(
            node_dim=node_dim,
            hist_dim=0,  # Not using history features
            pos_dim=pos_dim,
            hidden_dim=hidden_dim,
        )
        
        self.norm = nn.LayerNorm(node_dim)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, node_dim * 2),
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        edge_index: torch.Tensor, 
        pos: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Node features (num_nodes, node_dim)
            edge_index: Edge indices (2, num_edges)
            pos: Node positions (num_nodes, pos_dim)
            t_emb: Time embedding per node (num_nodes, time_dim)
        """
        # Message passing with position-aware messages
        h = self.conv(x, edge_index, pos)
        h = self.norm(h)
        
        # Time conditioning via scale and shift
        time_out = self.time_mlp(t_emb)
        scale, shift = time_out.chunk(2, dim=-1)
        h = h * (1 + scale) + shift
        
        h = F.silu(h)
        h = self.dropout(h)
        
        return h + x  # Residual connection


class PDEGNNScoreNetwork(nn.Module):
    """
    GNN score network using PDEMessagePassing layers.
    
    This network uses position-aware message passing that explicitly
    incorporates the Euclidean distance (x_i - x_j) between nodes
    in the message computation.
    
    Architecture:
    1. Input projection (noisy signal + position -> hidden dim)
    2. Time embedding  
    3. Stack of PDEGNNBlocks with time conditioning
    4. Output projection
    """
    def __init__(
        self,
        input_dim: int = 1,
        pos_dim: int = 2,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_layers: int = 6,
        dropout: float = 0.0,
        time_embedding: str = 'fourier',
        k_neighbors: int = 8,
        radius: float = None,
    ):
        """
        Args:
            input_dim: Dimension of input signal (typically 1 for scalar field)
            pos_dim: Dimension of positions (2 for 2D mesh)
            hidden_dim: Hidden dimension for GNN layers
            time_dim: Dimension of time embedding
            num_layers: Number of PDEGNNBlocks
            dropout: Dropout rate
            time_embedding: Type of time embedding ('sinusoidal' or 'fourier')
            k_neighbors: Number of nearest neighbors for graph construction
            radius: Radius for radius graph (if None, uses k-NN)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.radius = radius
        
        # Time embedding
        if time_embedding == 'sinusoidal':
            self.time_embed = SinusoidalTimeEmbedding(time_dim)
        else:
            self.time_embed = FourierTimeEmbedding(time_dim)
        
        # Input projection: signal + position -> hidden
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + pos_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # PDE-style GNN layers
        self.gnn_layers = nn.ModuleList([
            PDEGNNBlock(
                node_dim=hidden_dim,
                pos_dim=pos_dim,
                hidden_dim=hidden_dim,
                time_dim=time_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
        # Graph cache
        self.register_buffer('_cached_edge_index', None, persistent=False)
        self._cached_num_points = None
        
    def build_graph(self, pos: torch.Tensor) -> torch.Tensor:
        """Build graph from positions using k-nearest neighbors."""
        from torch_geometric.nn import knn_graph, radius_graph
        
        if self.radius is not None:
            edge_index = radius_graph(pos, r=self.radius, loop=False)
        else:
            edge_index = knn_graph(pos, k=self.k_neighbors, loop=False)
            
        # Make undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        return edge_index
    
    def set_mesh(self, pos: torch.Tensor):
        """
        Pre-compute and cache the graph structure for a fixed mesh.
        
        Args:
            pos: Position tensor (num_points, pos_dim)
        """
        edge_index = self.build_graph(pos)
        self._cached_edge_index = edge_index
        self._cached_num_points = pos.shape[0]
        
    def forward(
        self, 
        inp: torch.Tensor, 
        t: torch.Tensor, 
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with position-aware message passing.
        
        Args:
            inp: Input tensor (batch, channels, num_points) where channels = pos_dim + input_dim
            t: Time tensor (batch,)
            pos: Position tensor (batch, 1, num_points, pos_dim)
            
        Returns:
            Output tensor (batch, input_dim, num_points)
        """
        batch_size = inp.shape[0]
        num_points = inp.shape[2]
        device = inp.device
        
        # Extract positions: (batch, num_points, pos_dim)
        positions = pos.squeeze(1)
        
        # inp: (batch, channels, num_points) -> (batch, num_points, channels)
        inp = inp.permute(0, 2, 1)
        
        # Time embedding
        t_emb = self.time_embed(t)  # (batch, time_dim)
        
        # Use cached graph or build new one
        if self._cached_edge_index is not None and self._cached_num_points == num_points:
            base_edge_index = self._cached_edge_index.to(device)
        else:
            base_edge_index = self.build_graph(positions[0])
        
        # Create batched edge index by offsetting
        edge_indices = []
        for b in range(batch_size):
            offset = b * num_points
            edge_indices.append(base_edge_index + offset)
        batched_edge_index = torch.cat(edge_indices, dim=1)
        
        # Flatten batch dimension
        x = inp.reshape(batch_size * num_points, -1)  # (batch * num_points, channels)
        batched_pos = positions.reshape(batch_size * num_points, -1)  # (batch * num_points, pos_dim)
        
        # Expand time embedding
        batch_idx = torch.arange(batch_size, device=device).repeat_interleave(num_points)
        t_emb_expanded = t_emb[batch_idx]  # (batch * num_points, time_dim)
        
        # Forward through network
        h = self.input_proj(x)
        
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, batched_edge_index, batched_pos, t_emb_expanded)
        
        out = self.output_proj(h)
        
        # Reshape back to (batch, input_dim, num_points)
        out = out.view(batch_size, num_points, self.input_dim)
        out = out.permute(0, 2, 1)
        
        return out


class MeshPDEGNNScoreNetwork(nn.Module):
    """
    GNN score network using PDEMessagePassing with mesh connectivity.
    
    Unlike PDEGNNScoreNetwork which builds graphs via k-NN, this version
    uses pre-defined mesh connectivity (e.g., from triangular mesh adjacency).
    The edge_index must be set via set_mesh() before forward pass.
    
    Uses position-aware message passing: m_ij = phi(f_i, f_j, x_i - x_j)
    """
    def __init__(
        self,
        input_dim: int = 1,
        pos_dim: int = 2,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_layers: int = 6,
        dropout: float = 0.0,
        time_embedding: str = 'fourier',
    ):
        """
        Args:
            input_dim: Dimension of input signal (typically 1 for scalar field)
            pos_dim: Dimension of positions (2 for 2D mesh)
            hidden_dim: Hidden dimension for GNN layers
            time_dim: Dimension of time embedding
            num_layers: Number of PDEGNNBlocks
            dropout: Dropout rate
            time_embedding: Type of time embedding ('sinusoidal' or 'fourier')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim
        
        # Time embedding
        if time_embedding == 'sinusoidal':
            self.time_embed = SinusoidalTimeEmbedding(time_dim)
        else:
            self.time_embed = FourierTimeEmbedding(time_dim)
        
        # Input projection: signal + position -> hidden
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + pos_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # PDE-style GNN layers
        self.gnn_layers = nn.ModuleList([
            PDEGNNBlock(
                node_dim=hidden_dim,
                pos_dim=pos_dim,
                hidden_dim=hidden_dim,
                time_dim=time_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
        # Mesh connectivity cache
        self.register_buffer('_cached_edge_index', None, persistent=False)
        self._cached_num_points = None
        
    def set_mesh(self, edge_index: torch.Tensor, num_points: int = None):
        """
        Set the mesh connectivity (dual graph edge_index).
        
        Args:
            edge_index: (2, num_edges) tensor from mesh cell adjacency
            num_points: Number of mesh cells/nodes (inferred from edge_index if not provided)
        """
        self._cached_edge_index = edge_index
        if num_points is not None:
            self._cached_num_points = num_points
        else:
            self._cached_num_points = edge_index.max().item() + 1
        
    def forward(
        self, 
        inp: torch.Tensor, 
        t: torch.Tensor, 
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass using mesh connectivity with position-aware message passing.
        
        Args:
            inp: Input tensor (batch, channels, num_points) where channels = pos_dim + input_dim
            t: Time tensor (batch,)
            pos: Position tensor (batch, 1, num_points, pos_dim)
            
        Returns:
            Output tensor (batch, input_dim, num_points)
        """
        if self._cached_edge_index is None:
            raise RuntimeError("Mesh not set. Call set_mesh(edge_index) before forward pass.")
            
        batch_size = inp.shape[0]
        num_points = inp.shape[2]
        device = inp.device
        
        # Move cached edge_index to correct device
        base_edge_index = self._cached_edge_index.to(device)
        
        # Extract positions: (batch, num_points, pos_dim)
        positions = pos.squeeze(1)
        
        # inp: (batch, channels, num_points) -> (batch, num_points, channels)
        inp = inp.permute(0, 2, 1)
        
        # Time embedding
        t_emb = self.time_embed(t)  # (batch, time_dim)
        
        # Create batched edge index by offsetting for each sample in batch
        edge_indices = []
        for b in range(batch_size):
            offset = b * num_points
            edge_indices.append(base_edge_index + offset)
        batched_edge_index = torch.cat(edge_indices, dim=1)
        
        # Flatten batch dimension
        x = inp.reshape(batch_size * num_points, -1)  # (batch * num_points, channels)
        batched_pos = positions.reshape(batch_size * num_points, -1)  # (batch * num_points, pos_dim)
        
        # Expand time embedding to all nodes
        batch_idx = torch.arange(batch_size, device=device).repeat_interleave(num_points)
        t_emb_expanded = t_emb[batch_idx]  # (batch * num_points, time_dim)
        
        # Forward through network
        h = self.input_proj(x)
        
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, batched_edge_index, batched_pos, t_emb_expanded)
        
        out = self.output_proj(h)
        
        # Reshape back to (batch, input_dim, num_points)
        out = out.view(batch_size, num_points, self.input_dim)
        out = out.permute(0, 2, 1)
        
        return out


class GNNScoreNetwork(nn.Module):
    """
    GNN-based score network for diffusion models on meshes.
    
    Architecture:
    1. Input projection (noisy signal + position -> hidden dim)
    2. Time embedding
    3. Stack of GNN blocks with time conditioning
    4. Output projection
    """
    def __init__(
        self,
        input_dim: int = 1,
        pos_dim: int = 2,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_layers: int = 6,
        conv_type: str = 'gcn',
        heads: int = 4,
        dropout: float = 0.0,
        time_embedding: str = 'fourier',
        k_neighbors: int = 8,
        radius: float = None,
    ):
        """
        Args:
            input_dim: Dimension of input signal (typically 1 for scalar field)
            pos_dim: Dimension of positions (2 for 2D mesh)
            hidden_dim: Hidden dimension for GNN layers
            time_dim: Dimension of time embedding
            num_layers: Number of GNN blocks
            conv_type: Type of graph convolution ('gcn' or 'gat')
            heads: Number of attention heads (for GAT)
            dropout: Dropout rate
            time_embedding: Type of time embedding ('sinusoidal' or 'fourier')
            k_neighbors: Number of nearest neighbors for graph construction
            radius: Radius for radius graph (if None, uses k-NN)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        self.radius = radius
        
        # Time embedding
        if time_embedding == 'sinusoidal':
            self.time_embed = SinusoidalTimeEmbedding(time_dim)
        elif time_embedding == 'fourier':
            self.time_embed = FourierTimeEmbedding(time_dim)
        else:
            raise ValueError(f"Unknown time_embedding: {time_embedding}")
        
        # Input projection: signal + position -> hidden
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + pos_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNBlock(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                time_dim=time_dim,
                conv_type=conv_type,
                heads=heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
        # Cache for edge_index (recomputed if positions change)
        self._cached_edge_index = None
        self._cached_pos_hash = None
        
    def build_graph(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Build graph from positions using k-nearest neighbors.
        
        Args:
            pos: Position tensor (num_nodes, pos_dim)
            
        Returns:
            edge_index: (2, num_edges)
        """
        from torch_geometric.nn import knn_graph, radius_graph
        
        if self.radius is not None:
            edge_index = radius_graph(pos, r=self.radius, loop=False)
        else:
            edge_index = knn_graph(pos, k=self.k_neighbors, loop=False)
            
        # Make undirected
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        return edge_index
    
    def forward(
        self, 
        inp: torch.Tensor, 
        t: torch.Tensor, 
        pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass compatible with ScoreModel wrapper.
        
        Args:
            inp: Input tensor (batch, channels, num_points) where channels = pos_dim + input_dim
                 This is torch.cat([pos.permute(0, 2, 1), xt], dim=1)
            t: Time tensor (batch,)
            pos: Position tensor (batch, 1, num_points, pos_dim) - used for building graph
            
        Returns:
            Output tensor (batch, input_dim, num_points)
        """
        batch_size = inp.shape[0]
        num_points = inp.shape[2]
        device = inp.device
        
        # Extract positions from pos input (batch, 1, num_points, 2) -> (batch, num_points, 2)
        positions = pos.squeeze(1)  # (batch, num_points, 2)
        
        # inp is (batch, channels, num_points) -> (batch, num_points, channels)
        inp = inp.permute(0, 2, 1)
        
        # For batched graphs, we process each sample and combine
        # Time embedding: (batch,) -> (batch, time_dim)
        t_emb = self.time_embed(t)
        
        # Build batched graph data
        batch_list = []
        for b in range(batch_size):
            # Build graph for this sample
            edge_index = self.build_graph(positions[b])
            
            data = Data(
                x=inp[b],  # (num_points, channels)
                edge_index=edge_index,
                pos=positions[b],
            )
            batch_list.append(data)
        
        # Batch all graphs
        batched_data = Batch.from_data_list(batch_list)
        
        # Expand time embedding to all nodes
        # batched_data.batch gives node -> graph mapping
        t_emb_expanded = t_emb[batched_data.batch]  # (total_nodes, time_dim)
        
        # Input projection
        h = self.input_proj(batched_data.x)  # (total_nodes, hidden_dim)
        
        # GNN layers with time conditioning
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, batched_data.edge_index, t_emb_expanded)
        
        # Output projection
        out = self.output_proj(h)  # (total_nodes, input_dim)
        
        # Reshape back to (batch, input_dim, num_points)
        out = out.view(batch_size, num_points, self.input_dim)
        out = out.permute(0, 2, 1)  # (batch, input_dim, num_points)
        
        return out


class GNNScoreNetworkOptimized(nn.Module):
    """
    Optimized version that caches the graph structure when positions are fixed.
    More efficient for fixed meshes during training/inference.
    """
    def __init__(
        self,
        input_dim: int = 1,
        pos_dim: int = 2,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_layers: int = 6,
        conv_type: str = 'gcn',
        heads: int = 4,
        dropout: float = 0.0,
        time_embedding: str = 'fourier',
        k_neighbors: int = 8,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim
        self.k_neighbors = k_neighbors
        
        # Time embedding
        if time_embedding == 'sinusoidal':
            self.time_embed = SinusoidalTimeEmbedding(time_dim)
        else:
            self.time_embed = FourierTimeEmbedding(time_dim)
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + pos_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNBlock(hidden_dim, hidden_dim, time_dim, conv_type, heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), 
            nn.Linear(hidden_dim, input_dim),
        )
        
        # Graph cache
        self.register_buffer('_cached_edge_index', None, persistent=False)
        self._cached_num_points = None
        
    def set_mesh(self, pos: torch.Tensor):
        """
        Pre-compute and cache the graph structure for a fixed mesh.
        
        Args:
            pos: Position tensor (num_points, pos_dim)
        """
        from torch_geometric.nn import knn_graph
        
        edge_index = knn_graph(pos, k=self.k_neighbors, loop=False)
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        self._cached_edge_index = edge_index
        self._cached_num_points = pos.shape[0]
        
    def forward(self, inp: torch.Tensor, t: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with cached graph (must call set_mesh first for best performance).
        """
        batch_size = inp.shape[0]
        num_points = inp.shape[2]
        device = inp.device
        
        positions = pos.squeeze(1)  # (batch, num_points, 2)
        inp = inp.permute(0, 2, 1)   # (batch, num_points, channels)
        
        # Time embedding
        t_emb = self.time_embed(t)  # (batch, time_dim)
        
        # Use cached graph or build new one
        if self._cached_edge_index is not None and self._cached_num_points == num_points:
            base_edge_index = self._cached_edge_index.to(device)
        else:
            # Build graph from first sample (assuming all have same positions)
            from torch_geometric.nn import knn_graph
            edge_index = knn_graph(positions[0], k=self.k_neighbors, loop=False)
            base_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        # Create batched edge index by offsetting
        edge_indices = []
        for b in range(batch_size):
            offset = b * num_points
            edge_indices.append(base_edge_index + offset)
        batched_edge_index = torch.cat(edge_indices, dim=1)
        
        # Flatten batch dimension
        x = inp.reshape(batch_size * num_points, -1)  # (batch * num_points, channels)
        
        # Expand time embedding
        batch_idx = torch.arange(batch_size, device=device).repeat_interleave(num_points)
        t_emb_expanded = t_emb[batch_idx]  # (batch * num_points, time_dim)
        
        # Forward through network
        h = self.input_proj(x)
        
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, batched_edge_index, t_emb_expanded)
        
        out = self.output_proj(h)
        
        # Reshape back
        out = out.view(batch_size, num_points, self.input_dim)
        out = out.permute(0, 2, 1)
        
        return out


class MeshGNNScoreNetwork(nn.Module):
    """
    GNN score network that uses mesh connectivity (dual graph) instead of k-NN.
    The edge_index is derived from mesh cell adjacency and must be set via set_mesh().
    """
    def __init__(
        self,
        input_dim: int = 1,
        pos_dim: int = 2,
        hidden_dim: int = 128,
        time_dim: int = 64,
        num_layers: int = 6,
        conv_type: str = 'gcn',
        heads: int = 4,
        dropout: float = 0.0,
        time_embedding: str = 'fourier',
    ):
        """
        Args:
            input_dim: Dimension of input signal (typically 1 for scalar field)
            pos_dim: Dimension of positions (2 for 2D mesh)
            hidden_dim: Hidden dimension for GNN layers
            time_dim: Dimension of time embedding
            num_layers: Number of GNN blocks
            conv_type: Type of graph convolution ('gcn' or 'gat')
            heads: Number of attention heads (for GAT)
            dropout: Dropout rate
            time_embedding: Type of time embedding ('sinusoidal' or 'fourier')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.pos_dim = pos_dim
        self.hidden_dim = hidden_dim
        
        # Time embedding
        if time_embedding == 'sinusoidal':
            self.time_embed = SinusoidalTimeEmbedding(time_dim)
        else:
            self.time_embed = FourierTimeEmbedding(time_dim)
        
        # Input projection: signal + position -> hidden
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim + pos_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GNNBlock(hidden_dim, hidden_dim, time_dim, conv_type, heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), 
            nn.Linear(hidden_dim, input_dim),
        )
        
        # Mesh connectivity cache
        self.register_buffer('_cached_edge_index', None, persistent=False)
        self._cached_num_points = None
        
    def set_mesh(self, edge_index: torch.Tensor, num_points: int = None):
        """
        Set the mesh connectivity (dual graph edge_index).
        
        Args:
            edge_index: (2, num_edges) tensor from mesh cell adjacency
            num_points: Number of mesh cells/nodes (inferred from edge_index if not provided)
        """
        self._cached_edge_index = edge_index
        if num_points is not None:
            self._cached_num_points = num_points
        else:
            self._cached_num_points = edge_index.max().item() + 1
        
    def forward(self, inp: torch.Tensor, t: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using mesh connectivity.
        
        Args:
            inp: Input tensor (batch, channels, num_points) where channels = pos_dim + input_dim
            t: Time tensor (batch,)
            pos: Position tensor (batch, 1, num_points, pos_dim) - for feature construction
            
        Returns:
            Output tensor (batch, input_dim, num_points)
        """
        if self._cached_edge_index is None:
            raise RuntimeError("Mesh not set. Call set_mesh(edge_index) before forward pass.")
            
        batch_size = inp.shape[0]
        num_points = inp.shape[2]
        device = inp.device
        
        # Move cached edge_index to correct device
        base_edge_index = self._cached_edge_index.to(device)
        
        # inp is (batch, channels, num_points) -> (batch, num_points, channels)
        inp = inp.permute(0, 2, 1)
        
        # Time embedding: (batch,) -> (batch, time_dim)
        t_emb = self.time_embed(t)
        
        # Create batched edge index by offsetting for each sample in batch
        edge_indices = []
        for b in range(batch_size):
            offset = b * num_points
            edge_indices.append(base_edge_index + offset)
        batched_edge_index = torch.cat(edge_indices, dim=1)
        
        # Flatten batch dimension
        x = inp.reshape(batch_size * num_points, -1)  # (batch * num_points, channels)
        
        # Expand time embedding to all nodes
        batch_idx = torch.arange(batch_size, device=device).repeat_interleave(num_points)
        t_emb_expanded = t_emb[batch_idx]  # (batch * num_points, time_dim)
        
        # Forward through network
        h = self.input_proj(x)
        
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, batched_edge_index, t_emb_expanded)
        
        out = self.output_proj(h)
        
        # Reshape back to (batch, input_dim, num_points)
        out = out.view(batch_size, num_points, self.input_dim)
        out = out.permute(0, 2, 1)
        
        return out


def create_gnn_score_model(
    hidden_dim: int = 128,
    time_dim: int = 64,
    num_layers: int = 6,
    conv_type: str = 'gcn',
    k_neighbors: int = 8,
    use_mesh_connectivity: bool = False,
    use_pde_message_passing: bool = False,
    **kwargs
) -> nn.Module:
    """
    Factory function to create a GNN score model.
    
    Args:
        hidden_dim: Hidden dimension
        time_dim: Time embedding dimension
        num_layers: Number of GNN layers
        conv_type: 'gcn' or 'gat'
        k_neighbors: Number of neighbors for graph construction (ignored if use_mesh_connectivity=True)
        use_mesh_connectivity: If True, use mesh-based connectivity (requires set_mesh(edge_index))
        use_pde_message_passing: If True, use PDEMessagePassing with position-aware messages
        
    Returns:
        GNN score network module
        
    Model combinations:
        - use_pde_message_passing=True, use_mesh_connectivity=True: MeshPDEGNNScoreNetwork
        - use_pde_message_passing=True, use_mesh_connectivity=False: PDEGNNScoreNetwork (k-NN)
        - use_pde_message_passing=False, use_mesh_connectivity=True: MeshGNNScoreNetwork
        - use_pde_message_passing=False, use_mesh_connectivity=False: GNNScoreNetworkOptimized (k-NN)
    """
    if use_pde_message_passing and use_mesh_connectivity:
        return MeshPDEGNNScoreNetwork(
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_layers=num_layers,
            **kwargs
        )
    elif use_pde_message_passing:
        return PDEGNNScoreNetwork(
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_layers=num_layers,
            k_neighbors=k_neighbors,
            **kwargs
        )
    elif use_mesh_connectivity:
        return MeshGNNScoreNetwork(
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_layers=num_layers,
            conv_type=conv_type,
            **kwargs
        )
    else:
        return GNNScoreNetworkOptimized(
            hidden_dim=hidden_dim,
            time_dim=time_dim,
            num_layers=num_layers,
            conv_type=conv_type,
            k_neighbors=k_neighbors,
            **kwargs
        )


if __name__ == "__main__":
    # Test the GNN score models
    from config import get_device
    device = get_device()
    print(f"Testing on device: {device}")
    
    batch_size = 4
    num_points = 200
    
    # Create model (k-NN based)
    model = GNNScoreNetworkOptimized(
        input_dim=1,
        pos_dim=2,
        hidden_dim=128,
        time_dim=64,
        num_layers=4,
        conv_type='gcn',
        k_neighbors=8,
    ).to(device)
    
    print(f"k-NN Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy inputs matching ScoreModel interface
    # pos: (batch, num_points, 2)
    pos = torch.rand(batch_size, num_points, 2).to(device)
    
    # xt: noisy signal (batch, 1, num_points)
    xt = torch.randn(batch_size, 1, num_points).to(device)
    
    # inp: concatenated [pos, xt] as in ScoreModel
    inp = torch.cat([pos.permute(0, 2, 1), xt], dim=1)  # (batch, 3, num_points)
    
    # t: time
    t = torch.rand(batch_size).to(device)
    
    # pos for graph construction: (batch, 1, num_points, 2)
    pos_for_model = pos.unsqueeze(1)
    
    # Pre-set mesh for optimized version
    model.set_mesh(pos[0])
    
    # Forward pass
    out = model(inp, t, pos_for_model)
    
    print(f"Input shape: {inp.shape}")
    print(f"Time shape: {t.shape}")
    print(f"Position shape: {pos_for_model.shape}")
    print(f"Output shape: {out.shape}")
    
    assert out.shape == (batch_size, 1, num_points), f"Expected {(batch_size, 1, num_points)}, got {out.shape}"
    print("k-NN GNN test passed!\n")
    
    # Test MeshGNNScoreNetwork
    print("Testing MeshGNNScoreNetwork...")
    
    mesh_model = MeshGNNScoreNetwork(
        input_dim=1,
        pos_dim=2,
        hidden_dim=128,
        time_dim=64,
        num_layers=4,
        conv_type='gcn',
    ).to(device)
    
    print(f"Mesh GNN Model parameters: {sum(p.numel() for p in mesh_model.parameters()):,}")
    
    # Create a simple triangular mesh edge_index for testing
    # Simulating dual graph: each node is a cell, connected if cells share an edge
    # For testing, create a random sparse connectivity
    num_edges = num_points * 3  # ~3 neighbors per cell on average
    src = torch.randint(0, num_points, (num_edges,))
    dst = torch.randint(0, num_points, (num_edges,))
    edge_index = torch.stack([src, dst], dim=0).to(device)
    # Make undirected
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    
    # Set mesh connectivity
    mesh_model.set_mesh(edge_index, num_points=num_points)
    
    # Forward pass
    out_mesh = mesh_model(inp, t, pos_for_model)
    
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Output shape: {out_mesh.shape}")
    
    assert out_mesh.shape == (batch_size, 1, num_points), f"Expected {(batch_size, 1, num_points)}, got {out_mesh.shape}"
    print("Mesh GNN test passed!\n")
    
    # Test PDEGNNScoreNetwork
    print("Testing PDEGNNScoreNetwork (with position-aware message passing)...")
    
    pde_model = PDEGNNScoreNetwork(
        input_dim=1,
        pos_dim=2,
        hidden_dim=128,
        time_dim=64,
        num_layers=4,
        dropout=0.0,
        k_neighbors=8,
    ).to(device)
    
    print(f"PDE GNN Model parameters: {sum(p.numel() for p in pde_model.parameters()):,}")
    
    # Pre-set mesh for caching
    pde_model.set_mesh(pos[0])
    
    # Forward pass
    out_pde = pde_model(inp, t, pos_for_model)
    
    print(f"Output shape: {out_pde.shape}")
    
    assert out_pde.shape == (batch_size, 1, num_points), f"Expected {(batch_size, 1, num_points)}, got {out_pde.shape}"
    print("PDEGNNScoreNetwork test passed!\n")
    
    # Test MeshPDEGNNScoreNetwork
    print("Testing MeshPDEGNNScoreNetwork (mesh connectivity + position-aware messages)...")
    
    mesh_pde_model = MeshPDEGNNScoreNetwork(
        input_dim=1,
        pos_dim=2,
        hidden_dim=128,
        time_dim=64,
        num_layers=4,
        dropout=0.0,
    ).to(device)
    
    print(f"Mesh PDE GNN Model parameters: {sum(p.numel() for p in mesh_pde_model.parameters()):,}")
    
    # Set mesh connectivity (reuse edge_index from MeshGNNScoreNetwork test)
    mesh_pde_model.set_mesh(edge_index, num_points=num_points)
    
    # Forward pass
    out_mesh_pde = mesh_pde_model(inp, t, pos_for_model)
    
    print(f"Output shape: {out_mesh_pde.shape}")
    
    assert out_mesh_pde.shape == (batch_size, 1, num_points), f"Expected {(batch_size, 1, num_points)}, got {out_mesh_pde.shape}"
    print("MeshPDEGNNScoreNetwork test passed!\n")
    
    print("All tests passed!")
