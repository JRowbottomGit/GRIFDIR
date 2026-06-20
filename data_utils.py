import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

try:
    from dolfinx import mesh
    from dolfinx.fem import functionspace
    from mpi4py import MPI
    DOLFINX_AVAILABLE = True
except ImportError:
    DOLFINX_AVAILABLE = False

from data_tools.gaussian_blob_utils import gen_conductivity


# ---------------------------------------------------------------------------
# Canonical multi-domain registry (fig 4). Order is fixed: the per-domain
# one-hot fed to the model is indexed by these ids, so the checkpoints depend
# on it. DOMAIN_FILES are the per-domain .pt datasets under data/.
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).resolve().parent / "data"

DOMAIN_REGISTRY = {
    "circle":           0,
    "circle_with_hole": 1,
    "e_shape":          2,
    "l_shape":          3,
    "plus":             4,
    "square":           5,
    "square_with_hole": 6,
    "x_shape":          7,
}
N_DOMAINS = len(DOMAIN_REGISTRY)           # 8
_ONEHOT = torch.eye(N_DOMAINS)             # index by domain id -> one-hot row

DOMAIN_FILES = {
    "circle":           _DATA_DIR / "conductivity_circle_maxh0.100_n1000.pt",
    "circle_with_hole": _DATA_DIR / "conductivity_circle_with_hole_maxh0.100_n1000.pt",
    "e_shape":          _DATA_DIR / "conductivity_e_shape_maxh0.100_n1000.pt",
    "l_shape":          _DATA_DIR / "conductivity_l_shape_maxh0.100_n1000.pt",
    "plus":             _DATA_DIR / "conductivity_plus_maxh0.100_n1000.pt",
    "square":           _DATA_DIR / "conductivity_nx32_ny32_n10000.pt",
    "square_with_hole": _DATA_DIR / "conductivity_square_with_hole_maxh0.100_n1000.pt",
    "x_shape":          _DATA_DIR / "conductivity_x_shape_maxh0.100_n1000.pt",
}


def domain_onehot(domain):
    """Return the fixed [N_DOMAINS] one-hot vector for a domain name."""
    if domain not in DOMAIN_REGISTRY:
        raise ValueError(f"Unknown domain '{domain}'. Known: {list(DOMAIN_REGISTRY)}")
    return _ONEHOT[DOMAIN_REGISTRY[domain]]


def cells_to_dual_edge_index(cells: np.ndarray) -> torch.Tensor:
    """
    Convert mesh cells (triangles) to a dual graph edge_index.
    In the dual graph, each cell is a node and cells sharing an edge are connected.
    
    Args:
        cells: (num_cells, 3) array of vertex indices for each triangle
        
    Returns:
        edge_index: (2, num_edges) tensor of cell-to-cell connections
    """
    from collections import defaultdict
    
    # Build edge -> cells mapping
    # An edge is defined by two vertices (as a sorted tuple)
    edge_to_cells = defaultdict(list)
    
    for cell_idx, cell in enumerate(cells):
        # Each triangle has 3 edges
        edges = [
            tuple(sorted([cell[0], cell[1]])),
            tuple(sorted([cell[1], cell[2]])),
            tuple(sorted([cell[2], cell[0]])),
        ]
        for edge in edges:
            edge_to_cells[edge].append(cell_idx)
    
    # Build dual graph edges: connect cells that share an edge
    src_list = []
    dst_list = []
    
    for edge, cell_list in edge_to_cells.items():
        if len(cell_list) == 2:
            # Interior edge: connects two cells
            c1, c2 = cell_list
            src_list.extend([c1, c2])
            dst_list.extend([c2, c1])
        # Boundary edges (len == 1) don't create connections
    
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    return edge_index


class PinballDataset(Dataset):
    """
    Load pinball data directly from HMSAE data directory.

    Expects:
      data_dir/Pinball_Ytrain.pt       [n_traj, n_times, n_spatial]
      data_dir/Pinball_MUtrain.pt      [n_traj, n_times, n_params]
      data_dir/Pinball_mesh_coords.pt  [n_spatial, 2]
      data_dir/Pinball_edge_index.pt   [2, E]  (or edge_index.pt)
    """

    def __init__(self, data_dir, num_samples=None, split: str = 'train'):
        data_dir = Path(data_dir)
        _suffix = {'train': 'train', 'val': 'valid', 'test': 'test'}.get(split)
        if _suffix is None:
            raise ValueError(f"Unknown split {split!r}; expected train/val/test")

        # ---- Field data [n_traj, n_times, n_spatial] ----
        Y = torch.load(data_dir / f'Pinball_Y{_suffix}.pt', weights_only=True)
        self.n_traj, self.n_times, self.n_spatial = Y.shape

        # ---- Parameter data [n_traj, n_times, n_params] ----
        mu_path = data_dir / f'Pinball_MU{_suffix}.pt'
        if not mu_path.exists():
            mu_path = data_dir.parent / f'Pinball_MU{_suffix}.pt'
        MU = torch.load(mu_path, weights_only=True)
        self.n_params = MU.shape[-1]

        # Flatten trajectories → independent samples
        self.fields = Y.reshape(-1, self.n_spatial).unsqueeze(1).float()   # [N, 1, n_spatial]
        self.mu     = MU.reshape(-1, self.n_params).float()                # [N, n_params]
        # Physical time index per sample: [0,1,...,30, 0,1,...,30, ...]
        self.time_idx = torch.arange(self.n_times).repeat(self.n_traj).float()  # [N]

        if num_samples is not None and len(self.fields) > num_samples:
            self.fields   = self.fields[:num_samples]
            self.mu       = self.mu[:num_samples]
            self.time_idx = self.time_idx[:num_samples]

        # ---- Vertex coordinates ----
        coords = torch.load(data_dir / 'Pinball_mesh_coords.pt', weights_only=False)
        if not isinstance(coords, torch.Tensor):
            coords = torch.from_numpy(np.asarray(coords)).float()
        self.mesh_coords = coords.float()                  # [n_spatial, 2] tensor
        self.mesh_pos = self.mesh_coords.numpy().copy()     # numpy for compat

        # ---- Edge index ----
        ei_path = data_dir / 'Pinball_edge_index.pt'
        if not ei_path.exists():
            ei_path = data_dir / 'edge_index.pt'
        if not ei_path.exists():
            raise FileNotFoundError(f"No edge_index .pt found in {data_dir}")
        self.edge_index = torch.load(ei_path, weights_only=True).long()

        print(f"PinballDataset: {len(self.fields)} samples "
              f"({self.n_traj} traj × {self.n_times} steps), "
              f"{self.n_spatial} vertices, {self.edge_index.shape[1]} edges, "
              f"mu dim={self.n_params}")

    def __len__(self):
        return len(self.fields)

    def __getitem__(self, idx):
        return self.fields[idx], self.mu[idx], self.time_idx[idx]

    def get_mesh_info(self):
        return self.mesh_pos, self.mesh_coords, self.edge_index


class ConductivityDataset(Dataset):
    """
    PyTorch Dataset that generates conductivity samples on a uniform unit square mesh.
    Samples are generated on-the-fly using gen_conductivity.
    """
    
    def __init__(self, num_samples: int, nx: int = 32, ny: int = 32, max_numInc: int = 3, backCond: float = 1.0):
        """
        Args:
            num_samples: Number of samples in the dataset
            nx: Number of cells in x direction
            ny: Number of cells in y direction
            max_numInc: Maximum number of inclusions for conductivity generation
            backCond: Background conductivity value
        """
        if not DOLFINX_AVAILABLE:
            raise ImportError("dolfinx is required for ConductivityDataset but not available. Use pre-generated data instead.")
        
        self.num_samples = num_samples
        self.max_numInc = max_numInc
        self.backCond = backCond
        
        # Create unit square mesh
        comm = MPI.COMM_WORLD
        domain = mesh.create_unit_square(comm, nx, ny, mesh.CellType.triangle)
        domain.topology.create_connectivity(1, 2)
        
        # Store mesh geometry
        self.xy = domain.geometry.x
        self.cells = domain.geometry.dofmap.reshape((-1, domain.topology.dim + 1))
        
        # Get mesh positions (cell centers for piecewise constant functions)
        V = functionspace(domain, ("DG", 0))
        self.mesh_pos = np.array(V.tabulate_dof_coordinates()[:, :2])
        # [num_cells, 2]
        
        # Build dual graph edge_index from mesh connectivity
        self.edge_index = cells_to_dual_edge_index(self.cells)

        self.domain = domain  # Store domain for potential use in conductivity generation
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        sigma_mesh = gen_conductivity(
            self.mesh_pos[:, 0], self.mesh_pos[:, 1], 
            max_numInc=self.max_numInc, 
            backCond=self.backCond
        )
        return torch.from_numpy(sigma_mesh).float().unsqueeze(0)
    
    def get_mesh_info(self):
        """Returns mesh positions, xy coordinates, cells, and edge_index for visualization and GNN."""
        return self.mesh_pos, self.xy, self.cells, self.edge_index


class CachedDataset(Dataset):
    """Cached conductivity dataset.

    Monolithic `.pt` files have no on-disk split.  We apply a deterministic
    runtime 90/10 train/test slice (matching the PhySense SST convention of
    train/test only — training-time val reuses the train set).
    """

    def __init__(self, path, split: str = 'train'):
        data = torch.load(path, weights_only=False)
        samples = data['samples']
        self.mesh_pos = data['mesh_pos'].numpy()
        self.xy = data['xy'].numpy()
        self.cells = data['cells'].numpy()
        self.edge_index = data['edge_index']

        N = len(samples)
        n_train = int(0.9 * N)
        if split == 'train':
            self.samples = samples[:n_train]
        elif split == 'test':
            self.samples = samples[n_train:]
        else:
            raise ValueError(f"Unknown split {split!r}; expected train/test")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def get_mesh_info(self):
        return self.mesh_pos, self.xy, self.cells, self.edge_index


def get_conductivity_dataloader(
    num_samples: int,
    batch_size: int = 32,
    nx: int = 10,
    ny: int = 10,
    max_numInc: int = 3,
    backCond: float = 1.0,
    shuffle: bool = True,
    num_workers: int = 0,
):
    """
    Creates a DataLoader for conductivity samples on a uniform unit square mesh.
    
    Args:
        num_samples: Number of samples in the dataset
        batch_size: Batch size for the DataLoader
        nx: Number of cells in x direction
        ny: Number of cells in y direction
        max_numInc: Maximum number of inclusions
        backCond: Background conductivity
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader and the dataset
    """
    dataset = ConductivityDataset(
        num_samples=num_samples,
        nx=nx,
        ny=ny,
        max_numInc=max_numInc,
        backCond=backCond,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
    
    return dataloader, dataset


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    
    # Example usage
    dataloader, dataset = get_conductivity_dataloader(
        num_samples=100,
        batch_size=4,
        nx=32,
        ny=32,
        max_numInc=3,
        backCond=1.0,
    )
    
    # Get mesh info for visualization
    mesh_pos, xy, cells, edge_index = dataset.get_mesh_info()
    tri = Triangulation(xy[:, 0], xy[:, 1], cells)
    
    print(f"Mesh positions shape: {mesh_pos.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Number of cells: {len(cells)}, Number of edges in dual graph: {edge_index.shape[1] // 2}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Get a batch and visualize
    for batch in dataloader:
        print(f"Batch shape: {batch.shape}")
        
        # Plot first sample in batch
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        im = ax.tripcolor(tri, batch[0].flatten().numpy(), cmap='Blues', shading='flat', edgecolors='k')
        ax.axis('image')
        ax.set_aspect('equal', adjustable='box')
        ax.set_title("Conductivity Sample")
        ax.axis("off")
        fig.colorbar(im, ax=ax)
        plt.savefig("conductivity_sample.png")
        plt.show()
        break
