import torch 
import numpy as np
from numpy.typing import NDArray


def create_sample(mesh_pos):
    sigma_mesh = gen_conductivity(
        mesh_pos[:, 0], mesh_pos[:, 1], max_numInc=3, backCond=0.0
    )
    return torch.from_numpy(sigma_mesh).float().unsqueeze(0)


def draw_batch(batch_size, mesh_pos):
    x = [create_sample(mesh_pos) for _ in range(batch_size)]
    return torch.cat(x, dim=0)


def cart_ellipse_coords(x, y, h, k, a, b, alpha):
    """Compute normalized coordinates in rotated ellipse frame."""
    x_rot = (x - h) * np.cos(alpha) + (y - k) * np.sin(alpha)
    y_rot = -(x - h) * np.sin(alpha) + (y - k) * np.cos(alpha)
    return x_rot / a, y_rot / b


def get_ellipse_bounding_box(h, k, a, b, alpha):
    """
    Compute the axis-aligned bounding box of a rotated ellipse.
    
    Returns:
        x_min, x_max, y_min, y_max of the bounding box
    """
    # For a rotated ellipse, the bounding box extents are:
    # dx = sqrt(a^2 * cos^2(alpha) + b^2 * sin^2(alpha))
    # dy = sqrt(a^2 * sin^2(alpha) + b^2 * cos^2(alpha))
    cos_a = np.cos(alpha)
    sin_a = np.sin(alpha)
    
    dx = np.sqrt(a**2 * cos_a**2 + b**2 * sin_a**2)
    dy = np.sqrt(a**2 * sin_a**2 + b**2 * cos_a**2)
    
    return h - dx, h + dx, k - dy, k + dy


def sample_ellipse_unit_square(
    a_min: float = 0.08,
    a_max: float = 0.25,
    aspect_min: float = 0.5,
    aspect_max: float = 1.0,
    padding: float = 0.02,
    max_attempts: int = 100,
):
    """
    Sample a random ellipse that fits entirely within the unit square [0, 1] x [0, 1].
    
    Args:
        a_min: Minimum semi-major axis
        a_max: Maximum semi-major axis
        aspect_min: Minimum aspect ratio (b/a)
        aspect_max: Maximum aspect ratio (b/a), should be <= 1
        padding: Minimum distance from ellipse boundary to domain boundary
        max_attempts: Maximum sampling attempts before reducing size
        
    Returns:
        h, k: Center coordinates
        a, b: Semi-axes (a >= b)
        alpha: Rotation angle in radians
    """
    for attempt in range(max_attempts):
        # Sample semi-axes
        a = np.random.uniform(a_min, a_max)
        aspect = np.random.uniform(aspect_min, aspect_max)
        b = a * aspect
        
        # Sample rotation angle
        alpha = np.random.uniform(0, 2 * np.pi)
        
        # Compute bounding box half-widths for this ellipse at origin
        cos_a = np.cos(alpha)
        sin_a = np.sin(alpha)
        dx = np.sqrt(a**2 * cos_a**2 + b**2 * sin_a**2)
        dy = np.sqrt(a**2 * sin_a**2 + b**2 * cos_a**2)
        
        # Valid range for center to keep ellipse inside [0, 1] x [0, 1]
        h_min = dx + padding
        h_max = 1.0 - dx - padding
        k_min = dy + padding
        k_max = 1.0 - dy - padding
        
        # Check if valid placement exists
        if h_min < h_max and k_min < k_max:
            h = np.random.uniform(h_min, h_max)
            k = np.random.uniform(k_min, k_max)
            return h, k, a, b, alpha
        
        # If no valid placement, try smaller ellipse on next attempt
        a_max = max(a_min, a_max * 0.9)
    
    # Fallback: small circular ellipse in center region
    a = a_min
    b = a_min
    alpha = 0.0
    h = np.random.uniform(0.2, 0.8)
    k = np.random.uniform(0.2, 0.8)
    return h, k, a, b, alpha


def ellipses_overlap(h1, k1, a1, b1, alpha1, h2, k2, a2, b2, alpha2, min_gap: float = 0.02):
    """
    Check if two ellipses overlap (with a minimum gap between them).
    Uses a conservative bounding box check plus center distance.
    """
    # Get bounding boxes
    x1_min, x1_max, y1_min, y1_max = get_ellipse_bounding_box(h1, k1, a1, b1, alpha1)
    x2_min, x2_max, y2_min, y2_max = get_ellipse_bounding_box(h2, k2, a2, b2, alpha2)
    
    # Check bounding box overlap with gap
    if (x1_max + min_gap < x2_min or x2_max + min_gap < x1_min or
        y1_max + min_gap < y2_min or y2_max + min_gap < y1_min):
        return False
    
    # Additional center distance check (conservative)
    center_dist = np.sqrt((h1 - h2)**2 + (k1 - k2)**2)
    min_dist = max(a1, b1) + max(a2, b2) + min_gap
    
    return center_dist < min_dist


def sample_inclusions(
    numInc: int,
    a_min: float = 0.08,
    a_max: float = 0.25,
    aspect_min: float = 0.5,
    aspect_max: float = 1.0,
    padding: float = 0.02,
):
    """
    Sample multiple ellipses within the unit square (overlapping allowed).
    
    Args:
        numInc: Number of inclusions to sample
        a_min, a_max: Range for semi-major axis
        aspect_min, aspect_max: Range for aspect ratio (b/a)
        padding: Minimum distance from domain boundary
        
    Returns:
        h, k, a, b, alpha: Arrays of ellipse parameters
    """
    h = np.zeros(numInc)
    k = np.zeros(numInc)
    a = np.zeros(numInc)
    b = np.zeros(numInc)
    alpha = np.zeros(numInc)
    
    for i in range(numInc):
        h[i], k[i], a[i], b[i], alpha[i] = sample_ellipse_unit_square(
            a_min=a_min, a_max=a_max,
            aspect_min=aspect_min, aspect_max=aspect_max,
            padding=padding, max_attempts=50
        )
    
    return h, k, a, b, alpha


def gen_conductivity(
    x1: NDArray[np.float64],
    x2: NDArray[np.float64],
    max_numInc: int,
    backCond: float = 1.0,
    minCond: float = 0.01,
) -> NDArray[np.float64]:
    """
    Generate conductivity field with negative elliptical inclusions.
    
    Args:
        x1, x2: Coordinate arrays
        max_numInc: Maximum number of inclusions
        backCond: Background conductivity (default 1.0)
        minCond: Minimum conductivity at ellipse centers (default 0.01)
        
    Returns:
        Conductivity field with values in [minCond, backCond]
    """
    numInc = np.random.randint(1, max_numInc + 1)
    condOut = np.ones(x1.shape) * backCond
    h, k, a, b, alpha = sample_inclusions(numInc)

    for i in range(numInc):
        # Random depth: how far down towards minCond (0.5 to 1.0 means 50-100% of the way)
        depth = np.random.uniform(0.5, 1.0)
        target_min = backCond - depth * (backCond - minCond)
        
        x_norm, y_norm = cart_ellipse_coords(x1, x2, h[i], k[i], a[i], b[i], alpha[i])
        
        # Gaussian blob (elliptical)
        blob = np.exp(-0.5 * (x_norm**2 + y_norm**2))
        
        # Create dip: subtract from background, going towards target_min at center
        dip = backCond - (backCond - target_min) * blob
        
        # Blend: take the lower value (creates negative inclusions)
        condOut = np.minimum(condOut, dip)

    return condOut


def boundary_distance_mask(
    mesh_pos: NDArray[np.float64],
    xy: NDArray[np.float64],
    cells: NDArray[np.int64],
    margin: float = 0.12,
) -> NDArray[np.bool_]:
    """Return a boolean mask of cells whose centroid is ≥ margin·L from the boundary.

    Uses a mesh-based SDF approximation: identifies boundary vertices (those
    on edges shared by exactly one triangle), builds a KDTree, then keeps
    cell centres whose nearest-boundary-vertex distance exceeds ``margin * L``
    where L is the bounding-box diagonal.

    Parameters
    ----------
    mesh_pos : (N, 2)  DG0 cell-centre coordinates.
    xy       : (V, 2)  vertex coordinates.
    cells    : (N, 3)  triangle vertex indices.
    margin   : admissible fraction of domain diagonal (default 0.12 — roughly
               one blob-radius inward from the boundary).

    Returns
    -------
    (N,) bool array; True where a cell is admissible as a blob centre.
    """
    from collections import defaultdict
    from scipy.spatial import cKDTree

    # --- find boundary vertices -------------------------------------------
    edge_count: dict = defaultdict(int)
    for tri in cells:
        for e in (
            tuple(sorted([tri[0], tri[1]])),
            tuple(sorted([tri[1], tri[2]])),
            tuple(sorted([tri[2], tri[0]])),
        ):
            edge_count[e] += 1

    boundary_verts: set = set()
    for (u, v), count in edge_count.items():
        if count == 1:          # boundary edge: belongs to exactly one triangle
            boundary_verts.add(u)
            boundary_verts.add(v)

    if not boundary_verts:
        # Closed manifold with no boundary — accept all cells
        return np.ones(len(mesh_pos), dtype=bool)

    bv_coords = xy[list(boundary_verts)]   # (B, 2)

    # --- KDTree distance from each cell centre to nearest boundary vertex --
    tree = cKDTree(bv_coords)
    dist, _ = tree.query(mesh_pos, k=1)    # (N,)

    x_range = float(xy[:, 0].max() - xy[:, 0].min())
    y_range = float(xy[:, 1].max() - xy[:, 1].min())
    L = np.sqrt(x_range ** 2 + y_range ** 2)

    return dist >= margin * L


def gen_conductivity_on_mesh(
    mesh_pos: NDArray[np.float64],
    max_numInc: int = 3,
    backCond: float = 1.0,
    minCond: float = 0.01,
    blob_scale: float = 0.15,
    centre_mask: NDArray[np.bool_] | None = None,
) -> NDArray[np.float64]:
    """Generate a conductivity field on *any* 2D mesh domain.

    Blob centres are drawn from the subset of cells given by ``centre_mask``
    (an admissible interior region, e.g. from ``boundary_distance_mask``).
    If no mask is supplied, all cell positions are used — but callers are
    encouraged to pass a pre-computed mask so blobs stay well inside the
    domain and away from thin arms / holes.

    Blob semi-axes are scaled relative to the bounding-box diagonal so the
    spatial frequency matches ``gen_conductivity`` on the unit square.

    Parameters
    ----------
    mesh_pos : (N, 2) array of DG0 cell-centre coordinates.
    max_numInc : maximum number of Gaussian inclusions.
    backCond : background conductivity.
    minCond : minimum conductivity at blob centres.
    blob_scale : blob semi-major axis as a fraction of the domain diagonal
        (default 0.15; this matches the original unit-square function where
        a ∈ [0.08, 0.25] on a diagonal of √2 ≈ 1.41 → ~0.06–0.18·L).
    centre_mask : optional bool (N,) array restricting where blob centres may
        be placed.  Build with ``boundary_distance_mask`` before the loop so
        it is only computed once per mesh.

    Returns
    -------
    (N,) float64 array of conductivity values in [minCond, backCond].
    """
    x1 = mesh_pos[:, 0]
    x2 = mesh_pos[:, 1]

    # Characteristic length: bounding-box diagonal
    x_range = float(x1.max() - x1.min())
    y_range = float(x2.max() - x2.min())
    L = np.sqrt(x_range ** 2 + y_range ** 2)

    a_min = blob_scale * 0.5 * L
    a_max = blob_scale * L

    numInc = np.random.randint(1, max_numInc + 1)
    condOut = np.ones(len(mesh_pos)) * backCond

    # Admissible centre positions
    if centre_mask is not None and centre_mask.any():
        candidate_indices = np.where(centre_mask)[0]
    else:
        candidate_indices = np.arange(len(mesh_pos))

    centre_idx = np.random.choice(candidate_indices, size=numInc, replace=True)

    for i in range(numInc):
        h, k = mesh_pos[centre_idx[i]]
        a = np.random.uniform(a_min, a_max)
        b = a * np.random.uniform(0.5, 1.0)   # random aspect ratio
        alpha = np.random.uniform(0, 2 * np.pi)
        depth = np.random.uniform(0.5, 1.0)
        target_min = backCond - depth * (backCond - minCond)

        x_norm, y_norm = cart_ellipse_coords(x1, x2, h, k, a, b, alpha)
        blob = np.exp(-0.5 * (x_norm ** 2 + y_norm ** 2))
        dip = backCond - (backCond - target_min) * blob
        condOut = np.minimum(condOut, dip)

    return condOut
