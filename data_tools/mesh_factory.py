"""
Mesh factory for GRIFDIR multiscale meshes.

Supported domains
-----------------
  "square"  : unit square via dolfinx built-in  (kwargs: nx, ny)
  "l_shape" : L-shaped domain via gmsh + dolfinx (kwargs: maxh)

All builders return the same tuple:
  mesh_pos   : np.ndarray  [N, 2]  node coordinates (cell-centres)
  xy         : np.ndarray  [V, 2]  vertex coordinates
  cells      : np.ndarray  [C, 3]  per-triangle vertex indices
  edge_index : torch.Tensor [2, E] graph connectivity (dual graph)

Note: pinball mesh is loaded directly from HMSAE data dirs, not generated here.
"""

import numpy as np
import torch
from collections import defaultdict

# dolfinx / mpi4py imported lazily inside build_square_mesh only


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def cells_to_dual_edge_index(cells: np.ndarray) -> torch.Tensor:
    """Triangle cell array -> dual graph edge_index [2, E]."""
    edge_to_cells: dict = defaultdict(list)
    for idx, cell in enumerate(cells):
        for e in (
            tuple(sorted([cell[0], cell[1]])),
            tuple(sorted([cell[1], cell[2]])),
            tuple(sorted([cell[2], cell[0]])),
        ):
            edge_to_cells[e].append(idx)
    src, dst = [], []
    for cell_list in edge_to_cells.values():
        if len(cell_list) == 2:
            c1, c2 = cell_list
            src += [c1, c2]
            dst += [c2, c1]
    return torch.tensor([src, dst], dtype=torch.long)


def _extract_dolfinx(domain):
    """Extract arrays from a dolfinx mesh (requires dolfinx)."""
    from dolfinx.fem import functionspace
    domain.topology.create_connectivity(1, 2)
    xy = domain.geometry.x[:, :2].copy()
    cells = domain.geometry.dofmap.reshape((-1, domain.topology.dim + 1)).copy()
    V = functionspace(domain, ("DG", 0))
    mesh_pos = np.array(V.tabulate_dof_coordinates()[:, :2])
    edge_index = cells_to_dual_edge_index(cells)
    return mesh_pos, xy, cells, edge_index


def _extract_from_gmsh():
    """Extract mesh arrays directly from the active gmsh model (no dolfinx)."""
    import gmsh
    # Vertices
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    # node_tags are 1-based; build a map to 0-based
    max_tag = int(node_tags.max())
    tag_to_idx = np.full(max_tag + 1, -1, dtype=np.int64)
    tag_to_idx[node_tags.astype(int)] = np.arange(len(node_tags))
    xy = coords.reshape(-1, 3)[:, :2].copy()

    # Triangles (element type 2 = 3-node triangle)
    elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(dim=2)
    tri_idx = [i for i, t in enumerate(elem_types) if t == 2]
    if not tri_idx:
        raise RuntimeError("No triangles found in gmsh model")
    raw_nodes = elem_node_tags[tri_idx[0]].astype(int)
    cells = tag_to_idx[raw_nodes].reshape(-1, 3)

    # Cell centres (centroids)
    mesh_pos = xy[cells].mean(axis=1)

    edge_index = cells_to_dual_edge_index(cells)
    return mesh_pos, xy, cells, edge_index


# ---------------------------------------------------------------------------
# Domain builders
# ---------------------------------------------------------------------------

def build_square_mesh(nx: int, ny: int):
    """Unit square mesh via dolfinx."""
    from dolfinx import mesh as dolfin_mesh
    from mpi4py import MPI
    domain = dolfin_mesh.create_unit_square(
        MPI.COMM_WORLD, nx, ny, dolfin_mesh.CellType.triangle
    )
    return _extract_dolfinx(domain)


def build_lshape_mesh(maxh: float = 0.1):
    """
    L-shaped domain mesh via gmsh (Python API) + dolfinx.

    No Firedrake/Netgen required — only gmsh and dolfinx.
    gmsh is part of the standard dolfinx ecosystem.
    """
    import gmsh

    gmsh.initialize()
    gmsh.model.add("l_shape")

    # 6-vertex boundary polygon (CCW)
    pts = [
        gmsh.model.geo.addPoint(0, 0, 0, maxh),   # 0
        gmsh.model.geo.addPoint(1, 0, 0, maxh),   # 1
        gmsh.model.geo.addPoint(1, 1, 0, maxh),   # 2
        gmsh.model.geo.addPoint(2, 1, 0, maxh),   # 3
        gmsh.model.geo.addPoint(2, 2, 0, maxh),   # 4
        gmsh.model.geo.addPoint(0, 2, 0, maxh),   # 5
    ]
    lines = [gmsh.model.geo.addLine(pts[i], pts[(i + 1) % 6]) for i in range(6)]
    cl = gmsh.model.geo.addCurveLoop(lines)
    surf = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.add_physical_group(2, [surf], name="domain")
    gmsh.model.mesh.generate(2)

    result = _extract_from_gmsh()
    gmsh.finalize()
    return result


def build_plus_mesh(maxh: float = 0.1):
    """Plus / cross (+) shaped domain via gmsh.

    Bounding box [0,3]x[0,3], cross arms of width 1.
    """
    import gmsh
    gmsh.initialize()
    gmsh.model.add("plus")
    # 12-vertex boundary (CCW)
    pts = [
        gmsh.model.geo.addPoint(1, 0, 0, maxh),  # 0
        gmsh.model.geo.addPoint(2, 0, 0, maxh),  # 1
        gmsh.model.geo.addPoint(2, 1, 0, maxh),  # 2
        gmsh.model.geo.addPoint(3, 1, 0, maxh),  # 3
        gmsh.model.geo.addPoint(3, 2, 0, maxh),  # 4
        gmsh.model.geo.addPoint(2, 2, 0, maxh),  # 5
        gmsh.model.geo.addPoint(2, 3, 0, maxh),  # 6
        gmsh.model.geo.addPoint(1, 3, 0, maxh),  # 7
        gmsh.model.geo.addPoint(1, 2, 0, maxh),  # 8
        gmsh.model.geo.addPoint(0, 2, 0, maxh),  # 9
        gmsh.model.geo.addPoint(0, 1, 0, maxh),  # 10
        gmsh.model.geo.addPoint(1, 1, 0, maxh),  # 11
    ]
    lines = [gmsh.model.geo.addLine(pts[i], pts[(i + 1) % 12]) for i in range(12)]
    cl = gmsh.model.geo.addCurveLoop(lines)
    surf = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.add_physical_group(2, [surf], name="domain")
    gmsh.model.mesh.generate(2)
    result = _extract_from_gmsh()
    gmsh.finalize()
    return result


def build_e_shape_mesh(maxh: float = 0.1):
    """E-shaped domain via gmsh.

    Vertical spine [0,0.5]x[0,3] with three horizontal prongs at y=0, 1, 2
    extending to x=2, each of height 0.5.
    """
    import gmsh
    gmsh.initialize()
    gmsh.model.add("e_shape")
    pts = [
        gmsh.model.geo.addPoint(0,   0,   0, maxh),  # 0  bottom-left
        gmsh.model.geo.addPoint(2,   0,   0, maxh),  # 1  bottom prong right-bottom
        gmsh.model.geo.addPoint(2,   0.5, 0, maxh),  # 2  bottom prong right-top
        gmsh.model.geo.addPoint(0.5, 0.5, 0, maxh),  # 3  notch between bottom & mid
        gmsh.model.geo.addPoint(0.5, 1.0, 0, maxh),  # 4  mid prong inner-bottom
        gmsh.model.geo.addPoint(2,   1.0, 0, maxh),  # 5  mid prong right-bottom
        gmsh.model.geo.addPoint(2,   1.5, 0, maxh),  # 6  mid prong right-top
        gmsh.model.geo.addPoint(0.5, 1.5, 0, maxh),  # 7  notch between mid & top
        gmsh.model.geo.addPoint(0.5, 2.0, 0, maxh),  # 8  top prong inner-bottom
        gmsh.model.geo.addPoint(2,   2.0, 0, maxh),  # 9  top prong right-bottom
        gmsh.model.geo.addPoint(2,   2.5, 0, maxh),  # 10 top prong right-top
        gmsh.model.geo.addPoint(0,   2.5, 0, maxh),  # 11 top-left
    ]
    lines = [gmsh.model.geo.addLine(pts[i], pts[(i + 1) % 12]) for i in range(12)]
    cl = gmsh.model.geo.addCurveLoop(lines)
    surf = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.add_physical_group(2, [surf], name="domain")
    gmsh.model.mesh.generate(2)
    result = _extract_from_gmsh()
    gmsh.finalize()
    return result


def build_x_shape_mesh(maxh: float = 0.1):
    """X-shaped domain via gmsh.

    Union of two crossing diagonal bars with constant width (parallel edges).
    12-vertex polygon on [0,2]x[0,2]. Parameter d controls arm width:
    perpendicular width of each bar = 2*d*sqrt(2).
    """
    import gmsh
    gmsh.initialize()
    gmsh.model.add("x_shape")
    d = 0.3  # arm half-width parameter
    # 12-vertex polygon (CCW): 2 vertices per tip, 1 per concavity
    coords = [
        (2*d, 0),            # 0  BL tip, right edge
        (1, 1 - 2*d),        # 1  bottom concavity
        (2 - 2*d, 0),        # 2  BR tip, left edge
        (2, 2*d),            # 3  BR tip, right edge
        (1 + 2*d, 1),        # 4  right concavity
        (2, 2 - 2*d),        # 5  TR tip, left edge
        (2 - 2*d, 2),        # 6  TR tip, right edge
        (1, 1 + 2*d),        # 7  top concavity
        (2*d, 2),            # 8  TL tip, left edge
        (0, 2 - 2*d),        # 9  TL tip, right edge
        (1 - 2*d, 1),        # 10 left concavity
        (0, 2*d),            # 11 BL tip, left edge
    ]
    pts = [gmsh.model.geo.addPoint(x, y, 0, maxh) for x, y in coords]
    lines = [gmsh.model.geo.addLine(pts[i], pts[(i + 1) % len(pts)]) for i in range(len(pts))]
    cl = gmsh.model.geo.addCurveLoop(lines)
    surf = gmsh.model.geo.addPlaneSurface([cl])
    gmsh.model.geo.synchronize()
    gmsh.model.add_physical_group(2, [surf], name="domain")
    gmsh.model.mesh.generate(2)
    result = _extract_from_gmsh()
    gmsh.finalize()
    return result


def build_circle_mesh(maxh: float = 0.1, radius: float = 1.0,
                      cx: float = 1.0, cy: float = 1.0):
    """Circular (disk) domain via gmsh."""
    import gmsh
    gmsh.initialize()
    gmsh.model.add("circle")
    # OCC disk is simplest for circles
    gmsh.model.occ.addDisk(cx, cy, 0, radius, radius)
    gmsh.model.occ.synchronize()
    gmsh.model.add_physical_group(2, [1], name="domain")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", maxh)
    gmsh.model.mesh.generate(2)
    result = _extract_from_gmsh()
    gmsh.finalize()
    return result


def build_square_with_hole_mesh(maxh: float = 0.1, hole_radius: float = 0.25,
                                 cx: float = 1.0, cy: float = 1.0):
    """Unit square [0,2]x[0,2] with circular hole at (cx,cy)."""
    import gmsh
    gmsh.initialize()
    gmsh.model.add("square_with_hole")
    rect = gmsh.model.occ.addRectangle(0, 0, 0, 2, 2)
    hole = gmsh.model.occ.addDisk(cx, cy, 0, hole_radius, hole_radius)
    result, _ = gmsh.model.occ.cut([(2, rect)], [(2, hole)])
    gmsh.model.occ.synchronize()
    tags = [t for _, t in result]
    gmsh.model.add_physical_group(2, tags, name="domain")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", maxh)
    gmsh.model.mesh.generate(2)
    result = _extract_from_gmsh()
    gmsh.finalize()
    return result


def build_circle_with_hole_mesh(maxh: float = 0.1, outer_radius: float = 1.0,
                                 inner_radius: float = 0.3,
                                 cx: float = 1.0, cy: float = 1.0):
    """Annulus: outer circle minus inner circle."""
    import gmsh
    gmsh.initialize()
    gmsh.model.add("circle_with_hole")
    outer = gmsh.model.occ.addDisk(cx, cy, 0, outer_radius, outer_radius)
    inner = gmsh.model.occ.addDisk(cx, cy, 0, inner_radius, inner_radius)
    result, _ = gmsh.model.occ.cut([(2, outer)], [(2, inner)])
    gmsh.model.occ.synchronize()
    tags = [t for _, t in result]
    gmsh.model.add_physical_group(2, tags, name="domain")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", maxh)
    gmsh.model.mesh.generate(2)
    result = _extract_from_gmsh()
    gmsh.finalize()
    return result


# ---------------------------------------------------------------------------
# Factory entry point
# ---------------------------------------------------------------------------

_GMSH_DOMAINS = {
    'l_shape', 'plus', 'e_shape', 'x_shape',
    'circle', 'square_with_hole', 'circle_with_hole',
}

_DOMAIN_BUILDERS = {
    'l_shape':          build_lshape_mesh,
    'plus':             build_plus_mesh,
    'e_shape':          build_e_shape_mesh,
    'x_shape':          build_x_shape_mesh,
    'circle':           build_circle_mesh,
    'square_with_hole': build_square_with_hole_mesh,
    'circle_with_hole': build_circle_with_hole_mesh,
}


def build_mesh(domain: str, **kwargs):
    """
    Build mesh for a named domain.

    Parameters
    ----------
    domain : "square" | "l_shape" | "plus" | "e_shape" | "x_shape"
             | "circle" | "square_with_hole" | "circle_with_hole"
    kwargs :
        square  -> nx (int), ny (int)
        all others -> maxh (float, default 0.1), plus domain-specific params
    """
    if domain == "square":
        return build_square_mesh(kwargs["nx"], kwargs["ny"])
    elif domain in _DOMAIN_BUILDERS:
        builder = _DOMAIN_BUILDERS[domain]
        return builder(**{k: v for k, v in kwargs.items()})
    else:
        raise ValueError(
            f"Unknown domain '{domain}'. Supported: square, "
            + ", ".join(sorted(_DOMAIN_BUILDERS.keys()))
        )


def mesh_tag_for(domain: str, **kwargs) -> str:
    """Canonical filename tag for a domain + parameters."""
    if domain == "square":
        return f"nx{kwargs['nx']}_ny{kwargs['ny']}"
    maxh = kwargs.get('maxh', 0.1)
    return f"{domain}_maxh{maxh:.3f}"
