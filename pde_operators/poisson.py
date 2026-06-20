
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from scipy.sparse import csr_matrix
import torch

# FEniCSx/dolfinx is only needed to actually solve the Poisson forward problem
# (the fig-6 conditional sampling). Import lazily so this module loads torch-only.
try:
    from mpi4py import MPI
    import ufl
    from dolfinx import mesh, fem
    from dolfinx.io import gmshio
    from dolfinx.fem.petsc import assemble_matrix
    from dolfinx.fem import functionspace, form
    _DOLFINX_AVAILABLE = True
except ImportError:
    _DOLFINX_AVAILABLE = False

class PoissonOperator():
    def __init__(self, mesh, device="cuda"):
        if not _DOLFINX_AVAILABLE:
            raise ImportError(
                "PoissonOperator requires the FEniCSx/dolfinx stack (see environment.yml); "
                "not needed for torch-only train/eval."
            )
        self.omega = mesh
        self.omega.topology.create_connectivity(1, 2)
        #self.omega.topology.create_connectivity(1, 2)
        tdim = self.omega.topology.dim
        self.fdim = tdim - 1
        self.omega.topology.create_connectivity(self.fdim, tdim)

        self.V = functionspace(self.omega, ("Lagrange", 1)) # piecewise linear function space for solution u
        self.W = functionspace(self.omega, ("DG", 0)) # piecewise constant for parameter a 

        # degrees of freedom for the two discrete spaces
        u_ = fem.Function(self.V)
        a_ = fem.Function(self.W)

        self.dofs_pc = len(a_.x.array)
        self.dofs_pl = len(u_.x.array)

        self.device = device 
        self.M, self.Q = self.create_forward_operator()
        
        self.B = torch.eye(self.dofs_pl, device=self.device)

    def set_observation_operator(self, B):
        self.B = B

    def create_forward_operator(self):
        """
        Create the forward operator of the Poisson PDE 
            - laplace(u) = a in Omega 
                       u = 0 on boundary Omega
        
        The weak formulation is given by 

            int_Omega nabla(u) * nabla(v) dx = int_Omega a * v dx 

        In the discrete FEM version this will be a linear system 

                            A U = Q a
        """


        boundary_facets = mesh.exterior_facet_indices(self.omega.topology)
        boundary_dofs = fem.locate_dofs_topological(self.V, self.fdim, boundary_facets)

        uD = fem.Function(self.V)
        uD.interpolate(lambda x: 0*x[0]) # boundary value = 0 

        bc = fem.dirichletbc(uD, boundary_dofs)

        u = ufl.TrialFunction(self.V) 
        v = ufl.TestFunction(self.V)

        a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
        A = assemble_matrix(form(a), bcs=[bc])
        A.assemble()

        ai, aj, av = A.getValuesCSR()

        M = csr_matrix((av, aj, ai)) # M will be matrix used to solve problem
        M.resize(self.dofs_pl,self.dofs_pl) 

        # dont need A anymore 
        A.destroy() 
        
        w = ufl.TrialFunction(self.W)  
        # create the matrix to interpolate the piecewise constant a to the piecewise linear basis 
        bilinear_form = fem.form(ufl.inner(v, w) * ufl.dx)

        Q = fem.petsc.assemble_matrix(bilinear_form, [])
        Q.assemble()

        ai, aj, av = Q.getValuesCSR()

        Q_matrix = csr_matrix((av, aj, ai)) 
        Q.destroy() 

        Q_dense = torch.tensor(Q_matrix.toarray(), dtype=torch.float32).to(self.device)  # or float64
        M_dense = torch.tensor(M.toarray(), dtype=torch.float32).to(self.device)  # or float64

        return M_dense, Q_dense

    def solve_linear_system(self, a):
        rhs_torch = torch.matmul(self.Q, a)
        sol = torch.linalg.solve(self.M, rhs_torch)

        return sol 

    def forward(self, a):
        """
        Input - a: (..., dofs_pc)  e.g. (dofs_pc,), (1, dofs_pc), or (1, 1, dofs_pc)
        Output - u: (..., n_obs)   same leading dimensions, n_obs = dofs_pl (or fewer if B is set)
        """
        a = a.to(self.device)
        leading_shape = a.shape[:-1]
        a_flat = a.reshape(-1, a.shape[-1])  # (batch, dofs_pc)

        # Q @ a^T  -> (dofs_pl, batch),  then solve M x = Q a^T
        rhs = torch.matmul(self.Q, a_flat.T)          # (dofs_pl, batch)
        Minv_rhs = torch.linalg.solve(self.M, rhs)    # (dofs_pl, batch)
        res = torch.matmul(self.B, Minv_rhs)           # (n_obs, batch)

        # reshape back: (batch, n_obs) -> (*leading_shape, n_obs)
        return res.T.reshape(*leading_shape, -1)

    def adjoint(self, u):
        """
        
        u can have shape (..., n_obs) where n_obs is the number of observations (dofs_pl or fewer if B is set). The output will have shape (..., dofs_pc).
        """

        u = u.to(self.device)
        leading_shape = u.shape[:-1]
        u_flat = u.reshape(-1, u.shape[-1])  # (batch, n_obs)

        BTu = torch.matmul(self.B.T, u_flat.T)           # (dofs_pl, batch)
        MinvT_BTu = torch.linalg.solve(self.M, BTu)      # (dofs_pl, batch)
        result = torch.matmul(self.Q.T, MinvT_BTu)       # (dofs_pc, batch)

        return result.T.reshape(*leading_shape, -1)

