"""
Sparse sensor observation operator.

Forward operator that observes the solution at a fixed set of
randomly chosen mesh nodes (sensor locations).

    y = B @ u

where B is a binary (n_sensors x n_dofs) selection matrix
built once from a seeded random draw.
"""

import numpy as np
import torch


class SparseSensorOperator:
    """
    Observation operator that picks out point-wise values at
    ``n_sensors`` randomly selected degrees of freedom.

    Parameters
    ----------
    n_dofs : int
        Total number of degrees of freedom in the discrete solution.
    n_sensors : int
        Number of sensor locations to sample.
    seed : int
        Random seed used to draw the sensor indices (ensures
        reproducibility – sensors are sampled once at construction).
    device : str
        Torch device for all tensors (default ``"cuda"``).
    """

    def __init__(self, n_dofs: int, n_sensors: int, seed: int = 42,
                 device: str = "cuda"):
        assert n_sensors <= n_dofs, (
            f"n_sensors ({n_sensors}) must be <= n_dofs ({n_dofs})"
        )

        self.n_dofs = n_dofs
        self.n_sensors = n_sensors
        self.seed = seed
        self.device = device

        # --- sample sensor indices (fixed for the lifetime of the object) ---
        rng = np.random.RandomState(seed)
        self.sensor_indices = torch.from_numpy(
            rng.choice(n_dofs, size=n_sensors, replace=False)
        ).long().to(device)

        # --- build the sparse binary observation matrix B ---
        #   B has shape (n_sensors, n_dofs) with exactly one 1 per row.
        self.B = torch.zeros(n_sensors, n_dofs, device=device)
        self.B[torch.arange(n_sensors, device=device), self.sensor_indices] = 1.0

    # --------------------------------------------------------------------- #
    #  Forward / adjoint that match the PoissonOperator interface
    # --------------------------------------------------------------------- #
    def forward(self, u):
        """
        Apply the observation operator.

        Parameters
        ----------
        u : Tensor
            Solution vector(s).  Supports shapes
            ``(n_dofs,)`` or ``(batch, n_dofs)``.

        Returns
        -------
        y : Tensor
            Observed values at the sensor locations.
            Shape ``(n_sensors,)`` or ``(batch, n_sensors)``.
        """
        return torch.matmul(u, self.B.T) # (..., n_sensors)

    def adjoint(self, y):
        """
        Apply the adjoint (transpose) of the observation operator.

        Parameters
        ----------
        y : Tensor
            Sensor measurement(s).  Supports shapes
            ``(n_sensors,)`` or ``(batch, n_sensors)``.

        Returns
        -------
        u : Tensor
            Back-projected vector of shape ``(n_dofs,)`` or
            ``(batch, n_dofs)``.
        """
        return torch.matmul(y, self.B)   # (..., n_dofs)

    def to(self, device):
        """Move all tensors to *device* (in-place)."""
        self.device = device
        self.B = self.B.to(device)
        self.sensor_indices = self.sensor_indices.to(device)
        return self
