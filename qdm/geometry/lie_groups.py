"""
Lie Group Generators for QDM.

Implements infinitesimal generators for common symmetry groups:
  - SO(2): planar rotation (heading symmetry for robotics)
  - SO(3): 3D rotation (molecular conformer generation)
  - SE(3): rigid body (rotation + translation)

The generator xi_X(x) := d/dt [exp(t*xi) · x]|_{t=0}
spans the vertical space V_x = T_x(G · x).
"""

from __future__ import annotations
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


# --------------------------------------------------------------------------- #
#  SO(2) Generator  —  heading rotation around gravity axis                    #
# --------------------------------------------------------------------------- #

class SO2Generator(nn.Module):
    """
    Generator of SO(2) heading rotation acting on a robot state vector.

    The robot state x = (q, q_dot, R_flat, omega, tau) in R^d.
    SO(2) rotates only the heading-sensitive components: the base orientation
    yaw component and the horizontal velocity components.

    Args:
        state_dim: Total ambient state dimension d.
        heading_indices: Indices of components transformed by heading rotation.
                         If None, uses a default pattern for ANYmal-D (d=82).
    """

    def __init__(self, state_dim: int, heading_indices: Optional[list] = None):
        super().__init__()
        self.d = state_dim
        self.k = 1  # SO(2) is 1-dimensional

        if heading_indices is None:
            # Default for ANYmal-D: base orientation (yaw) + horiz. velocity
            # Indices are problem-specific; this is a sensible default
            heading_indices = list(range(24, 26))  # placeholder
        self.register_buffer(
            "heading_idx", torch.tensor(heading_indices, dtype=torch.long)
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute the infinitesimal generator vector field at x.

        Args:
            x: State tensor, shape (..., d).

        Returns:
            Generator vectors V_x, shape (..., d, 1).
        """
        *batch, d = x.shape
        assert d == self.d, f"Expected state dim {self.d}, got {d}"

        # Generator: rotation by 90 degrees in the heading plane
        # xi_SO2(x)_i = [-x_y, x_x] for heading (x,y) components, 0 elsewhere
        xi = torch.zeros(*batch, d, 1, device=x.device, dtype=x.dtype)

        # Pairs of heading-sensitive coordinates (x, y)
        for i in range(0, len(self.heading_idx) - 1, 2):
            ix = self.heading_idx[i]
            iy = self.heading_idx[i + 1]
            xi[..., ix, 0] = -x[..., iy]
            xi[..., iy, 0] = x[..., ix]

        return xi  # (..., d, 1)


# --------------------------------------------------------------------------- #
#  SO(3) Generator  —  3-D rotational symmetry for molecules                  #
# --------------------------------------------------------------------------- #

class SO3Generator(nn.Module):
    """
    Generator of SO(3) acting block-diagonally on N-atom conformations.

    For a molecule with N atoms in R^{3N}, the action of R ∈ SO(3) is
        R · x = (R q_1, R q_2, ..., R q_N)
    After centering (sum q_i = 0), translational dof are removed.

    The three generators correspond to infinitesimal rotations around
    the x-, y-, and z-axes (cross-product formula):
        (ξ^k_X(x))_i = e_k × q_i
    where e_k is the k-th standard basis vector.

    Args:
        n_atoms: Number of atoms N.
        center: If True, subtract center of mass before computing generators.
    """

    def __init__(self, n_atoms: int, center: bool = True):
        super().__init__()
        self.N = n_atoms
        self.d = 3 * n_atoms
        self.k = 3  # SO(3) is 3-dimensional
        self.center = center

        # Anti-symmetric matrices for Lie algebra generators
        # J_x, J_y, J_z ∈ so(3)
        Jx = torch.tensor([[0, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float32)
        Jy = torch.tensor([[0, 0, 1], [0, 0, 0], [-1, 0, 0]], dtype=torch.float32)
        Jz = torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float32)
        self.register_buffer("J", torch.stack([Jx, Jy, Jz], dim=0))  # (3, 3, 3)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute SO(3) generator matrix V_x = [ξ¹_X(x) | ξ²_X(x) | ξ³_X(x)].

        Args:
            x: Atom coordinates, shape (..., 3N).

        Returns:
            Generator matrix V_x, shape (..., 3N, 3).
        """
        *batch, d = x.shape
        assert d == self.d, f"Expected {self.d} dims (3×{self.N}), got {d}"

        # Reshape to (..., N, 3)
        coords = x.reshape(*batch, self.N, 3)

        if self.center:
            coords = coords - coords.mean(dim=-2, keepdim=True)

        # Compute ξ^k_X(x)_i = J_k @ q_i  for k=0,1,2
        # J: (3, 3, 3) →  apply each J_k to each atom
        # result: (..., N, 3, 3) → (..., 3N, 3)
        # xi[..., atom_i*3:(atom_i+1)*3, k] = J_k @ q_i
        V = torch.einsum("kab,...nb->...nka", self.J, coords)  # (..., N, 3, 3)
        V = V.reshape(*batch, self.N * 3, 3)  # (..., 3N, 3)
        return V  # (..., d, k=3)


# --------------------------------------------------------------------------- #
#  SE(3) Generator  —  rigid body (rotation + translation)                     #
# --------------------------------------------------------------------------- #

class SE3Generator(nn.Module):
    """
    Generator of SE(3) = SO(3) ⋉ R^3 on N-atom conformations.

    SE(3) has k = 6 generators: 3 rotational (from SO(3)) + 3 translational.
    Translational generators are constant unit vectors repeated for each atom.

    In practice, centering the coordinates removes the translational orbit,
    so this class is most useful when studying the full SE(3) quotient
    before centering.

    Args:
        n_atoms: Number of atoms N.
    """

    def __init__(self, n_atoms: int):
        super().__init__()
        self.N = n_atoms
        self.d = 3 * n_atoms
        self.k = 6  # SE(3) is 6-dimensional
        self.so3 = SO3Generator(n_atoms, center=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute SE(3) generator matrix V_x ∈ R^{3N × 6}.

        Args:
            x: Atom coordinates, shape (..., 3N).

        Returns:
            Generator matrix V_x, shape (..., 3N, 6).
        """
        *batch, d = x.shape

        # SO(3) part: first 3 columns
        V_rot = self.so3(x)  # (..., 3N, 3)

        # Translation part: last 3 columns
        # ξ^{3+k}_X(x)_i = e_k  (same for all atoms)
        # Tiled unit vectors in R^3 repeated N times
        e = torch.eye(3, device=x.device, dtype=x.dtype)  # (3, 3)
        V_trans = e.unsqueeze(0).expand(self.N, -1, -1)  # (N, 3, 3)
        V_trans = V_trans.reshape(self.d, 3)  # (3N, 3)
        # Broadcast over batch
        V_trans = V_trans.expand(*batch, -1, -1)  # (..., 3N, 3)

        V = torch.cat([V_rot, V_trans], dim=-1)  # (..., 3N, 6)
        return V


# --------------------------------------------------------------------------- #
#  Generic Lie Group Action                                                    #
# --------------------------------------------------------------------------- #

class LieGroupAction(nn.Module):
    """
    Abstract wrapper for any Lie group action.

    Subclass and implement `generator_matrix` and optionally `act`.

    Args:
        state_dim: Ambient dimension d.
        group_dim: Lie algebra dimension k.
    """

    def __init__(self, state_dim: int, group_dim: int):
        super().__init__()
        self.d = state_dim
        self.k = group_dim

    def generator_matrix(self, x: Tensor) -> Tensor:
        """Return V_x ∈ R^{d × k}, columns = Lie algebra generators at x."""
        raise NotImplementedError

    def act(self, x: Tensor, g: Tensor) -> Tensor:
        """Apply group element g to state x."""
        raise NotImplementedError

    def forward(self, x: Tensor) -> Tensor:
        return self.generator_matrix(x)
