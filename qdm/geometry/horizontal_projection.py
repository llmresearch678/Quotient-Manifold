"""
Horizontal Projection Operators for QDM.

Implements Algorithm 1 from the paper:
  - HorizontalProjection: exact orthogonal projection onto H_x = V_x^⊥
  - SoftHorizontalProjection: λ·Π_x + (1-λ)·I_d (Eq. 7)
  - ContextAdaptiveProjection: context-dependent V_x(τ) via learned encoder (Eq. 5-6)

All projections use a rank-k update (O(dk), never materializing a d×d matrix)
with Cholesky decomposition for numerical stability (float32 Gram matrix,
bfloat16 for the rest).

References:
  Proposition 1: smoothness and constant rank d-k.
  Theorem 4: Fisher information decomposition justifying variance reduction.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


# --------------------------------------------------------------------------- #
#  Utility: rank-k update projection                                           #
# --------------------------------------------------------------------------- #

def _rank_k_projection(
    v: Tensor,
    V: Tensor,
    eps: float = 1e-3,
) -> Tensor:
    """
    Compute Π_x · v  =  v  -  V (V^T V + ε I_k)^{-1} V^T v
    using Cholesky decomposition.  O(dk + k^3), never forms the d×d matrix.

    Args:
        v:   Vectors to project, shape (..., d) or (..., d, m).
        V:   Generator matrix, shape (..., d, k).
        eps: Regularisation constant ε > 0.

    Returns:
        Projected vectors, same shape as v.
    """
    squeeze = v.dim() == V.dim() - 1
    if squeeze:
        v = v.unsqueeze(-1)  # (..., d, 1)

    # Gram matrix in float32 for numerical stability
    V32 = V.float()
    v32 = v.float()

    G = V32.transpose(-2, -1) @ V32  # (..., k, k)
    G = G + eps * torch.eye(G.shape[-1], device=G.device, dtype=G.dtype)

    # Solve G L = V^T v  via Cholesky
    try:
        L = torch.linalg.cholesky(G)  # (..., k, k)
        VTv = V32.transpose(-2, -1) @ v32  # (..., k, m)
        # Two triangular solves: L y = VTv, then L^T z = y
        y = torch.linalg.solve_triangular(L, VTv, upper=False)
        z = torch.linalg.solve_triangular(L.transpose(-2, -1), y, upper=True)
    except torch.linalg.LinAlgError:
        # Fallback to pseudoinverse if Cholesky fails
        z = torch.linalg.lstsq(G, V32.transpose(-2, -1) @ v32).solution

    proj = v32 - V32 @ z  # (..., d, m)

    # Cast back to input dtype
    proj = proj.to(v.dtype)

    if squeeze:
        proj = proj.squeeze(-1)

    return proj


# --------------------------------------------------------------------------- #
#  Exact Horizontal Projection                                                 #
# --------------------------------------------------------------------------- #

class HorizontalProjection(nn.Module):
    """
    Exact orthogonal horizontal projection Π_x = I - V_x (V_x^T V_x + ε I)^{-1} V_x^T.

    Projects any ambient vector onto the horizontal subspace H_x = V_x^⊥,
    completely eliminating symmetry-orbit (vertical) components.

    This is the core operation underlying QDM's SDE-level symmetry reduction.

    Args:
        eps: Regularisation constant ε for Gram matrix inversion.
    """

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, v: Tensor, V: Tensor) -> Tensor:
        """
        Apply horizontal projection.

        Args:
            v: Input vectors, shape (..., d) or (..., d, m).
            V: Generator matrix, shape (..., d, k).

        Returns:
            Horizontally projected vectors, same shape as v.
        """
        return _rank_k_projection(v, V, eps=self.eps)

    def project_score(self, score: Tensor, V: Tensor) -> Tensor:
        """Alias for forward — project a score vector."""
        return self(score, V)

    def vertical_component(self, v: Tensor, V: Tensor) -> Tensor:
        """Return the vertical (symmetry-orbit) component of v."""
        return v - self(v, V)

    def vertical_fraction(self, v: Tensor, V: Tensor) -> Tensor:
        """
        Compute ||v_vert|| / ||v|| — the fraction of norm in vertical directions.
        Useful as a diagnostic (should be 0 for QDM, ~k/d for ambient).
        """
        v_vert = self.vertical_component(v, V)
        numer = v_vert.norm(dim=-1)
        denom = v.norm(dim=-1).clamp(min=1e-8)
        return numer / denom


# --------------------------------------------------------------------------- #
#  Soft Horizontal Projection                                                  #
# --------------------------------------------------------------------------- #

class SoftHorizontalProjection(nn.Module):
    """
    Soft horizontal projection: Π_x^soft = λ · Π_x + (1 - λ) · I_d.

    Interpolates between full quotient reduction (λ=1) and ambient diffusion
    (λ=0), where λ = exp(-γ · ε(τ)) is the symmetry confidence weight.

    This handles approximate symmetry gracefully: as terrain anisotropy grows
    or molecular flexibility increases, λ decreases and the projection relaxes.

    Args:
        eps: Regularisation for Gram matrix inversion.
        gamma: Decay rate γ in λ = exp(-γ · ε(τ)). Learned by default.
    """

    def __init__(self, eps: float = 1e-3, gamma: float = 2.1):
        super().__init__()
        self.eps = eps
        self.log_gamma = nn.Parameter(torch.log(torch.tensor(gamma)))
        self._proj = HorizontalProjection(eps=eps)

    @property
    def gamma(self) -> Tensor:
        return self.log_gamma.exp()

    def symmetry_confidence(self, symmetry_residual: Tensor) -> Tensor:
        """
        Compute λ(τ) = exp(-γ · ε(τ)).

        Args:
            symmetry_residual: ε(τ) ≥ 0, shape (...,).

        Returns:
            Confidence weight λ ∈ (0, 1], shape (...,).
        """
        return torch.exp(-self.gamma * symmetry_residual)

    def forward(
        self,
        v: Tensor,
        V: Tensor,
        symmetry_residual: Optional[Tensor] = None,
        lam: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply soft horizontal projection.

        Args:
            v:                  Input vectors, shape (..., d).
            V:                  Generator matrix, shape (..., d, k).
            symmetry_residual:  ε(τ), shape (...,). If None, lam must be given.
            lam:                Override λ directly, shape (...,). If None, computed from ε.

        Returns:
            Soft-projected vectors, shape (..., d).
        """
        if lam is None:
            if symmetry_residual is None:
                # Default: full projection (exact symmetry)
                lam = torch.ones(v.shape[:-1], device=v.device, dtype=v.dtype)
            else:
                lam = self.symmetry_confidence(symmetry_residual)

        # lam: (...,) → (..., 1) for broadcasting
        lam_e = lam.unsqueeze(-1)

        v_hor = self._proj(v, V)               # Π_x v
        v_soft = lam_e * v_hor + (1 - lam_e) * v  # Π_x^soft v
        return v_soft

    def extra_repr(self) -> str:
        return f"eps={self.eps}, gamma={self.gamma.item():.3f}"


# --------------------------------------------------------------------------- #
#  Context-Adaptive Projection                                                 #
# --------------------------------------------------------------------------- #

class ContextAdaptiveProjection(nn.Module):
    """
    Context-adaptive horizontal projection with learnable context encoder φ_ψ.

    Implements Eqs. 5–7 from the paper:
        V_x(τ) = φ_ψ(τ) ⊙ [ξ¹_X(x) | ... | ξᵏ_X(x)]   (Eq. 5)
        Π_x(τ) = I - V_x(τ) (V_x(τ)^T V_x(τ) + ε I)^{-1} V_x(τ)^T   (Eq. 6)
        Π_x^soft(τ) = λ(τ) Π_x(τ) + (1 - λ(τ)) I   (Eq. 7)

    The context encoder φ_ψ: T → R^{d×k} maps context features to a
    weighting of the generator directions, learned end-to-end.

    For robot locomotion: context is terrain height statistics, slopes, normals.
    For molecules: no context (falls back to exact SO(3) projection).

    Args:
        state_dim:    Ambient dimension d.
        context_dim:  Context feature dimension n_τ.
        group_dim:    Lie algebra dimension k.
        hidden_dim:   Hidden units in context encoder (default 64).
        eps:          Gram matrix regularisation.
        gamma_init:   Initial value of decay rate γ.
    """

    def __init__(
        self,
        state_dim: int,
        context_dim: int,
        group_dim: int,
        hidden_dim: int = 64,
        eps: float = 1e-3,
        gamma_init: float = 2.1,
    ):
        super().__init__()
        self.d = state_dim
        self.n_tau = context_dim
        self.k = group_dim
        self.eps = eps

        # Context encoder φ_ψ: T → R^k (scaling of each generator direction)
        # Outputs k weights (one per generator), applied column-wise to V_x.
        # φ_ψ(τ) = 1 recovers exact projection; < 1 relaxes that generator.
        self.context_encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, group_dim),
            nn.Sigmoid(),  # output in (0, 1]^k; scaled to (0, 1] below
        )

        # Symmetry residual estimator: predicts ε(τ) from context
        self.residual_estimator = nn.Sequential(
            nn.Linear(context_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # ε(τ) ≥ 0
        )

        self.log_gamma = nn.Parameter(torch.log(torch.tensor(gamma_init)))
        self._base_proj = HorizontalProjection(eps=eps)

    @property
    def gamma(self) -> Tensor:
        return self.log_gamma.exp()

    def encode_context(self, tau: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Encode context τ into (phi, epsilon).

        Args:
            tau: Context features, shape (..., n_τ).

        Returns:
            phi: Encoder weights ∈ (0, 1]^k, shape (..., k).
            eps_tau: Symmetry residual estimate ε(τ) ≥ 0, shape (..., 1).
        """
        # phi ∈ (0, 1]: at 1 → full generator retained; at 0 → suppressed
        phi = self.context_encoder(tau) + 1e-6  # (..., k)
        eps_tau = self.residual_estimator(tau)  # (..., 1)
        return phi, eps_tau

    def adaptive_generator_matrix(
        self, V_base: Tensor, phi: Tensor
    ) -> Tensor:
        """
        Compute context-adaptive generator matrix V_x(τ) = φ_ψ(τ) ⊙ V_base.

        Args:
            V_base: Base generator matrix, shape (..., d, k).
            phi:    Context weights, shape (..., k).

        Returns:
            Adaptive generator matrix, shape (..., d, k).
        """
        # phi: (..., k) → (..., 1, k) for column-wise scaling
        return V_base * phi.unsqueeze(-2)

    def forward(
        self,
        v: Tensor,
        V_base: Tensor,
        tau: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Apply context-adaptive soft horizontal projection.

        Args:
            v:       Input vectors, shape (..., d).
            V_base:  Base generator matrix, shape (..., d, k).
            tau:     Context features, shape (..., n_τ). If None, uses exact projection.

        Returns:
            Projected vectors, shape (..., d).
        """
        if tau is None:
            # No context → exact horizontal projection
            return self._base_proj(v, V_base)

        phi, eps_tau = self.encode_context(tau)

        # Adaptive generator matrix
        V_adapt = self.adaptive_generator_matrix(V_base, phi)  # (..., d, k)

        # Symmetry confidence weight λ(τ) = exp(-γ · ε(τ))
        lam = torch.exp(-self.gamma * eps_tau).squeeze(-1)  # (...,)

        # Horizontal projection with adaptive V
        v_hor = self._base_proj(v, V_adapt)  # Π_x(τ) v

        # Soft blend
        lam_e = lam.unsqueeze(-1)  # (..., 1)
        return lam_e * v_hor + (1 - lam_e) * v

    def symmetry_stats(self, tau: Tensor) -> dict:
        """Return diagnostic statistics for a context batch."""
        phi, eps_tau = self.encode_context(tau)
        lam = torch.exp(-self.gamma * eps_tau).squeeze(-1)
        return {
            "phi_mean": phi.mean().item(),
            "phi_min": phi.min().item(),
            "eps_tau_mean": eps_tau.mean().item(),
            "lambda_mean": lam.mean().item(),
            "lambda_min": lam.min().item(),
            "gamma": self.gamma.item(),
        }

    def extra_repr(self) -> str:
        return (
            f"state_dim={self.d}, context_dim={self.n_tau}, "
            f"group_dim={self.k}, eps={self.eps}, gamma={self.gamma.item():.3f}"
        )
