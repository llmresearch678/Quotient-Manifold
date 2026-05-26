"""
Quotient Manifold Constructions for QDM.

Implements:
  - QuotientManifold: single-stratum quotient M = X / G
  - StratifiedQuotientManifold: multi-mode quotient Q̄ = ⊔_c Q(c)
    with inter-stratum transition maps Ψ_{c→c'} and Haar symmetrization.

Theorem 1 (Invariant–Quotient Correspondence) is realized by the
push-forward / lift pair (π_#, lift_haar).

Theorem 2 (Probability Consistency Across Symmetry Transitions) is
implemented by `transition` + `haar_symmetrize`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Optional, Tuple, Callable

from .lie_groups import LieGroupAction
from .horizontal_projection import HorizontalProjection


class QuotientManifold:
    """
    Single-stratum quotient manifold Q = X / G.

    Provides:
      - quotient_project: π(x) → equivalence class representative
      - haar_symmetrize: average a measure over the orbit (Theorem 1)
      - fisher_decomposition: decompose Fisher information (Theorem 4)

    Args:
        action: A LieGroupAction specifying how G acts on X.
        eps: Gram matrix regularisation for horizontal projection.
    """

    def __init__(self, action: LieGroupAction, eps: float = 1e-3):
        self.action = action
        self.d = action.d
        self.k = action.k
        self.proj = HorizontalProjection(eps=eps)

    def generator_matrix(self, x: Tensor) -> Tensor:
        """V_x ∈ R^{d × k}: infinitesimal generators at x."""
        return self.action(x)

    def horizontal_project(self, score: Tensor, x: Tensor) -> Tensor:
        """Project score onto horizontal subspace H_x."""
        V = self.generator_matrix(x)
        return self.proj(score, V)

    def vertical_fisher_fraction(self, score: Tensor, x: Tensor) -> Tensor:
        """
        Empirical estimate of I_vert / I_X = fraction of Fisher information
        in symmetry-orbit directions.  Should be ~k/d for ambient, 0 for QDM.

        Args:
            score: Batch of score vectors, shape (B, d).
            x:     Corresponding states, shape (B, d).

        Returns:
            Scalar tensor: mean ||s_vert||^2 / ||s||^2.
        """
        V = self.generator_matrix(x)
        s_vert = score - self.proj(score, V)
        frac = s_vert.norm(dim=-1) ** 2 / score.norm(dim=-1).clamp(min=1e-8) ** 2
        return frac.mean()

    def theoretical_fisher_fraction(self) -> float:
        """Theoretical prediction k/d from Theorem 4."""
        return self.k / self.d

    def haar_symmetrize(
        self,
        x: Tensor,
        n_samples: int = 100,
    ) -> Tensor:
        """
        Approximate Haar averaging over the orbit G · x.

        For SO(3): sample random rotations, rotate x, average.
        Implements the symmetrization step from Theorem 1 (reverse direction).

        Args:
            x:         States, shape (..., d).
            n_samples: Monte Carlo samples for the Haar integral.

        Returns:
            Symmetrized representation (center of orbit), shape (..., d).
        """
        # Default implementation: return x unchanged (subclasses override)
        # Subclasses for SO(3)/SO(2) implement proper Haar sampling.
        return x


class SO3QuotientManifold(QuotientManifold):
    """
    Quotient manifold X / SO(3) for molecular conformer generation.

    Implements proper Haar averaging over SO(3) using random rotations.
    Theorem 1: μ ↔ π_# μ bijection between SO(3)-invariant measures on X
    and all measures on Q = X / SO(3).
    """

    def __init__(self, n_atoms: int, eps: float = 1e-3):
        from .lie_groups import SO3Generator
        action = SO3Generator(n_atoms, center=True)
        super().__init__(action, eps=eps)
        self.N = n_atoms

    def haar_symmetrize(self, x: Tensor, n_samples: int = 100) -> Tensor:
        """
        Average x over n_samples random SO(3) rotations (Haar measure).

        For SO(3), Haar measure = uniform measure over rotation matrices,
        sampled via QR decomposition of random Gaussian matrices.
        """
        *batch, d = x.shape
        assert d == 3 * self.N

        coords = x.reshape(*batch, self.N, 3)  # (..., N, 3)
        avg = torch.zeros_like(coords)

        for _ in range(n_samples):
            # Sample random rotation via QR of Gaussian matrix
            Z = torch.randn(3, 3, device=x.device, dtype=x.dtype)
            Q, R = torch.linalg.qr(Z)
            # Ensure proper rotation (det = +1)
            Q = Q * torch.sign(torch.linalg.det(Q)).unsqueeze(-1).unsqueeze(-1)
            rotated = coords @ Q.T  # (..., N, 3)
            avg = avg + rotated

        avg = avg / n_samples
        return avg.reshape(*batch, d)

    def canonicalize(self, x: Tensor) -> Tensor:
        """
        Return a canonical representative of the equivalence class [x].

        Uses PCA to align the principal axes, giving a deterministic
        quotient-space representative.

        Args:
            x: Atom coordinates, shape (B, 3N).

        Returns:
            Canonical representative, shape (B, 3N).
        """
        B, d = x.shape
        coords = x.reshape(B, self.N, 3)

        # Center
        center = coords.mean(dim=1, keepdim=True)
        coords_c = coords - center

        # PCA via SVD
        U, S, Vh = torch.linalg.svd(
            coords_c.reshape(B, -1, 3).transpose(-2, -1), full_matrices=False
        )
        # Align: rotate so that principal axes align with standard basis
        canonical = (U.transpose(-2, -1) @ coords_c.transpose(-2, -1)).transpose(-2, -1)
        return canonical.reshape(B, d)


class StratifiedQuotientManifold(nn.Module):
    """
    Stratified quotient space Q̄ = ⊔_{c ∈ C} Q(c).

    Each mode c selects an active symmetry group G(c) and quotient manifold
    Q(c) = X(c) / G(c). Inter-stratum transition maps Ψ_{c→c'} transport
    probability mass consistently (Theorem 2).

    Used for legged robot locomotion with contact-configuration stratification.

    Args:
        strata:       Dict mapping mode label c → QuotientManifold Q(c).
        transition_maps: Dict mapping (c, c') → Callable[Tensor, Tensor].
                         If None, identity transitions are used.
    """

    def __init__(
        self,
        strata: Dict[str, QuotientManifold],
        transition_maps: Optional[Dict[Tuple[str, str], Callable]] = None,
    ):
        super().__init__()
        self.strata = strata
        self.modes = list(strata.keys())
        self.transition_maps = transition_maps or {}

    def get_stratum(self, mode: str) -> QuotientManifold:
        return self.strata[mode]

    def transition(self, y: Tensor, from_mode: str, to_mode: str) -> Tensor:
        """
        Apply inter-stratum transition map Ψ_{c→c'}(y).

        Implements Theorem 2, Part 1: pushforward of the probability measure
        to the target stratum.

        Args:
            y:         State on source stratum Q(from_mode), shape (..., d).
            from_mode: Source mode label c.
            to_mode:   Target mode label c'.

        Returns:
            Transported state on Q(to_mode), shape (..., d).
        """
        key = (from_mode, to_mode)
        if key in self.transition_maps:
            return self.transition_maps[key](y)
        # Default: identity transition (mode change doesn't alter state coords)
        return y

    def horizontal_project(
        self,
        score: Tensor,
        x: Tensor,
        mode: str,
        tau: Optional[Tensor] = None,
    ) -> Tensor:
        """Project score in the active quotient stratum Q(mode)."""
        return self.strata[mode].horizontal_project(score, x)

    def active_quotient_dim(self, mode: str) -> int:
        """Quotient manifold dimension d - k for mode c."""
        q = self.strata[mode]
        return q.d - q.k

    def mode_fisher_fractions(
        self, scores: Dict[str, Tensor], states: Dict[str, Tensor]
    ) -> Dict[str, float]:
        """
        Compute vertical Fisher fraction I_vert/I_X per stratum.
        Useful diagnostic for verifying Theorem 4 empirically.
        """
        out = {}
        for mode in self.modes:
            if mode in scores:
                frac = self.strata[mode].vertical_fisher_fraction(
                    scores[mode], states[mode]
                )
                out[mode] = frac.item()
        return out
