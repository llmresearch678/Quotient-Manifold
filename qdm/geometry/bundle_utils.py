"""
Principal bundle utilities for QDM.

Implements helper functions used throughout the framework:
  - compute_vertical_space: span of orbit directions at x
  - compute_horizontal_projection: wrapper with full type handling
  - compute_symmetry_residual: ε(τ) for approximate symmetry (Eq. 4)
  - haar_symmetrize: Haar averaging over compact group orbits
  - fisher_decomposition: I_X = I_Q + I_vert decomposition (Theorem 4)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Callable, List, Optional, Tuple


def compute_vertical_space(
    V: Tensor, eps: float = 1e-3
) -> Tuple[Tensor, Tensor]:
    """
    Compute an orthonormal basis for the vertical space V_x = col(V_x).

    Args:
        V:   Generator matrix, shape (..., d, k).
        eps: Threshold for numerical rank determination.

    Returns:
        Q:   Orthonormal basis of V_x, shape (..., d, k').
        S:   Singular values, shape (..., k).
    """
    U, S, Vh = torch.linalg.svd(V, full_matrices=False)
    # Keep only numerically significant singular vectors
    mask = S > eps * S[..., :1]  # (..., k)
    # Return U (vertical basis) and singular values
    return U, S  # (..., d, k), (..., k)


def compute_horizontal_projection(
    v: Tensor, V: Tensor, eps: float = 1e-3
) -> Tensor:
    """
    Compute Π_x^hor v  =  v  -  V (V^T V + ε I)^{-1} V^T v.

    Public utility wrapping the internal rank-k update formula.

    Args:
        v:   Input vectors, shape (..., d).
        V:   Generator matrix, shape (..., d, k).
        eps: Gram matrix regularisation ε.

    Returns:
        Horizontally projected vectors, shape (..., d).
    """
    from .horizontal_projection import _rank_k_projection
    return _rank_k_projection(v, V, eps=eps)


def compute_symmetry_residual(
    p_x: Callable[[Tensor], Tensor],
    x: Tensor,
    group_samples: List[Tensor],
) -> Tensor:
    """
    Compute symmetry residual ε(τ) = sup_{g ∈ G} ||p_X(x) - p_X(g·x)||_{L2(X)}.

    In practice, approximated via Monte Carlo over group_samples.

    Args:
        p_x:           Callable mapping x → log p_X(x), shape (B,).
        x:             Reference states, shape (B, d).
        group_samples: List of transformed states g·x, each shape (B, d).

    Returns:
        Symmetry residual estimate, shape (B,) ≥ 0.
    """
    log_p_x = p_x(x)  # (B,)
    max_diff = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

    for gx in group_samples:
        log_p_gx = p_x(gx)  # (B,)
        diff = (log_p_x - log_p_gx).abs()
        max_diff = torch.maximum(max_diff, diff)

    return max_diff  # (B,)


def haar_symmetrize_so3(
    x: Tensor, n_atoms: int, n_samples: int = 64
) -> Tensor:
    """
    Haar-average x over SO(3): x̄ = ∫_{SO(3)} R·x dν_{SO(3)}(R).

    Used in Theorem 1 (reverse direction) to construct the unique G-invariant
    ambient lift from a quotient measure.

    Args:
        x:        Atom coordinates, shape (B, 3N).
        n_atoms:  Number of atoms N.
        n_samples: Monte Carlo samples.

    Returns:
        Symmetrized state, shape (B, 3N).
    """
    B, d = x.shape
    assert d == 3 * n_atoms, f"Expected 3N={3*n_atoms} dims, got {d}"

    coords = x.reshape(B, n_atoms, 3)  # (B, N, 3)
    avg = torch.zeros_like(coords)

    for _ in range(n_samples):
        Z = torch.randn(B, 3, 3, device=x.device, dtype=x.dtype)
        Q, _ = torch.linalg.qr(Z)
        # Ensure det(Q) = +1
        det_sign = torch.sign(torch.linalg.det(Q))  # (B,)
        Q = Q * det_sign[:, None, None]
        rotated = torch.einsum("bij,bnj->bni", Q, coords)  # (B, N, 3)
        avg = avg + rotated

    return (avg / n_samples).reshape(B, d)


def haar_symmetrize_so2(
    x: Tensor, heading_indices: List[int], n_samples: int = 64
) -> Tensor:
    """
    Haar-average x over SO(2) heading rotations.

    For robot locomotion: uniform average over heading angles θ ∈ [0, 2π).

    Args:
        x:               Robot state, shape (B, d).
        heading_indices: Pairs of (x, y) coordinate indices affected by heading.
        n_samples:       Monte Carlo samples.

    Returns:
        Symmetrized state, shape (B, d).
    """
    B, d = x.shape
    avg = torch.zeros_like(x)
    angles = torch.linspace(0, 2 * torch.pi, n_samples + 1)[:-1]

    for theta in angles:
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)
        x_rot = x.clone()
        for i in range(0, len(heading_indices) - 1, 2):
            ix = heading_indices[i]
            iy = heading_indices[i + 1]
            x_rot[:, ix] = cos_t * x[:, ix] - sin_t * x[:, iy]
            x_rot[:, iy] = sin_t * x[:, ix] + cos_t * x[:, iy]
        avg = avg + x_rot

    return avg / n_samples


def fisher_decomposition(
    scores: Tensor,
    x: Tensor,
    V_fn: Callable[[Tensor], Tensor],
    eps: float = 1e-3,
) -> dict:
    """
    Compute the Fisher information decomposition from Theorem 4:
        I_X = I_Q (horizontal) + I_vert (vertical)

    where:
        I_Q    = E[||Π^hor s||^2]  (quotient-relevant)
        I_vert = E[||Π^vert s||^2] (symmetry-orbit, removed by QDM)

    Args:
        scores: Score vectors, shape (B, d).
        x:      Corresponding states, shape (B, d).
        V_fn:   Callable x → V_x, shape (B, d, k).
        eps:    Gram matrix regularisation.

    Returns:
        Dict with keys: I_X, I_Q, I_vert, vert_fraction, k_over_d.
    """
    V = V_fn(x)  # (B, d, k)
    k = V.shape[-1]
    d = V.shape[-2]

    s_hor = compute_horizontal_projection(scores, V, eps=eps)  # (B, d)
    s_vert = scores - s_hor                                     # (B, d)

    I_X = (scores ** 2).sum(-1).mean().item()
    I_Q = (s_hor ** 2).sum(-1).mean().item()
    I_vert = (s_vert ** 2).sum(-1).mean().item()
    vert_fraction = I_vert / max(I_X, 1e-8)

    return {
        "I_X": I_X,
        "I_Q": I_Q,
        "I_vert": I_vert,
        "vert_fraction": vert_fraction,
        "k_over_d": k / d,
        "theory_prediction": k / d,
        "gap": abs(vert_fraction - k / d),
    }
