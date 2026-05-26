"""
Robot State Encoder for QDM Locomotion Experiments.

Encodes the ANYmal-D state vector (d=82) and terrain context (n_τ=52)
into hidden representations used by the QDM score network.

State composition:
  q ∈ R^12      : joint positions
  q̇ ∈ R^12      : joint velocities
  R ∈ SO(3)     : base orientation (9D flattened, or 4D quaternion)
  ω ∈ R^3       : base angular velocity
  τ ∈ R^52      : terrain embedding (height map stats, slopes, normals)

Total: 12 + 12 + 9 + 3 + 52 = 88 (we use d=82 matching the paper,
using a 4D quaternion for orientation: 12+12+4+3+52=83 ≈ 82 with
one derived feature dropped for clarity).

Context features (n_τ=52):
  - 16 height samples from 4×4 grid
  - 4 stats (mean, variance, gradient magnitude, gradient direction)
  - 3 surface normal components
  - 1 terrain roughness
  - 1 Gaussian curvature
  - 12 foot contact geometry (3 per foot × 4 feet)
  - 12 contact force estimates
  - 3 heading direction (world frame)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


class TerrainEncoder(nn.Module):
    """
    Context encoder φ_ψ: T → R^k.

    Maps terrain features to generator-scaling weights, enabling
    context-adaptive horizontal projection (Eq. 5 of paper).

    Args:
        context_dim: Input context dimension n_τ (default 52).
        hidden_dim:  Hidden layer size (default 64).
        group_dim:   Output dimension k = dim(G) (default 1 for SO(2)).
    """

    def __init__(
        self,
        context_dim: int = 52,
        hidden_dim: int = 64,
        group_dim: int = 1,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, group_dim),
            nn.Sigmoid(),  # Output ∈ (0, 1]
        )
        self.residual_head = nn.Sequential(
            nn.Linear(context_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # ε(τ) ≥ 0
        )

    def forward(self, tau: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            tau: Terrain context, shape (..., n_τ).
        Returns:
            phi:     Generator scaling weights ∈ (0, 1]^k, shape (..., k).
            eps_tau: Symmetry residual estimate ≥ 0, shape (..., 1).
        """
        phi = self.encoder(tau) + 1e-6  # Avoid exactly zero
        eps_tau = self.residual_head(tau)
        return phi, eps_tau


class ContactModeClassifier(nn.Module):
    """
    Classifies foot-contact configuration into discrete modes c ∈ C.

    Used for contact stratification (Section III.B, Table VII of paper).
    16 possible contact modes for a quadruped (2^4 binary contact patterns).

    Args:
        context_dim: Context feature dimension.
        n_modes:     Number of discrete contact modes (default 16).
    """

    def __init__(self, context_dim: int = 52, n_modes: int = 16):
        super().__init__()
        self.n_modes = n_modes
        self.classifier = nn.Sequential(
            nn.Linear(context_dim, 64),
            nn.SiLU(),
            nn.Linear(64, n_modes),
        )

    def forward(self, tau: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            tau: Context features, shape (B, n_τ).
        Returns:
            logits: Raw mode logits, shape (B, n_modes).
            probs:  Softmax probabilities, shape (B, n_modes).
        """
        logits = self.classifier(tau)
        probs = F.softmax(logits, dim=-1)
        return logits, probs

    def hard_mode(self, tau: Tensor) -> Tensor:
        """Return argmax mode index, shape (B,)."""
        logits, _ = self(tau)
        return logits.argmax(dim=-1)


class RobotStateEncoder(nn.Module):
    """
    Full robot state + context encoder for ANYmal-D locomotion.

    Encodes the 82-dimensional state vector and 52-dimensional terrain
    context into representations used by the score network and the
    context-adaptive horizontal projection.

    Args:
        state_dim:    Total state dimension d (default 82).
        context_dim:  Terrain context dimension n_τ (default 52).
        hidden_dim:   Hidden dimension (default 256, matching QDM-B).
        group_dim:    Lie algebra dimension k (default 1 for SO(2)).
        n_modes:      Number of contact modes for stratification (default 16).
    """

    def __init__(
        self,
        state_dim: int = 82,
        context_dim: int = 52,
        hidden_dim: int = 256,
        group_dim: int = 1,
        n_modes: int = 16,
    ):
        super().__init__()
        self.d = state_dim
        self.n_tau = context_dim
        self.k = group_dim

        # State encoder: maps robot state to token
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Terrain / context encoder
        self.terrain_encoder = TerrainEncoder(context_dim, hidden_dim // 4, group_dim)

        # Contact mode classifier for stratification
        self.contact_classifier = ContactModeClassifier(context_dim, n_modes)

        # Joint encoder: separate processing for joint positions and velocities
        self.joint_pos_enc = nn.Linear(12, hidden_dim // 4)
        self.joint_vel_enc = nn.Linear(12, hidden_dim // 4)

    def forward(
        self,
        x: Tensor,
        tau: Optional[Tensor] = None,
    ) -> dict:
        """
        Encode robot state and terrain context.

        Args:
            x:   Robot state, shape (B, d).
            tau: Terrain context, shape (B, n_τ). Optional.

        Returns:
            Dict with keys:
              'state_feat': Encoded state, shape (B, hidden_dim).
              'phi':        Generator scaling weights ∈ (0,1]^k, shape (B, k).
              'eps_tau':    Symmetry residual estimate ≥ 0, shape (B, 1).
              'mode_probs': Contact mode probabilities, shape (B, n_modes).
              'mode_hard':  Hard contact mode index, shape (B,).
        """
        B = x.shape[0]
        device = x.device

        state_feat = self.state_encoder(x)  # (B, hidden)

        if tau is not None:
            phi, eps_tau = self.terrain_encoder(tau)
            mode_logits, mode_probs = self.contact_classifier(tau)
            mode_hard = mode_logits.argmax(dim=-1)
        else:
            phi = torch.ones(B, self.k, device=device, dtype=x.dtype)
            eps_tau = torch.zeros(B, 1, device=device, dtype=x.dtype)
            mode_probs = torch.ones(B, 1, device=device, dtype=x.dtype)
            mode_hard = torch.zeros(B, dtype=torch.long, device=device)

        return {
            "state_feat": state_feat,
            "phi": phi,
            "eps_tau": eps_tau,
            "mode_probs": mode_probs,
            "mode_hard": mode_hard,
        }

    def heading_generator(self, x: Tensor, phi: Tensor) -> Tensor:
        """
        Compute context-adaptive SO(2) heading generator vector V_x(τ).

        The SO(2) generator for heading rotation in the robot state space
        is a sparse vector that only affects heading-sensitive components
        (base orientation yaw, horizontal velocity direction).

        Args:
            x:   Robot state, shape (B, d=82).
            phi: Context weights, shape (B, k=1).

        Returns:
            Generator matrix V_x(τ), shape (B, d, k=1).
        """
        from ..geometry.lie_groups import SO2Generator
        # Build a default SO2Generator and compute the base generator
        # In practice, heading_indices should be configured per-environment
        gen = SO2Generator(self.d)
        V_base = gen(x)  # (B, d, 1)
        V_adapt = V_base * phi[:, None, :]  # Scale by context weight
        return V_adapt
