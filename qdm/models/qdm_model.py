"""
QDM: Main Model Wrapper.

Integrates all QDM components:
  - Score network (QDMScoreNet)
  - Geometry (generator matrix, horizontal projection)
  - Context-adaptive soft projection
  - Forward and reverse diffusion processes

Provides a clean API for training, sampling, and evaluation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Dict, Optional, Tuple

from .score_net import QDMScoreNet, build_score_net
from ..geometry.horizontal_projection import (
    SoftHorizontalProjection,
    ContextAdaptiveProjection,
)
from ..geometry.lie_groups import SO3Generator, SO2Generator
from ..diffusion.sde import VP_SDE, QDMForwardProcess, QDMReverseProcess
from ..diffusion.score_matching import QDMScoreMatchingLoss


class QDM(nn.Module):
    """
    Quotient Diffusion Model.

    End-to-end model for symmetry-aware generative learning on quotient manifolds.

    Supports:
      - Molecular conformer generation (SO(3) / SE(3) quotient)
      - Legged robot locomotion (SO(2) quotient, contact-stratified)

    Args:
        task:         "molecular" or "robot".
        variant:      Score network size: "QDM-S", "QDM-B", "QDM-L".
        n_atoms:      Number of atoms (molecular) or state chunks (robot).
        context_dim:  Context / terrain feature dimension (0 for molecular).
        n_modes:      Number of symmetry modes (1 for single-stratum).
        eps_proj:     Gram matrix regularisation ε for horizontal projection.
        gamma_init:   Initial symmetry confidence decay rate γ.
        beta_min:     VP-SDE β_min.
        beta_max:     VP-SDE β_max.
        n_steps:      Denoising steps for sampling.
        method:       Reverse SDE solver ("em" or "heun").
    """

    def __init__(
        self,
        task: str = "molecular",
        variant: str = "QDM-B",
        n_atoms: int = 9,
        context_dim: int = 0,
        n_modes: int = 1,
        eps_proj: float = 1e-3,
        gamma_init: float = 2.1,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        n_steps: int = 100,
        method: str = "em",
    ):
        super().__init__()
        self.task = task
        self.variant = variant
        self.n_atoms = n_atoms
        self.d = 3 * n_atoms if task == "molecular" else n_atoms

        # Score network
        self.score_net = build_score_net(
            variant,
            n_atoms=n_atoms,
            context_dim=context_dim,
            n_modes=n_modes,
        )

        # Lie group generator
        if task == "molecular":
            self.generator = SO3Generator(n_atoms, center=True)
            self.k = 3
        else:
            self.generator = SO2Generator(n_atoms)
            self.k = 1

        # Context-adaptive soft horizontal projection
        if context_dim > 0:
            self.proj = ContextAdaptiveProjection(
                state_dim=self.d,
                context_dim=context_dim,
                group_dim=self.k,
                eps=eps_proj,
                gamma_init=gamma_init,
            )
        else:
            self.proj = SoftHorizontalProjection(eps=eps_proj, gamma=gamma_init)

        # Diffusion schedule
        self.sde = VP_SDE(beta_min=beta_min, beta_max=beta_max)

        # Forward and reverse processes
        soft_proj_for_forward = SoftHorizontalProjection(eps=eps_proj, gamma=gamma_init)
        self.forward_process = QDMForwardProcess(self.sde, soft_proj_for_forward)
        self.reverse_process = QDMReverseProcess(
            self.sde, self.score_net, soft_proj_for_forward, n_steps=n_steps, method=method
        )

        # Training objective
        self.loss_fn = QDMScoreMatchingLoss(self.sde, self.proj)

    def generator_matrix(self, x: Tensor) -> Tensor:
        """Compute V_x ∈ R^{d × k} at state x."""
        return self.generator(x)

    def score(
        self,
        x: Tensor,
        t: Tensor,
        tau: Optional[Tensor] = None,
        mode: str = "default",
        **kwargs,
    ) -> Tensor:
        """
        Compute the horizontally projected score f_θ(x, t, τ, c).

        The score network output is projected onto H_x before returning,
        ensuring zero vertical component in exact-symmetry regimes.

        Args:
            x:    State, shape (B, d).
            t:    Time, shape (B,).
            tau:  Context features, shape (B, n_τ). Optional.
            mode: Symmetry mode string.

        Returns:
            Projected score, shape (B, d).
        """
        V = self.generator_matrix(x)  # (B, d, k)
        raw_score = self.score_net(x, t, tau=tau, mode=mode, **kwargs)  # (B, d)

        if isinstance(self.proj, ContextAdaptiveProjection) and tau is not None:
            score_proj = self.proj(raw_score, V, tau=tau)
        else:
            lam = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
            score_proj = self.proj(raw_score, V, lam=lam)

        return score_proj

    def training_loss(
        self,
        x0: Tensor,
        tau: Optional[Tensor] = None,
        mode: str = "default",
        t: Optional[Tensor] = None,
    ) -> Tuple[Tensor, dict]:
        """
        Compute QDM training loss on a batch of clean data.

        Args:
            x0:   Clean data, shape (B, d).
            tau:  Context features, shape (B, n_τ). Optional.
            mode: Active symmetry mode.
            t:    Diffusion times. Sampled if None.

        Returns:
            loss:    Scalar training loss.
            metrics: Dict of diagnostic values.
        """
        V_fn = self.generator_matrix
        return self.loss_fn(self.score_net, x0, V_fn=V_fn, tau=tau, mode=mode, t=t)

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        tau: Optional[Tensor] = None,
        mode: str = "default",
        return_trajectory: bool = False,
    ) -> Tensor:
        """
        Generate samples from the QDM.

        Args:
            batch_size:        Number of samples.
            tau:               Context features, shape (B, n_τ). Optional.
            mode:              Symmetry mode.
            return_trajectory: Return full denoising trajectory.

        Returns:
            Samples, shape (batch_size, d).
        """
        device = next(self.parameters()).device
        V_fn = self.generator_matrix

        return self.reverse_process.sample(
            shape=(batch_size, self.d),
            V_fn=V_fn,
            device=device,
            tau=tau,
            mode=mode,
            return_trajectory=return_trajectory,
        )

    def vertical_fisher_fraction(self, score: Tensor, x: Tensor) -> float:
        """
        Compute I_vert / I_X: fraction of Fisher info in vertical directions.
        Should be ~0 for QDM, ~k/d for ambient (Theorem 4 diagnostic).
        """
        from ..geometry.bundle_utils import fisher_decomposition
        decomp = fisher_decomposition(score, x, self.generator_matrix)
        return decomp["vert_fraction"]

    def n_parameters(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"QDM(task={self.task!r}, variant={self.variant!r}, "
            f"d={self.d}, k={self.k}, "
            f"params={self.n_parameters():,})"
        )
