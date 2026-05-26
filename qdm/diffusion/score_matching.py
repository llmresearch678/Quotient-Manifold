"""
Context-Conditioned Denoising Score Matching for QDM.

Implements the training objective from Eq. 11 of the paper:

    L(θ) = E_{t, X_t, τ, c} [
        || Π^soft(τ,c) · f_θ(X_t, t, τ, c)
           - (dπ_{X_t}^c)† · s_t^c(π^c(X_t)) ||²_g
    ]

Key properties:
  - In exact-symmetry regimes (λ=1): vertical orbit directions receive
    zero gradient signal (score network is not penalised for ignoring them).
  - As symmetry weakens (λ<1): vertical components are gradually reintroduced.
  - Time weighting w(t) = 1/(1-t)² up-weights near-data times.

This objective recovers the intrinsic quotient score s_t^c through
the horizontal lift, as shown in Section III.D.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Dict, Optional, Tuple

from .sde import VP_SDE, QDMForwardProcess
from ..geometry.horizontal_projection import (
    SoftHorizontalProjection,
    ContextAdaptiveProjection,
)


class QDMScoreMatchingLoss(nn.Module):
    """
    Context-conditioned denoising score matching loss for QDM.

    The target for the score network is the noise ε (noise prediction
    parametrisation), consistent with the DSM identity.
    The horizontal projection is applied to both the prediction and the
    target before computing the MSE, confining gradient signal to H_x.

    Args:
        sde:          VP-SDE schedule.
        soft_proj:    SoftHorizontalProjection (or ContextAdaptiveProjection).
        time_weighting: If True, weight loss by w(t) = 1/(1-t)².
        lambda_vert:  Additional weight for vertical loss penalty (default 0,
                      set > 0 to explicitly penalise vertical score components).
    """

    def __init__(
        self,
        sde: VP_SDE,
        soft_proj: nn.Module,
        time_weighting: bool = True,
        lambda_vert: float = 0.0,
    ):
        super().__init__()
        self.sde = sde
        self.soft_proj = soft_proj
        self.time_weighting = time_weighting
        self.lambda_vert = lambda_vert
        self.forward_process = QDMForwardProcess(sde, soft_proj)

    def forward(
        self,
        score_model: nn.Module,
        x0: Tensor,
        V_fn: Callable[[Tensor], Tensor],
        tau: Optional[Tensor] = None,
        mode: str = "default",
        t: Optional[Tensor] = None,
    ) -> Tuple[Tensor, dict]:
        """
        Compute the QDM score matching loss on a batch.

        Args:
            score_model: f_θ(x, t, tau, mode) → score ∈ R^d.
            x0:          Clean data, shape (B, d).
            V_fn:        Callable x → V_x, shape (B, d, k).
            tau:         Context features, shape (B, n_τ). Optional.
            mode:        Active symmetry mode string.
            t:           Diffusion times, shape (B,). Sampled if None.

        Returns:
            loss:   Scalar loss tensor.
            metrics: Dict with diagnostic values.
        """
        B, d = x0.shape
        device = x0.device

        # Sample diffusion time
        if t is None:
            t = torch.rand(B, device=device) * (self.sde.T - 1e-3) + 1e-3

        # Compute V at x0 (used for forward process projection)
        V0 = V_fn(x0)  # (B, d, k)

        # Forward process: add horizontal-projected noise
        noise = torch.randn_like(x0)
        xt, noise_proj = self.forward_process(x0, t, V0, tau=tau, noise=noise)

        # Score model prediction
        score_pred = score_model(xt, t, tau=tau, mode=mode)  # (B, d)

        # Project prediction onto horizontal subspace at xt
        Vt = V_fn(xt.detach())  # (B, d, k) at noisy state
        if isinstance(self.soft_proj, ContextAdaptiveProjection):
            score_pred_proj = self.soft_proj(score_pred, Vt, tau=tau)
        elif tau is not None:
            lam = getattr(self.soft_proj, "symmetry_confidence",
                          lambda e: torch.ones(B, device=device))(
                torch.zeros(B, device=device)
            )
            score_pred_proj = self.soft_proj(score_pred, Vt, lam=lam)
        else:
            lam = torch.ones(B, device=device)
            score_pred_proj = self.soft_proj(score_pred, Vt, lam=lam)

        # Target: noise prediction (DSM identity; true score ∝ -noise/σ)
        # The noise was already projected in the forward process; we match
        # the projected noise in the horizontal subspace.
        _, std = self.sde.marginal_prob(x0, t)  # (B, 1)
        target = -noise_proj / std  # horizontal score target

        # Also project the target for consistency
        if isinstance(self.soft_proj, ContextAdaptiveProjection):
            target_proj = self.soft_proj(target, Vt, tau=tau)
        else:
            target_proj = self.soft_proj(target, Vt, lam=lam)

        # MSE loss in horizontal subspace
        diff = score_pred_proj - target_proj  # (B, d)
        loss_per_sample = (diff ** 2).sum(-1)  # (B,)

        # Time weighting w(t) = 1/(1-t)²
        if self.time_weighting:
            w = self.sde.time_weight(t)  # (B,)
            loss_per_sample = w * loss_per_sample

        loss = loss_per_sample.mean()

        # Optional: penalise vertical score components
        if self.lambda_vert > 0:
            s_vert = score_pred - score_pred_proj  # (B, d)
            vert_loss = (s_vert ** 2).sum(-1).mean()
            loss = loss + self.lambda_vert * vert_loss
        else:
            vert_loss = torch.tensor(0.0, device=device)

        # Diagnostics
        with torch.no_grad():
            s_vert_mag = (score_pred - score_pred_proj).norm(dim=-1).mean()
            s_total_mag = score_pred.norm(dim=-1).mean()
            vert_frac = (s_vert_mag / s_total_mag.clamp(min=1e-8)).item()

        metrics = {
            "loss": loss.item(),
            "vert_loss": vert_loss.item(),
            "vert_fraction": vert_frac,
            "t_mean": t.mean().item(),
            "noise_proj_norm": noise_proj.norm(dim=-1).mean().item(),
        }

        return loss, metrics


class AmbientScoreMatchingLoss(nn.Module):
    """
    Standard ambient denoising score matching (baseline, no projection).

    Used for ablation: QDM-ambient in Table VI of the paper.
    """

    def __init__(self, sde: VP_SDE, time_weighting: bool = True):
        super().__init__()
        self.sde = sde
        self.time_weighting = time_weighting

    def forward(
        self,
        score_model: nn.Module,
        x0: Tensor,
        t: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, dict]:
        B, d = x0.shape
        device = x0.device

        if t is None:
            t = torch.rand(B, device=device) * (self.sde.T - 1e-3) + 1e-3

        mean, std = self.sde.marginal_prob(x0, t)
        noise = torch.randn_like(x0)
        xt = mean + std * noise

        score_pred = score_model(xt, t, **kwargs)
        target = -noise / std

        loss_per_sample = ((score_pred - target) ** 2).sum(-1)

        if self.time_weighting:
            w = self.sde.time_weight(t)
            loss_per_sample = w * loss_per_sample

        loss = loss_per_sample.mean()
        return loss, {"loss": loss.item()}
