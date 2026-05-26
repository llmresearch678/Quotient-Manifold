"""
QDM Stochastic Differential Equations.

Implements the lifted quotient diffusion SDE (Eq. 8, 9 of the paper):

  Forward (Stratonovich):
    dX_t = σ · Π_x^soft(τ_t, c_t) ∘ dB_t

  Reverse:
    dX_t = b̃_t(X_t, τ_t, c_t) dt + σ · Π_x^soft(τ_t, c_t) ∘ dB̄_t

where the lifted reverse drift b̃_t is given by Eq. 9:
    b̃_t(x, τ, c) = (dπ_x^c)† [-σ² s_t^c(π^c(x))]

Theorem 3 (Well-Posedness) is satisfied under standard Lipschitz assumptions.

The module provides:
  - VP_SDE: Variance-Preserving SDE schedule (default for all experiments)
  - QDMForwardProcess: adds horizontal-projected noise
  - QDMReverseProcess: runs denoising with a given score model
  - StratifiedQDMProcess: handles mode transitions in Q̄
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable, Dict, List, Optional, Tuple

from ..geometry.horizontal_projection import SoftHorizontalProjection


# --------------------------------------------------------------------------- #
#  VP-SDE Noise Schedule                                                       #
# --------------------------------------------------------------------------- #

class VP_SDE:
    """
    Variance-Preserving SDE schedule from Song et al. (2021).

        β(t) = β_min + t (β_max - β_min)
        α(t) = exp(-∫_0^t β(s) ds) = exp(-t β_min - t²/2 (β_max - β_min))
        σ(t) = sqrt(1 - α(t)²)

    Args:
        beta_min: Minimum noise level (default 0.1).
        beta_max: Maximum noise level (default 20.0).
        T: Terminal time (default 1.0).
    """

    def __init__(
        self,
        beta_min: float = 0.1,
        beta_max: float = 20.0,
        T: float = 1.0,
    ):
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.T = T

    def beta(self, t: Tensor) -> Tensor:
        """β(t): noise schedule rate."""
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def marginal_prob(self, x0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Mean and std of q(X_t | X_0) = N(α(t) x0, σ(t)² I).

        Args:
            x0: Clean data, shape (..., d).
            t:  Time, shape (...,).

        Returns:
            mean: α(t) x0, shape (..., d).
            std:  σ(t), shape (..., 1) for broadcasting.
        """
        # ∫_0^t β(s) ds = t β_min + t²/2 (β_max - β_min)
        log_alpha = -0.5 * t * (self.beta_min + 0.5 * t * (self.beta_max - self.beta_min))
        alpha = log_alpha.exp()  # (...,)
        sigma = (1 - alpha ** 2).clamp(min=1e-5).sqrt()

        alpha_e = alpha[..., None]  # (..., 1)
        sigma_e = sigma[..., None]  # (..., 1)
        mean = alpha_e * x0
        return mean, sigma_e

    def diffusion_coeff(self, t: Tensor) -> Tensor:
        """σ(t) diffusion coefficient for the forward SDE."""
        log_alpha = -0.5 * t * (self.beta_min + 0.5 * t * (self.beta_max - self.beta_min))
        alpha = log_alpha.exp()
        return (1 - alpha ** 2).clamp(min=1e-5).sqrt()

    def prior_sample(self, shape: tuple, device, dtype) -> Tensor:
        """Sample X_T ~ N(0, I) as the prior."""
        return torch.randn(*shape, device=device, dtype=dtype)

    def time_weight(self, t: Tensor) -> Tensor:
        """
        Time-weighting w(t) = 1/(1-t)² used in the training objective.
        Up-weights near-data times where vertical score variance is most harmful.
        """
        return 1.0 / (1.0 - t).clamp(min=1e-3) ** 2


# --------------------------------------------------------------------------- #
#  QDM Forward Process                                                         #
# --------------------------------------------------------------------------- #

class QDMForwardProcess(nn.Module):
    """
    Quotient-projected forward diffusion process.

    Instead of adding isotropic Brownian noise in R^d, adds noise only
    in the horizontal subspace H_x at each step, eliminating symmetry-orbit
    noise from the forward process.

    In the Stratonovich SDE (Eq. 8):
        dX_t = σ · Π_x^soft(τ, c) ∘ dB_t

    The noise injection is Π^soft dB_t, a rank-(d-k) projected Brownian motion.

    Args:
        sde:            VP-SDE schedule.
        soft_proj:      SoftHorizontalProjection module.
    """

    def __init__(self, sde: VP_SDE, soft_proj: SoftHorizontalProjection):
        super().__init__()
        self.sde = sde
        self.soft_proj = soft_proj

    def forward(
        self,
        x0: Tensor,
        t: Tensor,
        V: Tensor,
        tau: Optional[Tensor] = None,
        noise: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Sample X_t from the quotient forward process.

        In the marginal distribution, X_t = α(t) x0 + σ(t) Π^soft ε,
        where ε ~ N(0, I_d) is standard Brownian noise.

        Args:
            x0:    Clean data, shape (B, d).
            t:     Diffusion time ∈ [0, T], shape (B,).
            V:     Generator matrix at x0, shape (B, d, k).
            tau:   Context features, shape (B, n_τ). If None, uses exact proj.
            noise: Pre-sampled noise ε ~ N(0, I); sampled internally if None.

        Returns:
            xt:    Noisy state, shape (B, d).
            noise_proj: Projected noise Π^soft ε, shape (B, d).
        """
        mean, std = self.sde.marginal_prob(x0, t)

        if noise is None:
            noise = torch.randn_like(x0)  # (B, d)

        # Project noise onto horizontal subspace
        if tau is not None:
            noise_proj = self.soft_proj(noise, V, tau=tau)
        else:
            noise_proj = self.soft_proj(noise, V, lam=torch.ones(x0.shape[0], device=x0.device))

        xt = mean + std * noise_proj
        return xt, noise_proj


# --------------------------------------------------------------------------- #
#  QDM Reverse Process (Denoising)                                             #
# --------------------------------------------------------------------------- #

class QDMReverseProcess(nn.Module):
    """
    Quotient-projected reverse diffusion process.

    Runs the reverse SDE (Eq. 9) using a learned score model:
        dX_t = b̃_t(X_t, τ, c) dt + σ · Π^soft ∘ dB̄_t

    where the drift is the horizontally lifted quotient score:
        b̃_t = -σ² · (dπ_x)† s_t(π(x))

    In ambient coordinates, the score model outputs f_θ(X_t, t, τ, c) ∈ R^d,
    and the drift is -σ² · Π^soft · f_θ.

    Args:
        sde:         VP-SDE schedule.
        score_model: nn.Module mapping (x, t, tau, c) → score ∈ R^d.
        soft_proj:   SoftHorizontalProjection module.
        n_steps:     Number of denoising steps (default 100).
        method:      ODE/SDE solver: 'em' (Euler-Maruyama) or 'heun'.
    """

    def __init__(
        self,
        sde: VP_SDE,
        score_model: nn.Module,
        soft_proj: SoftHorizontalProjection,
        n_steps: int = 100,
        method: str = "em",
    ):
        super().__init__()
        self.sde = sde
        self.score_model = score_model
        self.soft_proj = soft_proj
        self.n_steps = n_steps
        self.method = method

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        V_fn: Callable[[Tensor], Tensor],
        device,
        dtype=torch.float32,
        tau: Optional[Tensor] = None,
        mode: str = "default",
        return_trajectory: bool = False,
    ) -> Tensor:
        """
        Generate samples by running the reverse SDE.

        Args:
            shape:    Shape of output (B, d).
            V_fn:     Callable x → V_x, shape (B, d, k).
            device:   Torch device.
            dtype:    Data type.
            tau:      Context features, shape (B, n_τ).
            mode:     Active symmetry mode (for stratified QDM).
            return_trajectory: If True, return all intermediate states.

        Returns:
            x0_hat: Generated samples, shape (B, d).
        """
        B, d = shape
        x = self.sde.prior_sample((B, d), device, dtype)

        timesteps = torch.linspace(self.sde.T, 1e-3, self.n_steps + 1, device=device)
        trajectory = [x] if return_trajectory else None

        for i in range(self.n_steps):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_cur  # negative (going backward)

            t_batch = t_cur.expand(B)

            if self.method == "em":
                x = self._em_step(x, t_batch, dt, V_fn, tau, mode)
            elif self.method == "heun":
                x = self._heun_step(x, t_batch, dt, V_fn, tau, mode)
            else:
                raise ValueError(f"Unknown method: {self.method}")

            if return_trajectory:
                trajectory.append(x.clone())

        if return_trajectory:
            return x, torch.stack(trajectory, dim=1)
        return x

    def _score_and_project(
        self,
        x: Tensor,
        t: Tensor,
        V_fn: Callable,
        tau: Optional[Tensor],
        mode: str,
    ) -> Tuple[Tensor, Tensor]:
        """Compute score and its horizontal projection."""
        V = V_fn(x)  # (B, d, k)
        score = self.score_model(x, t, tau=tau, mode=mode)  # (B, d)
        if tau is not None:
            score_proj = self.soft_proj(score, V, tau=tau)
        else:
            lam = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
            score_proj = self.soft_proj(score, V, lam=lam)
        return score_proj, V

    def _em_step(
        self,
        x: Tensor,
        t: Tensor,
        dt: Tensor,
        V_fn: Callable,
        tau: Optional[Tensor],
        mode: str,
    ) -> Tensor:
        """Single Euler-Maruyama step of the reverse SDE."""
        sigma_t = self.sde.diffusion_coeff(t)  # (B,)
        score_proj, V = self._score_and_project(x, t, V_fn, tau, mode)

        # Drift: -σ² · s_proj · dt
        drift = (-sigma_t[:, None] ** 2 * score_proj) * dt

        # Diffusion: σ · Π^soft dB  (only in reverse if using DDPM-type SDE)
        noise = torch.randn_like(x)
        if tau is not None:
            noise_proj = self.soft_proj(noise, V, tau=tau)
        else:
            lam = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
            noise_proj = self.soft_proj(noise, V, lam=lam)
        diffusion = sigma_t[:, None] * noise_proj * (-dt).sqrt().clamp(min=0)

        return x + drift + diffusion

    def _heun_step(
        self,
        x: Tensor,
        t: Tensor,
        dt: Tensor,
        V_fn: Callable,
        tau: Optional[Tensor],
        mode: str,
    ) -> Tensor:
        """Heun predictor-corrector step (2nd order, ODE only)."""
        sigma_t = self.sde.diffusion_coeff(t)
        score_proj, _ = self._score_and_project(x, t, V_fn, tau, mode)
        d1 = -sigma_t[:, None] ** 2 * score_proj

        x_pred = x + d1 * dt

        t_next = (t + dt).clamp(min=1e-3)
        sigma_next = self.sde.diffusion_coeff(t_next)
        score_proj2, _ = self._score_and_project(x_pred, t_next, V_fn, tau, mode)
        d2 = -sigma_next[:, None] ** 2 * score_proj2

        return x + 0.5 * (d1 + d2) * dt


# --------------------------------------------------------------------------- #
#  Stratified QDM: mode-switching diffusion on Q̄                              #
# --------------------------------------------------------------------------- #

class StratifiedQDMProcess(nn.Module):
    """
    Stratified quotient diffusion with contact-configuration mode switching.

    Wraps QDMReverseProcess and handles transitions between symmetry strata.
    Implements the stratified SDE from Section III.B of the paper:
        Y_t^c ∈ Q(c) for t ∈ [t_c^in, t_c^out]
        Y_{t_{c'}^in}^{c'} = Ψ_{c→c'}(Y_{t_c^out}^c)  at transitions

    Args:
        sde:         VP-SDE schedule.
        score_model: Score model conditioned on (x, t, tau, c).
        strata_proj: Dict mode → SoftHorizontalProjection.
        transition_maps: Dict (c, c') → Callable for inter-stratum transport.
        n_steps:     Total denoising steps.
    """

    def __init__(
        self,
        sde: VP_SDE,
        score_model: nn.Module,
        strata_proj: Dict[str, SoftHorizontalProjection],
        transition_maps: Optional[Dict[Tuple[str, str], Callable]] = None,
        n_steps: int = 100,
    ):
        super().__init__()
        self.sde = sde
        self.score_model = score_model
        self.strata_proj = nn.ModuleDict(strata_proj)
        self.transition_maps = transition_maps or {}
        self.n_steps = n_steps

    def transition(self, x: Tensor, from_mode: str, to_mode: str) -> Tensor:
        """Apply inter-stratum transition map Ψ_{c→c'}."""
        key = (from_mode, to_mode)
        if key in self.transition_maps:
            return self.transition_maps[key](x)
        return x  # identity

    @torch.no_grad()
    def sample(
        self,
        shape: tuple,
        V_fn_dict: Dict[str, Callable],
        mode_schedule: List[Tuple[int, str]],  # [(step_idx, mode), ...]
        device,
        tau: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Generate samples with a scheduled mode sequence.

        Args:
            shape:          Output shape (B, d).
            V_fn_dict:      Dict mode → V_fn callable.
            mode_schedule:  List of (step_index, mode) breakpoints.
                            E.g. [(0, 'stance'), (50, 'swing'), (80, 'stance')]
            device:         Torch device.
            tau:            Context features.

        Returns:
            Generated samples (B, d).
        """
        B, d = shape
        x = self.sde.prior_sample((B, d), device, torch.float32)
        timesteps = torch.linspace(self.sde.T, 1e-3, self.n_steps + 1, device=device)

        # Build step → mode mapping
        step_mode = {}
        for i, (start, mode) in enumerate(mode_schedule):
            end = mode_schedule[i + 1][0] if i + 1 < len(mode_schedule) else self.n_steps
            for step in range(start, end):
                step_mode[step] = mode

        current_mode = mode_schedule[0][1]

        for i in range(self.n_steps):
            new_mode = step_mode.get(i, current_mode)

            # Handle mode transition
            if new_mode != current_mode:
                x = self.transition(x, current_mode, new_mode)
                current_mode = new_mode

            t = timesteps[i].expand(B)
            dt = timesteps[i + 1] - timesteps[i]
            sigma_t = self.sde.diffusion_coeff(t)

            V = V_fn_dict[current_mode](x)
            proj = self.strata_proj[current_mode]
            score = self.score_model(x, t, tau=tau, mode=current_mode)

            if tau is not None:
                score_proj = proj(score, V, tau=tau)
            else:
                lam = torch.ones(B, device=device)
                score_proj = proj(score, V, lam=lam)

            drift = -sigma_t[:, None] ** 2 * score_proj * dt
            noise = torch.randn_like(x)
            if tau is not None:
                noise_proj = proj(noise, V, tau=tau)
            else:
                noise_proj = proj(noise, V, lam=lam)
            diffusion = sigma_t[:, None] * noise_proj * (-dt).abs().sqrt()

            x = x + drift + diffusion

        return x
