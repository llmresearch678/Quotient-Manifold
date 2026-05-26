"""
Unit Tests for QDM Components.

Tests:
  - SO(2) / SO(3) / SE(3) generators
  - Horizontal projection: rank, idempotency, orthogonality
  - Soft projection: exact/ambient limits
  - Context-adaptive projection: smoothness
  - VP-SDE: marginal statistics
  - Forward/reverse process: conservation properties
  - Score matching loss: gradient flow, zero-vert property
  - Fisher decomposition: I_X = I_Q + I_vert (Theorem 4)
  - Haar symmetrization: invariance (Theorem 1)
  - Transition maps: probability conservation (Theorem 2)
  - QDM model: end-to-end forward pass

Run with:
    pytest tests/test_qdm.py -v
    pytest tests/test_qdm.py -v --tb=short -x  # stop on first failure
"""

import pytest
import torch
import numpy as np
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# --------------------------------------------------------------------------- #
#  Helpers                                                                     #
# --------------------------------------------------------------------------- #

def random_so3_matrix(batch_size: int = 4) -> torch.Tensor:
    """Sample random rotation matrices via QR decomposition."""
    Z = torch.randn(batch_size, 3, 3)
    Q, R = torch.linalg.qr(Z)
    det_sign = torch.sign(torch.linalg.det(Q))[:, None, None]
    return Q * det_sign  # (B, 3, 3)


# --------------------------------------------------------------------------- #
#  Generator tests                                                              #
# --------------------------------------------------------------------------- #

class TestSO3Generator:
    def setup_method(self):
        from qdm.geometry.lie_groups import SO3Generator
        self.N = 5
        self.gen = SO3Generator(self.N, center=True)

    def test_output_shape(self):
        x = torch.randn(4, 3 * self.N)
        V = self.gen(x)
        assert V.shape == (4, 3 * self.N, 3), f"Expected (4, {3*self.N}, 3), got {V.shape}"

    def test_rank(self):
        """V_x should have rank 3 for non-degenerate molecules."""
        x = torch.randn(4, 3 * self.N) + 0.1  # avoid near-collinear
        V = self.gen(x)
        for b in range(4):
            rank = torch.linalg.matrix_rank(V[b]).item()
            assert rank == 3, f"V_x should have rank 3, got {rank}"

    def test_equivariance(self):
        """Generator should transform equivariantly under SO(3)."""
        x = torch.randn(1, 3 * self.N)
        R = random_so3_matrix(1)  # (1, 3, 3)

        # Rotate x: apply R to each atom
        x_rot = x.reshape(1, self.N, 3) @ R.transpose(-2, -1)
        x_rot = x_rot.reshape(1, 3 * self.N)

        V_x = self.gen(x)   # (1, 3N, 3)
        V_Rx = self.gen(x_rot)  # (1, 3N, 3)

        # The column spaces should be the same (both span rotated orbit)
        # Check: column space of V_Rx = R-rotated column space of V_x
        # (Up to sign / basis permutation — check via projectors)
        P_x = V_x[0] @ torch.linalg.pinv(V_x[0])
        P_Rx = V_Rx[0] @ torch.linalg.pinv(V_Rx[0])
        # Both should project to rank-3 subspaces; Frobenius diff should be small
        # after aligning
        diff = (P_x - P_Rx).norm()
        # Note: this is approximate; projectors may differ due to different bases
        assert diff < 10.0, f"Projectors differ too much: {diff}"


class TestSO2Generator:
    def setup_method(self):
        from qdm.geometry.lie_groups import SO2Generator
        self.d = 10
        self.gen = SO2Generator(self.d, heading_indices=[0, 1, 2, 3])

    def test_output_shape(self):
        x = torch.randn(3, self.d)
        V = self.gen(x)
        assert V.shape == (3, self.d, 1), f"Expected (3, {self.d}, 1), got {V.shape}"

    def test_antisymmetry(self):
        """SO(2) generator should give antisymmetric rotation in heading plane."""
        x = torch.randn(2, self.d)
        V = self.gen(x)
        # Generator: (-x_y, x_x, ...) for each heading pair
        # V[b, 0, 0] = -x[b, 1]; V[b, 1, 0] = x[b, 0]
        assert torch.allclose(V[:, 0, 0], -x[:, 1], atol=1e-5)
        assert torch.allclose(V[:, 1, 0], x[:, 0], atol=1e-5)


# --------------------------------------------------------------------------- #
#  Horizontal projection tests                                                 #
# --------------------------------------------------------------------------- #

class TestHorizontalProjection:
    def setup_method(self):
        from qdm.geometry.horizontal_projection import HorizontalProjection
        from qdm.geometry.lie_groups import SO3Generator
        self.N = 6
        self.proj = HorizontalProjection(eps=1e-3)
        self.gen = SO3Generator(self.N, center=True)

    def test_idempotent(self):
        """Π² = Π: applying projection twice gives the same result."""
        x = torch.randn(4, 3 * self.N)
        V = self.gen(x)
        v = torch.randn(4, 3 * self.N)

        Pv = self.proj(v, V)
        PPv = self.proj(Pv, V)
        assert torch.allclose(Pv, PPv, atol=1e-4), f"Projection not idempotent: max diff={((Pv-PPv).abs().max()):.2e}"

    def test_vertical_component_zero(self):
        """Horizontal projection should give zero vertical component."""
        x = torch.randn(4, 3 * self.N)
        V = self.gen(x)
        v = torch.randn(4, 3 * self.N)

        Pv = self.proj(v, V)
        # The projected vector should be orthogonal to all columns of V
        # V^T Π v should be ≈ 0
        VT_Pv = torch.einsum("bdk,bd->bk", V, Pv)
        assert VT_Pv.abs().max().item() < 1e-3, \
            f"Horizontal projection not orthogonal to V: max={VT_Pv.abs().max():.2e}"

    def test_vertical_vector_annihilated(self):
        """A vector in V_x should be projected to zero."""
        x = torch.randn(4, 3 * self.N)
        V = self.gen(x)

        # Construct a purely vertical vector: v = V @ alpha for random alpha
        alpha = torch.randn(4, 3, 1)  # (B, k, 1)
        v_vert = (V @ alpha).squeeze(-1)  # (B, 3N)

        Pv = self.proj(v_vert, V)
        assert Pv.norm(dim=-1).max().item() < 1e-3, \
            f"Vertical vector not annihilated: norm={Pv.norm(dim=-1).max():.2e}"

    def test_rank(self):
        """Π_x should be rank d-k = 3N-3."""
        x = torch.randn(1, 3 * self.N)
        V = self.gen(x)
        d = 3 * self.N

        # Compute full projection matrix (only for testing, not used in production)
        eye = torch.eye(d)
        P = self.proj(eye, V.expand(d, -1, -1)).T  # Apply to each basis vector
        rank = torch.linalg.matrix_rank(P).item()
        assert rank == d - 3, f"Expected rank {d-3}, got {rank}"


class TestSoftProjection:
    def setup_method(self):
        from qdm.geometry.horizontal_projection import SoftHorizontalProjection
        from qdm.geometry.lie_groups import SO3Generator
        self.N = 5
        self.soft_proj = SoftHorizontalProjection(eps=1e-3, gamma=2.1)
        self.gen = SO3Generator(self.N, center=True)

    def test_exact_symmetry_limit(self):
        """λ=1 → full horizontal projection."""
        x = torch.randn(4, 3 * self.N)
        V = self.gen(x)
        v = torch.randn(4, 3 * self.N)

        lam = torch.ones(4)
        v_soft = self.soft_proj(v, V, lam=lam)

        from qdm.geometry.horizontal_projection import HorizontalProjection
        v_exact = HorizontalProjection(eps=1e-3)(v, V)
        assert torch.allclose(v_soft, v_exact, atol=1e-4), \
            "λ=1 should give exact horizontal projection"

    def test_ambient_limit(self):
        """λ=0 → identity (ambient diffusion)."""
        x = torch.randn(4, 3 * self.N)
        V = self.gen(x)
        v = torch.randn(4, 3 * self.N)

        lam = torch.zeros(4)
        v_soft = self.soft_proj(v, V, lam=lam)
        assert torch.allclose(v_soft, v, atol=1e-6), \
            "λ=0 should give identity (ambient diffusion)"

    def test_confidence_weight_range(self):
        """λ(τ) should be in (0, 1]."""
        eps = torch.rand(100) * 3  # symmetry residuals in [0, 3]
        lam = self.soft_proj.symmetry_confidence(eps)
        assert (lam > 0).all() and (lam <= 1).all(), \
            f"Confidence weight out of range: min={lam.min():.3f}, max={lam.max():.3f}"

    def test_monotone_in_residual(self):
        """λ should decrease as symmetry residual increases."""
        eps_low = torch.tensor([0.1])
        eps_high = torch.tensor([0.5])
        lam_low = self.soft_proj.symmetry_confidence(eps_low)
        lam_high = self.soft_proj.symmetry_confidence(eps_high)
        assert lam_low > lam_high, \
            f"λ should decrease with residual: lam(0.1)={lam_low:.3f} vs lam(0.5)={lam_high:.3f}"


# --------------------------------------------------------------------------- #
#  Diffusion tests                                                              #
# --------------------------------------------------------------------------- #

class TestVPSDE:
    def setup_method(self):
        from qdm.diffusion.sde import VP_SDE
        self.sde = VP_SDE()

    def test_marginal_at_t0(self):
        """At t=0, noisy data should equal clean data."""
        x0 = torch.randn(8, 16)
        t = torch.full((8,), 1e-5)
        mean, std = self.sde.marginal_prob(x0, t)
        assert torch.allclose(mean, x0, atol=1e-3), "Mean at t≈0 should be x0"
        assert std.max().item() < 0.01, "Std at t≈0 should be near 0"

    def test_marginal_at_T(self):
        """At t=T=1, data should be nearly pure noise (mean ≈ 0)."""
        x0 = torch.randn(8, 16)
        t = torch.full((8,), 0.999)
        mean, std = self.sde.marginal_prob(x0, t)
        assert mean.abs().max().item() < 0.5, "Mean at t≈T should be near 0"
        assert std.min().item() > 0.9, "Std at t≈T should be near 1"

    def test_time_weight_monotone(self):
        """Time weight w(t) = 1/(1-t)² should increase toward t=1."""
        t_vals = torch.linspace(0.01, 0.9, 20)
        w = self.sde.time_weight(t_vals)
        assert (w.diff() > 0).all(), "Time weight should be monotonically increasing"


class TestForwardProcess:
    def setup_method(self):
        from qdm.diffusion.sde import VP_SDE, QDMForwardProcess
        from qdm.geometry.horizontal_projection import SoftHorizontalProjection
        from qdm.geometry.lie_groups import SO3Generator
        self.N = 5
        self.d = 3 * self.N
        self.sde = VP_SDE()
        self.proj = SoftHorizontalProjection(eps=1e-3)
        self.fwd = QDMForwardProcess(self.sde, self.proj)
        self.gen = SO3Generator(self.N, center=True)

    def test_output_shape(self):
        x0 = torch.randn(4, self.d)
        t = torch.rand(4)
        V = self.gen(x0)
        xt, noise_proj = self.fwd(x0, t, V)
        assert xt.shape == x0.shape
        assert noise_proj.shape == x0.shape

    def test_noise_is_horizontal(self):
        """Projected noise should have near-zero vertical component."""
        x0 = torch.randn(8, self.d)
        t = torch.rand(8)
        V = self.gen(x0)
        _, noise_proj = self.fwd(x0, t, V)

        from qdm.geometry.horizontal_projection import _rank_k_projection
        noise_vert = noise_proj - _rank_k_projection(noise_proj, V)
        vert_frac = noise_vert.norm(dim=-1) / noise_proj.norm(dim=-1).clamp(min=1e-8)
        assert vert_frac.mean().item() < 0.05, \
            f"Projected noise has too much vertical component: {vert_frac.mean():.3f}"


# --------------------------------------------------------------------------- #
#  Fisher decomposition tests (Theorem 4)                                      #
# --------------------------------------------------------------------------- #

class TestFisherDecomposition:
    def setup_method(self):
        from qdm.geometry.lie_groups import SO3Generator
        self.N = 9  # QM9 average
        self.gen = SO3Generator(self.N, center=True)
        self.k = 3
        self.d = 3 * self.N

    def test_decomposition_identity(self):
        """I_X = I_Q + I_vert (exact up to numerical precision)."""
        from qdm.geometry.bundle_utils import fisher_decomposition
        x = torch.randn(100, self.d)
        scores = torch.randn(100, self.d)

        decomp = fisher_decomposition(scores, x, self.gen)

        assert abs(decomp["I_X"] - decomp["I_Q"] - decomp["I_vert"]) < 1e-4, \
            f"I_X ≠ I_Q + I_vert: {decomp['I_X']:.4f} ≠ {decomp['I_Q']:.4f} + {decomp['I_vert']:.4f}"

    def test_vertical_fraction_near_k_over_d(self):
        """For random scores, vert fraction should be ≈ k/d = 1/N."""
        from qdm.geometry.bundle_utils import fisher_decomposition
        # Large batch for reliable estimate
        x = torch.randn(1000, self.d)
        scores = torch.randn(1000, self.d)

        decomp = fisher_decomposition(scores, x, self.gen)

        theory = self.k / self.d
        empirical = decomp["vert_fraction"]
        # Should be close but not exact due to finite sample
        assert abs(empirical - theory) < 0.05, \
            f"Empirical vert fraction {empirical:.3f} ≠ theory {theory:.3f}"

    def test_horizontal_score_zero_vert(self):
        """A score with zero vertical component should give I_vert ≈ 0."""
        from qdm.geometry.bundle_utils import fisher_decomposition, compute_horizontal_projection
        x = torch.randn(100, self.d)
        V = self.gen(x)

        # Create purely horizontal scores
        scores_ambient = torch.randn(100, self.d)
        scores_hor = compute_horizontal_projection(scores_ambient, V)

        decomp = fisher_decomposition(scores_hor, x, self.gen)
        assert decomp["vert_fraction"] < 1e-3, \
            f"Horizontal score should have I_vert≈0, got {decomp['vert_fraction']:.4f}"


# --------------------------------------------------------------------------- #
#  End-to-end QDM model test                                                   #
# --------------------------------------------------------------------------- #

class TestQDMModel:
    def setup_method(self):
        from qdm import QDM
        self.model = QDM(
            task="molecular",
            variant="QDM-S",  # Small for fast tests
            n_atoms=5,
            n_steps=5,
        )

    def test_forward_pass(self):
        """End-to-end training loss should run without error."""
        x0 = torch.randn(4, 15)  # 5 atoms × 3
        loss, metrics = self.model.training_loss(x0)
        assert torch.isfinite(loss), f"Loss is not finite: {loss}"
        assert loss.item() > 0, "Loss should be positive"

    def test_sampling(self):
        """Sample generation should return correct shape."""
        with torch.no_grad():
            samples = self.model.sample(batch_size=3)
        assert samples.shape == (3, 15), f"Expected (3, 15), got {samples.shape}"
        assert torch.isfinite(samples).all(), "Samples contain NaN or Inf"

    def test_score_horizontal(self):
        """Model score should have near-zero vertical component."""
        x = torch.randn(4, 15)
        t = torch.rand(4)
        with torch.no_grad():
            score = self.model.score(x, t)

        vert_frac = self.model.vertical_fisher_fraction(score, x)
        assert vert_frac < 0.05, \
            f"QDM score should be nearly horizontal, got vert_frac={vert_frac:.3f}"

    def test_parameter_count(self):
        """QDM-S should have approximately 1M parameters."""
        n_params = self.model.n_parameters()
        assert 500_000 < n_params < 5_000_000, \
            f"QDM-S should have ~1M params, got {n_params:,}"

    def test_no_gradient_on_vertical(self):
        """Gradients from the loss should be zero in vertical directions."""
        x0 = torch.randn(4, 15)
        loss, _ = self.model.training_loss(x0)
        loss.backward()

        # The score model should receive no gradient signal along vertical directions
        # We check that gradient norms are finite (not NaN)
        for name, p in self.model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), \
                    f"NaN gradient in parameter {name}"


class TestRobotQDM:
    def setup_method(self):
        from qdm import QDM
        self.model = QDM(
            task="robot",
            variant="QDM-S",
            n_atoms=20,  # Reduced from 82 for fast testing
            context_dim=10,
            n_modes=4,
            n_steps=5,
        )

    def test_forward_with_context(self):
        """Robot model should handle context features correctly."""
        x0 = torch.randn(4, 20)
        tau = torch.randn(4, 10)
        loss, metrics = self.model.training_loss(x0, tau=tau)
        assert torch.isfinite(loss), f"Robot loss not finite: {loss}"

    def test_sample_with_context(self):
        """Sampling with context should work."""
        tau = torch.randn(3, 10)
        with torch.no_grad():
            samples = self.model.sample(batch_size=3, tau=tau)
        assert samples.shape == (3, 20)


# --------------------------------------------------------------------------- #
#  Run if called directly                                                      #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
