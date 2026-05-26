"""
QDM Evaluation Metrics.

Implements all metrics from the paper:

Molecular conformer generation (Tables I-III):
  - COV-R / COV-P: Coverage recall/precision (%)
  - AMR-R / AMR-P: Average minimum RMSD recall/precision (Å)
  - Boltzmann-weighted ensemble property MAE

Robot locomotion (Table VI):
  - Success Rate (SR, %)
  - Traversal Speed (TS, m/s)
  - Energy Efficiency (EE, m/J)
  - Foot Slip Rate (FSR, %)
  - Base Stability (BS, rad/s)

Statistical:
  - Vertical Fisher fraction I_vert / I_X (Theorem 4 diagnostic)
  - Wilcoxon p-values and Cohen's d (Appendix I)
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple


# --------------------------------------------------------------------------- #
#  RMSD utilities                                                               #
# --------------------------------------------------------------------------- #

def compute_rmsd(pred: Tensor, ref: Tensor) -> Tensor:
    """
    Compute per-sample RMSD between predicted and reference conformers.

    Args:
        pred: Predicted coordinates, shape (B, 3N).
        ref:  Reference coordinates, shape (B, 3N).

    Returns:
        RMSD per sample, shape (B,).
    """
    diff = (pred - ref).reshape(pred.shape[0], -1, 3)
    return diff.norm(dim=-1).pow(2).mean(dim=-1).sqrt()


def minimum_rmsd(
    preds: Tensor,  # (n_pred, 3N)
    refs: Tensor,   # (n_ref, 3N)
) -> Tuple[Tensor, Tensor]:
    """
    Compute minimum RMSD between each predicted and all reference conformers.

    Args:
        preds: Predicted conformers, shape (n_pred, 3N).
        refs:  Reference conformers, shape (n_ref, 3N).

    Returns:
        min_rmsd_pred2ref: For each pred, min RMSD over refs. Shape (n_pred,).
        min_rmsd_ref2pred: For each ref, min RMSD over preds. Shape (n_ref,).
    """
    n_pred, d = preds.shape
    n_ref, _ = refs.shape

    # Pairwise RMSD matrix
    diff = preds[:, None, :] - refs[None, :, :]  # (n_pred, n_ref, 3N)
    rmsd_matrix = diff.reshape(n_pred, n_ref, -1, 3).norm(dim=-1).pow(2).mean(dim=-1).sqrt()

    min_pred2ref = rmsd_matrix.min(dim=1).values  # (n_pred,) — min over refs
    min_ref2pred = rmsd_matrix.min(dim=0).values  # (n_ref,)  — min over preds

    return min_pred2ref, min_ref2pred


# --------------------------------------------------------------------------- #
#  Conformer generation metrics                                                 #
# --------------------------------------------------------------------------- #

def conformer_metrics(
    predicted: List[Tensor],  # List of (n_pred_i, 3N) tensors
    references: List[Tensor],  # List of (n_ref_i, 3N) tensors
    threshold: float = 0.5,   # δ in Å (0.5 for QM9, 0.75 for DRUGS)
) -> dict:
    """
    Compute COV-R, COV-P, AMR-R, AMR-P for a batch of molecules.

    Coverage Recall (COV-R): fraction of reference conformers
        that have at least one predicted conformer within δ Å RMSD.
    Coverage Precision (COV-P): fraction of predicted conformers
        that have at least one reference conformer within δ Å RMSD.
    AMR-R: mean minimum RMSD from each reference to closest prediction.
    AMR-P: mean minimum RMSD from each prediction to closest reference.

    Args:
        predicted:  List of (n_pred, 3N) tensors for each molecule.
        references: List of (n_ref, 3N) tensors for each molecule.
        threshold:  RMSD threshold δ for coverage computation.

    Returns:
        Dict with keys: COV_R_mean, COV_R_median, AMR_R_mean, AMR_R_median,
                        COV_P_mean, COV_P_median, AMR_P_mean, AMR_P_median.
    """
    cov_r_list, cov_p_list, amr_r_list, amr_p_list = [], [], [], []

    for preds, refs in zip(predicted, references):
        preds = preds.float()
        refs = refs.float()

        min_p2r, min_r2p = minimum_rmsd(preds, refs)

        # Coverage Recall: fraction of refs covered by preds
        cov_r = (min_r2p < threshold).float().mean().item()
        # Coverage Precision: fraction of preds covered by refs
        cov_p = (min_p2r < threshold).float().mean().item()
        # AMR
        amr_r = min_r2p.mean().item()
        amr_p = min_p2r.mean().item()

        cov_r_list.append(cov_r)
        cov_p_list.append(cov_p)
        amr_r_list.append(amr_r)
        amr_p_list.append(amr_p)

    cov_r = np.array(cov_r_list)
    cov_p = np.array(cov_p_list)
    amr_r = np.array(amr_r_list)
    amr_p = np.array(amr_p_list)

    return {
        "COV_R_mean": float(cov_r.mean() * 100),
        "COV_R_median": float(np.median(cov_r) * 100),
        "AMR_R_mean": float(amr_r.mean()),
        "AMR_R_median": float(np.median(amr_r)),
        "COV_P_mean": float(cov_p.mean() * 100),
        "COV_P_median": float(np.median(cov_p) * 100),
        "AMR_P_mean": float(amr_p.mean()),
        "AMR_P_median": float(np.median(amr_p)),
    }


# --------------------------------------------------------------------------- #
#  Robot locomotion metrics                                                    #
# --------------------------------------------------------------------------- #

class RobotMetricsTracker:
    """
    Tracks and aggregates robot locomotion metrics over episodes.

    Metrics (Section IV.B of paper):
      SR:  Success Rate (% episodes without fall, within T steps)
      TS:  Traversal Speed (m/s, mean forward velocity)
      EE:  Energy Efficiency (m/J, distance / energy)
      FSR: Foot Slip Rate (% steps with slip)
      BS:  Base Stability (rad/s, mean base angular velocity magnitude)
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._successes = []
        self._speeds = []
        self._energies = []
        self._distances = []
        self._slips = []
        self._angular_vels = []

    def update(
        self,
        success: bool,
        speed: float,        # m/s
        energy: float,       # J
        distance: float,     # m
        slip_steps: int,
        total_steps: int,
        angular_vel: float,  # rad/s
    ):
        self._successes.append(float(success))
        self._speeds.append(speed)
        self._energies.append(energy)
        self._distances.append(distance)
        self._slips.append(slip_steps / max(total_steps, 1))
        self._angular_vels.append(angular_vel)

    def compute(self) -> dict:
        """Compute and return all metrics."""
        n = len(self._successes)
        if n == 0:
            return {}

        sr = float(np.mean(self._successes) * 100)
        ts = float(np.mean(self._speeds))
        ee = float(np.mean([
            d / max(e, 1e-8)
            for d, e in zip(self._distances, self._energies)
        ]))
        fsr = float(np.mean(self._slips) * 100)
        bs = float(np.mean(self._angular_vels))

        return {
            "SR": sr,         # %, higher is better
            "TS": ts,         # m/s, higher is better
            "EE": ee,         # m/J, higher is better
            "FSR": fsr,       # %, lower is better
            "BS": bs,         # rad/s, lower is better
            "n_episodes": n,
        }

    def summary(self) -> str:
        m = self.compute()
        return (
            f"SR={m.get('SR', 0):.1f}% | TS={m.get('TS', 0):.2f}m/s | "
            f"EE={m.get('EE', 0):.3f}m/J | FSR={m.get('FSR', 0):.1f}% | "
            f"BS={m.get('BS', 0):.3f}rad/s (n={m.get('n_episodes', 0)})"
        )


# --------------------------------------------------------------------------- #
#  Fisher information diagnostics                                               #
# --------------------------------------------------------------------------- #

def vertical_fisher_diagnostic(
    scores: Tensor,
    x: Tensor,
    V_fn,
    theoretical_k_over_d: float,
) -> dict:
    """
    Compute the empirical vertical Fisher fraction and compare to theory.

    Implements the diagnostic from Fig. 7A-B and Theorem 4.
    QDM should give vert_frac ≈ 0; ambient should give ≈ k/d.

    Args:
        scores:              Score vectors, shape (B, d).
        x:                   Corresponding states, shape (B, d).
        V_fn:                Callable x → V_x generator matrix.
        theoretical_k_over_d: k/d prediction from Theorem 4.

    Returns:
        Dict with diagnostic values.
    """
    from .geometry.bundle_utils import fisher_decomposition
    decomp = fisher_decomposition(scores, x, V_fn)
    decomp["theoretical_k_over_d"] = theoretical_k_over_d
    decomp["gap_from_theory"] = abs(decomp["vert_fraction"] - theoretical_k_over_d)
    return decomp


# --------------------------------------------------------------------------- #
#  Statistical tests                                                            #
# --------------------------------------------------------------------------- #

def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size between two samples."""
    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(
        ((n_a - 1) * a.std() ** 2 + (n_b - 1) * b.std() ** 2)
        / (n_a + n_b - 2)
    )
    return float((a.mean() - b.mean()) / max(pooled_std, 1e-8))


def wilcoxon_p_value(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Wilcoxon signed-rank test p-value (paired, two-sided)."""
    try:
        from scipy.stats import wilcoxon
        diff = a - b
        if np.all(diff == 0):
            return 1.0
        stat, p = wilcoxon(diff, alternative="two-sided")
        return float(p)
    except ImportError:
        # Fallback: simple t-test approximation
        diff = a - b
        n = len(diff)
        t = diff.mean() / (diff.std() / np.sqrt(n) + 1e-8)
        # Approximate p-value
        return float(2 * (1 - min(abs(t) / (abs(t) + n ** 0.5), 1.0)))


def statistical_summary(
    qdm_scores: np.ndarray,
    baseline_scores: np.ndarray,
    metric_name: str = "metric",
    higher_is_better: bool = True,
) -> dict:
    """
    Compute statistical significance summary for QDM vs. baseline.

    Returns Cohen's d, Wilcoxon p-value, and improvement percentage.
    """
    if higher_is_better:
        d = cohens_d(qdm_scores, baseline_scores)
        improv_pct = (qdm_scores.mean() - baseline_scores.mean()) / (
            abs(baseline_scores.mean()) + 1e-8
        ) * 100
    else:
        d = cohens_d(baseline_scores, qdm_scores)  # positive d = QDM better
        improv_pct = (baseline_scores.mean() - qdm_scores.mean()) / (
            abs(baseline_scores.mean()) + 1e-8
        ) * 100

    p = wilcoxon_p_value(qdm_scores, baseline_scores)

    return {
        "metric": metric_name,
        "qdm_mean": float(qdm_scores.mean()),
        "qdm_std": float(qdm_scores.std()),
        "baseline_mean": float(baseline_scores.mean()),
        "baseline_std": float(baseline_scores.std()),
        "cohens_d": float(d),
        "wilcoxon_p": float(p),
        "significant_05": p < 0.05,
        "significant_01": p < 0.01,
        "improvement_pct": float(improv_pct),
    }
