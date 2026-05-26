"""
QDM: Quotient Diffusion Models
==============================

Generative learning on symmetry-reduced quotient manifolds.

Paper: "Quotient Diffusion Models: Generative Learning on Symmetry-Reduced Manifolds"
       IEEE Transactions on Pattern Analysis and Machine Intelligence (under review)

GitHub: https://github.com/llmresearch678/Quotient-Manifold

Quick start::

    from qdm import QDM

    # Molecular conformer generation (SO(3) quotient)
    model = QDM(task="molecular", variant="QDM-B", n_atoms=9)
    samples = model.sample(batch_size=16)  # shape: (16, 27)

    # Robot locomotion (SO(2) quotient, contact-stratified)
    model = QDM(task="robot", variant="QDM-B", n_atoms=82, context_dim=52)
    samples = model.sample(batch_size=4)   # shape: (4, 82)
"""

__version__ = "1.0.0"
__author__ = "Anonymous Authors"

from .models import QDM, QDMScoreNet, build_score_net
from .geometry import (
    HorizontalProjection,
    SoftHorizontalProjection,
    ContextAdaptiveProjection,
    SO2Generator,
    SO3Generator,
    SE3Generator,
    QuotientManifold,
)
from .diffusion import (
    VP_SDE,
    QDMForwardProcess,
    QDMReverseProcess,
    QDMScoreMatchingLoss,
)
from .training import QDMTrainer

__all__ = [
    # Main model
    "QDM",
    "QDMScoreNet",
    "build_score_net",
    # Geometry
    "HorizontalProjection",
    "SoftHorizontalProjection",
    "ContextAdaptiveProjection",
    "SO2Generator",
    "SO3Generator",
    "SE3Generator",
    "QuotientManifold",
    # Diffusion
    "VP_SDE",
    "QDMForwardProcess",
    "QDMReverseProcess",
    "QDMScoreMatchingLoss",
    # Training
    "QDMTrainer",
]
