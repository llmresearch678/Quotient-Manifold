"""
QDM Geometry: Quotient Manifold constructions, horizontal projections,
and principal bundle utilities.
"""

from .horizontal_projection import (
    HorizontalProjection,
    SoftHorizontalProjection,
    ContextAdaptiveProjection,
)
from .quotient_manifold import QuotientManifold, StratifiedQuotientManifold
from .lie_groups import SO2Generator, SO3Generator, SE3Generator, LieGroupAction
from .bundle_utils import (
    compute_vertical_space,
    compute_horizontal_projection,
    compute_symmetry_residual,
    haar_symmetrize_so3 as haar_symmetrize,
)

__all__ = [
    "HorizontalProjection",
    "SoftHorizontalProjection",
    "ContextAdaptiveProjection",
    "QuotientManifold",
    "StratifiedQuotientManifold",
    "SO2Generator",
    "SO3Generator",
    "SE3Generator",
    "LieGroupAction",
    "compute_vertical_space",
    "compute_horizontal_projection",
    "compute_symmetry_residual",
    "haar_symmetrize",
]
