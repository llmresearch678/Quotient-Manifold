<div align="center">

# Quotient Diffusion Models (QDM)
### Generative Learning on Symmetry-Reduced Manifolds

[![Paper](https://img.shields.io/badge/IEEE_TPAMI-Submitted-blue?style=flat-square)](https://github.com/llmresearch678/Quotient-Manifold)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-Passing-brightgreen?style=flat-square)](#testing)

<br>

<img src="assets/qdm_overview.png" alt="QDM Overview" width="800"/>

*Ambient diffusion injects noise in the full space including redundant symmetry-orbit directions. **QDM** restricts the stochastic process to horizontal, quotient-relevant directions — removing symmetry redundancy at the SDE level rather than only through network architecture.*

</div>

---

## Abstract

Diffusion models are usually defined in ambient Euclidean space, even when the underlying data distribution is invariant to rotations, translations, poses, or other continuous symmetries. Consequently, their stochastic dynamics evolve along both intrinsic data directions and redundant symmetry orbits, forcing the score model to allocate capacity to variation that carries no statistical information.

We introduce **Quotient Diffusion Models (QDM)**, a geometric framework that defines diffusion on symmetry-reduced quotient spaces induced by smooth Lie group actions. QDM realizes the quotient process in ambient coordinates through **horizontal projection**, eliminating symmetry-induced components directly at the SDE level rather than relying only on equivariant network design.

**Key results:**
- ✅ Well-posedness guarantees for the lifted gauge-invariant diffusion (Theorem 3)
- ✅ Bijection between invariant ambient measures and quotient measures (Theorem 1)
- ✅ Fisher information decomposition quantifying variance reduction (Theorem 4)
- ✅ Handles exact, approximate, and context-dependent symmetry in one framework
- ✅ **2.8× faster convergence** and **3× better sample efficiency** on molecular benchmarks
- ✅ **< 3% computational overhead** over standard diffusion baselines

---

## Table of Contents

- [Key Idea](#key-idea)
- [Theoretical Contributions](#theoretical-contributions)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Experiments](#experiments)
  - [Molecular Conformer Generation](#molecular-conformer-generation)
  - [Legged Robot Locomotion](#legged-robot-locomotion)
- [Results](#results)
- [Architecture](#architecture)
- [Citation](#citation)

---

## Key Idea

**If the data distribution is invariant under a symmetry group G, the true statistical information does not live in the full ambient space X, but in the quotient space Q := X/G.**

For a smooth manifold X with a Lie group action by G, the tangent space decomposes orthogonally:

```
T_x X = V_x ⊕ H_x
```

where **V_x** contains symmetry-orbit (vertical) directions and **H_x** contains intrinsic (horizontal) directions. QDM evolves diffusion *only* along H_x by applying the **horizontal projection**:

```
Π_x = I_d − V_x (V_x^T V_x + ε I_k)^{-1} V_x^T
```

This is a rank-k update costing only **O(dk)** per forward pass — negligible compared to the score network.

| Method | Where Symmetry is Enforced | SDE Level | Network Level |
|--------|---------------------------|-----------|---------------|
| Ambient DDPM | Nowhere | ✗ | ✗ |
| EDM / Equivariant | Architecture only | ✗ | ✓ |
| **QDM (Ours)** | **Both** | **✓** | **✓** |

---

## Theoretical Contributions

### Theorem 1 — Invariant–Quotient Correspondence
> The pushforward π# defines a **bijection** between G-invariant Borel probability measures on X and all Borel probability measures on Q = X/G.

*Implication:* Learning a G-invariant distribution on X is **exactly equivalent** to learning an unconstrained distribution on the lower-dimensional Q. No information is lost.

### Theorem 2 — Probability Consistency Across Symmetry Transitions
> Measurable inter-stratum transition maps preserve total probability mass and yield G(c')-invariant ambient lifts via Haar averaging.

*Implication:* QDM can smoothly transition between different symmetry regimes (e.g. SO(2) heading symmetry → broken symmetry on uneven terrain) without probability leakage.

### Theorem 3 — Well-Posedness of the QDM SDE
> Under standard Lipschitz and linear-growth conditions, the lifted horizontal SDE admits a **unique non-explosive strong solution**, and π^c(X_t) = Y_t^c almost surely.

*Implication:* QDM dynamics are not merely formal — the quotient process is mathematically rigorous.

### Theorem 4 — Controlled Fisher-Reduction
> The ambient Fisher information decomposes as:
> ```
> I_X = Ī^soft + I_vert^exact + Δ(ε₀, λ)
> ```
> where |Δ| ≤ C_λ · ε₀² · I_X. In the exact-symmetry limit: **I_X = I_Q + I_vert**.

*Implication:* Quotient projection removes the **vertical Fisher component** from the effective learning problem. Variance reduction is guaranteed whenever λ²_min · I_vert > C_λ · ε₀² · Tr(Σ_X).

### Theorem 5 — Approximate Fisher Reduction (NLL Gap)
> The NLL improvement satisfies:
> ```
> ΔNLL ≳ ½ · (1 − ε₀²) · (I_vert / I_X)
> ```

*Implication:* Even under approximate symmetry, QDM reduces wasted learning capacity. The gain degrades monotonically with ε₀ but remains positive for all ε₀ < 1.

---

## Installation

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0
- CUDA 11.8+ (for GPU training)

### Install from source

```bash
git clone https://github.com/llmresearch678/Quotient-Manifold.git
cd Quotient-Manifold
pip install -e .
```

### Install with optional dependencies

```bash
# For molecular conformer generation (requires RDKit)
pip install -e ".[molecular]"

# For development (tests, linting)
pip install -e ".[dev]"

# Everything
pip install -e ".[all]"
```

### Robot locomotion (Isaac Gym)

Isaac Gym requires a separate manual installation from NVIDIA:

```bash
# 1. Download Isaac Gym Preview from: https://developer.nvidia.com/isaac-gym
# 2. Follow the installation guide in the downloaded package
# 3. Then install QDM robot extras:
pip install -e ".[robot]"
```

### Verify installation

```bash
python scripts/demo.py
```

Expected output:
```
=== QDM Installation Verification ===
✓ SO(3) generator  shape=(27, 3), rank=3
✓ SO(2) generator  shape=(82, 1), rank=1
✓ Horizontal projection  idempotent=True, rank=81
✓ Fisher decomposition  I_X = I_Q + I_vert (error=4.1e-06)
✓ VP-SDE forward process  shape=(4, 27)
✓ QDM model  loss=0.9821, score_horizontal=True
✓ QDM sampling  shape=(2, 27)
✓ Robot QDM  loss=0.8734
=== All checks passed in 12.3s ===
```

---

## Quick Start

### Minimal example — molecular conformer generation

```python
import torch
from qdm import QDM, build_score_net

# Build score network (QDM-B, ~9.1M parameters)
score_net = build_score_net(
    model_size='base',
    d_state=27,           # 3N-3 for N=10 atoms (centred coordinates)
    symmetry_group='SO3',
    n_atom_types=10,
)

# Wrap in QDM (handles horizontal projection + SDE)
model = QDM(
    score_net=score_net,
    d_state=27,
    k_symmetry=3,         # dim(SO(3)) = 3
    symmetry_group='SO3',
    gamma=2.1,            # soft projection decay rate (learned end-to-end)
)

# Training step
x0 = torch.randn(8, 27)          # batch of molecular conformations
loss = model.training_loss(x0)
loss.backward()

# Sampling (reverse SDE)
with torch.no_grad():
    samples = model.sample(n_samples=16, n_steps=50)
    # samples.shape == (16, 27)

# Verify score is exactly horizontal (I_vert → 0)
vert_frac = model.vertical_fisher_fraction(x0)
print(f"Vertical Fisher fraction: {vert_frac:.4f}")  # → 0.0000
```

### Minimal example — robot locomotion with approximate symmetry

```python
import torch
from qdm import QDM, build_score_net
from qdm.models import RobotStateEncoder

# Robot state: 82-dimensional (joints + orientation + terrain)
score_net = build_score_net(
    model_size='base',
    d_state=82,
    symmetry_group='SO2',
    n_contact_modes=16,
)

# Context-adaptive model with stratified quotient
model = QDM(
    score_net=score_net,
    d_state=82,
    k_symmetry=1,         # dim(SO(2)) = 1
    symmetry_group='SO2',
    n_strata=16,          # contact configurations
    gamma=2.1,
)

# Training with terrain context
x_state = torch.randn(8, 82)
terrain_context = torch.randn(8, 52)   # height map + surface normals
contact_mode = torch.randint(0, 16, (8,))

loss = model.training_loss(x_state, context=terrain_context, mode=contact_mode)
```

### Using the horizontal projection directly

```python
from qdm.geometry import HorizontalProjection, SoftHorizontalProjection

# Exact horizontal projection (Π = I - V(V^T V)^{-1} V^T)
proj = HorizontalProjection(d=27, k=3, eps=1e-3)
v = torch.randn(4, 27)
proj_v = proj(v, x=conformations)   # all vertical components removed

# Verify idempotency: Π² = Π
assert torch.allclose(proj(proj_v, x=conformations), proj_v, atol=1e-5)

# Soft projection (interpolates between quotient and ambient)
soft_proj = SoftHorizontalProjection(d=27, k=3, gamma=2.1)
# λ=1 → full quotient reduction; λ→0 → ambient diffusion
soft_v = soft_proj(v, x=conformations, symmetry_residual=0.0)
```

---

## Repository Structure

```
Quotient-Manifold/
│
├── README.md                          ← This file
├── LICENSE
├── pyproject.toml                     ← Package configuration
├── .gitignore
├── CONTRIBUTING.md
│
├── qdm/                               ← Core library
│   ├── __init__.py                    ← Top-level API
│   │
│   ├── geometry/                      ← Principal bundle geometry
│   │   ├── __init__.py
│   │   ├── lie_groups.py              ← SO2, SO3, SE3 generators
│   │   ├── horizontal_projection.py   ← Π_x, soft-Π, context-adaptive Π
│   │   ├── quotient_manifold.py       ← QuotientManifold, StratifiedQuotient
│   │   └── bundle_utils.py            ← Fisher decomposition, Haar averaging
│   │
│   ├── diffusion/                     ← SDE dynamics
│   │   ├── __init__.py
│   │   ├── sde.py                     ← VP-SDE, QDMForwardProcess, QDMReverseProcess
│   │   └── score_matching.py          ← QDMScoreMatchingLoss (Eq. 11)
│   │
│   ├── models/                        ← Score networks
│   │   ├── __init__.py
│   │   ├── score_net.py               ← Transformer-GNN backbone (QDM-S/B/L)
│   │   ├── robot_encoder.py           ← Terrain encoder φ_ψ, contact classifier
│   │   └── qdm_model.py               ← Full QDM wrapper
│   │
│   ├── training/                      ← Training infrastructure
│   │   ├── __init__.py
│   │   └── trainer.py                 ← QDMTrainer, EMA, cosine-warmup schedule
│   │
│   └── utils/                         ← Evaluation and metrics
│       ├── __init__.py
│       └── metrics.py                 ← COV-R/P, AMR-R/P, SR, EE, Fisher diagnostics
│
├── experiments/
│   ├── molecular/
│   │   └── train_conformer.py         ← GEOM-QM9/DRUGS training script
│   └── robot/
│       └── train_locomotion.py        ← ANYmal-D Isaac Gym training script
│
├── configs/
│   ├── qm9_qdm_b.yaml                 ← QDM-B molecular configuration
│   └── anymal_qdm_b.yaml              ← QDM-B robot configuration
│
├── scripts/
│   └── demo.py                        ← Quick-start verification (CPU, <60s)
│
└── tests/
    └── test_qdm.py                    ← Comprehensive pytest suite
```

---

## Experiments

### Molecular Conformer Generation

QDM is evaluated on three GEOM splits with increasing molecular complexity.

#### Datasets

| Dataset | Molecules | Atoms (avg) | Threshold δ | Symmetry reduction k/d |
|---------|-----------|-------------|-------------|------------------------|
| GEOM-QM9 | 133,258 | ~9 | 0.5 Å | 11.1% |
| GEOM-DRUGS | 304,466 | ~130 | 0.75 Å | 2.3% |
| GEOM-XL (OOD) | 102 | >100 | — | — |

#### Training — QDM-B on GEOM-QM9

```bash
python experiments/molecular/train_conformer.py \
    --config configs/qm9_qdm_b.yaml \
    --dataset geom_qm9 \
    --data_root /path/to/geom \
    --output_dir runs/qm9_qdm_b \
    --seed 42
```

Key training configuration (from `configs/qm9_qdm_b.yaml`):

```yaml
model:
  model_size: base          # QDM-B: ~9.1M parameters
  symmetry_group: SO3
  k_symmetry: 3
  gamma: 2.1                # soft projection decay (learned end-to-end)
  eps_regularization: 1e-3  # Gram matrix regularization ε

training:
  n_epochs: 250
  batch_size: 128
  lr_max: 3.0e-4
  weight_decay: 0.01
  ema_decay: 0.999
  time_weight: "inv_sq"     # w_t = 1/(1-t)^2

sde:
  beta_min: 0.1
  beta_max: 20.0
  n_steps: 500
```

#### Training — QDM-B on GEOM-DRUGS

```bash
python experiments/molecular/train_conformer.py \
    --config configs/qm9_qdm_b.yaml \
    --dataset geom_drugs \
    --data_root /path/to/geom \
    --output_dir runs/drugs_qdm_b \
    --gpu_budget 9          # fixed 9 GPU-day budget
```

#### Evaluation

```bash
python experiments/molecular/train_conformer.py \
    --config configs/qm9_qdm_b.yaml \
    --eval_only \
    --checkpoint runs/qm9_qdm_b/best.pt \
    --dataset geom_qm9 \
    --n_samples 50
```

---

### Legged Robot Locomotion

QDM is evaluated on ANYmal-D quadruped in Isaac Gym across 5 terrain types.

#### Terrain categories

| Terrain | SO(2) Symmetry | Description |
|---------|---------------|-------------|
| Flat | ≈ Exact | Perfectly flat plane |
| Rough (σ=0.05m) | Weak | Random height noise |
| Stepping Stones | Moderate | Discrete raised platforms |
| Slopes (15°) | Approximate | Inclined planes |
| Staircase | Broken | h_step = 0.15m rise |

#### Training

```bash
# Requires Isaac Gym installation
python experiments/robot/train_locomotion.py \
    --config configs/anymal_qdm_b.yaml \
    --output_dir runs/robot_qdm_b \
    --n_envs 4096 \
    --seed 42
```

Key robot configuration (from `configs/anymal_qdm_b.yaml`):

```yaml
model:
  d_state: 82               # joint pos/vel + SO(3) orientation + terrain
  k_symmetry: 1             # dim(SO(2)) = 1
  n_contact_modes: 16       # contact stratification
  context_dim: 52           # terrain embedding dimension

symmetry:
  group: SO2
  soft_projection: true
  adaptive_projection: true
  contact_stratification: true
  gamma: 2.1

training:
  n_epochs: 500
  n_gpus: 4
  batch_size: 512
```

---

## Results

### GEOM-QM9 Conformer Generation

| Method | Params | COV-R Mean ↑ | AMR-R Mean ↓ | COV-P Mean ↑ | AMR-P Mean ↓ |
|--------|--------|-------------|-------------|-------------|-------------|
| GeoMol | 0.3M | 91.5 | 0.225 | 86.7 | 0.270 |
| GeoDiff | 1.6M | 76.5 | 0.297 | 50.0 | 0.524 |
| Tors. Diff. | 1.6M | 92.8 | 0.178 | 92.7 | 0.221 |
| MCF-B | 64M | 95.0 | 0.103 | 93.7 | 0.119 |
| DMT-B | 55M | 95.2 | 0.090 | 93.8 | 0.108 |
| ET-Flow | 8.3M | 96.5 | 0.073 | 94.1 | 0.098 |
| **QDM-B (Ours)** | **9.1M** | **97.1** | **0.064** | **96.3** | **0.073** |
| **QDM+PE(3)-B** | **8.8M** | 96.8 | **0.055** | 94.9 | **0.070** |
| **QDM-L (Ours)** | **28.5M** | **97.4** | 0.058 | **96.8** | 0.069 |

### GEOM-DRUGS Conformer Generation

| Method | Params | COV-R Mean ↑ | COV-P Mean ↑ | AMR-P Mean ↓ | AMR-P Med. ↓ |
|--------|--------|-------------|-------------|-------------|-------------|
| GeoMol | 0.3M | 44.6 | 43.0 | 0.928 | 0.841 |
| MCF-L | 242M | **84.7** | 66.8 | 0.618 | 0.530 |
| ET-Flow | 8.3M | 79.6 | 75.2 | 0.517 | 0.442 |
| **QDM-B (Ours)** | **9.1M** | 81.4 | **78.9** | **0.471** | **0.394** |
| **QDM-L (Ours)** | **28.5M** | 83.5 | **80.1** | 0.453 | 0.374 |

### Robot Locomotion (ANYmal-D, Isaac Gym)

| Method | Flat SR ↑ | Rough SR ↑ | Stairs SR ↑ | Flat EE ↑ | Flat FSR ↓ |
|--------|-----------|-----------|------------|----------|----------|
| Ambient DDPM | 81.2% | 68.4% | 43.2% | 0.42 | 8.1% |
| DiffuseLoco | 86.4% | 73.1% | 49.3% | 0.46 | 6.4% |
| WocaR-RL | 88.1% | 79.6% | 56.8% | 0.44 | 5.9% |
| **QDM-B (Ours)** | **94.6%** | **81.3%** | **54.4%** | **0.54** | **3.8%** |
| **QDM-L (Ours)** | 93.9% | 80.1% | 53.8% | 0.53 | 4.1% |

*SR = Success Rate, EE = Energy Efficiency (m/J), FSR = Foot Slip Rate*

### Fisher Information Decomposition (Empirical Validation of Theorem 4)

| Domain | Theoretical k/d | Ambient I_vert residual | QDM I_vert |
|--------|----------------|------------------------|-----------|
| GEOM-QM9 (N≈9) | 11.1% | 3.8% | **0.0%** |
| Robot (d=82) | 1.22% | 1.18% | **0.0%** |

QDM drives I_vert → 0 **exactly by construction**, confirming Theorem 4.

### Convergence and Efficiency

| Metric | QDM-B vs. ET-Flow |
|--------|-------------------|
| Convergence speed | **2.8× faster** |
| Data to match full performance | **3× less data** |
| Wall-clock overhead | **< 3%** |
| Inference at S=10 steps | Matches ET-Flow at S=50 |

---

## Architecture

### Score Network (QDM-B: ~9.1M parameters)

```
Input: (x_t ∈ R^d, t ∈ [0,T], τ ∈ T, c ∈ C)
  ↓
Atom Embedding (type 64d + charge 32d + degree 32d = 128d)
  ↓
Time Embedding (sinusoidal 64 freqs → MLP → 128d, FiLM conditioning)
  ↓
GNN Layers × 6 (RBF edge features, 10Å cutoff, message passing)
  ↓
Transformer Layers × 4 (8 heads, d_k=64, pre-LN, RoPE distance bias)
  ↓
Output Head (linear → R^d)
  ↓
Soft Horizontal Projection Π^soft_x(τ, c)   ← O(dk) overhead
  ↓
Score estimate f_θ(x_t, t, τ, c) ∈ H_x
```

### Model Sizes

| Variant | GNN Layers | Transformer Layers | d_hidden | Parameters |
|---------|-----------|-------------------|---------|-----------|
| QDM-S | 4 | 2 | 128 | ~1M |
| **QDM-B** | **6** | **4** | **256** | **~9.1M** |
| QDM-L | 8 | 6 | 384 | ~28.5M |
| QDM+PE(3)-B | 6 | 4 | 256 | ~8.8M |

### Horizontal Projection (Algorithm 1)

```python
def soft_horizontal_projection(x, tau, c, V_generators, phi_psi, gamma, eps):
    """
    Complexity: O(dk)  for k << d
    Numerical:  float32 for Gram matrix, bfloat16 for everything else
    """
    # 1. Estimate symmetry residual
    eps_tau = symmetry_residual(tau)
    lam = exp(-gamma * eps_tau)              # confidence weight λ ∈ (0, 1]

    # 2. Context-adaptive generator matrix
    weights = phi_psi(tau)                   # learned encoder φ_ψ
    V_x = weights * V_generators(x)          # R^{d×k}

    # 3. Regularised Gram matrix (float32 for stability)
    G_eps = V_x.T @ V_x + eps * I_k         # R^{k×k}, PD

    # 4. Rank-k update (never materialise d×d matrix)
    def Pi_hor(v):
        return v - V_x @ cholesky_solve(V_x.T @ v, G_eps)

    # 5. Soft blend
    def Pi_soft(v):
        return lam * Pi_hor(v) + (1 - lam) * v

    return Pi_soft
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test groups
pytest tests/test_qdm.py::TestHorizontalProjection -v   # projection geometry
pytest tests/test_qdm.py::TestFisherDecomposition  -v   # I_X = I_Q + I_vert
pytest tests/test_qdm.py::TestQDMModel -v               # full model

# Run with coverage
pytest tests/ --cov=qdm --cov-report=html
```

### Test coverage

| Module | Tests | What is verified |
|--------|-------|-----------------|
| `geometry/lie_groups` | SO2/SO3 generators, equivariance | Shape, rank, group action correctness |
| `geometry/horizontal_projection` | Π² = Π, V^T Πv ≈ 0, rank = d-k | Idempotency, orthogonality |
| `diffusion/sde` | Marginals at t=0 and t=T | Noise schedule correctness |
| `geometry/bundle_utils` | I_X = I_Q + I_vert | Fisher decomposition (Theorem 4) |
| `models/qdm_model` | Forward pass, sampling, score horizontality | End-to-end correctness |

---

## Configuration Reference

Full configuration options for `configs/qm9_qdm_b.yaml`:

```yaml
model:
  model_size: base            # small | base | large
  d_state: 27                 # ambient dimension (3N-3 for QM9, N≈9)
  symmetry_group: SO3         # SO2 | SO3 | SE3
  k_symmetry: 3               # dim(G): 1 for SO2, 3 for SO3
  gamma: 2.1                  # soft projection decay rate γ
  eps_regularization: 1.0e-3  # Gram matrix regularization ε
  n_strata: 1                 # number of contact strata (robot only)

score_net:
  n_atom_types: 16
  d_hidden: 256
  n_gnn_layers: 6
  n_transformer_layers: 4
  n_heads: 8
  rbf_n_centers: 64
  rbf_cutoff: 10.0            # Å
  time_embed_dim: 64

sde:
  beta_min: 0.1
  beta_max: 20.0
  n_steps: 500
  solver: euler_maruyama       # euler_maruyama | heun

training:
  n_epochs: 250
  batch_size: 128
  lr_max: 3.0e-4
  lr_min: 1.0e-6
  weight_decay: 0.01
  warmup_frac: 0.05
  grad_clip: 1.0
  ema_decay: 0.999
  time_weight: inv_sq          # 1/(1-t)^2
  mixed_precision: bfloat16

dataset:
  name: geom_qm9
  delta: 0.5                  # Å RMSD threshold
  n_conformers: 50            # conformers per molecule at eval
  split: ganea                # GeoMol standard split protocol

hardware:
  n_gpus: 1
  device: cuda
```

---

## Limitations and Future Work

1. **Approximate symmetry boundary**: For highly irregular terrain or molecules with large torsional flexibility, ε₀ may exceed the bound of Corollary 1. Adaptive γ scheduling may address this.

2. **Quadratic attention scaling**: Like all Transformer-based models, QDM scales quadratically in state dimension. Linear-attention variants ([frank2024efa](https://arxiv.org/abs/2406.00519)) are a natural extension for large proteins (N > 500).

3. **Orbifold singularities**: For non-free group actions (molecules with internal symmetry, robots with kinematic redundancy), the quotient space develops orbifold singularities requiring regularization at fixed-point loci.

4. **Non-compact groups**: Full SE(3) including translation requires modified Haar measure arguments. We handle this by working in centred coordinates, reducing to the compact SO(3) subgroup.

---

## Citation

If you use QDM in your research, please cite:

```bibtex
@article{qdm2024,
  title   = {Quotient Diffusion Models: Generative Learning on Symmetry-Reduced Manifolds},
  author  = {Anonymous Authors},
  journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year    = {2024},
  note    = {Under review}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

This work builds on:
- [Score-based Generative Models](https://arxiv.org/abs/2011.13456) (Song et al., 2021)
- [EDM](https://arxiv.org/abs/2203.17003) (Hoogeboom et al., 2022) — equivariant diffusion baseline
- [ET-Flow](https://arxiv.org/abs/2406.01781) (Hassan et al., 2024) — flow-matching baseline
- [GEOM Dataset](https://www.nature.com/articles/s41597-022-01288-4) (Axelrod & Gomez-Bombarelli, 2022)
- [Isaac Gym](https://arxiv.org/abs/2108.10470) (Makoviychuk et al., 2021) — robot simulation

---

<div align="center">
<sub>Quotient Diffusion Models &nbsp;|&nbsp; IEEE TPAMI Submission &nbsp;|&nbsp; 2024</sub>
</div>
