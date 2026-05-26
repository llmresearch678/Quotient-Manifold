#!/usr/bin/env python3
"""
QDM Quick-Start Demo
====================
Verifies the installation by running all core components on CPU.
Expected runtime: < 60 seconds.

Usage:
    python scripts/demo.py
"""

import sys
import time
import torch

sys.path.insert(0, ".")

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"

def check(name, fn):
    try:
        result = fn()
        print(f"  {PASS} {name}  {result}")
        return True
    except Exception as e:
        print(f"  {FAIL} {name}")
        print(f"    Error: {e}")
        return False


def main():
    print("\n" + "=" * 55)
    print("  QDM Installation Verification")
    print("=" * 55)
    start = time.time()
    all_ok = True

    # ------------------------------------------------------------------ #
    # 1. Geometry                                                          #
    # ------------------------------------------------------------------ #
    print("\n[1/4] Geometry (Lie group generators, horizontal projection)")

    from qdm.geometry.lie_groups import SO3Generator, SO2Generator
    from qdm.geometry.horizontal_projection import HorizontalProjection, SoftHorizontalProjection
    from qdm.geometry.bundle_utils import fisher_decomposition

    g3 = SO3Generator(n_atoms=9)
    g2 = SO2Generator(state_dim=82)
    proj = HorizontalProjection(eps=1e-3)

    x9 = torch.randn(4, 27)
    V3 = g3(x9)

    ok = check(
        "SO(3) generator",
        lambda: f"shape={tuple(V3.shape)}, rank={torch.linalg.matrix_rank(V3[0]).item()}"
    )
    all_ok &= ok

    x82 = torch.randn(4, 82)
    V2 = g2(x82)
    ok = check(
        "SO(2) generator",
        lambda: f"shape={tuple(V2.shape)}, rank={torch.linalg.matrix_rank(V2[0]).item()}"
    )
    all_ok &= ok

    # Idempotency test
    v = torch.randn(4, 27)
    Pv = proj(v, V3)
    PPv = proj(Pv, V3)
    err = (Pv - PPv).norm().item()
    ok = check(
        "Horizontal projection idempotency (Π²=Π)",
        lambda: f"error={err:.2e}"
    )
    all_ok &= ok and (err < 1e-3)

    # Orthogonality: V^T Π v ≈ 0
    dot = (V3.permute(0, 2, 1) @ Pv.unsqueeze(-1)).abs().max().item()
    ok = check(
        "Orthogonality (V^T Πv ≈ 0)",
        lambda: f"max_dot={dot:.2e}"
    )
    all_ok &= ok and (dot < 1e-2)

    # ------------------------------------------------------------------ #
    # 2. Fisher decomposition (Theorem 4)                                 #
    # ------------------------------------------------------------------ #
    print("\n[2/4] Fisher Information Decomposition (Theorem 4)")

    scores = torch.randn(100, 27)
    x_fd = torch.randn(100, 27)
    fd = fisher_decomposition(scores, x_fd, V_fn=g3)

    gap = abs(fd["I_X"] - fd["I_Q"] - fd["I_vert"])
    ok = check(
        "I_X = I_Q + I_vert",
        lambda: f"gap={gap:.2e}  (I_X={fd['I_X']:.3f}, I_vert={fd['I_vert']:.3f})"
    )
    all_ok &= ok and (gap < 1e-3 * fd["I_X"])

    # QDM: horizontal scores have I_vert → 0
    V_fd = g3(x_fd)
    h_scores = proj(scores, V_fd)
    fd_qdm = fisher_decomposition(h_scores, x_fd, V_fn=g3)
    ok = check(
        "QDM I_vert → 0 by construction",
        lambda: f"I_vert={fd_qdm['I_vert']:.6f}"
    )
    all_ok &= ok and (fd_qdm["I_vert"] < 1e-5)

    # ------------------------------------------------------------------ #
    # 3. VP-SDE forward process                                           #
    # ------------------------------------------------------------------ #
    print("\n[3/4] VP-SDE Forward Process")

    from qdm.diffusion.sde import VP_SDE

    sde = VP_SDE()
    x0 = torch.randn(4, 27)
    t = torch.rand(4)
    mean, std = sde.marginal_prob(x0, t)
    xt = mean + std.view(-1, 1) * torch.randn_like(x0)

    ok = check(
        "VP-SDE forward",
        lambda: f"xt shape={tuple(xt.shape)}, std range=[{std.min():.3f}, {std.max():.3f}]"
    )
    all_ok &= ok

    # ------------------------------------------------------------------ #
    # 4. Full QDM Model                                                   #
    # ------------------------------------------------------------------ #
    print("\n[4/4] QDM Model (QDM-S, molecular)")

    from qdm.models.qdm_model import QDM

    model = QDM(task="molecular", variant="QDM-S", n_atoms=9, n_steps=5)
    n_params = model.n_parameters()

    ok = check(
        "Model instantiation",
        lambda: f"{n_params/1e6:.2f}M parameters"
    )
    all_ok &= ok

    x0 = torch.randn(4, 27)
    loss, info = model.training_loss(x0)
    ok = check(
        "Training loss",
        lambda: f"loss={loss.item():.4f}"
    )
    all_ok &= ok

    loss.backward()
    ok = check("Backward pass", lambda: "OK")
    all_ok &= ok

    with torch.no_grad():
        t_test = torch.rand(4)
        score = model.score(x0, t_test)
        vf = model.vertical_fisher_fraction(score, x0)
    ok = check(
        "Score is exactly horizontal",
        lambda: f"I_vert/I_X={vf:.6f}  [should be 0]"
    )
    all_ok &= ok and (vf < 1e-5)

    with torch.no_grad():
        samples = model.sample(batch_size=2)
    ok = check(
        "Sampling (reverse SDE)",
        lambda: f"shape={tuple(samples.shape)}"
    )
    all_ok &= ok

    # ------------------------------------------------------------------ #
    # Summary                                                              #
    # ------------------------------------------------------------------ #
    elapsed = time.time() - start
    print()
    print("=" * 55)
    if all_ok:
        print(f"\033[92m  ✓ All checks passed in {elapsed:.1f}s\033[0m")
    else:
        print(f"\033[91m  ✗ Some checks FAILED (see above)\033[0m")
    print("=" * 55 + "\n")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
