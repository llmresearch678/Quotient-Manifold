#!/usr/bin/env python3
"""
Molecular Conformer Generation Experiment.

Reproduces Table I (GEOM-QM9) and Table II (GEOM-DRUGS) from the paper.

Usage:
    # Train QDM-B on GEOM-QM9
    python experiments/molecular/train_conformer.py \
        --dataset QM9 \
        --variant QDM-B \
        --output_dir outputs/qm9_qdm_b \
        --n_epochs 250

    # Evaluate a trained checkpoint
    python experiments/molecular/train_conformer.py \
        --dataset QM9 \
        --variant QDM-B \
        --eval_only \
        --checkpoint outputs/qm9_qdm_b/best.pt
"""

import argparse
import logging
import sys
import os
from pathlib import Path

import torch
import numpy as np

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from qdm import QDM, QDMTrainer
from qdm.utils.metrics import conformer_metrics, statistical_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


DATASET_CONFIGS = {
    "QM9": {
        "n_atoms": 9,
        "threshold": 0.5,     # δ in Å (COV threshold)
        "n_epochs": 250,
        "n_steps": 100,
        "context_dim": 0,
        "k_over_d": 1 / 9,   # Theoretical Fisher reduction
    },
    "DRUGS": {
        "n_atoms": 44,        # Mean heavy atoms ≈ 44 for DRUGS
        "threshold": 0.75,
        "n_epochs": 200,
        "n_steps": 100,
        "context_dim": 0,
        "k_over_d": 3 / 132,  # k=3, d=3N=132
    },
}


def make_synthetic_dataset(n_atoms: int, n_samples: int = 10000, seed: int = 42):
    """
    Create a synthetic dataset of random conformers for unit testing.

    In a real experiment, replace with GEOM dataset loader.
    """
    torch.manual_seed(seed)
    # Random Gaussian conformers centred at origin
    coords = torch.randn(n_samples, 3 * n_atoms)
    coords = coords - coords.reshape(n_samples, n_atoms, 3).mean(dim=1, keepdim=True).reshape(n_samples, -1)
    return torch.utils.data.TensorDataset(coords)


def train(args):
    cfg = DATASET_CONFIGS[args.dataset]

    logger.info(f"=== QDM Molecular Conformer Generation ===")
    logger.info(f"Dataset: {args.dataset} | Variant: {args.variant}")
    logger.info(f"n_atoms={cfg['n_atoms']} | threshold={cfg['threshold']}Å")
    logger.info(f"Theoretical Fisher reduction: k/d = {cfg['k_over_d']:.3f} = {cfg['k_over_d']*100:.1f}%")

    # Model
    model = QDM(
        task="molecular",
        variant=args.variant,
        n_atoms=cfg["n_atoms"],
        context_dim=cfg["context_dim"],
        n_steps=cfg["n_steps"],
        method=args.method,
    )
    logger.info(f"Model: {model}")

    # Dataset (replace with real GEOM loader)
    logger.info("Building dataset (synthetic placeholder; replace with GEOM loader)...")
    train_ds = make_synthetic_dataset(cfg["n_atoms"], n_samples=args.n_train, seed=42)
    val_ds = make_synthetic_dataset(cfg["n_atoms"], n_samples=args.n_val, seed=99)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.n_workers, pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.n_workers, pin_memory=True
    )

    # Trainer
    trainer = QDMTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        output_dir=args.output_dir,
        lr=args.lr,
        n_epochs=cfg["n_epochs"],
        grad_clip=1.0,
        ema_decay=0.999,
        use_amp=args.amp,
        log_every=args.log_every,
        eval_every=args.eval_every,
        save_every=args.save_every,
    )

    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    trainer.train()

    logger.info("Training complete. Run with --eval_only to evaluate.")


def evaluate(args):
    cfg = DATASET_CONFIGS[args.dataset]

    model = QDM(
        task="molecular",
        variant=args.variant,
        n_atoms=cfg["n_atoms"],
        n_steps=args.n_eval_steps,
    )
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info(f"Loaded EMA checkpoint from {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    logger.info(f"Evaluating {args.variant} on {args.dataset}...")
    logger.info(f"Theoretical Fisher reduction: {cfg['k_over_d']*100:.1f}%")

    # For real evaluation: load GEOM test set and generate conformers
    # Here we demonstrate the evaluation pipeline with synthetic data
    n_mols = 50
    n_pred = 5
    n_ref = 10

    all_preds, all_refs = [], []
    with torch.no_grad():
        for mol_idx in range(n_mols):
            preds = model.sample(n_pred)  # (n_pred, 3N)
            refs = torch.randn(n_ref, 3 * cfg["n_atoms"])
            all_preds.append(preds.cpu())
            all_refs.append(refs)

    metrics = conformer_metrics(all_preds, all_refs, threshold=cfg["threshold"])

    logger.info("=== Conformer Generation Results ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.3f}")

    logger.info(f"\nTheoretical Fisher reduction k/d = {cfg['k_over_d']*100:.1f}%")
    logger.info("(This equals ~11.1% for QM9, ~2.3% for DRUGS)")


def main():
    parser = argparse.ArgumentParser(description="QDM Molecular Conformer Generation")
    parser.add_argument("--dataset", choices=["QM9", "DRUGS"], default="QM9")
    parser.add_argument("--variant", choices=["QDM-S", "QDM-B", "QDM-L"], default="QDM-B")
    parser.add_argument("--output_dir", default="outputs/molecular")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--n_train", type=int, default=5000)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--method", choices=["em", "heun"], default="em")
    parser.add_argument("--n_eval_steps", type=int, default=100)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--save_every", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.eval_only:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
