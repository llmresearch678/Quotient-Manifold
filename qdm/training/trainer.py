"""
QDM Training Engine.

Implements the training protocol from Section III.D / Appendix C:
  - AdamW optimiser (β1=0.9, β2=0.999, weight_decay=1e-2)
  - Cosine LR annealing with linear warm-up (5% of steps)
  - Gradient clipping (global norm 1.0)
  - EMA model with decay 0.999
  - Mixed precision (bfloat16 forward/backward, float32 projection + optimiser)
  - Time-weighted loss w(t) = 1/(1-t)²
  - Vertical Fisher fraction tracked as diagnostic (Theorem 4)
"""

from __future__ import annotations

import math
import time
import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  EMA                                                                         #
# --------------------------------------------------------------------------- #

class EMA:
    """Exponential moving average of model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {
            k: v.clone().detach()
            for k, v in model.named_parameters()
            if v.requires_grad
        }

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name].mul_(self.decay).add_(
                param.data, alpha=1 - self.decay
            )

    def apply_shadow(self, model: nn.Module):
        """Load EMA weights into model (for evaluation)."""
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module, original: dict):
        """Restore original weights after EMA evaluation."""
        for name, param in model.named_parameters():
            if name in original:
                param.data.copy_(original[name])

    def state_dict(self) -> dict:
        return {"shadow": self.shadow, "decay": self.decay}

    def load_state_dict(self, state: dict):
        self.shadow = state["shadow"]
        self.decay = state["decay"]


# --------------------------------------------------------------------------- #
#  Learning Rate Schedule                                                      #
# --------------------------------------------------------------------------- #

def cosine_with_warmup(
    warmup_steps: int, total_steps: int, min_lr_ratio: float = 0.01
) -> Callable[[int], float]:
    """Cosine annealing with linear warm-up, per-step multiplier."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine

    return lr_lambda


# --------------------------------------------------------------------------- #
#  Trainer                                                                     #
# --------------------------------------------------------------------------- #

class QDMTrainer:
    """
    QDM training engine.

    Args:
        model:          QDM model instance.
        train_loader:   DataLoader yielding (x0, [tau], [mode]).
        val_loader:     Validation DataLoader. Optional.
        output_dir:     Directory for checkpoints and logs.
        lr:             Peak learning rate (default 3e-4).
        weight_decay:   AdamW weight decay (default 1e-2).
        n_epochs:       Number of training epochs.
        grad_clip:      Global gradient clipping norm (default 1.0).
        ema_decay:      EMA weight decay (default 0.999).
        warmup_frac:    Fraction of steps for linear warm-up (default 0.05).
        use_amp:        Use bfloat16 mixed precision (default True).
        log_every:      Log metrics every N steps.
        eval_every:     Evaluate every N steps.
        save_every:     Save checkpoint every N steps.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader=None,
        output_dir: str = "outputs",
        lr: float = 3e-4,
        weight_decay: float = 1e-2,
        n_epochs: int = 250,
        grad_clip: float = 1.0,
        ema_decay: float = 0.999,
        warmup_frac: float = 0.05,
        use_amp: bool = True,
        log_every: int = 100,
        eval_every: int = 1000,
        save_every: int = 5000,
        device: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.lr = lr
        self.n_epochs = n_epochs
        self.grad_clip = grad_clip
        self.log_every = log_every
        self.eval_every = eval_every
        self.save_every = save_every
        self.use_amp = use_amp

        # Device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = self.model.to(self.device)

        # Optimiser: separate weight decay groups (no decay on LN/bias/embeds)
        decay_params, no_decay_params = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if any(nd in name for nd in ["bias", "norm", "embedding", "emb"]):
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        self.optimizer = AdamW(
            [
                {"params": decay_params, "weight_decay": weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # LR scheduler
        steps_per_epoch = len(train_loader)
        total_steps = n_epochs * steps_per_epoch
        warmup_steps = max(1, int(warmup_frac * total_steps))
        self.scheduler = LambdaLR(
            self.optimizer,
            cosine_with_warmup(warmup_steps, total_steps),
        )

        # EMA
        self.ema = EMA(model, decay=ema_decay)

        # AMP scaler (use GradScaler for float16; not needed for bfloat16)
        self.scaler = GradScaler(enabled=(use_amp and self.device.type == "cuda"))
        self.amp_dtype = torch.bfloat16 if use_amp else torch.float32

        # State
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_history: List[dict] = []

        logger.info(f"QDMTrainer initialised | device={device} | params={sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    def _parse_batch(self, batch) -> Tuple[Tensor, Optional[Tensor], str]:
        """Unpack (x0, [tau], [mode]) from dataloader batch."""
        if isinstance(batch, (tuple, list)):
            x0 = batch[0].to(self.device)
            tau = batch[1].to(self.device) if len(batch) > 1 else None
            mode = batch[2] if len(batch) > 2 else "default"
            if isinstance(mode, (list, tuple)):
                mode = mode[0]  # Take first string if batched
        else:
            x0 = batch.to(self.device)
            tau = None
            mode = "default"
        return x0, tau, mode

    def train_step(self, batch) -> dict:
        """Single training step."""
        self.model.train()
        x0, tau, mode = self._parse_batch(batch)

        with torch.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=self.use_amp,
        ):
            loss, metrics = self.model.training_loss(x0, tau=tau, mode=mode)

        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)

        # Gradient clipping
        grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.grad_clip
        )
        metrics["grad_norm"] = grad_norm.item()

        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        self.scheduler.step()
        self.ema.update(self.model)

        self.global_step += 1
        return metrics

    @torch.no_grad()
    def eval_step(self, batch) -> dict:
        """Single validation step using EMA weights."""
        self.model.eval()
        # Temporarily apply EMA weights
        original = {n: p.data.clone() for n, p in self.model.named_parameters()}
        self.ema.apply_shadow(self.model)

        x0, tau, mode = self._parse_batch(batch)
        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp):
            loss, metrics = self.model.training_loss(x0, tau=tau, mode=mode)

        self.ema.restore(self.model, original)
        return metrics

    def train(self):
        """Full training loop."""
        logger.info(f"Starting training for {self.n_epochs} epochs")
        t0 = time.time()

        for epoch in range(self.n_epochs):
            epoch_metrics: List[dict] = []

            for batch in self.train_loader:
                metrics = self.train_step(batch)
                epoch_metrics.append(metrics)

                # Logging
                if self.global_step % self.log_every == 0:
                    avg = {k: sum(m[k] for m in epoch_metrics[-self.log_every:])
                           / len(epoch_metrics[-self.log_every:])
                           for k in metrics}
                    lr = self.scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    logger.info(
                        f"Step {self.global_step:6d} | epoch {epoch:3d} | "
                        f"loss={avg['loss']:.4f} | "
                        f"vert_frac={avg.get('vert_fraction', 0):.4f} | "
                        f"lr={lr:.2e} | "
                        f"grad_norm={avg.get('grad_norm', 0):.3f} | "
                        f"elapsed={elapsed/60:.1f}min"
                    )
                    self.train_history.append({"step": self.global_step, **avg, "lr": lr})

                # Validation
                if self.val_loader and self.global_step % self.eval_every == 0:
                    val_metrics = self._validate()
                    if val_metrics["loss"] < self.best_val_loss:
                        self.best_val_loss = val_metrics["loss"]
                        self.save_checkpoint("best.pt")

                # Checkpoint
                if self.global_step % self.save_every == 0:
                    self.save_checkpoint(f"step_{self.global_step}.pt")

        self.save_checkpoint("final.pt")
        logger.info(f"Training complete. Best val loss: {self.best_val_loss:.4f}")

    def _validate(self) -> dict:
        """Full validation pass."""
        all_metrics: List[dict] = []
        for batch in self.val_loader:
            m = self.eval_step(batch)
            all_metrics.append(m)
        avg = {k: sum(m[k] for m in all_metrics) / len(all_metrics) for k in all_metrics[0]}
        logger.info(f"  [VAL] step={self.global_step} | " +
                    " | ".join(f"{k}={v:.4f}" for k, v in avg.items()))
        return avg

    def save_checkpoint(self, filename: str):
        """Save model, EMA, optimiser, and scheduler state."""
        ckpt = {
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "ema_state_dict": self.ema.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_history": self.train_history,
        }
        path = self.output_dir / filename
        torch.save(ckpt, path)
        logger.info(f"  Checkpoint saved → {path}")

    def load_checkpoint(self, path: str):
        """Load checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.ema.load_state_dict(ckpt["ema_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.global_step = ckpt["global_step"]
        self.best_val_loss = ckpt["best_val_loss"]
        self.train_history = ckpt.get("train_history", [])
        logger.info(f"Loaded checkpoint from {path} (step={self.global_step})")
