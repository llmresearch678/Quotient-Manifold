"""
QDM Score Network: Transformer-GNN Backbone.

Implements the score network f_θ: X × [0,T] × T × C → TX from Section III.D.

Architecture:
  - Atom-type + charge + degree embeddings → 128-dim tokens
  - FiLM-conditioned time embedding
  - Alternating GNN (message passing) and Transformer (self-attention) layers
  - Radial basis function edge features from pairwise distances
  - Rotary position encoding (RoPE) for approximate translation invariance
  - Linear output head → per-atom 3D score vectors

Three sizes:
  - QDM-S (~1M): 4 GNN + 2 Transformer layers, hidden=128
  - QDM-B (~9.1M): 6 GNN + 4 Transformer layers, hidden=256 [DEFAULT]
  - QDM-L (~28.5M): 8 GNN + 6 Transformer layers, hidden=384
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple


# --------------------------------------------------------------------------- #
#  Utility modules                                                              #
# --------------------------------------------------------------------------- #

class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal time embedding, projected to model dimension via FiLM."""

    def __init__(self, hidden_dim: int, n_freqs: int = 64):
        super().__init__()
        self.n_freqs = n_freqs
        freqs = torch.exp(
            torch.linspace(0, math.log(1e4), n_freqs)
        )
        self.register_buffer("freqs", freqs)

        # FiLM: predict (gamma, beta) for feature modulation
        self.film = nn.Sequential(
            nn.Linear(2 * n_freqs, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

    def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Args:
            t: Diffusion time, shape (B,).
        Returns:
            gamma, beta: FiLM parameters, each shape (B, hidden_dim).
        """
        # Sinusoidal encoding
        t_e = t[:, None] * self.freqs[None, :]  # (B, n_freqs)
        t_enc = torch.cat([t_e.sin(), t_e.cos()], dim=-1)  # (B, 2*n_freqs)

        film_out = self.film(t_enc)  # (B, 2 * hidden_dim)
        gamma, beta = film_out.chunk(2, dim=-1)  # each (B, hidden_dim)
        return gamma, beta


class RBFEdgeFeatures(nn.Module):
    """
    Radial basis function (RBF) edge features from pairwise distances.

    Centers uniformly spaced on [0, r_max] with Gaussian kernels.
    """

    def __init__(self, n_rbf: int = 64, r_max: float = 10.0, sigma: float = 0.3):
        super().__init__()
        centers = torch.linspace(0, r_max, n_rbf)
        self.register_buffer("centers", centers)
        self.sigma = sigma

    def forward(self, dist: Tensor) -> Tensor:
        """
        Args:
            dist: Pairwise distances, shape (...).
        Returns:
            RBF features, shape (..., n_rbf).
        """
        d_e = dist[..., None]  # (..., 1)
        return torch.exp(-0.5 * ((d_e - self.centers) / self.sigma) ** 2)


class AtomEmbedding(nn.Module):
    """Atom-type + formal charge + degree embedding."""

    def __init__(
        self,
        n_atom_types: int = 16,
        n_charge: int = 5,
        n_degree: int = 7,
        hidden_dim: int = 128,
    ):
        super().__init__()
        type_dim = hidden_dim // 2
        charge_dim = hidden_dim // 4
        degree_dim = hidden_dim - type_dim - charge_dim

        self.type_emb = nn.Embedding(n_atom_types, type_dim)
        self.charge_emb = nn.Embedding(n_charge, charge_dim)   # charges: -2,-1,0,+1,+2
        self.degree_emb = nn.Embedding(n_degree, degree_dim)   # degree 0..6
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self, atom_type: Tensor, charge: Tensor, degree: Tensor
    ) -> Tensor:
        """
        Args:
            atom_type: shape (B, N), integers in [0, n_atom_types).
            charge:    shape (B, N), integers in [0, n_charge).
            degree:    shape (B, N), integers in [0, n_degree).
        Returns:
            Atom tokens, shape (B, N, hidden_dim).
        """
        h = torch.cat([
            self.type_emb(atom_type),
            self.charge_emb(charge),
            self.degree_emb(degree),
        ], dim=-1)
        return F.silu(self.proj(h))


class GNNLayer(nn.Module):
    """
    Message-passing GNN layer with RBF edge features.

    m_{ij} = MLP([h_i || h_j || e_{ij}])
    h_i' = h_i + LayerNorm(sum_j m_{ij})
    """

    def __init__(self, hidden_dim: int, n_rbf: int = 64):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim + n_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_update = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: Tensor,        # (B, N, hidden)
        edge_feat: Tensor, # (B, N, N, n_rbf)
        mask: Optional[Tensor] = None,  # (B, N) bool, True = valid atom
    ) -> Tensor:
        B, N, D = h.shape

        # Pairwise token features
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, D)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, N, D)
        msg_in = torch.cat([h_i, h_j, edge_feat], dim=-1)  # (B, N, N, 2D+rbf)

        # Edge messages
        msg = self.edge_mlp(msg_in)  # (B, N, N, D)

        # Mask padding atoms
        if mask is not None:
            msg = msg * mask[:, None, :, None].float()

        # Aggregate: sum over neighbours j
        agg = msg.sum(dim=2)  # (B, N, D)
        agg = self.node_update(agg)

        return self.norm(h + agg)


class TransformerLayer(nn.Module):
    """
    Pre-LN Transformer self-attention layer with RoPE distance bias.

    Attention score bias from pairwise distances (approximate translation inv.).
    """

    def __init__(self, hidden_dim: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Distance bias projection (geometric prior)
        self.dist_bias = nn.Linear(64, n_heads, bias=False)  # 64 RBF → n_heads

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        h: Tensor,          # (B, N, D)
        dist_feat: Tensor,  # (B, N, N, 64) RBF of pairwise distances
        mask: Optional[Tensor] = None,  # (B, N)
    ) -> Tensor:
        B, N, D = h.shape
        H = self.n_heads
        Dh = self.head_dim

        # Pre-LN
        h_norm = self.norm1(h)
        QKV = self.qkv(h_norm).reshape(B, N, 3, H, Dh).permute(2, 0, 3, 1, 4)
        Q, K, V = QKV[0], QKV[1], QKV[2]  # each (B, H, N, Dh)

        # Attention scores
        scale = Dh ** -0.5
        attn = (Q @ K.transpose(-2, -1)) * scale  # (B, H, N, N)

        # Distance bias (geometric prior)
        dist_b = self.dist_bias(dist_feat).permute(0, 3, 1, 2)  # (B, H, N, N)
        attn = attn + dist_b

        # Mask padding
        if mask is not None:
            inf_mask = (~mask[:, None, None, :]).float() * -1e9
            attn = attn + inf_mask

        attn = self.attn_drop(F.softmax(attn, dim=-1))

        out = (attn @ V).transpose(1, 2).reshape(B, N, D)  # (B, N, D)
        h = h + self.out_proj(out)

        # FFN
        h = h + self.ffn(self.norm2(h))
        return h


# --------------------------------------------------------------------------- #
#  QDM Score Network                                                           #
# --------------------------------------------------------------------------- #

class QDMScoreNet(nn.Module):
    """
    QDM Score Network: Transformer-GNN backbone.

    f_θ: (x, t, τ, c) → score ∈ R^{3N}

    After computing the score, the caller applies the soft horizontal
    projection Π^soft to confine the score to H_x.

    Args:
        n_atoms:      Maximum number of atoms N (or robot state chunks).
        hidden_dim:   Token dimension (128 / 256 / 384 for S / B / L).
        n_gnn_layers: Number of GNN layers.
        n_attn_layers: Number of Transformer layers.
        n_rbf:        Number of RBF centers for edge features.
        r_cutoff:     Cutoff radius for edge construction (Å).
        context_dim:  Terrain/context feature dimension (0 = no context).
        n_modes:      Number of symmetry modes (for mode embedding).
        dropout:      Dropout rate.
    """

    def __init__(
        self,
        n_atoms: int,
        hidden_dim: int = 256,
        n_gnn_layers: int = 6,
        n_attn_layers: int = 4,
        n_rbf: int = 64,
        r_cutoff: float = 10.0,
        context_dim: int = 0,
        n_modes: int = 1,
        n_atom_types: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.N = n_atoms
        self.hidden_dim = hidden_dim
        self.r_cutoff = r_cutoff

        # Atom embedding
        self.atom_emb = AtomEmbedding(n_atom_types, hidden_dim=hidden_dim)

        # Time embedding (FiLM conditioning)
        self.time_emb = SinusoidalTimeEmbedding(hidden_dim)

        # RBF edge features
        self.rbf = RBFEdgeFeatures(n_rbf=n_rbf, r_max=r_cutoff)

        # Edge feature projection
        self.edge_proj = nn.Linear(n_rbf, n_rbf)

        # Optional context encoder (terrain features for robot)
        if context_dim > 0:
            self.ctx_encoder = nn.Sequential(
                nn.Linear(context_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            self.ctx_encoder = None

        # Mode embedding
        self.mode_emb = nn.Embedding(max(n_modes, 1), hidden_dim)

        # Alternating GNN and Transformer layers
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, n_rbf) for _ in range(n_gnn_layers)
        ])
        self.attn_layers = nn.ModuleList([
            TransformerLayer(hidden_dim, dropout=dropout)
            for _ in range(n_attn_layers)
        ])

        # Output head: hidden → 3 (per-atom score)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 3),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        x: Tensor,
        t: Tensor,
        atom_type: Optional[Tensor] = None,
        charge: Optional[Tensor] = None,
        degree: Optional[Tensor] = None,
        tau: Optional[Tensor] = None,
        mode: str = "default",
        mask: Optional[Tensor] = None,
        mode_idx: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Compute score prediction f_θ(x, t, τ, c).

        Args:
            x:         Atom coordinates (flattened), shape (B, 3N).
            t:         Diffusion times, shape (B,).
            atom_type: Atom type indices, shape (B, N). Defaults to zeros.
            charge:    Formal charge indices, shape (B, N). Defaults to zeros.
            degree:    Bond degree indices, shape (B, N). Defaults to zeros.
            tau:       Context features, shape (B, context_dim). Optional.
            mode:      Mode string (for logging; use mode_idx for embedding).
            mask:      Atom validity mask (B, N). Defaults to all True.
            mode_idx:  Mode index tensor, shape (B,). Defaults to zeros.

        Returns:
            Score, shape (B, 3N). Caller applies horizontal projection.
        """
        B = x.shape[0]
        N = self.N
        device = x.device

        # Defaults
        if atom_type is None:
            atom_type = torch.zeros(B, N, dtype=torch.long, device=device)
        if charge is None:
            charge = torch.full((B, N), 2, dtype=torch.long, device=device)  # charge 0
        if degree is None:
            degree = torch.ones(B, N, dtype=torch.long, device=device)
        if mask is None:
            mask = torch.ones(B, N, dtype=torch.bool, device=device)
        if mode_idx is None:
            mode_idx = torch.zeros(B, dtype=torch.long, device=device)

        # Atom embedding
        h = self.atom_emb(atom_type, charge, degree)  # (B, N, hidden)

        # Time conditioning via FiLM
        gamma, beta = self.time_emb(t)  # each (B, hidden)
        h = (1 + gamma[:, None, :]) * h + beta[:, None, :]

        # Context conditioning (terrain / task)
        if self.ctx_encoder is not None and tau is not None:
            ctx = self.ctx_encoder(tau)  # (B, hidden)
            h = h + ctx[:, None, :]

        # Mode embedding
        mode_feat = self.mode_emb(mode_idx)  # (B, hidden)
        h = h + mode_feat[:, None, :]

        # Pairwise distances and RBF edge features
        coords = x.reshape(B, N, 3)
        diff = coords[:, :, None, :] - coords[:, None, :, :]  # (B, N, N, 3)
        dist = diff.norm(dim=-1)  # (B, N, N)
        edge_feat = self.rbf(dist)  # (B, N, N, n_rbf)
        edge_feat = F.silu(self.edge_proj(edge_feat))

        # Cutoff mask: only include edges within r_cutoff
        edge_mask = (dist < self.r_cutoff) & mask[:, :, None] & mask[:, None, :]

        # Interleave GNN and Transformer layers
        n_gnn = len(self.gnn_layers)
        n_attn = len(self.attn_layers)
        attn_step = max(1, n_gnn // max(n_attn, 1))

        attn_idx = 0
        for gnn_idx, gnn in enumerate(self.gnn_layers):
            h = gnn(h, edge_feat, mask=mask)
            # Apply attention after every attn_step GNN layers
            if attn_idx < n_attn and (gnn_idx + 1) % attn_step == 0:
                h = self.attn_layers[attn_idx](h, edge_feat, mask=mask)
                attn_idx += 1

        # Remaining attention layers
        while attn_idx < n_attn:
            h = self.attn_layers[attn_idx](h, edge_feat, mask=mask)
            attn_idx += 1

        # Output: per-atom 3D score
        score_per_atom = self.output_head(h)  # (B, N, 3)

        # Zero out padding atoms
        score_per_atom = score_per_atom * mask[:, :, None].float()

        return score_per_atom.reshape(B, 3 * N)  # (B, 3N)


# --------------------------------------------------------------------------- #
#  Factory: QDM-S / QDM-B / QDM-L                                             #
# --------------------------------------------------------------------------- #

_MODEL_CONFIGS = {
    "QDM-S": dict(hidden_dim=128, n_gnn_layers=4, n_attn_layers=2),
    "QDM-B": dict(hidden_dim=256, n_gnn_layers=6, n_attn_layers=4),
    "QDM-L": dict(hidden_dim=384, n_gnn_layers=8, n_attn_layers=6),
}


def build_score_net(
    variant: str,
    n_atoms: int,
    context_dim: int = 0,
    n_modes: int = 1,
    **kwargs,
) -> QDMScoreNet:
    """
    Build a QDM score network by variant name.

    Args:
        variant:     "QDM-S", "QDM-B", or "QDM-L".
        n_atoms:     Number of atoms (or state chunks).
        context_dim: Context feature dimension.
        n_modes:     Number of symmetry modes.

    Returns:
        Initialised QDMScoreNet.
    """
    if variant not in _MODEL_CONFIGS:
        raise ValueError(f"Unknown variant {variant!r}. Choose from {list(_MODEL_CONFIGS)}")
    cfg = _MODEL_CONFIGS[variant].copy()
    cfg.update(kwargs)
    return QDMScoreNet(
        n_atoms=n_atoms,
        context_dim=context_dim,
        n_modes=n_modes,
        **cfg,
    )
