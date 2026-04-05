"""
Gravitational Attention: Physics-based replacement for transformer attention.

Tokens are treated as matter with mass (importance) and position (semantic
location). Attention strength is the gravitational force between token pairs:

    Force(i,j) = G * (M_i * M_j) / (Distance(i,j)^2 + epsilon)

This replaces the standard Query/Key/Value dot-product mechanism and allows
information to "orbit" massive semantic concepts, preventing information
collapse via an event-horizon limit analogous to Hawking radiation.
"""

from __future__ import annotations

import math
import numpy as np
from typing import Optional, Tuple


_MIN_MASS: float = 0.01  # Minimum token mass to prevent gravitational singularities


class GravitationalAttention:
    """
    Gravitational Attention layer.

    Replaces Query/Key/Value projections with:
      - Position projection  (where is this token in semantic space?)
      - Mass projection       (how important / how much does it attract?)
      - Value projection      (what information does it carry?)

    The force matrix acts as the attention weight matrix.
    """

    def __init__(
        self,
        dim_model: int,
        dim_position: int = 64,
        num_heads: int = 4,
        gravitational_constant: float = 1.0,
        event_horizon: float = 1e-6,
        max_force: Optional[float] = None,
        curvature: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            dim_model:              Model dimension (must be divisible by num_heads).
            dim_position:           Semantic position space dimension.
            num_heads:              Number of independent gravitational heads.
            gravitational_constant: Initial value of G (strength of attention).
            event_horizon:          Epsilon preventing division-by-zero.
            max_force:              Hawking-radiation cap on maximum force.
            curvature:              Spacetime curvature (0 = flat Euclidean).
            seed:                   Optional random seed for reproducibility.
        """
        if dim_model % num_heads != 0:
            raise ValueError(
                f"dim_model ({dim_model}) must be divisible by num_heads ({num_heads})"
            )

        self.dim_model = dim_model
        self.dim_position = dim_position
        self.num_heads = num_heads
        self.head_dim = dim_model // num_heads
        self.G = gravitational_constant
        self.event_horizon = event_horizon
        self.max_force = max_force
        self.curvature = curvature

        rng = np.random.default_rng(seed)

        def _glorot(shape: Tuple[int, ...]) -> np.ndarray:
            limit = math.sqrt(6.0 / (shape[0] + shape[1]))
            return rng.uniform(-limit, limit, shape).astype(np.float64)

        # Per-head projections
        self.W_position = [_glorot((dim_model, dim_position)) for _ in range(num_heads)]
        self.W_mass = [_glorot((dim_model, 1)) for _ in range(num_heads)]
        self.W_value = [_glorot((dim_model, self.head_dim)) for _ in range(num_heads)]
        self.W_output = _glorot((dim_model, dim_model))

        # Spacetime metric tensor
        self.metric = self._build_metric()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_metric(self) -> np.ndarray:
        """Construct the spacetime metric tensor."""
        M = np.eye(self.dim_position, dtype=np.float64)
        if self.curvature > 0:
            for i in range(self.dim_position):
                for j in range(self.dim_position):
                    if i != j:
                        M[i, j] = self.curvature * math.exp(
                            -abs(i - j) / self.dim_position
                        )
        return M

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable row-wise softmax."""
        shifted = x - x.max(axis=-1, keepdims=True)
        exp_x = np.exp(shifted)
        return exp_x / (exp_x.sum(axis=-1, keepdims=True) + 1e-12)

    @staticmethod
    def _softplus(x: np.ndarray) -> np.ndarray:
        """Softplus: log(1 + e^x), ensuring positive mass."""
        return np.log1p(np.exp(np.clip(x, -30, 30)))

    def _geodesic_distance_sq(self, positions: np.ndarray) -> np.ndarray:
        """
        Pairwise squared geodesic distances.

        Args:
            positions: (seq_len, dim_position)

        Returns:
            dist_sq: (seq_len, seq_len)
        """
        diff = positions[:, None, :] - positions[None, :, :]  # (L, L, P)
        if self.curvature > 0:
            temp = diff @ self.metric  # (L, L, P)
            dist_sq = (temp * diff).sum(axis=-1)
        else:
            dist_sq = (diff ** 2).sum(axis=-1)
        return dist_sq

    def _compute_force_matrix(
        self, positions: np.ndarray, masses: np.ndarray
    ) -> np.ndarray:
        """
        Gravitational force matrix for one head.

        Args:
            positions: (seq_len, dim_position)
            masses:    (seq_len,)

        Returns:
            force: (seq_len, seq_len)
        """
        dist_sq = self._geodesic_distance_sq(positions)
        mass_outer = np.outer(masses, masses)
        force = self.G * mass_outer / (dist_sq + self.event_horizon)
        if self.max_force is not None:
            force = np.minimum(force, self.max_force)
        return force

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Apply multi-head gravitational attention.

        Args:
            X: (seq_len, dim_model) or (batch, seq_len, dim_model)

        Returns:
            Output of same shape as X.
        """
        squeeze = X.ndim == 2
        if squeeze:
            X = X[None]

        batch, seq_len, _ = X.shape
        head_outputs = []

        for h in range(self.num_heads):
            out_head = np.zeros((batch, seq_len, self.head_dim), dtype=np.float64)
            for b in range(batch):
                x_b = X[b]
                pos = x_b @ self.W_position[h]
                raw_mass = (x_b @ self.W_mass[h]).squeeze(-1)
                masses = self._softplus(raw_mass) + _MIN_MASS
                values = x_b @ self.W_value[h]
                force = self._compute_force_matrix(pos, masses)
                attn = self._softmax(force)
                out_head[b] = attn @ values

            head_outputs.append(out_head)

        concat = np.concatenate(head_outputs, axis=-1)
        output = concat @ self.W_output
        return output.squeeze(0) if squeeze else output
