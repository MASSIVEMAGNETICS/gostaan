"""
TokenFormer: Advanced token-based architecture with gravitational attention.

The TokenFormer processes sequences of tokens using a stack of layers that
each combine gravitational attention with a feed-forward network and optional
parameter tokens — learnable "slots" that encode persistent knowledge across
processing steps without being part of the input stream.

Parameter tokens (pTokens) are the key innovation: they allow the model to
maintain a compact, learned knowledge base that modulates every layer,
analogous to long-term implicit memory.
"""

from __future__ import annotations

import math
import numpy as np
from typing import List, Optional

from gostaan.attention.gravitational import GravitationalAttention


class TokenFormerLayer:
    """Single TokenFormer layer: gravitational attention + FFN + parameter tokens."""

    def __init__(
        self,
        dim_model: int,
        dim_ff: int,
        num_heads: int = 4,
        num_param_tokens: int = 16,
        dim_position: int = 32,
        dropout_rate: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        self.dim_model = dim_model
        self.dim_ff = dim_ff
        self.num_param_tokens = num_param_tokens
        self.dropout_rate = dropout_rate

        rng = np.random.default_rng(seed)

        def _glorot(shape):
            lim = math.sqrt(6.0 / (shape[0] + shape[1]))
            return rng.uniform(-lim, lim, shape).astype(np.float64)

        # Gravitational attention
        self.attn = GravitationalAttention(
            dim_model=dim_model,
            dim_position=dim_position,
            num_heads=num_heads,
            seed=seed,
        )

        # Persistent parameter tokens (learnable knowledge slots)
        self.param_tokens = rng.standard_normal((num_param_tokens, dim_model)).astype(
            np.float64
        ) * 0.02

        # Feed-forward network weights
        self.W_ff1 = _glorot((dim_model, dim_ff))
        self.b_ff1 = np.zeros(dim_ff, dtype=np.float64)
        self.W_ff2 = _glorot((dim_ff, dim_model))
        self.b_ff2 = np.zeros(dim_model, dtype=np.float64)

        # Layer norm parameters
        self.ln1_gamma = np.ones(dim_model, dtype=np.float64)
        self.ln1_beta = np.zeros(dim_model, dtype=np.float64)
        self.ln2_gamma = np.ones(dim_model, dtype=np.float64)
        self.ln2_beta = np.zeros(dim_model, dtype=np.float64)

    @staticmethod
    def _layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return gamma * (x - mean) / (np.sqrt(var) + 1e-5) + beta

    @staticmethod
    def _gelu(x: np.ndarray) -> np.ndarray:
        """GELU activation: x * Φ(x)."""
        return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for one TokenFormer layer.

        Parameter tokens are prepended to the sequence before attention so
        they modulate all token interactions, then stripped from the output.

        Args:
            x: (seq_len, dim_model)

        Returns:
            output: (seq_len, dim_model)
        """
        seq_len = x.shape[0]

        # Prepend parameter tokens
        augmented = np.concatenate([self.param_tokens, x], axis=0)  # (P+L, D)

        # --- Sub-layer 1: gravitational attention with residual ---
        normed = self._layer_norm(augmented, self.ln1_gamma, self.ln1_beta)
        attended = self.attn.forward(normed)  # (P+L, D)
        augmented = augmented + attended

        # Strip parameter tokens, keep only input tokens
        x = augmented[self.num_param_tokens:]  # (L, D)

        # --- Sub-layer 2: FFN with residual ---
        normed2 = self._layer_norm(x, self.ln2_gamma, self.ln2_beta)
        ff = self._gelu(normed2 @ self.W_ff1 + self.b_ff1) @ self.W_ff2 + self.b_ff2
        x = x + ff

        return x


class TokenFormer:
    """
    Multi-layer TokenFormer: stacked gravitational-attention transformer
    with persistent parameter tokens at every layer.

    Architecture highlights:
    - No positional encoding: positions emerge from gravitational distances
    - Parameter tokens carry persistent intra-layer knowledge
    - Each layer operates on the full augmented sequence
    """

    def __init__(
        self,
        dim_model: int = 128,
        dim_ff: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        num_param_tokens: int = 16,
        dim_position: int = 32,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            dim_model:        Embedding / hidden dimension.
            dim_ff:           Feed-forward intermediate dimension.
            num_layers:       Number of TokenFormer layers.
            num_heads:        Gravitational attention heads per layer.
            num_param_tokens: Learnable parameter-token slots per layer.
            dim_position:     Semantic-space position dimension.
            seed:             Random seed for reproducibility.
        """
        self.dim_model = dim_model
        self.num_layers = num_layers

        layer_seeds = (
            [None] * num_layers
            if seed is None
            else [seed + i for i in range(num_layers)]
        )

        self.layers: List[TokenFormerLayer] = [
            TokenFormerLayer(
                dim_model=dim_model,
                dim_ff=dim_ff,
                num_heads=num_heads,
                num_param_tokens=num_param_tokens,
                dim_position=dim_position,
                seed=layer_seeds[i],
            )
            for i in range(num_layers)
        ]

    def encode(self, token_embeddings: np.ndarray) -> np.ndarray:
        """
        Encode a sequence of token embeddings through all layers.

        Args:
            token_embeddings: (seq_len, dim_model)

        Returns:
            Encoded representation: (seq_len, dim_model)
        """
        x = token_embeddings.astype(np.float64)
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def encode_batch(self, batch: np.ndarray) -> np.ndarray:
        """
        Encode a batch of token sequences.

        Args:
            batch: (batch_size, seq_len, dim_model)

        Returns:
            (batch_size, seq_len, dim_model)
        """
        return np.stack([self.encode(batch[i]) for i in range(batch.shape[0])])

    def pool(self, token_embeddings: np.ndarray) -> np.ndarray:
        """
        Encode and mean-pool to a single sequence representation.

        Args:
            token_embeddings: (seq_len, dim_model)

        Returns:
            pooled: (dim_model,)
        """
        return self.encode(token_embeddings).mean(axis=0)
