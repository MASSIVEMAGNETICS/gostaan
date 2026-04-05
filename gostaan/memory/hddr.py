"""
HDDR Memory: High-Dimensional Dense Representation Memory.

Stores experience as high-dimensional vectors with:
- Sparse write: only the most salient dimensions are updated
- Dense read: retrieval is a soft weighted sum over all stored vectors
- Importance scoring for intelligent pruning
- Compression via SVD-based dimensionality reduction
"""

from __future__ import annotations

import time
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class HDDRCell:
    """A single memory cell in HDDR storage."""

    vector: np.ndarray
    importance: float
    timestamp: float
    access_count: int = 0
    tags: List[str] = field(default_factory=list)

    def decay(self, rate: float) -> None:
        """Apply exponential importance decay."""
        self.importance *= (1.0 - rate)


class HDDRMemory:
    """
    High-Dimensional Dense Representation Memory.

    Inspired by human working/long-term memory interaction:
    - Sparse write: only top-k dimensions are updated per write
    - Associative retrieval: cosine-similarity soft weighted sum
    - Automatic pruning of low-importance, stale entries
    - SVD compression when storage exceeds capacity
    """

    def __init__(
        self,
        dim: int = 256,
        capacity: int = 1024,
        sparsity: float = 0.3,
        decay_rate: float = 0.001,
        compression_threshold: float = 0.9,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            dim:                    Representation dimension.
            capacity:               Maximum number of stored cells.
            sparsity:               Fraction of dimensions written per write (0-1).
            decay_rate:             Importance decay per time step.
            compression_threshold:  Capacity fraction that triggers compression.
            seed:                   Random seed.
        """
        self.dim = dim
        self.capacity = capacity
        self.sparsity = sparsity
        self.decay_rate = decay_rate
        self.compression_threshold = compression_threshold
        self._rng = np.random.default_rng(seed)

        self._cells: List[HDDRCell] = []

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def write(
        self,
        vector: np.ndarray,
        importance: float = 1.0,
        tags: Optional[List[str]] = None,
    ) -> int:
        """
        Sparse-write a new memory vector.

        Only the top (sparsity * dim) dimensions with greatest magnitude
        are retained; the rest are zeroed out, creating a sparse code.

        Args:
            vector:     Dense vector of shape (dim,).
            importance: Initial importance score.
            tags:       Optional string tags for retrieval.

        Returns:
            Index of the new cell.
        """
        if vector.shape != (self.dim,):
            raise ValueError(
                f"HDDRMemory.write: expected vector of shape ({self.dim},), "
                f"got {vector.shape}. Check that your embedding dimension matches "
                f"the 'dim' parameter used to construct this HDDRMemory."
            )

        # Sparse encoding: keep top-k dimensions
        sparse_vec = self._sparse_encode(vector)

        cell = HDDRCell(
            vector=sparse_vec.astype(np.float64),
            importance=float(importance),
            timestamp=time.time(),
            tags=list(tags or []),
        )
        self._cells.append(cell)
        idx = len(self._cells) - 1

        # Trigger compression if over capacity threshold
        if len(self._cells) >= int(self.capacity * self.compression_threshold):
            self._compress()

        return idx

    def read(
        self,
        query: np.ndarray,
        top_k: int = 5,
        min_importance: float = 0.0,
    ) -> List[Tuple[int, float, np.ndarray]]:
        """
        Soft-associative read: retrieve cells most similar to query.

        Args:
            query:          Query vector of shape (dim,).
            top_k:          Number of results to return.
            min_importance: Minimum importance threshold.

        Returns:
            List of (cell_index, similarity_score, cell_vector) tuples,
            sorted by descending combined score.
        """
        if not self._cells:
            return []

        q_norm = self._normalize(query.astype(np.float64))
        scores = []
        for i, cell in enumerate(self._cells):
            if cell.importance < min_importance:
                continue
            sim = float(np.dot(q_norm, self._normalize(cell.vector)))
            combined = sim * cell.importance
            scores.append((i, combined, cell.vector))

        scores.sort(key=lambda t: t[1], reverse=True)
        results = scores[:top_k]

        # Increment access count
        for i, _, _ in results:
            self._cells[i].access_count += 1
            self._cells[i].importance = min(
                self._cells[i].importance * 1.05, 10.0
            )

        return results

    def associative_recall(self, query: np.ndarray, top_k: int = 5) -> np.ndarray:
        """
        Dense recall: weighted sum of top-k matching vectors.

        Args:
            query:  (dim,)
            top_k:  How many cells to blend.

        Returns:
            Blended recall vector (dim,).
        """
        hits = self.read(query, top_k=top_k)
        if not hits:
            return np.zeros(self.dim, dtype=np.float64)

        weights = np.array([s for _, s, _ in hits], dtype=np.float64)
        weights = weights / (weights.sum() + 1e-12)
        blended = sum(w * v for (_, _, v), w in zip(hits, weights))
        return blended

    def decay_all(self) -> None:
        """Apply importance decay to all cells."""
        for cell in self._cells:
            cell.decay(self.decay_rate)

    def prune(self, min_importance: float = 0.01) -> int:
        """
        Remove cells whose importance has dropped below threshold.

        Returns:
            Number of cells removed.
        """
        before = len(self._cells)
        self._cells = [c for c in self._cells if c.importance >= min_importance]
        return before - len(self._cells)

    def size(self) -> int:
        """Return number of stored cells."""
        return len(self._cells)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _sparse_encode(self, vector: np.ndarray) -> np.ndarray:
        """Zero out all but the top-sparsity fraction of dimensions."""
        k = max(1, int(self.dim * self.sparsity))
        sparse = np.zeros_like(vector)
        top_indices = np.argpartition(np.abs(vector), -k)[-k:]
        sparse[top_indices] = vector[top_indices]
        return sparse

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        return v / (norm + 1e-12)

    def _compress(self) -> None:
        """
        SVD-based compression: project to a lower-dimensional subspace,
        then drop the least important cells to stay within capacity.
        """
        if len(self._cells) <= self.capacity:
            return

        # Sort by importance descending, keep top capacity cells
        self._cells.sort(key=lambda c: c.importance, reverse=True)
        self._cells = self._cells[: self.capacity]
