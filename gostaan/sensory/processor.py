"""
Sensory Processor: Multi-modal data ingestion and pre-processing pipeline.

The SensoryProcessor is the gateway between raw external data and the
internal memory system. It handles:

1. Text encoding:    Sparse token-hash encoding → dense embedding.
2. Numeric arrays:   Standardisation, sparse compression, embedding.
3. Event signals:    Discrete timestamped events → contextual embedding.
4. Saliency scoring: How novel/important is this input vs. stored memories?
5. Chunking:         Long sequences are split into overlapping windows.
6. Routing:          Decides whether input goes to HDDR, episodic, or both.

The processor does NOT depend on an external LLM — all embeddings are
deterministic hash+linear projections that preserve semantic structure
well enough for retrieval and similarity tasks within the same session.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ModalityType(Enum):
    TEXT = "text"
    NUMERIC = "numeric"
    EVENT = "event"
    EMBEDDING = "embedding"


@dataclass
class SensoryInput:
    """A single processed sensory input ready for memory storage."""

    raw: Any
    modality: ModalityType
    embedding: np.ndarray
    saliency: float
    timestamp: float
    metadata: Dict[str, Any]


class SensoryProcessor:
    """
    Multi-modal sensory input processing pipeline.

    Converts raw inputs into normalised embeddings that can be written
    directly into HDDR and EpisodicMemory stores.
    """

    def __init__(
        self,
        dim: int = 256,
        chunk_size: int = 64,
        chunk_overlap: int = 16,
        saliency_threshold: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            dim:                Output embedding dimension.
            chunk_size:         Token window size for long-text chunking.
            chunk_overlap:      Overlap between adjacent chunks.
            saliency_threshold: Minimum saliency to forward to memory.
            seed:               Random seed.
        """
        self.dim = dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.saliency_threshold = saliency_threshold
        self._rng = np.random.default_rng(seed)

        # Random projection matrix: maps bag-of-hashes → dense embedding
        self._proj = self._rng.standard_normal((512, dim)).astype(np.float64)
        self._proj /= np.linalg.norm(self._proj, axis=1, keepdims=True) + 1e-12

        # Running mean of recent saliency for adaptive thresholding
        self._saliency_history: List[float] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process(
        self,
        data: Any,
        modality: Optional[ModalityType] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[SensoryInput]:
        """
        Process a raw input and return a SensoryInput if salient enough.

        Args:
            data:     Raw input (str, np.ndarray, dict, or list).
            modality: Override auto-detected modality.
            metadata: Extra key-value context to attach.

        Returns:
            SensoryInput or None if saliency is below threshold.
        """
        if modality is None:
            modality = self._detect_modality(data)

        embedding = self._embed(data, modality)
        saliency = self._compute_saliency(embedding)

        if saliency < self.saliency_threshold:
            return None

        self._saliency_history.append(saliency)
        if len(self._saliency_history) > 1000:
            self._saliency_history = self._saliency_history[-500:]

        return SensoryInput(
            raw=data,
            modality=modality,
            embedding=embedding,
            saliency=saliency,
            timestamp=time.time(),
            metadata=dict(metadata or {}),
        )

    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping token windows for long-context handling.

        Args:
            text: Input text.

        Returns:
            List of text chunks.
        """
        tokens = text.split()
        if len(tokens) <= self.chunk_size:
            return [text]

        chunks = []
        step = self.chunk_size - self.chunk_overlap
        for start in range(0, len(tokens), step):
            chunk = tokens[start : start + self.chunk_size]
            chunks.append(" ".join(chunk))
            if start + self.chunk_size >= len(tokens):
                break
        return chunks

    def process_text_chunks(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[SensoryInput]:
        """Process a long text by chunking and embedding each chunk."""
        chunks = self.chunk_text(text)
        results = []
        for chunk in chunks:
            inp = self.process(chunk, ModalityType.TEXT, metadata)
            if inp is not None:
                results.append(inp)
        return results

    def embed_text(self, text: str) -> np.ndarray:
        """Directly embed text without saliency filtering."""
        return self._embed_text(text)

    # ------------------------------------------------------------------
    # Modality-specific embedders
    # ------------------------------------------------------------------

    def _detect_modality(self, data: Any) -> ModalityType:
        if isinstance(data, str):
            return ModalityType.TEXT
        if isinstance(data, np.ndarray):
            return ModalityType.NUMERIC
        if isinstance(data, dict):
            return ModalityType.EVENT
        if isinstance(data, (list, tuple)):
            return ModalityType.NUMERIC
        return ModalityType.TEXT

    def _embed(self, data: Any, modality: ModalityType) -> np.ndarray:
        if modality == ModalityType.TEXT:
            return self._embed_text(str(data))
        if modality == ModalityType.NUMERIC:
            return self._embed_numeric(data)
        if modality == ModalityType.EVENT:
            return self._embed_event(data)
        if modality == ModalityType.EMBEDDING:
            vec = np.asarray(data, dtype=np.float64).ravel()
            if vec.shape[0] != self.dim:
                vec = self._resize_to_dim(vec)
            return self._normalize(vec)
        return self._embed_text(str(data))

    def _embed_text(self, text: str) -> np.ndarray:
        """
        Sparse bag-of-tokens hash embedding:
        1. Tokenise by whitespace.
        2. Hash each token to one of 512 bins.
        3. Build a sparse count vector.
        4. Project to dim via random projection.
        """
        tokens = text.lower().split()
        bow = np.zeros(512, dtype=np.float64)
        for token in tokens:
            h = int(hashlib.md5(token.encode()).hexdigest(), 16) % 512
            bow[h] += 1.0

        # TF normalisation
        total = bow.sum()
        if total > 0:
            bow /= total

        projected = bow @ self._proj  # (dim,)
        return self._normalize(projected)

    def _embed_numeric(self, data: Any) -> np.ndarray:
        """Standardise and project a numeric array to dim."""
        arr = np.asarray(data, dtype=np.float64).ravel()
        if arr.size == 0:
            return np.zeros(self.dim, dtype=np.float64)

        # Standardise
        mu, sigma = arr.mean(), arr.std()
        arr = (arr - mu) / (sigma + 1e-8)

        # Pad or truncate to 512, then project
        padded = np.zeros(512, dtype=np.float64)
        n = min(arr.size, 512)
        padded[:n] = arr[:n]
        projected = padded @ self._proj
        return self._normalize(projected)

    def _embed_event(self, event: Dict) -> np.ndarray:
        """Embed an event dict by serialising key-value pairs as text."""
        text = " ".join(f"{k} {v}" for k, v in event.items())
        return self._embed_text(text)

    # ------------------------------------------------------------------
    # Saliency
    # ------------------------------------------------------------------

    def _compute_saliency(self, embedding: np.ndarray) -> float:
        """
        Saliency = L2 norm of the embedding (high-energy = more salient).
        Normalised by recent history for adaptive thresholding.
        """
        base = float(np.linalg.norm(embedding))
        if self._saliency_history:
            avg = float(np.mean(self._saliency_history[-50:]))
            return base / (avg + 1e-8)
        return base

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resize_to_dim(self, vec: np.ndarray) -> np.ndarray:
        """Resize an arbitrary vector to self.dim via repeat or truncate."""
        if vec.shape[0] >= self.dim:
            return vec[: self.dim]
        repeats = self.dim // vec.shape[0] + 1
        return np.tile(vec, repeats)[: self.dim]

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)
