"""
Self-Inference Engine: Self-referential reasoning and novel idea synthesis.

The engine supports:
1. Self-inference: draw new conclusions from existing memory by projecting
   stored embeddings through the identity anchor and TokenFormer.
2. Concept synthesis: combine two or more memory embeddings to produce a
   novel blended concept.
3. Analogical reasoning: find the analogy A:B :: C:? via vector arithmetic.
4. Hypothesis generation: generate candidate hypotheses from a query by
   searching episodic memory and projecting through identity space.
5. Idea novelty scoring: measure how different a new concept is from
   everything already stored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Inference:
    """A single inference result."""

    source_ids: List[str]
    content: str
    embedding: np.ndarray
    novelty_score: float
    confidence: float
    tags: List[str] = field(default_factory=list)


class SelfInferenceEngine:
    """
    Self-Inference and Idea Synthesis Engine.

    Works in concert with:
    - EpisodicMemory  (source of existing knowledge)
    - IdentityAnchor  (filter and bias for self-consistency)
    - TokenFormer     (encode and blend representations)
    """

    def __init__(
        self,
        dim: int = 256,
        novelty_threshold: float = 0.3,
        max_hypotheses: int = 8,
        seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            dim:                Embedding dimension.
            novelty_threshold:  Minimum novelty score to keep a hypothesis.
            max_hypotheses:     Maximum number of hypotheses generated per call.
            seed:               Random seed.
        """
        self.dim = dim
        self.novelty_threshold = novelty_threshold
        self.max_hypotheses = max_hypotheses
        self._rng = np.random.default_rng(seed)

        # Pluggable components (set externally)
        self.episodic_memory = None    # EpisodicMemory
        self.identity_anchor = None    # IdentityAnchor
        self.tokenformer = None        # TokenFormer

    # ------------------------------------------------------------------
    # Core inference operations
    # ------------------------------------------------------------------

    def infer_from_memory(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> List[Inference]:
        """
        Retrieve relevant memories and project through identity to infer
        new concepts.

        Args:
            query_embedding: Seed concept embedding (dim,).
            top_k:           Number of memory cues to use.

        Returns:
            List of Inference objects sorted by novelty descending.
        """
        if self.episodic_memory is None:
            return []

        hits = self.episodic_memory.recall(query_embedding, top_k=top_k)
        if not hits:
            return []

        inferences = []
        for ep, score in hits:
            # Project through identity space
            projected = self._project_through_identity(ep.embedding)

            # Blend with query to create something new
            blended = self._synthesise(query_embedding, projected, alpha=0.6)

            novelty = self._novelty_score(blended, [e.embedding for e, _ in hits])
            if novelty < self.novelty_threshold:
                continue

            inferences.append(
                Inference(
                    source_ids=[ep.episode_id],
                    content=f"[inferred from: {ep.content[:80]}]",
                    embedding=blended,
                    novelty_score=novelty,
                    confidence=float(score),
                    tags=ep.tags,
                )
            )

        inferences.sort(key=lambda i: i.novelty_score, reverse=True)
        return inferences[: self.max_hypotheses]

    def synthesise_concepts(
        self,
        embeddings: List[np.ndarray],
        labels: Optional[List[str]] = None,
        mode: str = "mean",
    ) -> np.ndarray:
        """
        Synthesise a novel concept from multiple input embeddings.

        Args:
            embeddings: List of (dim,) vectors to blend.
            labels:     Optional text labels for each embedding.
            mode:       "mean" for equal blending, "weighted" for
                        importance-weighted blend (requires identity_anchor).

        Returns:
            Novel blended embedding (dim,).
        """
        if not embeddings:
            return np.zeros(self.dim, dtype=np.float64)

        arrays = np.stack([e.astype(np.float64) for e in embeddings])

        if mode == "weighted" and self.identity_anchor is not None:
            weights = np.array(
                [max(0.01, self.identity_anchor.identity_score(e) + 1.0)
                 for e in embeddings],
                dtype=np.float64,
            )
            weights /= weights.sum()
            blended = (arrays * weights[:, None]).sum(axis=0)
        else:
            blended = arrays.mean(axis=0)

        return self._normalize(blended)

    def analogical_infer(
        self,
        a: np.ndarray,
        b: np.ndarray,
        c: np.ndarray,
    ) -> np.ndarray:
        """
        Vector analogy: A is to B as C is to ?

        Computes:  result = normalize(b - a + c)

        Args:
            a, b, c: Embedding vectors (dim,).

        Returns:
            Inferred analogy vector (dim,).
        """
        result = (
            self._normalize(b.astype(np.float64))
            - self._normalize(a.astype(np.float64))
            + self._normalize(c.astype(np.float64))
        )
        return self._normalize(result)

    def generate_hypotheses(
        self,
        premise_embedding: np.ndarray,
        n: int = 4,
        noise_scale: float = 0.15,
    ) -> List[np.ndarray]:
        """
        Generate multiple diverse hypotheses by adding structured noise to
        the premise, then projecting each through identity space.

        Args:
            premise_embedding: Starting concept (dim,).
            n:                 Number of hypotheses.
            noise_scale:       Magnitude of perturbation.

        Returns:
            List of n hypothesis embeddings.
        """
        hypotheses = []
        for _ in range(n):
            noise = self._rng.standard_normal(self.dim) * noise_scale
            candidate = premise_embedding.astype(np.float64) + noise
            candidate = self._normalize(candidate)
            projected = self._project_through_identity(candidate)
            hypotheses.append(projected)
        return hypotheses

    def novelty_score_against_memory(self, embedding: np.ndarray) -> float:
        """
        Score how novel an embedding is relative to episodic memory.

        Returns a value in [0, 1]: 1 = completely novel, 0 = already known.
        """
        if self.episodic_memory is None or self.episodic_memory.size() == 0:
            return 1.0

        all_eps = self.episodic_memory.get_all()
        stored = [ep.embedding for ep in all_eps]
        return self._novelty_score(embedding.astype(np.float64), stored)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _project_through_identity(self, vector: np.ndarray) -> np.ndarray:
        """Project a vector through identity space if anchor is available."""
        if self.identity_anchor is not None:
            return self.identity_anchor.infer(vector, strength=0.3)
        return self._normalize(vector.astype(np.float64))

    @staticmethod
    def _synthesise(a: np.ndarray, b: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Linear interpolation between two vectors, then normalised."""
        blended = alpha * a + (1.0 - alpha) * b
        norm = np.linalg.norm(blended)
        return blended / (norm + 1e-12)

    @staticmethod
    def _novelty_score(embedding: np.ndarray, existing: List[np.ndarray]) -> float:
        """
        Novelty = 1 - max_cosine_similarity with any existing embedding.
        """
        if not existing:
            return 1.0
        e_norm = embedding / (np.linalg.norm(embedding) + 1e-12)
        sims = []
        for ex in existing:
            ex_norm = ex / (np.linalg.norm(ex) + 1e-12)
            sims.append(float(np.dot(e_norm, ex_norm)))
        return float(1.0 - max(sims))

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)
