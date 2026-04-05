"""
Identity Anchor: Identity-locked persistent memory seeded from a user's
"I am" paragraph statement.

The identity anchor:
1. Parses a free-form "I am ..." self-description paragraph.
2. Builds a high-dimensional identity vector that gates memory writes.
3. Scores incoming information for identity relevance.
4. Supports self-inference: generates novel assertions consistent with
   the identity by linear interpolation in identity space.

The "I am" statement is the root of identity — all memories are stored
relative to it, and pruning preserves identity-consistent memories
preferentially.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class IdentityCore:
    """Parsed, embedded identity representation."""

    raw_statement: str
    traits: List[str]
    identity_vector: np.ndarray
    timestamp: float
    version: int = 1
    refinements: List[str] = field(default_factory=list)


class IdentityAnchor:
    """
    Identity Anchor built from a user's "I am" self-description.

    Once initialised, the anchor:
    - Provides an identity_vector that gates and weights memory operations.
    - Scores any input vector for identity alignment.
    - Accumulates refinements over time without losing the original core.
    - Supports self-inference: project into identity space to suggest
      internally consistent new assertions.
    """

    def __init__(self, dim: int = 256, seed: Optional[int] = None) -> None:
        """
        Args:
            dim:  Embedding dimension (must match HDDR and episodic stores).
            seed: Random seed.
        """
        self.dim = dim
        self._rng = np.random.default_rng(seed)
        self._core: Optional[IdentityCore] = None
        self._trait_embeddings: Dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialise(self, statement: str) -> IdentityCore:
        """
        Parse and embed a free-form "I am" self-description paragraph.

        The statement is tokenised by whitespace; adjectives and nouns
        following "I am", "I feel", "I value", "I believe", and "I want"
        are extracted as identity traits.  Each trait is hashed to a
        deterministic embedding vector; the identity vector is their
        normalised sum.

        Args:
            statement: Free-form self-description beginning with "I am".

        Returns:
            The constructed IdentityCore.
        """
        traits = self._extract_traits(statement)
        trait_embeds = [self._embed_trait(t) for t in traits]

        if trait_embeds:
            identity_vector = np.sum(trait_embeds, axis=0)
        else:
            # Fallback: hash the whole statement
            identity_vector = self._hash_to_vector(statement)

        identity_vector = self._normalize(identity_vector)

        self._core = IdentityCore(
            raw_statement=statement,
            traits=traits,
            identity_vector=identity_vector,
            timestamp=time.time(),
        )
        return self._core

    def refine(self, additional_statement: str) -> IdentityCore:
        """
        Add a refinement to the identity.

        The new traits are blended into the existing identity vector with
        a small weight (0.15) to preserve core identity while allowing growth.

        Args:
            additional_statement: New "I am" / "I believe" / etc. statement.

        Returns:
            Updated IdentityCore.
        """
        if self._core is None:
            return self.initialise(additional_statement)

        new_traits = self._extract_traits(additional_statement)
        if not new_traits:
            return self._core

        new_embeds = [self._embed_trait(t) for t in new_traits]
        delta = self._normalize(np.sum(new_embeds, axis=0))

        # Blend: keep 85% core, blend 15% new
        blended = 0.85 * self._core.identity_vector + 0.15 * delta
        self._core.identity_vector = self._normalize(blended)
        self._core.traits.extend(new_traits)
        self._core.refinements.append(additional_statement)
        self._core.version += 1

        return self._core

    # ------------------------------------------------------------------
    # Scoring & inference
    # ------------------------------------------------------------------

    def identity_score(self, vector: np.ndarray) -> float:
        """
        Score how well a vector aligns with the identity.

        Returns:
            Cosine similarity in [-1, 1]; higher means more identity-aligned.
        """
        if self._core is None:
            return 0.0
        return float(
            np.dot(self._normalize(vector.astype(np.float64)),
                   self._core.identity_vector)
        )

    def gate(self, vector: np.ndarray, threshold: float = 0.0) -> Tuple[bool, float]:
        """
        Decide whether an input vector should be stored based on identity
        alignment.

        Args:
            vector:    Input embedding to evaluate.
            threshold: Minimum identity score to allow storage.

        Returns:
            (should_store, identity_score)
        """
        score = self.identity_score(vector)
        return score >= threshold, score

    def infer(self, seed_vector: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Self-inference: project a seed vector towards the identity core
        to generate an identity-consistent new representation.

        Args:
            seed_vector: Starting embedding (dim,).
            strength:    How strongly to pull towards identity (0=none, 1=full).

        Returns:
            New embedding biased towards identity (dim,).
        """
        if self._core is None:
            return seed_vector.astype(np.float64)

        seed = self._normalize(seed_vector.astype(np.float64))
        identity = self._core.identity_vector
        blended = (1.0 - strength) * seed + strength * identity
        return self._normalize(blended)

    @property
    def core(self) -> Optional[IdentityCore]:
        """The current IdentityCore, or None if not initialised."""
        return self._core

    @property
    def identity_vector(self) -> Optional[np.ndarray]:
        """The current identity vector, or None if not initialised."""
        return self._core.identity_vector if self._core else None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    _TRIGGER_PHRASES = (
        "i am", "i feel", "i value", "i believe", "i want",
        "i love", "i think", "i know", "i create", "i build",
    )

    def _extract_traits(self, statement: str) -> List[str]:
        """Extract key trait words from a free-form statement."""
        words = statement.lower().split()
        traits = []
        i = 0
        while i < len(words):
            for phrase in self._TRIGGER_PHRASES:
                phrase_words = phrase.split()
                n = len(phrase_words)
                if words[i : i + n] == phrase_words and i + n <= len(words):
                    # Take the next 1-3 words as the trait
                    trait_words = []
                    for offset in range(1, 4):
                        if i + n + offset - 1 < len(words):
                            w = words[i + n + offset - 1].strip(".,;:!?\"'")
                            if w and w not in ("a", "an", "the", "and", "or", "but"):
                                trait_words.append(w)
                    if trait_words:
                        traits.append(" ".join(trait_words))
                    break
            i += 1
        return list(dict.fromkeys(traits))  # deduplicate preserving order

    def _embed_trait(self, trait: str) -> np.ndarray:
        """Deterministically embed a trait string to a unit vector."""
        if trait in self._trait_embeddings:
            return self._trait_embeddings[trait]
        vec = self._hash_to_vector(trait)
        normed = self._normalize(vec)
        self._trait_embeddings[trait] = normed
        return normed

    def _hash_to_vector(self, text: str) -> np.ndarray:
        """Hash text to a reproducible float64 vector of shape (dim,)."""
        digest = hashlib.sha512(text.encode()).digest()
        # Repeat digest bytes to fill dim floats
        needed_bytes = self.dim * 8
        repeated = (digest * (needed_bytes // len(digest) + 1))[:needed_bytes]
        vec = np.frombuffer(repeated, dtype=np.uint8).astype(np.float64)
        vec = (vec / 127.5) - 1.0  # scale to [-1, 1]
        return vec[: self.dim]

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)
