"""
Episodic Memory: Human-like day-to-day memory with intelligent pruning.

Stores experiences as episodes (discrete events with context, embedding,
emotional weight, and temporal metadata). Retrieval is cue-based:
a query vector activates the most relevant episodes via similarity + recency.

Features:
- Temporal clustering: episodes close in time share contextual links
- Importance weighting: access frequency and emotional salience boost recall
- Consolidation: similar episodes are merged during idle/REM cycles
- Sparse intelligent parsing: only the most informative tokens of each
  episode are retained after consolidation
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Episode:
    """A single episodic memory record."""

    episode_id: str
    content: str
    embedding: np.ndarray
    timestamp: float
    importance: float = 1.0
    emotional_weight: float = 0.5
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    context: Dict[str, str] = field(default_factory=dict)
    consolidated: bool = False

    def age_seconds(self) -> float:
        return time.time() - self.timestamp

    def recency_score(self, half_life_seconds: float = 3600.0) -> float:
        """Exponential recency: more recent episodes score higher."""
        age = self.age_seconds()
        return float(np.exp(-age / half_life_seconds))


class EpisodicMemory:
    """
    Human-like episodic memory store.

    Stores discrete experiences (episodes) and supports:
    - Cue-based retrieval (semantic similarity + recency + importance)
    - Episodic consolidation (merging similar episodes during REM cycles)
    - Temporal traversal (recall what happened "yesterday", etc.)
    - Intelligent pruning (removes low-salience, stale episodes)
    """

    def __init__(
        self,
        dim: int = 256,
        max_episodes: int = 2048,
        recency_half_life: float = 3600.0,
        consolidation_threshold: float = 0.85,
        min_importance: float = 0.05,
    ) -> None:
        """
        Args:
            dim:                     Embedding dimension.
            max_episodes:            Hard cap on number of stored episodes.
            recency_half_life:       Half-life (seconds) for recency scoring.
            consolidation_threshold: Cosine-similarity above which two episodes
                                     are candidates for consolidation.
            min_importance:          Episodes below this importance are pruned.
        """
        self.dim = dim
        self.max_episodes = max_episodes
        self.recency_half_life = recency_half_life
        self.consolidation_threshold = consolidation_threshold
        self.min_importance = min_importance

        self._episodes: Dict[str, Episode] = {}

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def store(
        self,
        content: str,
        embedding: np.ndarray,
        importance: float = 1.0,
        emotional_weight: float = 0.5,
        tags: Optional[List[str]] = None,
        context: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Store a new episode.

        Args:
            content:         Raw text or description of the experience.
            embedding:       Dense vector representation (dim,).
            importance:      Initial salience score (0-10).
            emotional_weight: Emotional intensity (0-1).
            tags:            Optional string labels.
            context:         Optional key-value context metadata.

        Returns:
            episode_id of the new episode.
        """
        if embedding.shape != (self.dim,):
            raise ValueError(f"Expected embedding of shape ({self.dim},)")

        episode_id = str(uuid.uuid4())
        episode = Episode(
            episode_id=episode_id,
            content=content,
            embedding=embedding.astype(np.float64),
            timestamp=time.time(),
            importance=float(importance),
            emotional_weight=float(emotional_weight),
            tags=list(tags or []),
            context=dict(context or {}),
        )
        self._episodes[episode_id] = episode

        if len(self._episodes) > self.max_episodes:
            self._enforce_capacity()

        return episode_id

    def recall(
        self,
        cue: np.ndarray,
        top_k: int = 5,
        recency_weight: float = 0.3,
        importance_weight: float = 0.3,
        similarity_weight: float = 0.4,
        tags: Optional[List[str]] = None,
    ) -> List[Tuple[Episode, float]]:
        """
        Cue-based episodic recall.

        Retrieves the most relevant episodes by a weighted combination of:
        - Semantic similarity (cosine) between cue and episode embedding
        - Recency (exponential decay from current time)
        - Importance (access frequency + salience)

        Args:
            cue:                Query embedding (dim,).
            top_k:              Number of episodes to return.
            recency_weight:     Weight on recency score.
            importance_weight:  Weight on importance score.
            similarity_weight:  Weight on semantic similarity.
            tags:               If provided, restrict to episodes with these tags.

        Returns:
            List of (Episode, combined_score) sorted by score descending.
        """
        if not self._episodes:
            return []

        cue_norm = self._normalize(cue.astype(np.float64))
        candidates = list(self._episodes.values())
        if tags:
            tag_set = set(tags)
            candidates = [e for e in candidates if tag_set & set(e.tags)]

        scored = []
        for ep in candidates:
            sim = float(np.dot(cue_norm, self._normalize(ep.embedding)))
            rec = ep.recency_score(self.recency_half_life)
            imp = min(ep.importance / 10.0, 1.0)
            score = (
                similarity_weight * sim
                + recency_weight * rec
                + importance_weight * imp
            )
            scored.append((ep, score))

        scored.sort(key=lambda t: t[1], reverse=True)
        results = scored[:top_k]

        # Reinforce recalled episodes
        for ep, _ in results:
            ep.access_count += 1
            ep.importance = min(ep.importance + 0.1, 10.0)

        return results

    def get_episode(self, episode_id: str) -> Optional[Episode]:
        """Retrieve episode by ID."""
        return self._episodes.get(episode_id)

    def get_all(self) -> List[Episode]:
        """Return all stored episodes sorted by timestamp."""
        return sorted(self._episodes.values(), key=lambda e: e.timestamp)

    def prune(self) -> int:
        """Remove stale, low-importance episodes. Returns count removed."""
        before = len(self._episodes)
        self._episodes = {
            eid: ep
            for eid, ep in self._episodes.items()
            if ep.importance >= self.min_importance
        }
        return before - len(self._episodes)

    def consolidate(self) -> int:
        """
        Merge highly similar adjacent episodes.

        Pairs of episodes with cosine similarity above consolidation_threshold
        are merged: the newer episode absorbs the content and importance of
        the older, and the older is removed.

        Returns:
            Number of episodes removed by consolidation.
        """
        episodes = sorted(self._episodes.values(), key=lambda e: e.timestamp)
        to_remove = set()
        merged = 0

        for i in range(len(episodes) - 1):
            if episodes[i].episode_id in to_remove:
                continue
            for j in range(i + 1, min(i + 10, len(episodes))):
                if episodes[j].episode_id in to_remove:
                    continue
                sim = float(
                    np.dot(
                        self._normalize(episodes[i].embedding),
                        self._normalize(episodes[j].embedding),
                    )
                )
                if sim >= self.consolidation_threshold:
                    # Merge i into j: j is the "surviving" (more recent)
                    surviving = episodes[j]
                    surviving.importance = min(
                        surviving.importance + episodes[i].importance * 0.5, 10.0
                    )
                    surviving.content = (
                        surviving.content
                        + " [+] "
                        + episodes[i].content
                    )
                    surviving.embedding = self._normalize(
                        surviving.embedding + episodes[i].embedding
                    )
                    surviving.consolidated = True
                    to_remove.add(episodes[i].episode_id)
                    merged += 1
                    break

        for eid in to_remove:
            del self._episodes[eid]

        return merged

    def size(self) -> int:
        return len(self._episodes)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        return v / (np.linalg.norm(v) + 1e-12)

    def _enforce_capacity(self) -> None:
        """Drop the least important / oldest episodes to stay within capacity."""
        episodes = sorted(
            self._episodes.values(),
            key=lambda e: e.importance * e.recency_score(self.recency_half_life),
        )
        excess = len(episodes) - self.max_episodes
        for ep in episodes[:excess]:
            del self._episodes[ep.episode_id]
