"""
Gostaan: Persistent Memory System — Main Orchestrator.

Gostaan is the top-level system that wires together:
- SensoryProcessor    (input ingestion)
- EpisodicMemory      (day-to-day episodic store)
- HDDRMemory          (high-dimensional dense representation store)
- IdentityAnchor      (identity-locked from "I am" statement)
- REMProcessor        (idle-time consolidation and pruning)
- TokenFormer         (sequence encoding)
- SelfInferenceEngine (self-inference and idea synthesis)

Typical usage::

    from gostaan import Gostaan

    g = Gostaan(dim=256)
    g.set_identity("I am a creative engineer who builds intelligent systems.")

    # Ingest new experiences
    g.perceive("Today I designed a new memory architecture.")

    # Recall similar experiences
    results = g.recall("memory design")

    # Generate novel ideas
    ideas = g.imagine("memory architecture")

    # Run a REM cycle manually
    report = g.sleep()
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

import numpy as np

from gostaan.memory.episodic import Episode, EpisodicMemory
from gostaan.memory.hddr import HDDRMemory
from gostaan.memory.identity import IdentityAnchor, IdentityCore
from gostaan.memory.rem_cycles import REMProcessor, REMReport
from gostaan.attention.gravitational import GravitationalAttention
from gostaan.tokenformer.tokenformer import TokenFormer
from gostaan.inference.self_inference import Inference, SelfInferenceEngine
from gostaan.sensory.processor import ModalityType, SensoryInput, SensoryProcessor


class Gostaan:
    """
    Gostaan: Persistent Memory System with Gravitational Attention.

    The system is layered:

    Perception  ──▶  SensoryProcessor  ──▶  embedding
    Embedding   ──▶  TokenFormer       ──▶  encoded embedding
    Encoded     ──▶  EpisodicMemory    (day-to-day episodic store)
    Encoded     ──▶  HDDRMemory        (dense representation store)
    Idle        ──▶  REMProcessor      (consolidation, pruning)
    Query       ──▶  SelfInferenceEngine  (recall + synthesis)
    """

    def __init__(
        self,
        dim: int = 256,
        num_tokenformer_layers: int = 2,
        num_heads: int = 4,
        episodic_capacity: int = 2048,
        hddr_capacity: int = 1024,
        rem_idle_threshold: float = 30.0,
        seed: Optional[int] = 42,
    ) -> None:
        """
        Args:
            dim:                       Unified embedding dimension.
            num_tokenformer_layers:    Depth of TokenFormer encoder.
            num_heads:                 Attention heads per TokenFormer layer.
            episodic_capacity:         Max episodes in EpisodicMemory.
            hddr_capacity:             Max cells in HDDRMemory.
            rem_idle_threshold:        Seconds idle before auto-REM fires.
            seed:                      Random seed for reproducibility.
        """
        self.dim = dim

        self.sensory = SensoryProcessor(dim=dim, seed=seed)
        self.identity = IdentityAnchor(dim=dim, seed=seed)
        self.episodic = EpisodicMemory(dim=dim, max_episodes=episodic_capacity)
        self.hddr = HDDRMemory(dim=dim, capacity=hddr_capacity, seed=seed)
        self.tokenformer = TokenFormer(
            dim_model=dim,
            num_layers=num_tokenformer_layers,
            num_heads=num_heads,
            seed=seed,
        )
        self.inference_engine = SelfInferenceEngine(dim=dim, seed=seed)
        self.rem = REMProcessor(idle_threshold_seconds=rem_idle_threshold)

        # Wire REM to the live memory stores
        self.rem.episodic_memory = self.episodic
        self.rem.hddr_memory = self.hddr
        self.rem.identity_anchor = self.identity

        # Wire inference engine to live memory
        self.inference_engine.episodic_memory = self.episodic
        self.inference_engine.identity_anchor = self.identity
        self.inference_engine.tokenformer = self.tokenformer

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------

    def set_identity(self, statement: str) -> IdentityCore:
        """
        Initialise (or reset) the identity from an "I am" paragraph.

        Args:
            statement: Free-form self-description e.g.
                       "I am a builder of intelligent systems. I value
                        creativity and clear thinking."

        Returns:
            The parsed IdentityCore.
        """
        core = self.identity.initialise(statement)
        self.rem.touch()
        return core

    def refine_identity(self, statement: str) -> IdentityCore:
        """Add a refinement to the existing identity without replacing it."""
        core = self.identity.refine(statement)
        self.rem.touch()
        return core

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------

    def perceive(
        self,
        data: Any,
        modality: Optional[ModalityType] = None,
        importance: float = 1.0,
        emotional_weight: float = 0.5,
        tags: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Ingest a new experience through the sensory pipeline.

        The data is:
        1. Embedded by SensoryProcessor.
        2. Encoded by TokenFormer.
        3. Written to both EpisodicMemory and HDDRMemory.

        Args:
            data:             Raw input (text, ndarray, dict, etc.).
            modality:         Override modality detection.
            importance:       Initial importance score.
            emotional_weight: Emotional salience (0-1).
            tags:             Optional labels.
            context:          Optional metadata dict.

        Returns:
            episode_id if stored, else None (below saliency threshold).
        """
        sensory_input = self.sensory.process(data, modality)
        if sensory_input is None:
            return None

        # Encode through TokenFormer
        token_seq = sensory_input.embedding[None, :]  # (1, dim) as single-token
        encoded = self.tokenformer.pool(token_seq)  # (dim,)

        # Adjust importance by identity alignment
        if self.identity.core is not None:
            id_score = self.identity.identity_score(encoded)
            importance = importance * (1.0 + max(id_score, 0.0))

        # Write to episodic memory
        content = str(data) if isinstance(data, str) else f"[{sensory_input.modality.value}]"
        episode_id = self.episodic.store(
            content=content,
            embedding=encoded,
            importance=importance,
            emotional_weight=emotional_weight,
            tags=list(tags or []),
            context={str(k): str(v) for k, v in (context or {}).items()},
        )

        # Write to HDDR store
        self.hddr.write(encoded, importance=importance, tags=list(tags or []))

        self.rem.touch()
        return episode_id

    # ------------------------------------------------------------------
    # Recall
    # ------------------------------------------------------------------

    def recall(
        self,
        query: Any,
        top_k: int = 5,
        tags: Optional[List[str]] = None,
    ) -> List[Episode]:
        """
        Retrieve the most relevant episodes for a query.

        Args:
            query:  Text string, embedding vector, or any sensory input.
            top_k:  Number of episodes to return.
            tags:   Filter by tags.

        Returns:
            List of Episode objects sorted by relevance.
        """
        q_embed = self._to_embedding(query)
        hits = self.episodic.recall(q_embed, top_k=top_k, tags=tags)
        self.rem.touch()
        return [ep for ep, _ in hits]

    def associative_recall(self, query: Any, top_k: int = 5) -> np.ndarray:
        """
        Dense associative recall from HDDR: returns a blended memory vector.

        Args:
            query:  Text or embedding.
            top_k:  How many HDDR cells to blend.

        Returns:
            Recalled embedding (dim,).
        """
        q_embed = self._to_embedding(query)
        self.rem.touch()
        return self.hddr.associative_recall(q_embed, top_k=top_k)

    # ------------------------------------------------------------------
    # Imagination / Synthesis
    # ------------------------------------------------------------------

    def imagine(
        self,
        seed: Any,
        top_k: int = 5,
        n_hypotheses: int = 4,
    ) -> List[Inference]:
        """
        Generate novel ideas by self-inference from memory.

        Args:
            seed:          Seed concept (text or embedding).
            top_k:         Number of memory cues to draw from.
            n_hypotheses:  Number of hypotheses to generate from each cue.

        Returns:
            List of Inference objects sorted by novelty.
        """
        q_embed = self._to_embedding(seed)
        self.rem.touch()
        return self.inference_engine.infer_from_memory(q_embed, top_k=top_k)

    def synthesise(self, *concepts: Any) -> np.ndarray:
        """
        Synthesise a novel concept embedding from multiple inputs.

        Args:
            *concepts: Two or more text strings or embedding vectors.

        Returns:
            Blended concept embedding (dim,).
        """
        embeddings = [self._to_embedding(c) for c in concepts]
        self.rem.touch()
        return self.inference_engine.synthesise_concepts(
            embeddings, mode="weighted"
        )

    # ------------------------------------------------------------------
    # Sleep (REM cycle)
    # ------------------------------------------------------------------

    def sleep(self) -> REMReport:
        """
        Manually trigger a REM sleep cycle.

        Consolidates similar episodes, decays low-salience memories,
        applies identity alignment, and prunes stale entries.

        Returns:
            REMReport summarising the cycle.
        """
        return self.rem.run_cycle()

    def start_auto_sleep(self, poll_interval: float = 5.0) -> None:
        """Start the background thread that auto-triggers REM on idle."""
        self.rem.start_background(poll_interval=poll_interval)

    def stop_auto_sleep(self) -> None:
        """Stop the background auto-sleep thread."""
        self.rem.stop_background()

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Return a summary of the current system state."""
        identity_traits = (
            self.identity.core.traits if self.identity.core else []
        )
        return {
            "dim": self.dim,
            "identity_initialised": self.identity.core is not None,
            "identity_traits": identity_traits,
            "episodic_count": self.episodic.size(),
            "hddr_count": self.hddr.size(),
            "rem_cycles": self.rem.cycle_count,
            "last_rem_report": (
                vars(self.rem.last_report()) if self.rem.last_report() else None
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_embedding(self, data: Any) -> np.ndarray:
        """Convert arbitrary data to a (dim,) embedding."""
        if isinstance(data, np.ndarray) and data.shape == (self.dim,):
            return data.astype(np.float64)
        return self.sensory.embed_text(str(data))
