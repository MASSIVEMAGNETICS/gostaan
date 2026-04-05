"""End-to-end tests for the Gostaan core orchestrator."""

import time
import numpy as np
import pytest

from gostaan import Gostaan


IDENTITY_STATEMENT = (
    "I am a builder of intelligent systems. "
    "I value creativity and precision. "
    "I believe persistent memory is fundamental to intelligence. "
    "I love designing elegant architectures."
)


class TestGostaanCore:
    @pytest.fixture
    def g(self):
        return Gostaan(
            dim=64,
            num_tokenformer_layers=1,
            num_heads=2,
            episodic_capacity=100,
            hddr_capacity=50,
            seed=42,
        )

    def test_set_identity(self, g):
        core = g.set_identity(IDENTITY_STATEMENT)
        assert core is not None
        assert len(core.traits) > 0

    def test_perceive_returns_episode_id(self, g):
        g.set_identity(IDENTITY_STATEMENT)
        eid = g.perceive("Today I designed a new memory architecture.")
        assert eid is not None
        assert isinstance(eid, str)

    def test_perceive_and_recall(self, g):
        g.set_identity(IDENTITY_STATEMENT)
        g.perceive("I built a gravitational attention layer.")
        g.perceive("The tokenformer encodes sequences beautifully.")
        g.perceive("REM cycles consolidate memories during idle time.")
        results = g.recall("attention mechanism")
        assert len(results) > 0

    def test_associative_recall_shape(self, g):
        g.set_identity(IDENTITY_STATEMENT)
        g.perceive("Memory architecture experiment.")
        g.perceive("Gravitational forces in semantic space.")
        recalled = g.associative_recall("memory")
        assert recalled.shape == (64,)

    def test_imagine_returns_inferences(self, g):
        g.set_identity(IDENTITY_STATEMENT)
        for i in range(5):
            g.perceive(f"Experiment {i} with intelligent systems and memory.")
        ideas = g.imagine("intelligent memory", top_k=3)
        assert isinstance(ideas, list)

    def test_synthesise(self, g):
        g.set_identity(IDENTITY_STATEMENT)
        blended = g.synthesise("memory architecture", "gravitational attention")
        assert blended.shape == (64,)

    def test_sleep_returns_report(self, g):
        g.set_identity(IDENTITY_STATEMENT)
        for i in range(5):
            g.perceive(f"Experience {i}")
        report = g.sleep()
        assert report is not None
        assert report.cycle_id == 1

    def test_sleep_prunes_on_repeat(self, g):
        g.set_identity(IDENTITY_STATEMENT)
        # Store an episode with very low importance, then sleep to prune it
        vec = np.random.default_rng(0).standard_normal(64)
        eid = g.episodic.store("low-importance", vec, importance=0.001)
        before = g.episodic.size()
        g.sleep()
        after = g.episodic.size()
        assert after <= before

    def test_status(self, g):
        g.set_identity(IDENTITY_STATEMENT)
        g.perceive("Test experience for status check.")
        status = g.status()
        assert status["identity_initialised"] is True
        assert status["episodic_count"] >= 1
        assert "identity_traits" in status

    def test_refine_identity(self, g):
        g.set_identity(IDENTITY_STATEMENT)
        v0 = g.identity.core.version
        g.refine_identity("I love exploration and discovery.")
        assert g.identity.core.version == v0 + 1

    def test_multiple_perceive_recall(self, g):
        g.set_identity(IDENTITY_STATEMENT)
        texts = [
            "Gravitational attention replaces dot-product attention.",
            "Token-former uses parameter tokens for persistent knowledge.",
            "Episodic memory stores day-to-day experiences.",
            "REM sleep consolidates and prunes memories.",
            "Identity anchoring ensures memory coherence.",
        ]
        for t in texts:
            g.perceive(t)
        assert g.episodic.size() == len(texts)
        results = g.recall("episodic memory")
        assert len(results) > 0

    def test_no_identity_perceive(self, g):
        """Perceive without identity should still work."""
        eid = g.perceive("Experience without identity.")
        assert eid is not None

    def test_perceive_with_tags(self, g):
        g.set_identity(IDENTITY_STATEMENT)
        eid = g.perceive("Tagged experience.", tags=["work", "memory"])
        results = g.recall("experience", tags=["work"])
        assert any(ep.episode_id == eid for ep in results)
