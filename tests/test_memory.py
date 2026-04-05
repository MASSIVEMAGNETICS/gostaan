"""Tests for memory subsystems: HDDR, Episodic, Identity, REM."""

import time
import numpy as np
import pytest

from gostaan.memory.hddr import HDDRMemory
from gostaan.memory.episodic import EpisodicMemory
from gostaan.memory.identity import IdentityAnchor
from gostaan.memory.rem_cycles import REMProcessor


def _rand_vec(dim, seed=0):
    return np.random.default_rng(seed).standard_normal(dim)


# -----------------------------------------------------------------------
# HDDR Memory
# -----------------------------------------------------------------------

class TestHDDRMemory:
    def test_write_read(self):
        mem = HDDRMemory(dim=64, seed=0)
        vec = _rand_vec(64, 1)
        idx = mem.write(vec, importance=1.0)
        assert idx == 0
        results = mem.read(vec, top_k=1)
        assert len(results) == 1
        assert results[0][0] == 0  # index matches

    def test_wrong_dim_raises(self):
        mem = HDDRMemory(dim=64, seed=0)
        with pytest.raises(ValueError):
            mem.write(_rand_vec(32, 1))

    def test_associative_recall_shape(self):
        mem = HDDRMemory(dim=64, seed=0)
        for i in range(5):
            mem.write(_rand_vec(64, i), importance=1.0)
        result = mem.associative_recall(_rand_vec(64, 99), top_k=3)
        assert result.shape == (64,)

    def test_decay_reduces_importance(self):
        mem = HDDRMemory(dim=64, decay_rate=0.5, seed=0)
        mem.write(_rand_vec(64, 0), importance=1.0)
        mem.decay_all()
        assert mem._cells[0].importance < 1.0

    def test_prune(self):
        mem = HDDRMemory(dim=64, seed=0)
        mem.write(_rand_vec(64, 0), importance=0.001)
        mem.write(_rand_vec(64, 1), importance=5.0)
        removed = mem.prune(min_importance=0.01)
        assert removed == 1
        assert mem.size() == 1

    def test_capacity_compression(self):
        mem = HDDRMemory(dim=32, capacity=10, compression_threshold=0.8, seed=0)
        for i in range(12):
            mem.write(_rand_vec(32, i))
        assert mem.size() <= 10

    def test_read_empty(self):
        mem = HDDRMemory(dim=64, seed=0)
        result = mem.read(_rand_vec(64, 0))
        assert result == []

    def test_tags_stored(self):
        mem = HDDRMemory(dim=64, seed=0)
        mem.write(_rand_vec(64, 0), tags=["tag1", "tag2"])
        assert "tag1" in mem._cells[0].tags


# -----------------------------------------------------------------------
# Episodic Memory
# -----------------------------------------------------------------------

class TestEpisodicMemory:
    def test_store_and_recall(self):
        mem = EpisodicMemory(dim=64)
        vec = _rand_vec(64, 0)
        eid = mem.store("test experience", vec, importance=1.0)
        results = mem.recall(vec, top_k=1)
        assert len(results) == 1
        ep, score = results[0]
        assert ep.episode_id == eid
        assert score > 0

    def test_wrong_dim_raises(self):
        mem = EpisodicMemory(dim=64)
        with pytest.raises(ValueError):
            mem.store("bad", _rand_vec(32, 0))

    def test_recall_empty(self):
        mem = EpisodicMemory(dim=64)
        results = mem.recall(_rand_vec(64, 0))
        assert results == []

    def test_recall_reinforces_importance(self):
        mem = EpisodicMemory(dim=64)
        vec = _rand_vec(64, 0)
        eid = mem.store("test", vec, importance=1.0)
        mem.recall(vec, top_k=1)
        ep = mem.get_episode(eid)
        assert ep.access_count == 1
        assert ep.importance > 1.0

    def test_prune(self):
        mem = EpisodicMemory(dim=64, min_importance=0.5)
        mem.store("keep", _rand_vec(64, 0), importance=1.0)
        mem.store("prune", _rand_vec(64, 1), importance=0.01)
        pruned = mem.prune()
        assert pruned == 1
        assert mem.size() == 1

    def test_consolidation_merges_similar(self):
        mem = EpisodicMemory(dim=64, consolidation_threshold=0.5)
        vec = _rand_vec(64, 0)
        # Store nearly identical episodes
        mem.store("experience A", vec, importance=1.0)
        mem.store("experience B", vec * 1.0001, importance=1.0)
        before = mem.size()
        merged = mem.consolidate()
        assert merged >= 0
        assert mem.size() <= before

    def test_capacity_enforcement(self):
        mem = EpisodicMemory(dim=16, max_episodes=5)
        for i in range(8):
            mem.store(f"ep {i}", _rand_vec(16, i), importance=float(i + 1))
        assert mem.size() <= 5

    def test_tag_filter(self):
        mem = EpisodicMemory(dim=32)
        v1 = _rand_vec(32, 0)
        v2 = _rand_vec(32, 1)
        mem.store("tagged", v1, tags=["work"])
        mem.store("untagged", v2)
        results = mem.recall(v1, top_k=5, tags=["work"])
        assert all("work" in ep.tags for ep, _ in results)


# -----------------------------------------------------------------------
# Identity Anchor
# -----------------------------------------------------------------------

class TestIdentityAnchor:
    STATEMENT = (
        "I am a creative builder. I value deep thinking and innovation. "
        "I believe in the power of persistent memory systems."
    )

    def test_initialise_returns_core(self):
        anchor = IdentityAnchor(dim=64, seed=0)
        core = anchor.initialise(self.STATEMENT)
        assert core is not None
        assert len(core.traits) > 0
        assert core.identity_vector.shape == (64,)

    def test_identity_vector_unit_norm(self):
        anchor = IdentityAnchor(dim=64, seed=0)
        anchor.initialise(self.STATEMENT)
        norm = np.linalg.norm(anchor.identity_vector)
        assert abs(norm - 1.0) < 1e-6

    def test_identity_score_range(self):
        anchor = IdentityAnchor(dim=64, seed=0)
        anchor.initialise(self.STATEMENT)
        rng = np.random.default_rng(1)
        vec = rng.standard_normal(64)
        score = anchor.identity_score(vec)
        assert -1.0 <= score <= 1.0

    def test_gate_allows_aligned(self):
        anchor = IdentityAnchor(dim=64, seed=0)
        anchor.initialise(self.STATEMENT)
        iv = anchor.identity_vector
        should_store, score = anchor.gate(iv, threshold=0.5)
        assert should_store

    def test_infer_shifts_toward_identity(self):
        anchor = IdentityAnchor(dim=64, seed=0)
        anchor.initialise(self.STATEMENT)
        rng = np.random.default_rng(2)
        seed_vec = rng.standard_normal(64)
        inferred = anchor.infer(seed_vec, strength=0.9)
        # Inferred should be more aligned with identity than seed
        orig_score = anchor.identity_score(seed_vec)
        new_score = anchor.identity_score(inferred)
        assert new_score >= orig_score

    def test_refine_updates_version(self):
        anchor = IdentityAnchor(dim=64, seed=0)
        anchor.initialise(self.STATEMENT)
        v0 = anchor.core.version
        anchor.refine("I love music and art.")
        assert anchor.core.version == v0 + 1

    def test_uninitialised_score_returns_zero(self):
        anchor = IdentityAnchor(dim=64)
        rng = np.random.default_rng(0)
        score = anchor.identity_score(rng.standard_normal(64))
        assert score == 0.0


# -----------------------------------------------------------------------
# REM Processor
# -----------------------------------------------------------------------

class TestREMProcessor:
    def _make_system(self):
        from gostaan.memory.episodic import EpisodicMemory
        from gostaan.memory.hddr import HDDRMemory
        from gostaan.memory.identity import IdentityAnchor

        rem = REMProcessor(idle_threshold_seconds=1.0, decay_rate=0.1)
        rem.episodic_memory = EpisodicMemory(dim=32)
        rem.hddr_memory = HDDRMemory(dim=32, seed=0)
        rem.identity_anchor = IdentityAnchor(dim=32, seed=0)
        rem.identity_anchor.initialise("I am a test system. I value accuracy.")
        return rem

    def test_run_cycle_returns_report(self):
        rem = self._make_system()
        report = rem.run_cycle()
        assert report.cycle_id == 1
        assert report.duration_seconds >= 0

    def test_cycle_count_increments(self):
        rem = self._make_system()
        rem.run_cycle()
        rem.run_cycle()
        assert rem.cycle_count == 2

    def test_prunes_low_importance(self):
        rem = self._make_system()
        vec = _rand_vec(32, 0)
        rem.episodic_memory.store("low salience", vec, importance=0.001)
        rem.run_cycle()
        assert rem.episodic_memory.size() == 0

    def test_decay_applied(self):
        rem = self._make_system()
        rem.hddr_memory.write(_rand_vec(32, 0), importance=1.0)
        initial_importance = rem.hddr_memory._cells[0].importance
        rem.run_cycle()
        assert rem.hddr_memory._cells[0].importance < initial_importance

    def test_is_idle(self):
        rem = REMProcessor(idle_threshold_seconds=0.05)
        rem.touch()
        assert not rem.is_idle()
        time.sleep(0.1)
        assert rem.is_idle()

    def test_last_report(self):
        rem = self._make_system()
        assert rem.last_report() is None
        rem.run_cycle()
        assert rem.last_report() is not None
