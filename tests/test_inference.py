"""Tests for SelfInferenceEngine."""

import numpy as np
import pytest

from gostaan.inference.self_inference import SelfInferenceEngine
from gostaan.memory.episodic import EpisodicMemory
from gostaan.memory.identity import IdentityAnchor


def _rand_vec(dim, seed=0):
    return np.random.default_rng(seed).standard_normal(dim)


def _make_engine(dim=64, n_episodes=5):
    mem = EpisodicMemory(dim=dim)
    anchor = IdentityAnchor(dim=dim, seed=0)
    anchor.initialise("I am a creative thinker. I love building new systems.")
    for i in range(n_episodes):
        vec = _rand_vec(dim, i)
        mem.store(f"experience {i}", vec, importance=float(i + 1))

    engine = SelfInferenceEngine(dim=dim, seed=0)
    engine.episodic_memory = mem
    engine.identity_anchor = anchor
    return engine


class TestSelfInferenceEngine:
    def test_infer_from_memory_returns_list(self):
        engine = _make_engine()
        q = _rand_vec(64, 99)
        results = engine.infer_from_memory(q, top_k=3)
        assert isinstance(results, list)

    def test_infer_from_memory_no_memory(self):
        engine = SelfInferenceEngine(dim=64)
        results = engine.infer_from_memory(_rand_vec(64, 0))
        assert results == []

    def test_synthesise_two_concepts(self):
        engine = _make_engine()
        a, b = _rand_vec(64, 0), _rand_vec(64, 1)
        result = engine.synthesise_concepts([a, b])
        assert result.shape == (64,)
        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < 1e-5

    def test_synthesise_weighted(self):
        engine = _make_engine()
        a, b = _rand_vec(64, 0), _rand_vec(64, 1)
        result = engine.synthesise_concepts([a, b], mode="weighted")
        assert result.shape == (64,)

    def test_analogical_infer(self):
        engine = _make_engine()
        a = _rand_vec(64, 0)
        b = _rand_vec(64, 1)
        c = _rand_vec(64, 2)
        result = engine.analogical_infer(a, b, c)
        assert result.shape == (64,)
        assert not np.any(np.isnan(result))

    def test_generate_hypotheses(self):
        engine = _make_engine()
        premise = _rand_vec(64, 5)
        hypotheses = engine.generate_hypotheses(premise, n=4)
        assert len(hypotheses) == 4
        for h in hypotheses:
            assert h.shape == (64,)

    def test_novelty_score_new_concept(self):
        engine = _make_engine(n_episodes=3)
        # A random vector should be somewhat novel
        v = _rand_vec(64, 999)
        score = engine.novelty_score_against_memory(v)
        assert 0.0 <= score <= 1.0

    def test_novelty_score_no_memory(self):
        engine = SelfInferenceEngine(dim=64)
        score = engine.novelty_score_against_memory(_rand_vec(64, 0))
        assert score == 1.0

    def test_synthesise_empty(self):
        engine = _make_engine()
        result = engine.synthesise_concepts([])
        assert result.shape == (64,)
        assert np.allclose(result, 0.0)
