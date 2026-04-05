"""Tests for SensoryProcessor."""

import numpy as np
import pytest

from gostaan.sensory.processor import SensoryProcessor, ModalityType, SensoryInput


class TestSensoryProcessor:
    def test_process_text_returns_sensory_input(self):
        sp = SensoryProcessor(dim=64, saliency_threshold=0.0, seed=0)
        result = sp.process("Hello world this is a test input sentence")
        assert result is not None
        assert isinstance(result, SensoryInput)
        assert result.embedding.shape == (64,)
        assert result.modality == ModalityType.TEXT

    def test_process_numeric_array(self):
        sp = SensoryProcessor(dim=64, saliency_threshold=0.0, seed=0)
        arr = np.random.default_rng(1).standard_normal(128)
        result = sp.process(arr, ModalityType.NUMERIC)
        assert result is not None
        assert result.embedding.shape == (64,)

    def test_process_event_dict(self):
        sp = SensoryProcessor(dim=64, saliency_threshold=0.0, seed=0)
        event = {"action": "click", "target": "button", "value": "42"}
        result = sp.process(event, ModalityType.EVENT)
        assert result is not None
        assert result.embedding.shape == (64,)

    def test_saliency_threshold_filters(self):
        sp = SensoryProcessor(dim=64, saliency_threshold=1e9, seed=0)
        result = sp.process("some text")
        assert result is None

    def test_embed_text_shape(self):
        sp = SensoryProcessor(dim=128, seed=0)
        emb = sp.embed_text("the quick brown fox")
        assert emb.shape == (128,)

    def test_embed_text_normalised(self):
        sp = SensoryProcessor(dim=64, seed=0)
        emb = sp.embed_text("test sentence")
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-6

    def test_chunk_text_short(self):
        sp = SensoryProcessor(dim=64, chunk_size=10, seed=0)
        text = "short text"
        chunks = sp.chunk_text(text)
        assert chunks == [text]

    def test_chunk_text_long(self):
        sp = SensoryProcessor(dim=64, chunk_size=4, chunk_overlap=1, seed=0)
        text = "word1 word2 word3 word4 word5 word6 word7 word8"
        chunks = sp.chunk_text(text)
        assert len(chunks) > 1
        assert all(len(c.split()) <= 4 for c in chunks)

    def test_process_text_chunks(self):
        sp = SensoryProcessor(dim=32, chunk_size=3, chunk_overlap=1,
                               saliency_threshold=0.0, seed=0)
        long_text = " ".join(f"word{i}" for i in range(20))
        results = sp.process_text_chunks(long_text)
        assert len(results) > 1

    def test_modality_auto_detect(self):
        sp = SensoryProcessor(dim=32, saliency_threshold=0.0, seed=0)
        # str → TEXT
        r = sp.process("hello")
        assert r.modality == ModalityType.TEXT
        # numpy → NUMERIC
        r2 = sp.process(np.zeros(10))
        assert r2.modality == ModalityType.NUMERIC
        # dict → EVENT
        r3 = sp.process({"key": "val"})
        assert r3.modality == ModalityType.EVENT

    def test_different_texts_differ(self):
        sp = SensoryProcessor(dim=64, seed=0)
        e1 = sp.embed_text("memory architecture")
        e2 = sp.embed_text("quantum physics")
        assert not np.allclose(e1, e2)
