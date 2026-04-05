"""Tests for TokenFormer."""

import numpy as np
import pytest

from gostaan.tokenformer.tokenformer import TokenFormer, TokenFormerLayer


def _rand(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape)


class TestTokenFormerLayer:
    def test_output_shape(self):
        layer = TokenFormerLayer(
            dim_model=32, dim_ff=64, num_heads=2, num_param_tokens=4, dim_position=16, seed=0
        )
        x = _rand((6, 32))
        out = layer.forward(x)
        assert out.shape == (6, 32)

    def test_no_nan(self):
        layer = TokenFormerLayer(
            dim_model=32, dim_ff=64, num_heads=2, num_param_tokens=4, dim_position=16, seed=1
        )
        x = _rand((5, 32))
        out = layer.forward(x)
        assert not np.any(np.isnan(out))

    def test_param_tokens_not_in_output(self):
        num_param = 8
        layer = TokenFormerLayer(
            dim_model=16, dim_ff=32, num_heads=2, num_param_tokens=num_param,
            dim_position=8, seed=2
        )
        x = _rand((4, 16))
        out = layer.forward(x)
        # Output should have same seq_len as input
        assert out.shape[0] == x.shape[0]


class TestTokenFormer:
    def test_encode_shape(self):
        tf = TokenFormer(dim_model=32, dim_ff=64, num_layers=2, num_heads=2,
                         num_param_tokens=4, dim_position=16, seed=0)
        x = _rand((8, 32))
        out = tf.encode(x)
        assert out.shape == (8, 32)

    def test_pool_shape(self):
        tf = TokenFormer(dim_model=32, dim_ff=64, num_layers=2, num_heads=2,
                         num_param_tokens=4, dim_position=16, seed=0)
        x = _rand((6, 32))
        pooled = tf.pool(x)
        assert pooled.shape == (32,)

    def test_encode_batch(self):
        tf = TokenFormer(dim_model=32, dim_ff=64, num_layers=2, num_heads=2,
                         num_param_tokens=4, dim_position=16, seed=0)
        batch = _rand((3, 5, 32))
        out = tf.encode_batch(batch)
        assert out.shape == (3, 5, 32)

    def test_single_layer(self):
        tf = TokenFormer(dim_model=16, num_layers=1, num_heads=2, seed=5)
        x = _rand((3, 16))
        out = tf.encode(x)
        assert out.shape == (3, 16)

    def test_no_nan(self):
        tf = TokenFormer(dim_model=32, num_layers=2, num_heads=2, seed=3)
        x = _rand((10, 32))
        out = tf.encode(x)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))
