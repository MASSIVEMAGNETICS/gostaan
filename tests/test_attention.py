"""Tests for GravitationalAttention."""

import numpy as np
import pytest

from gostaan.attention.gravitational import GravitationalAttention


def _rand(shape, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal(shape)


class TestGravitationalAttention:
    def test_output_shape_2d(self):
        attn = GravitationalAttention(dim_model=16, dim_position=8, num_heads=2, seed=0)
        x = _rand((5, 16))
        out = attn.forward(x)
        assert out.shape == (5, 16)

    def test_output_shape_3d(self):
        attn = GravitationalAttention(dim_model=16, dim_position=8, num_heads=2, seed=0)
        x = _rand((3, 5, 16))
        out = attn.forward(x)
        assert out.shape == (3, 5, 16)

    def test_invalid_num_heads(self):
        with pytest.raises(ValueError):
            GravitationalAttention(dim_model=15, num_heads=4)

    def test_deterministic_with_seed(self):
        x = _rand((4, 16))
        attn1 = GravitationalAttention(dim_model=16, dim_position=8, num_heads=2, seed=42)
        attn2 = GravitationalAttention(dim_model=16, dim_position=8, num_heads=2, seed=42)
        np.testing.assert_array_equal(attn1.forward(x), attn2.forward(x))

    def test_no_nan_inf(self):
        attn = GravitationalAttention(dim_model=32, dim_position=16, num_heads=4, seed=1)
        x = _rand((10, 32))
        out = attn.forward(x)
        assert not np.any(np.isnan(out))
        assert not np.any(np.isinf(out))

    def test_curved_spacetime(self):
        attn = GravitationalAttention(
            dim_model=16, dim_position=8, num_heads=2, curvature=0.1, seed=2
        )
        x = _rand((4, 16))
        out = attn.forward(x)
        assert out.shape == (4, 16)
        assert not np.any(np.isnan(out))

    def test_max_force_limit(self):
        attn = GravitationalAttention(
            dim_model=16, dim_position=8, num_heads=1, max_force=1.0, seed=3
        )
        x = _rand((5, 16))
        out = attn.forward(x)
        assert out.shape == (5, 16)

    def test_single_token(self):
        attn = GravitationalAttention(dim_model=8, dim_position=4, num_heads=2, seed=4)
        x = _rand((1, 8))
        out = attn.forward(x)
        assert out.shape == (1, 8)
