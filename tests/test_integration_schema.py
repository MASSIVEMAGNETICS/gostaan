"""
Unit tests for the integration schema and security primitives.

These tests run without binding any network ports.
"""

from __future__ import annotations

import json

import pytest

from gostaan.integration.schema import PlatformEvent, SCHEMA_VERSION, EVENT_TYPES
from gostaan.integration.security import RateLimiter, SecurityPolicy


# ---------------------------------------------------------------------------
# PlatformEvent
# ---------------------------------------------------------------------------


class TestPlatformEvent:
    def test_defaults_populated(self):
        ev = PlatformEvent(type="perceive", source="test")
        assert ev.version == SCHEMA_VERSION
        assert ev.id  # non-empty UUID
        assert ev.timestamp  # non-empty ISO string
        assert ev.payload == {}
        assert ev.trace_id is None

    def test_to_dict_contains_all_fields(self):
        ev = PlatformEvent(type="recall", source="sys-a", payload={"query": "memory"})
        d = ev.to_dict()
        assert d["type"] == "recall"
        assert d["source"] == "sys-a"
        assert d["payload"]["query"] == "memory"
        assert "id" in d
        assert "version" in d
        assert "timestamp" in d

    def test_json_roundtrip(self):
        ev = PlatformEvent(
            type="heartbeat",
            source="gostaan",
            payload={"key": "val"},
            trace_id="req-abc",
        )
        recovered = PlatformEvent.from_json(ev.to_json())
        assert recovered.id == ev.id
        assert recovered.type == ev.type
        assert recovered.source == ev.source
        assert recovered.payload == ev.payload
        assert recovered.trace_id == ev.trace_id

    def test_from_dict_basic(self):
        data = {
            "id": "abc-123",
            "version": "1.0.0",
            "type": "sleep",
            "source": "system-a",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "payload": {},
            "trace_id": None,
        }
        ev = PlatformEvent.from_dict(data)
        assert ev.id == "abc-123"
        assert ev.type == "sleep"
        assert ev.source == "system-a"

    def test_from_dict_ignores_unknown_fields(self):
        data = {
            "id": "x",
            "version": "1.0.0",
            "type": "perceive",
            "source": "a",
            "timestamp": "2026-01-01T00:00:00Z",
            "payload": {},
            "extra_future_field": "should be ignored",
        }
        ev = PlatformEvent.from_dict(data)
        assert ev.type == "perceive"

    def test_validate_all_known_types(self):
        for t in EVENT_TYPES:
            ev = PlatformEvent(type=t, source="x")
            ev.validate()  # must not raise

    def test_validate_unknown_type_raises(self):
        ev = PlatformEvent(type="unknown_type", source="x")
        with pytest.raises(ValueError, match="Unknown event type"):
            ev.validate()

    def test_validate_empty_source_raises(self):
        ev = PlatformEvent(type="perceive", source="")
        with pytest.raises(ValueError, match="source"):
            ev.validate()

    def test_trace_id_preserved(self):
        ev = PlatformEvent(type="result", source="g", trace_id="orig-999")
        d = ev.to_dict()
        assert d["trace_id"] == "orig-999"
        recovered = PlatformEvent.from_dict(d)
        assert recovered.trace_id == "orig-999"

    def test_from_json_invalid_raises(self):
        with pytest.raises((json.JSONDecodeError, Exception)):
            PlatformEvent.from_json("not json {{{")

    def test_unique_ids(self):
        ids = {PlatformEvent(type="heartbeat", source="x").id for _ in range(50)}
        assert len(ids) == 50


# ---------------------------------------------------------------------------
# RateLimiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_allows_within_limit(self):
        rl = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert rl.allow("client-a") is True

    def test_blocks_over_limit(self):
        rl = RateLimiter(max_requests=3, window_seconds=60)
        for _ in range(3):
            rl.allow("client-b")
        assert rl.allow("client-b") is False

    def test_different_clients_independent(self):
        rl = RateLimiter(max_requests=2, window_seconds=60)
        rl.allow("c1")
        rl.allow("c1")
        assert rl.allow("c1") is False
        # c2 has its own bucket
        assert rl.allow("c2") is True

    def test_zero_limit_always_blocks(self):
        rl = RateLimiter(max_requests=0, window_seconds=60)
        assert rl.allow("any") is False


# ---------------------------------------------------------------------------
# SecurityPolicy
# ---------------------------------------------------------------------------


class TestSecurityPolicy:
    def test_no_tokens_configured_allows_all(self):
        policy = SecurityPolicy(require_auth=True)
        # No tokens configured → open server
        assert policy.check_token("any-token") is True
        assert policy.check_token(None) is True

    def test_with_tokens_rejects_bad_token(self):
        policy = SecurityPolicy(auth_tokens={"good"}, require_auth=True)
        assert policy.check_token("good") is True
        assert policy.check_token("bad") is False
        assert policy.check_token(None) is False

    def test_require_auth_false_bypasses_check(self):
        policy = SecurityPolicy(auth_tokens={"secret"}, require_auth=False)
        assert policy.check_token(None) is True
        assert policy.check_token("wrong") is True

    def test_no_origin_allowlist_accepts_any(self):
        policy = SecurityPolicy()
        assert policy.check_origin(None) is True
        assert policy.check_origin("http://anywhere.com") is True

    def test_origin_allowlist_rejects_unknown(self):
        policy = SecurityPolicy(allowed_origins={"http://trusted.internal"})
        assert policy.check_origin("http://trusted.internal") is True
        assert policy.check_origin("http://evil.com") is False
        assert policy.check_origin(None) is False

    def test_extract_bearer_token(self):
        policy = SecurityPolicy()
        assert policy.extract_token("Bearer abc123") == "abc123"
        assert policy.extract_token("abc123") == "abc123"
        assert policy.extract_token(None) is None

    def test_add_token(self):
        policy = SecurityPolicy(require_auth=True)
        policy.add_token("new-token")
        assert policy.check_token("new-token") is True

    def test_add_origin(self):
        policy = SecurityPolicy(allowed_origins={"http://a.com"})
        policy.add_origin("http://b.com")
        assert policy.check_origin("http://b.com") is True

    def test_rate_limit_integration(self):
        rl = RateLimiter(max_requests=1, window_seconds=60)
        policy = SecurityPolicy(rate_limiter=rl)
        assert policy.check_rate("ip-1") is True
        assert policy.check_rate("ip-1") is False
