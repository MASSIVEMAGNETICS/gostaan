"""
End-to-end tests for the integration server, client, and GostaanBridge.

A real HTTP server is bound on port 17700 for the duration of the test module.
Each test sends live HTTP requests and validates the response payloads.
"""

from __future__ import annotations

import json
import time
import urllib.request

import pytest

from gostaan import Gostaan
from gostaan.integration.schema import PlatformEvent
from gostaan.integration.security import SecurityPolicy
from gostaan.integration.server import EventServer
from gostaan.integration.client import EventClient
from gostaan.integration.bridge import GostaanBridge

# Port reserved for this test module (unlikely to conflict with real services).
TEST_PORT = 17700
BASE_URL = f"http://127.0.0.1:{TEST_PORT}"

# Secondary port for auth tests.
AUTH_PORT = 17701
AUTH_URL = f"http://127.0.0.1:{AUTH_PORT}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_bridge(port: int, require_auth: bool = False, tokens=None) -> GostaanBridge:
    g = Gostaan(dim=64, seed=42)
    g.set_identity(
        "I am a test cognitive system. I value accurate and verifiable results."
    )
    policy = SecurityPolicy(
        auth_tokens=set(tokens or []),
        require_auth=require_auth,
    )
    server = EventServer(host="127.0.0.1", port=port, security=policy)
    return GostaanBridge(gostaan=g, server=server, source_name="gostaan-test")


@pytest.fixture(scope="module")
def bridge():
    b = _make_bridge(TEST_PORT, require_auth=False)
    b.start(blocking=False)
    time.sleep(0.15)  # give the OS time to bind the port
    yield b
    b.stop()


@pytest.fixture
def client(bridge):  # noqa: ARG001
    return EventClient(base_url=BASE_URL, source="pytest-client")


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_returns_200(self, bridge):  # noqa: ARG002
        with urllib.request.urlopen(f"{BASE_URL}/health") as resp:
            assert resp.status == 200
            data = json.loads(resp.read())
            assert data["status"] == "ok"

    def test_client_health(self, bridge, client):  # noqa: ARG002
        assert client.health() is True


# ---------------------------------------------------------------------------
# Event round-trips
# ---------------------------------------------------------------------------


class TestPerceive:
    def test_returns_result_with_episode_id(self, bridge, client):  # noqa: ARG002
        result = client.emit("perceive", {"data": "Integration test perception."})
        assert result is not None
        assert result.type == "result"
        assert isinstance(result.payload.get("episode_id"), str)

    def test_trace_id_propagated(self, bridge, client):  # noqa: ARG002
        # The bridge sets response.trace_id = request.id (request/response
        # correlation), not the request's own trace_id field.
        ev = PlatformEvent(
            type="perceive",
            source="test",
            payload={"data": "Trace test."},
        )
        result = client.send(ev)
        assert result is not None
        assert result.trace_id == ev.id

    def test_perceive_with_tags(self, bridge, client):  # noqa: ARG002
        result = client.emit(
            "perceive",
            {"data": "Tagged event.", "tags": ["alpha", "beta"]},
        )
        assert result is not None
        assert result.type == "result"


class TestRecall:
    def test_returns_memories_list(self, bridge, client):  # noqa: ARG002
        client.emit("perceive", {"data": "Recall target experience."})
        result = client.emit("recall", {"query": "Recall target", "top_k": 3})
        assert result is not None
        assert result.type == "result"
        assert "memories" in result.payload
        assert isinstance(result.payload["memories"], list)
        assert "count" in result.payload

    def test_empty_query_still_returns_result(self, bridge, client):  # noqa: ARG002
        result = client.emit("recall", {"query": ""})
        assert result is not None
        assert result.type == "result"


class TestSleep:
    def test_returns_rem_report(self, bridge, client):  # noqa: ARG002
        result = client.emit("sleep", {})
        assert result is not None
        assert result.type == "result"
        assert "cycle_id" in result.payload
        assert "episodes_consolidated" in result.payload
        assert "episodes_pruned" in result.payload


class TestImagine:
    def test_returns_ideas_list(self, bridge, client):  # noqa: ARG002
        client.emit("perceive", {"data": "Memory architecture experiment."})
        client.emit("perceive", {"data": "Cognitive systems and intelligence."})
        result = client.emit("imagine", {"seed": "memory", "top_k": 3})
        assert result is not None
        assert result.type == "result"
        assert "ideas" in result.payload
        assert isinstance(result.payload["ideas"], list)


class TestSynthesise:
    def test_returns_blend_vector(self, bridge, client):  # noqa: ARG002
        result = client.emit(
            "synthesise", {"concepts": ["memory", "attention"]}
        )
        assert result is not None
        assert result.type == "result"
        blend = result.payload.get("blend")
        assert isinstance(blend, list)
        assert len(blend) == 64  # matches dim=64 fixture

    def test_fewer_than_two_concepts_returns_error(self, bridge, client):  # noqa: ARG002
        result = client.emit("synthesise", {"concepts": ["only-one"]})
        assert result is not None
        assert result.type == "error"


class TestHeartbeat:
    def test_returns_status(self, bridge, client):  # noqa: ARG002
        result = client.emit("heartbeat", {})
        assert result is not None
        assert result.type == "result"
        assert "status" in result.payload


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_invalid_event_type_returns_400(self, bridge):  # noqa: ARG002
        ev = PlatformEvent(type="perceive", source="test", payload={})
        # Bypass validate() and corrupt the type
        ev.type = "not_a_real_type"
        c = EventClient(base_url=BASE_URL, source="test")
        with pytest.raises(ConnectionError, match="400"):
            c.send(ev)

    def test_unknown_route_returns_404(self, bridge):  # noqa: ARG002
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(f"{BASE_URL}/unknown-path")
        assert exc_info.value.code == 404

    def test_invalid_json_returns_400(self, bridge):  # noqa: ARG002
        import urllib.request as _req

        req = _req.Request(
            f"{BASE_URL}/events",
            data=b"not valid json {{{",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            _req.urlopen(req)
        assert exc_info.value.code == 400


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------


class TestAuthentication:
    @pytest.fixture(scope="class")
    def auth_bridge(self):
        b = _make_bridge(AUTH_PORT, require_auth=True, tokens=["correct-token"])
        b.start(blocking=False)
        time.sleep(0.15)
        yield b
        b.stop()

    def test_no_token_returns_401(self, auth_bridge):  # noqa: ARG002
        c = EventClient(base_url=AUTH_URL, source="noauth")
        with pytest.raises(ConnectionError, match="401"):
            c.emit("heartbeat", {})

    def test_wrong_token_returns_401(self, auth_bridge):  # noqa: ARG002
        c = EventClient(
            base_url=AUTH_URL, auth_token="wrong-token", source="wrongauth"
        )
        with pytest.raises(ConnectionError, match="401"):
            c.emit("heartbeat", {})

    def test_correct_token_succeeds(self, auth_bridge):  # noqa: ARG002
        c = EventClient(
            base_url=AUTH_URL, auth_token="correct-token", source="goodauth"
        )
        result = c.emit("heartbeat", {})
        assert result is not None
        assert result.type == "result"
