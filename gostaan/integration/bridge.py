"""
GostaanBridge: wires a Gostaan instance to the multi-system event bus.

Inbound side
------------
An ``EventServer`` receives ``PlatformEvent`` objects and dispatches them to
the appropriate ``Gostaan`` method:

    perceive   → g.perceive(...)
    recall     → g.recall(...)
    sleep      → g.sleep()
    imagine    → g.imagine(...)
    synthesise → g.synthesise(...)
    heartbeat  → g.status()

Each inbound event produces a ``result`` (or ``error``) ``PlatformEvent``
returned synchronously in the HTTP response.

Outbound side
-------------
Zero or more ``EventClient`` instances can be registered under logical names
(e.g. ``"conscious-river"``, ``"agi_council"``).  Use ``broadcast()`` to
push a ``PlatformEvent`` to all registered peers simultaneously.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from gostaan.integration.schema import PlatformEvent
from gostaan.integration.server import EventServer
from gostaan.integration.client import EventClient


class GostaanBridge:
    """
    Bridge between a ``Gostaan`` cognitive memory instance and the event bus.

    Parameters
    ----------
    gostaan     : A ``Gostaan`` instance (type-hinted as ``Any`` to avoid a
                  circular import at module level).
    server      : ``EventServer`` that receives inbound events.  If ``None``
                  the bridge operates in client-only mode.
    source_name : ``source`` string used in outbound response events.
    """

    def __init__(
        self,
        gostaan: Any,
        server: Optional[EventServer] = None,
        source_name: str = "gostaan",
    ) -> None:
        self._g = gostaan
        self._server = server
        self._source = source_name
        self._clients: Dict[str, EventClient] = {}

        if self._server is not None:
            self._server.set_handler(self._handle_event)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, blocking: bool = False) -> None:
        """Start the inbound event server."""
        if self._server is not None:
            self._server.start(blocking=blocking)

    def stop(self) -> None:
        """Stop the inbound event server."""
        if self._server is not None:
            self._server.stop()

    # ------------------------------------------------------------------
    # Peer management
    # ------------------------------------------------------------------

    def add_client(self, name: str, client: EventClient) -> None:
        """Register a downstream peer client (e.g. ``"conscious-river"``)."""
        self._clients[name] = client

    def remove_client(self, name: str) -> None:
        """Deregister a peer client."""
        self._clients.pop(name, None)

    # ------------------------------------------------------------------
    # Outbound broadcast
    # ------------------------------------------------------------------

    def broadcast(
        self, event: PlatformEvent
    ) -> Dict[str, Optional[PlatformEvent]]:
        """
        Send *event* to every registered downstream client.

        Returns a dict ``{peer_name: response_or_error_event}``.
        Connection failures are captured as ``error`` events rather than
        raised, so a single unreachable peer does not abort the broadcast.
        """
        results: Dict[str, Optional[PlatformEvent]] = {}
        for name, client in self._clients.items():
            try:
                results[name] = client.send(event)
            except Exception as exc:  # noqa: BLE001
                results[name] = PlatformEvent(
                    type="error",
                    source=name,
                    payload={"detail": str(exc)},
                    trace_id=event.id,
                )
        return results

    # ------------------------------------------------------------------
    # Inbound dispatch
    # ------------------------------------------------------------------

    def _handle_event(self, event: PlatformEvent) -> Optional[PlatformEvent]:
        """Route an inbound event to the correct Gostaan method."""
        _dispatch = {
            "perceive": self._on_perceive,
            "recall": self._on_recall,
            "sleep": self._on_sleep,
            "imagine": self._on_imagine,
            "synthesise": self._on_synthesise,
            "heartbeat": self._on_heartbeat,
        }
        handler = _dispatch.get(event.type)
        if handler is None:
            return self._make_error(event, f"Unsupported event type: {event.type!r}")
        try:
            return handler(event)
        except Exception as exc:  # noqa: BLE001
            return self._make_error(event, str(exc))

    # ------------------------------------------------------------------
    # Per-type handlers
    # ------------------------------------------------------------------

    def _on_perceive(self, event: PlatformEvent) -> PlatformEvent:
        p = event.payload
        episode_id = self._g.perceive(
            data=p.get("data", ""),
            importance=float(p.get("importance", 1.0)),
            emotional_weight=float(p.get("emotional_weight", 0.5)),
            tags=p.get("tags") or [],
            context=p.get("context") or {},
        )
        return self._make_result(event, {"episode_id": episode_id})

    def _on_recall(self, event: PlatformEvent) -> PlatformEvent:
        p = event.payload
        episodes = self._g.recall(
            query=p.get("query", ""),
            top_k=int(p.get("top_k", 5)),
            tags=p.get("tags"),
        )
        memories: List[Dict[str, Any]] = [
            {
                "episode_id": ep.episode_id,
                "content": ep.content,
                "importance": ep.importance,
                "timestamp": ep.timestamp,
                "tags": ep.tags,
            }
            for ep in episodes
        ]
        return self._make_result(event, {"memories": memories, "count": len(memories)})

    def _on_sleep(self, event: PlatformEvent) -> PlatformEvent:
        report = self._g.sleep()
        return self._make_result(
            event,
            {
                "cycle_id": report.cycle_id,
                "episodes_consolidated": report.episodes_consolidated,
                "episodes_pruned": report.episodes_pruned,
            },
        )

    def _on_imagine(self, event: PlatformEvent) -> PlatformEvent:
        p = event.payload
        inferences = self._g.imagine(
            seed=p.get("seed", ""),
            top_k=int(p.get("top_k", 5)),
        )
        ideas: List[Dict[str, Any]] = [
            {
                "content": inf.content,
                "confidence": inf.confidence,
                "novelty_score": inf.novelty_score,
            }
            for inf in inferences
        ]
        return self._make_result(event, {"ideas": ideas})

    def _on_synthesise(self, event: PlatformEvent) -> PlatformEvent:
        concepts = event.payload.get("concepts", [])
        if len(concepts) < 2:
            return self._make_error(
                event, "'synthesise' requires at least 2 concepts in payload."
            )
        blended = self._g.synthesise(*concepts)
        return self._make_result(event, {"blend": blended.tolist()})

    def _on_heartbeat(self, event: PlatformEvent) -> PlatformEvent:
        return self._make_result(event, {"status": self._g.status()})

    # ------------------------------------------------------------------
    # Response helpers
    # ------------------------------------------------------------------

    def _make_result(
        self, trigger: PlatformEvent, payload: Dict[str, Any]
    ) -> PlatformEvent:
        return PlatformEvent(
            type="result",
            source=self._source,
            payload=payload,
            trace_id=trigger.id,
        )

    def _make_error(self, trigger: PlatformEvent, detail: str) -> PlatformEvent:
        return PlatformEvent(
            type="error",
            source=self._source,
            payload={"detail": detail},
            trace_id=trigger.id,
        )
