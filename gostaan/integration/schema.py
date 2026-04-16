"""
Canonical event schema for the MASSIVEMAGNETICS multi-system platform.

All inter-system messages use ``PlatformEvent``, a versioned, JSON-serialisable
dataclass that carries a typed payload and an optional correlation trace-id.

Supported event types (``EVENT_TYPES``):

``perceive``   — ingest a new experience into Gostaan
``recall``     — retrieve relevant memories from Gostaan
``sleep``      — trigger a REM consolidation cycle
``imagine``    — generate novel ideas from memory
``synthesise`` — blend two or more concepts
``result``     — response from Gostaan (or any system)
``error``      — error/exception notification
``heartbeat``  — keep-alive / status ping
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, Optional

SCHEMA_VERSION = "1.0.0"

EVENT_TYPES: frozenset = frozenset(
    {
        "perceive",
        "recall",
        "sleep",
        "imagine",
        "synthesise",
        "result",
        "error",
        "heartbeat",
    }
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PlatformEvent:
    """
    Canonical inter-system event.

    Fields
    ------
    type        : One of EVENT_TYPES.
    source      : Identifier of the originating system (e.g. ``"gostaan"``).
    payload     : Event-specific data (arbitrary JSON-serialisable dict).
    id          : UUID v4 string — auto-generated if not supplied.
    version     : Schema semver string — defaults to SCHEMA_VERSION.
    timestamp   : ISO 8601 UTC string — auto-generated if not supplied.
    trace_id    : Optional correlation ID linking a request to its response.
    """

    type: str
    source: str
    payload: Dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: str = field(default=SCHEMA_VERSION)
    timestamp: str = field(default_factory=_utc_now)
    trace_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ``ValueError`` if the event is malformed."""
        if self.type not in EVENT_TYPES:
            raise ValueError(
                f"Unknown event type: {self.type!r}. "
                f"Must be one of {sorted(EVENT_TYPES)}"
            )
        if not self.source:
            raise ValueError("Event 'source' must not be empty.")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Return a plain dict suitable for JSON serialisation."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict())

    # ------------------------------------------------------------------
    # Deserialisation
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlatformEvent":
        """
        Construct from a dict.

        Unknown keys are silently ignored so that future schema additions
        remain backward-compatible.
        """
        known = {"id", "version", "type", "source", "payload", "timestamp", "trace_id"}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_json(cls, raw: str) -> "PlatformEvent":
        """Construct from a JSON string."""
        return cls.from_dict(json.loads(raw))
