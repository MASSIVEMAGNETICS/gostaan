"""
Gostaan Integration Layer

Provides a canonical event schema and lightweight HTTP adapters so that
Gostaan can exchange events with other MASSIVEMAGNETICS systems:

- conscious-river                            (Python)
- Defensive-Cognitive-Runtime                (Python)
- agi_council                                (TypeScript)
- complete-active-aware-repo-intelligence    (TypeScript)

Quick start::

    from gostaan import Gostaan
    from gostaan.integration import GostaanBridge, EventServer, SecurityPolicy

    g = Gostaan(dim=256)
    g.set_identity("I am a multi-system cognitive platform.")

    security = SecurityPolicy(auth_tokens={"my-secret"})
    server   = EventServer(host="127.0.0.1", port=7700, security=security)
    bridge   = GostaanBridge(gostaan=g, server=server)
    bridge.start()
"""

from gostaan.integration.schema import PlatformEvent, SCHEMA_VERSION, EVENT_TYPES
from gostaan.integration.security import SecurityPolicy, RateLimiter
from gostaan.integration.server import EventServer
from gostaan.integration.client import EventClient
from gostaan.integration.bridge import GostaanBridge

__all__ = [
    "PlatformEvent",
    "SCHEMA_VERSION",
    "EVENT_TYPES",
    "SecurityPolicy",
    "RateLimiter",
    "EventServer",
    "EventClient",
    "GostaanBridge",
]
