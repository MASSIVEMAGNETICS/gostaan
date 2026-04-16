"""
Integration Demo: Gostaan ↔ multi-system event bus.

Starts a Gostaan cognitive memory instance bound to an HTTP event server.
Other systems (Python or TypeScript) can send events to it, and it responds
with structured results.

Run
---
    python examples/integration_demo.py

Then, in a second terminal, send test events with curl:

    curl -X POST http://127.0.0.1:7700/events \\
      -H "Content-Type: application/json" \\
      -d '{
        "id": "demo-001",
        "version": "1.0.0",
        "type": "perceive",
        "source": "curl-demo",
        "timestamp": "2026-01-01T00:00:00Z",
        "payload": {"data": "Hello from curl!", "importance": 0.9}
      }'

Or run the TypeScript client demo (Node.js ≥ 18):

    cd integration/ts_client
    npx ts-node examples/demo.ts
"""

import time

from gostaan import Gostaan
from gostaan.integration import GostaanBridge, EventServer, EventClient, SecurityPolicy

IDENTITY = (
    "I am a multi-system cognitive platform. I value collaboration between "
    "different AI systems. I believe inter-system communication enables "
    "emergent intelligence that no single system can achieve alone."
)


def _build_bridge(port: int = 7700, require_auth: bool = False) -> GostaanBridge:
    g = Gostaan(dim=256, seed=42)
    g.set_identity(IDENTITY)
    g.perceive("Integration layer initialised. Ready to receive platform events.")

    security = SecurityPolicy(require_auth=require_auth)
    server = EventServer(host="127.0.0.1", port=port, security=security)
    bridge = GostaanBridge(gostaan=g, server=server, source_name="gostaan")
    return bridge


def main() -> None:
    print("=" * 60)
    print("  Gostaan Multi-System Integration Demo")
    print("=" * 60)

    bridge = _build_bridge(port=7700)
    print(f"\nStarting event server on {bridge._server.url} …")
    bridge.start(blocking=False)
    print("✓ Server started.")
    print()
    print("Accepted event types: perceive, recall, sleep, imagine, synthesise, heartbeat")
    print()

    # ── Self-test: local round-trip via EventClient ───────────────────────
    client = EventClient(base_url="http://127.0.0.1:7700", source="self-test")
    time.sleep(0.05)

    print("→ Self-test: perceive …")
    r = client.emit("perceive", {"data": "Self-test experience.", "importance": 0.7})
    print(f"  episode_id = {r.payload.get('episode_id') if r else 'n/a'}")

    print("→ Self-test: recall …")
    r = client.emit("recall", {"query": "integration", "top_k": 3})
    count = r.payload.get("count", 0) if r else 0
    print(f"  memories returned = {count}")

    print("→ Self-test: heartbeat …")
    r = client.emit("heartbeat", {})
    status = r.payload.get("status", {}) if r else {}
    print(f"  episodic_count = {status.get('episodic_count')}")
    print()
    print("Self-tests passed ✓")
    print()
    print("Listening for events. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping…")
        bridge.stop()
        print("✓ Done")


if __name__ == "__main__":
    main()
