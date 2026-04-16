# Multi-System Integration Layer

Gostaan exposes a lightweight HTTP event bus that lets it exchange structured
events with the other MASSIVEMAGNETICS systems:

| System | Language | Role |
|--------|----------|------|
| **conscious-river** | Python | Sensory inputs, attention, data merging |
| **Defensive-Cognitive-Runtime** | Python | Protector / defensive runtime |
| **agi_council** | TypeScript | Multi-agent cross-reasoning engine |
| **complete-active-aware-repo-intelligence** | TypeScript | Repo intelligence |

---

## Quick Start

### 1 — Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2 — Start the Gostaan event server

```bash
python examples/integration_demo.py
```

The server starts at `http://127.0.0.1:7700`.  
You should see:

```
============================================================
  Gostaan Multi-System Integration Demo
============================================================

Starting event server on http://127.0.0.1:7700 …
✓ Server started.
...
Listening for events. Press Ctrl+C to stop.
```

### 3 — Send events with `curl`

```bash
# Store a new experience
curl -s -X POST http://127.0.0.1:7700/events \
  -H "Content-Type: application/json" \
  -d '{
    "id":        "curl-001",
    "version":   "1.0.0",
    "type":      "perceive",
    "source":    "curl",
    "timestamp": "2026-01-01T00:00:00Z",
    "payload":   {"data": "Hello from curl!", "importance": 0.9}
  }' | python -m json.tool

# Query memories
curl -s -X POST http://127.0.0.1:7700/events \
  -H "Content-Type: application/json" \
  -d '{
    "id":        "curl-002",
    "version":   "1.0.0",
    "type":      "recall",
    "source":    "curl",
    "timestamp": "2026-01-01T00:00:01Z",
    "payload":   {"query": "Hello", "top_k": 3}
  }' | python -m json.tool
```

### 4 — TypeScript client (Node.js ≥ 18)

```bash
cd integration/ts_client
npm install          # installs typescript + ts-node (dev deps only)
npx ts-node examples/demo.ts
```

---

## Event Schema

All messages use `PlatformEvent` (see `integration/schema/event_schema.json`).

| Field | Type | Required | Description |
|-------|------|:--------:|-------------|
| `id` | UUID string | ✅ | Unique event identifier |
| `version` | semver string | ✅ | Schema version (`"1.0.0"`) |
| `type` | enum | ✅ | See table below |
| `source` | string | ✅ | Originating system name |
| `timestamp` | ISO 8601 | ✅ | UTC creation time |
| `payload` | object | ✅ | Type-specific data |
| `trace_id` | string | ❌ | Correlation ID (request → response) |

### Event Types

| Type | Direction | Payload fields |
|------|-----------|----------------|
| `perceive` | → Gostaan | `data`, `importance`, `emotional_weight`, `tags`, `context` |
| `recall` | → Gostaan | `query`, `top_k`, `tags` |
| `sleep` | → Gostaan | *(none)* |
| `imagine` | → Gostaan | `seed`, `top_k` |
| `synthesise` | → Gostaan | `concepts` (list of ≥ 2) |
| `heartbeat` | → Gostaan | *(none)* |
| `result` | ← Gostaan | Varies by trigger type |
| `error` | ← Gostaan | `detail` |

---

## Security

### Auth tokens

```python
from gostaan.integration import SecurityPolicy, EventServer

policy = SecurityPolicy(
    auth_tokens={"my-secret-token"},   # accepted bearer tokens
    require_auth=True,
)
server = EventServer(port=7700, security=policy)
```

Clients send the token in the request header:

```
Authorization: Bearer my-secret-token
```

### Origin allowlist

```python
policy = SecurityPolicy(
    auth_tokens={"secret"},
    allowed_origins={"http://trusted-service.internal"},
)
```

### Rate limiting

Default: **60 requests per minute per client IP**.  Customise with:

```python
from gostaan.integration import RateLimiter, SecurityPolicy

rl = RateLimiter(max_requests=120, window_seconds=60)
policy = SecurityPolicy(rate_limiter=rl)
```

---

## Connecting Other Systems

### Python peer (e.g. conscious-river)

```python
from gostaan.integration import EventClient

client = EventClient(
    base_url="http://gostaan-host:7700",
    auth_token="shared-secret",
    source="conscious-river",
)

# Store a sensory observation
result = client.emit("perceive", {
    "data": "Attention spike detected in sensory stream.",
    "importance": 0.85,
    "tags": ["sensory", "attention"],
})
print(result.payload["episode_id"])
```

### TypeScript peer (e.g. agi_council)

```typescript
import { EventClient } from "@massivemagnetics/gostaan-event-client";

const client = new EventClient({
  baseUrl: "http://gostaan-host:7700",
  authToken: process.env.GOSTAAN_TOKEN,
  source: "agi_council",
});

const result = await client.emit("recall", {
  query: "recent council decisions",
  top_k: 5,
});
console.log(result?.payload?.memories);
```

### Outbound broadcast (Gostaan → peers)

```python
from gostaan.integration import GostaanBridge, EventClient, PlatformEvent

bridge = GostaanBridge(gostaan=g, server=server)

# Register downstream systems
bridge.add_client("conscious-river", EventClient("http://cr-host:7701"))
bridge.add_client("dcr", EventClient("http://dcr-host:7702"))

bridge.start()

# Push an event to all registered peers
bridge.broadcast(PlatformEvent(
    type="result",
    source="gostaan",
    payload={"message": "Memory consolidated after REM cycle."},
))
```

---

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                  Multi-System Platform                    │
│                                                           │
│  conscious-river  ──────────┐                             │
│  Defensive-Cognitive  ──────┤  PlatformEvent (HTTP POST)  │
│  agi_council  ─────────────┼──▶  EventServer :7700        │
│  repo-intelligence  ────────┘         │                   │
│                                 GostaanBridge             │
│                                       │                   │
│                               ┌───────▼────────┐          │
│                               │   Gostaan Core │          │
│                               │  perceive()    │          │
│                               │  recall()      │          │
│                               │  sleep()       │          │
│                               │  imagine()     │          │
│                               └────────────────┘          │
└───────────────────────────────────────────────────────────┘
```

---

## Running Tests

```bash
# Schema + security unit tests
pytest tests/test_integration_schema.py -v

# End-to-end server/client/bridge tests (binds port 17700)
pytest tests/test_integration_server.py -v

# Full test suite
pytest tests/ -v
```
