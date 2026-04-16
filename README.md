# gostaan

**Gostaan** is a next-generation persistent memory system with gravitational attention — a revolutionary cognitive architecture that moves beyond traditional transformer attention.

## Architecture

```
Perception  ──▶  SensoryProcessor  ──▶  embedding
Embedding   ──▶  TokenFormer       ──▶  encoded embedding
Encoded     ──▶  EpisodicMemory    (day-to-day episodic store)
Encoded     ──▶  HDDRMemory        (high-dimensional dense representation)
Idle        ──▶  REMProcessor      (consolidation + intelligent pruning)
Query       ──▶  SelfInferenceEngine  (recall + novel idea synthesis)
```

### Core Modules

| Module | Description |
|--------|-------------|
| `gostaan/attention/` | **Gravitational Attention** — replaces QKV dot-product with physics-based gravitational forces between tokens |
| `gostaan/tokenformer/` | **TokenFormer** — multi-layer architecture with persistent *parameter tokens* (learnable knowledge slots) |
| `gostaan/memory/episodic.py` | **Episodic Memory** — human-like day-to-day experience store with cue-based recall and intelligent consolidation |
| `gostaan/memory/hddr.py` | **HDDR Memory** — High-Dimensional Dense Representation store with sparse write and associative recall |
| `gostaan/memory/identity.py` | **Identity Anchor** — parses an "I am" paragraph into an identity vector that gates all memory operations |
| `gostaan/memory/rem_cycles.py` | **REM Sleep Cycles** — idle-triggered consolidation, decay, identity alignment, and pruning |
| `gostaan/inference/` | **Self-Inference Engine** — analogical reasoning, concept synthesis, hypothesis generation |
| `gostaan/sensory/` | **Sensory Processor** — multi-modal input pipeline (text, numeric, event) with saliency filtering |
| `gostaan/core.py` | **Gostaan Orchestrator** — wires all components into a unified API |

## Quick Start

```python
from gostaan import Gostaan

# Create system
g = Gostaan(dim=256)

# Anchor identity from your "I am" statement
g.set_identity(
    "I am a creative engineer. I value intelligent systems and deep thinking. "
    "I believe persistent memory is the foundation of intelligence."
)

# Ingest experiences
g.perceive("Today I designed a gravitational attention layer.")
g.perceive("The TokenFormer encodes sequences using parameter tokens.")
g.perceive("REM cycles consolidate episodic memories during idle time.")

# Recall relevant memories
memories = g.recall("attention mechanism")
for ep in memories:
    print(ep.content)

# Generate novel ideas from memory
ideas = g.imagine("memory architecture", top_k=3)

# Synthesise a new concept from multiple inputs
blended = g.synthesise("gravitational attention", "episodic memory")

# Run REM sleep cycle (consolidation + pruning)
report = g.sleep()
print(f"Consolidated: {report.episodes_consolidated}, Pruned: {report.episodes_pruned}")

# System status
print(g.status())
```

## Key Concepts

### Gravitational Attention
Instead of `Attention(Q,K,V) = softmax(QK^T/√d) V`, gravitational attention computes:

```
Force(i,j) = G × (M_i × M_j) / (Distance(i,j)² + ε)
Attention(P, M, V) = softmax(Force) × V
```

Tokens with high mass attract nearby tokens strongly, allowing concepts to "orbit" central ideas. The event horizon `ε` prevents singularities (Hawking radiation analogue).

### Identity Anchor
The "I am" paragraph is parsed into traits and embedded into an **identity vector**. All memories are scored for identity alignment during REM cycles — consistent memories are reinforced, inconsistent ones decay. Self-inference projects new ideas through identity space to ensure coherence.

### REM Sleep Cycles
When idle, the system enters a REM cycle that:
1. Decays importance of all memories
2. Consolidates similar episodes (cosine similarity merge)
3. Applies identity alignment (boost/decay based on identity score)
4. Prunes stale low-importance memories
5. Compresses HDDR store to capacity

## Multi-System Integration

Gostaan includes a lightweight HTTP event bus that lets it communicate with
the other MASSIVEMAGNETICS systems:
**conscious-river**, **Defensive-Cognitive-Runtime**, **agi_council**, and
**complete-active-aware-repo-intelligence**.

All inter-system messages use a single canonical `PlatformEvent` schema
(see `integration/schema/event_schema.json`).

### Quick start

```bash
# Start the Gostaan event server (port 7700)
python examples/integration_demo.py

# Send a perceive event from any system (curl, Python, TypeScript…)
curl -X POST http://127.0.0.1:7700/events \
  -H "Content-Type: application/json" \
  -d '{
    "id": "demo-1", "version": "1.0.0", "type": "perceive",
    "source": "my-system", "timestamp": "2026-01-01T00:00:00Z",
    "payload": {"data": "Hello from another system!", "importance": 0.9}
  }'
```

### Python peer adapter

```python
from gostaan.integration import EventClient

client = EventClient(
    base_url="http://gostaan-host:7700",
    auth_token="shared-secret",
    source="conscious-river",
)
result = client.emit("perceive", {"data": "Sensory spike detected."})
print(result.payload["episode_id"])
```

### TypeScript peer adapter (Node.js ≥ 18, no extra deps)

```typescript
import { EventClient } from "@massivemagnetics/gostaan-event-client";

const client = new EventClient({
  baseUrl: "http://gostaan-host:7700",
  authToken: process.env.GOSTAAN_TOKEN,
  source: "agi_council",
});
const result = await client.emit("recall", { query: "recent decisions", top_k: 5 });
console.log(result?.payload?.memories);
```

### Run the TypeScript demo

```bash
cd integration/ts_client
npm install
npx ts-node examples/demo.ts
```

See **[docs/integration.md](docs/integration.md)** for the full schema reference,
security configuration (auth tokens, origin allowlist, rate limiting), and
architecture diagram.

## Installation

```bash
pip install -r requirements.txt
```

## Tests

```bash
# All tests (120 total)
pytest tests/ -v

# Integration-only
pytest tests/test_integration_schema.py tests/test_integration_server.py -v
```

## License

MIT
