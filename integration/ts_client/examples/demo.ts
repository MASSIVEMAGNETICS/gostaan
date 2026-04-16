/**
 * TypeScript integration demo.
 *
 * Demonstrates a full event-flow round-trip from a TypeScript client to the
 * Gostaan Python cognitive memory system.
 *
 * Prerequisites
 * -------------
 * 1. Start the Gostaan event server (Python):
 *      python examples/integration_demo.py
 *
 * 2. Run this demo (Node.js ≥ 18):
 *      cd integration/ts_client
 *      npx ts-node examples/demo.ts
 */

import { EventClient } from "../src";

const GOSTAAN_URL = process.env.GOSTAAN_URL ?? "http://127.0.0.1:7700";
const GOSTAAN_TOKEN = process.env.GOSTAAN_TOKEN;

async function main(): Promise<void> {
  const client = new EventClient({
    baseUrl: GOSTAAN_URL,
    authToken: GOSTAAN_TOKEN,
    source: "ts-demo",
  });

  console.log(`Connecting to Gostaan at ${GOSTAAN_URL} …`);

  // ── Health check ─────────────────────────────────────────────────────────
  const healthy = await client.health();
  console.log("Server healthy:", healthy);
  if (!healthy) {
    console.error(
      "\nCould not reach the Gostaan server.\n" +
        "Start it with:  python examples/integration_demo.py\n"
    );
    process.exit(1);
  }

  // ── perceive ─────────────────────────────────────────────────────────────
  console.log("\n→ perceive: storing a new experience …");
  const perceiveResult = await client.emit("perceive", {
    data: "TypeScript client connected to Gostaan cognitive memory system.",
    importance: 0.9,
    tags: ["integration", "typescript"],
  });
  console.log("← episode_id:", perceiveResult?.payload?.episode_id);

  // ── perceive a second event for recall to find ───────────────────────────
  await client.emit("perceive", {
    data: "agi_council is reasoning across multiple agents in real time.",
    importance: 0.8,
    tags: ["council", "multi-agent"],
  });

  // ── recall ────────────────────────────────────────────────────────────────
  console.log("\n→ recall: querying 'cognitive memory' …");
  const recallResult = await client.emit("recall", {
    query: "cognitive memory",
    top_k: 3,
  });
  interface Memory {
    episode_id: string;
    content: string;
    importance: number;
    timestamp: number;
    tags: string[];
  }
  const memories = recallResult?.payload?.memories as Memory[];
  console.log(`← ${memories?.length ?? 0} memories returned`);
  memories?.forEach((m, i) => console.log(`   [${i + 1}] ${m.content}`));

  // ── imagine ───────────────────────────────────────────────────────────────
  console.log("\n→ imagine: generating ideas from 'inter-system intelligence' …");
  const imagineResult = await client.emit("imagine", {
    seed: "inter-system intelligence",
    top_k: 3,
  });
  interface Idea {
    content: string;
    confidence: number;
    novelty_score: number;
  }
  const ideas = imagineResult?.payload?.ideas as Idea[];
  console.log(`← ${ideas?.length ?? 0} ideas generated`);
  ideas?.slice(0, 3).forEach((idea, i) =>
    console.log(`   [${i + 1}] ${idea.content}`)
  );

  // ── heartbeat / status ────────────────────────────────────────────────────
  console.log("\n→ heartbeat: fetching Gostaan status …");
  const hbResult = await client.emit("heartbeat", {});
  const status = hbResult?.payload?.status as Record<string, unknown>;
  console.log("← episodic_count:", status?.episodic_count);
  console.log("← rem_cycles:    ", status?.rem_cycles);

  console.log("\n✓ TypeScript ↔ Python integration demo complete.");
}

main().catch((err) => {
  console.error("Demo failed:", err);
  process.exit(1);
});
