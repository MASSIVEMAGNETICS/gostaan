/**
 * Canonical TypeScript interfaces for the MASSIVEMAGNETICS PlatformEvent schema.
 *
 * Keep in sync with integration/schema/event_schema.json and
 * gostaan/integration/schema.py.
 */

export const SCHEMA_VERSION = "1.0.0" as const;

export type EventType =
  | "perceive"
  | "recall"
  | "sleep"
  | "imagine"
  | "synthesise"
  | "result"
  | "error"
  | "heartbeat";

export interface PlatformEvent {
  /** Unique event identifier (UUID v4). */
  id: string;
  /** Schema version (semver, e.g. "1.0.0"). */
  version: string;
  /** Canonical event type. */
  type: EventType;
  /** Identifier of the originating system. */
  source: string;
  /** ISO 8601 UTC timestamp. */
  timestamp: string;
  /** Event-specific data. */
  payload: Record<string, unknown>;
  /** Optional correlation ID linking a request to its response. */
  trace_id?: string | null;
}

// ---------------------------------------------------------------------------
// Per-type payload shapes (informational — TypeScript does not enforce them
// at runtime, but they document the expected structure for each event type).
// ---------------------------------------------------------------------------

export interface PerceivePayload {
  /** Raw input: text, JSON object, or numeric array. */
  data: unknown;
  importance?: number;
  emotional_weight?: number;
  tags?: string[];
  context?: Record<string, string>;
}

export interface RecallPayload {
  query: string;
  top_k?: number;
  tags?: string[];
}

export interface ImaginePayload {
  seed: string;
  top_k?: number;
}

export interface SynthesisePayload {
  /** At least two concepts to blend. */
  concepts: [string, string, ...string[]];
}

export interface ResultPayload {
  [key: string]: unknown;
}

export interface ErrorPayload {
  detail: string;
}
