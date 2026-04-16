/**
 * Lightweight HTTP event client for the MASSIVEMAGNETICS platform.
 *
 * Uses only Node.js built-in modules (http / https / crypto).
 * No npm runtime dependencies required.
 *
 * Usage:
 *
 *   import { EventClient } from "@massivemagnetics/gostaan-event-client";
 *
 *   const client = new EventClient({
 *     baseUrl: "http://127.0.0.1:7700",
 *     authToken: process.env.GOSTAAN_TOKEN,
 *     source: "agi_council",
 *   });
 *
 *   const result = await client.emit("perceive", {
 *     data: "New signal received from council.",
 *     importance: 0.85,
 *   });
 */

import * as http from "http";
import * as https from "https";
import { randomUUID } from "crypto";
import { PlatformEvent, EventType, SCHEMA_VERSION } from "./schema";

export interface ClientOptions {
  /** Root URL of the remote Gostaan server (e.g. "http://host:7700"). */
  baseUrl: string;
  /** Bearer token sent in the Authorization header. */
  authToken?: string;
  /** Request timeout in milliseconds (default: 10 000). */
  timeoutMs?: number;
  /** Value used as the "source" field in emitted events (default: "ts-client"). */
  source?: string;
}

export class EventClient {
  private readonly eventsUrl: string;
  private readonly healthUrl: string;
  private readonly authToken?: string;
  private readonly timeoutMs: number;
  private readonly source: string;

  constructor(options: ClientOptions) {
    const base = options.baseUrl.replace(/\/$/, "");
    this.eventsUrl = `${base}/events`;
    this.healthUrl = `${base}/health`;
    this.authToken = options.authToken;
    this.timeoutMs = options.timeoutMs ?? 10_000;
    this.source = options.source ?? "ts-client";
  }

  // -----------------------------------------------------------------------
  // Public API
  // -----------------------------------------------------------------------

  /**
   * POST a fully-constructed {@link PlatformEvent} to the server.
   *
   * Returns the response {@link PlatformEvent} on HTTP 200,
   * `null` on HTTP 202 Accepted, or rejects on error.
   */
  async send(event: PlatformEvent): Promise<PlatformEvent | null> {
    const body = JSON.stringify(event);
    const parsed = new URL(this.eventsUrl);
    const useHttps = parsed.protocol === "https:";
    const mod = useHttps ? https : http;

    const headers: Record<string, string> = {
      "Content-Type": "application/json",
      "Content-Length": Buffer.byteLength(body).toString(),
    };
    if (this.authToken) {
      headers["Authorization"] = `Bearer ${this.authToken}`;
    }

    return new Promise<PlatformEvent | null>((resolve, reject) => {
      const req = mod.request(
        {
          hostname: parsed.hostname,
          port: parsed.port
            ? Number(parsed.port)
            : useHttps
            ? 443
            : 80,
          path: parsed.pathname + (parsed.search ?? ""),
          method: "POST",
          headers,
        },
        (res) => {
          let data = "";
          res.on("data", (chunk: Buffer) => (data += chunk.toString()));
          res.on("end", () => {
            const status = res.statusCode ?? 0;
            if (status === 200 || status === 202) {
              try {
                const parsed = JSON.parse(data) as Record<string, unknown>;
                if ("type" in parsed && "source" in parsed) {
                  resolve(parsed as unknown as PlatformEvent);
                } else {
                  resolve(null);
                }
              } catch {
                resolve(null);
              }
            } else {
              reject(new Error(`Server returned HTTP ${status}: ${data}`));
            }
          });
        }
      );

      req.setTimeout(this.timeoutMs, () => {
        req.destroy(new Error("Request timed out"));
      });

      req.on("error", reject);
      req.write(body);
      req.end();
    });
  }

  /**
   * Convenience: construct a {@link PlatformEvent} and send it.
   *
   * @param type     One of the canonical event type strings.
   * @param payload  Event-specific data.
   * @param traceId  Optional correlation ID.
   */
  async emit(
    type: EventType,
    payload: Record<string, unknown>,
    traceId?: string
  ): Promise<PlatformEvent | null> {
    const event: PlatformEvent = {
      id: randomUUID(),
      version: SCHEMA_VERSION,
      type,
      source: this.source,
      timestamp: new Date().toISOString(),
      payload,
      trace_id: traceId ?? null,
    };
    return this.send(event);
  }

  /**
   * Perform a GET /health check against the remote server.
   *
   * Returns `true` if the server responds with HTTP 200.
   */
  async health(): Promise<boolean> {
    const parsed = new URL(this.healthUrl);
    const useHttps = parsed.protocol === "https:";
    const mod = useHttps ? https : http;

    return new Promise<boolean>((resolve) => {
      const req = mod.request(
        {
          hostname: parsed.hostname,
          port: parsed.port
            ? Number(parsed.port)
            : useHttps
            ? 443
            : 80,
          path: parsed.pathname,
          method: "GET",
        },
        (res) => {
          resolve(res.statusCode === 200);
        }
      );
      req.setTimeout(this.timeoutMs, () => {
        req.destroy();
        resolve(false);
      });
      req.on("error", () => resolve(false));
      req.end();
    });
  }
}
