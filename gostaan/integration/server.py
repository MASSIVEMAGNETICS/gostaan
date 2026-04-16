"""
HTTP event server adapter for Gostaan.

Exposes a minimal HTTP API:

    POST /events   — receive a ``PlatformEvent`` JSON body
    GET  /health   — liveness probe

Security (via ``SecurityPolicy``):
    - Bearer token authentication
    - Origin allowlist
    - Per-IP sliding-window rate limiting
    - 1 MB request body cap

The server runs in a daemon thread so it does not block the host process.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Optional

from gostaan.integration.schema import PlatformEvent
from gostaan.integration.security import SecurityPolicy

_MAX_BODY_BYTES = 1_048_576  # 1 MiB

# Type alias for the event handler callback.
EventHandler = Callable[[PlatformEvent], Optional[PlatformEvent]]


class _Handler(BaseHTTPRequestHandler):
    """Internal request handler — configured via class attributes."""

    # These are set by EventServer before binding.
    _event_handler: EventHandler
    _security: SecurityPolicy

    # ------------------------------------------------------------------ helpers

    def log_message(self, fmt: str, *args: object) -> None:  # suppress noise
        pass

    def _send_json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------------ routes

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        if self.path != "/events":
            self._send_json(404, {"error": "not_found"})
            return

        client_ip: str = self.client_address[0]

        # --- Rate limit ---
        if not self._security.check_rate(client_ip):
            self._send_json(429, {"error": "rate_limit_exceeded"})
            return

        # --- Origin allowlist ---
        origin = self.headers.get("Origin")
        if not self._security.check_origin(origin):
            self._send_json(403, {"error": "origin_not_allowed"})
            return

        # --- Auth ---
        token = self._security.extract_token(self.headers.get("Authorization"))
        if not self._security.check_token(token):
            self._send_json(401, {"error": "unauthorized"})
            return

        # --- Read body ---
        try:
            length = int(self.headers.get("Content-Length", 0))
        except ValueError:
            self._send_json(400, {"error": "invalid_content_length"})
            return

        if length > _MAX_BODY_BYTES:
            self._send_json(413, {"error": "payload_too_large"})
            return

        raw = self.rfile.read(length)

        # --- Parse & validate ---
        try:
            event = PlatformEvent.from_json(raw.decode("utf-8", errors="replace"))
            event.validate()
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": "invalid_event", "detail": str(exc)})
            return

        # --- Dispatch to handler ---
        try:
            result = self._event_handler(event)
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": "handler_error", "detail": str(exc)})
            return

        if result is None:
            self._send_json(202, {"status": "accepted", "event_id": event.id})
        else:
            self._send_json(200, result.to_dict())


class EventServer:
    """
    Lightweight HTTP server that receives ``PlatformEvent`` JSON payloads.

    Parameters
    ----------
    host        : Bind address (default ``"127.0.0.1"``).
    port        : TCP port (default ``7700``).
    handler     : Callable ``(PlatformEvent) -> Optional[PlatformEvent]``.
                  Return a ``PlatformEvent`` to send a 200 response, or
                  ``None`` to send a 202 Accepted.
    security    : ``SecurityPolicy`` instance.  Defaults to no-auth mode.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7700,
        handler: Optional[EventHandler] = None,
        security: Optional[SecurityPolicy] = None,
    ) -> None:
        self.host = host
        self.port = port
        self._handler: EventHandler = handler or (lambda _e: None)
        self._security = security or SecurityPolicy(require_auth=False)
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def set_handler(self, handler: EventHandler) -> None:
        """Replace the event handler (must be called before ``start``)."""
        self._handler = handler

    # ------------------------------------------------------------------

    def start(self, blocking: bool = False) -> None:
        """
        Start the server.

        Parameters
        ----------
        blocking : When ``True`` the call blocks (useful for CLI scripts).
                   When ``False`` the server runs in a daemon thread.
        """
        policy = self._security
        fn = self._handler

        class _BoundHandler(_Handler):
            _event_handler = staticmethod(fn)
            _security = policy

        self._server = ThreadingHTTPServer((self.host, self.port), _BoundHandler)
        if blocking:
            self._server.serve_forever()
        else:
            self._thread = threading.Thread(
                target=self._server.serve_forever, daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        """Shut down the server and join the background thread."""
        if self._server:
            self._server.shutdown()
            self._server = None
        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

    @property
    def url(self) -> str:
        """Base URL of the server (e.g. ``"http://127.0.0.1:7700"``). """
        return f"http://{self.host}:{self.port}"
