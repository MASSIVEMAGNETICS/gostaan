"""
HTTP event client for the MASSIVEMAGNETICS multi-system platform.

Sends ``PlatformEvent`` objects to a remote ``EventServer`` (or any compatible
endpoint) using only Python stdlib (``urllib``).  No third-party dependencies.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

from gostaan.integration.schema import PlatformEvent


class EventClient:
    """
    Lightweight HTTP client for emitting ``PlatformEvent`` objects.

    Parameters
    ----------
    base_url    : Root URL of the remote server (e.g. ``"http://host:7700"``).
    auth_token  : Bearer token sent in ``Authorization`` header.
    timeout     : Request timeout in seconds (default 10).
    source      : ``source`` field value injected into convenience ``emit``
                  calls (default ``"gostaan"``).
    """

    def __init__(
        self,
        base_url: str,
        auth_token: Optional[str] = None,
        timeout: float = 10.0,
        source: str = "gostaan",
    ) -> None:
        self._events_url = base_url.rstrip("/") + "/events"
        self._health_url = base_url.rstrip("/") + "/health"
        self._token = auth_token
        self._timeout = timeout
        self._source = source

    # ------------------------------------------------------------------

    def send(self, event: PlatformEvent) -> Optional[PlatformEvent]:
        """
        POST a ``PlatformEvent`` to the remote server.

        Returns the response ``PlatformEvent`` when the server returns 200,
        ``None`` on 202 Accepted, or raises ``ConnectionError`` on failure.
        """
        body = event.to_json().encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        req = urllib.request.Request(
            self._events_url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                raw = resp.read().decode("utf-8")
                data: dict = json.loads(raw)
                if "type" in data and "source" in data:
                    return PlatformEvent.from_dict(data)
                return None
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ConnectionError(
                f"Server returned HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise ConnectionError(f"Could not connect: {exc.reason}") from exc

    def emit(
        self,
        event_type: str,
        payload: dict,
        trace_id: Optional[str] = None,
    ) -> Optional[PlatformEvent]:
        """
        Build a ``PlatformEvent`` and send it.

        Parameters
        ----------
        event_type  : One of the canonical event type strings.
        payload     : Event-specific data dict.
        trace_id    : Optional correlation ID.
        """
        event = PlatformEvent(
            type=event_type,
            source=self._source,
            payload=payload,
            trace_id=trace_id,
        )
        event.validate()
        return self.send(event)

    def health(self) -> bool:
        """Return ``True`` if the remote server is reachable and healthy."""
        try:
            with urllib.request.urlopen(self._health_url, timeout=self._timeout) as r:
                return r.status == 200
        except Exception:  # noqa: BLE001
            return False
