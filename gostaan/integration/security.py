"""
Security primitives for the integration layer.

Provides:

``RateLimiter``      — sliding-window request counter per client key.
``SecurityPolicy``   — composable policy: auth tokens, origin allowlist,
                       rate limiting.

All primitives are thread-safe.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Set


class RateLimiter:
    """
    Sliding-window rate limiter keyed by an arbitrary string (e.g. client IP).

    Parameters
    ----------
    max_requests    : Maximum allowed requests inside the window.
    window_seconds  : Length of the sliding window in seconds.
    """

    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: float = 60.0,
    ) -> None:
        self._max = max_requests
        self._window = window_seconds
        self._buckets: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        """
        Return ``True`` if the request is within the limit, ``False`` otherwise.
        """
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            timestamps = [t for t in self._buckets.get(key, []) if t > cutoff]
            if len(timestamps) >= self._max:
                self._buckets[key] = timestamps
                return False
            timestamps.append(now)
            self._buckets[key] = timestamps
            return True


class SecurityPolicy:
    """
    Composable security policy for the event server.

    Parameters
    ----------
    auth_tokens     : Set of accepted bearer tokens.  If empty *and*
                      ``require_auth`` is ``True``, all requests are allowed
                      (no-token-configured → open).
    allowed_origins : Set of allowed ``Origin`` header values (CORS-like
                      allowlist).  Empty set → accept any origin.
    rate_limiter    : Custom ``RateLimiter``.  Defaults to 60 req/min per IP.
    require_auth    : When ``False`` the token check is skipped entirely.
    """

    def __init__(
        self,
        auth_tokens: Optional[Set[str]] = None,
        allowed_origins: Optional[Set[str]] = None,
        rate_limiter: Optional[RateLimiter] = None,
        require_auth: bool = True,
    ) -> None:
        self._tokens: Set[str] = set(auth_tokens or [])
        self._origins: Set[str] = set(allowed_origins or [])
        self._rate = rate_limiter or RateLimiter()
        self._require_auth = require_auth

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def add_token(self, token: str) -> None:
        """Add an accepted auth token."""
        self._tokens.add(token)

    def add_origin(self, origin: str) -> None:
        """Add an allowed origin."""
        self._origins.add(origin)

    # ------------------------------------------------------------------
    # Check helpers
    # ------------------------------------------------------------------

    def check_token(self, token: Optional[str]) -> bool:
        """
        Return ``True`` if the token is acceptable.

        - ``require_auth=False``         → always ``True``
        - No tokens configured           → always ``True`` (open server)
        - Tokens configured              → exact match required
        """
        if not self._require_auth:
            return True
        if not self._tokens:
            return True
        return token in self._tokens

    def check_origin(self, origin: Optional[str]) -> bool:
        """
        Return ``True`` if the origin is on the allowlist (or no allowlist set).
        """
        if not self._origins:
            return True
        return origin in self._origins

    def check_rate(self, client_key: str) -> bool:
        """Return ``True`` if the client is within the rate limit."""
        return self._rate.allow(client_key)

    def extract_token(self, auth_header: Optional[str]) -> Optional[str]:
        """
        Extract the bearer token from an ``Authorization`` header value.

        Supports ``"Bearer <token>"`` and bare token strings.
        Returns ``None`` if no header was provided.
        """
        if not auth_header:
            return None
        if auth_header.startswith("Bearer "):
            return auth_header[len("Bearer "):]
        return auth_header
