# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The research package's only network access.

Rules (design section 4): https only; fixed hosts (the bare host: no user info or port), with CDN redirects allowed on `.hf.co` and
`.huggingface.co` because that is where Hugging Face serves file bytes; no Authorization header, so the
operator's token is never sent; a size cap on every body; a byte range must be answered with 206; successes
are cached by URL and range so a resume reads the same answers; `offline` makes no request at all."""
from __future__ import annotations

import base64
import datetime
import hashlib
import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from contextlib import closing
from pathlib import Path
from urllib.parse import urlsplit

ALLOWED_HOSTS = ("huggingface.co", "api.github.com", "raw.githubusercontent.com")
ALLOWED_SUFFIXES = (".hf.co", ".huggingface.co")
MAX_BODY = 8 * 1024 * 1024
TIMEOUT_S = 20
# One research run's budget, counted from its first network request (answers from the cache are free). A run on a
# slow or stalled network ends with "unavailable" rows and `unknown` estimates instead of hanging the preflight.
TIMEOUTS_TO_GIVE_UP = 2     # connection timeouts to one host before the rest of the run stops asking it
MAX_REQUESTS = 600
DEADLINE_S = 240


class FetchError(Exception):
    """A request that could not be answered. `reason` is the text the report shows."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def host_allowed(host) -> bool:
    return isinstance(host, str) and (host in ALLOWED_HOSTS or host.endswith(ALLOWED_SUFFIXES))


def url_allowed(url) -> bool:
    """https to an allowed host, and an authority that is that host alone: no user info, no port, no escape.
    urlsplit and urllib.request read an authority such as `evil.com\\@huggingface.co` differently (urlsplit's
    hostname is the allowed host; urllib connects to the whole string), so only a bare host is trusted."""
    try:
        parts = urlsplit(url)
    except ValueError:
        return False
    host = parts.hostname
    return parts.scheme == "https" and host_allowed(host) and parts.netloc.lower() == host


class _Redirects(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to an allowed host over https, checked again on every hop."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not url_allowed(newurl):
            raise FetchError(f"redirect to {urlsplit(newurl).netloc} refused")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _http_reason(exc: urllib.error.HTTPError) -> str:
    remaining = exc.headers.get("x-ratelimit-remaining") if exc.headers else None
    if exc.code == 429 or (exc.code == 403 and remaining == "0"):
        return "rate limited"
    if exc.code == 401:
        return "HTTP 401 (needs a login)"
    return f"HTTP {exc.code}"


class Fetcher:
    def __init__(self, cache_dir, *, opener=None, offline: bool = False, refresh: bool = False,
                 now=None, timeout: int = TIMEOUT_S, max_requests: int = MAX_REQUESTS, deadline_s: float = DEADLINE_S,
                 clock=time.monotonic):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.opener = opener or urllib.request.build_opener(_Redirects)
        self.offline, self.refresh, self.timeout = offline, refresh, timeout
        self.max_requests, self.deadline_s, self._clock = max_requests, deadline_s, clock
        self._requests, self._started = 0, None
        # Several threads read weight headers at once (headers.read_model). The budget, the start time and the
        # per-host failure state are shared, so each is changed under this lock.
        self._lock = threading.Lock()
        # Hosts that failed at the connection level (no route, DNS failure, timeout). Later requests to the
        # same host fail at once, so a machine with no network does not wait out the timeout once per request.
        self._unreachable: set[str] = set()
        self._timeouts: dict[str, int] = {}
        self._now = now or (lambda: datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

    def now(self) -> str:
        return self._now()

    # ---- cache -----------------------------------------------------------------------------------

    def _path(self, url: str, byte_range) -> Path | None:
        if self.cache_dir is None:
            return None
        key = hashlib.sha256(f"{url}\n{byte_range}".encode()).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, url: str, byte_range) -> bytes | None:
        path = self._path(url, byte_range)
        if path is None or self.refresh or not path.is_file():
            return None
        try:
            return base64.b64decode(json.loads(path.read_text(encoding="utf-8"))["body"])
        except (OSError, ValueError, KeyError):
            return None

    def _write_cache(self, url: str, byte_range, body: bytes) -> None:
        path = self._path(url, byte_range)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"url": url, "range": byte_range, "fetched_at": self.now(),
                                        "body": base64.b64encode(body).decode()}), encoding="utf-8")
        except OSError:
            pass                                    # an unwritable cache costs a repeat request on the next run

    # ---- requests --------------------------------------------------------------------------------

    def _spend(self) -> float:
        """Count one network request against the budget and return its timeout, which never runs past the
        deadline. A spent budget is a FetchError; it is not cached and does not mark the host unreachable."""
        with self._lock:                    # the check and the increment are one step, so the budget is exact
            now = self._clock()
            if self._started is None:
                self._started = now
            left = self.deadline_s - (now - self._started)
            if self._requests >= self.max_requests:
                raise FetchError(f"the research's request budget is spent ({self.max_requests} requests)")
            if left <= 0:
                raise FetchError(f"the research's time budget is spent ({self.deadline_s:g} s)")
            self._requests += 1
            return max(1.0, min(self.timeout, left))

    def get(self, url: str, *, byte_range=None, max_bytes: int = MAX_BODY) -> bytes:
        parts = urlsplit(url)
        if not url_allowed(url):
            raise FetchError(f"{parts.scheme}://{parts.netloc} is not allowed")
        if self.offline:
            raise FetchError("offline")
        cached = self._read_cache(url, byte_range)
        if cached is not None:
            return cached
        if parts.hostname in self._unreachable:
            raise FetchError(f"network unreachable: {parts.hostname}")
        timeout = self._spend()
        headers = {"User-Agent": "tt-orchard", "Accept": "*/*"}      # no Authorization, ever
        limit = max_bytes
        if byte_range is not None:
            start, end = byte_range
            headers["Range"] = f"bytes={start}-{end}"
            limit = min(max_bytes, end - start + 1)
        try:
            resp = self.opener.open(urllib.request.Request(url, headers=headers), timeout=timeout)
        except urllib.error.HTTPError as exc:
            raise FetchError(_http_reason(exc)) from None
        except FetchError:
            raise
        except (OSError, ValueError, http.client.HTTPException) as exc:
            self._note_connection_failure(parts.hostname, exc)
            raise FetchError(f"{type(exc).__name__}: {exc}") from None
        try:
            with closing(resp):
                if byte_range is not None and getattr(resp, "status", None) != 206:
                    raise FetchError("the server ignored the byte range")
                body = resp.read(limit + 1)
        except FetchError:
            raise
        except (OSError, ValueError, http.client.HTTPException) as exc:
            # The connection opened and then broke. That says nothing about whether the host can be reached.
            raise FetchError(f"{type(exc).__name__}: {exc}") from None
        if len(body) > limit:
            raise FetchError(f"response larger than {limit} bytes")
        self._write_cache(url, byte_range, body)
        return body

    def _note_connection_failure(self, host, exc) -> None:
        """A failure to open a connection. Most of them (no route, DNS, refused) mean the host cannot be reached
        and mark it. A timeout can be one slow answer, so it marks the host only on its TIMEOUTS_TO_GIVE_UP-th
        occurrence."""
        if not isinstance(exc, OSError):
            return
        reason = getattr(exc, "reason", exc)           # urllib wraps a connect timeout in a URLError
        with self._lock:
            if isinstance(reason, TimeoutError):
                self._timeouts[host] = self._timeouts.get(host, 0) + 1
                if self._timeouts[host] < TIMEOUTS_TO_GIVE_UP:
                    return
            self._unreachable.add(host)

    def json(self, url: str, **kw):
        try:
            return json.loads(self.get(url, **kw).decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            raise FetchError("the answer is not JSON") from None
