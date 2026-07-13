"""
ratelimit.py — process-wide, thread-safe, per-host rate limiting
================================================================
With the company loop parallelised, every worker thread shares the same
external services (SEC, Google News, GDELT, GLEIF, model endpoints).
Politeness delays inside each thread no longer coordinate anything —
eight threads sleeping 0.2s each still fire eight requests at once.

This module provides ONE limiter per hostname, shared across all threads:
a caller blocks until at least `interval` seconds have passed since the
previous request to that host, whichever thread made it. Holding the
per-host lock while waiting intentionally serialises callers to the same
host; requests to different hosts proceed in parallel.

Usage:
    from ratelimit import acquire_for_url
    acquire_for_url(url)      # blocks as needed, then returns
    resp = session.get(url)
"""

from __future__ import annotations

import threading
import time
import urllib.parse

# Minimum seconds between successive requests to the same host,
# process-wide. Unlisted hosts use _DEFAULT_INTERVAL.
HOST_INTERVALS: dict[str, float] = {
    # SEC guideline is 10 req/s — stay under it across ALL threads
    "efts.sec.gov":    0.15,
    "www.sec.gov":     0.13,
    "data.sec.gov":    0.13,
    "news.google.com": 0.25,
    "api.gdeltproject.org": 0.75,
    "content.guardianapis.com": 0.25,
    "en.wikipedia.org": 0.12,
    "api.gleif.org":   1.05,   # 60 req/min documented
    "api.company-information.service.gov.uk": 0.60,  # 600 / 5 min
    "www.prnewswire.com":    0.50,
    "feed.businesswire.com": 0.50,
}
_DEFAULT_INTERVAL = 0.20

_registry_lock = threading.Lock()
_hosts: dict[str, dict] = {}


def _entry(host: str) -> dict:
    with _registry_lock:
        e = _hosts.get(host)
        if e is None:
            e = {"lock": threading.Lock(), "last": 0.0,
                 "interval": HOST_INTERVALS.get(host, _DEFAULT_INTERVAL)}
            _hosts[host] = e
        return e


def acquire(host: str) -> None:
    """Block until this thread may hit `host`, respecting its interval."""
    e = _entry(host)
    with e["lock"]:
        now  = time.monotonic()
        wait = e["interval"] - (now - e["last"])
        if wait > 0:
            time.sleep(wait)
        e["last"] = time.monotonic()


def acquire_for_url(url: str) -> None:
    host = urllib.parse.urlsplit(url).netloc.lower()
    if host:
        acquire(host)
