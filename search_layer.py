"""
search_layer.py — robust search layer for market_signals_pipeline
==================================================================
Integrates Tavily + Brave Search with:
  - Per-source rate limit handling (exponential backoff + 429 detection)
  - In-memory + on-disk result cache (avoids redundant calls within a batch)
  - Text sanitisation (HTML strip, entity decode, whitespace normalise)
  - Context chunker that fits content to the prescreener LLM window
  - Graceful degradation: either source can fail independently
  - Usage tracker: shows Tavily + Brave call counts at end of run

Architecture
------------
                    ┌─────────────────────────────────────┐
  company name  ──► │        SearchLayer.fetch()           │
                    │                                     │
                    │  1. check cache (in-memory + disk)  │
                    │  2. Stage 1 free sources (official APIs only — no scraping)│
                    │  3. Brave Search (optional, keyed)   │
                    │  4. Tavily (gated by prescreener)    │
                    │  5. sanitise + deduplicate           │
                    │  6. chunk for LLM context window     │
                    └─────────────────────────────────────┘
                                     │
                    ┌────────────────▼─────────────────────┐
                    │         SearchResult                 │
                    │  .headline_text  → prescreener       │
                    │  .full_context   → classifier        │
                    │  .chunks[]       → LLM-ready slices  │
                    │  .sources[]      → Excel citations   │
                    │  .from_cache     → bool              │
                    └──────────────────────────────────────┘

Rate limits
-----------
  Tavily free : 1,000 queries/month — guarded by prescreener gate
  Brave free  : ~1,000 queries/month (credit-based as of Feb 2026),
                1 req/sec hard limit — handled by RateLimiter
  EDGAR       : 10 req/sec guideline — handled by RateLimiter
  Google News : no published limit — polite 0.2s delay
  GDELT       : no API key, no published limit — 0.5s polite delay
  Guardian    : free API key, 5,000 calls/day — 0.2s polite delay

Cache
-----
  - Layer 1: in-memory dict (current batch, instant lookup)
  - Layer 2: on-disk JSON (across runs, configurable TTL in hours)
  - Cache key: sha256(company_name_lower + date_bucket)
  - Date bucket: YYYY-WW (weekly) so cache expires naturally each week
"""

from __future__ import annotations

import hashlib
import html
import json
import os
import re
import textwrap
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Optional

import requests

import config


# ─────────────────────────────────────────────────────────────────────────────
# Optional imports (graceful if not installed)
# ─────────────────────────────────────────────────────────────────────────────
try:
    from tavily import TavilyClient
    _TAVILY_AVAILABLE = True
except ImportError:
    _TAVILY_AVAILABLE = False

# Brave uses plain requests — no extra package needed


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SearchResult:
    """
    Unified result object passed downstream to Prescreener and Classifier.

    headline_text  — compact titles + first sentences only, for Stage 1C
    full_context   — all content merged and truncated to MAX_CONTEXT chars
    chunks         — full_context split into LLM-sized slices (each ≤ CHUNK_SIZE)
    sources        — deduplicated URL list for Excel citation column
    from_cache     — True if result was served from cache (no API calls made)
    source_breakdown — chars contributed by each source (for diagnostics)
    tavily_calls   — how many Tavily queries were fired for this company
    brave_calls    — how many Brave queries were fired for this company
    """
    company:          str
    headline_text:    str
    full_context:     str
    chunks:           list[str]          = field(default_factory=list)
    sources:          list[str]          = field(default_factory=list)
    char_count:       int                = 0
    from_cache:       bool               = False
    source_breakdown: dict[str, int]     = field(default_factory=dict)
    tavily_calls:     int                = 0
    brave_calls:      int                = 0


# ─────────────────────────────────────────────────────────────────────────────
# Rate limiter — per-source token bucket
# ─────────────────────────────────────────────────────────────────────────────
class RateLimiter:
    """
    Simple token-bucket rate limiter.
    min_interval_sec: minimum seconds between successive calls to .acquire().
    Thread-safe via Lock.
    """

    def __init__(self, min_interval_sec: float):
        self._interval = min_interval_sec
        self._last     = 0.0
        self._lock     = Lock()

    def acquire(self) -> None:
        with self._lock:
            now     = time.monotonic()
            elapsed = now - self._last
            if elapsed < self._interval:
                time.sleep(self._interval - elapsed)
            self._last = time.monotonic()


# One limiter per source — shared across all SearchLayer instances
_LIMITERS: dict[str, RateLimiter] = {
    "tavily":       RateLimiter(0.35),   # ~3 req/sec (stay under burst limit)
    "brave":        RateLimiter(1.05),   # 1 req/sec hard limit on free tier
    "edgar":        RateLimiter(0.12),   # 8 req/sec (SEC guideline is 10)
    "google_news":  RateLimiter(0.20),   # polite delay
    "gdelt":        RateLimiter(0.50),   # polite delay — no published limit
    "guardian":     RateLimiter(0.20),   # 5,000 calls/day free — polite delay
    "prnewswire":   RateLimiter(0.50),
    "businesswire": RateLimiter(0.50),
    "wikipedia":    RateLimiter(0.10),
}


# ─────────────────────────────────────────────────────────────────────────────
# Text sanitiser
# ─────────────────────────────────────────────────────────────────────────────
_HTML_TAG_RE    = re.compile(r"<[^>]+>")
_MULTI_NL_RE    = re.compile(r"\n{3,}")
_MULTI_SPACE_RE = re.compile(r"[ \t]{2,}")
_CONTROL_RE     = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitise(text: str) -> str:
    """
    Strip HTML, decode entities, normalise whitespace.
    Safe to call on any raw string from any source.
    """
    if not text:
        return ""
    text = html.unescape(text)           # &amp; → & etc.
    text = _HTML_TAG_RE.sub(" ", text)   # <b>X</b> → X
    text = _CONTROL_RE.sub("", text)     # remove control chars
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def sanitise_snippet(text: str, max_chars: int = 600) -> str:
    """Sanitise + hard truncate to max_chars. Used for individual snippets."""
    clean = sanitise(text)
    if len(clean) > max_chars:
        # Truncate at last full sentence within limit
        truncated = clean[:max_chars]
        last_dot  = truncated.rfind(". ")
        if last_dot > max_chars // 2:
            truncated = truncated[:last_dot + 1]
        return truncated + " …"
    return clean


# ─────────────────────────────────────────────────────────────────────────────
# Context chunker
# ─────────────────────────────────────────────────────────────────────────────
def chunk_context(
    full_text:  str,
    chunk_size: int = 3_000,
    max_chunks: int = 5,
    overlap:    int = 200,
) -> list[str]:
    """
    Split full_text into overlapping chunks, each ≤ chunk_size chars.
    Overlap preserves context at boundaries (avoids cutting mid-sentence).

    Returns at most max_chunks chunks.
    Total context passed to LLM = chunk_size * max_chunks = 15,000 chars default.

    Strategy:
      1. Split at paragraph boundaries (double newline) when possible.
      2. Fall back to hard split at chunk_size with overlap.
    """
    if not full_text:
        return []

    # Cap total text fed in to max_chunks * chunk_size before splitting
    cap  = chunk_size * max_chunks + overlap
    text = full_text[:cap]

    # Try paragraph-aware split first
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para)
        if current_len + para_len + 2 > chunk_size and current:
            chunk_text = "\n\n".join(current)
            chunks.append(chunk_text)
            if len(chunks) >= max_chunks:
                break
            # Start next chunk with overlap: carry last paragraph forward
            current     = current[-1:] if current else []
            current_len = len(current[0]) if current else 0

        current.append(para)
        current_len += para_len + 2  # +2 for '\n\n'

    # Flush remaining
    if current and len(chunks) < max_chunks:
        chunks.append("\n\n".join(current))

    # Safety: hard-split any oversized chunk
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= chunk_size:
            final.append(chunk)
        else:
            # Hard split with overlap
            start = 0
            while start < len(chunk) and len(final) < max_chunks:
                end = start + chunk_size
                final.append(chunk[start:end])
                start = end - overlap  # overlap step back

    return final[:max_chunks]


def build_headline_text(sections: list[str]) -> str:
    """
    Extract just the first line of each section (source tag + title).
    Used for the Stage 1C prescreener — very compact.
    """
    lines = []
    for section in sections:
        for line in section.splitlines():
            line = line.strip()
            if line.startswith("[") and line:   # e.g. [GOOGLE NEWS] Title ...
                lines.append(line)
                break                           # one line per section only
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Cache
# ─────────────────────────────────────────────────────────────────────────────
def _cache_key(company: str) -> str:
    """Weekly bucket key — cache expires naturally after 7 days."""
    week_bucket = datetime.now().strftime("%Y-W%W")
    raw         = f"{company.lower().strip()}::{week_bucket}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class SearchCache:
    """
    Two-layer cache:
      - Layer 1: in-memory dict (instant, lives for the current process run)
      - Layer 2: on-disk JSON (persists across runs, TTL = 1 week via key bucket)

    The cache stores the full SearchResult as a dict (serialisable).
    """

    def __init__(self, cache_dir: str = config.CACHE_DIR, ttl_hours: int = config.CACHE_TTL_HOURS):
        self._mem:       dict[str, dict] = {}
        self._cache_dir  = cache_dir
        self._ttl_hours  = ttl_hours
        os.makedirs(cache_dir, exist_ok=True)

    def _disk_path(self, key: str) -> str:
        return os.path.join(self._cache_dir, f"{key}.json")

    def get(self, company: str) -> Optional[SearchResult]:
        key = _cache_key(company)

        # Layer 1: memory
        if key in self._mem:
            return _dict_to_result(self._mem[key])

        # Layer 2: disk
        path = self._disk_path(key)
        if os.path.exists(path):
            try:
                age_hours = (time.time() - os.path.getmtime(path)) / 3600
                if age_hours < self._ttl_hours:
                    with open(path, encoding="utf-8") as fh:
                        data = json.load(fh)
                    self._mem[key] = data          # promote to memory
                    return _dict_to_result(data)
                else:
                    os.remove(path)                # expired
            except Exception:
                pass                               # corrupt cache — ignore

        return None

    def put(self, company: str, result: SearchResult) -> None:
        key  = _cache_key(company)
        data = _result_to_dict(result)
        self._mem[key] = data

        path = self._disk_path(key)
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(data, fh, ensure_ascii=False, indent=2)
        except Exception:
            pass  # disk write failure is non-fatal

    def invalidate(self, company: str) -> None:
        key = _cache_key(company)
        self._mem.pop(key, None)
        path = self._disk_path(key)
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


def _result_to_dict(r: SearchResult) -> dict:
    return {
        "company":          r.company,
        "headline_text":    r.headline_text,
        "full_context":     r.full_context,
        "chunks":           r.chunks,
        "sources":          r.sources,
        "char_count":       r.char_count,
        "from_cache":       True,
        "source_breakdown": r.source_breakdown,
        "tavily_calls":     r.tavily_calls,
        "brave_calls":      r.brave_calls,
    }


def _dict_to_result(d: dict) -> SearchResult:
    return SearchResult(**{k: v for k, v in d.items() if k in SearchResult.__dataclass_fields__})


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helper with backoff
# ─────────────────────────────────────────────────────────────────────────────
_HTTP = requests.Session()
_HTTP.headers.update({
    "User-Agent": f"MarketSignalsPipeline/1.0 ({config.EDGAR_USER_AGENT_EMAIL})",
    "Accept":     "application/json, text/xml, */*",
})

# Standard browser User-Agent — used by _get(use_browser_ua=True) when needed
_BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
               "AppleWebKit/537.36 (KHTML, like Gecko) "
               "Chrome/124.0.0.0 Safari/537.36")


def _get(
    url:         str,
    source:      str,
    timeout:     int  = 12,
    use_browser_ua: bool = False,
    **kwargs,
) -> Optional[requests.Response]:
    """
    Rate-limited GET with exponential backoff on 429 / 5xx.
    source: key into _LIMITERS dict.
    """
    limiter = _LIMITERS.get(source)
    if limiter:
        limiter.acquire()

    headers = kwargs.pop("headers", {})
    if use_browser_ua:
        headers["User-Agent"] = _BROWSER_UA

    for attempt in range(3):
        try:
            r = _HTTP.get(url, timeout=timeout, headers=headers, **kwargs)

            if r.status_code == 200:
                return r

            if r.status_code == 429:
                # Respect Retry-After header if present
                retry_after = int(r.headers.get("Retry-After", 2 ** (attempt + 1)))
                retry_after = min(retry_after, 30)  # cap at 30s
                print(f"    [RATE LIMIT] {source} — waiting {retry_after}s")
                time.sleep(retry_after)
                if limiter:
                    limiter.acquire()
                continue

            if r.status_code in (500, 502, 503, 504):
                time.sleep(2 ** attempt)
                continue

            return None  # 4xx other than 429 — don't retry

        except requests.Timeout:
            print(f"    [TIMEOUT] {source} attempt {attempt + 1}")
            time.sleep(1)
        except requests.RequestException as e:
            print(f"    [ERROR] {source}: {e}")
            break

    return None


def _post(
    url:     str,
    source:  str,
    timeout: int = 12,
    **kwargs,
) -> Optional[requests.Response]:
    limiter = _LIMITERS.get(source)
    if limiter:
        limiter.acquire()

    for attempt in range(3):
        try:
            r = _HTTP.post(url, timeout=timeout, **kwargs)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 2 ** (attempt + 1)))
                wait = min(wait, 30)
                print(f"    [RATE LIMIT] {source} — waiting {wait}s")
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            return None
        except requests.RequestException as e:
            print(f"    [ERROR] {source}: {e}")
            break
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Individual source fetchers
# ─────────────────────────────────────────────────────────────────────────────
def _clean_name(name: str) -> str:
    suffixes = (r"\b(plc|ltd|llc|inc\.?|corp\.?|co\.?|s\.a\.b?\.?|ag|gmbh|bv|nv|"
                r"pty|holdings?|group|international|industries|enterprises|"
                r"limited|incorporated|associates|partners)\b\.?")
    clean = re.sub(suffixes, "", name, flags=re.IGNORECASE)
    clean = re.sub(r"[^\w\s]", " ", clean)
    return " ".join(clean.split()).strip()


# ── SEC EDGAR ─────────────────────────────────────────────────────────────────
def _source_edgar(company: str, seen: set[str]) -> tuple[list[str], list[str]]:
    clean  = _clean_name(company)
    params = {
        "q":       f'"{clean}"',
        "dateRange": "custom",
        "startdt": "2024-01-01",
        "enddt":   "2026-12-31",
        "forms":   "8-K",
    }
    r = _get("https://efts.sec.gov/LATEST/search-index", "edgar", params=params)
    if not r:
        return [], []

    try:
        data = r.json()
    except Exception:
        return [], []

    hits     = data.get("hits", {}).get("hits", [])[:6]
    sections, urls = [], []
    for hit in hits:
        src  = hit.get("_source", {})
        url  = "https://www.sec.gov" + src.get("file_url", "")
        if url in seen:
            continue
        seen.add(url)
        text = sanitise(
            f"[SEC EDGAR 8-K] {src.get('entity_name', company)} — {src.get('file_date','')}\n"
            f"{src.get('file_description', '')}\n{url}"
        )
        sections.append(text)
        urls.append(url)

    return sections, urls


# ── Google News RSS ───────────────────────────────────────────────────────────
def _source_google_news(company: str, seen: set[str], max_items: int = 12) -> tuple[list[str], list[str]]:
    clean   = _clean_name(company)
    queries = [
        f'"{clean}" merger OR acquisition OR bankruptcy OR shutdown OR restructuring OR rebranding OR renamed OR relocated',
        f'"{clean}" 2025 OR 2026',
    ]
    sections, urls = [], []
    base = "https://news.google.com/rss/search"

    for query in queries:
        r = _get(f"{base}?q={urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en",
                 "google_news", timeout=10)
        if not r:
            continue
        try:
            root = ET.fromstring(r.text)
        except ET.ParseError:
            continue

        for item in root.iter("item"):
            title = sanitise(item.findtext("title") or "")
            link  = (item.findtext("link") or "").strip()
            desc  = sanitise_snippet(item.findtext("description") or "", 400)
            pub   = (item.findtext("pubDate") or "")[:22]

            if not link or link in seen:
                continue
            seen.add(link)
            sections.append(f"[GOOGLE NEWS] {title} ({pub})\n{link}\n{desc}")
            urls.append(link)
            if len(sections) >= max_items:
                return sections, urls

    return sections, urls


# ── GDELT Full Text Search API ───────────────────────────────────────────────
_GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def _source_gdelt(company: str, seen: set[str], max_items: int = 10) -> tuple[list[str], list[str]]:
    """
    GDELT 2.0 Full Text Search API.
    No API key required. No hard rate limit — use polite delay.
    Monitors 100,000+ news sources in 65 languages, updated every 15 minutes.
    Returns structured JSON — no HTML parsing, no scraping.
    Docs: https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/
    """
    clean = _clean_name(company)
    query = (f'"{clean}" AND (merger OR acquisition OR bankruptcy OR shutdown OR '
             f'restructuring OR renamed OR rebranded OR relocated OR spinoff OR '
             f'headquarters OR "new name")')

    r = _get(
        _GDELT_URL,
        "gdelt",
        params={
            "query":      query,
            "mode":       "artlist",
            "maxrecords": max_items,
            "timespan":   getattr(config, "GDELT_TIMESPAN", "12m"),
            "sort":       "datedesc",
            "format":     "json",
        },
    )
    if not r:
        return [], []

    try:
        data = r.json()
    except Exception:
        return [], []

    articles = data.get("articles", [])
    sections, urls = [], []

    for article in articles:
        url      = article.get("url", "")
        title    = sanitise(article.get("title", ""))
        domain   = article.get("domain", "")
        pub_date = article.get("seendate", "")[:8]

        if not url or url in seen:
            continue
        seen.add(url)
        sections.append(f"[GDELT] {title} ({pub_date}) — {domain}\n{url}")
        urls.append(url)

        if len(sections) >= max_items:
            break

    return sections, urls


# ── The Guardian Open Platform ────────────────────────────────────────────────
_GUARDIAN_URL = "https://content.guardianapis.com/search"


def _source_guardian(company: str, seen: set[str], max_items: int = 5) -> tuple[list[str], list[str]]:
    """
    The Guardian Open Platform API.
    Free API key: https://open-platform.theguardian.com/access/
    Default key "test" works with lower rate limits.
    5,000 calls/day on free production key.
    Covers business, M&A, corporate restructuring, bankruptcy from Guardian content.
    """
    api_key = getattr(config, "GUARDIAN_API_KEY", "test")
    clean   = _clean_name(company)

    r = _get(
        _GUARDIAN_URL,
        "guardian",
        params={
            "q":           (f'"{clean}" AND (merger OR acquisition OR bankruptcy OR '
                            f'restructuring OR renamed OR redomicile OR spinoff OR shutdown)'),
            "api-key":     api_key,
            "section":     "business",
            "order-by":    "relevance",
            "page-size":   max_items,
            "show-fields": "headline,trailText,webPublicationDate",
            "from-date":   getattr(config, "DATE_START", "2025-01-01"),
        },
    )
    if not r:
        return [], []

    try:
        data = r.json()
    except Exception:
        return [], []

    results  = data.get("response", {}).get("results", [])
    sections, urls = [], []

    for item in results:
        url    = item.get("webUrl", "")
        fields = item.get("fields", {})
        title  = sanitise(fields.get("headline", item.get("webTitle", "")))
        trail  = sanitise_snippet(fields.get("trailText", ""), 400)
        pub    = item.get("webPublicationDate", "")[:10]

        if not url or url in seen:
            continue
        seen.add(url)
        sections.append(f"[GUARDIAN] {title} ({pub})\n{url}\n{trail}")
        urls.append(url)

    return sections, urls


# ── Brave Search ──────────────────────────────────────────────────────────────
_BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"


def _source_brave(
    company:   str,
    seen:      set[str],
    api_key:   str,
    max_items: int = 8,
) -> tuple[list[str], list[str]]:
    """
    Brave Web Search API.
    Rate limit: 1 req/sec (free tier) — handled by _LIMITERS["brave"].
    Returns up to max_items results per call (1 call per company to conserve credits).
    """
    clean = _clean_name(company)
    query = (f'"{clean}" '
             f'merger OR acquisition OR bankruptcy OR shutdown OR renamed OR '
             f'rebranded OR relocated OR spinoff 2025 OR 2026')

    r = _get(
        _BRAVE_ENDPOINT,
        "brave",
        params  = {
            "q":      query,
            "count":  max_items,
            "search_lang": "en",
            "country":     "us",
            "text_decorations": 0,
            "result_filter": "web,news",
        },
        headers = {
            "Accept":              "application/json",
            "Accept-Encoding":     "gzip",
            "X-Subscription-Token": api_key,
        },
    )
    if not r:
        return [], []

    try:
        data = r.json()
    except Exception:
        return [], []

    sections, urls = [], []

    # Web results
    for item in data.get("web", {}).get("results", []):
        url     = item.get("url", "")
        title   = sanitise(item.get("title", ""))
        snippet = sanitise_snippet(item.get("description", ""), 500)
        pub     = item.get("age", "")

        if not url or url in seen:
            continue
        seen.add(url)
        sections.append(f"[BRAVE WEB] {title} ({pub})\n{url}\n{snippet}")
        urls.append(url)

    # News results (dedicated news tab if returned)
    for item in data.get("news", {}).get("results", []):
        url     = item.get("url", "")
        title   = sanitise(item.get("title", ""))
        snippet = sanitise_snippet(item.get("description", ""), 400)
        pub     = item.get("age", "")

        if not url or url in seen:
            continue
        seen.add(url)
        sections.append(f"[BRAVE NEWS] {title} ({pub})\n{url}\n{snippet}")
        urls.append(url)

    return sections[:max_items], urls[:max_items]


# ── PR Newswire RSS ───────────────────────────────────────────────────────────
def _source_prnewswire(company: str, seen: set[str], max_items: int = 4) -> tuple[list[str], list[str]]:
    r = _get("https://www.prnewswire.com/rss/news-releases-list.rss", "prnewswire", timeout=10)
    if not r:
        return [], []
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return [], []

    clean    = _clean_name(company).lower()
    sections, urls = [], []
    for item in root.iter("item"):
        title = sanitise(item.findtext("title") or "")
        link  = (item.findtext("link") or "").strip()
        desc  = sanitise_snippet(item.findtext("description") or "", 400)
        pub   = (item.findtext("pubDate") or "")[:22]
        if clean not in (title + desc).lower() or link in seen:
            continue
        seen.add(link)
        sections.append(f"[PR NEWSWIRE] {title} ({pub})\n{link}\n{desc}")
        urls.append(link)
        if len(sections) >= max_items:
            break
    return sections, urls


# ── Business Wire RSS ─────────────────────────────────────────────────────────
def _source_businesswire(company: str, seen: set[str], max_items: int = 4) -> tuple[list[str], list[str]]:
    r = _get("https://feed.businesswire.com/rss/home/?rss=G1&rssid=20", "businesswire", timeout=10)
    if not r:
        return [], []
    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return [], []

    clean    = _clean_name(company).lower()
    sections, urls = [], []
    for item in root.iter("item"):
        title = sanitise(item.findtext("title") or "")
        link  = (item.findtext("link") or "").strip()
        desc  = sanitise_snippet(item.findtext("description") or "", 400)
        pub   = (item.findtext("pubDate") or "")[:22]
        if clean not in (title + desc).lower() or link in seen:
            continue
        seen.add(link)
        sections.append(f"[BUSINESS WIRE] {title} ({pub})\n{link}\n{desc}")
        urls.append(link)
        if len(sections) >= max_items:
            break
    return sections, urls


# ── Wikipedia ─────────────────────────────────────────────────────────────────
def _source_wikipedia(company: str, seen: set[str]) -> tuple[list[str], list[str]]:
    for name in [company, _clean_name(company)]:
        slug = urllib.parse.quote(name.replace(" ", "_"))
        r    = _get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{slug}", "wikipedia", timeout=8)
        if not r:
            continue
        try:
            data = r.json()
        except Exception:
            continue
        extract  = sanitise(data.get("extract", ""))
        title    = data.get("title", "")
        page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")
        if not extract or len(extract) < 50 or page_url in seen:
            continue
        seen.add(page_url)
        text = f"[WIKIPEDIA] {title}\n{page_url}\n{extract[:1200]}"
        return [text], [page_url] if page_url else []
    return [], []


# ── Tavily ────────────────────────────────────────────────────────────────────
def _source_tavily(
    company: str,
    seen:    set[str],
    client:  "TavilyClient",
) -> tuple[list[str], list[str], int]:
    """Returns (sections, urls, n_calls)."""
    clean   = _clean_name(company)
    q       = f'"{clean}"'
    queries = [
        f"{q} 2025 OR 2026",
        f"{q} merger acquisition spinoff divestiture 2025 2026",
        f"{q} headquarters relocation redomicile 2025 2026",
        f"{q} bankruptcy shutdown liquidation restructuring 2025 2026",
        f"{q} renamed rebranded pivot new name 2025 2026",
    ]
    sections, urls = [], []
    n_calls = 0

    for i, query in enumerate(queries):
        _LIMITERS["tavily"].acquire()
        try:
            resp = client.search(
                query               = query,
                max_results         = config.TAVILY_MAX_RESULTS,
                search_depth        = config.TAVILY_SEARCH_DEPTH,
                include_raw_content = False,
            )
            n_calls += 1
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                print(f"    [TAVILY RATE LIMIT] q{i+1} — skipping remaining queries")
                break
            print(f"    [TAVILY ERROR] q{i+1}: {e}")
            continue

        for r in resp.get("results", []):
            url     = r.get("url", "")
            title   = sanitise(r.get("title", ""))
            content = sanitise_snippet(r.get("content", "") or r.get("snippet", ""), 700)
            if not url or url in seen:
                continue
            seen.add(url)
            sections.append(f"[TAVILY] {title}\n{url}\n{content}")
            urls.append(url)

    return sections, urls, n_calls


# ─────────────────────────────────────────────────────────────────────────────
# Usage tracker
# ─────────────────────────────────────────────────────────────────────────────
class UsageTracker:
    """Accumulates API call counts across the full batch run."""

    def __init__(self):
        self.tavily_calls: int = 0
        self.brave_calls:  int = 0
        self.cache_hits:   int = 0
        self.total:        int = 0

    def record(self, result: SearchResult) -> None:
        self.total        += 1
        self.tavily_calls += result.tavily_calls
        self.brave_calls  += result.brave_calls
        if result.from_cache:
            self.cache_hits += 1

    def summary(self) -> str:
        lines = [
            f"  Batch summary ({self.total} companies):",
            f"    Cache hits   : {self.cache_hits}  ({self.total - self.cache_hits} fetched live)",
            f"    Tavily calls : {self.tavily_calls}  (~{self.tavily_calls} of 1,000 free/month)",
            f"    Brave calls  : {self.brave_calls}",
        ]
        if self.tavily_calls > 800:
            lines.append("  ⚠  WARNING: >80% of free Tavily quota used this run.")
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main SearchLayer class
# ─────────────────────────────────────────────────────────────────────────────
class SearchLayer:
    """
    Public interface for the search layer.

    Instantiate once per pipeline run:
        sl = SearchLayer(tavily_key="tvly-xxx", brave_key="BSA-xxx")

    Then call per company:
        result = sl.fetch(company, stage2=False)   # Stage 1 only (free)
        result = sl.fetch(company, stage2=True)    # + Tavily deep fetch

    After the batch:
        print(sl.usage.summary())
    """

    def __init__(
        self,
        tavily_key:   Optional[str] = None,
        brave_key:    Optional[str] = None,
        cache_dir:    str  = config.CACHE_DIR,
        cache_ttl_h:  int  = config.CACHE_TTL_HOURS,
        chunk_size:   int  = config.CONTEXT_CHUNK_SIZE,
        max_chunks:   int  = config.MAX_CONTEXT_CHUNKS,
    ):
        self._chunk_size = chunk_size
        self._max_chunks = max_chunks
        self._cache      = SearchCache(cache_dir=cache_dir, ttl_hours=cache_ttl_h)
        self.usage       = UsageTracker()

        # Tavily client
        self._tavily: Optional["TavilyClient"] = None
        key = tavily_key or config.TAVILY_API_KEY
        if _TAVILY_AVAILABLE and key and not key.startswith("tvly-YOUR"):
            self._tavily = TavilyClient(api_key=key)

        # Brave key
        self._brave_key: Optional[str] = None
        bkey = brave_key or getattr(config, "BRAVE_API_KEY", "")
        if bkey and not bkey.startswith("BSA-YOUR"):
            self._brave_key = bkey

    # ── public fetch ──────────────────────────────────────────────────────────
    def fetch(
        self,
        company:  str,
        stage2:   bool = False,    # True = add Tavily deep fetch
        nocache:  bool = False,    # True = bypass cache read (still writes)
    ) -> SearchResult:
        """
        Fetch signals for one company.

        stage2=False : Stage 1 only (free sources — no Tavily, no Brave credits)
        stage2=True  : Stage 1 + Brave + Tavily deep fetch

        Result is cached; subsequent calls for the same company within the
        weekly TTL are served from cache instantly.
        """
        # ── cache lookup ──────────────────────────────────────────────────────
        if not nocache:
            cached = self._cache.get(company)
            if cached is not None:
                cached.from_cache = True
                self.usage.record(cached)
                return cached

        # ── live fetch ────────────────────────────────────────────────────────
        seen_urls:   set[str]   = set()
        all_sections: list[str] = []
        all_urls:     list[str] = []
        breakdown:    dict      = {}
        tavily_calls  = 0
        brave_calls   = 0

        def _add(source_name: str, sections: list[str], urls: list[str]) -> None:
            if sections:
                all_sections.extend(sections)
                all_urls.extend(urls)
                breakdown[source_name] = sum(len(s) for s in sections)

        # 1. EDGAR (always — most authoritative for US corps)
        _add("edgar",       *_source_edgar(company, seen_urls))

        # 2. Google News (always — free, no limit)
        _add("google_news", *_source_google_news(company, seen_urls))

        # 3. GDELT Full Text Search (always — free, no key, global news)
        _add("gdelt",       *_source_gdelt(company, seen_urls))

        # 4. The Guardian Open Platform (always — free API, business section)
        _add("guardian",    *_source_guardian(company, seen_urls))

        # 5. PR Newswire + Business Wire (always — free)
        _add("prnewswire",   *_source_prnewswire(company, seen_urls))
        _add("businesswire", *_source_businesswire(company, seen_urls))

        # 6. Wikipedia (always — free)
        _add("wikipedia",   *_source_wikipedia(company, seen_urls))

        # 6. Brave Search (Stage 2 only, if key configured)
        if stage2 and self._brave_key:
            brave_secs, brave_urls = _source_brave(company, seen_urls, self._brave_key)
            _add("brave", brave_secs, brave_urls)
            brave_calls = 1   # 1 call per company for Brave

        # 7. Tavily deep search (Stage 2 only, if key configured)
        if stage2 and self._tavily:
            tav_secs, tav_urls, n = _source_tavily(company, seen_urls, self._tavily)
            _add("tavily", tav_secs, tav_urls)
            tavily_calls = n

        # ── assemble context ──────────────────────────────────────────────────
        full_context  = "\n\n".join(all_sections)
        headline_text = build_headline_text(all_sections)
        chunks        = chunk_context(full_context, self._chunk_size, self._max_chunks)

        max_chars = self._chunk_size * self._max_chunks
        result = SearchResult(
            company          = company,
            headline_text    = headline_text,
            full_context     = full_context[:max_chars],
            chunks           = chunks,
            sources          = list(dict.fromkeys(all_urls))[:15],
            char_count       = len(full_context),
            from_cache       = False,
            source_breakdown = breakdown,
            tavily_calls     = tavily_calls,
            brave_calls      = brave_calls,
        )

        # ── cache write ───────────────────────────────────────────────────────
        self._cache.put(company, result)
        self.usage.record(result)
        return result

    # ── batch convenience ─────────────────────────────────────────────────────
    def fetch_batch_stage1(self, companies: list[str]) -> dict[str, SearchResult]:
        """
        Fetch Stage 1 (free sources) for all companies.
        Used by pipeline to populate the prescreener input cheaply.
        Returns {company_name: SearchResult}.
        """
        results = {}
        for i, company in enumerate(companies, start=1):
            print(f"  [Stage 1 fetch {i:3d}/{len(companies)}] {company}", end="  ")
            r = self.fetch(company, stage2=False)
            tag = "CACHE" if r.from_cache else f"{r.char_count:,}c"
            bd  = "  ".join(f"{k}:{v:,}" for k, v in r.source_breakdown.items())
            print(f"{tag}  [{bd}]")
            results[company] = r
        return results

    def invalidate(self, company: str) -> None:
        """Force re-fetch for a specific company on next call."""
        self._cache.invalidate(company)
