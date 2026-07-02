"""
search.py — two-stage multi-source fetcher
===========================================
Stage 1 — free sources only, headlines + snippets (no Tavily)
Stage 2 — Tavily deep fetch, only triggered for pre-screened companies

Free sources (Stage 1):
  1. SEC EDGAR 8-K filings   — authoritative US public company disclosures
  2. Google News RSS          — broad news, no key, no limit
  3. PR Newswire RSS          — company press releases
  4. Business Wire RSS        — second press release network
  5. Wikipedia REST API       — confirmed shutdowns, rebrands, redomiciles
  6. DuckDuckGo HTML search   — Google-quality results, no key, no hard limit

Stage 2 (Tavily, gated):
  - Only fires when Prescreener says passed=True
  - 5 targeted queries per company
  - Free tier: 1,000/month → at ~20% trigger rate covers 200 companies per run
    (was consuming 500 calls for 100 companies in the old design)

Install:  pip install tavily-python requests
"""

from __future__ import annotations

import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import requests

import config

try:
    from tavily import TavilyClient
    _TAVILY_OK = True
except ImportError:
    _TAVILY_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class FetchResult:
    company:          str
    context:          str                              # merged text for classifier
    sources:          list[str] = field(default_factory=list)
    char_count:       int = 0
    source_breakdown: dict = field(default_factory=dict)


@dataclass
class StageOneResult:
    """Lightweight result from Stage 1 — headlines + snippets only."""
    company:          str
    headline_text:    str                              # for Prescreener
    full_context:     str                              # for classifier if Stage 2 skipped
    sources:          list[str] = field(default_factory=list)
    char_count:       int = 0
    source_breakdown: dict = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────
def _clean_name(name: str) -> str:
    suffixes = (r"\b(plc|ltd|llc|inc\.?|corp\.?|co\.?|s\.a\.b?\.?|ag|gmbh|bv|nv|"
                r"pty|holdings?|group|international|industries|enterprises|"
                r"limited|incorporated|associates|partners)\b\.?")
    clean = re.sub(suffixes, "", name, flags=re.IGNORECASE)
    clean = re.sub(r"[^\w\s]", " ", clean)
    return " ".join(clean.split()).strip()


def _truncate(text: str, n: int = 600) -> str:
    return text[:n] if len(text) > n else text


def _year_terms() -> str:
    return getattr(config, "SEARCH_YEAR_TERMS", "2025 OR 2026")


def _date_range() -> str:
    return getattr(config, "DATE_RANGE", "Last 12 months")


_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": f"MarketSignalsPipeline/1.0 ({config.EDGAR_USER_AGENT_EMAIL})",
    "Accept":     "application/json, text/xml, */*",
})


def _get(url: str, timeout: int = 12, **kwargs) -> Optional[requests.Response]:
    for attempt in range(2):
        try:
            r = _SESSION.get(url, timeout=timeout, **kwargs)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
        except requests.RequestException:
            time.sleep(0.5)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Source 1 — SEC EDGAR 8-K (free, no key)
# ─────────────────────────────────────────────────────────────────────────────
_EDGAR_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
_EDGAR_FILING_BASE = "https://www.sec.gov"


def _fetch_edgar(company: str, max_filings: int = 5) -> tuple[str, list[str]]:
    clean = _clean_name(company)
    params = {
        "q":       f'"{clean}"',
        "dateRange": "custom",
        "startdt": getattr(config, "DATE_START", "2025-01-01"),
        "enddt":   getattr(config, "DATE_END", "2026-12-31"),
        "forms":   "8-K",
    }
    r = _get(_EDGAR_SEARCH_URL, params=params)
    if not r:
        return "", []

    try:
        data = r.json()
    except Exception:
        return "", []

    hits = data.get("hits", {}).get("hits", [])[:max_filings]
    if not hits:
        return "", []

    sections, urls = [], []
    for hit in hits:
        src         = hit.get("_source", {})
        entity_name = src.get("entity_name", company)
        file_date   = src.get("file_date", "")
        form_type   = src.get("form_type", "8-K")
        description = src.get("file_description", "")
        filing_url  = _EDGAR_FILING_BASE + src.get("file_url", "")

        text = (f"[SEC EDGAR {form_type}] {entity_name} — {file_date}\n"
                f"{description}\n{filing_url}")
        sections.append(text)
        urls.append(filing_url)

    return "\n\n".join(sections), urls


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — Google News RSS (free, no key)
# ─────────────────────────────────────────────────────────────────────────────
_GNEWS_BASE = "https://news.google.com/rss/search"


def _fetch_google_news(company: str, seen_urls: set[str], max_items: int = 12) -> tuple[str, list[str]]:
    clean = _clean_name(company)
    years = _year_terms()
    queries = [
        f'"{clean}" merger OR acquisition OR bankruptcy OR shutdown OR restructuring OR rebranding OR renamed OR relocated {years}',
        f'"{clean}" {years}',
    ]
    sections, urls = [], []

    for query in queries:
        encoded = urllib.parse.quote(query)
        url     = f"{_GNEWS_BASE}?q={encoded}&hl=en-US&gl=US&ceid=US:en"
        r       = _get(url, timeout=10)
        if not r:
            continue

        try:
            root = ET.fromstring(r.text)
        except ET.ParseError:
            continue

        for item in root.iter("item"):
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link")  or "").strip()
            desc  = (item.findtext("description") or "").strip()
            pub   = (item.findtext("pubDate") or "").strip()

            if not link or link in seen_urls:
                continue
            seen_urls.add(link)

            sections.append(f"[GOOGLE NEWS] {title} ({pub})\n{link}\n{_truncate(desc, 400)}")
            urls.append(link)

            if len(sections) >= max_items:
                break

        time.sleep(0.15)

    return "\n\n".join(sections), urls


# ─────────────────────────────────────────────────────────────────────────────
# Source 3 — DuckDuckGo HTML search (free, no key, no hard limit)
# ─────────────────────────────────────────────────────────────────────────────
_DDG_URL = "https://html.duckduckgo.com/html/"


def _fetch_ddg(company: str, seen_urls: set[str], max_items: int = 8) -> tuple[str, list[str]]:
    """
    DuckDuckGo HTML endpoint — returns search result snippets.
    No API key. Reasonable use policy applies (don't hammer it).
    Uses a browser-like User-Agent to avoid bot blocks.
    """
    clean = _clean_name(company)
    query = f'"{clean}" (merger OR acquisition OR bankrupt OR shutdown OR renamed OR "new name" OR relocated OR spinoff) {_year_terms()}'

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        r = requests.post(
            _DDG_URL,
            data    = {"q": query, "kl": "us-en"},
            headers = headers,
            timeout = 12,
        )
        if r.status_code != 200:
            return "", []
    except requests.RequestException:
        return "", []

    # Parse results from HTML using simple regex (avoids BeautifulSoup dep)
    # DuckDuckGo wraps results in <div class="result__body"> with <a class="result__url">
    result_blocks = re.findall(
        r'class="result__title"[^>]*>.*?<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>.*?'
        r'class="result__snippet"[^>]*>(.*?)</div',
        r.text, re.DOTALL
    )

    sections, urls = [], []
    for href, title_html, snippet_html in result_blocks:
        # Unescape and clean HTML
        title   = re.sub(r"<[^>]+>", "", title_html).strip()
        snippet = re.sub(r"<[^>]+>", "", snippet_html).strip()
        url     = href.strip()

        # DDG wraps real URLs in redirect — extract uddg param if present
        if "uddg=" in url:
            m = re.search(r"uddg=([^&]+)", url)
            if m:
                url = urllib.parse.unquote(m.group(1))

        if not url.startswith("http") or url in seen_urls:
            continue
        seen_urls.add(url)

        text = f"[DUCKDUCKGO] {title}\n{url}\n{_truncate(snippet, 400)}"
        sections.append(text)
        urls.append(url)

        if len(sections) >= max_items:
            break

    return "\n\n".join(sections), urls


# ─────────────────────────────────────────────────────────────────────────────
# Source 4 — PR Newswire RSS (free, no key)
# ─────────────────────────────────────────────────────────────────────────────
_PRNEWSWIRE_RSS = "https://www.prnewswire.com/rss/news-releases-list.rss"


def _fetch_prnewswire(company: str, seen_urls: set[str], max_items: int = 4) -> tuple[str, list[str]]:
    r = _get(_PRNEWSWIRE_RSS, timeout=10)
    if not r:
        return "", []

    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return "", []

    clean    = _clean_name(company).lower()
    sections, urls = [], []

    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link  = (item.findtext("link")  or "").strip()
        desc  = (item.findtext("description") or "").strip()
        pub   = (item.findtext("pubDate") or "").strip()

        if clean not in (title + " " + desc).lower():
            continue
        if link in seen_urls:
            continue
        seen_urls.add(link)

        sections.append(f"[PR NEWSWIRE] {title} ({pub})\n{link}\n{_truncate(desc, 400)}")
        urls.append(link)

        if len(sections) >= max_items:
            break

    return "\n\n".join(sections), urls


# ─────────────────────────────────────────────────────────────────────────────
# Source 5 — Business Wire RSS (free, no key)
# ─────────────────────────────────────────────────────────────────────────────
_BUSINESSWIRE_RSS = "https://feed.businesswire.com/rss/home/?rss=G1&rssid=20"


def _fetch_businesswire(company: str, seen_urls: set[str], max_items: int = 4) -> tuple[str, list[str]]:
    r = _get(_BUSINESSWIRE_RSS, timeout=10)
    if not r:
        return "", []

    try:
        root = ET.fromstring(r.text)
    except ET.ParseError:
        return "", []

    clean    = _clean_name(company).lower()
    sections, urls = [], []

    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link  = (item.findtext("link")  or "").strip()
        desc  = (item.findtext("description") or "").strip()
        pub   = (item.findtext("pubDate") or "").strip()

        if clean not in (title + " " + desc).lower():
            continue
        if link in seen_urls:
            continue
        seen_urls.add(link)

        sections.append(f"[BUSINESS WIRE] {title} ({pub})\n{link}\n{_truncate(desc, 400)}")
        urls.append(link)

        if len(sections) >= max_items:
            break

    return "\n\n".join(sections), urls


# ─────────────────────────────────────────────────────────────────────────────
# Source 6 — Wikipedia REST API (free, no key)
# ─────────────────────────────────────────────────────────────────────────────
_WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{}"


def _fetch_wikipedia(company: str, seen_urls: set[str]) -> tuple[str, list[str]]:
    for name in [company, _clean_name(company)]:
        slug = urllib.parse.quote(name.replace(" ", "_"))
        r    = _get(_WIKI_API.format(slug), timeout=8)
        if not r:
            continue
        try:
            data = r.json()
        except Exception:
            continue

        extract  = data.get("extract", "")
        title    = data.get("title", "")
        page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")

        if not extract or len(extract) < 50 or page_url in seen_urls:
            continue
        seen_urls.add(page_url)

        text = f"[WIKIPEDIA] {title}\n{page_url}\n{_truncate(extract, 1200)}"
        return text, [page_url] if page_url else []

    return "", []


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 fetcher — free sources, headlines + snippets
# ─────────────────────────────────────────────────────────────────────────────
def fetch_stage1(company: str) -> StageOneResult:
    """
    Fetch headlines and snippets from all free sources.
    Fast (~2–4 sec per company). No Tavily used.
    Returns StageOneResult with:
      - headline_text: compact, for Prescreener keyword + LLM pass
      - full_context:  full merged text, used by classifier if Stage 2 not triggered
    """
    seen_urls:   set[str]   = set()
    all_sections: list[str] = []
    all_urls:     list[str] = []
    breakdown:    dict      = {}

    # 1. EDGAR (most authoritative for US public cos)
    ctx, urls = _fetch_edgar(company)
    if ctx:
        all_sections.append("=== SEC EDGAR ===\n" + ctx)
        all_urls.extend(urls)
        seen_urls.update(urls)
        breakdown["edgar"] = len(ctx)

    # 2. Google News RSS
    ctx, urls = _fetch_google_news(company, seen_urls)
    if ctx:
        all_sections.append("=== GOOGLE NEWS ===\n" + ctx)
        all_urls.extend(urls)
        breakdown["google_news"] = len(ctx)

    # 3. DuckDuckGo
    ctx, urls = _fetch_ddg(company, seen_urls)
    if ctx:
        all_sections.append("=== WEB SEARCH ===\n" + ctx)
        all_urls.extend(urls)
        breakdown["duckduckgo"] = len(ctx)

    # 4. PR Newswire
    ctx, urls = _fetch_prnewswire(company, seen_urls)
    if ctx:
        all_sections.append("=== PR NEWSWIRE ===\n" + ctx)
        all_urls.extend(urls)
        breakdown["prnewswire"] = len(ctx)

    # 5. Business Wire
    ctx, urls = _fetch_businesswire(company, seen_urls)
    if ctx:
        all_sections.append("=== BUSINESS WIRE ===\n" + ctx)
        all_urls.extend(urls)
        breakdown["businesswire"] = len(ctx)

    # 6. Wikipedia
    ctx, urls = _fetch_wikipedia(company, seen_urls)
    if ctx:
        all_sections.append("=== WIKIPEDIA ===\n" + ctx)
        all_urls.extend(urls)
        breakdown["wikipedia"] = len(ctx)

    full_context = "\n\n".join(all_sections)
    # Headline text: just titles + first line of each section (compact for prescreener)
    headline_lines = []
    for section in all_sections:
        for line in section.splitlines():
            line = line.strip()
            if line.startswith("[") or line.startswith("==="):
                headline_lines.append(line)
    headline_text = "\n".join(headline_lines)

    max_chars = config.CONTEXT_CHUNK_SIZE * config.MAX_CONTEXT_CHUNKS

    return StageOneResult(
        company          = company,
        headline_text    = headline_text,
        full_context     = full_context[:max_chars],
        sources          = list(dict.fromkeys(all_urls))[:15],
        char_count       = len(full_context),
        source_breakdown = breakdown,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 fetcher — Tavily deep search (only for triggered companies)
# ─────────────────────────────────────────────────────────────────────────────
def _build_tavily_queries(company: str) -> list[str]:
    c = _clean_name(company)
    q = f'"{c}"'
    years = _year_terms()
    window = _date_range()
    return [
        f"{q} corporate changes {window}",
        f"{q} merger acquisition spinoff divestiture takeover {years}",
        f"{q} headquarters relocation redomicile incorporated {years}",
        f"{q} bankruptcy shutdown liquidation closure restructuring {years}",
        f"{q} renamed rebranded sector pivot new name {years}",
    ]


def fetch_stage2(
    company:     str,
    stage1:      StageOneResult,
    tavily_client: "TavilyClient",
) -> FetchResult:
    """
    Deep fetch — appends Tavily results on top of Stage 1 context.
    Only called for companies that passed the Prescreener.
    """
    seen_urls = set(stage1.sources)
    extra_sections: list[str] = []
    extra_urls:     list[str] = []

    queries = _build_tavily_queries(company)

    for i, query in enumerate(queries):
        try:
            resp = tavily_client.search(
                query               = query,
                max_results         = config.TAVILY_MAX_RESULTS,
                search_depth        = config.TAVILY_SEARCH_DEPTH,
                include_raw_content = False,
            )
        except Exception as e:
            print(f"    [Tavily q{i+1} error] {e}")
            continue

        for r in resp.get("results", []):
            url     = r.get("url", "")
            title   = r.get("title", "")
            content = r.get("content", "") or r.get("snippet", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            extra_sections.append(f"[TAVILY] {title}\n{url}\n{_truncate(content, 800)}")
            extra_urls.append(url)

        if i < len(queries) - 1:
            time.sleep(0.3)

    tavily_ctx = "\n\n".join(extra_sections)

    # Merge: Stage 1 base + Tavily enrichment
    full_context = stage1.full_context
    if tavily_ctx:
        full_context = full_context + "\n\n=== TAVILY DEEP SEARCH ===\n" + tavily_ctx

    max_chars = config.CONTEXT_CHUNK_SIZE * config.MAX_CONTEXT_CHUNKS

    breakdown = dict(stage1.source_breakdown)
    if tavily_ctx:
        breakdown["tavily"] = len(tavily_ctx)

    return FetchResult(
        company          = company,
        context          = full_context[:max_chars],
        sources          = list(dict.fromkeys(stage1.sources + extra_urls))[:15],
        char_count       = len(full_context),
        source_breakdown = breakdown,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Convenience wrapper — used by pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
class MultiSourceFetcher:
    """
    Drop-in replacement for old MultiSourceFetcher.
    Stages are now split: use fetch_stage1() and fetch_stage2() directly
    in pipeline.py for the two-stage flow.
    This wrapper runs both stages unconditionally (for backward compat).
    """

    def __init__(self, tavily_key: Optional[str] = None):
        self._tavily_client: Optional["TavilyClient"] = None
        key = tavily_key or config.TAVILY_API_KEY
        if _TAVILY_OK and key and not key.startswith("tvly-YOUR"):
            self._tavily_client = TavilyClient(api_key=key)

    def fetch(self, company: str, sector_hint: str = "") -> FetchResult:
        s1 = fetch_stage1(company)
        if self._tavily_client:
            return fetch_stage2(company, s1, self._tavily_client)
        # No Tavily — return Stage 1 as FetchResult
        return FetchResult(
            company          = company,
            context          = s1.full_context,
            sources          = s1.sources,
            char_count       = s1.char_count,
            source_breakdown = s1.source_breakdown,
        )


# Backward compat alias
TavilySearcher = MultiSourceFetcher
