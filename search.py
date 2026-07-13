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
  6. GDELT Full Text Search   — global news, no key, no limit, 65 languages
  7. The Guardian Open Platform — business/M&A news API, free key, 5,000/day

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
from ratelimit import acquire_for_url

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
            acquire_for_url(url)   # process-wide, per-host, thread-safe
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


def _edgar_filing_url(hit: dict) -> str:
    """
    Build the document URL from an EDGAR full-text-search hit.
    (The FTS response has no 'file_url' field — the URL must be derived
    from _id = "<accession>:<filename>" plus the CIK.)
    """
    src = hit.get("_source", {})
    _id = hit.get("_id", "")
    if ":" not in _id:
        return ""
    adsh, filename = _id.split(":", 1)
    ciks = src.get("ciks") or []
    if not ciks or not filename:
        return ""
    try:
        cik = int(ciks[0])
    except (TypeError, ValueError):
        return ""
    return (f"{_EDGAR_FILING_BASE}/Archives/edgar/data/{cik}/"
            f"{adsh.replace('-', '')}/{filename}")


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
        entity_name = (src.get("display_names") or [company])[0]
        file_date   = src.get("file_date", "")
        form_type   = src.get("form", "8-K")
        description = src.get("file_description", "")
        items       = ", ".join(src.get("items") or [])
        filing_url  = _edgar_filing_url(hit)
        if not filing_url:
            continue

        text = (f"[SEC EDGAR {form_type}] {entity_name} — {file_date}"
                f"{'  (items ' + items + ')' if items else ''}\n"
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
# Source 3 — GDELT Full Text Search API (free, no key, global news)
# ─────────────────────────────────────────────────────────────────────────────
_GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"


def _fetch_gdelt(company: str, seen_urls: set[str], max_items: int = 10) -> tuple[str, list[str]]:
    """
    GDELT 2.0 Full Text Search API — no API key, no hard rate limit.
    Monitors 100,000+ news sources in 65 languages, updated every 15 minutes.
    Returns article titles + URLs sorted by relevance.
    Docs: https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/
    """
    clean = _clean_name(company)
    _kw = ("merger OR acquisition OR bankruptcy OR shutdown OR "
           "restructuring OR renamed OR rebranded OR relocated OR spinoff OR "
           'headquarters OR "new name"')
    query = f'"{clean}" AND ({_kw})'

    params = {
        "query":      query,
        "mode":       "artlist",       # return article list (not timeline/wordcloud)
        "maxrecords": max_items,
        "timespan":   getattr(config, "GDELT_TIMESPAN", "12m"),
        "sort":       "datedesc",      # newest first
        "format":     "json",
    }

    try:
        acquire_for_url(_GDELT_URL)
        r = _SESSION.get(_GDELT_URL, params=params, timeout=12)
        if r.status_code != 200:
            return "", []
        data = r.json()
    except Exception:
        return "", []

    articles = data.get("articles", [])
    sections, urls = [], []

    for article in articles:
        url      = article.get("url", "")
        title    = article.get("title", "").strip()
        domain   = article.get("domain", "")
        pub_date = article.get("seendate", "")[:8]  # YYYYMMDD

        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        text = f"[GDELT] {title} ({pub_date}) — {domain}\n{url}"
        sections.append(text)
        urls.append(url)

        if len(sections) >= max_items:
            break

    time.sleep(getattr(config, "GDELT_DELAY", 0.5))
    return "\n\n".join(sections), urls


# ─────────────────────────────────────────────────────────────────────────────
# Source 4 (alt) — The Guardian Open Platform (free API, 5,000 calls/day)
# ─────────────────────────────────────────────────────────────────────────────
_GUARDIAN_URL = "https://content.guardianapis.com/search"


def _fetch_guardian(company: str, seen_urls: set[str], max_items: int = 5) -> tuple[str, list[str]]:
    """
    The Guardian Open Platform API.
    Free key: https://open-platform.theguardian.com/access/
    Default key "test" works but with tighter rate limits.
    Covers business, M&A, corporate restructuring, bankruptcy from Guardian content.
    """
    api_key = getattr(config, "GUARDIAN_API_KEY", "test")
    clean   = _clean_name(company)

    _gkw = ("merger OR acquisition OR bankruptcy OR restructuring OR "
            'renamed OR redomicile OR spinoff OR shutdown OR "new name"')
    params = {
        "q":           f'"{clean}" AND ({_gkw})',
        "api-key":     api_key,
        "section":     "business",      # business section only — more relevant
        "order-by":    "relevance",
        "page-size":   max_items,
        "show-fields": "headline,trailText,webPublicationDate",
        "from-date":   getattr(config, "DATE_START", "2025-01-01"),
    }

    try:
        acquire_for_url(_GUARDIAN_URL)
        r = _SESSION.get(_GUARDIAN_URL, params=params, timeout=10)
        if r.status_code != 200:
            return "", []
        data = r.json()
    except Exception:
        return "", []

    results  = data.get("response", {}).get("results", [])
    sections, urls = [], []

    for item in results:
        url    = item.get("webUrl", "")
        fields = item.get("fields", {})
        title  = fields.get("headline", item.get("webTitle", "")).strip()
        trail  = fields.get("trailText", "").strip()
        pub    = item.get("webPublicationDate", "")[:10]

        if not url or url in seen_urls:
            continue
        seen_urls.add(url)

        text = f"[GUARDIAN] {title} ({pub})\n{url}\n{_truncate(trail, 400)}"
        sections.append(text)
        urls.append(url)

    time.sleep(getattr(config, "GUARDIAN_DELAY", 0.2))
    return "\n\n".join(sections), urls


# ─────────────────────────────────────────────────────────────────────────────
# Source 5 — PR Newswire RSS (free, no key)
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
# Source 6 — Business Wire RSS (free, no key)
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
# Source 7 — Wikipedia REST API (free, no key)
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

    # 3. GDELT Full Text Search (free, no key, global news)
    ctx, urls = _fetch_gdelt(company, seen_urls)
    if ctx:
        all_sections.append("=== GDELT NEWS ===\n" + ctx)
        all_urls.extend(urls)
        breakdown["gdelt"] = len(ctx)

    # 4. The Guardian (free API key, business section)
    ctx, urls = _fetch_guardian(company, seen_urls)
    if ctx:
        all_sections.append("=== THE GUARDIAN ===\n" + ctx)
        all_urls.extend(urls)
        breakdown["guardian"] = len(ctx)

    # 5. PR Newswire
    ctx, urls = _fetch_prnewswire(company, seen_urls)
    if ctx:
        all_sections.append("=== PR NEWSWIRE ===\n" + ctx)
        all_urls.extend(urls)
        breakdown["prnewswire"] = len(ctx)

    # 6. Business Wire
    ctx, urls = _fetch_businesswire(company, seen_urls)
    if ctx:
        all_sections.append("=== BUSINESS WIRE ===\n" + ctx)
        all_urls.extend(urls)
        breakdown["businesswire"] = len(ctx)

    # 7. Wikipedia
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
# Tavily targeted fetch — thin-evidence gate (raw content, budget-capped)
# ─────────────────────────────────────────────────────────────────────────────
def fetch_tavily_targeted(
    company:       str,
    seen_urls:     set[str],
    tavily_client: "TavilyClient",
    max_queries:   int,
) -> tuple[list[str], list[str], int]:
    """
    Credit-efficient Tavily fetch for companies whose FREE evidence came
    back thin (typically non-US/private companies with headline-only news).

    Differences from the legacy fetch_stage2:
      - 1-2 broad queries instead of 5 narrow ones
      - include_raw_content=True: actual article text, not 800-char
        snippets — the full-text depth the free stack couldn't provide
      - caller passes max_queries from a process-wide budget; returns the
        number actually spent so unused reservations can be refunded

    Returns (sections, urls, queries_spent).
    """
    c = _clean_name(company)
    years = _year_terms()
    queries = [
        f'"{c}" merger OR acquisition OR takeover OR divestiture OR spinoff {years}',
        f'"{c}" renamed OR rebranded OR bankruptcy OR liquidation OR restructuring OR relocated {years}',
    ][:max(0, max_queries)]

    raw_cap  = getattr(config, "TAVILY_RAW_CHARS_PER_RESULT", 3000)
    sections: list[str] = []
    urls:     list[str] = []
    spent = 0

    for query in queries:
        try:
            resp = tavily_client.search(
                query               = query,
                max_results         = min(config.TAVILY_MAX_RESULTS, 3),
                search_depth        = config.TAVILY_SEARCH_DEPTH,
                include_raw_content = True,
            )
            spent += 1
        except Exception as e:
            print(f"    [Tavily targeted error] {company}: {e}")
            break

        for r in resp.get("results", []):
            url = r.get("url", "")
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            title = (r.get("title") or "").strip()
            body  = (r.get("raw_content") or r.get("content") or "").strip()
            if not body:
                continue
            sections.append(f"[TAVILY FULL TEXT] {title}\n{url}\n{_truncate(body, raw_cap)}")
            urls.append(url)

        time.sleep(0.3)

    return sections, urls, spent


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
