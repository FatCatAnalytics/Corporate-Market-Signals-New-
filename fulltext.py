"""
fulltext.py — free full-text enrichment (official APIs only, no scraping)
=========================================================================
Deep-context alternative to Tavily that costs nothing and never runs out
of credits. Used when the prescreener flags a company, either alongside
Tavily or entirely without it.

Sources (all official APIs, no HTML scraping of third-party news sites):
  1. SEC EDGAR      — full text of actual 8-K filing documents & exhibits.
                      Free, no key (needs real email in User-Agent).
                      The single most authoritative source for M&A,
                      bankruptcy, renames, redomiciles of US registrants.
  2. The Guardian   — full article body via the Open Platform API
                      (show-fields=body). Free key, 5,000 calls/day.
  3. Wikipedia      — full plaintext article via the MediaWiki Action API
                      (prop=extracts&explaintext). Free, official.

What this module deliberately does NOT do:
  - No fetching of arbitrary news-site HTML (paywalls, bot-blockers,
    robots.txt issues). GDELT / Google News URLs stay headline-only.

Budgeting
---------
The classifier context window is capped at
CONTEXT_CHUNK_SIZE * MAX_CONTEXT_CHUNKS chars (15,000 default).
Full text could easily exceed that on its own, so enrichment reserves
room: the Stage 1 headline context is kept up to FULLTEXT_STAGE1_BUDGET
chars and full-text sections fill the remainder. Headlines keep breadth,
full text adds depth.
"""

from __future__ import annotations

import html as _html
import re
import time
import urllib.parse
from typing import Optional

import requests

import config
from search import FetchResult, StageOneResult, _clean_name

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": f"MarketSignalsPipeline/1.0 ({config.EDGAR_USER_AGENT_EMAIL})",
    "Accept":     "application/json, text/html, text/plain, */*",
})

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_TAG_RE          = re.compile(r"<[^>]+>")


def _strip_html(raw: str) -> str:
    """Convert an HTML document to readable plaintext (no external deps)."""
    if not raw:
        return ""
    # Bound the work on pathological multi-MB filings
    raw  = raw[:1_500_000]
    text = _SCRIPT_STYLE_RE.sub(" ", raw)
    # Preserve some block structure before stripping tags
    text = re.sub(r"</(p|div|tr|table|h[1-6]|li|br)[^>]*>", "\n", text, flags=re.I)
    text = _TAG_RE.sub(" ", text)
    text = _html.unescape(text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text.strip()


def _get(url: str, timeout: int = 15, **kwargs) -> Optional[requests.Response]:
    for attempt in range(2):
        try:
            r = _SESSION.get(url, timeout=timeout, **kwargs)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                time.sleep(2 * (attempt + 1))
                continue
            return None
        except requests.RequestException:
            time.sleep(0.5)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Source 1 — SEC EDGAR full filing text
# ─────────────────────────────────────────────────────────────────────────────
_EDGAR_FTS_URL = "https://efts.sec.gov/LATEST/search-index"


def _edgar_doc_url(hit: dict) -> str:
    """
    Build the primary document URL from an EDGAR full-text-search hit.
    _id has the form  "<accession-with-dashes>:<filename>"
    e.g. 0000055067-25-000190:exhibit991q3-2025.htm  →
    https://www.sec.gov/Archives/edgar/data/55067/000005506725000190/exhibit991q3-2025.htm
    """
    src  = hit.get("_source", {})
    _id  = hit.get("_id", "")
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
    return (f"https://www.sec.gov/Archives/edgar/data/{cik}/"
            f"{adsh.replace('-', '')}/{filename}")


def fetch_edgar_fulltext(
    company:       str,
    seen:          set[str],
    max_docs:      int = None,
    chars_per_doc: int = None,
) -> tuple[list[str], list[str]]:
    """
    Search EDGAR 8-K filings for the company and download the actual
    filing documents / exhibits (press releases filed as EX-99.1 are the
    richest signal source). One document per distinct filing (adsh).
    """
    max_docs      = max_docs      or config.FULLTEXT_EDGAR_MAX_DOCS
    chars_per_doc = chars_per_doc or config.FULLTEXT_EDGAR_CHARS_PER_DOC

    clean = _clean_name(company)
    r = _get(_EDGAR_FTS_URL, params={
        "q":         f'"{clean}"',
        "forms":     "8-K",
        "dateRange": "custom",
        "startdt":   getattr(config, "DATE_START", "2025-01-01"),
        "enddt":     getattr(config, "DATE_END",   "2026-12-31"),
    })
    if not r:
        return [], []
    try:
        hits = r.json().get("hits", {}).get("hits", [])
    except Exception:
        return [], []

    # EDGAR FTS matches the phrase anywhere in a filing, including filings
    # BY OTHER registrants that merely mention the target company (often
    # the M&A counterparty — useful; often pure noise — an earnings release
    # naming a customer). Handle both:
    #   - filings BY the company: take the document head (most relevant)
    #   - third-party mentions:   take a window AROUND the mention,
    #     processed only after own filings, so noise never crowds out signal.
    name_token = re.sub(r"[^a-z0-9 ]", "", clean.lower())

    def _is_own_filing(src: dict) -> bool:
        names = " ".join(src.get("display_names") or []).lower()
        names = re.sub(r"[^a-z0-9 ]", "", names)
        return name_token in names

    hits_sorted = sorted(hits, key=lambda h: not _is_own_filing(h.get("_source", {})))

    sections, urls = [], []
    seen_adsh: set[str] = set()

    for hit in hits_sorted:
        if len(sections) >= max_docs:
            break
        src  = hit.get("_source", {})
        adsh = src.get("adsh", "")
        if adsh in seen_adsh:
            continue
        url = _edgar_doc_url(hit)
        if not url:
            continue
        # Note: do NOT skip URLs already cited at Stage 1 — Stage 1 only
        # carried the filing description; the document text is new.
        if url.lower().endswith((".jpg", ".png", ".gif", ".pdf", ".xml", ".xsd")):
            continue

        entity = (src.get("display_names") or [company])[0]

        time.sleep(getattr(config, "FULLTEXT_EDGAR_DELAY", 0.15))
        doc = _get(url)
        if not doc:
            continue
        text = _strip_html(doc.text)
        if len(text) < 200:
            continue

        if _is_own_filing(src):
            excerpt = text[:chars_per_doc]
            tag     = "SEC FILING FULL TEXT"
        else:
            # Third-party filing: extract the window around the mention.
            pos = text.lower().find(clean.lower())
            if pos < 0:
                continue  # phrase split across markup — skip as unverifiable
            start   = max(0, pos - chars_per_doc // 3)
            excerpt = text[start : start + chars_per_doc]
            tag     = "SEC FILING MENTION"

        seen_adsh.add(adsh)
        seen.add(url)
        header = (f"[{tag}] {entity} — "
                  f"{src.get('form', '8-K')} {src.get('file_type', '')} "
                  f"filed {src.get('file_date', '')}\n{url}")
        sections.append(f"{header}\n{excerpt}")
        urls.append(url)

    return sections, urls


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — Guardian full article body
# ─────────────────────────────────────────────────────────────────────────────
_GUARDIAN_URL = "https://content.guardianapis.com/search"


def fetch_guardian_fulltext(
    company:       str,
    seen:          set[str],
    max_items:     int = None,
    chars_per_doc: int = None,
) -> tuple[list[str], list[str]]:
    """
    Same Guardian query as Stage 1, but requests show-fields=body which
    returns the COMPLETE article text through the official API.
    """
    max_items     = max_items     or config.FULLTEXT_GUARDIAN_MAX_ARTICLES
    chars_per_doc = chars_per_doc or config.FULLTEXT_GUARDIAN_CHARS_PER_ARTICLE

    api_key = getattr(config, "GUARDIAN_API_KEY", "test")
    clean   = _clean_name(company)
    _gkw = ("merger OR acquisition OR bankruptcy OR restructuring OR "
            'renamed OR redomicile OR spinoff OR shutdown OR "new name"')

    # Guardian phrase search is strict (e.g. "and" does not match "&"),
    # and the keyword filter can kill recall entirely. Try targeted first,
    # then fall back to phrase-only in the business section — downstream
    # classification decides relevance either way.
    queries = [
        {"q": f'"{clean}" AND ({_gkw})', "section": "business"},
        {"q": f'"{clean}"',              "section": "business"},
    ]
    if "and" in clean.lower().split():
        amp = re.sub(r"\band\b", "&", clean, flags=re.IGNORECASE)
        queries.insert(1, {"q": f'"{amp}" AND ({_gkw})', "section": "business"})

    results = []
    for qp in queries:
        r = _get(_GUARDIAN_URL, timeout=10, params={
            **qp,
            "api-key":     api_key,
            "order-by":    "relevance",
            "page-size":   max_items,
            "show-fields": "headline,body",
            "from-date":   getattr(config, "DATE_START", "2025-01-01"),
        })
        if not r:
            continue
        try:
            results = r.json().get("response", {}).get("results", [])
        except Exception:
            continue
        if results:
            break
        time.sleep(getattr(config, "GUARDIAN_DELAY", 0.2))
    if not results:
        return [], []

    sections, urls = [], []
    for item in results:
        url    = item.get("webUrl", "")
        fields = item.get("fields", {})
        title  = (fields.get("headline") or item.get("webTitle", "")).strip()
        body   = _strip_html(fields.get("body", ""))
        pub    = item.get("webPublicationDate", "")[:10]
        if not url or not body:
            continue
        # Do not skip URLs already seen at Stage 1 — Stage 1 only had the
        # trail text; the body is new information. Dedupe within this call.
        if url in urls:
            continue
        seen.add(url)
        sections.append(f"[GUARDIAN FULL TEXT] {title} ({pub})\n{url}\n"
                        f"{body[:chars_per_doc]}")
        urls.append(url)

    time.sleep(getattr(config, "GUARDIAN_DELAY", 0.2))
    return sections, urls


# ─────────────────────────────────────────────────────────────────────────────
# Source 3 — Wikipedia full article
# ─────────────────────────────────────────────────────────────────────────────
_WIKI_ACTION_API = "https://en.wikipedia.org/w/api.php"


def fetch_wikipedia_fulltext(
    company:   str,
    seen:      set[str],
    max_chars: int = None,
) -> tuple[list[str], list[str]]:
    """
    Full plaintext article via the MediaWiki Action API (prop=extracts,
    explaintext). Corporate history / ownership sections routinely record
    renames, acquisitions, HQ moves and shutdowns that summaries omit.
    """
    max_chars = max_chars or config.FULLTEXT_WIKIPEDIA_MAX_CHARS

    for name in [company, _clean_name(company)]:
        r = _get(_WIKI_ACTION_API, timeout=10, params={
            "action":      "query",
            "prop":        "extracts",
            "explaintext": 1,
            "redirects":   1,
            "format":      "json",
            "titles":      name,
        })
        if not r:
            continue
        try:
            pages = r.json().get("query", {}).get("pages", {})
        except Exception:
            continue

        for page_id, page in pages.items():
            if page_id == "-1":
                continue
            extract = (page.get("extract") or "").strip()
            title   = page.get("title", name)
            if len(extract) < 300:
                continue
            page_url = ("https://en.wikipedia.org/wiki/"
                        + urllib.parse.quote(title.replace(" ", "_")))
            seen.add(page_url)
            return ([f"[WIKIPEDIA FULL TEXT] {title}\n{page_url}\n"
                     f"{extract[:max_chars]}"], [page_url])

    return [], []


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator — enrich an existing FetchResult with free full text
# ─────────────────────────────────────────────────────────────────────────────
def enrich_with_fulltext(company: str, base: FetchResult) -> FetchResult:
    """
    Append free full-text context to a FetchResult (Stage 1-only or
    Stage 1 + Tavily). Reserves room in the context window so full text
    is not silently truncated away by the headline sections.
    """
    seen: set[str] = set(base.sources)
    sections: list[str] = []
    urls:     list[str] = []
    breakdown = dict(base.source_breakdown)

    for fetcher, tag in (
        (fetch_edgar_fulltext,     "edgar_fulltext"),
        (fetch_guardian_fulltext,  "guardian_fulltext"),
        (fetch_wikipedia_fulltext, "wikipedia_fulltext"),
    ):
        try:
            secs, u = fetcher(company, seen)
        except Exception as e:
            print(f"    [FULLTEXT {tag} error] {e}")
            continue
        if secs:
            sections.extend(secs)
            urls.extend(u)
            breakdown[tag] = sum(len(s) for s in secs)

    if not sections:
        return base

    max_chars    = config.CONTEXT_CHUNK_SIZE * config.MAX_CONTEXT_CHUNKS
    fulltext_ctx = "\n\n".join(sections)

    # Reserve room: keep headlines up to FULLTEXT_STAGE1_BUDGET, let full
    # text fill the rest of the window.
    reserve = min(len(fulltext_ctx) + 40,
                  max_chars - config.FULLTEXT_STAGE1_BUDGET)
    reserve = max(reserve, 0)
    base_ctx = base.context[: max_chars - reserve]

    context = (base_ctx
               + "\n\n=== FULL TEXT (FREE OFFICIAL SOURCES) ===\n"
               + fulltext_ctx)[:max_chars]

    return FetchResult(
        company          = company,
        context          = context,
        sources          = list(dict.fromkeys(base.sources + urls))[:15],
        char_count       = len(base.context) + len(fulltext_ctx),
        source_breakdown = breakdown,
    )


def stage1_to_fetchresult(s1: StageOneResult) -> FetchResult:
    """Convenience: wrap a StageOneResult as a FetchResult for enrichment."""
    return FetchResult(
        company          = s1.company,
        context          = s1.full_context,
        sources          = s1.sources,
        char_count       = s1.char_count,
        source_breakdown = dict(s1.source_breakdown),
    )
