"""
config.py — central configuration for market_signals_pipeline
=============================================================
Edit this file before your first run. Everything else reads from here.
"""

import calendar
import os
from datetime import date


def _bool_env(name: str, default: str = "False") -> bool:
    return os.environ.get(name, default).strip() in ("1", "True", "true", "YES", "yes")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)).strip())
    except Exception:
        return default


def _subtract_months(d: date, months: int) -> date:
    """Return the date `months` before `d`, clamping day-of-month safely."""
    month = d.month - months
    year = d.year
    while month <= 0:
        month += 12
        year -= 1
    last_day = calendar.monthrange(year, month)[1]
    return date(year, month, min(d.day, last_day))


def _build_date_window(months: int) -> tuple[str, str, str]:
    today = date.today()
    start = _subtract_months(today, months)
    return (
        start.isoformat(),
        today.isoformat(),
        f"{start.isoformat()} – {today.isoformat()} ({months} months)",
    )


# ─────────────────────────────────────────────────────────────────────────────
# SEC EDGAR  (free, no key — just needs a real email in User-Agent)
# ─────────────────────────────────────────────────────────────────────────────
# Replace with your real email — SEC blocks generic User-Agent strings
EDGAR_USER_AGENT_EMAIL: str = os.environ.get("EDGAR_EMAIL", "aetingu@gmail.com")

# ─────────────────────────────────────────────────────────────────────────────
# TAVILY SEARCH
# ─────────────────────────────────────────────────────────────────────────────
# Get a free API key at https://tavily.com  (1,000 searches/month free)
# Set here OR export as environment variable:  set TAVILY_API_KEY=tvly-xxx
TAVILY_API_KEY: str = os.environ.get("TAVILY_API_KEY", "tvly-YOUR_KEY_HERE")

# Searches per company (5 targeted queries × up to 5 results each)
TAVILY_MAX_RESULTS: int = 5       # results per query
TAVILY_SEARCH_DEPTH: str = "advanced"   # "basic" (faster) or "advanced" (better)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL BACKEND SELECTOR
# ─────────────────────────────────────────────────────────────────────────────
# True  = use Databricks Foundation Model APIs (recommended for Databricks runs)
# False = use local llama-server (for running on your Windows workstation)
USE_DATABRICKS_MODEL: bool = _bool_env("USE_DATABRICKS_MODEL", "False")

# ─────────────────────────────────────────────────────────────────────────────
# DATABRICKS FOUNDATION MODEL API / MODEL SERVING
# ─────────────────────────────────────────────────────────────────────────────
# Workspace URL — e.g. https://adb-1234567890123456.7.azuredatabricks.net
DATABRICKS_HOST:  str = os.environ.get("DATABRICKS_HOST",  "https://YOUR_WORKSPACE.azuredatabricks.net")

# Personal Access Token — generate in Databricks → Settings → Developer → Access Tokens
# Inside a Databricks notebook/job this is injected automatically by the runner.
DATABRICKS_TOKEN: str = os.environ.get("DATABRICKS_TOKEN", "")

# Endpoint for Stage 2 classifier (high-quality, larger model).
# Override from a Databricks Job parameter by setting DATABRICKS_CLASSIFIER_ENDPOINT.
# Examples:
#   databricks-qwen3-next-80b-a3b-instruct
#   qwen3-6-corporate-classifier
#   your-own-model-serving-endpoint-name
DATABRICKS_CLASSIFIER_ENDPOINT: str = os.environ.get(
    "DATABRICKS_CLASSIFIER_ENDPOINT",
    "databricks-qwen3-next-80b-a3b-instruct",
)

# Endpoint for Stage 1C prescreener (fast, cheap — fires for every company).
# Override from a Databricks Job parameter by setting DATABRICKS_PRESCREENER_ENDPOINT.
DATABRICKS_PRESCREENER_ENDPOINT: str = os.environ.get(
    "DATABRICKS_PRESCREENER_ENDPOINT",
    "databricks-meta-llama-3-1-8b-instruct",
)

# ─────────────────────────────────────────────────────────────────────────────
# LOCAL LLAMA-SERVER  (llama.cpp — any GGUF model)
# ─────────────────────────────────────────────────────────────────────────────
# This replaces the Databricks Foundation Model endpoints when
# USE_DATABRICKS_MODEL is False (the default).
#
# Two servers mirror the two Databricks endpoints:
#   Classifier  (quality, replaces qwen3-next-80b)  → port 8080
#   Prescreener (fast/cheap, replaces llama-3.1-8b) → port 8081
#
# macOS (Apple Silicon, Metal is automatic):
#   brew install llama.cpp
#
#   # Classifier — Qwen3-30B-A3B MoE, ~19 GB, fast (3B active params)
#   llama-server -hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M \
#     -c 16384 --port 8080
#
#   # Prescreener — Llama-3.1-8B, ~5 GB
#   llama-server -hf bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M \
#     -c 8192 --port 8081
#
# Or run scripts/run_local_models.sh to start both.
#
# If you only want ONE server, point both URLs at the same port —
# the classifier model handles prescreening fine, just slower.
LLAMA_SERVER_URL:      str = os.environ.get("LLAMA_SERVER_URL",      "http://127.0.0.1:8080")
LLAMA_PRESCREENER_URL: str = os.environ.get("LLAMA_PRESCREENER_URL", "http://127.0.0.1:8081")
LLAMA_SERVER_TIMEOUT: int = 180    # seconds — increase for large models / slow CPU

# ─────────────────────────────────────────────────────────────────────────────
# INPUT / OUTPUT
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_INPUT_CSV:  str = "company_list_sample.csv"
DEFAULT_OUTPUT_XLS: str = "market_signals_report.xlsx"

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE BEHAVIOUR
# ─────────────────────────────────────────────────────────────────────────────
# Time window used in search queries, classifier prompts, and report summaries.
# Databricks notebook/job can override with TIME_HORIZON_MONTHS = 6, 12, or 24.
_requested_months = _int_env("TIME_HORIZON_MONTHS", 12)
TIME_HORIZON_MONTHS: int = _requested_months if _requested_months in (6, 12, 24) else 12
_DEFAULT_DATE_START, _DEFAULT_DATE_END, _DEFAULT_DATE_RANGE = _build_date_window(TIME_HORIZON_MONTHS)
DATE_START: str = os.environ.get("DATE_START", _DEFAULT_DATE_START)
DATE_END:   str = os.environ.get("DATE_END",   _DEFAULT_DATE_END)
DATE_RANGE: str = os.environ.get("DATE_RANGE", _DEFAULT_DATE_RANGE)

# Useful search term string, e.g. "2025 OR 2026".
try:
    _start_year = int(DATE_START[:4])
    _end_year = int(DATE_END[:4])
    SEARCH_YEAR_TERMS: str = " OR ".join(str(y) for y in range(_start_year, _end_year + 1))
except Exception:
    SEARCH_YEAR_TERMS = "2025 OR 2026"

# Set to 0 to process all rows. Use 10 or 15 for testing.
DEFAULT_MAX_COMPANIES: int = max(0, _int_env("MAX_COMPANIES", 0))

# Minimum LLM confidence (1–5) to accept a signal.
# Raise to 3–4 for higher precision. Lower to 1 for maximum recall.
MIN_CONFIDENCE: int = 2

# Three-pass classifier (higher quality, ~3× slower).
# Set False to use single-pass (faster, ~70% quality).
USE_IMPROVED_CLASSIFIER: bool = True

# Characters of fetched content sent to the LLM per chunk
# With 6 sources now feeding context, increase the window to fit more content.
# 5 chunks × 3,000 = 15,000 chars — well within Qwen3.6's 8K token context
# (15,000 chars ≈ 3,750 tokens at ~4 chars/token).
CONTEXT_CHUNK_SIZE: int = 3_000
MAX_CONTEXT_CHUNKS: int = 5       # = 15,000 total chars (was 3 → 9,000)

# Seconds to wait between companies (avoids Tavily rate limits)
INTER_COMPANY_DELAY: float = 1.0

# ─────────────────────────────────────────────────────────────────────────────
# BRAVE SEARCH API  (optional — $5 monthly credit ~1,000 queries)
# Get key at: https://api-dashboard.search.brave.com
# Leave as-is if you don’t have a Brave key — pipeline works without it.
# ─────────────────────────────────────────────────────────────────────────────
BRAVE_API_KEY: str = os.environ.get("BRAVE_API_KEY", "BSA-YOUR_KEY_HERE")

# ─────────────────────────────────────────────────────────────────────────────
# GDELT FULL TEXT SEARCH API  (completely free, no key required)
# Monitors 100,000+ news sources in 65 languages, updated every 15 minutes.
# Endpoint: https://api.gdeltproject.org/api/v2/doc/doc
# No registration. No rate limit published — use 0.5s delay to be polite.
# Docs: https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/
# ─────────────────────────────────────────────────────────────────────────────
GDELT_MAX_RECORDS: int = 10      # articles per query (max 250)
GDELT_TIMESPAN: str = "12m"      # GDELT timespan param: 15min,1h,1d,1w,1m,12m etc.
GDELT_DELAY: float = 0.5         # polite delay between GDELT calls (seconds)

# ─────────────────────────────────────────────────────────────────────────────
# THE GUARDIAN OPEN PLATFORM  (free, 5,000 calls/day — key required)
# Register at: https://open-platform.theguardian.com/access/
# Covers: business, M&A, bankruptcy, corporate restructuring from Guardian content.
# Leave as-is if you don't have a key — pipeline falls back to GDELT-only.
# ─────────────────────────────────────────────────────────────────────────────
GUARDIAN_API_KEY: str = os.environ.get("GUARDIAN_API_KEY", "test")
# Note: "test" is a valid public key with lower rate limits.
# Register for a free production key for full 5,000 calls/day access.
GUARDIAN_MAX_RESULTS: int = 5    # articles per company query
GUARDIAN_DELAY: float = 0.2      # polite delay (seconds)

# ─────────────────────────────────────────────────────────────────────────────
# CORPORATE REGISTRY LOOKUPS  (free official APIs — ground-truth legal facts)
# ─────────────────────────────────────────────────────────────────────────────
# Runs at Stage 1 for every company. Registry facts (previous legal names,
# entity status, successor entities, liquidation status) feed both the
# prescreener and the classifier.
REGISTRY_ENABLED: bool = _bool_env("REGISTRY_ENABLED", "True")

# GLEIF LEI API — global, free, no key. Rate limit 60 req/min.
# NOTE: at 60/min this adds ~1s per company; a 5,000-company run spends
# ~85 min on GLEIF alone. Disable via REGISTRY_ENABLED=False for quick tests.
GLEIF_DELAY: float = 1.05

# UK Companies House — free key from
# https://developer.company-information.service.gov.uk
# Applies to UK-HQ companies only. Leave empty to skip Companies House.
COMPANIES_HOUSE_API_KEY: str = os.environ.get("COMPANIES_HOUSE_API_KEY", "")
COMPANIES_HOUSE_DELAY: float = 0.6   # 600 req / 5 min limit

# ─────────────────────────────────────────────────────────────────────────────
# FREE FULL-TEXT ENRICHMENT  (official APIs only — no Tavily credits needed)
# ─────────────────────────────────────────────────────────────────────────────
# When the prescreener flags a company, fetch full text from free official
# sources: actual SEC 8-K filing documents, Guardian article bodies
# (show-fields=body), and full Wikipedia articles. Runs alongside Tavily if
# a key is configured, or entirely instead of it if not.
FULLTEXT_ENABLED: bool = _bool_env("FULLTEXT_ENABLED", "True")

# SEC EDGAR filing documents (free, no key, 10 req/s guideline)
FULLTEXT_EDGAR_MAX_DOCS:      int   = _int_env("FULLTEXT_EDGAR_MAX_DOCS", 3)
FULLTEXT_EDGAR_CHARS_PER_DOC: int   = 4_000
FULLTEXT_EDGAR_DELAY:         float = 0.15   # seconds between doc fetches

# Guardian full article bodies (free key, 5,000 calls/day)
FULLTEXT_GUARDIAN_MAX_ARTICLES:      int = 3
FULLTEXT_GUARDIAN_CHARS_PER_ARTICLE: int = 3_000

# Wikipedia full article (free, official MediaWiki Action API)
FULLTEXT_WIKIPEDIA_MAX_CHARS: int = 4_000

# Chars of Stage 1 headline context preserved when full text is appended.
# The rest of the context window (CONTEXT_CHUNK_SIZE × MAX_CONTEXT_CHUNKS)
# is given to full text. Headlines keep breadth; full text adds depth.
FULLTEXT_STAGE1_BUDGET: int = 6_000

# ─────────────────────────────────────────────────────────────────────────────
# CACHE
# ─────────────────────────────────────────────────────────────────────────────
# Search results are cached to avoid redundant API calls within a batch.
# Cache lives alongside the output file. Each company is keyed by name + week
# number so cache auto-expires after 7 days.
CACHE_DIR:       str = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".search_cache")
CACHE_TTL_HOURS: int = 168   # 7 days

# ─────────────────────────────────────────────────────────────────────────────
# TWO-STAGE PRESCREENER
# ─────────────────────────────────────────────────────────────────────────────
# Minimum LLM score (1–5) to trigger Stage 2 Tavily deep fetch.
# 3 = plausible signal (recommended — balances recall vs Tavily usage)
# 4 = strong signal only (saves more Tavily credits, slightly lower recall)
PRESCREEN_MIN_SCORE: int = 3

# Set True to log prescreener decisions to a CSV alongside the output XLSX.
# Useful for tuning PRESCREEN_MIN_SCORE on your company list.
PRESCREEN_LOG: bool = True

# ─────────────────────────────────────────────────────────────────────────────
# EXCEL STYLE CONSTANTS  (exact values from reference file)
# ─────────────────────────────────────────────────────────────────────────────
# Dashboard column widths  [A, B, C, D, E, F, G, H, I, J, K]
DASH_COL_WIDTHS = [38, 11, 11, 10, 12, 14, 13, 9, 70, 13, 11]

# Detail sheet column widths  [Company, Detail, Sources]
DETAIL_COL_WIDTHS_DEFAULT = [34, 82, 52]
DETAIL_COL_WIDTHS_HQ      = [34, 14, 52, 52]  # extra USA Region col

HEADER_ROW_HEIGHT:   float = 36.0
DATA_ROW_HEIGHT:     float = 55.0   # default; auto-expand for long content

# Hex colours
COLOR_HEADER_FILL   = "1F4E79"   # dark navy
COLOR_ALT_ROW       = "EAF1FB"   # light blue
COLOR_WHITE_ROW     = "FFFFFF"
COLOR_TICK          = "1F7A1F"   # dark green
COLOR_DASH          = "9E9E9E"   # mid grey
COLOR_COMPANY_TEXT  = "1F4E79"   # navy bold
COLOR_BODY_TEXT     = "1A1A1A"   # near-black
COLOR_SOURCES_TEXT  = "505D6B"   # steel grey
COLOR_REGION_TEXT   = "1F4E79"   # navy bold (region labels)
COLOR_TOTAL_TEXT    = "1F4E79"   # navy bold (signal count)
COLOR_BORDER        = "C9D6E8"   # light blue-grey border

# USA region groups
USA_REGIONS = {
    "USA West", "USA New York", "USA Midwest",
    "USA South", "USA Boston",  "USA Northeast",
}
