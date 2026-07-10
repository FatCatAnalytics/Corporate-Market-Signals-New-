"""
registry.py — corporate registry lookups (free official APIs, no credits)
=========================================================================
News tells you what journalists wrote; registries tell you what legally
happened. For this pipeline's signal types — renames, redomiciles, M&A,
dissolutions, bankruptcies — registries are ground truth and directly
corroborate (or refute) news claims.

Sources:
  1. GLEIF LEI API (global, free, no key)
       https://api.gleif.org/api/v1/lei-records
     Legal name + PREVIOUS_LEGAL_NAME entries, entity status
     (ACTIVE/INACTIVE), expiration reason (DISSOLVED / CORPORATE_ACTION),
     successor entities, legal & HQ addresses.
     Rate limit: 60 req/min → 1.05 s delay.

  2. SEC submissions API (US registrants, free, no key)
       https://data.sec.gov/submissions/CIK##########.json
     `formerNames` with from/to dates — an authoritative rename detector.
     Company matching is done locally against the SEC ticker file
     (downloaded once per run), so most companies cost zero extra calls.

  3. UK Companies House API (UK companies, free key required)
       https://api.company-information.service.gov.uk
     company_status (liquidation / administration / dissolved),
     previous_company_names with change dates, registered office.
     Enabled only when COMPANIES_HOUSE_API_KEY is set.
     Rate limit: 600 req / 5 min → 0.6 s delay.

Matching caution
----------------
Registry name search is fuzzy and returns pension trusts, funds, and
subsidiaries (e.g. "Kellanova" → "ACTIONS KELLANOVA", a French fund).
Lookups therefore filter by the company's HQ country where known and
score candidates by token overlap; anything below threshold is dropped.
The matched legal name is always included in the output so the
downstream classifier can see exactly which entity the facts refer to.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

import config

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": f"MarketSignalsPipeline/1.0 ({config.EDGAR_USER_AGENT_EMAIL})",
    "Accept":     "application/json, application/vnd.api+json",
})


@dataclass
class RegistryResult:
    company:   str
    context:   str = ""                                # full sections for classifier
    headline:  str = ""                                # compact lines for prescreener
    sources:   list[str] = field(default_factory=list)
    breakdown: dict      = field(default_factory=dict)
    flags:     list[str] = field(default_factory=list)  # machine-readable signals


# ─────────────────────────────────────────────────────────────────────────────
# Country mapping — CSV "Country HQ" values → ISO 3166-1 alpha-2
# ─────────────────────────────────────────────────────────────────────────────
_COUNTRY_ISO = {
    "united kingdom": "GB", "japan": "JP", "china": "CN", "germany": "DE",
    "france": "FR", "canada": "CA", "india": "IN", "brazil": "BR",
    "south korea": "KR", "switzerland": "CH", "italy": "IT",
    "hong kong": "HK", "australia": "AU", "netherlands": "NL",
    "taiwan": "TW", "spain": "ES", "mexico": "MX", "singapore": "SG",
    "sweden": "SE", "belgium": "BE", "ireland": "IE", "malaysia": "MY",
    "south africa": "ZA", "united arab emirates": "AE", "indonesia": "ID",
    "austria": "AT", "norway": "NO", "turkey": "TR", "luxembourg": "LU",
    "denmark": "DK", "russia": "RU", "poland": "PL", "thailand": "TH",
    "israel": "IL", "chile": "CL", "finland": "FI", "philippines": "PH",
    "new zealand": "NZ", "saudi arabia": "SA", "greece": "GR",
    "colombia": "CO", "peru": "PE", "portugal": "PT", "argentina": "AR",
    "bermuda": "BM", "czech republic": "CZ", "vietnam": "VN",
    "kuwait": "KW", "hungary": "HU", "kazakhstan": "KZ", "egypt": "EG",
    "qatar": "QA", "jordan": "JO", "oman": "OM",
    "trinidad & tobago": "TT", "trinidad and tobago": "TT",
    "bahrain": "BH", "cyprus": "CY", "panama": "PA", "kenya": "KE",
    "uruguay": "UY", "mauritius": "MU", "lebanon": "LB", "jamaica": "JM",
    "guatemala": "GT", "costa rica": "CR", "pakistan": "PK",
    "cayman islands": "KY", "liechtenstein": "LI", "ukraine": "UA",
    "gabon": "GA", "ivory coast": "CI", "zambia": "ZM", "bulgaria": "BG",
    "macedonia": "MK", "north macedonia": "MK", "bangladesh": "BD",
    "sri lanka": "LK", "algeria": "DZ", "zimbabwe": "ZW", "estonia": "EE",
    "ethiopia": "ET", "haiti": "HT", "iceland": "IS", "morocco": "MA",
    "venezuela": "VE", "azerbaijan": "AZ", "romania": "RO", "nigeria": "NG",
    "usa": "US", "united states": "US",
}


def country_to_iso(country_hint: str) -> Optional[str]:
    """Map a CSV 'Country HQ' value to ISO alpha-2. USA regions → US."""
    if not country_hint:
        return None
    c = country_hint.strip().lower()
    if c.startswith("usa") or c.startswith("us "):
        return "US"
    return _COUNTRY_ISO.get(c)


# ─────────────────────────────────────────────────────────────────────────────
# Name normalisation + candidate scoring
# ─────────────────────────────────────────────────────────────────────────────
_LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "plc",
    "ltd", "limited", "llc", "llp", "lp", "sa", "sau", "sab", "se", "ag",
    "nv", "bv", "spa", "sl", "slu", "gmbh", "kk", "ab", "asa", "oyj",
    "aps", "pjsc", "psc", "jsc", "pt", "tbk", "bhd", "berhad", "pty",
    "sarl", "sas", "kgaa", "wll", "holdings", "holding", "group",
}


def _norm_tokens(name: str) -> tuple[str, ...]:
    name = name.lower()
    # Collapse dotted abbreviations first ("S.P.A." → "spa", "N.V" → "nv")
    # so legal-form suffixes are recognised instead of shattering into
    # single letters that wreck the match score.
    name = re.sub(r"\.", "", name)
    name = re.sub(r"[^\w\s]", " ", name)
    toks = [t for t in name.split() if t and t not in _LEGAL_SUFFIXES]
    return tuple(toks) if toks else tuple(name.split())


def _match_score(query: str, candidate: str) -> float:
    """
    3.0 exact (normalised) · ~2.x query ⊆ candidate (subsidiary/trust risk,
    penalised per extra token) · ~1.x candidate ⊆ query · else Jaccard.
    """
    q, c = _norm_tokens(query), _norm_tokens(candidate)
    if not q or not c:
        return 0.0
    if q == c:
        return 3.0
    qs, cs = set(q), set(c)
    if qs <= cs:
        return max(0.0, 2.2 - 0.3 * len(cs - qs))
    if cs <= qs:
        return max(0.0, 1.6 - 0.3 * len(qs - cs))
    inter = len(qs & cs)
    return 1.2 * inter / len(qs | cs)


_MATCH_THRESHOLD = 1.5


def _rate_limit(last: list[float], min_interval: float) -> None:
    now = time.monotonic()
    wait = min_interval - (now - last[0])
    if wait > 0:
        time.sleep(wait)
    last[0] = time.monotonic()


def _get(url: str, timeout: int = 15, **kwargs) -> Optional[requests.Response]:
    for attempt in range(2):
        try:
            r = _SESSION.get(url, timeout=timeout, **kwargs)
            if r.status_code == 200:
                return r
            if r.status_code == 429:
                time.sleep(3 * (attempt + 1))
                continue
            return None
        except requests.RequestException:
            time.sleep(0.5)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Source 1 — GLEIF LEI records (global)
# ─────────────────────────────────────────────────────────────────────────────
_GLEIF_URL   = "https://api.gleif.org/api/v1/lei-records"
_gleif_last  = [0.0]


def lookup_gleif(
    company:     str,
    iso_country: Optional[str],
    lei:         str = "",
) -> tuple[str, str, list[str], list[str]]:
    """
    Returns (context_section, headline_line, source_urls, flags).

    When the input list supplies an LEI, the record is fetched directly —
    deterministic, no fuzzy-match risk. Name search is the fallback.
    """
    _rate_limit(_gleif_last, getattr(config, "GLEIF_DELAY", 1.05))

    best, best_score = None, 0.0

    if lei:
        r = _get(f"{_GLEIF_URL}/{lei}")
        if r:
            try:
                best = r.json().get("data") or None
                best_score = 3.0   # exact by construction
            except Exception:
                best = None

    if best is None:
        params = {"filter[fulltext]": company, "page[size]": 10}
        if iso_country:
            params["filter[entity.legalAddress.country]"] = iso_country
        r = _get(_GLEIF_URL, params=params)
        if not r:
            return "", "", [], []
        try:
            records = r.json().get("data", [])
        except Exception:
            return "", "", [], []

        for rec in records:
            ent   = rec.get("attributes", {}).get("entity", {})
            lname = (ent.get("legalName") or {}).get("name", "")
            score = _match_score(company, lname)
            if score > best_score:
                best, best_score = rec, score
        if best is None or best_score < _MATCH_THRESHOLD:
            return "", "", [], []

    attrs = best.get("attributes", {})
    ent   = attrs.get("entity", {})
    reg   = attrs.get("registration", {})

    successor  = ent.get("successorEntity") or {}
    successors = ent.get("successorEntities") or []
    expiration = ent.get("expiration") or {}
    legal_addr = ent.get("legalAddress") or {}
    hq_addr    = ent.get("headquartersAddress") or {}

    rec = {
        "lei":           attrs.get("lei", ""),
        "legal_name":    (ent.get("legalName") or {}).get("name", ""),
        "status":        ent.get("status", ""),
        "reg_status":    reg.get("status", ""),
        "last_update":   str(reg.get("lastUpdateDate", ""))[:10],
        "prev_names":    [o.get("name") for o in (ent.get("otherNames") or [])
                          if o.get("type") == "PREVIOUS_LEGAL_NAME" and o.get("name")],
        "succ_names":    [n for n in
                          [(successor.get("name") if isinstance(successor, dict) else None)]
                          + [(s.get("name") if isinstance(s, dict) else None) for s in successors]
                          if n],
        "exp_date":      expiration.get("date") or "",
        "exp_reason":    expiration.get("reason") or "",
        "legal_city":    legal_addr.get("city", "?"),
        "legal_country": legal_addr.get("country", ""),
        "hq_city":       hq_addr.get("city", "?"),
        "hq_country":    hq_addr.get("country", ""),
        "parent_name":   "",   # only available via the Level 2 Delta path
    }
    return format_gleif_record(company, rec, iso_country)


def format_gleif_record(
    company:     str,
    rec:         dict,
    iso_country: Optional[str],
) -> tuple[str, str, list[str], list[str]]:
    """
    Render a normalised GLEIF record dict into (context_section,
    headline_line, source_urls, flags). Shared by the live-API path
    (lookup_gleif) and the Delta-table path (registry_delta.py).
    """
    lei   = rec.get("lei", "")
    lname = rec.get("legal_name", "")

    # Previous legal names — drop punctuation-only or translation-only
    # variants (same normalised tokens as the current name), which would
    # otherwise read as fake rename signals downstream.
    prev_names = [p for p in (rec.get("prev_names") or [])
                  if p and _norm_tokens(p) != _norm_tokens(lname)]
    succ_names = [s for s in (rec.get("succ_names") or []) if s]
    status     = rec.get("status", "")

    flags = []
    lines = [f"Matched legal entity: {lname} (LEI {lei})",
             f"Entity status: {status or 'unknown'} | Registration: {rec.get('reg_status','')}"]
    if prev_names:
        lines.append(f"PREVIOUS LEGAL NAME(S): {'; '.join(prev_names)}")
        flags.append("gleif_previous_name")
    if status == "INACTIVE":
        flags.append("gleif_inactive")
    if rec.get("exp_reason"):
        lines.append(f"Entity expired: {rec.get('exp_date','')} — reason: {rec['exp_reason']}")
        flags.append(f"gleif_expired_{rec['exp_reason'].lower()}")
    if succ_names:
        lines.append(f"SUCCESSOR ENTITY: {'; '.join(succ_names)}")
        flags.append("gleif_successor")
    if rec.get("parent_name"):
        lines.append(f"Ultimate parent: {rec['parent_name']}")
    lines.append(f"HQ: {rec.get('hq_city','?')}, {rec.get('hq_country','?')} | "
                 f"Legal seat: {rec.get('legal_city','?')}, {rec.get('legal_country','?')}")
    if iso_country and rec.get("legal_country") and rec["legal_country"] != iso_country:
        lines.append(f"NOTE: legal seat country {rec['legal_country']} differs "
                     f"from expected {iso_country} (possible redomicile or subsidiary match)")
        flags.append("gleif_country_mismatch")
    if rec.get("last_update"):
        lines.append(f"Record last updated: {rec['last_update']}")

    url      = f"https://search.gleif.org/#/record/{lei}"
    section  = f"[GLEIF REGISTRY] {company}\n{url}\n" + "\n".join(lines)
    hl_bits  = [f"status={status}"]
    if prev_names: hl_bits.append(f"prev-name: {prev_names[0]}")
    if succ_names: hl_bits.append(f"successor: {succ_names[0]}")
    # Lead with the CSV company name — the prescreener's keyword gate
    # requires signal words to co-occur with the *input* name, which can
    # differ from the registry legal name.
    headline = f"[GLEIF REGISTRY] {company}: {lname} — " + " | ".join(hl_bits)
    return section, headline, [url], flags


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — SEC submissions formerNames (US registrants)
# ─────────────────────────────────────────────────────────────────────────────
_SEC_TICKERS_URL     = "https://www.sec.gov/files/company_tickers.json"
_SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
_sec_last            = [0.0]
_sec_name_to_cik: Optional[dict[tuple[str, ...], int]] = None


def _load_sec_ticker_map() -> dict[tuple[str, ...], int]:
    """Download the SEC ticker file once and index by normalised name."""
    global _sec_name_to_cik
    if _sec_name_to_cik is not None:
        return _sec_name_to_cik
    _sec_name_to_cik = {}
    r = _get(_SEC_TICKERS_URL, timeout=20)
    if r:
        try:
            for item in r.json().values():
                toks = _norm_tokens(item.get("title", ""))
                if toks and toks not in _sec_name_to_cik:
                    _sec_name_to_cik[toks] = int(item.get("cik_str", 0))
        except Exception:
            pass
    return _sec_name_to_cik


_SEC_FTS_URL = "https://efts.sec.gov/LATEST/search-index"

# display_names look like "KELLANOVA  (K)  (CIK 0000055067)" — ticker optional
_DISPLAY_NAME_RE = re.compile(r"^(.*?)\s*(?:\([^)]*\)\s*)?\(CIK (\d+)\)\s*$")


def _sec_cik_by_fts(company: str) -> Optional[int]:
    """
    Resolve a CIK via EDGAR full-text search display_names. Unlike the
    ticker file, this still finds DELISTED registrants — companies that
    were acquired or went private (i.e. exactly the ones with signals),
    because their historical filings remain indexed.
    (The legacy browse-edgar atom API is broken for multi-match results —
    it emits Perl 'ARRAY(0x..)' artifacts instead of company names.)
    """
    _rate_limit(_sec_last, 0.15)
    r = _get(_SEC_FTS_URL, params={"q": f'"{company}"'})
    if not r:
        return None
    try:
        hits = r.json().get("hits", {}).get("hits", [])
    except Exception:
        return None

    best_cik, best_score = None, 0.0
    for hit in hits:
        for dn in hit.get("_source", {}).get("display_names") or []:
            m = _DISPLAY_NAME_RE.match(dn)
            if not m:
                continue
            score = _match_score(company, m.group(1))
            if score > best_score:
                best_cik, best_score = int(m.group(2)), score
    return best_cik if best_score >= _MATCH_THRESHOLD else None


def lookup_sec(company: str) -> tuple[str, str, list[str], list[str]]:
    """
    Name match against SEC registrants: first the local ticker file
    (zero API calls, current registrants only), then EDGAR full-text
    search (1 call, includes delisted registrants).
    """
    name_map = _load_sec_ticker_map()
    cik = name_map.get(_norm_tokens(company)) if name_map else None
    if not cik:
        cik = _sec_cik_by_fts(company)
    if not cik:
        return "", "", [], []

    _rate_limit(_sec_last, 0.15)
    r = _get(_SEC_SUBMISSIONS_URL.format(cik=cik))
    if not r:
        return "", "", [], []
    try:
        data = r.json()
    except Exception:
        return "", "", [], []

    current  = data.get("name", company)
    # Drop punctuation-only / identical former-name records (SEC updates
    # the record row on corporate events without an actual name change).
    formers  = [f for f in (data.get("formerNames") or [])
                if _norm_tokens(f.get("name", "")) != _norm_tokens(current)]
    state    = data.get("stateOfIncorporation", "")
    flags    = []
    lines    = [f"Current registrant name: {current} (CIK {cik})"]
    if state:
        lines.append(f"State/country of incorporation: {state}")
    for fn in formers[:5]:
        frm = str(fn.get("from", ""))[:10]
        to  = str(fn.get("to",   ""))[:10]
        lines.append(f"FORMER NAME: {fn.get('name','')} ({frm} → {to})")
    if formers:
        flags.append("sec_former_name")

    url      = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik:010d}"
    section  = f"[SEC REGISTRY] {company}\n{url}\n" + "\n".join(lines)
    headline = f"[SEC REGISTRY] {company}: {current}"
    if formers:
        headline += f" — former name: {formers[-1].get('name','')}"
    return section, headline, [url], flags


# ─────────────────────────────────────────────────────────────────────────────
# Source 3 — UK Companies House (key-gated)
# ─────────────────────────────────────────────────────────────────────────────
_CH_SEARCH_URL  = "https://api.company-information.service.gov.uk/search/companies"
_CH_COMPANY_URL = "https://api.company-information.service.gov.uk/company/{number}"
_ch_last        = [0.0]

_CH_DISTRESS_STATUSES = {
    "liquidation", "administration", "insolvency-proceedings",
    "receivership", "dissolved", "voluntary-arrangement",
}


def lookup_companies_house(company: str) -> tuple[str, str, list[str], list[str]]:
    api_key = getattr(config, "COMPANIES_HOUSE_API_KEY", "")
    if not api_key:
        return "", "", [], []

    _rate_limit(_ch_last, getattr(config, "COMPANIES_HOUSE_DELAY", 0.6))
    r = _get(_CH_SEARCH_URL, params={"q": company, "items_per_page": 5},
             auth=(api_key, ""))
    if not r:
        return "", "", [], []
    try:
        items = r.json().get("items", [])
    except Exception:
        return "", "", [], []

    best, best_score = None, 0.0
    for item in items:
        score = _match_score(company, item.get("title", ""))
        if score > best_score:
            best, best_score = item, score
    if best is None or best_score < _MATCH_THRESHOLD:
        return "", "", [], []

    number = best.get("company_number", "")
    _rate_limit(_ch_last, getattr(config, "COMPANIES_HOUSE_DELAY", 0.6))
    r = _get(_CH_COMPANY_URL.format(number=number), auth=(api_key, ""))
    if not r:
        return "", "", [], []
    try:
        prof = r.json()
    except Exception:
        return "", "", [], []

    status     = prof.get("company_status", "")
    detail     = prof.get("company_status_detail", "")
    prev_names = prof.get("previous_company_names") or []
    office     = prof.get("registered_office_address") or {}

    flags = []
    lines = [f"Matched company: {prof.get('company_name', best.get('title',''))} "
             f"(No. {number})",
             f"Status: {status}{' — ' + detail if detail else ''}"]
    if status in _CH_DISTRESS_STATUSES:
        flags.append(f"ch_status_{status}")
    for pn in prev_names[:4]:
        lines.append(f"PREVIOUS NAME: {pn.get('name','')} "
                     f"(until {pn.get('ceased_on','')})")
    if prev_names:
        flags.append("ch_previous_name")
    locality = ", ".join(x for x in (office.get("locality"),
                                     office.get("country")) if x)
    if locality:
        lines.append(f"Registered office: {locality}")

    url      = f"https://find-and-update.company-information.service.gov.uk/company/{number}"
    section  = f"[COMPANIES HOUSE] {company}\n{url}\n" + "\n".join(lines)
    headline = f"[COMPANIES HOUSE] {company}: {prof.get('company_name','')} — status={status}"
    if prev_names:
        headline += f" | prev-name: {prev_names[0].get('name','')}"
    return section, headline, [url], flags


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def lookup(
    company:      str,
    country_hint: str = "",
    gleif_map:    Optional[dict] = None,
    lei:          str = "",
) -> RegistryResult:
    """
    Run all applicable registry lookups for one company.
    country_hint is the raw CSV 'Country HQ' value (may be '').
    lei is the company's LEI from the input CSV ('' if absent) — when
    present, the GLEIF API record is fetched directly by ID.

    gleif_map: prebuilt {company: (section, headline, urls, flags)} from
    registry_delta.build_gleif_map (Databricks Delta tables). When given,
    the GLEIF API is NOT called — a miss in the full golden copy means the
    entity has no LEI, so there is nothing to fall back to.
    """
    iso    = country_to_iso(country_hint)
    result = RegistryResult(company=company)
    sections, headlines = [], []

    if gleif_map is not None:
        fetchers = [("gleif", lambda: gleif_map.get(company) or ("", "", [], []))]
    else:
        fetchers = [("gleif", lambda: lookup_gleif(company, iso, lei))]
    if iso == "US":
        fetchers.append(("sec_registry", lambda: lookup_sec(company)))
    if iso == "GB":
        fetchers.append(("companies_house", lambda: lookup_companies_house(company)))

    for tag, fn in fetchers:
        try:
            section, headline, urls, flags = fn()
        except Exception as e:
            print(f"    [REGISTRY {tag} error] {e}")
            continue
        if section:
            sections.append(section)
            headlines.append(headline)
            result.sources.extend(urls)
            result.flags.extend(flags)
            result.breakdown[tag] = len(section)

    result.context  = "\n\n".join(sections)
    result.headline = "\n".join(headlines)
    return result
