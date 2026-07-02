"""
prescreener.py — Stage 1 two-step pre-filter
=============================================
Stage 1B — keyword filter (instant, zero cost, zero LLM)
Stage 1C — fast LLM headline pass (local, ~3–5 sec, headlines only)

Only companies that pass BOTH stages are sent to Stage 2 (Tavily deep fetch).

Design principles:
  - 1B is a hard gate: if NO signal keywords appear near the company name,
    skip immediately — no LLM needed.
  - 1C is a soft gate: a fast single-pass asking the model to score 1–5.
    Score < PRESCREEN_MIN_SCORE → skip. Score >= threshold → Stage 2.
  - Both stages operate on HEADLINES + SNIPPETS only (100–500 chars each),
    not full articles. This keeps Stage 1 cheap and fast.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import config
from classifier import LlamaServerClient


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1B — keyword signal list
# ─────────────────────────────────────────────────────────────────────────────
# These are root-forms (regex will match word boundaries + common suffixes).
# Kept intentionally broad to maximise recall at this stage —
# false positives here cost very little (one fast LLM call).

_SIGNAL_ROOTS = [
    # M&A
    r"merger", r"merging", r"merged",
    r"acqui",           # acquires, acquired, acquisition
    r"takeover",
    r"spinoff", r"spin-off", r"divest",
    r"buyout", r"buy.?out",
    # HQ / domicile
    r"relocat",         # relocated, relocation
    r"redomicil",
    r"headquarter",
    r"moved? (to|its)",
    r"new hq",
    # Renaming / rebranding
    r"renam",           # renamed, renaming
    r"rebrand",
    r"new name",
    r"formerly known",
    r"changes? (its )?name",
    # Sector change
    r"pivot",
    r"shifts? (to|into|focus)",
    r"transforms?",
    r"new (business|strategy|direction)",
    # Operational / shutdown
    r"shutdown", r"shut.?down",
    r"clos(ing|ed|es|ure)",
    r"wind(ing)?.?down",
    r"dissolv",
    r"exit",
    r"layoff", r"lay.?off",
    r"redundanc",
    r"restructur",
    r"workforce reduction",
    # Bankruptcy
    r"bankrupt",
    r"chapter.?1[13]",
    r"insolvenc",
    r"liquidat",
    r"administration",  # UK/Aus equivalent of Chapter 11
    r"receivership",
]

_KW_PATTERN = re.compile(
    r"\b(" + "|".join(_SIGNAL_ROOTS) + r")\w*",
    flags=re.IGNORECASE,
)


@dataclass
class PrescreenResult:
    company:        str
    passed:         bool
    stage:          str    # "1B_keyword" | "1C_llm" | "passed" | "no_content"
    score:          int    # 1–5 from LLM; 0 if not reached
    reason:         str    # short human-readable explanation
    headline_text:  str    # the headlines/snippets used (for debugging)


def _keyword_hit(company: str, text: str) -> tuple[bool, str]:
    """
    Returns (hit, matched_keyword).
    Requires the signal keyword to appear within 300 chars of the company name
    (case-insensitive), or anywhere in a sentence that also contains the name.
    """
    if not text:
        return False, ""

    clean_name = re.sub(r"\b(plc|ltd|llc|inc\.?|corp\.?|group|holdings?)\b\.?",
                        "", company, flags=re.IGNORECASE).strip()
    name_pattern = re.compile(re.escape(clean_name), re.IGNORECASE)

    # Check each sentence separately — keyword must co-occur with company name
    sentences = re.split(r"[.\n!?]", text)
    for sentence in sentences:
        if not name_pattern.search(sentence):
            continue
        m = _KW_PATTERN.search(sentence)
        if m:
            return True, m.group(0).lower()

    # Fallback: proximity check — keyword within 300 chars of any name mention
    for name_match in name_pattern.finditer(text):
        start = max(0, name_match.start() - 300)
        end   = min(len(text), name_match.end() + 300)
        window = text[start:end]
        m = _KW_PATTERN.search(window)
        if m:
            return True, m.group(0).lower()

    return False, ""


# ─────────────────────────────────────────────────────────────────────────────
# Stage 1C — fast LLM headline scorer
# ─────────────────────────────────────────────────────────────────────────────
_PRESCREEN_SYSTEM = (
    "You are a corporate intelligence pre-screener. "
    "Your only job is to score whether the headlines indicate a genuine "
    "corporate action directly affecting the named company. "
    "Respond with ONLY a JSON object: {\"score\": <1-5>, \"reason\": \"<10 words max>\"}\n\n"
    "Score guide:\n"
    "  1 = No signal, just general industry news\n"
    "  2 = Vague or tangential mention\n"
    "  3 = Plausible signal, needs deeper investigation\n"
    "  4 = Strong signal likely affecting the company\n"
    "  5 = Definitive confirmed corporate action\n"
    "Do not explain. Return JSON only."
)

_PRESCREEN_PROMPT = """Company: {company}

Headlines and snippets:
{headlines}

Score whether these headlines indicate a genuine corporate action directly affecting {company}.
Return JSON only: {{"score": <1-5>, "reason": "<10 words>"}}"""


def _llm_score(
    company:    str,
    headlines:  str,
    client:     LlamaServerClient,
) -> tuple[int, str]:
    """
    Calls the local llama-server with headlines only (fast pass).
    Returns (score 1–5, reason string).
    Falls back to score=3 (pass through) if LLM is unavailable.
    """
    prompt = _PRESCREEN_PROMPT.format(
        company   = company,
        headlines = headlines[:2000],  # cap at 2000 chars for speed
    )

    try:
        raw = client.complete(
            system_prompt = _PRESCREEN_SYSTEM,
            user_prompt   = prompt,
            max_tokens    = 60,
            temperature   = 0.0,
        )
    except Exception as e:
        # If LLM unavailable, pass everything through (safe fallback)
        return 3, f"LLM unavailable: {e}"

    # Parse JSON response
    import json
    # Strip markdown code fences if model wraps in ```json
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    # Extract JSON object
    m = re.search(r"\{.*?\}", raw, re.DOTALL)
    if not m:
        return 3, "parse error — defaulting to pass"
    try:
        data   = json.loads(m.group(0))
        score  = int(data.get("score", 3))
        reason = str(data.get("reason", ""))
        score  = max(1, min(5, score))  # clamp 1–5
        return score, reason
    except Exception:
        return 3, "parse error — defaulting to pass"


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────
class Prescreener:
    """
    Two-step pre-filter.

    Usage:
        ps = Prescreener()
        result = ps.check(company="Goldman Sachs", headline_text="...")
        if result.passed:
            # run Stage 2 (Tavily deep fetch + 3-pass classifier)
    """

    def __init__(self, llm_client=None):
        if llm_client is not None:
            self._client = llm_client
        elif config.USE_DATABRICKS_MODEL:
            from databricks_client import DatabricksPrescreenClient
            self._client = DatabricksPrescreenClient()
        else:
            # Small/fast local model (port 8081 by default).
            # Falls back to the classifier server if no separate one is running.
            prescreen_url = getattr(config, "LLAMA_PRESCREENER_URL", config.LLAMA_SERVER_URL)
            try:
                self._client = LlamaServerClient(url=prescreen_url)
            except ConnectionError:
                if prescreen_url == config.LLAMA_SERVER_URL:
                    raise
                print(f"  [prescreener] no server at {prescreen_url} — "
                      f"using classifier server at {config.LLAMA_SERVER_URL}")
                self._client = LlamaServerClient(url=config.LLAMA_SERVER_URL)

    def check(self, company: str, headline_text: str) -> PrescreenResult:
        """
        Run Stage 1B then 1C.  Returns a PrescreenResult with passed=True/False.
        """
        # ── no content at all ─────────────────────────────────────────────────
        if not headline_text or len(headline_text.strip()) < 50:
            return PrescreenResult(
                company       = company,
                passed        = False,
                stage         = "no_content",
                score         = 0,
                reason        = "No headlines returned from free sources",
                headline_text = headline_text,
            )

        # ── Stage 1B: keyword filter ──────────────────────────────────────────
        hit, keyword = _keyword_hit(company, headline_text)
        if not hit:
            return PrescreenResult(
                company       = company,
                passed        = False,
                stage         = "1B_keyword",
                score         = 0,
                reason        = "No signal keywords near company name",
                headline_text = headline_text,
            )

        # ── Stage 1C: LLM fast pass ───────────────────────────────────────────
        score, reason = _llm_score(company, headline_text, self._client)

        passed = score >= config.PRESCREEN_MIN_SCORE

        return PrescreenResult(
            company       = company,
            passed        = passed,
            stage         = "passed" if passed else "1C_llm",
            score         = score,
            reason        = reason,
            headline_text = headline_text,
        )
