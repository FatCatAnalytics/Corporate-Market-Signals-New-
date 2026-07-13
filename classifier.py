"""
classifier.py — llama-server classifier factory with 3-pass detection
======================================================================
Connects to a locally running llama-server instance via HTTP.
Works with ANY model served by llama-server (Qwen3.6-35B-A3B-MTP,
Qwen3-14B, Mistral, etc.) — the server handles all model specifics.

Three-pass strategy (when USE_IMPROVED_CLASSIFIER=True):
  Pass 1 — Broad detection per chunk: "which signals exist?"
  Pass 2 — Deep extraction per signal: one focused prompt each
  Pass 3 — Self-verification: model checks its own answers

Single-pass fallback (USE_IMPROVED_CLASSIFIER=False):
  One combined prompt, faster but ~15-20% lower recall.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Optional

import requests

import config


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class SignalResult:
    company:             str
    sector_change:       bool = False
    hq_change:           bool = False
    hq_region:           str  = ""
    ma_spinoff:          bool = False
    renaming:            bool = False
    operational_change:  bool = False
    shutdown:            bool = False
    bankruptcy:          bool = False
    total_signals:       int  = 0

    sector_detail:       str = ""
    hq_detail:           str = ""
    ma_detail:           str = ""
    rename_detail:       str = ""
    ops_detail:          str = ""
    bankruptcy_detail:   str = ""
    summary:             str = ""
    sources:             list[str] = field(default_factory=list)

    def recount(self) -> None:
        """Recompute total_signals from boolean flags."""
        self.total_signals = sum([
            self.sector_change, self.hq_change, self.ma_spinoff,
            self.renaming, self.operational_change, self.bankruptcy,
        ])


# ─────────────────────────────────────────────────────────────────────────────
# llama-server HTTP backend
# ─────────────────────────────────────────────────────────────────────────────
class LlamaServerClient:
    """
    Thin OpenAI-compatible client for llama-server.
    Raises ConnectionError if server is not reachable at startup.
    """

    def __init__(
        self,
        url:     str = config.LLAMA_SERVER_URL,
        timeout: int = config.LLAMA_SERVER_TIMEOUT,
    ):
        self.completions_url = url.rstrip("/") + "/v1/chat/completions"
        self.health_url      = url.rstrip("/") + "/health"
        self.timeout         = timeout
        self._verify()

    def _verify(self) -> None:
        for attempt in range(3):
            try:
                r = requests.get(self.health_url, timeout=5)
                if r.status_code == 200:
                    print(f"  llama-server OK  →  {self.health_url.replace('/health','')}")
                    return
            except requests.ConnectionError:
                pass
            time.sleep(2)
        raise ConnectionError(
            f"\nllama-server not reachable at {self.health_url}\n\n"
            "Start it first (see scripts/run_local_models.sh), e.g.:\n"
            "  llama-server -hf unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M \\\n"
            "    -c 16384 --port 8080\n"
        )

    def chat(
        self,
        messages:    list[dict],
        max_tokens:  int   = 600,
        temperature: float = 0.0,
    ) -> str:
        """Call the server and return the assistant message content."""
        payload = {
            "messages":    messages,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "stream":      False,
        }
        try:
            r = requests.post(
                self.completions_url,
                json    = payload,
                timeout = self.timeout,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except requests.Timeout:
            print("  [llama-server timeout — returning empty]")
            return "{}"
        except Exception as e:
            print(f"  [llama-server error] {e}")
            return "{}"

    def complete(
        self,
        system_prompt: str,
        user_prompt:   str,
        max_tokens:    int   = 1200,
        temperature:   float = 0.0,
    ) -> str:
        """
        Same interface as DatabricksModelClient.complete(), so the two
        backends are interchangeable everywhere in the pipeline.
        """
        return self.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens  = max_tokens,
            temperature = temperature,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────
_SYSTEM = (
    "/no_think\n"
    "You are a senior corporate intelligence analyst. Be precise and conservative.\n"
    "Only flag signals explicitly supported by the text provided.\n"
    "Respond ONLY with valid JSON — no commentary, no markdown fences."
)

_SIGNAL_DEFS = """\
SIGNAL DEFINITIONS — route each distinct event to exactly ONE signal, the most
specific one that fits. Work down this list in order; the FIRST match wins.

- bankruptcy       : bankruptcy, administration, insolvency, or liquidation
                     filing (any jurisdiction's equivalent).
- shutdown         : entity fully wound down / dissolved / ceased to exist
                     with no successor operations.
- ma_spinoff       : any merger, acquisition, takeover, spinoff, or major
                     divestiture in which the company is EITHER the acquirer
                     OR the target/divested party. The company buying another
                     company IS an ma_spinoff signal. Announced or pending
                     deals count — note the deal status in the detail.
                     An acquisition, merger or divestiture is NEVER an
                     operational_change, even when framed as "strategy" or
                     "portfolio restructuring around a deal".
- renaming         : legal name change or major corporate rebrand (old name →
                     new name). If a rename accompanies a deal, report the
                     rename here AND the deal under ma_spinoff — that is the
                     one case where a story yields two signals, because it is
                     two events.
- hq_change        : HQ relocation or legal redomicile to another country or
                     state. Not branch/office openings or closings.
- sector_change    : the company's PRIMARY industry or business focus changed.
- operational_change: ONLY material operational events: a named restructuring
                     programme, significant site/plant closures, or workforce
                     reductions (>5% of staff or thousands of jobs).
                     NOT operational_change: executive hires or departures,
                     earnings results, guidance, product launches, store
                     openings, partnerships, buybacks, dividends, ordinary
                     capex, or generic "transformation" talk without concrete
                     closures/cuts. When in doubt, leave it out.
"""

_PASS1 = """\
Read the text about "{company}" and identify which corporate change signals are present.
Only consider corporate changes inside this time window: {date_range}.

""" + _SIGNAL_DEFS + """
TEXT:
{context}

Respond with JSON only:
{{"detected": ["signal_name", ...], "confidence": 1-5}}
If nothing: {{"detected": [], "confidence": 1}}"""

_PASS2 = """\
Signal "{signal}" was detected for company "{company}".
Extract the precise details from the text below.
Only confirm the signal if it happened inside this time window: {date_range}.

TEXT:
{context}

Respond with JSON only:
{{
  "confirmed": true/false,
  "detail": "one sentence: what happened, when, who was involved — cite a date if present",
  "hq_region": "if hq_change: USA West | USA New York | USA Midwest | USA South | USA Boston | USA Northeast — or country name. Else empty string.",
  "is_shutdown": true/false,
  "confidence": 1-5
}}
Rules:
- confirmed=false if evidence is weak, ambiguous, or outside the time window
- ma_spinoff: confirm whether the company is the ACQUIRER or the TARGET —
  both count; say which in the detail, plus deal status (announced/pending/completed)
- operational_change: confirm ONLY named restructuring programmes, significant
  site closures, or workforce cuts (>5% / thousands). Executive appointments,
  earnings, products, partnerships are NOT operational changes → confirmed=false
- is_shutdown=true ONLY if entity fully ceased (wound down, dissolved, absorbed with no brand successor)
- confidence 5=multiple corroborating sources with dates, 3=single clear source, 1=inferred"""

_PASS3 = """\
You extracted these signals for "{company}":
{extracted_json}

Re-read the source text and verify each is genuinely supported and inside this time window: {date_range}.
TEXT:
{context}

Respond with JSON only:
{{
  "verified": {{"signal_name": true/false, ...}},
  "corrections": {{"signal_name": "corrected detail if needed", ...}}
}}
Only include a key in "corrections" if the detail needs changing."""

_SINGLE_PASS = """\
Analyze the following content about "{company}" and extract all corporate change signals
from {date_range}.

""" + _SIGNAL_DEFS + """
CONTENT:
{context}

Return a JSON object with EXACTLY these fields:
{{
  "sector_change": true/false, "sector_detail": "one sentence or empty",
  "hq_change": true/false, "hq_detail": "FROM → TO with date, or empty",
  "hq_region": "USA sub-region or country if hq_change, else empty",
  "ma_spinoff": true/false, "ma_detail": "one sentence: acquirer/target roles, deal status, date — or empty",
  "renaming": true/false, "rename_detail": "old name → new name + date, or empty",
  "operational_change": true/false, "ops_detail": "one sentence or empty",
  "shutdown": true/false, "shutdown_detail": "one sentence or empty",
  "bankruptcy": true/false, "bankruptcy_detail": "filing type + date or empty",
  "summary": "2-3 sentence executive summary of all signals, or 'No significant corporate changes identified.'",
  "confidence": 1-5
}}
shutdown=true ONLY if entity fully ceased to exist.
hq_change=true ONLY for actual relocation or redomicile, not minor office moves."""


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────
def _parse_json(text: str) -> Optional[dict]:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$",           "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except Exception:
                pass
    return None


def _chunks(context: str) -> list[str]:
    out = []
    for i in range(0, min(len(context),
                          config.CONTEXT_CHUNK_SIZE * config.MAX_CONTEXT_CHUNKS),
                   config.CONTEXT_CHUNK_SIZE):
        chunk = context[i: i + config.CONTEXT_CHUNK_SIZE]
        if chunk.strip():
            out.append(chunk)
    return out or [context[:config.CONTEXT_CHUNK_SIZE]]


class Classifier:
    """
    Single entry point.  Automatically uses 3-pass or single-pass
    based on config.USE_IMPROVED_CLASSIFIER.
    """

    def __init__(self, client: Optional[LlamaServerClient] = None):
        self.client = client or LlamaServerClient()

    def _call(self, user_prompt: str, max_tokens: int = 500) -> str:
        """
        Call either the local llama-server client or the Databricks client.

        Local LlamaServerClient exposes .chat(messages=...).
        DatabricksModelClient exposes .complete(system_prompt=..., user_prompt=...).
        """
        if hasattr(self.client, "complete"):
            return self.client.complete(
                system_prompt=_SYSTEM,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=0.0,
            )

        return self.client.chat(
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens  = max_tokens,
            temperature = 0.0,
        )

    # ── single pass ──────────────────────────────────────────────────────────
    def _single_pass(self, company: str, context: str) -> SignalResult:
        r   = SignalResult(company=company)
        raw = self._call(
            _SINGLE_PASS.format(
                company    = company,
                context    = context[:config.CONTEXT_CHUNK_SIZE * config.MAX_CONTEXT_CHUNKS],
                date_range = config.DATE_RANGE,
            ),
            max_tokens=800,
        )
        data = _parse_json(raw)
        if not data:
            r.summary = f"[parse error] {raw[:120]}"
            return r

        conf = int(data.get("confidence", 1))
        if conf < config.MIN_CONFIDENCE:
            r.summary = f"[low confidence {conf}/5] {data.get('summary','')}"
            return r

        r.sector_change      = bool(data.get("sector_change"))
        r.sector_detail      = data.get("sector_detail", "")
        r.hq_change          = bool(data.get("hq_change"))
        r.hq_detail          = data.get("hq_detail", "")
        r.hq_region          = data.get("hq_region", "")
        r.ma_spinoff         = bool(data.get("ma_spinoff"))
        r.ma_detail          = data.get("ma_detail", "")
        r.renaming           = bool(data.get("renaming"))
        r.rename_detail      = data.get("rename_detail", "")
        r.operational_change = bool(data.get("operational_change"))
        r.shutdown           = bool(data.get("shutdown"))
        ops_parts = [data.get("ops_detail",""), data.get("shutdown_detail","")]
        r.ops_detail         = " | ".join(p for p in ops_parts if p)
        r.bankruptcy         = bool(data.get("bankruptcy"))
        r.bankruptcy_detail  = data.get("bankruptcy_detail", "")
        r.summary            = data.get("summary", "")
        r.recount()
        return r

    # ── three pass ───────────────────────────────────────────────────────────
    def _three_pass(self, company: str, context: str) -> SignalResult:
        r      = SignalResult(company=company)
        chunks = _chunks(context)

        # Pass 1 — detect across all chunks
        detected: set[str] = set()
        best_conf = 1
        for chunk in chunks:
            raw  = self._call(
                _PASS1.format(company=company, context=chunk, date_range=config.DATE_RANGE),
                150,
            )
            data = _parse_json(raw)
            if data:
                detected.update(data.get("detected") or [])
                best_conf = max(best_conf, int(data.get("confidence", 1)))

        if not detected or best_conf < config.MIN_CONFIDENCE:
            r.summary = "No significant corporate changes identified."
            return r

        # Pass 2 — extract details per signal
        full_ctx  = context[:config.CONTEXT_CHUNK_SIZE * config.MAX_CONTEXT_CHUNKS]
        extracted: dict[str, dict] = {}
        for signal in detected:
            raw  = self._call(_PASS2.format(
                signal=signal,
                company=company,
                context=full_ctx,
                date_range=config.DATE_RANGE,
            ), 300)
            data = _parse_json(raw)
            if (data
                    and data.get("confirmed")
                    and int(data.get("confidence", 1)) >= config.MIN_CONFIDENCE):
                extracted[signal] = data

        if not extracted:
            r.summary = "Signals detected but none confirmed after extraction."
            return r

        # Pass 3 — self-verify
        ext_summary = json.dumps(
            {k: v.get("detail", "") for k, v in extracted.items()}, ensure_ascii=False)
        raw_v = self._call(_PASS3.format(
            company=company,
            extracted_json=ext_summary,
            context=full_ctx,
            date_range=config.DATE_RANGE,
        ), 300)
        vdata = _parse_json(raw_v) or {"verified": {}}
        verified = vdata.get("verified", {})
        corrections = vdata.get("corrections", {})

        # Map to result fields
        for sig, d in extracted.items():
            if not verified.get(sig, True):
                continue
            detail = corrections.get(sig) or d.get("detail", "")
            if sig == "sector_change":
                r.sector_change = True; r.sector_detail = detail
            elif sig == "hq_change":
                r.hq_change = True; r.hq_detail = detail; r.hq_region = d.get("hq_region", "")
            elif sig == "ma_spinoff":
                r.ma_spinoff = True; r.ma_detail = detail
            elif sig == "renaming":
                r.renaming = True; r.rename_detail = detail
            elif sig == "operational_change":
                r.operational_change = True; r.ops_detail = detail
            elif sig == "bankruptcy":
                r.bankruptcy = True; r.bankruptcy_detail = detail
            elif sig == "shutdown":
                r.operational_change = True
                r.ops_detail = (r.ops_detail + " | " if r.ops_detail else "") + detail

        r.recount()
        if r.total_signals:
            details = []
            for label, det in [
                ("sector", r.sector_detail), ("hq", r.hq_detail),
                ("M&A", r.ma_detail), ("rename", r.rename_detail),
                ("ops", r.ops_detail), ("bankruptcy", r.bankruptcy_detail),
            ]:
                if det:
                    details.append(f"{label}: {det}")
            r.summary = "; ".join(details)[:500]
        else:
            r.summary = "No significant corporate changes identified after verification."
        return r

    def classify(
        self,
        company: str,
        context: str,
        sources: Optional[list[str]] = None,
    ) -> SignalResult:
        if not context or len(context.strip()) < 80:
            result = SignalResult(company=company, summary="Insufficient context to classify.")
        elif getattr(config, "USE_IMPROVED_CLASSIFIER", True):
            result = self._three_pass(company, context)
        else:
            result = self._single_pass(company, context)

        if sources:
            result.sources = list(sources)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def make_classifier() -> Classifier:
    """
    Factory that decides whether to use:
    - local llama-server client, or
    - Databricks Foundation Model client

    Controlled by config.USE_DATABRICKS_MODEL.
    """
    if getattr(config, "USE_DATABRICKS_MODEL", False):
        from databricks_client import make_databricks_classifier
        return Classifier(client=make_databricks_classifier())

    return Classifier()
