"""
pipeline.py — main orchestrator for market_signals_pipeline
============================================================
Reads a CSV of companies, fetches news via the two-stage search layer,
classifies signals, and writes an Excel report.

Usage
-----
  # Basic run (uses defaults from config.py)
  python pipeline.py

  # Custom input / output
  python pipeline.py --input my_companies.csv --output results.xlsx

  # Override Tavily key at runtime
  python pipeline.py --tavily-key tvly-XXXX

  # Test only the first 10 companies
  python pipeline.py --max-companies 10

  # Use a 6, 12, or 24 month horizon
  python pipeline.py --time-horizon-months 12

  # Skip checkpoint / start fresh
  python pipeline.py --no-resume

CSV format
----------
The input CSV must have at least one column that contains company names.
Supported column names (case-insensitive):  company, company_name, name, entity
Optionally a "sector" column is used for richer search queries.

Checkpoint / resume
-------------------
The pipeline writes a JSON checkpoint file alongside the output XLSX.
If interrupted, re-run with the same --output path to resume automatically.
Use --no-resume to force a clean start.
"""

from __future__ import annotations

import argparse
import csv
import re
import importlib
import json
import os
import sys
import threading
import time
from dataclasses import asdict
from typing import List, Optional, Tuple

import config
from search import fetch_stage1, fetch_tavily_targeted, FetchResult, StageOneResult
from fulltext import enrich_with_fulltext, stage1_to_fetchresult
from registry import lookup as registry_lookup
from classifier import Classifier, SignalResult, make_classifier
from excel_writer import write_report
from prescreener import Prescreener, PrescreenResult

try:
    from tavily import TavilyClient as _TavilyClient
    _TAVILY_OK = True
except ImportError:
    _TAVILY_OK = False


# ─────────────────────────────────────────────────────────────────────────────
# CSV reader
# ─────────────────────────────────────────────────────────────────────────────

_COMPANY_COL_ALIASES = {"company", "company_name", "name", "entity",
                        "coalition names", "company name"}
_SECTOR_COL_ALIASES  = {"sector", "industry", "subsector"}
_COUNTRY_COL_ALIASES = {"country hq", "country_hq", "country", "hq country", "hq_country"}
_LEI_COL_ALIASES     = {"lei", "lei id", "lei_id", "lei code", "lei_code"}


def _find_col(header: list[str], aliases: set[str]) -> Optional[int]:
    for i, h in enumerate(header):
        if h.strip().lower() in aliases:
            return i
    return None


def _find_name_col_fuzzy(header: list[str]) -> Optional[int]:
    """Last resort before column 0: any header containing 'name' or 'company'."""
    for i, h in enumerate(header):
        hl = h.strip().lower()
        if "name" in hl or "company" in hl:
            return i
    return None


_LEI_RE = re.compile(r"^[A-Z0-9]{20}$")


def _sniff_delimiter(csv_path: str) -> str:
    """
    Detect the field delimiter from the header line. Company lists are
    frequently exported as tab-separated files, and company names contain
    commas ("Aarons, Inc.") — a comma reader silently mangles those.
    """
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        first_line = fh.readline()
    for delim in ("\t", ";", ","):
        if delim in first_line:
            return delim
    return ","


def load_companies(csv_path: str) -> List[Tuple[str, str, str, str]]:
    """
    Return list of (company_name, sector_hint, country_hint, lei).
    sector_hint / country_hint / lei are empty strings if absent. LEI values
    that are not valid 20-char ISO 17442 codes (e.g. '-') come back as ''.
    Handles comma-, tab-, and semicolon-separated files.
    """
    companies: List[Tuple[str, str, str, str]] = []
    delimiter = _sniff_delimiter(csv_path)
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"CSV is empty: {csv_path}")

        name_col    = _find_col(header, _COMPANY_COL_ALIASES)
        sector_col  = _find_col(header, _SECTOR_COL_ALIASES)
        country_col = _find_col(header, _COUNTRY_COL_ALIASES)
        lei_col     = _find_col(header, _LEI_COL_ALIASES)

        if name_col is None:
            name_col = _find_name_col_fuzzy(header)
        if name_col is None:
            # Fall back: just use the first column
            print(f"  [WARNING] No recognised company column found. "
                  f"Headers: {header}\n  Using column 0: '{header[0]}'")
            name_col = 0

        for row in reader:
            if not row:
                continue
            name    = row[name_col].strip()
            sector  = row[sector_col].strip()  if sector_col  is not None and sector_col  < len(row) else ""
            country = row[country_col].strip() if country_col is not None and country_col < len(row) else ""
            lei     = row[lei_col].strip().upper() if lei_col is not None and lei_col < len(row) else ""
            if not _LEI_RE.match(lei):
                lei = ""
            if name:
                companies.append((name, sector, country, lei))

    return companies


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def _checkpoint_path(output_xlsx: str) -> str:
    base = os.path.splitext(output_xlsx)[0]
    return base + "_checkpoint.json"


def _load_checkpoint(output_xlsx: str) -> dict[str, dict]:
    """Return dict of {company_name: serialised SignalResult} from checkpoint."""
    cp_path = _checkpoint_path(output_xlsx)
    if os.path.exists(cp_path):
        try:
            with open(cp_path, encoding="utf-8") as fh:
                data = json.load(fh)
            print(f"  Loaded checkpoint: {len(data)} companies already processed.")
            return data
        except Exception as e:
            print(f"  [WARNING] Could not load checkpoint ({e}). Starting fresh.")
    return {}


def _save_checkpoint(output_xlsx: str, done: dict[str, dict]) -> None:
    cp_path = _checkpoint_path(output_xlsx)
    with open(cp_path, "w", encoding="utf-8") as fh:
        json.dump(done, fh, ensure_ascii=False, indent=2)


def _delete_checkpoint(output_xlsx: str) -> None:
    cp_path = _checkpoint_path(output_xlsx)
    if os.path.exists(cp_path):
        os.remove(cp_path)


def _result_from_dict(d: dict) -> SignalResult:
    """Reconstruct SignalResult from checkpoint dict."""
    return SignalResult(**{
        k: v for k, v in d.items()
        if k in SignalResult.__dataclass_fields__
    })


class _TavilyBudget:
    """Thread-safe credit budget shared by all workers. Hard stop: once
    spent == total, take() returns 0 and the run continues free-only."""

    def __init__(self, total: int):
        self._lock = threading.Lock()
        self.total = max(0, int(total))
        self.spent = 0

    def take(self, k: int) -> int:
        with self._lock:
            k = max(0, min(k, self.total - self.spent))
            self.spent += k
            return k

    def refund(self, k: int) -> None:
        with self._lock:
            self.spent = max(0, self.spent - k)

    @property
    def remaining(self) -> int:
        with self._lock:
            return self.total - self.spent


def _normalise_max_companies(max_companies: Optional[int]) -> int:
    if max_companies is None:
        return max(0, getattr(config, "DEFAULT_MAX_COMPANIES", 0))
    try:
        return max(0, int(max_companies))
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Core pipeline loop
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    input_csv:      str,
    output_xlsx:    str,
    tavily_key:     Optional[str] = None,
    resume:         bool          = True,
    max_companies:  Optional[int] = None,
    gleif_map:      Optional[dict] = None,
) -> List[SignalResult]:
    """
    Main pipeline.  Returns the list of SignalResult objects.

    max_companies:
      0 or None = process all companies.
      N > 0     = process only the first N companies from the input CSV.

    gleif_map:
      Prebuilt GLEIF facts from registry_delta.build_gleif_map (Databricks
      Delta tables). When provided, the GLEIF API is not called at all.
      When None (default, e.g. local runs), GLEIF is queried via its
      public API — by LEI directly when the input CSV has an LEI column.
    """
    # ── resolve paths ─────────────────────────────────────────────────────────
    input_csv   = os.path.abspath(input_csv)
    output_xlsx = os.path.abspath(output_xlsx)

    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # ── load companies ────────────────────────────────────────────────────────
    companies = load_companies(input_csv)
    original_company_count = len(companies)

    limit = _normalise_max_companies(max_companies)
    if limit > 0:
        companies = companies[:limit]

    print(f"\n{'='*60}")
    print(f"  Market Signals Pipeline")
    print(f"  Input  : {input_csv}")
    print(f"  Output : {output_xlsx}")
    print(f"  Date horizon: {config.DATE_RANGE}")
    print(f"  Companies in file: {original_company_count}")
    print(f"  Companies selected: {len(companies)}" + (f" (max_companies={limit})" if limit else " (all)"))
    print(f"  Classifier: {'3-pass' if config.USE_IMPROVED_CLASSIFIER else 'single-pass'}")
    print(f"  Min confidence: {config.MIN_CONFIDENCE}/5")
    print(f"{'='*60}\n")

    # ── checkpoint ───────────────────────────────────────────────────────────
    done_raw: dict[str, dict] = {}
    if resume:
        done_raw = _load_checkpoint(output_xlsx)

    # Only consider checkpoint entries for this selected batch.
    selected_companies = {name for name, _sector, _country, _lei in companies}
    done_raw = {k: v for k, v in done_raw.items() if k in selected_companies}
    done_set = set(done_raw.keys())

    remaining = [(n, s, c, l) for n, s, c, l in companies if n not in done_set]
    print(f"  To process: {len(remaining)} (skipping {len(done_set)} already done)\n")

    # ── initialise Tavily searcher ────────────────────────────────────────────
    key = tavily_key or config.TAVILY_API_KEY
    tavily_client = None
    if _TAVILY_OK and key and not key.startswith("tvly-YOUR"):
        tavily_client = _TavilyClient(api_key=key)
    else:
        print(
            "  [INFO] Tavily key not set or unavailable.\n"
            "  Stage 2 will use FREE full-text sources only\n"
            "  (SEC filing documents, Guardian article bodies, Wikipedia).\n"
            "  Set TAVILY_API_KEY to add targeted raw-content fetches for\n"
            "  thin-evidence companies (budget-capped by TAVILY_BUDGET)."
        )

    # Credit budget shared across all workers (0 when no client → never used)
    tavily_budget = _TavilyBudget(
        getattr(config, "TAVILY_BUDGET", 0) if tavily_client is not None else 0)
    if tavily_client is not None:
        print(f"  Tavily: targeted thin-evidence mode — budget {tavily_budget.total} queries, "
              f"thin threshold {getattr(config, 'TAVILY_THIN_THRESHOLD', 3000):,} chars, "
              f"{getattr(config, 'TAVILY_QUERIES_PER_COMPANY', 2)} queries/company")

    # ── initialise prescreener + classifier (stateless — shared by workers) ──
    prescreener: Prescreener = Prescreener()
    classifier:  Classifier  = make_classifier()

    # ── prescreener log ───────────────────────────────────────────────────────
    prescreen_log: list[dict] = []

    # ── per-company worker ────────────────────────────────────────────────────
    def _process_company(company: str, sector: str, country: str, lei: str,
                         ) -> tuple[SignalResult, dict, list[str]]:
        """
        Full per-company flow. Returns (result, prescreen_entry, log_lines).
        Pure function of its inputs + shared stateless clients — safe to run
        from multiple threads. Console output is buffered into log_lines so
        parallel workers don't interleave their lines.
        """
        out: list[str] = []

        # 1. Stage 1 — free sources (always runs)
        s1: StageOneResult = fetch_stage1(company)
        bd_str = "  ".join(f"{k}:{v:,}c" for k, v in s1.source_breakdown.items())
        out.append(f"          → Stage 1: {s1.char_count:,} chars  [{bd_str}]")

        # 1b. Registry facts (GLEIF / SEC / Companies House — free, official).
        #     Prepended so they reach BOTH the prescreener and the classifier:
        #     an INACTIVE status or previous legal name should trigger Stage 2
        #     even when the news headlines look quiet.
        if getattr(config, "REGISTRY_ENABLED", False):
            reg = registry_lookup(company, country, gleif_map=gleif_map, lei=lei)
            if reg.context:
                s1.full_context     = (reg.context + "\n\n" + s1.full_context)
                s1.headline_text    = (reg.headline + "\n" + s1.headline_text).strip()
                s1.sources          = list(dict.fromkeys(reg.sources + s1.sources))[:15]
                s1.source_breakdown = {**reg.breakdown, **s1.source_breakdown}
                s1.char_count      += len(reg.context)
                flag_str = f"  flags: {', '.join(reg.flags)}" if reg.flags else ""
                out.append(f"          → Registry: {len(reg.context):,} chars "
                           f"[{'  '.join(reg.breakdown)}]{flag_str}")

        # 2. Prescreener — decide whether to run Stage 2 deep evidence
        ps: PrescreenResult = prescreener.check(company, s1.headline_text)
        ps_entry = {
            "company":        company,
            "passed":         ps.passed,
            "stage":          ps.stage,
            "score":          ps.score,
            "reason":         ps.reason,
            # Filled below for passed companies — lets a free-only run
            # count its thin-evidence population and size a Tavily budget
            # from data before buying credits.
            "fulltext_chars": "",
            "thin_evidence":  "",
        }

        if ps.passed:
            out.append(f"          → Prescreener PASS (score={ps.score}: {ps.reason})")
            fetch = stage1_to_fetchresult(s1)

            # Free full-text enrichment (official APIs — no credits used).
            # Runs for every prescreener-passed company.
            ft_chars = 0
            if getattr(config, "FULLTEXT_ENABLED", False):
                fetch = enrich_with_fulltext(company, fetch)
                ft_chars = sum(v for k, v in fetch.source_breakdown.items()
                               if k.endswith("_fulltext"))
                if ft_chars:
                    ft_bd = "  ".join(f"{k}:{v:,}c"
                                      for k, v in fetch.source_breakdown.items()
                                      if k.endswith("_fulltext"))
                    out.append(f"          → Full text (free): +{ft_chars:,} chars  [{ft_bd}]")
                else:
                    out.append(f"          → Full text (free): no documents found")

            # Record evidence depth for every passed company — with or
            # without a Tavily key — so the prescreen log answers "how many
            # thin-evidence companies would spend credits?"
            is_thin = ft_chars < getattr(config, "TAVILY_THIN_THRESHOLD", 3000)
            ps_entry["fulltext_chars"] = ft_chars
            ps_entry["thin_evidence"]  = is_thin

            # Tavily thin-evidence gate: spend credits ONLY where the free
            # stack came back thin. Raw article content, budget-capped,
            # hard stop; the run continues free-only when credits run out.
            if tavily_client is not None and is_thin:
                allowed = tavily_budget.take(
                    getattr(config, "TAVILY_QUERIES_PER_COMPANY", 2))
                if allowed > 0:
                    secs, urls, used = fetch_tavily_targeted(
                        company, set(fetch.sources), tavily_client, allowed)
                    if used < allowed:
                        tavily_budget.refund(allowed - used)
                    if secs:
                        tav_ctx   = "\n\n".join(secs)
                        max_chars = config.CONTEXT_CHUNK_SIZE * config.MAX_CONTEXT_CHUNKS
                        reserve   = max(0, min(len(tav_ctx) + 44, max_chars - 6000))
                        base      = fetch.context[: max_chars - reserve]
                        fetch = FetchResult(
                            company          = company,
                            context          = (base + "\n\n=== TAVILY TARGETED (RAW CONTENT) ===\n"
                                                + tav_ctx)[:max_chars],
                            sources          = list(dict.fromkeys(fetch.sources + urls))[:15],
                            char_count       = fetch.char_count + len(tav_ctx),
                            source_breakdown = {**fetch.source_breakdown,
                                                "tavily_raw": len(tav_ctx)},
                        )
                        out.append(f"          → Tavily (thin evidence): +{len(tav_ctx):,} chars, "
                                   f"{used} credit(s), {tavily_budget.remaining} left")
                    else:
                        out.append(f"          → Tavily (thin evidence): no results, "
                                   f"{used} credit(s) spent")
                else:
                    out.append("          → Tavily budget exhausted — free evidence only")
        else:
            out.append(f"          → Prescreener SKIP ({ps.stage}, score={ps.score}: {ps.reason})")
            fetch = FetchResult(
                company          = company,
                context          = s1.full_context,
                sources          = s1.sources,
                char_count       = s1.char_count,
                source_breakdown = s1.source_breakdown,
            )

        # 3. Classify
        result: SignalResult = classifier.classify(
            company = company,
            context = fetch.context,
            sources = fetch.sources,
        )

        # 4. Brief status
        if result.total_signals > 0:
            flags = []
            if result.sector_change:      flags.append("Sector")
            if result.hq_change:          flags.append(f"HQ→{result.hq_region or '?'}")
            if result.ma_spinoff:         flags.append("M&A")
            if result.renaming:           flags.append("Rename")
            if result.operational_change: flags.append("Ops")
            if result.shutdown:           flags.append("SHUTDOWN")
            if result.bankruptcy:         flags.append("Bankruptcy")
            out.append(f"          → {result.total_signals} signal(s): {', '.join(flags)}")
        else:
            out.append(f"          → no signals")

        return result, ps_entry, out

    # ── process companies (parallel when PIPELINE_MAX_WORKERS > 1) ──────────
    start_time      = time.time()
    total_remaining = len(remaining)
    max_workers     = max(1, int(getattr(config, "PIPELINE_MAX_WORKERS", 1)))
    state_lock      = threading.Lock()   # guards done_raw, prescreen_log, counter
    completed       = [0]

    def _finish(company: str, result: SignalResult, ps_entry: dict,
                out: list[str]) -> None:
        """Record one company's outcome. Called under no lock; takes it."""
        with state_lock:
            completed[0] += 1
            idx = completed[0]
            prescreen_log.append(ps_entry)
            done_raw[company] = asdict(result)
            # Errored companies stay OUT of the checkpoint file so a
            # resumed run retries them instead of skipping them as done.
            _save_checkpoint(output_xlsx, {
                k: v for k, v in done_raw.items()
                if not str(v.get("summary", "")).startswith("[ERROR")
            })
            print("\n".join([f"  [{idx:3d}/{total_remaining}] {company}"] + out))

    if max_workers == 1 or len(remaining) <= 1:
        for company, sector, country, lei in remaining:
            result, ps_entry, out = _process_company(company, sector, country, lei)
            _finish(company, result, ps_entry, out)
            if completed[0] < total_remaining:
                time.sleep(config.INTER_COMPANY_DELAY)
    else:
        print(f"  Parallel mode: {max_workers} workers "
              f"(per-host rate limits shared across workers)\n")
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers,
                                thread_name_prefix="company") as pool:
            futures = {
                pool.submit(_process_company, company, sector, country, lei):
                company
                for company, sector, country, lei in remaining
            }
            for fut in as_completed(futures):
                company = futures[fut]
                try:
                    result, ps_entry, out = fut.result()
                except Exception as e:
                    result   = SignalResult(company=company,
                                            summary=f"[ERROR: {e}]")
                    ps_entry = {"company": company, "passed": False,
                                "stage": "error", "score": 0, "reason": str(e)[:80]}
                    out      = [f"          → ERROR: {e}"]
                _finish(company, result, ps_entry, out)

    elapsed = time.time() - start_time

    # ── reconstruct full results list in selected CSV order ──────────────────
    all_results: List[SignalResult] = []
    for company, _sector, _country, _lei in companies:
        if company in done_raw:
            all_results.append(_result_from_dict(done_raw[company]))
        else:
            # Should not happen, but guard gracefully
            all_results.append(SignalResult(
                company=company,
                summary="[skipped — not found in checkpoint]",
            ))

    # ── write prescreener log CSV ─────────────────────────────────────────────
    if config.PRESCREEN_LOG and prescreen_log:
        import csv
        log_path = os.path.splitext(output_xlsx)[0] + "_prescreen_log.csv"
        with open(log_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["company","passed","stage","score","reason",
                                                    "fulltext_chars","thin_evidence"])
            writer.writeheader()
            writer.writerows(prescreen_log)
        triggered  = sum(1 for r in prescreen_log if r["passed"])
        print(f"  Prescreener: {triggered}/{len(prescreen_log)} triggered Stage 2")
        thin = sum(1 for r in prescreen_log if r.get("thin_evidence") is True)
        est  = thin * getattr(config, "TAVILY_QUERIES_PER_COMPANY", 2)
        print(f"  Thin-evidence companies: {thin}/{triggered} passed "
              f"(a Tavily-enabled run would spend ~{est} credits at "
              f"{getattr(config, 'TAVILY_QUERIES_PER_COMPANY', 2)}/company)")
        if tavily_client is not None:
            print(f"  Tavily credits spent: {tavily_budget.spent} of {tavily_budget.total} budget "
                  f"(thin-evidence companies only)")
        else:
            print(f"  Tavily not used — Stage 2 served by free full-text sources")
        print(f"  Prescreener log: {log_path}")

    # ── write Excel ───────────────────────────────────────────────────────────
    print(f"\n  Writing Excel report …")
    saved_path = write_report(
        results     = all_results,
        output_path = output_xlsx,
        elapsed_sec = elapsed,
    )
    print(f"  Saved: {saved_path}")

    # ── clean up checkpoint on success ────────────────────────────────────────
    _delete_checkpoint(output_xlsx)

    mins, secs = divmod(int(elapsed), 60)
    print(f"\n{'='*60}")
    print(f"  Done.  {len(all_results)} companies processed in {mins}m {secs}s")
    signals_found = sum(1 for r in all_results if r.total_signals > 0)
    print(f"  Companies with signals: {signals_found} / {len(all_results)}")
    print(f"{'='*60}\n")

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="pipeline.py",
        description=(
            "Market Signals Pipeline — fetch corporate change signals "
            "for a list of companies using two-stage search + classifier."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py
  python pipeline.py --input companies.csv --output signals.xlsx
  python pipeline.py --tavily-key tvly-XXXX --no-resume
  python pipeline.py --max-companies 10 --time-horizon-months 12
        """,
    )
    p.add_argument(
        "--input", "-i",
        default=config.DEFAULT_INPUT_CSV,
        help=f"Path to input CSV (default: {config.DEFAULT_INPUT_CSV})",
    )
    p.add_argument(
        "--output", "-o",
        default=config.DEFAULT_OUTPUT_XLS,
        help=f"Path to output XLSX (default: {config.DEFAULT_OUTPUT_XLS})",
    )
    p.add_argument(
        "--tavily-key", "-k",
        default=None,
        help="Tavily API key (overrides config.py and TAVILY_API_KEY env var)",
    )
    p.add_argument(
        "--max-companies",
        type=int,
        default=config.DEFAULT_MAX_COMPANIES,
        help="Process only the first N companies. Use 0 for all companies.",
    )
    p.add_argument(
        "--time-horizon-months",
        type=int,
        choices=[6, 12, 24],
        default=config.TIME_HORIZON_MONTHS,
        help="Corporate-change lookback window from today: 6, 12, or 24 months.",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        default=False,
        help="Ignore any existing checkpoint and start from scratch",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args   = parser.parse_args()

    os.environ["TIME_HORIZON_MONTHS"] = str(args.time_horizon_months)
    os.environ["MAX_COMPANIES"] = str(args.max_companies)
    importlib.reload(config)

    try:
        run_pipeline(
            input_csv      = args.input,
            output_xlsx    = args.output,
            tavily_key     = args.tavily_key,
            resume         = not args.no_resume,
            max_companies  = args.max_companies,
        )
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}", file=sys.stderr)
        sys.exit(2)
    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoint saved; re-run to resume.", file=sys.stderr)
        sys.exit(130)


if __name__ == "__main__":
    main()
