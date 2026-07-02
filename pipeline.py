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
import importlib
import json
import os
import sys
import time
from dataclasses import asdict
from typing import List, Optional, Tuple

import config
from search import fetch_stage1, fetch_stage2, FetchResult, StageOneResult
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

_COMPANY_COL_ALIASES = {"company", "company_name", "name", "entity"}
_SECTOR_COL_ALIASES  = {"sector", "industry", "subsector"}


def _find_col(header: list[str], aliases: set[str]) -> Optional[int]:
    for i, h in enumerate(header):
        if h.strip().lower() in aliases:
            return i
    return None


def load_companies(csv_path: str) -> List[Tuple[str, str]]:
    """
    Return list of (company_name, sector_hint).
    sector_hint is empty string if the column is absent.
    """
    companies: List[Tuple[str, str]] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"CSV is empty: {csv_path}")

        name_col   = _find_col(header, _COMPANY_COL_ALIASES)
        sector_col = _find_col(header, _SECTOR_COL_ALIASES)

        if name_col is None:
            # Fall back: just use the first column
            print(f"  [WARNING] No recognised company column found. "
                  f"Headers: {header}\n  Using column 0: '{header[0]}'")
            name_col = 0

        for row in reader:
            if not row:
                continue
            name   = row[name_col].strip()
            sector = row[sector_col].strip() if sector_col is not None and sector_col < len(row) else ""
            if name:
                companies.append((name, sector))

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
) -> List[SignalResult]:
    """
    Main pipeline.  Returns the list of SignalResult objects.

    max_companies:
      0 or None = process all companies.
      N > 0     = process only the first N companies from the input CSV.
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
    selected_companies = {name for name, _sector in companies}
    done_raw = {k: v for k, v in done_raw.items() if k in selected_companies}
    done_set = set(done_raw.keys())

    remaining = [(n, s) for n, s in companies if n not in done_set]
    print(f"  To process: {len(remaining)} (skipping {len(done_set)} already done)\n")

    # ── initialise Tavily searcher ────────────────────────────────────────────
    key = tavily_key or config.TAVILY_API_KEY
    tavily_client = None
    if _TAVILY_OK and key and not key.startswith("tvly-YOUR"):
        tavily_client = _TavilyClient(api_key=key)
    else:
        print(
            "  [INFO] Tavily key not set or unavailable.\n"
            "  Stage 2 deep fetch disabled — free sources only.\n"
            "  Set TAVILY_API_KEY in config.py for deeper coverage."
        )

    # ── initialise prescreener + classifier ───────────────────────────────────
    prescreener: Prescreener = Prescreener()
    classifier:  Classifier  = make_classifier()

    # ── prescreener log ───────────────────────────────────────────────────────
    prescreen_log: list[dict] = []

    # ── process each company ─────────────────────────────────────────────────
    start_time = time.time()

    for idx, (company, sector) in enumerate(remaining, start=1):
        total_remaining = len(remaining)
        print(f"  [{idx:3d}/{total_remaining}] {company}")

        # 1. Stage 1 — free sources (always runs)
        s1: StageOneResult = fetch_stage1(company)
        bd_str = "  ".join(f"{k}:{v:,}c" for k, v in s1.source_breakdown.items())
        print(f"          → Stage 1: {s1.char_count:,} chars  [{bd_str}]")

        # 2. Prescreener — decide whether to spend Tavily credits
        ps: PrescreenResult = prescreener.check(company, s1.headline_text)
        prescreen_log.append({
            "company": company,
            "passed":  ps.passed,
            "stage":   ps.stage,
            "score":   ps.score,
            "reason":  ps.reason,
        })

        if ps.passed and tavily_client is not None:
            print(f"          → Prescreener PASS (score={ps.score}: {ps.reason}) — running Stage 2")
            fetch: FetchResult = fetch_stage2(company, s1, tavily_client)
            print(f"          → Stage 2: {fetch.char_count:,} chars total")
        else:
            if not ps.passed:
                print(f"          → Prescreener SKIP ({ps.stage}, score={ps.score}: {ps.reason})")
            else:
                print(f"          → Tavily not configured — using Stage 1 context only")
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

        # 4. Print brief status
        if result.total_signals > 0:
            flags = []
            if result.sector_change:      flags.append("Sector")
            if result.hq_change:          flags.append(f"HQ→{result.hq_region or '?'}")
            if result.ma_spinoff:         flags.append("M&A")
            if result.renaming:           flags.append("Rename")
            if result.operational_change: flags.append("Ops")
            if result.shutdown:           flags.append("SHUTDOWN")
            if result.bankruptcy:         flags.append("Bankruptcy")
            print(f"          → {result.total_signals} signal(s): {', '.join(flags)}")
        else:
            print(f"          → no signals")

        # 5. Persist to checkpoint
        done_raw[company] = asdict(result)
        _save_checkpoint(output_xlsx, done_raw)

        # 6. Rate-limit between companies
        if idx < total_remaining:
            time.sleep(config.INTER_COMPANY_DELAY)

    elapsed = time.time() - start_time

    # ── reconstruct full results list in selected CSV order ──────────────────
    all_results: List[SignalResult] = []
    for company, _sector in companies:
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
            writer = csv.DictWriter(fh, fieldnames=["company","passed","stage","score","reason"])
            writer.writeheader()
            writer.writerows(prescreen_log)
        triggered  = sum(1 for r in prescreen_log if r["passed"])
        print(f"  Prescreener: {triggered}/{len(prescreen_log)} triggered Stage 2")
        print(f"  Tavily calls used: ~{triggered * 5} of 1,000 free/month")
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
