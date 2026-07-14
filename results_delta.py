"""
results_delta.py — persist pipeline results to a Unity Catalog Delta table
==========================================================================
The Excel report is a deliverable; the Delta table is the system of
record. Each run APPENDS one row per company with a run timestamp and
run name, so history is kept and the latest state is a window query away.

Key columns for the Tavily workflow:
  fulltext_chars — free full-text evidence depth (from the prescreen log)
  thin_evidence  — True when below TAVILY_THIN_THRESHOLD (prescreener-passed only)
  tavily_used    — True when Tavily raw content was actually fetched

Typical flow:
  1. Full run WITHOUT a Tavily key →
       write_results_table(spark, TABLE, results, companies, prescreen_csv,
                           run_name="full-free-2026-07-14")
  2. Size the budget:
       SELECT COUNT(*) FROM <table>
       WHERE run_name='full-free-2026-07-14' AND passed AND thin_evidence
  3. Export those companies as a new input CSV:
       export_thin_companies_csv(spark, TABLE, "full-free-2026-07-14",
                                 "/Volumes/.../thin_rerun.csv")
  4. Re-run the pipeline on that CSV WITH the Tavily key and a fresh
       output name; write_results_table again with run_name="thin-tavily-...".
       The gate re-fires (free evidence is still thin) and spends credits
       only on these companies.
"""

from __future__ import annotations

import csv as _csv
from dataclasses import asdict
from typing import Optional


def _load_prescreen(prescreen_csv: str) -> dict[str, dict]:
    """{company: prescreen row} from the CSV the pipeline wrote."""
    out: dict[str, dict] = {}
    try:
        with open(prescreen_csv, newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                out[row.get("company", "")] = row
    except FileNotFoundError:
        pass
    return out


def _to_bool(v) -> Optional[bool]:
    if v in (True, "True", "true", 1, "1"):
        return True
    if v in (False, "False", "false", 0, "0"):
        return False
    return None


def write_results_table(
    spark,
    table:         str,
    results:       list,            # list[SignalResult] from run_pipeline
    companies:     list[tuple],     # load_companies() output (name, sector, country, lei)
    prescreen_csv: str,             # path of the *_prescreen_log.csv the run wrote
    run_name:      str = "",
) -> int:
    """Append one row per company to `table` (created on first write).
    Returns the number of rows written."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import (StructType, StructField, StringType,
                                   BooleanType, IntegerType)

    meta = {name: (sector, country, lei) for name, sector, country, lei in companies}
    pres = _load_prescreen(prescreen_csv)

    rows = []
    for r in results:
        d = asdict(r)
        sector, country, lei = meta.get(r.company, ("", "", ""))
        p = pres.get(r.company, {})
        ft_raw = p.get("fulltext_chars", "")
        rows.append({
            "run_name":           run_name,
            "company":            r.company,
            "lei":                lei,
            "country_hq":         country,
            "sector":             sector,
            "passed":             _to_bool(p.get("passed")),
            "prescreen_stage":    p.get("stage", ""),
            "prescreen_score":    int(p.get("score") or 0),
            "prescreen_reason":   p.get("reason", ""),
            "fulltext_chars":     int(ft_raw) if str(ft_raw).strip().isdigit() else None,
            "thin_evidence":      _to_bool(p.get("thin_evidence")),
            "tavily_used":        bool(str(p.get("tavily_chars") or "").strip().isdigit()
                                       and int(p["tavily_chars"]) > 0),
            "sector_change":      bool(d.get("sector_change")),
            "hq_change":          bool(d.get("hq_change")),
            "hq_region":          d.get("hq_region", ""),
            "ma_spinoff":         bool(d.get("ma_spinoff")),
            "renaming":           bool(d.get("renaming")),
            "operational_change": bool(d.get("operational_change")),
            "shutdown":           bool(d.get("shutdown")),
            "bankruptcy":         bool(d.get("bankruptcy")),
            "total_signals":      int(d.get("total_signals") or 0),
            "sector_detail":      d.get("sector_detail", ""),
            "hq_detail":          d.get("hq_detail", ""),
            "ma_detail":          d.get("ma_detail", ""),
            "rename_detail":      d.get("rename_detail", ""),
            "ops_detail":         d.get("ops_detail", ""),
            "bankruptcy_detail":  d.get("bankruptcy_detail", ""),
            "summary":            d.get("summary", ""),
            "sources":            "; ".join(d.get("sources") or []),
        })

    schema = StructType(
        [StructField("run_name", StringType()), StructField("company", StringType()),
         StructField("lei", StringType()), StructField("country_hq", StringType()),
         StructField("sector", StringType()), StructField("passed", BooleanType()),
         StructField("prescreen_stage", StringType()), StructField("prescreen_score", IntegerType()),
         StructField("prescreen_reason", StringType()), StructField("fulltext_chars", IntegerType()),
         StructField("thin_evidence", BooleanType()), StructField("tavily_used", BooleanType()),
         StructField("sector_change", BooleanType()), StructField("hq_change", BooleanType()),
         StructField("hq_region", StringType()), StructField("ma_spinoff", BooleanType()),
         StructField("renaming", BooleanType()), StructField("operational_change", BooleanType()),
         StructField("shutdown", BooleanType()), StructField("bankruptcy", BooleanType()),
         StructField("total_signals", IntegerType()), StructField("sector_detail", StringType()),
         StructField("hq_detail", StringType()), StructField("ma_detail", StringType()),
         StructField("rename_detail", StringType()), StructField("ops_detail", StringType()),
         StructField("bankruptcy_detail", StringType()), StructField("summary", StringType()),
         StructField("sources", StringType())])

    df = (spark.createDataFrame(rows, schema)
               .withColumn("run_ts", F.current_timestamp()))
    df.write.mode("append").option("mergeSchema", "true").saveAsTable(table)

    thin = sum(1 for x in rows if x["thin_evidence"] is True)
    print(f"  [RESULTS DELTA] {len(rows)} rows appended to {table} "
          f"(run '{run_name}'): {thin} thin-evidence companies")
    return len(rows)


def export_thin_companies_csv(
    spark,
    table:    str,
    run_name: str,
    out_path: str,
) -> int:
    """
    Write the thin-evidence companies of a run as a tab-separated input
    CSV the pipeline can consume directly (company/Sector/Country HQ/LEI).
    Returns the number of companies exported.
    """
    rows = (spark.table(table)
                 .where(f"run_name = '{run_name}'")
                 .where("passed = true AND thin_evidence = true")
                 .select("company", "sector", "country_hq", "lei")
                 .distinct()
                 .collect())

    lines = ["company\tSector\tCountry HQ\tLEI"]
    for r in rows:
        lines.append(f"{r['company']}\t{r['sector'] or ''}\t"
                     f"{r['country_hq'] or ''}\t{r['lei'] or '-'}")
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"  [RESULTS DELTA] {len(rows)} thin-evidence companies → {out_path}")
    return len(rows)
