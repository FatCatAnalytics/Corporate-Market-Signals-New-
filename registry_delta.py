"""
registry_delta.py — GLEIF lookups from Delta tables (Databricks)
================================================================
When the GLEIF golden copy already lives in Unity Catalog Delta tables,
per-company API calls (60 req/min → ~85 min for a 5,000-company list) are
replaced by ONE batch join at pipeline start. Registry facts become
effectively free and instant.

Usage (Databricks notebook, before run_pipeline):

    from pipeline import load_companies, run_pipeline
    from registry_delta import build_gleif_map

    companies = load_companies(INPUT_CSV)
    gleif_map = build_gleif_map(
        spark,
        [(name, country) for name, _sector, country in companies],
        level1_table = "catalog.schema.gleif_level1",
        level2_table = "catalog.schema.gleif_level2_rr",   # optional
    )
    results = run_pipeline(..., gleif_map=gleif_map)

Column names
------------
GLEIF golden-copy loads differ (dots become underscores, casing varies).
Columns are resolved by *signature*: names are lowercased and stripped of
non-alphanumerics before matching, so `Entity.LegalName`,
`Entity_LegalName` and `entity_legal_name` all resolve. Pass
`columns={...}` to override explicitly if resolution fails — the error
message lists both what was searched for and what the table actually has.

Matching strategy
-----------------
1. Exact match on normalised name (legal suffixes stripped), preferring
   candidates whose legal/HQ country matches the CSV country hint.
2. For the remainder: candidate join on the first name token (+ country
   when known), then the same token-overlap scoring as the API path
   (registry._match_score, threshold 1.5).

Freshness caution: a Delta copy is only as current as its last load.
GLEIF publishes daily golden copies — if the table is stale, entity
status changes and renames will lag accordingly.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Optional

from registry import (
    _match_score,
    _norm_tokens,
    _MATCH_THRESHOLD,
    country_to_iso,
    format_gleif_record,
)

# ─────────────────────────────────────────────────────────────────────────────
# Column resolution — find real column names by signature
# ─────────────────────────────────────────────────────────────────────────────
# key → list of candidate signatures (lowercase, alphanumerics only)
_L1_COLUMNS = {
    "lei":           ["lei"],
    "legal_name":    ["entitylegalname", "legalname", "entitylegalnamename"],
    "status":        ["entityentitystatus", "entitystatus"],
    "reg_status":    ["registrationregistrationstatus", "registrationstatus"],
    "last_update":   ["registrationlastupdatedate", "lastupdatedate"],
    "legal_city":    ["entitylegaladdresscity", "legaladdresscity"],
    "legal_country": ["entitylegaladdresscountry", "legaladdresscountry"],
    "hq_city":       ["entityheadquartersaddresscity", "headquartersaddresscity"],
    "hq_country":    ["entityheadquartersaddresscountry", "headquartersaddresscountry"],
    # optional
    "exp_date":      ["entityentityexpirationdate", "entityexpirationdate"],
    "exp_reason":    ["entityentityexpirationreason", "entityexpirationreason"],
    "successor_name": ["entitysuccessorentity", "entitysuccessorentityentityname",
                       "successorentityname", "entitysuccessorentityname"],
}
_L1_REQUIRED = {"lei", "legal_name", "status", "legal_country"}

_L2_COLUMNS = {
    "child_lei":  ["relationshipstartnodenodeid", "startnodenodeid", "childlei"],
    "parent_lei": ["relationshipendnodenodeid", "endnodenodeid", "parentlei"],
    "rel_type":   ["relationshiprelationshiptype", "relationshiptype"],
    "rel_status": ["relationshiprelationshipstatus", "relationshipstatus",
                   "registrationregistrationstatus"],
}


def _sig(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _resolve_columns(
    actual_cols: list[str],
    wanted:      dict[str, list[str]],
    required:    set[str],
    overrides:   Optional[dict] = None,
    table_label: str = "table",
) -> dict[str, Optional[str]]:
    overrides = overrides or {}
    by_sig    = {_sig(c): c for c in actual_cols}
    resolved: dict[str, Optional[str]] = {}
    for key, candidates in wanted.items():
        if key in overrides:
            resolved[key] = overrides[key]
            continue
        resolved[key] = next(
            (by_sig[c] for c in candidates if c in by_sig), None
        )
    missing = [k for k in required if not resolved.get(k)]
    if missing:
        raise ValueError(
            f"Could not resolve required GLEIF columns {missing} in {table_label}. "
            f"Searched signatures: { {k: wanted[k] for k in missing} }. "
            f"Table columns: {actual_cols}. "
            f"Pass columns={{'{missing[0]}': '<actual column name>', ...}} to override."
        )
    return resolved


def _find_prev_name_columns(actual_cols: list[str]) -> list[tuple[str, Optional[str]]]:
    """
    GLEIF golden-copy CSVs carry other/previous names as numbered column
    pairs (Entity.OtherEntityName.1 / Entity.OtherEntityName.1.Type …).
    Return [(name_col, type_col_or_None), …]; when a type column exists the
    caller filters for PREVIOUS_LEGAL_NAME, otherwise all values are kept.
    """
    pairs = []
    for col in actual_cols:
        s = _sig(col)
        if ("otherentityname" in s or "othername" in s) and not s.endswith("type"):
            type_col = next(
                (c for c in actual_cols if _sig(c) == s + "type"), None
            )
            pairs.append((col, type_col))
    return pairs


def _norm_str(name: str) -> str:
    return " ".join(_norm_tokens(name))


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────
def build_gleif_map(
    spark,
    companies:        list[tuple],
    level1_table:     str,
    level2_table:     Optional[str] = None,
    columns:          Optional[dict] = None,
    candidate_limit:  int = 50,
) -> dict[str, tuple]:
    """
    Batch-match companies against a GLEIF Level 1 Delta table.

    companies: [(company_name, country_hint), …] or
               [(company_name, country_hint, lei), …].
               country_hint is the raw CSV 'Country HQ' value ('' if
               unknown). When an LEI is present the record is joined
               DIRECTLY on LEI — deterministic, no name matching at all.
               Name matching is used only for companies without an LEI.
    Returns {company_name: (section, headline, urls, flags)} — the exact
    shape registry.lookup consumes via its gleif_map parameter. Companies
    with no acceptable match are absent from the dict.
    """
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType

    df   = spark.table(level1_table)
    cols = _resolve_columns(df.columns, _L1_COLUMNS, _L1_REQUIRED,
                            columns, level1_table)
    prev_pairs = _find_prev_name_columns(df.columns)

    # ── target sets: LEI-keyed (exact) vs name-matched ───────────────────────
    lei_targets:  list[tuple] = []   # (company, lei, iso)
    name_targets: list[tuple] = []   # (company, norm, first_token, iso)
    for row in companies:
        name, country = row[0], row[1]
        lei = str(row[2]).strip().upper() if len(row) > 2 and row[2] else ""
        if not name:
            continue
        iso = country_to_iso(country) or ""
        if re.fullmatch(r"[A-Z0-9]{20}", lei):
            lei_targets.append((name, lei, iso))
        else:
            name_targets.append((name, _norm_str(name),
                                 (_norm_tokens(name) or ("",))[0], iso))
    if not lei_targets and not name_targets:
        return {}

    # ── project + normalise the GLEIF side ───────────────────────────────────
    keep = [F.col(c).alias(k) for k, c in cols.items() if c]
    for i, (ncol, tcol) in enumerate(prev_pairs):
        keep.append(F.col(ncol).alias(f"_prev_{i}"))
        if tcol:
            keep.append(F.col(tcol).alias(f"_prevtype_{i}"))

    base = df.select(*keep).where(F.col("legal_name").isNotNull())

    result: dict[str, tuple] = {}   # company → (rec, iso)

    # ── path 1: direct LEI join (deterministic — no scoring needed) ─────────
    if lei_targets:
        lei_df = spark.createDataFrame(
            lei_targets, ["company", "target_lei", "iso"]
        )
        lei_rows = (base.join(
                        F.broadcast(lei_df),
                        F.upper(F.col("lei")) == F.col("target_lei"),
                        "inner")
                    .collect())
        for r in lei_rows:
            result[r["company"]] = (_row_to_rec(r, cols, prev_pairs), r["iso"] or None)
        misses = len(lei_targets) - len(lei_rows)
        if misses:
            print(f"  [REGISTRY DELTA] {misses} provided LEIs not found in "
                  f"{level1_table} — check table freshness.")

    # ── path 2: name matching for companies without an LEI ──────────────────
    rows = []
    if name_targets:
        tgt_df = spark.createDataFrame(
            name_targets, ["company", "norm_name", "first_token", "iso"]
        )
        norm_udf  = F.udf(_norm_str, StringType())
        first_udf = F.udf(lambda s: (_norm_tokens(s) or ("",))[0], StringType())
        gl = (base.withColumn("g_norm",  norm_udf(F.col("legal_name")))
                  .withColumn("g_first", first_udf(F.col("legal_name"))))

        # Candidate join: same first name token, and country agreement when
        # the target has a hint (legal OR HQ country). Exact-name matches are
        # just the highest-scoring candidates — one join serves both.
        hq_country = (F.col("hq_country") if cols.get("hq_country")
                      else F.col("legal_country"))
        joined = (gl.join(
                    F.broadcast(tgt_df),
                    (F.col("g_first") == F.col("first_token"))
                    & ((F.col("iso") == "")
                       | (F.col("legal_country") == F.col("iso"))
                       | (hq_country == F.col("iso"))),
                    "inner")
                  .withColumn("_exact", (F.col("g_norm") == F.col("norm_name")).cast("int")))

        # Bound rows per company before collecting to the driver.
        from pyspark.sql.window import Window
        w = (Window.partitionBy("company")
                   .orderBy(F.desc("_exact"), F.length("g_norm").asc()))
        rows = (joined.withColumn("_rank", F.row_number().over(w))
                      .where(F.col("_rank") <= candidate_limit)
                      .collect())

    # ── score name-match candidates (identical logic to the API path) ───────
    by_company: dict[str, list] = defaultdict(list)
    for r in rows:
        by_company[r["company"]].append(r)

    iso_by_company = {t[0]: t[3] for t in name_targets}

    for company, cands in by_company.items():
        best, best_score = None, 0.0
        for r in cands:
            score = _match_score(company, r["legal_name"] or "")
            if score > best_score:
                best, best_score = r, score
        if best is None or best_score < _MATCH_THRESHOLD:
            continue
        result[company] = (_row_to_rec(best, cols, prev_pairs),
                           iso_by_company.get(company) or None)

    # ── Level 2: current ultimate parent (optional) ──────────────────────────
    if level2_table and result:
        try:
            _attach_parents(spark, result, level1_table, level2_table, cols)
        except Exception as e:
            print(f"  [REGISTRY DELTA] Level 2 parent lookup skipped: {e}")

    # ── render with the shared formatter ─────────────────────────────────────
    rendered = {
        company: format_gleif_record(company, rec, iso)
        for company, (rec, iso) in result.items()
    }
    n_total = len(lei_targets) + len(name_targets)
    print(f"  [REGISTRY DELTA] GLEIF matches: {len(rendered)}/{n_total} companies "
          f"({len(lei_targets)} joined by LEI, {len(name_targets)} name-matched)")
    return rendered


def _row_to_rec(row, cols: dict, prev_pairs: list) -> dict:
    """Convert a joined Spark Row into the shared GLEIF record dict."""
    fields = set(row.__fields__)

    def g(key, default=""):
        return (row[key] if key in fields else default) or default

    prev_names = []
    for i, (_ncol, tcol) in enumerate(prev_pairs):
        val = g(f"_prev_{i}", None)
        if not val:
            continue
        if tcol:
            ptype = g(f"_prevtype_{i}", None)
            if ptype and "PREVIOUS" not in str(ptype).upper():
                continue
        prev_names.append(val)

    return {
        "lei":           g("lei"),
        "legal_name":    g("legal_name"),
        "status":        g("status"),
        "reg_status":    g("reg_status"),
        "last_update":   str(g("last_update"))[:10],
        "prev_names":    prev_names,
        "succ_names":    [g("successor_name")] if g("successor_name") else [],
        "exp_date":      g("exp_date"),
        "exp_reason":    g("exp_reason"),
        "legal_city":    g("legal_city", "?"),
        "legal_country": g("legal_country"),
        "hq_city":       g("hq_city", "?"),
        "hq_country":    g("hq_country"),
        "parent_name":   "",
    }


def _attach_parents(spark, result: dict, level1_table: str,
                    level2_table: str, l1_cols: dict) -> None:
    """Fill rec['parent_name'] with the ACTIVE ultimate parent's legal name."""
    from pyspark.sql import functions as F

    rr      = spark.table(level2_table)
    l2_cols = _resolve_columns(rr.columns, _L2_COLUMNS,
                               {"child_lei", "parent_lei", "rel_type"},
                               table_label=level2_table)

    leis = [rec["lei"] for rec, _iso in result.values() if rec["lei"]]
    rel  = (rr.select(
                F.col(l2_cols["child_lei"]).alias("child"),
                F.col(l2_cols["parent_lei"]).alias("parent"),
                F.col(l2_cols["rel_type"]).alias("rtype"),
                *( [F.col(l2_cols["rel_status"]).alias("rstatus")]
                   if l2_cols.get("rel_status") else [] ))
              .where(F.col("child").isin(leis))
              .where(F.col("rtype") == "IS_ULTIMATELY_CONSOLIDATED_BY"))
    if l2_cols.get("rel_status"):
        rel = rel.where(F.col("rstatus") == "ACTIVE")

    l1 = spark.table(level1_table).select(
        F.col(l1_cols["lei"]).alias("parent"),
        F.col(l1_cols["legal_name"]).alias("parent_name"),
    )
    parents = {r["child"]: r["parent_name"]
               for r in rel.join(l1, "parent", "left").collect()}

    for rec, _iso in result.values():
        pname = parents.get(rec["lei"])
        # Suppress self-referential or trivial parent lines
        if pname and _norm_tokens(pname) != _norm_tokens(rec["legal_name"]):
            rec["parent_name"] = pname
