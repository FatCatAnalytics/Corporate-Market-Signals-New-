# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Market Signals Parallel Runner
# MAGIC %md
# MAGIC # Market Signals Corporates — Spark Parallel Runner
# MAGIC
# MAGIC Processes corporate market signals at scale using `mapInPandas` for parallelism.
# MAGIC Results are written to a Delta table for persistence and resume support.
# MAGIC
# MAGIC **Architecture:** Same pipeline logic (search → prescreener → classifier) distributed
# MAGIC across Spark partitions. Each partition processes a batch of companies independently.
# MAGIC
# MAGIC **Target:** `client_intelligence_analytics.corporate_market_signals.signals_results`

# COMMAND ----------

# DBTITLE 1,Install requirements
# MAGIC %pip install tavily-python requests openpyxl -q

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Imports and fallback pip
import os
import sys
import time
import uuid
import subprocess
import importlib
from datetime import datetime, timezone

# Fallback pip install (ensures deps survive restartPython)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "tavily-python", "requests", "openpyxl"
])
import pandas as pd

# COMMAND ----------

# DBTITLE 1,Widgets
dbutils.widgets.text("pipeline_dir", "/Workspace/Users/aksel.etingu@crisil.com/Market Signals Corporates")
dbutils.widgets.text("input_csv", "/Volumes/data_poc_ws/default/client_intelligence_analytics/market_signals/input/corporate_100.csv")
dbutils.widgets.dropdown("resume", "true", ["true", "false"])
dbutils.widgets.dropdown("use_tavily", "true", ["true", "false"])
dbutils.widgets.dropdown("time_horizon_months", "12", ["6", "12", "24"])
dbutils.widgets.text("classifier_endpoint", "databricks-meta-llama-3-3-70b-instruct")
dbutils.widgets.text("prescreener_endpoint", "databricks-meta-llama-3-1-8b-instruct")
dbutils.widgets.text("num_partitions", "200")
dbutils.widgets.text("limit", "")
dbutils.widgets.dropdown("export_only", "false", ["true", "false"])
dbutils.widgets.text("export_run_id", "latest")

# COMMAND ----------

# DBTITLE 1,Configuration and credentials
# ── Read widget values ─────────────────────────────────────────────────────────
PIPELINE_DIR         = dbutils.widgets.get("pipeline_dir")
INPUT_CSV            = dbutils.widgets.get("input_csv")
RESUME               = dbutils.widgets.get("resume").lower() == "true"
USE_TAVILY           = dbutils.widgets.get("use_tavily").lower() == "true"
TIME_HORIZON_MONTHS  = dbutils.widgets.get("time_horizon_months")
CLASSIFIER_ENDPOINT  = dbutils.widgets.get("classifier_endpoint")
PRESCREENER_ENDPOINT = dbutils.widgets.get("prescreener_endpoint")
NUM_PARTITIONS       = int(dbutils.widgets.get("num_partitions") or "200")
_limit_raw           = dbutils.widgets.get("limit").strip()
LIMIT                = int(_limit_raw) if _limit_raw else None
EXPORT_ONLY          = dbutils.widgets.get("export_only").lower() == "true"
EXPORT_RUN_ID        = dbutils.widgets.get("export_run_id").strip()

# Generate a unique run ID
RUN_ID = str(uuid.uuid4())

# Delta table target
CATALOG = "client_intelligence_analytics"
SCHEMA  = "corporate_market_signals"
TABLE   = "signals_results"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{TABLE}"

# ── Databricks credentials ─────────────────────────────────────────────────
ctx   = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
TOKEN = ctx.apiToken().get()
HOST  = ctx.apiUrl().get()

os.environ["DATABRICKS_TOKEN"] = TOKEN
os.environ["DATABRICKS_HOST"]  = HOST
os.environ["USE_DATABRICKS_MODEL"] = "True"
os.environ["DATABRICKS_CLASSIFIER_ENDPOINT"]  = CLASSIFIER_ENDPOINT
os.environ["DATABRICKS_PRESCREENER_ENDPOINT"] = PRESCREENER_ENDPOINT
os.environ["TIME_HORIZON_MONTHS"] = TIME_HORIZON_MONTHS

# ── Tavily key ──────────────────────────────────────────────────────────────
TAVILY_KEY = ""
if USE_TAVILY:
    try:
        TAVILY_KEY = dbutils.secrets.get(scope="market-signals", key="tavily-api-key")
        print("Tavily key loaded from Databricks Secrets")
    except Exception:
        print("Tavily secret not found — Stage 2 will use free sources only")

# ── Worker config (closure-captured, no SparkContext needed on serverless) ────
WORKER_CONFIG = {
    "token": TOKEN,
    "host": HOST,
    "tavily_key": TAVILY_KEY,
    "classifier_endpoint": CLASSIFIER_ENDPOINT,
    "prescreener_endpoint": PRESCREENER_ENDPOINT,
    "pipeline_dir": PIPELINE_DIR,
    "use_tavily": USE_TAVILY,
    "time_horizon_months": TIME_HORIZON_MONTHS,
    "run_id": RUN_ID,
}

print(f"Run ID: {RUN_ID}")
print(f"Pipeline dir: {PIPELINE_DIR}")
print(f"Input CSV: {INPUT_CSV}")
print(f"Target table: {FULL_TABLE_NAME}")
print(f"Partitions: {NUM_PARTITIONS}")
print(f"Limit: {LIMIT}")
print(f"Resume: {RESUME}")
print(f"Tavily: {USE_TAVILY}")
print(f"Classifier: {CLASSIFIER_ENDPOINT}")
print(f"Export only: {EXPORT_ONLY}")

# COMMAND ----------

# DBTITLE 1,Worker function: process_company_batch
from pyspark.sql.types import (
    StructType, StructField, StringType, BooleanType, IntegerType, TimestampType
)

RESULT_SCHEMA = StructType([
    StructField("company", StringType(), False),
    StructField("sector_change", BooleanType(), True),
    StructField("hq_change", BooleanType(), True),
    StructField("hq_region", StringType(), True),
    StructField("ma_spinoff", BooleanType(), True),
    StructField("renaming", BooleanType(), True),
    StructField("operational_change", BooleanType(), True),
    StructField("shutdown", BooleanType(), True),
    StructField("bankruptcy", BooleanType(), True),
    StructField("total_signals", IntegerType(), True),
    StructField("sector_detail", StringType(), True),
    StructField("hq_detail", StringType(), True),
    StructField("ma_detail", StringType(), True),
    StructField("rename_detail", StringType(), True),
    StructField("ops_detail", StringType(), True),
    StructField("bankruptcy_detail", StringType(), True),
    StructField("summary", StringType(), True),
    StructField("sources", StringType(), True),
    StructField("processed_at", TimestampType(), True),
    StructField("run_id", StringType(), True),
    StructField("confidence_score", IntegerType(), True),
    StructField("error", StringType(), True),
])


def process_company_batch(pdf_iterator):
    """mapInPandas worker: processes a batch of companies through the pipeline."""
    import os, sys, json, time, subprocess
    from datetime import datetime, timezone

    cfg = WORKER_CONFIG
    pipeline_dir = cfg["pipeline_dir"]
    run_id = cfg["run_id"]

    # Setup path and working directory
    if pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)
    os.chdir(pipeline_dir)

    # Set env vars for config.py
    os.environ["DATABRICKS_TOKEN"] = cfg["token"]
    os.environ["DATABRICKS_HOST"] = cfg["host"]
    os.environ["USE_DATABRICKS_MODEL"] = "True"
    os.environ["DATABRICKS_CLASSIFIER_ENDPOINT"] = cfg["classifier_endpoint"]
    os.environ["DATABRICKS_PRESCREENER_ENDPOINT"] = cfg["prescreener_endpoint"]
    os.environ["TIME_HORIZON_MONTHS"] = cfg["time_horizon_months"]
    if cfg["tavily_key"]:
        os.environ["TAVILY_API_KEY"] = cfg["tavily_key"]

    # Ensure deps
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
        "tavily-python", "requests", "openpyxl"])

    # Import pipeline modules
    import config
    import importlib
    importlib.reload(config)
    import search
    from classifier import SignalResult, make_classifier
    from prescreener import Prescreener

    # Init Tavily
    tavily_client = None
    if cfg["use_tavily"] and cfg["tavily_key"]:
        try:
            from tavily import TavilyClient
            tavily_client = TavilyClient(api_key=cfg["tavily_key"])
        except ImportError:
            pass

    # Init classifier and prescreener
    classifier = make_classifier()
    prescreener = Prescreener()

    def _process_single(company):
        """Process one company through the full pipeline."""
        # Stage 1 — free sources
        s1 = search.fetch_stage1(company)

        # Prescreener
        ps = prescreener.check(company, s1.headline_text)

        # Stage 2 if prescreener passes and Tavily available
        if ps.passed and tavily_client is not None:
            fetch = search.fetch_stage2(company, s1, tavily_client)
        else:
            fetch = search.FetchResult(
                company=company,
                context=s1.full_context,
                sources=s1.sources,
                char_count=s1.char_count,
                source_breakdown=s1.source_breakdown,
            )

        # Classify
        result = classifier.classify(
            company=company,
            context=fetch.context,
            sources=fetch.sources,
        )
        return result

    # Process each batch
    import pandas as _pd
    for pdf in pdf_iterator:
        rows = []
        for _, row in pdf.iterrows():
            company = row["company"]
            try:
                result = _process_single(company)
                rows.append({
                    "company": company,
                    "sector_change": result.sector_change,
                    "hq_change": result.hq_change,
                    "hq_region": result.hq_region or "",
                    "ma_spinoff": result.ma_spinoff,
                    "renaming": result.renaming,
                    "operational_change": result.operational_change,
                    "shutdown": result.shutdown,
                    "bankruptcy": result.bankruptcy,
                    "total_signals": result.total_signals,
                    "sector_detail": result.sector_detail or "",
                    "hq_detail": result.hq_detail or "",
                    "ma_detail": result.ma_detail or "",
                    "rename_detail": result.rename_detail or "",
                    "ops_detail": result.ops_detail or "",
                    "bankruptcy_detail": result.bankruptcy_detail or "",
                    "summary": result.summary or "",
                    "sources": json.dumps(result.sources or []),
                    "processed_at": datetime.now(timezone.utc),
                    "run_id": run_id,
                    "confidence_score": getattr(result, "confidence_score", 0) or 0,
                    "error": None,
                })
            except Exception as e:
                rows.append({
                    "company": company,
                    "sector_change": False, "hq_change": False, "hq_region": "",
                    "ma_spinoff": False, "renaming": False,
                    "operational_change": False, "shutdown": False, "bankruptcy": False,
                    "total_signals": 0,
                    "sector_detail": "", "hq_detail": "", "ma_detail": "",
                    "rename_detail": "", "ops_detail": "", "bankruptcy_detail": "",
                    "summary": "", "sources": "[]",
                    "processed_at": datetime.now(timezone.utc),
                    "run_id": run_id,
                    "confidence_score": 0,
                    "error": str(e)[:500],
                })
        yield _pd.DataFrame(rows)

print("✔ Worker function defined")

# COMMAND ----------

# DBTITLE 1,Main execution: read CSV, anti-join, repartition, mapInPandas, write Delta
from pyspark.sql import functions as F

if EXPORT_ONLY:
    print("✔ EXPORT_ONLY mode — skipping pipeline processing.")
    print(f"  Table: {FULL_TABLE_NAME}")
    print(f"  Export run ID: {EXPORT_RUN_ID}")
else:
    # ── 1. Create target Delta table ───────────────────────────────────────────
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {FULL_TABLE_NAME} (
            company STRING NOT NULL,
            sector_change BOOLEAN,
            hq_change BOOLEAN,
            hq_region STRING,
            ma_spinoff BOOLEAN,
            renaming BOOLEAN,
            operational_change BOOLEAN,
            shutdown BOOLEAN,
            bankruptcy BOOLEAN,
            total_signals INT,
            sector_detail STRING,
            hq_detail STRING,
            ma_detail STRING,
            rename_detail STRING,
            ops_detail STRING,
            bankruptcy_detail STRING,
            summary STRING,
            sources STRING,
            processed_at TIMESTAMP,
            run_id STRING,
            confidence_score INT,
            error STRING
        )
        USING DELTA
        COMMENT 'Corporate market signal results - parallel pipeline output'
    """)
    print(f"Target table ready: {FULL_TABLE_NAME}")

    # ── 2. Load input CSV ─────────────────────────────────────────────────────
    df_input = spark.read.csv(INPUT_CSV, header=True, inferSchema=True)

    # Normalize: find company column
    input_cols = [c.lower() for c in df_input.columns]
    company_col = None
    for candidate in ["company", "company_name", "name", "entity"]:
        if candidate in input_cols:
            company_col = df_input.columns[input_cols.index(candidate)]
            break
    if company_col is None:
        company_col = df_input.columns[0]

    df_companies = (
        df_input
        .select(F.col(company_col).alias("company"))
        .filter(F.col("company").isNotNull())
        .filter(F.trim(F.col("company")) != "")
        .dropDuplicates(["company"])
    )

    if LIMIT:
        df_companies = df_companies.limit(LIMIT)

    total_input = df_companies.count()
    print(f"Input companies (deduplicated): {total_input}")

    # ── 3. Resume: anti-join against already-processed ──────────────────────
    if RESUME:
        try:
            df_existing = spark.table(FULL_TABLE_NAME).select("company").distinct()
            df_todo = df_companies.join(df_existing, on="company", how="left_anti")
            already_done = total_input - df_todo.count()
            print(f"Already processed (resume): {already_done}")
        except Exception:
            df_todo = df_companies
            print("No existing results found — processing all companies")
    else:
        df_todo = df_companies

    todo_count = df_todo.count()
    print(f"Companies to process this run: {todo_count}")

    if todo_count == 0:
        print("Nothing to do — all companies already processed!")
    else:
        # ── 4. Repartition and process ─────────────────────────────────────
        effective_partitions = min(NUM_PARTITIONS, todo_count)
        df_partitioned = df_todo.repartition(effective_partitions)
        print(f"Repartitioned into {effective_partitions} partitions")

        print(f"\nStarting parallel processing at {datetime.now(timezone.utc).isoformat()}...")
        start_time = time.time()

        df_results = df_partitioned.mapInPandas(process_company_batch, schema=RESULT_SCHEMA)

        (
            df_results
            .write
            .format("delta")
            .mode("append")
            .saveAsTable(FULL_TABLE_NAME)
        )

        elapsed = time.time() - start_time
        mins, secs = divmod(int(elapsed), 60)
        print(f"\nParallel processing complete in {mins}m {secs}s")
        print(f"Results appended to: {FULL_TABLE_NAME}")

# COMMAND ----------

# DBTITLE 1,Summary stats
# ── Summary statistics ─────────────────────────────────────────────────────────
from pyspark.sql import functions as F

# Ensure variables are defined (handles fresh kernel after restart)
try:
    EXPORT_ONLY
except NameError:
    EXPORT_ONLY = dbutils.widgets.get("export_only").lower() == "true"
    EXPORT_RUN_ID = dbutils.widgets.get("export_run_id").strip()
    RUN_ID = "latest"
    FULL_TABLE_NAME = "client_intelligence_analytics.corporate_market_signals.signals_results"

# Determine which run to display
if EXPORT_ONLY:
    _target_run_id = EXPORT_RUN_ID
else:
    _target_run_id = RUN_ID

if _target_run_id.lower() == "all":
    df_run = spark.table(FULL_TABLE_NAME)
    _run_label = "ALL RUNS"
elif _target_run_id.lower() == "latest":
    _latest = (
        spark.table(FULL_TABLE_NAME)
        .select("run_id", "processed_at")
        .orderBy(F.col("processed_at").desc())
        .limit(1).collect()
    )
    if _latest:
        _target_run_id = _latest[0]["run_id"]
        df_run = spark.table(FULL_TABLE_NAME).filter(F.col("run_id") == _target_run_id)
        _run_label = f"LATEST RUN ({_target_run_id[:8]}...)"
    else:
        df_run = spark.table(FULL_TABLE_NAME)
        _run_label = "NO RUNS FOUND"
else:
    df_run = spark.table(FULL_TABLE_NAME).filter(F.col("run_id") == _target_run_id)
    _run_label = f"RUN {_target_run_id[:8]}..."

total_processed = df_run.count()
total_errors = df_run.filter(F.col("error").isNotNull()).count()
total_with_signals = df_run.filter(F.col("total_signals") > 0).count()

print("=" * 60)
print(f"CORPORATE MARKET SIGNALS — {_run_label}")
print("=" * 60)
print(f"Run ID             : {_target_run_id}")
print(f"Companies processed: {total_processed}")
print(f"Errors             : {total_errors}")
print(f"With signals       : {total_with_signals}")
print(f"Signal rate        : {total_with_signals / max(total_processed, 1) * 100:.1f}%")
print("=" * 60)

# Category breakdown
df_cats = df_run.agg(
    F.sum(F.col("sector_change").cast("int")).alias("sector"),
    F.sum(F.col("hq_change").cast("int")).alias("hq"),
    F.sum(F.col("ma_spinoff").cast("int")).alias("ma"),
    F.sum(F.col("renaming").cast("int")).alias("rename"),
    F.sum(F.col("operational_change").cast("int")).alias("ops"),
    F.sum(F.col("shutdown").cast("int")).alias("shutdown"),
    F.sum(F.col("bankruptcy").cast("int")).alias("bankruptcy"),
    F.sum("total_signals").alias("total_signals"),
).collect()[0]

print(f"\nSignal breakdown:")
print(f"  Sector changes     : {df_cats['sector']}")
print(f"  HQ relocations     : {df_cats['hq']}")
print(f"  M&A / Spinoffs     : {df_cats['ma']}")
print(f"  Renamings          : {df_cats['rename']}")
print(f"  Operational changes: {df_cats['ops']}")
print(f"  Shutdowns          : {df_cats['shutdown']}")
print(f"  Bankruptcies       : {df_cats['bankruptcy']}")
print(f"  Total signals      : {df_cats['total_signals']}")

# All-time table stats
total_all_time = spark.table(FULL_TABLE_NAME).count()
print(f"\nAll-time table size: {total_all_time} companies")

# Show errors if any
if total_errors > 0:
    print(f"\n⚠️ {total_errors} companies had errors:")
    display(
        df_run.filter(F.col("error").isNotNull())
        .select("company", "error").limit(20)
    )

# COMMAND ----------

# DBTITLE 1,Excel export (signals only)
# Export companies with signals to Excel using the pipeline's excel_writer.
import os
import sys
import shutil
import tempfile
import json as _json
from datetime import datetime, timezone
from pyspark.sql import functions as F

# Ensure variables are defined (handles fresh kernel after restart)
try:
    PIPELINE_DIR
except NameError:
    PIPELINE_DIR = dbutils.widgets.get("pipeline_dir")
    FULL_TABLE_NAME = "client_intelligence_analytics.corporate_market_signals.signals_results"
try:
    _target_run_id
except NameError:
    _export_run_id = dbutils.widgets.get("export_run_id").strip()
    if _export_run_id.lower() == "latest":
        _latest = (
            spark.table(FULL_TABLE_NAME)
            .select("run_id", "processed_at")
            .orderBy(F.col("processed_at").desc())
            .limit(1).collect()
        )
        _target_run_id = _latest[0]["run_id"] if _latest else "all"
    else:
        _target_run_id = _export_run_id

if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)
os.chdir(PIPELINE_DIR)

_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
OUTPUT_XLSX = f"/Volumes/client_intelligence_analytics/corporate_market_signals/corporate_market_signals/output/market_signals_report_{_timestamp}.xlsx"
EXPORT_LIMIT = 5000

# Load from Delta — only companies with signals
_export_base = spark.table(FULL_TABLE_NAME)
if _target_run_id.lower() != "all":
    _export_base = _export_base.filter(F.col("run_id") == _target_run_id)

df_export = (
    _export_base
    .filter(F.col("total_signals") > 0)
    .orderBy(F.col("confidence_score").desc(), F.col("total_signals").desc())
    .limit(EXPORT_LIMIT)
    .toPandas()
)
print(f"Exporting {len(df_export)} companies with signals (run: {_target_run_id[:8] if len(_target_run_id) > 8 else _target_run_id})")

if len(df_export) == 0:
    print("No signals found — skipping Excel export.")
else:
    from classifier import SignalResult
    import excel_writer

    signal_results = []
    for _, row in df_export.iterrows():
        sr = SignalResult(
            company=row["company"],
            sector_change=bool(row.get("sector_change", False)),
            hq_change=bool(row.get("hq_change", False)),
            hq_region=row.get("hq_region", "") or "",
            ma_spinoff=bool(row.get("ma_spinoff", False)),
            renaming=bool(row.get("renaming", False)),
            operational_change=bool(row.get("operational_change", False)),
            shutdown=bool(row.get("shutdown", False)),
            bankruptcy=bool(row.get("bankruptcy", False)),
            sector_detail=row.get("sector_detail", "") or "",
            hq_detail=row.get("hq_detail", "") or "",
            ma_detail=row.get("ma_detail", "") or "",
            rename_detail=row.get("rename_detail", "") or "",
            ops_detail=row.get("ops_detail", "") or "",
            bankruptcy_detail=row.get("bankruptcy_detail", "") or "",
            summary=row.get("summary", "") or "",
        )
        try:
            sr.sources = _json.loads(row.get("sources", "[]") or "[]")
        except (TypeError, _json.JSONDecodeError):
            sr.sources = []
        sr.recount()
        signal_results.append(sr)

    _tmp_dir = tempfile.mkdtemp()
    _tmp_xlsx = os.path.join(_tmp_dir, "market_signals_report.xlsx")

    excel_writer.write_report(
        results=signal_results,
        output_path=_tmp_xlsx,
    )

    # Copy to UC Volume (avoids FUSE seek issue)
    os.makedirs(os.path.dirname(OUTPUT_XLSX), exist_ok=True)
    shutil.copy2(_tmp_xlsx, OUTPUT_XLSX)
    shutil.rmtree(_tmp_dir)

    print(f"\n{'='*60}")
    print(f"EXCEL REPORT EXPORTED")
    print(f"{'='*60}")
    print(f"Companies with signals: {len(signal_results)}")
    print(f"Output: {OUTPUT_XLSX}")
