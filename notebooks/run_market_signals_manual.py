# Databricks notebook source
# MAGIC %md
# MAGIC # Market Signals Pipeline — Manual Databricks Runner
# MAGIC
# MAGIC This notebook is intended to be run manually or from a Databricks Job.
# MAGIC It reads a company list CSV from a Unity Catalog Volume, runs the pipeline,
# MAGIC and writes the Excel report back to a Volume.

# COMMAND ----------

# MAGIC %pip install tavily-python requests openpyxl

# COMMAND ----------

import os
import sys
import time
import importlib

# COMMAND ----------

# Databricks Job parameters. These can be overridden with "Run now with different parameters".
dbutils.widgets.text(
    "pipeline_dir",
    "/Workspace/Users/aksel.etingu@crisil.com/market-signals-pipeline",
)

dbutils.widgets.text(
    "input_csv",
    "/Volumes/data_poc_ws/default/client_intelligence_analytics/market_signals/input/corporate_100.csv",
)

dbutils.widgets.text(
    "output_xlsx",
    "/Volumes/data_poc_ws/default/client_intelligence_analytics/market_signals/output/market_signals_report.xlsx",
)

dbutils.widgets.dropdown(
    "resume",
    "true",
    ["true", "false"],
)

dbutils.widgets.dropdown(
    "use_tavily",
    "true",
    ["true", "false"],
)

dbutils.widgets.dropdown(
    "time_horizon_months",
    "12",
    ["6", "12", "24"],
)

dbutils.widgets.text(
    "max_companies",
    "0",
)

dbutils.widgets.text(
    "classifier_endpoint",
    "databricks-qwen3-next-80b-a3b-instruct",
)

dbutils.widgets.text(
    "prescreener_endpoint",
    "databricks-meta-llama-3-1-8b-instruct",
)

# GLEIF golden-copy Delta tables (Unity Catalog). When gleif_level1_table is
# set, GLEIF registry facts come from ONE batch join instead of per-company
# API calls (60 req/min → ~85 min saved on a 5,000-company list).
# Leave empty to use the GLEIF public API instead.
dbutils.widgets.text(
    "gleif_level1_table",
    "",
)

dbutils.widgets.text(
    "gleif_level2_table",   # relationships (RR) — enables ultimate-parent facts
    "",
)

# COMMAND ----------

PIPELINE_DIR = dbutils.widgets.get("pipeline_dir")
INPUT_CSV = dbutils.widgets.get("input_csv")
OUTPUT_XLSX = dbutils.widgets.get("output_xlsx")
RESUME = dbutils.widgets.get("resume").lower() == "true"
USE_TAVILY = dbutils.widgets.get("use_tavily").lower() == "true"
TIME_HORIZON_MONTHS = dbutils.widgets.get("time_horizon_months")
MAX_COMPANIES = int((dbutils.widgets.get("max_companies") or "0").strip())
CLASSIFIER_ENDPOINT = dbutils.widgets.get("classifier_endpoint")
PRESCREENER_ENDPOINT = dbutils.widgets.get("prescreener_endpoint")

if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

os.makedirs(os.path.dirname(OUTPUT_XLSX), exist_ok=True)

print(f"Pipeline dir         : {PIPELINE_DIR}")
print(f"Input CSV            : {INPUT_CSV}")
print(f"Output XLSX          : {OUTPUT_XLSX}")
print(f"Resume               : {RESUME}")
print(f"Use Tavily           : {USE_TAVILY}")
print(f"Time horizon months  : {TIME_HORIZON_MONTHS}")
print(f"Max companies        : {MAX_COMPANIES if MAX_COMPANIES else 'ALL'}")
print(f"Classifier endpoint  : {CLASSIFIER_ENDPOINT}")
print(f"Prescreener endpoint : {PRESCREENER_ENDPOINT}")
print(f"Pipeline dir exists  : {os.path.exists(PIPELINE_DIR)}")
print(f"Input CSV exists     : {os.path.exists(INPUT_CSV)}")

if not os.path.exists(PIPELINE_DIR):
    raise FileNotFoundError(f"Pipeline dir not found: {PIPELINE_DIR}")
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"Input CSV not found: {INPUT_CSV}")

# COMMAND ----------

# Auto-configure Databricks credentials from the notebook/job context.
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()

os.environ["DATABRICKS_TOKEN"] = token
os.environ["DATABRICKS_HOST"] = host
os.environ["USE_DATABRICKS_MODEL"] = "True"
os.environ["DATABRICKS_CLASSIFIER_ENDPOINT"] = CLASSIFIER_ENDPOINT
os.environ["DATABRICKS_PRESCREENER_ENDPOINT"] = PRESCREENER_ENDPOINT
os.environ["TIME_HORIZON_MONTHS"] = TIME_HORIZON_MONTHS
os.environ["MAX_COMPANIES"] = str(MAX_COMPANIES)

print(f"Detected Databricks host: {host}")

# COMMAND ----------

# Optional Tavily key. Preferred storage:
#   scope = market-signals
#   key   = tavily-api-key
if USE_TAVILY:
    try:
        tavily_key = dbutils.secrets.get(
            scope="market-signals",
            key="tavily-api-key",
        )
        os.environ["TAVILY_API_KEY"] = tavily_key
        print("Tavily key loaded from Databricks Secrets.")
    except Exception:
        os.environ["TAVILY_API_KEY"] = "tvly-YOUR_KEY_HERE"
        print("Tavily secret not found. Stage 2 will use free sources only.")
else:
    os.environ["TAVILY_API_KEY"] = "tvly-YOUR_KEY_HERE"
    print("Tavily disabled for this run.")

# COMMAND ----------

import config
importlib.reload(config)

print(f"USE_DATABRICKS_MODEL   : {config.USE_DATABRICKS_MODEL}")
print(f"DATABRICKS_HOST        : {config.DATABRICKS_HOST[:40]}...")
print(f"DATABRICKS_CLASSIFIER  : {config.DATABRICKS_CLASSIFIER_ENDPOINT}")
print(f"DATABRICKS_PRESCREENER : {config.DATABRICKS_PRESCREENER_ENDPOINT}")
print(f"TAVILY_API_KEY set     : {not config.TAVILY_API_KEY.startswith('tvly-YOUR')}")
print(f"TIME_HORIZON_MONTHS    : {config.TIME_HORIZON_MONTHS}")
print(f"DATE_START             : {config.DATE_START}")
print(f"DATE_END               : {config.DATE_END}")
print(f"DATE_RANGE             : {config.DATE_RANGE}")
print(f"MAX_COMPANIES default  : {config.DEFAULT_MAX_COMPANIES}")
print(f"MIN_CONFIDENCE         : {config.MIN_CONFIDENCE}")
print(f"PRESCREEN_MIN_SCORE    : {config.PRESCREEN_MIN_SCORE}")

# COMMAND ----------

# Quick model connectivity test. This catches endpoint-name or permission issues early.
from databricks_client import DatabricksModelClient

print("Testing classifier endpoint...")

test_client = DatabricksModelClient(
    endpoint=config.DATABRICKS_CLASSIFIER_ENDPOINT,
)

response = test_client.complete(
    system_prompt="Reply with exactly OK.",
    user_prompt="Test",
    max_tokens=10,
)

print(f"Model test response: {response.strip()}")

# COMMAND ----------

# Reload pipeline modules after config/env changes so repeated notebook runs stay clean.
import search
import classifier
import prescreener
import pipeline

importlib.reload(search)
importlib.reload(classifier)
importlib.reload(prescreener)
importlib.reload(pipeline)

import shutil
import tempfile
from pipeline import load_companies, run_pipeline

# ── GLEIF from Delta tables (one batch join instead of ~85 min of API calls) ──
GLEIF_L1 = dbutils.widgets.get("gleif_level1_table").strip()
GLEIF_L2 = dbutils.widgets.get("gleif_level2_table").strip() or None

gleif_map = None
if GLEIF_L1:
    import registry_delta
    importlib.reload(registry_delta)
    _companies = load_companies(INPUT_CSV)
    gleif_map = registry_delta.build_gleif_map(
        spark,
        [(name, country, lei) for name, _sector, country, lei in _companies],
        level1_table=GLEIF_L1,
        level2_table=GLEIF_L2,
    )

# Write Excel to local temp file first — Volume FUSE doesn't support seek for zip/xlsx writes
_tmp_xlsx = os.path.join(tempfile.gettempdir(), os.path.basename(OUTPUT_XLSX))

start = time.time()

results = run_pipeline(
    input_csv=INPUT_CSV,
    output_xlsx=_tmp_xlsx,
    tavily_key=config.TAVILY_API_KEY if not config.TAVILY_API_KEY.startswith("tvly-YOUR") else None,
    resume=RESUME,
    max_companies=MAX_COMPANIES,
    gleif_map=gleif_map,
)

shutil.copy2(_tmp_xlsx, OUTPUT_XLSX)
os.remove(_tmp_xlsx)

elapsed = time.time() - start
mins, secs = divmod(int(elapsed), 60)

print(f"Pipeline complete: {len(results)} companies in {mins}m {secs}s")
print(f"Output written to: {OUTPUT_XLSX}")

# COMMAND ----------

signals_found = sum(1 for r in results if r.total_signals > 0)

print("=" * 60)
print("MARKET SIGNALS SUMMARY")
print("=" * 60)
print(f"Total companies : {len(results)}")
print(f"With signals    : {signals_found}")
print(f"Date range      : {config.DATE_RANGE}")
print(f"Output          : {OUTPUT_XLSX}")
print("=" * 60)

for r in results:
    if r.total_signals > 0:
        print(f"{r.company:45s} {r.total_signals} signal(s) - {r.summary}")

dbutils.notebook.exit(
    f"Completed. Companies={len(results)}, Signals={signals_found}, Horizon={config.DATE_RANGE}, Output={OUTPUT_XLSX}"
)
