# Databricks notebook source
# MAGIC %md
# MAGIC # Market Signals Pipeline — Databricks
# MAGIC
# MAGIC This notebook runs the full corporate change signal detection pipeline
# MAGIC on Databricks using Foundation Model APIs (no local GPU or llama-server needed).
# MAGIC
# MAGIC **Architecture:**
# MAGIC - Stage 1 — Free sources (EDGAR, Google News, DuckDuckGo, PR Newswire, Business Wire, Wikipedia)
# MAGIC - Stage 1C — Prescreener: `databricks-meta-llama-3-1-8b-instruct` (fast, cheap)
# MAGIC - Stage 2 — Tavily deep fetch (gated — only for companies that pass prescreener)
# MAGIC - Stage 2 Classifier — `databricks-qwen3-next-80b-a3b-instruct` (high quality)
# MAGIC - Output — pixel-perfect Excel workbook written to DBFS, downloadable from Files tab

# COMMAND ----------
# MAGIC %md ## Cell 1 — Install dependencies

# COMMAND ----------

# %pip install tavily-python requests openpyxl
# Uncomment the line above and run this cell first, then restart the Python kernel.
# After kernel restart, continue from Cell 2 onwards.

# COMMAND ----------
# MAGIC %md ## Cell 2 — Upload pipeline files & configure paths

# COMMAND ----------

import os, sys

# ── Where you uploaded the pipeline zip on DBFS ──────────────────────────────
# After uploading market_signals_pipeline_v4.zip via the Databricks UI:
#   Workspace → your folder → Import → select the zip
# Files land at /Workspace/Users/<your-email>/market_signals_pipeline/
# OR upload to DBFS via: Catalog → + Add → Upload files → /FileStore/

PIPELINE_DIR = "/Workspace/Users/aetingu@gmail.com/market_signals_pipeline"
# If you uploaded to DBFS instead:
# PIPELINE_DIR = "/dbfs/FileStore/market_signals_pipeline"

# Add pipeline dir to Python path so imports work
if PIPELINE_DIR not in sys.path:
    sys.path.insert(0, PIPELINE_DIR)

# ── Input / Output paths ─────────────────────────────────────────────────────
# Upload your company CSV to DBFS: Catalog → + Add → Upload files
INPUT_CSV    = "/dbfs/FileStore/company_list_sample.csv"
OUTPUT_XLSX  = "/dbfs/FileStore/market_signals_report.xlsx"

print(f"Pipeline dir : {PIPELINE_DIR}")
print(f"Input CSV    : {INPUT_CSV}")
print(f"Output XLSX  : {OUTPUT_XLSX}")
print(f"Pipeline dir exists: {os.path.exists(PIPELINE_DIR)}")
print(f"Input CSV exists   : {os.path.exists(INPUT_CSV)}")

# COMMAND ----------
# MAGIC %md ## Cell 3 — Auto-configure Databricks credentials
# MAGIC
# MAGIC Inside a Databricks notebook the token and host are available automatically.
# MAGIC We inject them into config so the pipeline uses the FMAPI.

# COMMAND ----------

# Auto-inject Databricks credentials from notebook context
try:
    ctx   = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    token = ctx.apiToken().get()
    host  = ctx.apiUrl().get()
    print(f"Auto-detected host : {host}")
    print(f"Token              : dapi...{token[-6:]}")
except Exception as e:
    # Running outside a Databricks notebook (e.g. local test)
    # Set these manually:
    token = os.environ.get("DATABRICKS_TOKEN", "")
    host  = os.environ.get("DATABRICKS_HOST",  "")
    print(f"Manual config — host: {host}")

# Inject into environment so config.py picks them up
os.environ["DATABRICKS_TOKEN"]       = token
os.environ["DATABRICKS_HOST"]        = host
os.environ["USE_DATABRICKS_MODEL"]   = "True"

# Also set your Tavily key here if you have one
# os.environ["TAVILY_API_KEY"] = "tvly-YOUR_KEY_HERE"

# COMMAND ----------
# MAGIC %md ## Cell 4 — Verify imports and config

# COMMAND ----------

import importlib, config
importlib.reload(config)   # reload so env vars are picked up

print(f"USE_DATABRICKS_MODEL        : {config.USE_DATABRICKS_MODEL}")
print(f"DATABRICKS_HOST             : {config.DATABRICKS_HOST[:40]}...")
print(f"DATABRICKS_CLASSIFIER       : {config.DATABRICKS_CLASSIFIER_ENDPOINT}")
print(f"DATABRICKS_PRESCREENER      : {config.DATABRICKS_PRESCREENER_ENDPOINT}")
print(f"TAVILY_API_KEY set          : {not config.TAVILY_API_KEY.startswith('tvly-YOUR')}")
print(f"MIN_CONFIDENCE              : {config.MIN_CONFIDENCE}")
print(f"PRESCREEN_MIN_SCORE         : {config.PRESCREEN_MIN_SCORE}")

# COMMAND ----------
# MAGIC %md ## Cell 5 — Quick connectivity test (FMAPI + Tavily)

# COMMAND ----------

from databricks_client import DatabricksModelClient

# Test prescreener model
print("Testing prescreener model...")
test_client = DatabricksModelClient(endpoint=config.DATABRICKS_PRESCREENER_ENDPOINT)
resp = test_client.complete(
    system_prompt = "You are a tester. Reply with exactly: OK",
    user_prompt   = "Are you working?",
    max_tokens    = 10,
)
print(f"  Prescreener response: {resp.strip()}")

# Test classifier model
print("Testing classifier model...")
test_client2 = DatabricksModelClient(endpoint=config.DATABRICKS_CLASSIFIER_ENDPOINT)
resp2 = test_client2.complete(
    system_prompt = "You are a tester. Reply with exactly: OK",
    user_prompt   = "Are you working?",
    max_tokens    = 10,
)
print(f"  Classifier response : {resp2.strip()}")

# Test Tavily (optional)
if not config.TAVILY_API_KEY.startswith("tvly-YOUR"):
    from tavily import TavilyClient
    tc   = TavilyClient(api_key=config.TAVILY_API_KEY)
    tres = tc.search("Goldman Sachs merger 2025", max_results=1)
    print(f"  Tavily test: {len(tres.get('results', []))} result(s) returned ✓")
else:
    print("  Tavily key not set — Stage 2 deep fetch will be disabled")

print("\nAll connectivity tests passed ✓")

# COMMAND ----------
# MAGIC %md ## Cell 6 — Run the pipeline

# COMMAND ----------

from pipeline import run_pipeline
import time

start = time.time()

results = run_pipeline(
    input_csv   = INPUT_CSV,
    output_xlsx = OUTPUT_XLSX,
    tavily_key  = config.TAVILY_API_KEY if not config.TAVILY_API_KEY.startswith("tvly-YOUR") else None,
    resume      = True,   # set False to start fresh (ignores checkpoint)
)

elapsed = time.time() - start
mins, secs = divmod(int(elapsed), 60)
print(f"\nPipeline complete: {len(results)} companies in {mins}m {secs}s")
print(f"Output written to: {OUTPUT_XLSX}")

# COMMAND ----------
# MAGIC %md ## Cell 7 — Summary & download link

# COMMAND ----------

signals_found  = sum(1 for r in results if r.total_signals > 0)
sector_count   = sum(1 for r in results if r.sector_change)
hq_count       = sum(1 for r in results if r.hq_change)
ma_count        = sum(1 for r in results if r.ma_spinoff)
rename_count   = sum(1 for r in results if r.renaming)
ops_count      = sum(1 for r in results if r.operational_change)
bk_count       = sum(1 for r in results if r.bankruptcy)
shutdown_count = sum(1 for r in results if r.shutdown)

print("=" * 60)
print(f"  PIPELINE SUMMARY")
print("=" * 60)
print(f"  Total companies     : {len(results)}")
print(f"  With signals        : {signals_found}")
print(f"  Sector changes      : {sector_count}")
print(f"  HQ changes          : {hq_count}")
print(f"  M&A / Spinoffs      : {ma_count}")
print(f"  Renamings           : {rename_count}")
print(f"  Operational changes : {ops_count}")
print(f"  Bankruptcies        : {bk_count}")
print(f"  Confirmed shutdowns : {shutdown_count}")
print("=" * 60)

# Companies with signals
flagged = [r.company for r in results if r.total_signals > 0]
if flagged:
    print(f"\n  Flagged companies:")
    for c in flagged:
        r = next(x for x in results if x.company == c)
        print(f"    {c:45s} {r.total_signals} signal(s)")

print(f"\n  Download output from:")
print(f"  Catalog → DBFS → FileStore → market_signals_report.xlsx")
print(f"\n  OR run this in a cell:")
print(f"  files.download('{OUTPUT_XLSX}')")

# COMMAND ----------
# MAGIC %md ## Cell 8 — Download the Excel file (optional)

# COMMAND ----------

# Uncomment to trigger browser download of the Excel file
# files.download(OUTPUT_XLSX)

# OR copy to your Workspace folder for persistent storage:
# dbutils.fs.cp(f"dbfs:/FileStore/market_signals_report.xlsx",
#               f"/Workspace/Users/aetingu@gmail.com/market_signals_report.xlsx")

# COMMAND ----------
# MAGIC %md ## Cell 9 — Schedule as a Databricks Job (optional)
# MAGIC
# MAGIC To run this pipeline on a schedule (e.g. every Monday at 7am):
# MAGIC
# MAGIC 1. Click **Schedule** (top right of this notebook)
# MAGIC 2. Set the schedule: Weekly, Monday, 07:00 UTC
# MAGIC 3. Set the cluster: use a **Serverless** cluster or a small CPU cluster (Standard_DS3_v2)
# MAGIC 4. The job will automatically:
# MAGIC    - Install dependencies from requirements.txt
# MAGIC    - Run all cells top to bottom
# MAGIC    - Write the output XLSX to DBFS
# MAGIC    - Send a completion email (configure in Job settings → Notifications)

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC **Pipeline version:** v4 | **Models:** Qwen3-Next 80B (classifier) + Llama 3.1 8B (prescreener)
# MAGIC **Output:** `/dbfs/FileStore/market_signals_report.xlsx`
