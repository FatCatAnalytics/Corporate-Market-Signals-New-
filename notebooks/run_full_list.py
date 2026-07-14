# Databricks notebook source
# MAGIC %md # Market Signals — full-list production run
# MAGIC All parameters are notebook widgets (top of the UI, or "Run now with
# MAGIC different parameters" when scheduled as a job). Free sources +
# MAGIC registry (GLEIF Delta) → Excel + results Delta table with
# MAGIC thin-evidence flags. Durable resume. Optional Tavily thin-rerun at
# MAGIC the bottom.

# COMMAND ----------

# Parameters — edit in the widget bar before running.
dbutils.widgets.text("input_csv",  "/Volumes/workspace/default/market_signals/full.csv", "1. Input CSV (Volume path)")
dbutils.widgets.text("run_name",   "full-free-v1",                                       "2. Run name (tags results table)")
dbutils.widgets.text("results_table", "workspace.default.market_signals_results",        "3. Results Delta table")
dbutils.widgets.text("output_volume", "/Volumes/workspace/default/market_signals",       "4. Output Volume dir")
dbutils.widgets.dropdown("time_horizon_months", "12", ["6", "12", "24"],                 "5. Lookback (months)")
dbutils.widgets.text("max_companies", "0",                                               "6. Max companies (0 = all)")
dbutils.widgets.dropdown("resume", "true", ["true", "false"],                            "7. Resume from checkpoint")
dbutils.widgets.text("pipeline_max_workers", "8",                                        "8. Parallel workers")
dbutils.widgets.text("model_max_concurrent", "4",                                        "9. Concurrent model calls")
dbutils.widgets.text("classifier_endpoint",  "databricks-qwen3-next-80b-a3b-instruct",   "10. Classifier endpoint")
dbutils.widgets.text("prescreener_endpoint", "databricks-meta-llama-3-1-8b-instruct",    "11. Prescreener endpoint")
dbutils.widgets.text("gleif_level1_table", "gleif_project.gleif_db.bronze_gleif_lei_cdf","12. GLEIF L1 table ('' = API)")
dbutils.widgets.text("gleif_level2_table", "gleif_project.gleif_db.silver_gleif_relationships", "13. GLEIF L2 table (optional)")
dbutils.widgets.text("tavily_api_key", "",                                               "14. Tavily key ('' = free-only)")
dbutils.widgets.text("tavily_budget", "2000",                                            "15. Tavily budget (queries)")
dbutils.widgets.text("tavily_thin_threshold", "3000",                                    "16. Thin threshold (chars)")
dbutils.widgets.text("tavily_queries_per_company", "2",                                  "17. Tavily queries/company")

# COMMAND ----------

# MAGIC %pip install openpyxl requests tavily-python --quiet

# COMMAND ----------

import os, sys, importlib

W = lambda name: dbutils.widgets.get(name).strip()

CODE          = "/Workspace/Users/aetingu@gmail.com/market-signals-pipeline"
VOL           = W("output_volume")
INPUT         = W("input_csv")
RUN_NAME      = W("run_name")
RESULTS_TABLE = W("results_table")
GLEIF_L1      = W("gleif_level1_table")
GLEIF_L2      = W("gleif_level2_table") or None
TAVILY_KEY    = W("tavily_api_key")   # tip: prefer dbutils.secrets.get(...) over pasting

sys.path.insert(0, CODE)
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
os.environ["DATABRICKS_HOST"]                 = ctx.apiUrl().get()
os.environ["DATABRICKS_TOKEN"]                = ctx.apiToken().get()
os.environ["USE_DATABRICKS_MODEL"]            = "True"
os.environ["EDGAR_EMAIL"]                     = "aetingu@gmail.com"
os.environ["TIME_HORIZON_MONTHS"]             = W("time_horizon_months")
os.environ["PIPELINE_MAX_WORKERS"]            = W("pipeline_max_workers")
os.environ["MODEL_MAX_CONCURRENT"]            = W("model_max_concurrent")
os.environ["DATABRICKS_CLASSIFIER_ENDPOINT"]  = W("classifier_endpoint")
os.environ["DATABRICKS_PRESCREENER_ENDPOINT"] = W("prescreener_endpoint")
os.environ["CHECKPOINT_DIR"]                  = VOL
os.environ["TAVILY_BUDGET"]                   = W("tavily_budget")
os.environ["TAVILY_THIN_THRESHOLD"]           = W("tavily_thin_threshold")
os.environ["TAVILY_QUERIES_PER_COMPANY"]      = W("tavily_queries_per_company")
os.environ["TAVILY_API_KEY"]                  = TAVILY_KEY or "tvly-YOUR_KEY_HERE"

import config
importlib.reload(config)
print(f"run '{RUN_NAME}' | window {config.DATE_RANGE}")
print(f"workers {config.PIPELINE_MAX_WORKERS} | model concurrency {os.environ['MODEL_MAX_CONCURRENT']}")
print(f"tavily: {'ON — budget ' + W('tavily_budget') if TAVILY_KEY else 'OFF (free-only)'}")

# COMMAND ----------

from pipeline import load_companies, run_pipeline
from registry_delta import build_gleif_map

companies = load_companies(INPUT)
print(f"{len(companies)} companies, {sum(1 for c in companies if c[3])} with LEI")

gleif_map = None
if GLEIF_L1:
    gleif_map = build_gleif_map(
        spark,
        [(name, country, lei) for name, _sector, country, lei in companies],
        level1_table=GLEIF_L1,
        level2_table=GLEIF_L2,
    )

# COMMAND ----------

import shutil, glob

OUT_TMP = f"/tmp/market_signals_{RUN_NAME}.xlsx"
results = run_pipeline(
    input_csv     = INPUT,
    output_xlsx   = OUT_TMP,
    tavily_key    = TAVILY_KEY or None,
    resume        = (W("resume") == "true"),
    max_companies = int(W("max_companies") or 0),
    gleif_map     = gleif_map,
)

shutil.copy2(OUT_TMP, f"{VOL}/market_signals_{RUN_NAME}.xlsx")
prescreen_csv = OUT_TMP.replace(".xlsx", "_prescreen_log.csv")
for f in glob.glob(prescreen_csv):
    shutil.copy2(f, VOL)

# COMMAND ----------

# Persist results to the Delta table (system of record).
import results_delta
importlib.reload(results_delta)

results_delta.write_results_table(
    spark, RESULTS_TABLE, results, companies, prescreen_csv, run_name=RUN_NAME)

display(spark.sql(f"""
  SELECT thin_evidence, COUNT(*) AS companies
  FROM {RESULTS_TABLE} WHERE run_name = '{RUN_NAME}' AND passed = true
  GROUP BY thin_evidence"""))

# COMMAND ----------

# MAGIC %md ## Tavily re-run for thin-evidence companies
# MAGIC 1. Set widget 14 (tavily_api_key) and change widget 2 (run_name) to
# MAGIC    e.g. `full-free-v1-tavily`.
# MAGIC 2. Uncomment and run the cell below with the ORIGINAL run name as
# MAGIC    SOURCE_RUN — it exports that run's thin companies and pushes only
# MAGIC    those back through the pipeline; credits go exactly there.

# COMMAND ----------

# import results_delta, shutil
# from pipeline import load_companies, run_pipeline
#
# SOURCE_RUN = "full-free-v1"
# THIN_CSV   = f"{VOL}/thin_rerun_{SOURCE_RUN}.csv"
# n = results_delta.export_thin_companies_csv(spark, RESULTS_TABLE, SOURCE_RUN, THIN_CSV)
# print(f"{n} companies → estimated ≤{n * int(W('tavily_queries_per_company'))} Tavily credits")
#
# thin_companies = load_companies(THIN_CSV)
# OUT2 = f"/tmp/market_signals_{RUN_NAME}.xlsx"
# results2 = run_pipeline(input_csv=THIN_CSV, output_xlsx=OUT2,
#                         tavily_key=TAVILY_KEY, resume=True,
#                         max_companies=0, gleif_map=gleif_map)
# shutil.copy2(OUT2, f"{VOL}/market_signals_{RUN_NAME}.xlsx")
# results_delta.write_results_table(spark, RESULTS_TABLE, results2, thin_companies,
#                                   OUT2.replace(".xlsx", "_prescreen_log.csv"),
#                                   run_name=RUN_NAME)
