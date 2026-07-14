# Databricks notebook source
# MAGIC %md # Market Signals — full-list production run
# MAGIC Free sources + registry (GLEIF Delta) → results Delta table.
# MAGIC Durable resume (checkpoint on Volume). Optional Tavily thin-evidence
# MAGIC re-run at the bottom.

# COMMAND ----------

# MAGIC %pip install openpyxl requests tavily-python --quiet

# COMMAND ----------

import os, sys, importlib

CODE = "/Workspace/Users/aetingu@gmail.com/market-signals-pipeline"
VOL  = "/Volumes/workspace/default/market_signals"
RESULTS_TABLE = "workspace.default.market_signals_results"
RUN_NAME      = "full-free-v1"          # change per run; used to query results

INPUT = f"{VOL}/full.csv"               # your company list (with LEI column)

sys.path.insert(0, CODE)
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
os.environ["DATABRICKS_HOST"]      = ctx.apiUrl().get()
os.environ["DATABRICKS_TOKEN"]     = ctx.apiToken().get()
os.environ["USE_DATABRICKS_MODEL"] = "True"
os.environ["EDGAR_EMAIL"]          = "aetingu@gmail.com"
os.environ["PIPELINE_MAX_WORKERS"] = "8"
os.environ["CHECKPOINT_DIR"]       = VOL         # resume survives cluster restarts
os.environ["TAVILY_API_KEY"]       = "tvly-YOUR_KEY_HERE"   # placeholder = free-only

import config
importlib.reload(config)
print("date window:", config.DATE_RANGE, "| workers:", config.PIPELINE_MAX_WORKERS)

# COMMAND ----------

from pipeline import load_companies, run_pipeline
from registry_delta import build_gleif_map

companies = load_companies(INPUT)
print(f"{len(companies)} companies, {sum(1 for c in companies if c[3])} with LEI")

gleif_map = build_gleif_map(
    spark,
    [(name, country, lei) for name, _sector, country, lei in companies],
    level1_table="gleif_project.gleif_db.bronze_gleif_lei_cdf",
    level2_table="gleif_project.gleif_db.silver_gleif_relationships",
)

# COMMAND ----------

import shutil, glob

OUT_TMP = f"/tmp/market_signals_{RUN_NAME}.xlsx"
results = run_pipeline(
    input_csv     = INPUT,
    output_xlsx   = OUT_TMP,
    tavily_key    = None if config.TAVILY_API_KEY.startswith("tvly-YOUR") else config.TAVILY_API_KEY,
    resume        = True,        # checkpoint on the Volume — safe to re-run
    max_companies = 0,           # all
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

# MAGIC %md ## Tavily re-run for thin-evidence companies (run when you have credits)
# MAGIC 1. Set a REAL TAVILY_API_KEY in the env cell above and re-run that cell.
# MAGIC 2. Run the cells below — they export the thin companies from the run
# MAGIC    named above and push only those back through the pipeline. The
# MAGIC    thin-evidence gate spends credits on exactly these companies.

# COMMAND ----------

# from pipeline import load_companies, run_pipeline
# import results_delta, shutil
#
# THIN_RUN = f"{RUN_NAME}-tavily"
# THIN_CSV = f"{VOL}/thin_rerun.csv"
# n = results_delta.export_thin_companies_csv(spark, RESULTS_TABLE, RUN_NAME, THIN_CSV)
# print(f"{n} companies → estimated ≤{n*2} Tavily credits")
#
# thin_companies = load_companies(THIN_CSV)
# OUT2 = f"/tmp/market_signals_{THIN_RUN}.xlsx"
# results2 = run_pipeline(input_csv=THIN_CSV, output_xlsx=OUT2,
#                         tavily_key=config.TAVILY_API_KEY,
#                         resume=True, max_companies=0, gleif_map=gleif_map)
# shutil.copy2(OUT2, f"{VOL}/market_signals_{THIN_RUN}.xlsx")
# results_delta.write_results_table(spark, RESULTS_TABLE, results2, thin_companies,
#                                   OUT2.replace(".xlsx", "_prescreen_log.csv"),
#                                   run_name=THIN_RUN)
