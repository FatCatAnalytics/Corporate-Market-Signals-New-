# Databricks notebook source
# MAGIC %md # Market Signals — 20-company validation run (no Tavily)
# MAGIC Free sources + registry layer (GLEIF from Delta) + Databricks foundation models.

# COMMAND ----------

# MAGIC %pip install openpyxl requests --quiet

# COMMAND ----------

import os, sys, importlib

CODE = "/Workspace/Users/aetingu@gmail.com/market-signals-pipeline"
sys.path.insert(0, CODE)

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
os.environ["DATABRICKS_HOST"]      = ctx.apiUrl().get()
os.environ["DATABRICKS_TOKEN"]     = ctx.apiToken().get()
os.environ["USE_DATABRICKS_MODEL"] = "True"
os.environ["TAVILY_API_KEY"]       = "tvly-YOUR_KEY_HERE"   # explicit: NO Tavily
os.environ["EDGAR_EMAIL"]          = "aetingu@gmail.com"

import config
importlib.reload(config)
print("classifier endpoint :", config.DATABRICKS_CLASSIFIER_ENDPOINT)
print("prescreener endpoint:", config.DATABRICKS_PRESCREENER_ENDPOINT)
print("registry enabled    :", config.REGISTRY_ENABLED, "| fulltext:", config.FULLTEXT_ENABLED)

# COMMAND ----------

# Build the GLEIF map from the Delta tables — one batch join, no API calls.
from pipeline import load_companies, run_pipeline
from registry_delta import build_gleif_map

INPUT = "/Volumes/workspace/default/market_signals/full.csv"
companies = load_companies(INPUT)[:20]
print(f"{len(companies)} companies, {sum(1 for c in companies if c[3])} with LEI")

gleif_map = build_gleif_map(
    spark,
    [(name, country, lei) for name, _sector, country, lei in companies],
    level1_table="gleif_project.gleif_db.bronze_gleif_lei_cdf",
    level2_table="gleif_project.gleif_db.silver_gleif_relationships",
)
for name, (section, _hl, _urls, flags) in list(gleif_map.items())[:3]:
    print("=" * 50); print(section); print("flags:", flags)

# COMMAND ----------

import shutil, glob

OUT_TMP = "/tmp/market_signals_test20.xlsx"
results = run_pipeline(
    input_csv     = INPUT,
    output_xlsx   = OUT_TMP,
    tavily_key    = None,
    resume        = False,
    max_companies = 20,
    gleif_map     = gleif_map,
)

VOL = "/Volumes/workspace/default/market_signals"
shutil.copy2(OUT_TMP, f"{VOL}/market_signals_test20.xlsx")
for f in glob.glob("/tmp/market_signals_test20_prescreen_log.csv"):
    shutil.copy2(f, VOL)

n_signals = sum(1 for r in results if r.total_signals > 0)
print(f"PIPELINE_DONE companies={len(results)} with_signals={n_signals}")
for r in results:
    if r.total_signals:
        print(f"  SIGNAL {r.company}: {r.total_signals} — {r.summary[:150]}")
