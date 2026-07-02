# Market Signals Pipeline

Automated corporate change signal detection for a list of companies.

The recommended production path is to run this from **Databricks** as a manually triggered Job, using a company-list CSV stored in a Unity Catalog Volume and a Databricks Foundation Model / Model Serving endpoint for classification.

## What it does

1. Reads a CSV of company names.
2. Fetches corporate-change evidence from free sources and, optionally, Tavily.
3. Prescreens companies using a fast model endpoint.
4. Classifies corporate change signals using a stronger model endpoint such as Qwen.
5. Writes an Excel report with dashboard and detail sheets.

## Signal types detected

- Sector/Subsector Change — company moved to a different industry or business focus.
- HQ/Domicile Change — HQ relocation or legal redomicile.
- M&A / Spinoffs — mergers, acquisitions, takeovers, divestitures, or spinoffs.
- Renaming/Rebranding — legal name change or major brand rename.
- Operational Change/Shutdown — restructuring, site closures, major workforce reductions, or full wind-down.
- Bankruptcy/Liquidation — insolvency, administration, liquidation, or bankruptcy filing.

## Recommended Databricks architecture

```text
GitHub repo
  FatCatAnalytics/market-signals-pipeline
        ↓
Databricks Git folder
        ↓
Manual Databricks Job
        ↓
Input CSV in Unity Catalog Volume
        ↓
Databricks Foundation Model / Model Serving endpoint
        ↓
Excel output written back to Volume
```

## Required repository files

```text
classifier.py
config.py
databricks_client.py
excel_writer.py
pipeline.py
prescreener.py
search.py
search_layer.py
requirements.txt
notebooks/run_market_signals_manual.py
```

## Input CSV format

The input CSV must contain one company-name column.

Supported company column names:

- `company`
- `company_name`
- `name`
- `entity`

Optional sector column names:

- `sector`
- `industry`
- `subsector`

Example:

```csv
company,sector
Microsoft,Technology
JPMorgan Chase,Banking
Amazon,Technology
BlackRock,Asset Management
PayPal,Payments
```

## Databricks setup

### 1. Connect Databricks to GitHub

In Databricks:

```text
User menu → Settings → Developer → Git integration
```

Then create a Git folder from:

```text
https://github.com/FatCatAnalytics/market-signals-pipeline
```

### 2. Create Volume folders

Example paths:

```text
/Volumes/data_poc_ws/default/client_intelligence_analytics/market_signals/input
/Volumes/data_poc_ws/default/client_intelligence_analytics/market_signals/output
/Volumes/data_poc_ws/default/client_intelligence_analytics/market_signals/archive
```

### 3. Upload the company list

Upload your CSV to:

```text
/Volumes/data_poc_ws/default/client_intelligence_analytics/market_signals/input/company_list.csv
```

### 4. Add Tavily key as a Databricks secret

The pipeline can run without Tavily, but Tavily improves Stage 2 evidence gathering.

Recommended secret location:

```text
scope = market-signals
key   = tavily-api-key
```

CLI example:

```bash
databricks secrets create-scope market-signals
databricks secrets put-secret market-signals tavily-api-key
```

### 5. Confirm your model endpoint names

In Databricks:

```text
Serving → Endpoints
```

Use the exact endpoint name for the classifier parameter.

Examples:

```text
databricks-qwen3-next-80b-a3b-instruct
qwen3-6-corporate-classifier
your-own-qwen-endpoint-name
```

Do not guess the endpoint name. It must match a serving endpoint available in your workspace.

## Manual Databricks Job

Create a Databricks Job with a notebook task:

```text
Notebook path: /Workspace/Users/aetingu@gmail.com/market-signals-pipeline/notebooks/run_market_signals_manual
```

Recommended parameters:

| Parameter | Example |
|---|---|
| `pipeline_dir` | `/Workspace/Users/aetingu@gmail.com/market-signals-pipeline` |
| `input_csv` | `/Volumes/data_poc_ws/default/client_intelligence_analytics/market_signals/input/company_list.csv` |
| `output_xlsx` | `/Volumes/data_poc_ws/default/client_intelligence_analytics/market_signals/output/market_signals_report.xlsx` |
| `resume` | `true` |
| `use_tavily` | `true` |
| `classifier_endpoint` | `databricks-qwen3-next-80b-a3b-instruct` |
| `prescreener_endpoint` | `databricks-meta-llama-3-1-8b-instruct` |

Run manually with:

```text
Workflows → Jobs → Run now
```

For a one-off run with a different input/output/model endpoint, use:

```text
Run now with different parameters
```

## Local run option

The repo still supports local execution with a local `llama-server`, useful for development outside Databricks.

```bash
pip install -r requirements.txt
python pipeline.py --input company_list_sample.csv --output market_signals_report.xlsx
```

For local Qwen/GGUF usage, start `llama-server` first and keep `USE_DATABRICKS_MODEL=False`.

## Output sheets

| Sheet | Contents |
|---|---|
| Signal Summary Dashboard | All companies, tick/dash grid, totals, summaries |
| Sector Changes | Companies with sector change detail |
| HQ Changes | HQ moves + USA region classification |
| M&A & Spinoffs | Merger/acquisition/divestiture detail |
| Renaming Rebranding | Name change detail |
| Operational Changes | Restructuring, shutdowns, significant cuts |
| Bankruptcy | Filing type and date |
| Pipeline Stats | Run metadata, signal counts, runtime |

## Generated files

Typical generated outputs:

```text
market_signals_report.xlsx
market_signals_report_checkpoint.json
market_signals_report_prescreen_log.csv
```

These files should be written to a Databricks Volume and are ignored by Git.
