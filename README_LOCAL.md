# Running locally (no Databricks)

The pipeline runs fully on your laptop. Databricks foundation models are replaced by two llama.cpp servers; `USE_DATABRICKS_MODEL` stays `False` (the default) and no Databricks credentials are needed.

| Role | Databricks endpoint (before) | Local model (after) | RAM |
|---|---|---|---|
| Classifier | qwen3-next-80b-a3b-instruct | Qwen3-30B-A3B-Instruct Q4_K_M, port 8080 | ~19 GB |
| Prescreener | meta-llama-3-1-8b-instruct | Llama-3.1-8B-Instruct Q4_K_M, port 8081 | ~5 GB |

Both together use ~25 GB — comfortable on 64 GB.

## One-time setup

```bash
brew install llama.cpp
cd market-signals-pipeline
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Set your API keys (or edit `config.py`):

```bash
export TAVILY_API_KEY=tvly-...        # free at tavily.com
export EDGAR_EMAIL=you@example.com    # any real email
```

## Each run

```bash
./scripts/run_local_models.sh         # starts both model servers (first run downloads GGUFs)
python pipeline.py --input company_list_sample.csv --max 5   # smoke test
./scripts/run_local_models.sh stop    # when done
```

Output lands in `market_signals_report.xlsx`.

## Notes

- **One server instead of two:** skip the prescreener server — the pipeline automatically falls back to the classifier server on :8080. Slower per company, simpler to run.
- **Higher quality:** swap the classifier for `unsloth/Qwen3-32B-GGUF:Q4_K_M` (dense, ~20 GB, noticeably slower than the MoE).
- **Faster/cheaper:** prescreener can drop to `bartowski/Qwen2.5-3B-Instruct-GGUF:Q4_K_M`.
- Override URLs with `LLAMA_SERVER_URL` / `LLAMA_PRESCREENER_URL` env vars.
- `pipeline_databricks.py` and `notebooks/` are Databricks-only entry points — ignore them locally; `pipeline.py` is the local entry point.
