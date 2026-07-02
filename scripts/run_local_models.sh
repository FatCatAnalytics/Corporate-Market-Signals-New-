#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_local_models.sh — start both llama.cpp servers for the pipeline
# ─────────────────────────────────────────────────────────────────────────────
# Replaces the two Databricks Foundation Model endpoints:
#   :8080  classifier   (was databricks-qwen3-next-80b-a3b-instruct)
#   :8081  prescreener  (was databricks-meta-llama-3-1-8b-instruct)
#
# Install llama.cpp first:   brew install llama.cpp
#
# First run downloads the GGUFs from Hugging Face into ~/Library/Caches/llama.cpp
# (~19 GB + ~5 GB). Subsequent runs start instantly.
#
# Usage:
#   ./scripts/run_local_models.sh          # start both
#   ./scripts/run_local_models.sh stop     # stop both
set -euo pipefail

CLASSIFIER_MODEL="unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF:Q4_K_M"
PRESCREENER_MODEL="bartowski/Meta-Llama-3.1-8B-Instruct-GGUF:Q4_K_M"
LOG_DIR="${TMPDIR:-/tmp}"

if [[ "${1:-}" == "stop" ]]; then
  pkill -f "llama-server.*--port 808[01]" && echo "Stopped." || echo "Nothing running."
  exit 0
fi

command -v llama-server >/dev/null || { echo "llama-server not found — run: brew install llama.cpp"; exit 1; }

echo "Starting classifier  (${CLASSIFIER_MODEL}) on :8080 ..."
nohup llama-server -hf "$CLASSIFIER_MODEL" -c 16384 --port 8080 \
  > "$LOG_DIR/llama-classifier.log" 2>&1 &

echo "Starting prescreener (${PRESCREENER_MODEL}) on :8081 ..."
nohup llama-server -hf "$PRESCREENER_MODEL" -c 8192 --port 8081 \
  > "$LOG_DIR/llama-prescreener.log" 2>&1 &

echo -n "Waiting for servers"
for port in 8080 8081; do
  until curl -s "http://127.0.0.1:${port}/health" | grep -q ok; do
    echo -n "."; sleep 3
  done
done
echo
echo "Both servers up:  classifier http://127.0.0.1:8080   prescreener http://127.0.0.1:8081"
echo "Logs: $LOG_DIR/llama-classifier.log  $LOG_DIR/llama-prescreener.log"
