"""
databricks_client.py — Databricks Foundation Model API client
=============================================================
Drop-in replacement for LlamaServerClient.
Uses the OpenAI-compatible /invocations endpoint on Databricks Model Serving.

Switching between local llama-server and Databricks is controlled by
USE_DATABRICKS_MODEL in config.py — no other files need to change.

Supported endpoints (set in config.py):
  Prescreener (fast/cheap):
    databricks-meta-llama-3-1-8b-instruct
    databricks-gpt-oss-20b

  Classifier (high quality):
    databricks-qwen3-next-80b-a3b-instruct   ← recommended
    databricks-qwen35-122b-a10b              ← highest quality (preview)
    databricks-meta-llama-3-3-70b-instruct
"""

from __future__ import annotations

import os
import random
import threading
import time
from typing import Optional

import requests

import config


# Process-wide cap on CONCURRENT model-endpoint calls, shared by every
# client instance and every pipeline worker thread. Serving endpoints
# enforce their own concurrency/QPS limits — queueing here converts what
# would be a 429 storm into orderly waiting. Tune via MODEL_MAX_CONCURRENT.
_MODEL_SEMAPHORE = threading.Semaphore(
    max(1, int(__import__("os").environ.get("MODEL_MAX_CONCURRENT", "4")))
)


class DatabricksModelClient:
    """
    Calls Databricks Foundation Model APIs using the OpenAI-compatible
    chat completions format.

    Authentication:
      Set DATABRICKS_TOKEN and DATABRICKS_HOST as environment variables,
      OR configure them in config.py.

    Inside a Databricks notebook, both are available automatically:
      from databricks.sdk.runtime import dbutils
      token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
      host  = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    """

    def __init__(
        self,
        endpoint:    str           = None,
        token:       Optional[str] = None,
        host:        Optional[str] = None,
        timeout_sec: int           = 90,
    ):
        self.endpoint    = endpoint or config.DATABRICKS_CLASSIFIER_ENDPOINT
        self.token       = token   or config.DATABRICKS_TOKEN
        self.host        = (host   or config.DATABRICKS_HOST).rstrip("/")
        self.timeout_sec = timeout_sec
        self.url         = f"{self.host}/serving-endpoints/{self.endpoint}/invocations"

        if not self.token:
            raise ValueError(
                "Databricks token not set.\n"
                "  Option 1: set DATABRICKS_TOKEN in config.py\n"
                "  Option 2: set env var DATABRICKS_TOKEN=dapiXXX\n"
                "  Option 3: inside a notebook, token is auto-injected"
            )
        if not self.host or self.host == "https://YOUR_WORKSPACE.azuredatabricks.net":
            raise ValueError(
                "Databricks host not set.\n"
                "  Set DATABRICKS_HOST in config.py to your workspace URL.\n"
                "  Example: https://adb-1234567890.12.azuredatabricks.net"
            )

    def complete(
        self,
        system_prompt: str,
        user_prompt:   str,
        max_tokens:    int   = 1200,
        temperature:   float = 0.0,
    ) -> str:
        """
        Send a chat completion request. Returns the response text.
        Retries up to 3 times on transient errors (429, 503).
        """
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        return self._invoke(payload)

    def chat(
        self,
        messages:    list[dict],
        max_tokens:  int   = 600,
        temperature: float = 0.0,
    ) -> str:
        """
        Compatibility wrapper for local llama-server style callers.

        Some existing pipeline code calls client.chat(messages=[...]).
        Databricks serving accepts the same message payload, so this wrapper
        keeps older code paths working while complete(...) remains preferred.
        """
        payload = {
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        return self._invoke(payload)

    def _invoke(self, payload: dict) -> str:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type":  "application/json",
        }

        # With the pipeline loop parallelised, N workers can hit the model
        # endpoint simultaneously and trigger a 429 storm where every
        # thread's retries collide and all give up together. Two defences:
        #   1. a process-wide semaphore caps concurrent model calls
        #      (news fetching stays fully parallel; only LLM calls queue);
        #   2. more attempts with jittered exponential backoff, so retries
        #      from different threads spread out instead of re-colliding.
        attempts = 6
        for attempt in range(attempts):
            try:
                with _MODEL_SEMAPHORE:
                    r = requests.post(
                        self.url,
                        headers = headers,
                        json    = payload,
                        timeout = self.timeout_sec,
                    )

                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]

                if r.status_code == 429:
                    retry_after = r.headers.get("Retry-After")
                    if retry_after:
                        wait = min(int(retry_after), 60)
                    else:
                        wait = min(3 * (2 ** attempt), 60)
                    wait += random.uniform(0, 2)          # jitter — de-sync threads
                    print(f"    [DATABRICKS RATE LIMIT] waiting {wait:.0f}s "
                          f"(attempt {attempt+1}/{attempts})")
                    time.sleep(wait)
                    continue

                if r.status_code in (500, 502, 503, 504):
                    time.sleep(min(3 * (2 ** attempt), 45) + random.uniform(0, 2))
                    continue

                # Non-retryable error
                r.raise_for_status()

            except requests.Timeout:
                print(f"    [DATABRICKS TIMEOUT] attempt {attempt+1} — endpoint: {self.endpoint}")
                time.sleep(2 + random.uniform(0, 2))
            except requests.RequestException as e:
                raise ConnectionError(
                    f"Databricks API call failed: {e}\n"
                    f"  Endpoint: {self.url}\n"
                    f"  Check DATABRICKS_HOST and DATABRICKS_TOKEN in config.py"
                ) from e

        raise ConnectionError(
            f"Databricks API failed after {attempts} attempts.\n"
            f"  Endpoint: {self.endpoint}\n"
            f"  Check your workspace URL and token."
        )


class DatabricksPrescreenClient(DatabricksModelClient):
    """
    Lightweight client for Stage 1C prescreener.
    Uses a smaller/cheaper model (8B or 20B) for fast yes/no decisions.
    """
    def __init__(self, **kwargs):
        kwargs.setdefault("endpoint", config.DATABRICKS_PRESCREENER_ENDPOINT)
        super().__init__(**kwargs)


def make_databricks_classifier() -> "DatabricksModelClient":
    """Factory — returns classifier client using config settings."""
    return DatabricksModelClient(endpoint=config.DATABRICKS_CLASSIFIER_ENDPOINT)


def make_databricks_prescreener() -> "DatabricksPrescreenClient":
    """Factory — returns prescreener client using config settings."""
    return DatabricksPrescreenClient()
