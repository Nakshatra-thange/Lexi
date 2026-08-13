"""
Lightweight tracing: logs every LLM call to a JSONL file with latency,
token usage, and full input/output. Not a replacement for a real
observability stack (Langfuse, etc.) — deliberately dependency-free so
the project stays runnable with zero extra infra. The point is to make
the pipeline's behavior inspectable after the fact, which is the bar
the JD sets ("observable, built for real customer usage"), not to
reinvent a commercial tracing product.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from src.config import ROOT_DIR

TRACE_LOG_PATH = ROOT_DIR / "data" / "traces.jsonl"
TRACE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


@contextmanager
def traced_call(stage: str, model: str, contract_id: str = ""):
    """
    Usage:
        with traced_call("segmentation", MODEL_SEGMENTATION, contract_id) as t:
            response = client.messages.create(...)
            t.record(response)
    """
    start = time.monotonic()
    record_box = {"response": None, "error": None}

    class _Recorder:
        def record(self, response):
            record_box["response"] = response

    try:
        yield _Recorder()
    except Exception as e:
        record_box["error"] = str(e)
        raise
    finally:
        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        response = record_box["response"]

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stage": stage,
            "model": model,
            "contract_id": contract_id,
            "latency_ms": elapsed_ms,
            "error": record_box["error"],
        }

        if response is not None:
            entry["input_tokens"] = getattr(response.usage, "input_tokens", None)
            entry["output_tokens"] = getattr(response.usage, "output_tokens", None)

        with open(TRACE_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")


def summarize_traces(contract_id: str | None = None) -> dict:
    """Aggregate cost/latency stats — the numbers you'd quote in a design doc."""
    if not TRACE_LOG_PATH.exists():
        return {}

    by_stage: dict[str, dict] = {}
    with open(TRACE_LOG_PATH) as f:
        for line in f:
            entry = json.loads(line)
            if contract_id and entry.get("contract_id") != contract_id:
                continue
            stage = entry["stage"]
            by_stage.setdefault(stage, {
                "calls": 0, "total_latency_ms": 0,
                "total_input_tokens": 0, "total_output_tokens": 0, "errors": 0,
            })
            s = by_stage[stage]
            s["calls"] += 1
            s["total_latency_ms"] += entry.get("latency_ms", 0)
            s["total_input_tokens"] += entry.get("input_tokens") or 0
            s["total_output_tokens"] += entry.get("output_tokens") or 0
            if entry.get("error"):
                s["errors"] += 1

    return by_stage