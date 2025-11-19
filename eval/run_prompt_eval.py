#!/usr/bin/env python3
"""
Offline prompt evaluation that works with current environment:
- Attempts to register the A1 prompt in the MLflow Prompt Registry (if available)
- Generates predictions using the A1 prompt against the first N tweets
- Computes simple metrics (heuristic ground truth vs model outputs)
- Logs everything to a single MLflow run with artifacts
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow

from utils_runtime import call_native_generate

PROMPT_FILE = Path("prompts/freemind_prompts.json")
DEFAULT_INPUT = Path("data/free tweet export.csv")
RESULTS_DIR = Path("data/results")
ISSUE_REGEX = re.compile(r"(panne|proble|incident|coupure|rupture|lent|bug|erreur|service)", re.IGNORECASE)


def load_prompt_template(agent_key: str) -> Optional[str]:
    try:
        cfg = json.loads(PROMPT_FILE.read_text(encoding="utf-8"))
        return (
            cfg.get("agents", {})
            .get(agent_key, {})
            .get("prompt_template")
        )
    except Exception:
        return None


def register_prompt_if_possible(name: str, template: str) -> Optional[str]:
    try:
        from mlflow.genai import register_prompt  # type: ignore[attr-defined]
    except Exception:
        register_prompt = None  # type: ignore
    if register_prompt is None:
        return None
    try:
        pr = register_prompt(name=name, template=template, commit_message="Eval import")
        return f"prompts:/{getattr(pr, 'name', name)}@{getattr(pr, 'version', 'latest')}"
    except Exception:
        return None


def extract_value(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            for k in ["status", "issue", "utile"]:
                if k in data:
                    return str(data[k]).strip()
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, str):
                return first.strip()
    except Exception:
        pass
    m = re.search(r"(problem|not_problem|unclear|true|false)", s, re.IGNORECASE)
    return (m.group(1).lower() if m else s[:64])


def is_issue(text: str) -> bool:
    return bool(ISSUE_REGEX.search(text or ""))


def score(pred: str, text: str) -> Tuple[int, int, int, int]:
    y = is_issue(text)
    yhat = pred in {"problem", "true"}
    tp = int(yhat and y)
    tn = int((not yhat) and (not y))
    fp = int(yhat and (not y))
    fn = int((not yhat) and y)
    return tp, tn, fp, fn


def run_eval(
    input_csv: Path,
    max_rows: int,
    model_name: str,
    agent_key: str = "A1_utile",
    tracking_uri: str = "file:./mlruns",
    experiment: str = "freemind_prompt_eval",
) -> None:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    # Data
    rows: List[Dict[str, Any]] = []
    with input_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            rows.append(r)
            if max_rows and i + 1 >= max_rows:
                break

    if not rows:
        print("No rows to evaluate.")
        return

    # Prompt
    template = load_prompt_template(agent_key) or "Decide issue: Return JSON {\"status\": \"problem|not_problem|unclear\"}\nTweet:\n{{full_text}}"
    registry_uri = register_prompt_if_possible(agent_key, template)

    # Run
    run_name = f"prompt_eval_{agent_key}_{int(time.time())}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("input_csv", str(input_csv))
        mlflow.log_param("max_rows", max_rows)
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("agent_key", agent_key)
        if registry_uri:
            mlflow.log_param("prompt_uri", registry_uri)
        mlflow.log_dict({"name": agent_key, "template": template}, artifact_file=f"prompts/{agent_key}.json")

        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        eval_records: List[Dict[str, Any]] = []

        tp = tn = fp = fn = 0
        for idx, r in enumerate(rows, 1):
            text = r.get("full_text", "")
            prompt = template.replace("{{full_text}}", text)
            raw = call_native_generate(ollama_host, model_name, prompt)
            extracted = extract_value(raw)
            a, b, c, d = score(extracted, text)
            tp += a
            tn += b
            fp += c
            fn += d
            eval_records.append({
                "row_index": idx,
                "tweet_id": r.get("id"),
                "screen_name": r.get("screen_name"),
                "full_text": text,
                "raw_output": raw,
                "prediction": extracted,
                "heuristic_is_issue": is_issue(text),
            })

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("tp", tp)
        mlflow.log_metric("tn", tn)
        mlflow.log_metric("fp", fp)
        mlflow.log_metric("fn", fn)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_csv = RESULTS_DIR / f"prompt_eval_{agent_key}.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as wf:
            writer = csv.DictWriter(wf, fieldnames=list(eval_records[0].keys()))
            writer.writeheader()
            writer.writerows(eval_records)
        mlflow.log_artifact(str(out_csv), artifact_path="eval")
        print(f"Saved eval CSV: {out_csv}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline prompt evaluation (env-compatible).")
    p.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--max-rows", type=int, default=100)
    p.add_argument("--model-name", type=str, default="llama3.2:3b")
    p.add_argument("--agent-key", type=str, default="A1_utile")
    p.add_argument("--tracking-uri", type=str, default="file:./mlruns")
    p.add_argument("--experiment", type=str, default="freemind_prompt_eval")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_eval(
        input_csv=args.input_csv,
        max_rows=args.max_rows,
        model_name=args.model_name,
        agent_key=args.agent_key,
        tracking_uri=args.tracking_uri,
        experiment=args.experiment,
    )


