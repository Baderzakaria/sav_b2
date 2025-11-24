

import csv
import re
import json
import os
import time
from pathlib import Path
from typing import TypedDict, Dict, Any, Optional, List

import mlflow
import mlflow.langchain
from langgraph.graph import StateGraph, START, END
from utils_runtime import call_native_generate
from judges.mlflow_judge import run_judge_on_log

from freemind_env import load_environment

load_environment()

INPUT_CSV = os.path.join("data", "free tweet export.csv")
OUTPUT_CSV: Optional[str] = None
LOG_CSV: Optional[str] = None
RESULTS_DIR = os.path.join("data", "results")
MAX_ROWS: int = 100
MODELS: List[str] = ["llama3.2:3b"]
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "freemind_quick_check")

class State(TypedDict):
    row: Dict[str, Any]
    model_name: str
    text_value: str
    prompt_template: Optional[str]
    result: str

def _extract_single_value(raw: str) -> str:
    s = (raw or "").strip()
    if not s:
        return ""
    try:
        data = json.loads(s)
        if isinstance(data, dict):
            keys = list(data.keys())
            if len(keys) == 1:
                return str(data[keys[0]]).strip()

            for k in [
                "issue",
                "status",
                "utile",
                "categorie",
                "sentiment",
                "type_probleme",
                "score_gravite",
            ]:
                if k in data:
                    return str(data.get(k, "")).strip()
            return ""
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, (str, int, float, bool)):
                return str(first)
            if isinstance(first, dict):
                for k in ["value", "label", "status", "issue"]:
                    if k in first:
                        return str(first.get(k, "")).strip()
            return ""

        return str(data).strip()
    except Exception:
        pass
    m = re.search(r"(problem|not_problem|unclear)", s, re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return s

def _map_to_utile(value: str) -> str:
    v = (value or "").strip().lower()
    if v in {"problem", "probleme", "issue", "incident", "panne"}:
        return "true"
    if v in {"not_problem", "false", "no_issue", "pas_de_probleme", "non"}:
        return "false"
    if v in {"true", "false"}:
        return v
    return ""

def get_text(row: Dict[str, Any]) -> str:
    return row.get("full_text", " ".join(str(v) for v in row.values() if v))

def load_prompt_template(agent_key: str = "A1_utile") -> Optional[str]:
    try:
        with open(os.path.join("prompts", "freemind_prompts.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
            template = (
                cfg.get("agents", {})
                .get(agent_key, {})
                .get("prompt_template")
            )
            return template.strip() if isinstance(template, str) and template.strip() else None
    except Exception:
        return None

def worker_node(state: State) -> Dict[str, Any]:
    model_name = state["model_name"]
    text_value = state["text_value"]
    prompt_template = state.get("prompt_template")

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

    if isinstance(prompt_template, str) and prompt_template.strip():
        prompt = prompt_template.replace("{{full_text}}", text_value)
    else:
        prompt = (
            "Determine if the following tweet reports an issue or problem that might require attention. "
            "Return only JSON with a single key '' and value 'problem', 'not_problem', or 'unclear'. "
            "Consider context such as complaints, incidents, service disruptions, or requests for help.\n\n"
            "Tweet:\n" + text_value
        )

    model_output = call_native_generate(ollama_host, model_name, prompt)
    return {"result": model_output}

def build_graph():
    g = StateGraph(State)
    g.add_node("worker", worker_node)
    g.add_edge(START, "worker")
    g.add_edge("worker", END)
    return g.compile()

def run_quick_check(
    input_csv: str,
    output_csv: Optional[str],
    log_csv: Optional[str],
    max_rows: int,
    models: List[str],
) -> List[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    fieldnames: list[str] = []
    with open(input_csv, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        for i, row in enumerate(reader):
            rows.append(row)
            if max_rows and i + 1 >= max_rows:
                break

    if not rows:
        return []

    base_name = os.path.splitext(os.path.basename(input_csv))[0]
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    mlflow.langchain.autolog()
    run_name = f"quick_check_{base_name}_{int(time.time())}"

    graph = build_graph()
    prompt_template = load_prompt_template("A1_utile")

    augmented_rows: list[Dict[str, Any]] = [dict(r) for r in rows]
    log_entries: list[Dict[str, Any]] = []
    first_model = models[0] if models else ""

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("input_csv", input_csv)
        mlflow.log_param("max_rows", max_rows)
        mlflow.log_param("models", ",".join(models))

        if prompt_template:
            mlflow.log_dict(
                {"agent": "A1_utile", "prompt_template": prompt_template},
                artifact_file="prompts/A1_utile.json",
            )

        for model_name in models:
            print(f"\n=== Model: {model_name} ===")
            total_start = time.time()
            for idx, row in enumerate(rows, start=1):
                row_start = time.time()
                text_value = get_text(row)
                out_state = graph.invoke({
                    "row": row,
                    "model_name": model_name,
                    "text_value": text_value,
                    "prompt_template": prompt_template,
                })
                row_time = time.time() - row_start
                extracted = _extract_single_value(out_state.get("result", ""))
                utile_value = _map_to_utile(extracted)
                if model_name == first_model and utile_value:
                    augmented_rows[idx - 1]["A1_utile"] = utile_value

                print(f"[{idx}] ({row_time:.2f}s) Tweet: {text_value[:1000]}{'...' if len(text_value) > 1000 else ''}")
                if model_name == first_model:
                    print(f"     Utile(A1): {utile_value}\n")
                else:
                    print()

                log_entries.append({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "model": model_name,
                    "row_index": idx,
                    "tweet_id": row.get("id"),
                    "screen_name": row.get("screen_name"),
                    "elapsed_sec": f"{row_time:.3f}",
                    "raw_output": out_state.get("result", ""),
                    "extracted_value": extracted,
                    "mapped_utile": utile_value if utile_value else "",
                    "prompt_used": "custom" if prompt_template else "default",
                    "text_preview": text_value[:280],
                })

            total_time = time.time() - total_start
            avg_time = total_time / len(rows)
            print(f"Total: {total_time:.2f}s | Average: {avg_time:.2f}s per row\n")
            mlflow.log_metric(f"{model_name}_total_latency_sec", total_time)
            mlflow.log_metric(f"{model_name}_avg_latency_sec", avg_time)

        mlflow.log_metric("tweets_processed", len(rows))

        if not output_csv:
            output_csv = os.path.join(RESULTS_DIR, f"{base_name}_results.csv")
        if not log_csv:
            log_csv = os.path.join(RESULTS_DIR, f"{base_name}_log.csv")

        out_dir = os.path.dirname(output_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        extra_cols = ["A1_utile"]
        final_headers = [*fieldnames]
        for col in extra_cols:
            if col not in final_headers:
                final_headers.append(col)

        with open(output_csv, "w", encoding="utf-8", newline="") as wf:
            writer = csv.DictWriter(wf, fieldnames=final_headers)
            writer.writeheader()
            for r in augmented_rows:
                writer.writerow(r)
        print(f"Saved CSV: {output_csv}")

        log_fields = [
            "timestamp",
            "model",
            "row_index",
            "tweet_id",
            "screen_name",
            "elapsed_sec",
            "raw_output",
            "extracted_value",
            "mapped_utile",
            "prompt_used",
            "text_preview",
        ]
        with open(log_csv, "w", encoding="utf-8", newline="") as lf:
            writer = csv.DictWriter(lf, fieldnames=log_fields)
            writer.writeheader()
            for entry in log_entries:
                writer.writerow(entry)
        print(f"Saved log CSV: {log_csv}")

        mlflow.log_param("output_csv", output_csv)
        mlflow.log_param("log_csv", log_csv)
        mlflow.log_artifact(output_csv, artifact_path="results")
        mlflow.log_artifact(log_csv, artifact_path="logs")

        judge_csv_path = run_judge_on_log(
            log_csv=Path(log_csv),
            output_csv=None,
            judge_key="J_utile_consistency",
            tracking_uri=MLFLOW_TRACKING_URI,
            experiment_name=MLFLOW_EXPERIMENT,
            start_run=False,
        )
        mlflow.log_param("judge_csv", str(judge_csv_path))

    return augmented_rows

def main() -> None:
    run_quick_check(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        log_csv=LOG_CSV,
        max_rows=MAX_ROWS,
        models=MODELS,
    )

if __name__ == "__main__":
    main()

