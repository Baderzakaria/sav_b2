import csv
import json
import os
import time
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, TypedDict, Any, Dict, List, Optional

import mlflow
from langgraph.graph import StateGraph, START, END
from mlflow.tracing.fluent import start_span

from monitoring.gpu_watchdog import GPUWatcher, capture_nvidia_smi
from utils_runtime import call_native_generate

# --- Configuration Defaults ---
INPUT_CSV = "data/cleaned/free_tweet_export-latest.csv"
RESULTS_DIR = "data/results"
MAX_ROWS = 10
MODEL_NAME = "llama3.2:1b"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MLFLOW_EXPERIMENT_NAME = "FreeMind_Orchestrator"
PROMPT_FILE = Path("prompts/freemind_prompts.json")


# --- Data Classes ---
@dataclass
class PipelineConfig:
    input_csv: str = INPUT_CSV
    results_dir: str = RESULTS_DIR
    max_rows: int = MAX_ROWS
    model_name: str = MODEL_NAME
    ollama_host: str = OLLAMA_HOST
    mlflow_experiment: str = MLFLOW_EXPERIMENT_NAME
    run_name: Optional[str] = None
    prompt_overrides: Optional[Dict[str, str]] = None
    enable_live_log: bool = True
    gpu_mem_floor_mb: int = 2048
    gpu_temp_ceiling_c: int = 80
    gpu_watch_workers: int = 4
    gpu_poll_interval_sec: float = 2.0
    watchdog_backoff_sec: float = 5.0
    metadata_tags: Dict[str, Any] = field(default_factory=dict)
    live_log_dir: Optional[str] = None


def ensure_results_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_prompts(overrides: Optional[Dict[str, str]] = None) -> Dict[str, Dict[str, Any]]:
    with PROMPT_FILE.open("r", encoding="utf-8") as f:
        data = json.load(f)
    prompts = data.get("agents", {})
    if overrides:
        for key, template in overrides.items():
            if key in prompts and template:
                prompts[key]["prompt_template"] = template
    return prompts

# --- Helper: Clean JSON Extraction ---
def extract_json_value(response: str, key: str = None, aliases: Optional[list[str]] = None) -> Any:
    """Extract a key (with optional aliases) from a JSON-like response."""
    if response is None:
        return None
    cleaned_response = response.strip()
    keys_to_try = [key] if key else []
    if aliases:
        keys_to_try.extend([alias for alias in aliases if alias and alias not in keys_to_try])

    # 1. Try pure JSON parsing (fastest)
    try:
        data = json.loads(cleaned_response)
        if keys_to_try and isinstance(data, dict):
            for candidate in keys_to_try:
                if candidate in data:
                    return data.get(candidate)
        return data
    except json.JSONDecodeError:
        pass

    # 2. Try finding JSON blob inside text ```json ... ``` or { ... }
    try:
        match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
        if match:
            json_str = match.group(0)
            data = json.loads(json_str)
            if keys_to_try and isinstance(data, dict):
                for candidate in keys_to_try:
                    if candidate in data:
                        return data.get(candidate)
            return data
    except:
        pass

    # 3. Regex fallback for each candidate key
    for candidate in keys_to_try:
        pattern = f'"{candidate}"\\s*:\\s*("([^"]*)"|(\\d+|true|false|null))'
        match = re.search(pattern, cleaned_response, re.IGNORECASE)
        if match:
            val_str = match.group(2) or match.group(3)
            if isinstance(val_str, str):
                lowered = val_str.lower()
                if lowered == "true": return True
                if lowered == "false": return False
                if lowered == "null": return None
            if isinstance(val_str, str) and val_str.isdigit():
                return int(val_str)
            return val_str

    return None

class State(TypedDict, total=False):
    full_text: str
    a1_result: str
    a2_result: str
    a3_result: str
    a4_result: str
    a5_result: str
    final_json: Dict[str, Any]


def build_pipeline(active_prompts: Dict[str, Dict[str, Any]], model_name: str, ollama_host: str):
    def run_agent(state: State, agent_key: str, result_key: str):
        prompt_cfg = active_prompts[agent_key]
        prompt = prompt_cfg["prompt_template"].replace("{{full_text}}", state["full_text"])

        with start_span(
            name=f"{agent_key}_agent",
            attributes={"agent_key": agent_key, "model_name": model_name},
        ) as span:
            span.set_inputs({"prompt": prompt, "model": model_name})
            response = call_native_generate(ollama_host, model_name, prompt)
            span.set_outputs({"raw_response": response})
        return {result_key: response}

    def node_a1(state): return run_agent(state, "A1_utile", "a1_result")
    def node_a2(state): return run_agent(state, "A2_categorie", "a2_result")
    def node_a3(state): return run_agent(state, "A3_sentiment", "a3_result")
    def node_a4(state): return run_agent(state, "A4_type_probleme", "a4_result")
    def node_a5(state): return run_agent(state, "A5_gravite", "a5_result")

    def node_a6_checker(state):
        results = {
            "A1": state.get("a1_result"),
            "A2": state.get("a2_result"),
            "A3": state.get("a3_result"),
            "A4": state.get("a4_result"),
            "A5": state.get("a5_result"),
        }
        results_str = json.dumps(results, indent=2)

        prompt_cfg = active_prompts["A6_checker"]
        prompt = prompt_cfg["prompt_template"].replace("{{agent_results}}", results_str)

        with start_span(
            name="A6_checker",
            attributes={"agent_key": "A6_checker", "model_name": model_name},
        ) as span:
            span.set_inputs({"prompt": prompt, "aggregated_results": results})
            response = call_native_generate(ollama_host, model_name, prompt)
            span.set_outputs({"checker_response": response})

        final_json = extract_json_value(response)
        if not isinstance(final_json, dict):
            final_json = {"status": "error", "raw_response": response}
        else:
            if "utile" not in final_json and "useful" in final_json:
                final_json["utile"] = final_json.get("useful")

        return {"final_json": final_json}

    workflow = StateGraph(State)
    workflow.add_node("A1", node_a1)
    workflow.add_node("A2", node_a2)
    workflow.add_node("A3", node_a3)
    workflow.add_node("A4", node_a4)
    workflow.add_node("A5", node_a5)
    workflow.add_node("A6", node_a6_checker)

    workflow.add_edge(START, "A1")
    workflow.add_edge(START, "A2")
    workflow.add_edge(START, "A3")
    workflow.add_edge(START, "A4")
    workflow.add_edge(START, "A5")

    workflow.add_edge("A1", "A6")
    workflow.add_edge("A2", "A6")
    workflow.add_edge("A3", "A6")
    workflow.add_edge("A4", "A6")
    workflow.add_edge("A5", "A6")
    workflow.add_edge("A6", END)
    return workflow.compile()

def append_jsonl(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as fp:
        fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_pipeline(
    config: PipelineConfig,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_signal: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    ensure_results_dir(config.results_dir)
    live_dir = config.live_log_dir or config.results_dir

    timestamp = int(time.time())
    run_name = config.run_name or f"run_{timestamp}"
    log_csv_path = os.path.join(config.results_dir, f"freemind_log_{timestamp}.csv")
    metadata_json_path = os.path.join(config.results_dir, f"run_metadata_{timestamp}.json")
    live_log_path = os.path.join(live_dir, f"live_run_{timestamp}.jsonl")

    prompts = load_prompts(config.prompt_overrides)
    app = build_pipeline(prompts, config.model_name, config.ollama_host)

    mlflow.set_experiment(config.mlflow_experiment)
    mlflow.langchain.autolog()

    rows: List[Dict[str, Any]] = []
    if not os.path.exists(config.input_csv):
        raise FileNotFoundError(
            f"Input CSV file not found: {config.input_csv}. "
            "Run clean_free_tweets.py first or point PipelineConfig.input_csv to an existing file."
        )
    
    with open(config.input_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i >= config.max_rows:
                break
            rows.append(r)
    
    if not rows:
        raise ValueError(f"No rows found in {config.input_csv} or file is empty")

    watcher: Optional[GPUWatcher] = None
    watchdog_alerts: List[Dict[str, Any]] = []

    def _watchdog_alert(kind: str, metric) -> None:
        alert = {
            "kind": kind,
            "timestamp": metric.timestamp,
            "gpu_util": metric.gpu_util,
            "mem_used_mb": metric.mem_used_mb,
            "mem_total_mb": metric.mem_total_mb,
            "temperature": metric.temperature,
        }
        watchdog_alerts.append(alert)
        print(f"[Watchdog] {kind} at {time.strftime('%X', time.localtime(metric.timestamp))}: {alert}")

    try:
        watcher = GPUWatcher(
            mem_floor_mb=config.gpu_mem_floor_mb,
            temp_ceiling_c=config.gpu_temp_ceiling_c,
            num_workers=config.gpu_watch_workers,
            poll_interval_sec=config.gpu_poll_interval_sec,
        )
        watcher.start(_watchdog_alert)
    except Exception as exc:
        print(f"GPU watcher not started: {exc}")
        watcher = None

    log_entries: List[Dict[str, Any]] = []
    start_pipeline = time.time()
    print(f"=== FreeMind Orchestrator {run_name} ===")
    print(f"Input file: {config.input_csv} | Rows planned: {len(rows)} | Model: {config.model_name}")

    mlflow_tags = {"run_name": run_name, **config.metadata_tags}

    with mlflow.start_run(run_name=f"pipeline_{timestamp}", tags=mlflow_tags):
        mlflow.set_tags({"gpu_mem_floor_mb": config.gpu_mem_floor_mb})
        stopped_early = False
        for i, row in enumerate(rows, 1):
            if stop_signal and stop_signal():
                print("Stop signal received. Ending run early.")
                stopped_early = True
                break
            full_text = row.get("full_text", "") or str(row)
            row_start = time.time()

            latest_gpu = watcher.latest() if watcher else None
            if latest_gpu and (latest_gpu["mem_total_mb"] - latest_gpu["mem_used_mb"]) < config.gpu_mem_floor_mb:
                print(f"[Watchdog] Pausing before row {i} due to low free memory.")
                time.sleep(config.watchdog_backoff_sec)
                latest_gpu = watcher.latest() if watcher else latest_gpu

            print(f"\n[{i}/{len(rows)}] Tweet ID={row.get('id')}")
            print(f"  Text: {full_text[:500]}{'...' if len(full_text) > 500 else ''}")

            inputs = {"full_text": full_text}
            with start_span(
                name="tweet_pipeline",
                attributes={"tweet_id": row.get("id"), "row_index": i},
            ) as pipeline_span:
                pipeline_span.set_inputs({"full_text": full_text})
                output = app.invoke(inputs)
                final = output.get("final_json", {})
                pipeline_span.set_outputs(
                    {
                        "final_json": final,
                        "agents": {
                            "a1": output.get("a1_result"),
                            "a2": output.get("a2_result"),
                            "a3": output.get("a3_result"),
                            "a4": output.get("a4_result"),
                            "a5": output.get("a5_result"),
                        },
                    }
                )

            elapsed = time.time() - row_start

            a1_val = extract_json_value(output.get("a1_result"), "utile", aliases=["useful"])
            a2_val = extract_json_value(output.get("a2_result"), "categorie")
            a3_val = extract_json_value(output.get("a3_result"), "sentiment")
            a4_val = extract_json_value(output.get("a4_result"), "type_probleme")
            a5_val = extract_json_value(output.get("a5_result"), "score_gravite")

            print(f"  A1 utile      : {a1_val}")
            print(f"  A2 categorie  : {a2_val}")
            print(f"  A3 sentiment  : {a3_val}")
            print(f"  A4 type       : {a4_val}")
            print(f"  A5 gravité    : {a5_val}")
            print(f"  Checker status: {final.get('status', 'unknown')} | utile={final.get('utile')} | gravité={final.get('score_gravite')}")
            print(f"  Row latency   : {elapsed:.2f}s")

            mlflow.log_dict(final, f"results/tweet_{row.get('id', i)}.json")

            entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "row_index": i,
                "tweet_id": row.get("id"),
                "screen_name": row.get("screen_name"),
                "full_text": full_text,
                "elapsed_sec": round(elapsed, 2),
                "A1_utile": a1_val,
                "A2_categorie": a2_val,
                "A3_sentiment": a3_val,
                "A4_type": a4_val,
                "A5_gravity": a5_val,
                "Final_utile": final.get("utile"),
                "Final_categorie": final.get("categorie"),
                "Final_sentiment": final.get("sentiment"),
                "Final_gravity": final.get("score_gravite"),
                "status": final.get("status", "unclear"),
                "json_valid": "yes" if "status" in final else "no",
            }
            if latest_gpu:
                entry["gpu_util"] = latest_gpu["gpu_util"]
                entry["gpu_mem_used_mb"] = latest_gpu["mem_used_mb"]
                entry["gpu_mem_total_mb"] = latest_gpu["mem_total_mb"]
                entry["gpu_temp_c"] = latest_gpu["temperature"]

            log_entries.append(entry)
            if config.enable_live_log:
                append_jsonl(live_log_path, entry)
            if progress_callback:
                progress_callback(entry)

    total_duration = time.time() - start_pipeline

    if log_entries:
        keys = log_entries[0].keys()
        with open(log_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(log_entries)
        print(f"Done! Log saved to {log_csv_path}")

        metadata = {
            "run_id": run_name,
            "timestamp": timestamp,
            "input_file": config.input_csv,
            "rows_processed": len(log_entries),
            "total_duration_sec": round(total_duration, 2),
            "avg_sec_per_row": round(total_duration / len(log_entries), 2) if log_entries else 0,
            "model": config.model_name,
            "experiment": config.mlflow_experiment,
            "output_files": {"log_csv": log_csv_path, "results_dir": config.results_dir},
            "watchdog_alerts": watchdog_alerts,
            "nvidia_smi": capture_nvidia_smi(),
            "metadata_tags": config.metadata_tags,
            "stopped_early": stopped_early,
        }
        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_json_path}")
    else:
        print("No rows processed; nothing to log.")

    if watcher:
        history = watcher.history()
        watcher.stop()
    else:
        history = []

    return {
        "log_csv": log_csv_path,
        "metadata": metadata_json_path,
        "live_log": live_log_path if config.enable_live_log else None,
        "watchdog_alerts": watchdog_alerts,
        "gpu_history": history,
        "stopped_early": locals().get("stopped_early", False),
    }


def main():
    config = PipelineConfig()
    run_pipeline(config)


if __name__ == "__main__":
    main()
