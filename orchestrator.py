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

INPUT_CSV = "data/cleaned/free_tweet_export-latest.csv"
RESULTS_DIR = "data/results"
MAX_ROWS = 3087
MODEL_NAME = "llama3.2:3b"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MLFLOW_EXPERIMENT_NAME = "FreeMind_Orchestrator"
PROMPT_FILE = Path("prompts/freemind_prompts.json")

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
    ollama_options: Dict[str, Any] = field(default_factory=dict)
    enable_model_warmup: bool = True
    warmup_prompt: str = "Analyse ce texte pour t'initialiser."
    warmup_repeat: int = 1
    enable_adaptive_mem_floor: bool = True
    adaptive_mem_floor_timeout_sec: float = 8.0
    adaptive_mem_floor_slack_mb: int = 1024
    adaptive_mem_floor_min_mb: int = 1024

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

def run_model_warmup(config: PipelineConfig) -> None:
    if not config.enable_model_warmup:
        return
    prompt = (config.warmup_prompt or "").strip() or "Ping de préchauffage."
    attempts = max(1, config.warmup_repeat)
    for idx in range(attempts):
        try:
            call_native_generate(
                config.ollama_host,
                config.model_name,
                prompt,
                options=config.ollama_options,
                timeout=30,
            )
        except Exception as exc:
            print(f"Warmup attempt {idx + 1}/{attempts} skipped: {exc}")
            break

def wait_for_gpu_metric(watcher: Optional[GPUWatcher], timeout: float) -> Optional[Dict[str, float]]:
    if watcher is None:
        return None
    deadline = time.time() + max(timeout, 0)
    metric = watcher.latest()
    while metric is None and time.time() < deadline:
        time.sleep(min(watcher.poll_interval_sec, 0.5))
        metric = watcher.latest()
    return metric

def adapt_gpu_mem_floor(watcher: Optional[GPUWatcher], config: PipelineConfig) -> None:
    if watcher is None or not config.enable_adaptive_mem_floor:
        return
    metric = wait_for_gpu_metric(watcher, config.adaptive_mem_floor_timeout_sec)
    if not metric:
        print("Adaptive GPU mem floor skipped: no metrics yet.")
        return
    mem_free = max(metric["mem_total_mb"] - metric["mem_used_mb"], 0)
    target_floor = max(config.adaptive_mem_floor_min_mb, mem_free - config.adaptive_mem_floor_slack_mb)
    watcher.mem_floor_mb = max(config.adaptive_mem_floor_min_mb, target_floor)
    print(
        f"[Watchdog] Adaptive memory floor set to {watcher.mem_floor_mb:.0f} MB "
        f"(baseline free ~{mem_free:.0f} MB)."
    )

def algorithmic_checker(
    utile: Any,
    categorie: Any,
    sentiment: Any,
    type_probleme: Any,
    score_gravite: Any,
) -> Dict[str, Any]:

    final = {
        "utile": utile,
        "categorie": categorie,
        "sentiment": sentiment,
        "type_probleme": type_probleme,
        "score_gravite": score_gravite,
        "status": "success",
    }

    has_problem_or_question = (
        categorie in ["probleme", "question"] or
        (type_probleme and type_probleme != "autre") or
        (isinstance(score_gravite, (int, float)) and score_gravite <= -3)
    )
    if has_problem_or_question and final["utile"] is not True:
        final["utile"] = True

    if final["utile"] is False:
        if isinstance(score_gravite, (int, float)):
            if score_gravite < -1:
                final["score_gravite"] = -1
            elif score_gravite > 1:
                final["score_gravite"] = 1
        elif score_gravite is None:
            final["score_gravite"] = 0

    negative_sentiments = {"colere", "frustration", "deception", "inquietude"}
    positive_sentiments = {"satisfaction", "enthousiasme"}
    neutral_sentiments = {"neutre"}

    if final.get("sentiment") in negative_sentiments:
        if isinstance(final.get("score_gravite"), (int, float)) and final["score_gravite"] > -4:
            final["score_gravite"] = max(-10, -4)
        elif final.get("score_gravite") is None:
            final["score_gravite"] = -5
    elif final.get("sentiment") in positive_sentiments:
        if isinstance(final.get("score_gravite"), (int, float)) and final["score_gravite"] < 4:
            final["score_gravite"] = min(10, 4)
        elif final.get("score_gravite") is None:
            final["score_gravite"] = 5
    elif final.get("sentiment") in neutral_sentiments and final.get("score_gravite") is None:
        final["score_gravite"] = 0

    if isinstance(final["score_gravite"], (int, float)):
        final["score_gravite"] = max(-10, min(10, int(final["score_gravite"])))

    if final.get("utile") is False:
        sg = final.get("score_gravite")
        final["score_gravite"] = 0 if sg is None else max(-1, min(1, int(sg)))

    valid_sentiments = [
        "colere", "frustration", "deception", "inquietude",
        "neutre", "satisfaction", "enthousiasme"
    ]
    if final["sentiment"] == "outrage_critique":
        final["sentiment"] = "colere"
    elif final["sentiment"] and final["sentiment"] not in valid_sentiments:

        pass

    valid_categories = ["probleme", "question", "retour_client"]
    if final["categorie"] and final["categorie"] not in valid_categories:

        pass

    valid_types = ["panne", "facturation", "abonnement", "resiliation", "information", "autre"]
    if final["type_probleme"] and final["type_probleme"] not in valid_types:

        pass

    return final

def extract_json_value(response: str, key: str = None, aliases: Optional[list[str]] = None) -> Any:
    if response is None:
        return None
    cleaned_response = response.strip()
    keys_to_try = [key] if key else []
    if aliases:
        keys_to_try.extend([alias for alias in aliases if alias and alias not in keys_to_try])

    try:
        data = json.loads(cleaned_response)
        if keys_to_try and isinstance(data, dict):
            for candidate in keys_to_try:
                if candidate in data:
                    return data.get(candidate)
        return data
    except json.JSONDecodeError:
        pass

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
    clean_text: str
    a1_result: str
    a2_result: str
    a3_result: str
    a4_result: str
    a5_result: str
    final_json: Dict[str, Any]

def build_pipeline(
    active_prompts: Dict[str, Dict[str, Any]],
    model_name: str,
    ollama_host: str,
    ollama_options: Optional[Dict[str, Any]] = None,
):
    def run_agent(state: State, agent_key: str, result_key: str):
        prompt_cfg = active_prompts[agent_key]
        clean_text = state.get("clean_text", "")
        prompt = prompt_cfg["prompt_template"].replace("{{full_text}}", clean_text).replace("{{clean_text}}", clean_text)

        with start_span(
            name=f"{agent_key}_agent",
            attributes={"agent_key": agent_key, "model_name": model_name},
        ) as span:
            span.set_inputs({"prompt": prompt, "model": model_name})
            response = call_native_generate(
                ollama_host,
                model_name,
                prompt,
                options=ollama_options,
            )
            span.set_outputs({"raw_response": response})
        return {result_key: response}

    def node_a1(state): return run_agent(state, "A1_utile", "a1_result")
    def node_a2(state): return run_agent(state, "A2_categorie", "a2_result")
    def node_a3(state): return run_agent(state, "A3_sentiment", "a3_result")
    def node_a4(state): return run_agent(state, "A4_type_probleme", "a4_result")
    def node_a5(state): return run_agent(state, "A5_gravite", "a5_result")

    def node_a6_checker(state):

        a1_raw = state.get("a1_result", "")
        a2_raw = state.get("a2_result", "")
        a3_raw = state.get("a3_result", "")
        a4_raw = state.get("a4_result", "")
        a5_raw = state.get("a5_result", "")

        a1_parsed = extract_json_value(a1_raw, "utile", aliases=["useful"])
        a2_parsed = extract_json_value(a2_raw, "categorie")
        a3_parsed = extract_json_value(a3_raw, "sentiment")
        a4_parsed = extract_json_value(a4_raw, "type_probleme")
        a5_parsed = extract_json_value(a5_raw, "score_gravite")

        with start_span(
            name="A6_checker",
            attributes={"agent_key": "A6_checker", "model_name": "algorithmic"},
        ) as span:
            span.set_inputs({
                "a1_utile": a1_parsed,
                "a2_categorie": a2_parsed,
                "a3_sentiment": a3_parsed,
                "a4_type_probleme": a4_parsed,
                "a5_score_gravite": a5_parsed,
            })

            final_json = algorithmic_checker(
                utile=a1_parsed,
                categorie=a2_parsed,
                sentiment=a3_parsed,
                type_probleme=a4_parsed,
                score_gravite=a5_parsed,
            )

            span.set_outputs({"final_json": final_json})

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
    log_csv_latest_path = os.path.join(config.results_dir, "freemind_log_latest.csv")
    metadata_json_path = os.path.join(config.results_dir, f"run_metadata_{timestamp}.json")
    metadata_json_latest_path = os.path.join(config.results_dir, "run_metadata_latest.json")
    live_log_path = os.path.join(live_dir, f"live_run_{timestamp}.jsonl")

    prompts = load_prompts(config.prompt_overrides)
    app = build_pipeline(prompts, config.model_name, config.ollama_host, config.ollama_options)

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

    if config.enable_model_warmup:
        run_model_warmup(config)
        adapt_gpu_mem_floor(watcher, config)

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
            clean_text = row.get("clean_text", "") or row.get("full_text", "") or str(row)
            row_start = time.time()

            latest_gpu = watcher.latest() if watcher else None
            if latest_gpu and (latest_gpu["mem_total_mb"] - latest_gpu["mem_used_mb"]) < config.gpu_mem_floor_mb:
                print(f"[Watchdog] Pausing before row {i} due to low free memory.")
                time.sleep(config.watchdog_backoff_sec)
                latest_gpu = watcher.latest() if watcher else latest_gpu

            print(f"\n[{i}/{len(rows)}] Tweet ID={row.get('id')}")
            print(f"  Clean Text: {clean_text[:500]}{'...' if len(clean_text) > 500 else ''}")

            inputs = {"clean_text": clean_text}
            with start_span(
                name="tweet_pipeline",
                attributes={"tweet_id": row.get("id"), "row_index": i},
            ) as pipeline_span:
                pipeline_span.set_inputs({"clean_text": clean_text})
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
                "full_text": row.get("full_text", ""),
                "date_iso": row.get("date_iso"),
                "clean_text": clean_text,
                "favorite_count": row.get("favorite_count"),
                "reply_count": row.get("reply_count"),
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

            log_entries.append(entry)
            if config.enable_live_log:
                append_jsonl(live_log_path, entry)
            if progress_callback:
                progress_callback(entry)

    total_duration = time.time() - start_pipeline

    if log_entries:
        keys = log_entries[0].keys()
        for output_path in (log_csv_path, log_csv_latest_path):
            with open(output_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(log_entries)
        print(f"Done! Log saved to {log_csv_path} and {log_csv_latest_path}")

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
        for output_path in (metadata_json_path, metadata_json_latest_path):
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_json_path} and {metadata_json_latest_path}")
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
