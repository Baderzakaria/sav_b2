import json
import logging
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

import mlflow
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit.runtime.scriptrunner import (
    RerunData,
    add_script_run_ctx,
    get_script_run_ctx,
)

from monitoring.gpu_watchdog import capture_nvidia_smi, GPUWatcher
from orchestrator import (
    PipelineConfig,
    load_prompts,
    run_pipeline,
    PROMPT_FILE,
    INPUT_CSV,
    RESULTS_DIR,
    MODEL_NAME,
)

SESSION_LOCK = threading.Lock()
SMI_HISTORY_LIMIT = 120
LOGGER = logging.getLogger(__name__)

def get_ollama_models() -> List[str]:

    cache_key = "cached_ollama_models"
    if cache_key in st.session_state:
        return st.session_state[cache_key]

    try:
        result = subprocess.run(
            ["ollama", "ls"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False
        )
        if result.returncode != 0:
            return []

        lines = result.stdout.strip().split("\n")
        models = []

        if len(lines) <= 1:

            return []

        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if parts:
                model_name = parts[0]

                if model_name:
                    models.append(model_name)

        models = sorted(set(models))

        st.session_state[cache_key] = models
        return models

    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        LOGGER.debug(f"Failed to fetch Ollama models: {e}")
        return []
    except Exception as e:
        LOGGER.debug(f"Unexpected error fetching Ollama models: {e}")
        return []

def _request_streamlit_rerun(is_auto: bool = True) -> None:
    ctx = get_script_run_ctx(suppress_warning=True)
    if not ctx or not ctx.script_requests:
        return

    try:
        ctx.script_requests.request_rerun(
            RerunData(
                query_string=ctx.query_string,
                page_script_hash=ctx.page_script_hash,
                is_auto_rerun=is_auto,
            )
        )
    except Exception as exc:
        LOGGER.debug("Auto-rerun request failed: %s", exc)

def init_session_state() -> None:
    defaults = {
        "run_thread": None,
        "run_stop_event": threading.Event(),
        "live_entries": [],
        "last_run_result": None,
        "run_error": "",
        "dvc_logs": "",
        "config_input_csv": INPUT_CSV,
        "config_model": MODEL_NAME,
        "config_model_custom": MODEL_NAME,
        "config_max_rows": 50,
        "config_parallel_agents": True,
        "config_gpu_mem_floor": 2048,
        "config_gpu_temp": 80,
        "config_gpu_threads": 1,
        "config_gpu_poll": 2.0,
        "dvc_target_path": "data/results",
        "smi_history": [],
        "current_run_target_rows": 0,
        "gpu_watcher": None,
        "gpu_metrics": [],
        "pipeline_status": "idle",
        "live_log_path": None,
        "jsonl_read_position": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def start_gpu_watcher() -> None:
    if st.session_state.gpu_watcher is not None:

        if hasattr(st.session_state.gpu_watcher, '_threads'):
            threads_alive = any(t.is_alive() for t in st.session_state.gpu_watcher._threads)
            if threads_alive:
                return

        try:
            stop_gpu_watcher()
        except Exception:
            pass

    try:
        watcher = GPUWatcher(
            mem_floor_mb=st.session_state.config_gpu_mem_floor,
            temp_ceiling_c=st.session_state.config_gpu_temp,
            num_workers=st.session_state.config_gpu_threads,
            poll_interval_sec=st.session_state.config_gpu_poll,
        )

        def update_metrics(kind: str, metric):
            with SESSION_LOCK:
                entry = {
                    "timestamp": time.time(),
                    "gpu_util": metric.gpu_util,
                    "gpu_mem_used_mb": metric.mem_used_mb,
                    "gpu_mem_total_mb": metric.mem_total_mb,
                    "gpu_temp_c": metric.temperature,
                }
                if "gpu_metrics" not in st.session_state:
                    st.session_state.gpu_metrics = []
                st.session_state.gpu_metrics.append(entry)

                if len(st.session_state.gpu_metrics) > 300:
                    st.session_state.gpu_metrics.pop(0)

        watcher.start(update_metrics)
        st.session_state.gpu_watcher = watcher

    except Exception as exc:
        st.session_state.gpu_watcher = None

def stop_gpu_watcher() -> None:
    if st.session_state.gpu_watcher is not None:
        try:
            st.session_state.gpu_watcher.stop()
        except Exception:
            pass
        st.session_state.gpu_watcher = None

def update_local_prompt(agent_key: str, template: str, version_label: str) -> None:
    with PROMPT_FILE.open("r", encoding="utf-8") as fp:
        cfg = json.load(fp)
    agents = cfg.setdefault("agents", {})
    agent_entry = agents.setdefault(agent_key, {"name": agent_key})
    agent_entry["prompt_template"] = template
    agent_entry["version"] = version_label
    agents[agent_key] = agent_entry
    cfg["agents"] = agents
    with PROMPT_FILE.open("w", encoding="utf-8") as fp:
        json.dump(cfg, fp, indent=2, ensure_ascii=False)

def register_prompt_via_mlflow(agent_key: str, template: str) -> Dict[str, str]:
    try:
        from mlflow.genai import register_prompt
    except Exception as exc:
        return {"success": False, "message": f"MLflow Prompt Registry unavailable: {exc}"}

    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("freemind_prompt_registry")
    run_name = f"update_prompt_{agent_key}_{int(time.time())}"

    try:
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("agent_key", agent_key)
            result = register_prompt(
                name=agent_key,
                template=template,
                commit_message=f"Streamlit update for {agent_key}",
            )
            new_version = getattr(result, "version", None)
            version_label = f"v{new_version}" if new_version is not None else f"v{int(time.time())}"
            mlflow.log_param("new_version", version_label)
    except Exception as exc:
        return {"success": False, "message": f"Failed to register prompt: {exc}"}

    update_local_prompt(agent_key, template, version_label)
    return {"success": True, "message": f"Prompt registered as {version_label}.", "version": version_label}

def launch_pipeline(config: PipelineConfig) -> None:
    st.session_state.run_stop_event.clear()
    st.session_state.live_entries = []
    st.session_state.current_run_target_rows = int(config.max_rows)
    st.session_state.pipeline_status = "running"
    st.session_state.run_error = ""
    st.session_state.jsonl_read_position = 0

    import time
    timestamp = int(time.time())
    live_dir = config.live_log_dir or config.results_dir
    expected_live_log_path = os.path.join(live_dir, f"live_run_{timestamp}.jsonl")
    st.session_state.live_log_path = expected_live_log_path

    def progress(entry):
        with SESSION_LOCK:
            st.session_state.live_entries.append(entry)

            if st.session_state.gpu_watcher:
                latest = st.session_state.gpu_watcher.latest()
                if latest:
                    entry.update({
                        "gpu_util": latest.get("gpu_util"),
                        "gpu_mem_used_mb": latest.get("mem_used_mb"),
                        "gpu_mem_total_mb": latest.get("mem_total_mb"),
                        "gpu_temp_c": latest.get("temperature"),
                    })
        _request_streamlit_rerun()

    def should_stop():
        return st.session_state.run_stop_event.is_set()

    def runner():
        import traceback
        try:
            result = run_pipeline(config, progress_callback=progress, stop_signal=should_stop)
            with SESSION_LOCK:
                st.session_state.last_run_result = result
                st.session_state.run_error = ""
                st.session_state.pipeline_status = "completed"

                if result and "live_log" in result and result["live_log"]:
                    st.session_state.live_log_path = result["live_log"]
            _request_streamlit_rerun(is_auto=False)
        except Exception as exc:
            with SESSION_LOCK:
                st.session_state.run_error = traceback.format_exc()
                st.session_state.pipeline_status = "error"
            _request_streamlit_rerun(is_auto=False)
        finally:
            with SESSION_LOCK:
                st.session_state.run_thread = None
            _request_streamlit_rerun(is_auto=False)

    thread = threading.Thread(target=runner, daemon=True)
    st.session_state.run_thread = thread
    add_script_run_ctx(thread)
    thread.start()

def stop_pipeline():
    st.session_state.run_stop_event.set()

def run_dvc_command(args: List[str]) -> None:
    try:
        result = subprocess.run(args, capture_output=True, text=True, check=False)
        output = f"$ {' '.join(args)}\n{result.stdout}\n{result.stderr}"
    except FileNotFoundError:
        output = "DVC executable not found. Install DVC or adjust PATH."
    st.session_state.dvc_logs = output.strip()

def load_run_history(limit: int = 20) -> pd.DataFrame:
    records: List[Dict] = []
    metadata_dir = Path(RESULTS_DIR)
    for meta_path in sorted(metadata_dir.glob("run_metadata_*.json"), reverse=True)[:limit]:
        try:
            with meta_path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
                data["metadata_path"] = str(meta_path)
                records.append(data)
        except Exception:
            continue
    return pd.DataFrame(records)

def render_gpu_chart(entries: List[Dict]) -> None:
    df = pd.DataFrame(entries)
    if df.empty or "gpu_util" not in df.columns:
        st.info("No GPU samples yet.")
        return
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    color_map = {
        "gpu_util": "#1f77b4",
        "gpu_mem_used_mb": "#ff7f0e",
        "gpu_temp_c": "#d62728"
    }

    fig = px.line(
        df,
        x="timestamp",
        y=["gpu_util", "gpu_mem_used_mb", "gpu_temp_c"],
        labels={
            "value": "Value",
            "variable": "Metric",
            "timestamp": "Time",
        },
        title="GPU Utilization / Memory / Temperature",
        color_discrete_map=color_map
    )
    fig.update_layout(
        margin=dict(l=40, r=20, t=40, b=40),
        height=340,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            title=""
        ),
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
        }
    )

def render_history_chart(history_df: pd.DataFrame) -> None:
    if history_df.empty:
        st.info("No historical runs found.")
        return
    history_df = history_df.copy()
    history_df["datetime"] = pd.to_datetime(history_df["timestamp"], unit="s")
    fig = px.bar(
        history_df,
        x="datetime",
        y="rows_processed",
        color="avg_sec_per_row",
        labels={"rows_processed": "Rows", "avg_sec_per_row": "Avg sec/row"},
        title="Recent Pipeline Runs",
    )
    st.plotly_chart(fig, use_container_width=True)

def render_prompt_editor():
    st.subheader("Prompt Management")
    prompts = load_prompts()
    for agent_key, cfg in prompts.items():
        with st.expander(f"{agent_key}"):
            template = st.text_area(
                f"Template for {agent_key}",
                value=cfg.get("prompt_template", ""),
                height=200,
                key=f"prompt_{agent_key}",
            )
            st.caption(f"Current version: {cfg.get('version', 'unknown')}")
            if st.button(f"Save via MLflow · {agent_key}", key=f"save_{agent_key}"):
                result = register_prompt_via_mlflow(agent_key, template)
                if result.get("success"):
                    st.success(result["message"])
                else:
                    st.error(result["message"])

def render_dvc_section():
    st.subheader("DVC Actions")
    target = st.text_input("Target path", value=st.session_state.get("dvc_target_path", "data/results"), key="dvc_target_path_input")
    st.session_state["dvc_target_path"] = target
    dvc_ready = (Path(".") / ".dvc").exists()
    if not dvc_ready:
        st.warning("DVC repository not initialized. Run `dvc init` in the project root to enable these commands.")
    cols = st.columns(3)
    if cols[0].button("dvc add", use_container_width=True, disabled=not dvc_ready):
        run_dvc_command(["dvc", "add", st.session_state["dvc_target_path"]])
    if cols[1].button("dvc commit", use_container_width=True, disabled=not dvc_ready):
        run_dvc_command(["dvc", "commit", st.session_state["dvc_target_path"]])
    if cols[2].button("dvc push", use_container_width=True, disabled=not dvc_ready):
        run_dvc_command(["dvc", "push", st.session_state["dvc_target_path"]])
    if st.session_state.dvc_logs:
        st.code(st.session_state.dvc_logs, language="bash")

def read_jsonl_file(file_path: str, start_position: int = 0) -> tuple[List[Dict], int]:
    if not file_path or not os.path.exists(file_path):
        return [], start_position

    new_entries = []
    new_position = start_position

    try:
        with open(file_path, "r", encoding="utf-8") as f:

            if start_position > 0:
                f.seek(start_position)

            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    new_entries.append(entry)
                except json.JSONDecodeError:

                    continue

            new_position = f.tell()
    except (IOError, OSError) as e:
        LOGGER.debug(f"Error reading JSONL file {file_path}: {e}")
        return [], start_position

    return new_entries, new_position

def find_latest_jsonl_file() -> Optional[str]:
    results_dir = Path(RESULTS_DIR)
    if not results_dir.exists():
        return None

    jsonl_files = list(results_dir.glob("live_run_*.jsonl"))
    if not jsonl_files:
        return None

    latest = max(jsonl_files, key=lambda p: p.stat().st_mtime)
    return str(latest)

def get_live_entries_from_file() -> List[Dict]:
    live_entries = list(st.session_state.live_entries)
    live_log_path = st.session_state.get("live_log_path")

    if not live_log_path or not os.path.exists(live_log_path):
        latest_file = find_latest_jsonl_file()
        if latest_file:
            live_log_path = latest_file
            st.session_state.live_log_path = latest_file
            st.session_state.jsonl_read_position = 0

    if live_log_path and os.path.exists(live_log_path):

        read_pos = st.session_state.get("jsonl_read_position", 0)
        new_entries, new_position = read_jsonl_file(live_log_path, read_pos)

        if new_entries:

            st.session_state.jsonl_read_position = new_position

            existing_indices = {e.get("row_index") for e in live_entries if e.get("row_index")}
            for entry in new_entries:
                row_idx = entry.get("row_index")
                if row_idx and row_idx not in existing_indices:
                    live_entries.append(entry)
                    existing_indices.add(row_idx)
                elif not row_idx:

                    live_entries.append(entry)

            if all(e.get("row_index") for e in live_entries):
                live_entries.sort(key=lambda x: x.get("row_index", 0))

    return live_entries

def render_live_results():
    st.subheader("Live Run Output")
    running = st.session_state.run_thread is not None and st.session_state.run_thread.is_alive()

    live_entries = get_live_entries_from_file()

    if live_entries:
        st.session_state.live_entries = live_entries

    rows_done = len(live_entries)
    target_rows = max(st.session_state.get("current_run_target_rows", rows_done), 1)

    status = st.session_state.get("pipeline_status", "idle")
    status_colors = {
        "idle": "⚪",
        "running": "🟢",
        "completed": "✅",
        "error": "🔴"
    }
    status_text = {
        "idle": "Ready",
        "running": "Running",
        "completed": "Completed",
        "error": "Error"
    }

    live_log_path = st.session_state.get("live_log_path")
    status_display = f"Status: {status_colors.get(status, '⚪')} {status_text.get(status, 'Unknown')}"
    if live_log_path and os.path.exists(live_log_path):
        file_size = os.path.getsize(live_log_path)
        status_display += f" | 📄 Reading from: {os.path.basename(live_log_path)} ({file_size:,} bytes)"
    st.caption(status_display)

    if target_rows > 0:
        progress_value = min(rows_done / target_rows, 1.0)
        progress_text = f"{rows_done} / {target_rows} rows processed"
        if running:
            progress_text = f"🔄 {progress_text}"
        st.progress(progress_value, text=progress_text)
    else:
        st.progress(0.0, text="Not started")

    stream_container = st.container()
    with stream_container:
        if live_entries:
            st.caption("📊 Streaming feed (latest 50 rows)")
            stream_placeholder = st.empty()

            def stream_generator():
                for entry in live_entries[-50:]:
                    summary = {
                        "row": entry.get("row_index"),
                        "tweet_id": entry.get("tweet_id"),
                        "utile": entry.get("Final_utile"),
                        "categorie": entry.get("Final_categorie"),
                        "sentiment": entry.get("Final_sentiment"),
                        "gravity": entry.get("Final_gravity"),
                        "status": entry.get("status"),
                        "elapsed_sec": entry.get("elapsed_sec"),
                    }
                    yield json.dumps(summary, ensure_ascii=False) + "\n"

            stream_placeholder.write_stream(stream_generator)

            if len(live_entries) > 0:
                live_df = pd.DataFrame(live_entries)
                st.dataframe(live_df.tail(50), use_container_width=True, height=400)

                if any("gpu_util" in e for e in live_entries):
                    render_gpu_chart(live_entries)
        else:
            if running:
                st.info("⏳ Pipeline running… waiting for first row to be processed.")
            elif status == "error":
                st.error("❌ Pipeline encountered an error. Check error details below.")
            else:
                st.info("📋 No rows processed yet. Launch a run to start processing.")

def render_run_history():
    st.subheader("Run History")
    history_df = load_run_history()
    render_history_chart(history_df)
    if not history_df.empty:
        st.dataframe(history_df.head(10), use_container_width=True)

def latest_gpu_entry() -> Optional[Dict]:

    if st.session_state.gpu_watcher:
        try:
            latest = st.session_state.gpu_watcher.latest()
            if latest and latest.get("gpu_util") is not None:
                return {
                    "gpu_util": latest.get("gpu_util", 0),
                    "gpu_mem_used_mb": latest.get("mem_used_mb", 0),
                    "gpu_mem_total_mb": latest.get("mem_total_mb", 0),
                    "gpu_temp_c": latest.get("temperature", 0),
                }
        except Exception:
            pass

    if st.session_state.get("gpu_metrics"):
        metrics = st.session_state.gpu_metrics
        if metrics and len(metrics) > 0:
            latest = metrics[-1]
            if latest.get("gpu_util") is not None:
                return latest

    if st.session_state.get("smi_history"):
        history = st.session_state.smi_history
        if history and len(history) > 0:
            latest_smi = history[-1]

            if latest_smi.get("gpu_util") is not None or latest_smi.get("mem_used_mb") is not None:
                return {
                    "gpu_util": latest_smi.get("gpu_util", 0),
                    "gpu_mem_used_mb": latest_smi.get("mem_used_mb", 0),
                    "gpu_mem_total_mb": latest_smi.get("mem_total_mb", 0),
                    "gpu_temp_c": latest_smi.get("temperature", 0),
                }

    for entry in reversed(st.session_state.live_entries):
        if all(k in entry for k in ("gpu_util", "gpu_mem_used_mb", "gpu_temp_c")):
            if entry.get("gpu_util") is not None:
                return entry
    return None

def record_smi_snapshot(snapshot: str) -> None:
    if not snapshot or "NVIDIA-SMI" not in snapshot:
        return

    history = st.session_state.get("smi_history", [])

    mem_match = re.search(r"(\d+)\s*MiB\s*/\s*(\d+)\s*MiB", snapshot) or re.search(r"(\d+)\s*/\s*(\d+)\s*MiB", snapshot)

    util_match = re.search(r"(\d+)\s*%\s+Default", snapshot) or re.search(r"(\d+)\s*%", snapshot.split("\n")[0])

    temp_match = re.search(r"\|\s*(\d+)\s*C", snapshot) or re.search(r"(\d+)\s*C", snapshot)

    entry = {
        "timestamp": time.time(),
        "gpu_util": float(util_match.group(1)) if util_match else None,
        "mem_used_mb": float(mem_match.group(1)) if mem_match else None,
        "mem_total_mb": float(mem_match.group(2)) if mem_match else None,
        "temperature": float(temp_match.group(1)) if temp_match else None,
    }

    if any(v is not None for v in [entry["gpu_util"], entry["mem_used_mb"], entry["temperature"]]):
        history.append(entry)
        history = history[-SMI_HISTORY_LIMIT:]
        st.session_state["smi_history"] = history

def render_smi_history():
    history = st.session_state.get("smi_history", [])
    if not history:
        st.info("📊 nvidia-smi history will appear once metrics are captured.")
        return
    df = pd.DataFrame(history)

    for col in ["gpu_util", "mem_used_mb", "temperature"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if df["timestamp"].dtype == "float64" or df["timestamp"].dtype == "int64":
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    else:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_long = df.melt(
        id_vars=["timestamp"],
        value_vars=["gpu_util", "mem_used_mb", "temperature"],
        var_name="metric",
        value_name="value",
    ).dropna(subset=["value"])
    if df_long.empty:
        st.info("📊 nvidia-smi history will appear once metrics are captured.")
        return

    color_map = {
        "gpu_util": "#1f77b4",
        "mem_used_mb": "#ff7f0e",
        "temperature": "#d62728"
    }

    fig = px.line(
        df_long,
        x="timestamp",
        y="value",
        color="metric",
        labels={
            "value": "Value",
            "metric": "Metric",
            "timestamp": "Time",
        },
        title="",
        color_discrete_map=color_map
    )
    fig.update_layout(
        margin=dict(l=40, r=20, t=20, b=40),
        height=400,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.25,
            xanchor="center",
            x=0.5,
            title=""
        ),
        xaxis_title="Time",
        yaxis_title="Value",
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "displaylogo": False,
            "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
        }
    )

def render_run_controls():
    running = st.session_state.run_thread is not None and st.session_state.run_thread.is_alive()
    st.subheader("Run Controls")

    available_models = get_ollama_models()
    current_model = st.session_state.get("config_model", MODEL_NAME)

    if not available_models:
        available_models = ["llama3.2:3b", "mistral:latest", "llama3.1:8b"]
        if st.session_state.get("ollama_models_warning_shown", False) == False:
            st.warning("⚠️ Could not fetch models from Ollama. Using default presets. Make sure Ollama is running and 'ollama ls' works.")
            st.session_state.ollama_models_warning_shown = True

    if current_model in available_models:
        default_choice = current_model
    else:
        default_choice = available_models[0] if available_models else "Custom"

    model_col1, model_col2 = st.columns([4, 1])
    with model_col2:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True, help="Refresh model list from 'ollama ls'"):

            if "cached_ollama_models" in st.session_state:
                del st.session_state.cached_ollama_models
            st.rerun()

    with st.form("run_control_form"):
        st.text_input("Input CSV", key="config_input_csv", value=st.session_state.get("config_input_csv", INPUT_CSV))
        st.number_input(
            "Number of tweets",
            min_value=1,
            max_value=10000,
            key="config_max_rows",
        )

        model_options = available_models + ["Custom"]
        try:
            default_index = model_options.index(default_choice)
        except ValueError:
            default_index = 0

        model_choice = st.selectbox(
            "🤖 Ollama Model",
            options=model_options,
            index=default_index,
            key="config_model_choice",
            help="Select from available Ollama models (fetched from 'ollama ls') or choose Custom to enter a model name manually.",
        )

        if model_choice == "Custom":
            custom_default = current_model if current_model not in available_models else st.session_state.get("config_model_custom", "")
            st.text_input(
                "Custom model name",
                key="config_model_custom",
                value=custom_default,
                help="Enter any Ollama model name (e.g., 'llama3.2:3b', 'mistral:latest')",
            )
            st.session_state.config_model = st.session_state.config_model_custom
        else:
            st.session_state.config_model = model_choice

        if available_models:
            st.caption(f"📋 Found {len(available_models)} model(s) from Ollama")

        st.toggle("Parallel agents (5)", key="config_parallel_agents", help="Agents already execute in parallel; toggle for bookkeeping.")
        start_clicked = st.form_submit_button("Launch Run", disabled=running)

    if start_clicked:
        input_csv = (st.session_state.config_input_csv or "").strip() or INPUT_CSV
        max_rows = int(st.session_state.config_max_rows or MAX_ROWS)
        model_name = st.session_state.config_model or MODEL_NAME
        gpu_mem_floor = int(st.session_state.config_gpu_mem_floor or PipelineConfig.gpu_mem_floor_mb)
        gpu_temp = int(st.session_state.config_gpu_temp or PipelineConfig.gpu_temp_ceiling_c)
        gpu_threads = int(st.session_state.config_gpu_threads or PipelineConfig.gpu_watch_workers)
        gpu_poll = float(st.session_state.config_gpu_poll or PipelineConfig.gpu_poll_interval_sec)

        config = PipelineConfig(
            input_csv=input_csv,
            max_rows=max_rows,
            model_name=model_name,
            gpu_mem_floor_mb=gpu_mem_floor,
            gpu_temp_ceiling_c=gpu_temp,
            gpu_watch_workers=gpu_threads,
            gpu_poll_interval_sec=gpu_poll,
            metadata_tags={
                "parallel_agents": st.session_state.config_parallel_agents,
                "gpu_threads": gpu_threads,
            },
        )
        launch_pipeline(config)
        st.success("Run started.")

    if st.button("Launch Default (CLI) Run", disabled=running, use_container_width=True):
        launch_pipeline(PipelineConfig())
        st.success("Default run started (same as `python orchestrator.py`).")

    if st.button("Stop Run", disabled=not running, use_container_width=True):
        stop_pipeline()
        st.warning("Stop signal sent.")

    if st.session_state.get("run_error"):
        st.error("❌ Pipeline Error")
        with st.expander("Error Details", expanded=True):
            st.code(st.session_state["run_error"], language="python")

    if st.session_state.last_run_result and not running:
        with st.expander("Last Run Summary", expanded=False):
            result = st.session_state.last_run_result
            st.json({
                "log_csv": result.get("log_csv"),
                "metadata": result.get("metadata"),
                "watchdog_alerts": len(result.get("watchdog_alerts", [])),
                "gpu_history_samples": len(result.get("gpu_history", [])),
                "stopped_early": result.get("stopped_early", False),
            })

def render_gpu_panel():

    header_col1, header_col2 = st.columns([3, 1])
    with header_col1:
        st.markdown("### 🖥️ GPU & Watchdog")
    with header_col2:
        st.markdown("<br>", unsafe_allow_html=True)

    watcher_error = None
    if st.session_state.gpu_watcher is None:
        try:
            start_gpu_watcher()
        except Exception as exc:
            watcher_error = str(exc)

    status_container = st.container()
    with status_container:
        if watcher_error:
            st.error(f"⚠️ **GPU monitoring unavailable:** {watcher_error}")
            st.info("💡 Install pynvml: `pip install pynvml` or use nvidia-smi snapshot below")
        elif st.session_state.gpu_watcher:
            try:
                threads_alive = any(t.is_alive() for t in st.session_state.gpu_watcher._threads) if hasattr(st.session_state.gpu_watcher, '_threads') else False
                if threads_alive:
                    st.success("🟢 **GPU Watcher: Active**")
                else:
                    st.warning("🟡 **GPU Watcher: Starting...**")
            except Exception:
                st.success("🟢 **GPU Watcher: Active**")
        else:
            st.info("⚪ **GPU Watcher: Inactive**")

    st.markdown("---")

    latest = latest_gpu_entry()

    if latest:

        mem_total = latest.get('gpu_mem_total_mb', 1)
        mem_used = latest.get('gpu_mem_used_mb', 0)
        mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0

        metric_cols = st.columns(4)

        with metric_cols[0]:
            st.markdown("#### 📊 GPU Utilization")
            st.markdown(f"### {latest['gpu_util']:.0f}%")

            st.progress(latest['gpu_util'] / 100, text="")

        with metric_cols[1]:
            st.markdown("#### 💾 VRAM Usage")
            st.markdown(f"### {mem_used:.0f} / {mem_total:.0f} MB")
            st.progress(mem_percent / 100, text=f"{mem_percent:.1f}%")

        with metric_cols[2]:
            st.markdown("#### 🌡️ Temperature")
            temp = latest['gpu_temp_c']
            temp_color = "🟢" if temp < 70 else "🟡" if temp < 80 else "🔴"
            st.markdown(f"### {temp_color} {temp:.0f}°C")

            temp_progress = min(temp / 90, 1.0)
            st.progress(temp_progress, text="")

        with metric_cols[3]:
            st.markdown("#### 🔄 Free Memory")
            free_mem = mem_total - mem_used
            st.markdown(f"### {free_mem:.0f} MB")
            free_percent = (free_mem / mem_total * 100) if mem_total > 0 else 0
            st.progress(free_percent / 100, text=f"{free_percent:.1f}%")
    else:
        metric_cols = st.columns(4)
        with metric_cols[0]:
            st.markdown("#### 📊 GPU Utilization")
            st.markdown("### --")
        with metric_cols[1]:
            st.markdown("#### 💾 VRAM Usage")
            st.markdown("### --")
        with metric_cols[2]:
            st.markdown("#### 🌡️ Temperature")
            st.markdown("### --")
        with metric_cols[3]:
            st.markdown("#### 🔄 Free Memory")
            st.markdown("### --")
        if not watcher_error:
            st.info("⏳ Waiting for GPU metrics... (checking nvidia-smi)")

    st.markdown("---")

    tab_controls, tab_snapshot, tab_timeline = st.tabs(["⚙️ Configuration", "📋 NVIDIA-SMI", "📈 Timeline"])

    with tab_controls:
        st.markdown("#### Watchdog Settings")
        st.markdown("Configure GPU monitoring thresholds and polling behavior.")

        control_col1, control_col2 = st.columns(2)

        with control_col1:
            workers = st.slider(
                "🔧 GPU Multithreading (Watchdog Workers)",
                min_value=1,
                max_value=4,
                value=st.session_state.get("config_gpu_threads", 1),
                key="config_gpu_threads",
                help="Number of watcher threads sampling GPU metrics. More threads = more frequent sampling.",
            )

            mem_floor = st.slider(
                "💾 GPU Free Memory Floor (MB)",
                min_value=512,
                max_value=8192,
                step=256,
                value=st.session_state.get("config_gpu_mem_floor", 2048),
                key="config_gpu_mem_floor",
                help="Pipeline will pause if free GPU memory drops below this threshold.",
            )

        with control_col2:
            poll_interval = st.number_input(
                "⏱️ Watchdog Poll Interval (seconds)",
                min_value=0.5,
                max_value=10.0,
                step=0.5,
                value=st.session_state.get("config_gpu_poll", 2.0),
                key="config_gpu_poll",
                help="Lower values capture GPU spikes faster at the cost of CPU overhead.",
            )

            temp_ceiling = st.slider(
                "🌡️ GPU Temperature Ceiling (°C)",
                min_value=60,
                max_value=90,
                step=5,
                value=st.session_state.get("config_gpu_temp", 80),
                key="config_gpu_temp",
                help="Alert if GPU temperature exceeds this value.",
            )

        if st.session_state.gpu_watcher:
            try:
                watcher = st.session_state.gpu_watcher
                watcher.mem_floor_mb = mem_floor
                watcher.temp_ceiling_c = temp_ceiling
                watcher.poll_interval_sec = poll_interval
                watcher.num_workers = workers
            except Exception:
                pass

    with tab_snapshot:
        st.markdown("#### Current NVIDIA-SMI Snapshot")

        refresh_col1, refresh_col2 = st.columns([4, 1])
        with refresh_col1:
            st.caption("Raw output from nvidia-smi command")
        with refresh_col2:
            if st.button("🔄 Refresh", use_container_width=True, type="primary"):
                snapshot = capture_nvidia_smi()
                record_smi_snapshot(snapshot)
                st.rerun()

        snapshot = capture_nvidia_smi()
        st.code(
            snapshot,
            language="bash",
        )

        record_smi_snapshot(snapshot)

    with tab_timeline:
        st.markdown("#### GPU Metrics Timeline")
        st.caption("Real-time visualization of GPU utilization, memory usage, and temperature")

        if st.session_state.gpu_metrics:
            gpu_df = pd.DataFrame(st.session_state.gpu_metrics[-100:])
            if not gpu_df.empty:
                gpu_df["timestamp"] = pd.to_datetime(gpu_df["timestamp"], unit="s")

                color_map = {
                    "gpu_util": "#1f77b4",
                    "gpu_mem_used_mb": "#ff7f0e",
                    "gpu_temp_c": "#d62728"
                }

                fig = px.line(
                    gpu_df,
                    x="timestamp",
                    y=["gpu_util", "gpu_mem_used_mb", "gpu_temp_c"],
                    labels={
                        "value": "Value",
                        "variable": "Metric",
                        "timestamp": "Time",
                        "gpu_util": "GPU Utilization (%)",
                        "gpu_mem_used_mb": "Memory Used (MB)",
                        "gpu_temp_c": "Temperature (°C)"
                    },
                    title="",
                    color_discrete_map={
                        "gpu_util": color_map["gpu_util"],
                        "gpu_mem_used_mb": color_map["gpu_mem_used_mb"],
                        "gpu_temp_c": color_map["gpu_temp_c"]
                    }
                )

                fig.update_layout(
                    margin=dict(l=40, r=20, t=20, b=40),
                    height=400,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=-0.25,
                        xanchor="center",
                        x=0.5,
                        title=""
                    ),
                    xaxis_title="Time",
                    yaxis_title="Value",
                    hovermode="x unified",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True,
                    config={
                        "displayModeBar": True,
                        "displaylogo": False,
                        "modeBarButtonsToRemove": ["pan2d", "lasso2d"],
                    }
                )
        else:
            render_smi_history()

def render_admin_panel():
    st.subheader("Artifacts & DVC")
    render_dvc_section()
    history_df = load_run_history(limit=5)
    if not history_df.empty:
        st.caption("Recent runs")
        st.dataframe(history_df[["run_id", "rows_processed", "avg_sec_per_row"]], use_container_width=True)
    else:
        st.info("No run metadata yet.")

def render_last_run_summary():
    if st.session_state.last_run_result:
        st.json(st.session_state.last_run_result)
    else:
        st.info("Run the pipeline to see summary data.")

def main():
    init_session_state()

    if st.session_state.gpu_watcher is None:
        try:
            start_gpu_watcher()
        except Exception:
            pass

    running = st.session_state.run_thread is not None and st.session_state.run_thread.is_alive()

    if running:

        auto_refresh_placeholder = st.empty()
        with auto_refresh_placeholder.container():
            col1, col2, col3 = st.columns([8, 1, 1])
            with col1:
                st.info("🔄 Pipeline is running. Auto-refreshing every 2 seconds...")
            with col2:
                if st.button("🔄 Refresh", help="Manual refresh", use_container_width=True):
                    st.rerun()
            with col3:
                if st.button("⏸️ Pause", help="Pause auto-refresh", use_container_width=True):
                    st.session_state.auto_refresh_paused = not st.session_state.get("auto_refresh_paused", False)
                    st.rerun()

        if not st.session_state.get("auto_refresh_paused", False):
            time.sleep(2.0)
            st.rerun()
    else:

        if st.session_state.get("auto_refresh_paused", False):
            st.session_state.auto_refresh_paused = False

    st.title("Experiment Control & Watchdog")
    st.caption("Launch orchestrations, edit prompts, monitor GPU, and manage DVC artifacts.")

    render_gpu_panel()

    st.divider()

    render_run_controls()

    st.divider()

    render_admin_panel()

    tab_live, tab_prompts, tab_history, tab_summary = st.tabs(
        ["Live Output", "Prompt Editor", "Run History", "Last Run Summary"]
    )
    with tab_live:
        render_live_results()
    with tab_prompts:
        render_prompt_editor()
    with tab_history:
        render_run_history()
    with tab_summary:
        render_last_run_summary()

if __name__ == "__main__":
    main()

