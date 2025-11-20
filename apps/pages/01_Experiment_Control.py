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

# Ensure project root is importable when Streamlit runs from apps/
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


def _request_streamlit_rerun(is_auto: bool = True) -> None:
    """Ask Streamlit to rerun the current session (safe no-op outside Streamlit)."""
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
    except Exception as exc:  # pragma: no cover - defensive guardrail
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
        "pipeline_status": "idle",  # idle, running, completed, error
        "live_log_path": None,  # Path to current JSONL file
        "jsonl_read_position": 0,  # Track last read position in JSONL file
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def start_gpu_watcher() -> None:
    """Start background GPU monitoring thread."""
    if st.session_state.gpu_watcher is not None:
        # Check if watcher is still alive
        if hasattr(st.session_state.gpu_watcher, '_threads'):
            threads_alive = any(t.is_alive() for t in st.session_state.gpu_watcher._threads)
            if threads_alive:
                return
        # Watcher exists but threads are dead, restart
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
                # Keep last 300 entries
                if len(st.session_state.gpu_metrics) > 300:
                    st.session_state.gpu_metrics.pop(0)
        
        watcher.start(update_metrics)
        st.session_state.gpu_watcher = watcher
        # Don't sleep here - it blocks the UI thread
    except Exception as exc:
        st.session_state.gpu_watcher = None
        # Don't show warning here, let render_gpu_panel handle it


def stop_gpu_watcher() -> None:
    """Stop background GPU monitoring."""
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
    st.session_state.jsonl_read_position = 0  # Reset read position for new run
    
    # Store the expected live log path (will be created by orchestrator)
    import time
    timestamp = int(time.time())
    live_dir = config.live_log_dir or config.results_dir
    expected_live_log_path = os.path.join(live_dir, f"live_run_{timestamp}.jsonl")
    st.session_state.live_log_path = expected_live_log_path

    def progress(entry):
        with SESSION_LOCK:
            st.session_state.live_entries.append(entry)
            # Add latest GPU metrics to entry if available
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
                # Update live_log_path from result if available
                if result and "live_log" in result and result["live_log"]:
                    st.session_state.live_log_path = result["live_log"]
            _request_streamlit_rerun(is_auto=False)
        except Exception as exc:  # pragma: no cover - surfaced in UI
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
    fig = px.line(
        df,
        x="timestamp",
        y=["gpu_util", "gpu_mem_used_mb", "gpu_temp_c"],
        labels={"value": "Metric", "variable": "Series"},
        title="GPU Utilization / Memory / Temperature",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True, height=340)


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
    """
    Read JSONL file incrementally from a given position.
    Returns (new_entries, new_position).
    """
    if not file_path or not os.path.exists(file_path):
        return [], start_position
    
    new_entries = []
    new_position = start_position
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            # Seek to last known position
            if start_position > 0:
                f.seek(start_position)
            
            # Read new lines
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    new_entries.append(entry)
                except json.JSONDecodeError:
                    # Skip malformed lines
                    continue
            
            # Update position to end of file
            new_position = f.tell()
    except (IOError, OSError) as e:
        LOGGER.debug(f"Error reading JSONL file {file_path}: {e}")
        return [], start_position
    
    return new_entries, new_position


def find_latest_jsonl_file() -> Optional[str]:
    """Find the most recent live_run_*.jsonl file in results directory."""
    results_dir = Path(RESULTS_DIR)
    if not results_dir.exists():
        return None
    
    jsonl_files = list(results_dir.glob("live_run_*.jsonl"))
    if not jsonl_files:
        return None
    
    # Sort by modification time, most recent first
    latest = max(jsonl_files, key=lambda p: p.stat().st_mtime)
    return str(latest)


def get_live_entries_from_file() -> List[Dict]:
    """
    Read live entries from JSONL file, combining with session state entries.
    This ensures we get real-time updates even if callbacks miss some entries.
    """
    live_entries = list(st.session_state.live_entries)  # Start with callback entries
    live_log_path = st.session_state.get("live_log_path")
    
    # Fallback: find latest JSONL file if path not set
    if not live_log_path or not os.path.exists(live_log_path):
        latest_file = find_latest_jsonl_file()
        if latest_file:
            live_log_path = latest_file
            st.session_state.live_log_path = latest_file
            st.session_state.jsonl_read_position = 0  # Reset position for new file
    
    if live_log_path and os.path.exists(live_log_path):
        # Read new entries from file
        read_pos = st.session_state.get("jsonl_read_position", 0)
        new_entries, new_position = read_jsonl_file(live_log_path, read_pos)
        
        if new_entries:
            # Update session state position
            st.session_state.jsonl_read_position = new_position
            
            # Merge new entries (avoid duplicates by row_index)
            existing_indices = {e.get("row_index") for e in live_entries if e.get("row_index")}
            for entry in new_entries:
                row_idx = entry.get("row_index")
                if row_idx and row_idx not in existing_indices:
                    live_entries.append(entry)
                    existing_indices.add(row_idx)
                elif not row_idx:
                    # If no row_index, append anyway (shouldn't happen but be safe)
                    live_entries.append(entry)
            
            # Sort by row_index if available
            if all(e.get("row_index") for e in live_entries):
                live_entries.sort(key=lambda x: x.get("row_index", 0))
    
    return live_entries


def render_live_results():
    st.subheader("Live Run Output")
    running = st.session_state.run_thread is not None and st.session_state.run_thread.is_alive()
    
    # Read entries from file in real-time
    live_entries = get_live_entries_from_file()
    
    # Update session state for backward compatibility
    if live_entries:
        st.session_state.live_entries = live_entries
    
    rows_done = len(live_entries)
    target_rows = max(st.session_state.get("current_run_target_rows", rows_done), 1)
    
    # Status indicator
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
    
    # Show file source info
    live_log_path = st.session_state.get("live_log_path")
    status_display = f"Status: {status_colors.get(status, '⚪')} {status_text.get(status, 'Unknown')}"
    if live_log_path and os.path.exists(live_log_path):
        file_size = os.path.getsize(live_log_path)
        status_display += f" | 📄 Reading from: {os.path.basename(live_log_path)} ({file_size:,} bytes)"
    st.caption(status_display)
    
    # Progress bar
    if target_rows > 0:
        progress_value = min(rows_done / target_rows, 1.0)
        progress_text = f"{rows_done} / {target_rows} rows processed"
        if running:
            progress_text = f"🔄 {progress_text}"
        st.progress(progress_value, text=progress_text)
    else:
        st.progress(0.0, text="Not started")

    # Streaming output
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
                        "status": entry.get("status"),  # Show status from JSON
                        "elapsed_sec": entry.get("elapsed_sec"),
                    }
                    yield json.dumps(summary, ensure_ascii=False) + "\n"
            
            stream_placeholder.write_stream(stream_generator)
            
            # Data table
            if len(live_entries) > 0:
                live_df = pd.DataFrame(live_entries)
                st.dataframe(live_df.tail(50), use_container_width=True, height=400)
                
                # GPU chart if we have GPU data
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
    # First try to get from GPU watcher (most recent)
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
    
    # Fallback to GPU metrics from session state
    if st.session_state.get("gpu_metrics"):
        metrics = st.session_state.gpu_metrics
        if metrics and len(metrics) > 0:
            latest = metrics[-1]
            if latest.get("gpu_util") is not None:
                return latest
    
    # Try parsing from nvidia-smi snapshot history
    if st.session_state.get("smi_history"):
        history = st.session_state.smi_history
        if history and len(history) > 0:
            latest_smi = history[-1]
            # Check if we have valid metrics
            if latest_smi.get("gpu_util") is not None or latest_smi.get("mem_used_mb") is not None:
                return {
                    "gpu_util": latest_smi.get("gpu_util", 0),
                    "gpu_mem_used_mb": latest_smi.get("mem_used_mb", 0),
                    "gpu_mem_total_mb": latest_smi.get("mem_total_mb", 0),
                    "gpu_temp_c": latest_smi.get("temperature", 0),
                }
    
    # Last resort: check live entries
    for entry in reversed(st.session_state.live_entries):
        if all(k in entry for k in ("gpu_util", "gpu_mem_used_mb", "gpu_temp_c")):
            if entry.get("gpu_util") is not None:
                return entry
    return None


def record_smi_snapshot(snapshot: str) -> None:
    """Parse nvidia-smi output and record metrics."""
    if not snapshot or "NVIDIA-SMI" not in snapshot:
        return
    
    history = st.session_state.get("smi_history", [])
    
    # More flexible regex patterns for nvidia-smi
    # Memory: "1234MiB / 8192MiB" or "1234 / 8192 MiB"
    mem_match = re.search(r"(\d+)\s*MiB\s*/\s*(\d+)\s*MiB", snapshot) or re.search(r"(\d+)\s*/\s*(\d+)\s*MiB", snapshot)
    
    # GPU Util: "45% Default" or "45 %"
    util_match = re.search(r"(\d+)\s*%\s+Default", snapshot) or re.search(r"(\d+)\s*%", snapshot.split("\n")[0])
    
    # Temperature: "| 65C" or "65C" or "65 C"
    temp_match = re.search(r"\|\s*(\d+)\s*C", snapshot) or re.search(r"(\d+)\s*C", snapshot)
    
    entry = {
        "timestamp": time.time(),
        "gpu_util": float(util_match.group(1)) if util_match else None,
        "mem_used_mb": float(mem_match.group(1)) if mem_match else None,
        "mem_total_mb": float(mem_match.group(2)) if mem_match else None,
        "temperature": float(temp_match.group(1)) if temp_match else None,
    }
    
    # Only add if we got at least one metric
    if any(v is not None for v in [entry["gpu_util"], entry["mem_used_mb"], entry["temperature"]]):
        history.append(entry)
        history = history[-SMI_HISTORY_LIMIT:]
        st.session_state["smi_history"] = history


def render_smi_history():
    history = st.session_state.get("smi_history", [])
    if not history:
        st.info("nvidia-smi history will appear once metrics are captured.")
        return
    df = pd.DataFrame(history)
    # Ensure proper dtypes
    for col in ["gpu_util", "mem_used_mb", "temperature"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Convert timestamp - handle both float (Unix timestamp) and datetime
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
        st.info("nvidia-smi history will appear once metrics are captured.")
        return
    fig = px.line(df_long, x="timestamp", y="value", color="metric", title="nvidia-smi Timeline")
    fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True, height=260)


def render_run_controls():
    running = st.session_state.run_thread is not None and st.session_state.run_thread.is_alive()
    st.subheader("Run Controls")
    preset_models = ["llama3.2:3b", "mistral:latest", "llama3.1:8b"]
    current_model = st.session_state.get("config_model", MODEL_NAME)
    default_choice = current_model if current_model in preset_models else "Custom"

    with st.form("run_control_form"):
        st.text_input("Input CSV", key="config_input_csv", value=st.session_state.get("config_input_csv", INPUT_CSV))
        st.number_input(
            "Number of tweets",
            min_value=1,
            max_value=10000,
            key="config_max_rows",
        )
        model_options = preset_models + ["Custom"]
        model_choice = st.selectbox(
            "Model preset",
            options=model_options,
            index=model_options.index(default_choice),
            key="config_model_choice",
        )
        if model_choice == "Custom":
            custom_default = current_model if default_choice == "Custom" else st.session_state.get("config_model_custom", "")
            st.text_input(
                "Custom model name",
                key="config_model_custom",
                value=custom_default,
            )
            st.session_state.config_model = st.session_state.config_model_custom
        else:
            st.session_state.config_model = model_choice
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

    # Option to run the exact CLI-style defaults (PipelineConfig with no overrides)
    if st.button("Launch Default (CLI) Run", disabled=running, use_container_width=True):
        launch_pipeline(PipelineConfig())
        st.success("Default run started (same as `python orchestrator.py`).")

    if st.button("Stop Run", disabled=not running, use_container_width=True):
        stop_pipeline()
        st.warning("Stop signal sent.")
    
    # Show error if any
    if st.session_state.get("run_error"):
        st.error("❌ Pipeline Error")
        with st.expander("Error Details", expanded=True):
            st.code(st.session_state["run_error"], language="python")
    
    # Show last run result summary
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
    st.subheader("GPU & Watchdog")
    
    # Auto-start GPU watcher if not running
    watcher_error = None
    if st.session_state.gpu_watcher is None:
        try:
            start_gpu_watcher()
        except Exception as exc:
            watcher_error = str(exc)
    
    # Show status and error if any
    if watcher_error:
        st.error(f"⚠️ GPU monitoring unavailable: {watcher_error}")
        st.info("💡 Install pynvml: `pip install pynvml` or use nvidia-smi snapshot below")
    elif st.session_state.gpu_watcher:
        # Check if threads are alive
        try:
            threads_alive = any(t.is_alive() for t in st.session_state.gpu_watcher._threads) if hasattr(st.session_state.gpu_watcher, '_threads') else False
            watcher_status = "🟢 Active" if threads_alive else "🟡 Starting..."
            st.caption(f"GPU Watcher: {watcher_status}")
        except Exception:
            st.caption("GPU Watcher: 🟢 Active")
    else:
        st.caption("GPU Watcher: ⚪ Inactive")
    
    latest = latest_gpu_entry()
    metric_cols = st.columns(3)
    
    if latest:
        metric_cols[0].metric("GPU Util %", f"{latest['gpu_util']:.0f}%")
        mem_display = f"{latest['gpu_mem_used_mb']:.0f}/{latest.get('gpu_mem_total_mb', 0):.0f} MB"
        metric_cols[1].metric("VRAM Used", mem_display)
        metric_cols[2].metric("Temp °C", f"{latest['gpu_temp_c']:.0f}")
    else:
        metric_cols[0].metric("GPU Util %", "--")
        metric_cols[1].metric("VRAM Used", "--")
        metric_cols[2].metric("Temp °C", "--")
        if not watcher_error:
            st.info("⏳ Waiting for GPU metrics... (checking nvidia-smi)")

    control_col, snapshot_col = st.columns(2)
    with control_col:
        workers = st.slider(
            "GPU multithreading (watchdog workers)",
            min_value=1,
            max_value=4,
            key="config_gpu_threads",
            help="Number of watcher threads sampling GPU metrics.",
        )
        mem_floor = st.slider(
            "GPU free memory floor (MB)",
            min_value=512,
            max_value=8192,
            step=256,
            key="config_gpu_mem_floor",
            help="Pipeline will pause if free GPU memory drops below this threshold.",
        )
        poll_interval = st.number_input(
            "Watchdog poll interval (sec)",
            min_value=0.5,
            max_value=10.0,
            step=0.5,
            key="config_gpu_poll",
            help="Lower values capture GPU spikes faster at the cost of CPU overhead.",
        )
        temp_ceiling = st.slider(
            "GPU temperature ceiling (°C)",
            min_value=60,
            max_value=90,
            step=5,
            key="config_gpu_temp",
            help="Alert if GPU temperature exceeds this value.",
        )
        
        # Restart watcher if config changed
        if st.session_state.gpu_watcher:
            try:
                # Update watcher config
                watcher = st.session_state.gpu_watcher
                watcher.mem_floor_mb = mem_floor
                watcher.temp_ceiling_c = temp_ceiling
                watcher.poll_interval_sec = poll_interval
                watcher.num_workers = workers
            except Exception:
                pass
    
    with snapshot_col:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption("Current nvidia-smi snapshot")
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                # Force refresh GPU metrics
                snapshot = capture_nvidia_smi()
                record_smi_snapshot(snapshot)
                st.rerun()
        
        snapshot = capture_nvidia_smi()
        st.text_area(
            "nvidia-smi",
            value=snapshot,
            height=220,
            label_visibility="collapsed",
            key="nvidia_smi_display",
        )
        # Always record snapshot to ensure metrics are available
        record_smi_snapshot(snapshot)
        
        # Show GPU metrics timeline from watcher
        if st.session_state.gpu_metrics:
            gpu_df = pd.DataFrame(st.session_state.gpu_metrics[-100:])
            if not gpu_df.empty:
                gpu_df["timestamp"] = pd.to_datetime(gpu_df["timestamp"], unit="s")
                fig = px.line(
                    gpu_df,
                    x="timestamp",
                    y=["gpu_util", "gpu_mem_used_mb", "gpu_temp_c"],
                    labels={"value": "Metric", "variable": "Series"},
                    title="Live GPU Metrics Timeline",
                )
                fig.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    height=200,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                st.plotly_chart(fig, use_container_width=True, height=200)
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
    
    # Initialize GPU watcher early - before any rendering
    if st.session_state.gpu_watcher is None:
        try:
            start_gpu_watcher()
        except Exception:
            pass  # GPU monitoring optional
    
    # Check if pipeline is running
    running = st.session_state.run_thread is not None and st.session_state.run_thread.is_alive()
    
    # Auto-refresh mechanism for real-time updates
    if running:
        # Use st.empty() for auto-refreshing content
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
        
        # Auto-refresh when not paused
        if not st.session_state.get("auto_refresh_paused", False):
            time.sleep(2.0)  # Wait 2 seconds for new data
            st.rerun()
    else:
        # Clear pause state when not running
        if st.session_state.get("auto_refresh_paused", False):
            st.session_state.auto_refresh_paused = False

    st.title("Experiment Control & Watchdog")
    st.caption("Launch orchestrations, edit prompts, monitor GPU, and manage DVC artifacts.")

    # Line 1: GPU & Watchdog (full width, splits its own inner columns) - FIRST
    render_gpu_panel()

    st.divider()
    # Line 2: Run controls (full width)
    render_run_controls()

    st.divider()
    # Line 3: Artifacts & DVC
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

