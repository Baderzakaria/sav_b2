

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import pandas as pd
import streamlit as st
from langgraph.graph import START, END, StateGraph

RESULTS_DIR = Path("data/results")
DEFAULT_RESULTS_PATH = RESULTS_DIR / "free tweet export_results.csv"
DEFAULT_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
DEFAULT_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "freemind_streamlit_inspector")

@dataclass
class AppConfig:
    results_path: Path
    tracking_uri: str
    experiment_name: str
    enable_mlflow: bool

@st.cache_data(show_spinner=False)
def load_results(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")
    df = pd.read_csv(path)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    if "id" not in df.columns:
        if "tweet_id" in df.columns:
            df["id"] = df["tweet_id"]
        elif "row_index" in df.columns:
            df["id"] = df["row_index"]
        else:
            df["id"] = df.index
    if "row_index" not in df.columns:
        df["row_index"] = df.index + 1
    boolish_cols = [col for col in df.columns if col.endswith("_utile")]
    for col in ["Final_utile", "A1_utile"]:
        if col in df.columns and col not in boolish_cols:
            boolish_cols.append(col)
    for col in boolish_cols:
        df[col] = coerce_bool_series(df[col])
    return df

def coerce_bool_series(series: pd.Series) -> pd.Series:
    def _parse(value):
        if pd.isna(value):
            return pd.NA
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            text = value.strip()
            lower = text.lower()
            truthy = {"true", "yes", "1", "vrai"}
            falsy = {"false", "no", "0", "faux"}
            if lower in truthy:
                return True
            if lower in falsy:
                return False

            if lower.startswith("{") and "utile" in lower:
                try:
                    payload = json.loads(lower.replace("'", '"'))
                    for key in ("utile", "useful", "utilite"):
                        if isinstance(payload, dict) and isinstance(payload.get(key), bool):
                            return payload[key]
                except json.JSONDecodeError:
                    pass
            match = re.search(r"\b(true|false)\b", lower)
            if match:
                return match.group(1) == "true"
        return pd.NA

    parsed = series.apply(_parse)
    return parsed.astype("boolean")

def apply_filters(
    df: pd.DataFrame,
    authors: List[str],
    search_query: str,
) -> pd.DataFrame:
    filtered = df.copy()
    if authors:
        filtered = filtered[filtered["screen_name"].isin(authors)]
    if search_query:
        query = search_query.lower()
        mask = (
            filtered["full_text"].fillna("").str.lower().str.contains(query)
            | filtered["name"].fillna("").str.lower().str.contains(query)
        )
        filtered = filtered[mask]
    return filtered

def safe_json(value: Optional[str]) -> Optional[Dict]:
    if not isinstance(value, str) or not value.strip():
        return None
    if value.strip().lower() in {"null", "none"}:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return json.loads(value.replace("'", '"'))
        except json.JSONDecodeError:
            return None

def derive_row_identifier(row: pd.Series, fallback: str) -> str:
    for key in ("id", "tweet_id", "row_index"):
        value = row.get(key)
        if value is not None and f"{value}".strip():
            return str(value)
    return fallback

def list_result_files() -> List[tuple[str, Path]]:
    files: List[tuple[str, Path]] = []
    for path in RESULTS_DIR.glob("freemind_log_*.csv"):
        ts_part = path.stem.replace("freemind_log_", "")
        try:
            ts = datetime.fromtimestamp(int(ts_part))
            label = f"{ts.strftime('%Y-%m-%d %H:%M:%S')} • {path.name}"
        except ValueError:
            label = path.name
        files.append((label, path))
    legacy = RESULTS_DIR / "free tweet export_results.csv"
    if legacy.exists():
        files.append(("Legacy • free tweet export_results.csv", legacy))
    return sorted(files, key=lambda item: item[1].stat().st_mtime, reverse=True)

def log_selection_to_mlflow(row: pd.Series, config: AppConfig) -> Optional[str]:
    if not config.enable_mlflow:
        return None

    mlflow.set_tracking_uri(config.tracking_uri)
    experiment = mlflow.set_experiment(config.experiment_name)
    run_name = f"inspect_{row.get('id', 'unknown')}"
    with mlflow.start_run(run_name=run_name):
        params = {
            "tweet_id": str(row.get("id")),
            "author": row.get("screen_name"),
            "useful": row.get("A1_utile"),
            "created_at": row.get("created_at"),
            "url": row.get("url"),
        }
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        text = row.get("full_text", "")
        if isinstance(text, str):
            mlflow.log_text(text, artifact_file="full_text.txt")
        row_path = Path("artifacts") / f"row_{row.get('id', 'unknown')}.json"
        mlflow.log_dict(row.to_dict(), artifact_file=str(row_path))
        return mlflow.active_run().info.run_id

def build_pipeline_mermaid() -> str:
    graph = StateGraph(dict)
    for node in [
        "Load CSV",
        "Preprocess",
        "Safety Gate",
        "Agents",
        "Checker",
        "Writer",
        "Persist Results",
        "Streamlit Dashboard",
    ]:
        graph.add_node(node, lambda state, _node=node: state)
    graph.add_edge(START, "Load CSV")
    graph.add_edge("Load CSV", "Preprocess")
    graph.add_edge("Preprocess", "Safety Gate")
    graph.add_edge("Safety Gate", "Agents")
    graph.add_edge("Agents", "Checker")
    graph.add_edge("Checker", "Writer")
    graph.add_edge("Writer", "Persist Results")
    graph.add_edge("Persist Results", "Streamlit Dashboard")
    graph.add_edge("Streamlit Dashboard", END)
    compiled = graph.compile()
    return compiled.get_graph().draw_mermaid()

def render_mermaid(diagram: str) -> str:
    return f"""
    <div class="mermaid">
    {diagram}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <script>mermaid.initialize({{startOnLoad: true}});</script>

