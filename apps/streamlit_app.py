"""
Streamlit dashboard for inspecting tweet classification results.

Features:
- Load the latest CSV results exported by the pipeline.
- Filter and review tweets with their metadata and labels.
- Log inspected samples to MLflow for downstream analysis.
- Visualize the LangGraph agent pipeline via Mermaid diagrams.
"""

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

    # Normalize identifier columns so downstream UI logic always has sensible defaults
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
            # Try JSON or dict-like substrings
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
    """Choose the best available identifier for a tweet row."""
    for key in ("id", "tweet_id", "row_index"):
        value = row.get(key)
        if value is not None and f"{value}".strip():
            return str(value)
    return fallback


def list_result_files() -> List[tuple[str, Path]]:
    """Return available result CSVs with human-friendly labels."""
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
    """


def render_tweet_details(row: pd.Series):
    st.subheader("Tweet Details")
    meta_cols = st.columns(3)
    meta_cols[0].metric("Author", row.get("screen_name"))
    meta_cols[1].metric("Useful?", str(row.get("A1_utile")))
    meta_cols[2].metric("Favorites", row.get("favorite_count", 0))
    st.write("**Text**")
    st.write(row.get("full_text", ""))
    if row.get("url"):
        st.markdown(f"[Open on Twitter]({row['url']})")
    media_data = safe_json(row.get("media"))
    if media_data:
        st.write("**Media**")
        st.json(media_data)


def sidebar_config() -> AppConfig:
    with st.sidebar:
        st.header("Configuration")
        result_choices = list_result_files()
        if not result_choices:
            st.error("No result CSVs found in data/results/. Run the pipeline first.")
            st.stop()
        labels = [label for label, _ in result_choices]
        default_index = 0
        for idx, (_, path) in enumerate(result_choices):
            if path == DEFAULT_RESULTS_PATH:
                default_index = idx
                break
        selected_label = st.selectbox("Results CSV", labels, index=default_index)
        selected_path = dict(result_choices)[selected_label]
        tracking_uri = st.text_input(
            "MLflow tracking URI",
            value=DEFAULT_TRACKING_URI,
        )
        experiment_name = st.text_input(
            "MLflow experiment",
            value=DEFAULT_EXPERIMENT,
        )
        enable_mlflow = st.toggle("Enable MLflow logging", value=True)
    return AppConfig(
        results_path=selected_path,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        enable_mlflow=enable_mlflow,
    )


def main():
    st.set_page_config(
        page_title="FreeMind Results Explorer",
        layout="wide",
        page_icon="📊",
    )
    st.title("FreeMind Results Explorer")
    st.caption("Inspect pipeline outputs, log insights to MLflow, and view the agent graph.")

    config = sidebar_config()
    try:
        df = load_results(str(config.results_path))
    except FileNotFoundError as err:
        st.error(str(err))
        return

    st.subheader("Dataset Overview")
    cols = st.columns(4)
    cols[0].metric("Rows", len(df))
    cols[1].metric("Authors", df["screen_name"].nunique())
    if "A1_utile" in df.columns:
        utiles = df["A1_utile"]
        if utiles.notna().any():
            useful_pct = utiles.dropna().mean() * 100
            cols[2].metric("Useful %", f"{useful_pct:.1f}%")
        else:
            cols[2].metric("Useful %", "n/a")
    else:
        cols[2].metric("Useful %", "n/a")
    if "created_at" in df.columns and df["created_at"].notna().any():
        date_min = df["created_at"].min()
        date_max = df["created_at"].max()
        cols[3].metric("Date range", f"{date_min.date()} → {date_max.date()}")
    else:
        cols[3].metric("Date range", "n/a")

    st.subheader("Filters")
    author_options = sorted(df["screen_name"].dropna().unique())
    selected_authors = st.multiselect("Authors", author_options)
    query = st.text_input("Search text", placeholder="Enter keyword...")
    filtered = apply_filters(df, selected_authors, query)

    st.write(f"Showing {len(filtered)} tweets")
    preferred_cols = [
        "row_index",
        "id",
        "tweet_id",
        "created_at",
        "screen_name",
        "full_text",
        "A1_utile",
        "A2_categorie",
        "A3_sentiment",
        "A4_type",
        "A5_gravity",
        "Final_utile",
        "Final_categorie",
        "Final_sentiment",
        "Final_gravity",
        "status",
    ]
    display_cols = [c for c in preferred_cols if c in filtered.columns]
    if not display_cols:
        display_cols = filtered.columns.tolist()[:8]
    display_df = filtered[display_cols].copy()
    bool_cols = [col for col in ["A1_utile", "Final_utile"] if col in display_df.columns]
    for col in bool_cols:
        display_df[col] = display_df[col].map(
            lambda v: "True" if v is True else ("False" if v is False else None)
        )
    st.dataframe(display_df, width="stretch")

    st.subheader("Inspect a Tweet")
    if filtered.empty:
        st.info("No tweets to display with current filters.")
    else:
        option_labels: Dict[str, pd.Series] = {}
        for idx, row in filtered.iterrows():
            identifier = derive_row_identifier(row, str(idx))
            label = f"{identifier} • {row.get('screen_name', 'unknown')}"
            option_labels[label] = row
        selected_label = st.selectbox("Choose tweet", list(option_labels.keys()))
        selected_row = option_labels[selected_label]
        render_tweet_details(selected_row)
        if st.button("Log selection to MLflow", disabled=not config.enable_mlflow):
            run_id = log_selection_to_mlflow(selected_row, config)
            if run_id:
                st.success(f"Logged to MLflow run {run_id}")
            else:
                st.info("MLflow logging disabled.")

    st.subheader("Agent Pipeline (LangGraph)")
    mermaid_diagram = build_pipeline_mermaid()
    st.components.v1.html(render_mermaid(mermaid_diagram), height=500, scrolling=True)


if __name__ == "__main__":
    main()

