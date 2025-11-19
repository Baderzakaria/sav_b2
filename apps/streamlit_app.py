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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import pandas as pd
import streamlit as st
from langgraph.graph import START, END, StateGraph


DEFAULT_RESULTS_PATH = Path("data/results/free tweet export_results.csv")
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
    return df


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
        results_path = st.text_input(
            "Results CSV path",
            value=str(DEFAULT_RESULTS_PATH),
            help="Path to the CSV generated by the pipeline.",
        )
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
        results_path=Path(results_path),
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
        utiles = df["A1_utile"].fillna(False)
        cols[2].metric("Useful %", f"{(utiles.mean() * 100):.1f}%")
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
    display_cols = [c for c in ["id", "created_at", "screen_name", "full_text", "A1_utile"] if c in filtered.columns]
    st.dataframe(filtered[display_cols], use_container_width=True)

    st.subheader("Inspect a Tweet")
    if filtered.empty:
        st.info("No tweets to display with current filters.")
    else:
        row_map = {str(row["id"]): row for _, row in filtered.iterrows()}
        selected_id = st.selectbox("Choose tweet", list(row_map.keys()))
        selected_row = row_map[selected_id]
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

