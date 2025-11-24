import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from freemind_env import load_environment
from orchestrator import (  # noqa: E402
    INPUT_CSV,
    OPENROUTER_APP_TITLE,
    OPENROUTER_BASE_URL,
    OPENROUTER_MODEL,
    OPENROUTER_REFERER,
    PipelineConfig,
    build_pipeline,
    extract_json_value,
    load_prompts,
)

load_environment()

RESULTS_DIR = Path("data/results")


@st.cache_data(show_spinner=False)
def load_results_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "row_index" not in df.columns:
        df["row_index"] = df.index + 1
    if "tweet_id" not in df.columns:
        df["tweet_id"] = df.get("id", df.index + 1)
    return df


def list_result_files(limit: int = 25) -> List[Tuple[str, Path]]:
    files: List[Tuple[str, Path]] = []
    if RESULTS_DIR.exists():
        for csv_path in sorted(RESULTS_DIR.glob("freemind_log_*.csv"), reverse=True)[:limit]:
            label = csv_path.name.replace("freemind_log_", "")
            files.append((label, csv_path))
    return files


def _provider_options() -> List[str]:
    return ["ollama", "openrouter"]


def _provider_defaults() -> Dict[str, str]:
    return {
        "provider": st.session_state.get("config_llm_provider", "ollama"),
        "openrouter_model": st.session_state.get("config_openrouter_model", OPENROUTER_MODEL),
        "openrouter_api_key": st.session_state.get("config_openrouter_api_key", os.environ.get("OPENROUTER_API_KEY", "")),
        "openrouter_base_url": st.session_state.get("config_openrouter_base_url", OPENROUTER_BASE_URL),
        "openrouter_referer": st.session_state.get("config_openrouter_referer", OPENROUTER_REFERER),
        "openrouter_title": st.session_state.get("config_openrouter_title", OPENROUTER_APP_TITLE),
        "ollama_model": st.session_state.get("config_model", "mistral:7b"),
    }


def _render_provider_settings() -> Tuple[str, Dict[str, str]]:
    defaults = _provider_defaults()
    provider_opts = _provider_options()
    try:
        default_idx = provider_opts.index(defaults["provider"])
    except ValueError:
        default_idx = 0

    provider_choice = st.selectbox(
        "LLM Provider",
        options=provider_opts,
        index=default_idx,
        key="tweet_workbench_provider",
    )

    settings: Dict[str, str] = {}
    if provider_choice == "ollama":
        settings["model_name"] = st.text_input(
            "Ollama Model",
            value=defaults["ollama_model"],
            help="Must be available locally via `ollama ls` (e.g., mistral:7b).",
            key="tweet_workbench_ollama_model",
        )
    else:
        settings["openrouter_model"] = st.text_input(
            "OpenRouter Model ID",
            value=defaults["openrouter_model"],
            help="Model slug from openrouter.ai (e.g., mistralai/mistral-small-3.1-24b-instruct:free).",
            key="tweet_workbench_openrouter_model",
        )
        settings["openrouter_api_key"] = st.text_input(
            "OpenRouter API Key",
            value=defaults["openrouter_api_key"],
            type="password",
            help="Stored only in Streamlit session state for this session.",
            key="tweet_workbench_openrouter_api_key",
        )
        settings["openrouter_base_url"] = st.text_input(
            "OpenRouter Base URL",
            value=defaults["openrouter_base_url"],
            key="tweet_workbench_openrouter_base",
        )
        settings["openrouter_referer"] = st.text_input(
            "HTTP Referer (optional)",
            value=defaults["openrouter_referer"],
            key="tweet_workbench_openrouter_referer",
        )
        settings["openrouter_title"] = st.text_input(
            "App Title (optional)",
            value=defaults["openrouter_title"],
            key="tweet_workbench_openrouter_title",
        )

    return provider_choice, settings


def _format_row_option(row: pd.Series) -> str:
    snippet = (row.get("clean_text") or row.get("full_text") or "")[:80]
    tweet_id = row.get("tweet_id") or row.get("id") or row.get("row_index")
    return f"[{row.get('row_index', '?')}] @{row.get('screen_name', 'unknown')}: {snippet}… (id={tweet_id})"


def render_tweet_selector() -> Optional[pd.Series]:
    files = list_result_files()
    manual_text = st.text_area("Tweet text", value="", height=200, key="tweet_workbench_manual_input")

    if not files:
        st.info("No log files found. Enter text manually or run the orchestrator first.")
        if manual_text.strip():
            return pd.Series({"clean_text": manual_text.strip(), "tweet_id": "manual"})
        return None

    labels = [label for label, _ in files]
    default_idx = 0
    selected_label = st.selectbox("Results CSV", options=labels, index=default_idx)
    selected_path = dict(files)[selected_label]

    try:
        df = load_results_csv(str(selected_path))
    except Exception as exc:
        st.error(f"Failed to load {selected_path.name}: {exc}")
        return None

    options = [_format_row_option(row) for _, row in df.iterrows()]
    if not options:
        st.warning("Selected CSV is empty. Enter text manually instead.")
        if manual_text.strip():
            return pd.Series({"clean_text": manual_text.strip(), "tweet_id": "manual"})
        return None

    choice = st.selectbox("Select tweet", options=options, index=0, key="tweet_workbench_row_choice")
    idx = options.index(choice)
    selected_row = df.iloc[idx]

    default_text = selected_row.get("clean_text") or selected_row.get("full_text") or ""
    clean_text = st.text_area(
        "Review / edit text before running agents",
        value=default_text,
        height=220,
        key="tweet_workbench_clean_text",
    )

    if manual_text.strip():
        clean_text = manual_text.strip()

    selected_row = selected_row.copy()
    selected_row["clean_text"] = clean_text
    return selected_row


def run_agents_for_text(text: str, config: PipelineConfig) -> Dict:
    prompts = load_prompts()
    app = build_pipeline(prompts, config)
    output = app.invoke({"clean_text": text})
    final = output.get("final_json", {})
    agents = {
        "a1": output.get("a1_result"),
        "a2": output.get("a2_result"),
        "a3": output.get("a3_result"),
        "a4": output.get("a4_result"),
        "a5": output.get("a5_result"),
    }
    return {"final": final, "agents": agents}


def render_results_section(result: Dict) -> None:
    final = result.get("final", {})
    agents = result.get("agents", {})

    st.subheader("Final Checker Output")
    st.json(final)

    st.markdown("#### Agent Responses")
    cols = st.columns(5)
    for idx, key in enumerate(["a1", "a2", "a3", "a4", "a5"]):
        val = final.get(["utile", "categorie", "sentiment", "type_probleme", "score_gravite"][idx], None) if final else None
        label = ["A1 utile", "A2 catégorie", "A3 sentiment", "A4 type", "A5 gravité"][idx]
        cols[idx].metric(label, val if val is not None else "—")

    with st.expander("Raw Agent Outputs", expanded=False):
        for agent_key, payload in agents.items():
            st.code(payload or "<empty>", language="json")


def main():
    st.title("Tweet Workbench")
    st.caption("Select a tweet (or paste text) and run the multi-agent pipeline on demand.")

    selected_row = render_tweet_selector()
    provider_choice, provider_settings = _render_provider_settings()

    st.markdown("---")
    run_button = st.button("Run agents on this tweet", type="primary", use_container_width=True, disabled=selected_row is None or not selected_row.get("clean_text"))

    if run_button and selected_row is not None:
        text = (selected_row.get("clean_text") or "").strip()
        if not text:
            st.warning("Please provide tweet text first.")
            return

        model_name = provider_settings.get("model_name", "mistral:7b") if provider_choice == "ollama" else provider_settings.get("openrouter_model", OPENROUTER_MODEL)
        config = PipelineConfig(
            input_csv=INPUT_CSV,
            max_rows=1,
            model_name=model_name,
            llm_provider=provider_choice,
            enable_live_log=False,
            enable_model_warmup=False,
            prompt_overrides=None,
            openrouter_model=provider_settings.get("openrouter_model", OPENROUTER_MODEL),
            openrouter_api_key=provider_settings.get("openrouter_api_key") or os.environ.get("OPENROUTER_API_KEY"),
            openrouter_base_url=provider_settings.get("openrouter_base_url", OPENROUTER_BASE_URL),
            openrouter_headers={
                "HTTP-Referer": provider_settings.get("openrouter_referer", OPENROUTER_REFERER),
                "X-Title": provider_settings.get("openrouter_title", OPENROUTER_APP_TITLE),
            },
        )

        with st.spinner("Running agents..."):
            try:
                result = run_agents_for_text(text, config)
                render_results_section(result)
            except Exception as exc:
                st.error(f"Agent run failed: {exc}")


if __name__ == "__main__":
    main()

