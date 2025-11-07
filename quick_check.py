#!/usr/bin/env python3

import csv
import re
import json
import os
import time
from typing import TypedDict, Dict, Any, Optional, List

from langgraph.graph import StateGraph, START, END
from utils_runtime import call_native_generate


# Configuration variables (edit as needed)
INPUT_CSV = os.path.join("data", "free tweet export.csv")
OUTPUT_CSV: Optional[str] = None  # None => will default under RESULTS_DIR
RESULTS_DIR = os.path.join("data", "results")
MAX_ROWS: int = 20
MODELS: List[str] = ["llama3.2:3b"]


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
            # fall back to common keys
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
        # primitives
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


def load_prompt_template() -> Optional[str]:
    try:
        with open(os.path.join("prompts", "freemind_prompts.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)
            t = cfg.get("a1_problem_detection", {}).get("prompt_template")
            return t if isinstance(t, str) and t.strip() else None
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

    graph = build_graph()
    prompt_template = load_prompt_template()

    # Prepare container for augmented rows (original row + single utile column)
    augmented_rows: list[Dict[str, Any]] = [dict(r) for r in rows]
    first_model = models[0] if models else ""

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

        total_time = time.time() - total_start
        print(f"Total: {total_time:.2f}s | Average: {total_time/len(rows):.2f}s per row\n")

    # Resolve default output path if not provided (place under RESULTS_DIR)
    if not output_csv:
        base_name = os.path.splitext(os.path.basename(input_csv))[0]
        output_csv = os.path.join(RESULTS_DIR, f"{base_name}_results.csv")

    # Export
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Compose final headers: original + single utile column
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
    return augmented_rows


def main() -> None:
    run_quick_check(
        input_csv=INPUT_CSV,
        output_csv=OUTPUT_CSV,
        max_rows=MAX_ROWS,
        models=MODELS,
    )


if __name__ == "__main__":
    main()


