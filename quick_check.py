#!/usr/bin/env python3

import csv
import os
import sys
import time
from typing import TypedDict, Dict, Any

from langgraph.graph import StateGraph, START, END
from utils_runtime import call_native_generate


class State(TypedDict):
    row: Dict[str, Any]
    model_name: str
    result: str


def worker_node(state: State) -> Dict[str, Any]:
    row = state["row"]
    model_name = state["model_name"]
    text_value = row.get("full_text", " ".join(str(v) for v in row.values() if v))

    ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

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


def main() -> None:
    csv_path = os.environ.get("CSV_PATH", os.path.join("data", "free tweet export.csv"))
    models = sys.argv[1:] if len(sys.argv) > 1 else ["mistral:7b", "llama3.2:3b"]

    rows = []
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            rows.append(row)
            if i + 1 >= 20:
                break

    graph = build_graph()

    for model_name in models:
        print(f"\n=== Model: {model_name} ===")
        total_start = time.time()
        for idx, row in enumerate(rows, start=1):
            row_start = time.time()
            text_value = row.get("full_text", " ".join(str(v) for v in row.values() if v))
            out_state = graph.invoke({"row": row, "model_name": model_name})
            row_time = time.time() - row_start
            print(f"[{idx}] ({row_time:.2f}s) Tweet: {text_value[:1000]}{'...' if len(text_value) > 1000 else ''}")
            print(f"     Result: {out_state.get('result', '')}\n")
        total_time = time.time() - total_start
        print(f"Total: {total_time:.2f}s | Average: {total_time/len(rows):.2f}s per row\n")


if __name__ == "__main__":
    main()


