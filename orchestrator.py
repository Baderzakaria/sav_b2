import csv
import json
import os
import time
import re
import mlflow
from typing import TypedDict, Any, Dict, Optional
from langgraph.graph import StateGraph, START, END
from utils_runtime import call_native_generate

# --- Configuration ---
INPUT_CSV = "data/free tweet export.csv"
RESULTS_DIR = "data/results"
MAX_ROWS = 6000
MODEL_NAME = "llama3.2:3b"
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MLFLOW_EXPERIMENT_NAME = "FreeMind_Orchestrator"

# --- Load Prompts ---
with open("prompts/freemind_prompts.json", "r", encoding="utf-8") as f:
    PROMPTS = json.load(f)["agents"]

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

# --- 1. Define State ---
class State(TypedDict):
    full_text: str
    # Outputs from agents
    a1_result: str
    a2_result: str
    a3_result: str
    a4_result: str
    a5_result: str
    final_json: Dict[str, Any]

# --- 2. Generic Node Function ---
def run_agent(state: State, agent_key: str, result_key: str):
    prompt_cfg = PROMPTS[agent_key]
    prompt = prompt_cfg["prompt_template"].replace("{{full_text}}", state["full_text"])
    response = call_native_generate(OLLAMA_HOST, MODEL_NAME, prompt)
    return {result_key: response}

# Specific Nodes
def node_a1(state): return run_agent(state, "A1_utile", "a1_result")
def node_a2(state): return run_agent(state, "A2_categorie", "a2_result")
def node_a3(state): return run_agent(state, "A3_sentiment", "a3_result")
def node_a4(state): return run_agent(state, "A4_type_probleme", "a4_result")
def node_a5(state): return run_agent(state, "A5_gravite", "a5_result")

def node_a6_checker(state):
    # Aggregate previous results for the checker
    results = {
        "A1": state.get("a1_result"),
        "A2": state.get("a2_result"),
        "A3": state.get("a3_result"),
        "A4": state.get("a4_result"),
        "A5": state.get("a5_result"),
    }
    results_str = json.dumps(results, indent=2)
    
    prompt_cfg = PROMPTS["A6_checker"]
    prompt = prompt_cfg["prompt_template"].replace("{{agent_results}}", results_str)
    response = call_native_generate(OLLAMA_HOST, MODEL_NAME, prompt)
    
    # Use robust extractor for the final JSON
    final_json = extract_json_value(response)
    if not isinstance(final_json, dict):
         final_json = {"status": "error", "raw_response": response}
    else:
        if "utile" not in final_json and "useful" in final_json:
            final_json["utile"] = final_json.get("useful")
        
    return {"final_json": final_json}

# --- 3. Build Graph (Parallel Execution) ---
def build_pipeline():
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("A1", node_a1)
    workflow.add_node("A2", node_a2)
    workflow.add_node("A3", node_a3)
    workflow.add_node("A4", node_a4)
    workflow.add_node("A5", node_a5)
    workflow.add_node("A6", node_a6_checker)
    
    # Parallel edges: Start -> A1..A5
    workflow.add_edge(START, "A1")
    workflow.add_edge(START, "A2")
    workflow.add_edge(START, "A3")
    workflow.add_edge(START, "A4")
    workflow.add_edge(START, "A5")
    
    # Converge: A1..A5 -> A6
    workflow.add_edge("A1", "A6")
    workflow.add_edge("A2", "A6")
    workflow.add_edge("A3", "A6")
    workflow.add_edge("A4", "A6")
    workflow.add_edge("A5", "A6")
    
    workflow.add_edge("A6", END)
    return workflow.compile()

# --- 4. Execution & Logging ---
def main():
    app = build_pipeline()
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    mlflow.langchain.autolog() 
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # New: Timestamped Log file (don't overwrite old logs)
    timestamp = int(time.time())
    log_csv_path = os.path.join(RESULTS_DIR, f"freemind_log_{timestamp}.csv")
    metadata_json_path = os.path.join(RESULTS_DIR, f"run_metadata_{timestamp}.json")
    
    # Read CSV
    rows = []
    with open(INPUT_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for i, r in enumerate(reader):
            if i >= MAX_ROWS:
                break
            rows.append(r)

    log_entries = []
    start_pipeline = time.time()
    
    print(f"=== FreeMind Orchestrator run_{timestamp} ===")
    print(f"Input file: {INPUT_CSV} | Rows planned: {len(rows)} | Model: {MODEL_NAME}")

    with mlflow.start_run(run_name=f"pipeline_run_{timestamp}"):
        for i, row in enumerate(rows, 1):
            full_text = row.get("full_text", "") or str(row)
            row_start = time.time()
            
            print(f"\n[{i}/{len(rows)}] Tweet ID={row.get('id')}")
            print(f"  Text: {full_text[:500]}{'...' if len(full_text) > 500 else ''}")
            
            # Run Pipeline
            inputs = {"full_text": full_text}
            output = app.invoke(inputs)
            
            elapsed = time.time() - row_start
            final = output.get("final_json", {})
            
            # Extract Clean Values from Raw Agent Outputs
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

            # Log clean JSON artifact
            mlflow.log_dict(final, f"results/tweet_{row.get('id', i)}.json")

            # --- Prepare Log Entry ---
            entry = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "row_index": i,
                "tweet_id": row.get("id"),
                "screen_name": row.get("screen_name"),
                "full_text": full_text,
                "elapsed_sec": round(elapsed, 2),
                
                # Clean Agent Columns (Value Only)
                "A1_utile": a1_val,
                "A2_categorie": a2_val,
                "A3_sentiment": a3_val,
                "A4_type": a4_val,
                "A5_gravity": a5_val,

                # Consolidated Final Labels (from A6)
                "Final_utile": final.get("utile"),
                "Final_categorie": final.get("categorie"),
                "Final_sentiment": final.get("sentiment"),
                "Final_gravity": final.get("score_gravite"),
                
                # Diagnostics
                "status": final.get("status", "unclear"),
                "json_valid": "yes" if "status" in final else "no",
            }
            log_entries.append(entry)
            
    total_duration = time.time() - start_pipeline

    # Save to CSV
    if log_entries:
        keys = log_entries[0].keys()
        with open(log_csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(log_entries)
        print(f"Done! Log saved to {log_csv_path}")
        
        # Save Metadata (DVC-friendly JSON)
        metadata = {
            "run_id": f"run_{timestamp}",
            "timestamp": timestamp,
            "input_file": INPUT_CSV,
            "rows_processed": len(log_entries),
            "total_duration_sec": round(total_duration, 2),
            "avg_sec_per_row": round(total_duration / len(log_entries), 2) if log_entries else 0,
            "model": MODEL_NAME,
            "experiment": MLFLOW_EXPERIMENT_NAME,
            "output_files": {
                "log_csv": log_csv_path,
                "results_dir": RESULTS_DIR
            }
        }
        with open(metadata_json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {metadata_json_path}")
        print(f"\n=== Run Summary ===")
        print(f"Rows processed : {len(log_entries)}")
        print(f"Total duration : {total_duration:.2f}s")
        print(f"Avg per row    : {metadata['avg_sec_per_row']:.2f}s")
        print(f"CSV            : {log_csv_path}")
        print(f"Metadata       : {metadata_json_path}")
    else:
        print("No rows processed; nothing to log.")

if __name__ == "__main__":
    main()
