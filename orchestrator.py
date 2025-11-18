"""LangGraph orchestrator for FreeMind A2A pipeline with guardrails."""

import json
import time
import os
from typing import Dict, Any, TypedDict
from datetime import datetime
import subprocess

from langgraph.graph import StateGraph, START, END

from models.labels import Labels, Affect, CheckerOutput
from guardrails.safety_gate import SafetyGate
from checker.rules import apply_checker_rules, load_taxonomy
from storage.sqlite_writer import SQLiteWriter
from utils_runtime import call_native_generate
from config.settings import get_settings


class State(TypedDict):
    """Shared state for the pipeline."""
    tweet: Dict[str, Any]
    context: Dict[str, str]
    results: Dict[str, Any]
    checked: Dict[str, Any]
    meta: Dict[str, Any]
    guardrails: Dict[str, Any]
    error: str


def get_run_id() -> str:
    """Generate run ID with timestamp and git SHA."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M")
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_sha = "nogit"
    return f"{timestamp}-{git_sha}"


def load_prompts() -> Dict[str, Any]:
    """Load prompt configuration."""
    settings = get_settings()
    with open(settings.prompt_file, "r", encoding="utf-8") as f:
        return json.load(f)


def render_prompt(template: str, tweet: Dict[str, Any], context: Dict[str, str]) -> str:
    """Render prompt template with placeholders."""
    return (template
            .replace("{{full_text}}", tweet.get("full_text", ""))
            .replace("{{ctx_before}}", context.get("ctx_before", ""))
            .replace("{{ctx_after}}", context.get("ctx_after", ""))
            .replace("{{ctx_refs}}", context.get("ctx_refs", "")))


def preprocess_node(state: State) -> Dict[str, Any]:
    """Build context fields non-destructively."""
    settings = get_settings()
    tweet = state["tweet"]
    
    # Build context (simplified for now - would need full tweet graph)
    context = {
        "ctx_before": "",  # Would lookup parent tweet
        "ctx_after": "",   # Would lookup replies
        "ctx_refs": ""     # Would lookup RT/quote content
    }
    
    # Truncate to limits
    for key in context:
        if context[key] and len(context[key]) > settings.ctx_max_length:
            context[key] = context[key][:settings.ctx_max_length] + "..."
    
    return {"context": context}


def safety_gate_node(state: State) -> Dict[str, Any]:
    """Check guardrails and decide allow/warn/refuse."""
    settings = get_settings()
    
    if not settings.enable_guardrails:
        return {"guardrails": {"refused": False, "warned": False, "reason": "", "flags": {}}}
    
    gate = SafetyGate()
    decision = gate.check(state["tweet"].get("full_text", ""), state["context"])
    
    guardrails_dict = gate.to_dict(decision)
    
    # If refused, skip agents
    if decision.action == "refuse":
        return {
            "guardrails": guardrails_dict,
            "results": {},
            "checked": {
                "final": None,
                "checker_status": "fail",
                "a2a_trace": {"refused_by_guardrails": True}
            }
        }
    
    return {"guardrails": guardrails_dict}


def invoke_agent(
    agent_name: str,
    prompt_template: str,
    tweet: Dict[str, Any],
    context: Dict[str, str],
    model: str,
    ollama_host: str
) -> Dict[str, Any]:
    """Invoke a single agent and parse JSON response."""
    prompt = render_prompt(prompt_template, tweet, context)
    
    try:
        raw_output = call_native_generate(ollama_host, model, prompt)
        # Try to parse JSON
        result = json.loads(raw_output)
        return result
    except json.JSONDecodeError:
        # Try to extract JSON from markdown or text
        import re
        json_match = re.search(r'\{[^{}]*\}', raw_output)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except Exception:
                pass
        return {"error": f"Invalid JSON from {agent_name}", "raw": raw_output}
    except Exception as e:
        return {"error": f"Agent {agent_name} failed: {str(e)}"}


def agents_parallel_node(state: State) -> Dict[str, Any]:
    """Run A1-A6 agents in parallel (simulated sequential for now)."""
    # Check if refused by guardrails
    if state.get("guardrails", {}).get("refused"):
        return {"results": {}}
    
    settings = get_settings()
    prompts_config = load_prompts()
    agents_config = prompts_config.get("agents", {})
    
    tweet = state["tweet"]
    context = state["context"]
    
    results = {}
    
    # A1: utile
    if "A1_utile" in agents_config:
        results["A1"] = invoke_agent(
            "A1_utile",
            agents_config["A1_utile"]["prompt_template"],
            tweet,
            context,
            settings.model_name,
            settings.ollama_host
        )
    
    # A2: categorie
    if "A2_categorie" in agents_config:
        results["A2"] = invoke_agent(
            "A2_categorie",
            agents_config["A2_categorie"]["prompt_template"],
            tweet,
            context,
            settings.model_name,
            settings.ollama_host
        )
    
    # A3: sentiment
    if "A3_sentiment" in agents_config:
        results["A3"] = invoke_agent(
            "A3_sentiment",
            agents_config["A3_sentiment"]["prompt_template"],
            tweet,
            context,
            settings.model_name,
            settings.ollama_host
        )
    
    # A4: type_probleme
    if "A4_type_probleme" in agents_config:
        results["A4"] = invoke_agent(
            "A4_type_probleme",
            agents_config["A4_type_probleme"]["prompt_template"],
            tweet,
            context,
            settings.model_name,
            settings.ollama_host
        )
    
    # A5: gravite
    if "A5_gravite" in agents_config:
        results["A5"] = invoke_agent(
            "A5_gravite",
            agents_config["A5_gravite"]["prompt_template"],
            tweet,
            context,
            settings.model_name,
            settings.ollama_host
        )
    
    # A6: affect
    if "A6_affect" in agents_config:
        results["A6"] = invoke_agent(
            "A6_affect",
            agents_config["A6_affect"]["prompt_template"],
            tweet,
            context,
            settings.model_name,
            settings.ollama_host
        )
    
    return {"results": results}


def checker_node(state: State) -> Dict[str, Any]:
    """Apply A2A checker rules."""
    # Check if refused by guardrails
    if state.get("guardrails", {}).get("refused"):
        return {"checked": state.get("checked", {})}
    
    results = state["results"]
    context = state["context"]
    taxonomy = load_taxonomy()
    
    try:
        checker_output = apply_checker_rules(results, context, taxonomy)
        
        return {
            "checked": {
                "final": checker_output.final.model_dump(),
                "checker_status": checker_output.checker_status,
                "a2a_trace": checker_output.a2a_trace
            }
        }
    except Exception as e:
        return {
            "checked": {
                "final": None,
                "checker_status": "fail",
                "a2a_trace": {"error": str(e)}
            },
            "error": f"Checker failed: {str(e)}"
        }


def writer_node(state: State) -> Dict[str, Any]:
    """Write to SQLite and handle HITL escalation."""
    settings = get_settings()
    writer = SQLiteWriter(settings.db_path)
    
    tweet = state["tweet"]
    context = state["context"]
    checked = state["checked"]
    meta = state["meta"]
    guardrails = state.get("guardrails", {})
    
    # If refused by guardrails, enqueue for review
    if guardrails.get("refused"):
        writer.enqueue_for_review(
            tweet_id=tweet.get("id"),
            reason="guardrail_refuse",
            original_labels={}
        )
        return {}
    
    # If checker failed or warned, consider HITL
    checker_status = checked.get("checker_status", "fail")
    
    if checker_status == "fail":
        writer.enqueue_for_review(
            tweet_id=tweet.get("id"),
            reason="checker_fail",
            original_labels=checked.get("a2a_trace", {}).get("inputs", {})
        )
    
    # Build CheckerOutput for storage
    final_dict = checked.get("final")
    if final_dict:
        try:
            # Reconstruct Labels from dict
            affect_dict = final_dict.get("affect")
            affect_obj = Affect(**affect_dict) if affect_dict else None
            
            labels = Labels(
                utile=final_dict["utile"],
                categorie=final_dict["categorie"],
                sentiment=final_dict["sentiment"],
                type_probleme=final_dict["type_probleme"],
                score_gravite=final_dict["score_gravite"],
                affect=affect_obj
            )
            
            checker_output = CheckerOutput(
                final=labels,
                checker_status=checker_status,
                a2a_trace=checked.get("a2a_trace", {})
            )
            
            writer.upsert_tweet(tweet, context, checker_output, meta, guardrails)
            
            # Log feedback for learning
            if checker_status in ["warn", "fail"]:
                writer.log_feedback(
                    tweet_id=tweet.get("id"),
                    feedback_type="auto_flag",
                    original_labels=checked.get("a2a_trace", {}).get("inputs", {}),
                    feedback_source="auto",
                    notes=f"Status: {checker_status}"
                )
        except Exception as e:
            print(f"Error writing tweet {tweet.get('id')}: {e}")
    
    return {}


def feedback_sink_node(state: State) -> Dict[str, Any]:
    """Collect feedback for continuous learning (placeholder)."""
    # Would write to data/feedback/training_set.jsonl
    # For now, just pass through
    return {}


def build_graph() -> StateGraph:
    """Build the LangGraph pipeline."""
    graph = StateGraph(State)
    
    # Add nodes
    graph.add_node("preprocess", preprocess_node)
    graph.add_node("safety_gate", safety_gate_node)
    graph.add_node("agents_parallel", agents_parallel_node)
    graph.add_node("checker", checker_node)
    graph.add_node("writer", writer_node)
    graph.add_node("feedback_sink", feedback_sink_node)
    
    # Add edges
    graph.add_edge(START, "preprocess")
    graph.add_edge("preprocess", "safety_gate")
    graph.add_edge("safety_gate", "agents_parallel")
    graph.add_edge("agents_parallel", "checker")
    graph.add_edge("checker", "writer")
    graph.add_edge("writer", "feedback_sink")
    graph.add_edge("feedback_sink", END)
    
    return graph.compile()


def process_tweet(tweet: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    """Process a single tweet through the pipeline."""
    settings = get_settings()
    graph = build_graph()
    
    initial_state: State = {
        "tweet": tweet,
        "context": {},
        "results": {},
        "checked": {},
        "meta": {
            "model": settings.model_name,
            "prompt_version": settings.prompt_version,
            "run_id": run_id
        },
        "guardrails": {},
        "error": ""
    }
    
    start_time = time.time()
    final_state = graph.invoke(initial_state)
    elapsed = time.time() - start_time
    
    checked = final_state.get("checked", {}) or {}
    guardrails = final_state.get("guardrails", {}) or {}
    result = {
        "tweet_id": tweet.get("id"),
        "elapsed": elapsed,
        "checker_status": checked.get("checker_status"),
        "refused": guardrails.get("refused", False),
        "final_labels": checked.get("final"),
        "agent_results": final_state.get("results", {}),
        "guardrails_info": guardrails,
        "checker_trace": checked.get("a2a_trace")
    }
    error = final_state.get("error")
    if error:
        result["error"] = error
    return result

