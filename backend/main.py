import sys
import os
import asyncio
import csv
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Generator, List, Optional

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

# Add parent directory to path so we can import orchestrator and other modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from orchestrator import PipelineConfig, run_pipeline, build_pipeline, load_prompts

app = FastAPI()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LATEST_RESULTS_FILE = PROJECT_ROOT / "data" / "results" / "freemind_log_latest.csv"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn")
MAX_INTERFACE_SAMPLES = int(os.environ.get("INTERFACE_TEST_MAX", "20"))


class InterfaceSample(BaseModel):
    id: Optional[str] = None
    text: str


class InterfaceTestRequest(BaseModel):
    samples: List[InterfaceSample]
    model: Optional[str] = None
    temperature: Optional[float] = None


def _determine_severity(ai_gravity: Optional[str]) -> str:
    """Approximate urgency bucket from AI gravity score."""
    try:
        score = int(float(ai_gravity or 0))
    except (ValueError, TypeError):
        return "faible"

    if score <= -7:
        return "critique"
    if score < 0:
        return "élevée"
    if score > 3:
        return "faible"
    return "moyenne"


def _parse_int(value: Optional[str]) -> int:
    try:
        return int(float(value))  # Handles "7.0" as well
    except (TypeError, ValueError):
        return 0


def _parse_bool(value: Optional[str]) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"true", "1", "yes"}


def _row_to_ticket(row: Dict[str, str]) -> Dict[str, Any]:
    ai_gravity = row.get("Final_gravity") or row.get("A5_gravity") or "0"
    category = row.get("Final_categorie") or row.get("A2_categorie") or "autre"
    sentiment = row.get("Final_sentiment") or row.get("A3_sentiment") or "neutre"
    ticket_type = row.get("A4_type") or "autre"
    utile = _parse_bool(row.get("Final_utile")) or _parse_bool(row.get("A1_utile"))

    description = row.get("full_text") or "Pas de contenu"
    subject = f"{category.upper()}: {description[:40]}..."

    return {
        "id": row.get("tweet_id") or f"csv-{row.get('row_index')}",
        "source": row.get("source") or "twitter",
        "subject": subject,
        "description": description,
        "clean_text": row.get("clean_text"),
        "customer": f"@{row['screen_name']}" if row.get("screen_name") else "Anonyme",
        "created_at": row.get("date_iso") or row.get("timestamp") or "",
        "status": "nouveau",
        "severity": _determine_severity(ai_gravity),
        "channel": "Twitter",
        "type": ticket_type,
        "sentiment": sentiment,
        "categorie": category,
        "gravity": _parse_int(ai_gravity),
        "favorite_count": _parse_int(row.get("favorite_count")),
        "reply_count": _parse_int(row.get("reply_count")),
        "utile": utile,
        "agentResponses": {
            "utile": row.get("A1_utile"),
            "categorie": row.get("A2_categorie"),
            "sentiment": row.get("A3_sentiment"),
            "type": row.get("A4_type"),
            "gravity": row.get("A5_gravity"),
        },
    }


@app.get("/tickets/latest")
async def get_latest_tickets(limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Return the latest orchestrator output from the CSV file so the dashboard
    has data on load.
    """
    if not LATEST_RESULTS_FILE.exists():
        raise HTTPException(status_code=404, detail="Latest results file not found")

    try:
        with LATEST_RESULTS_FILE.open("r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            tickets: List[Dict[str, Any]] = []
            for row in reader:
                tickets.append(_row_to_ticket(row))
                if limit and len(tickets) >= limit:
                    break

        return {
            "source": str(LATEST_RESULTS_FILE.name),
            "count": len(tickets),
            "tickets": tickets,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to load latest tickets: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to read latest results")

@app.get("/run-orchestrator")
async def run_orchestrator(request: Request):
    """
    Runs the orchestrator pipeline and streams the logs back to the client via SSE.
    Does NOT save results to disk (save_to_disk=False).
    """
    
    async def event_generator():
        queue = asyncio.Queue()
        stop_event = asyncio.Event()

        def progress_callback(entry: Dict[str, Any]):
            # Put the entry into the queue
            # We use a non-async wrapper to put into async queue via run_coroutine_threadsafe 
            # or just assume the loop is running. 
            # Since orchestrator is blocking, we should run it in a separate thread.
            asyncio.run_coroutine_threadsafe(queue.put(entry), loop)

        loop = asyncio.get_running_loop()
        
        # Configuration for the run
        config = PipelineConfig(
            save_to_disk=False,
            enable_live_log=False, # We stream instead
            max_rows=10, # Limit rows for testing, or maybe make it a parameter
            # We can accept query params to override config if needed
        )

        # Run pipeline in a separate thread to avoid blocking the event loop
        def run_in_thread():
            try:
                run_pipeline(
                    config=config,
                    progress_callback=progress_callback,
                    stop_signal=lambda: stop_event.is_set()
                )
            except Exception as e:
                logger.error(f"Error in pipeline: {e}")
                asyncio.run_coroutine_threadsafe(queue.put({"error": str(e)}), loop)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop) # Sentinel to stop

        import threading
        thread = threading.Thread(target=run_in_thread)
        thread.start()

        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                stop_event.set()
                break

            data = await queue.get()
            if data is None:
                break
            
            yield {
                "event": "log",
                "data": json.dumps(data)
            }

    return EventSourceResponse(event_generator())


def _execute_interface_tests(payload: InterfaceTestRequest) -> Dict[str, Any]:
    if not payload.samples:
        return {"model": payload.model, "count": 0, "results": []}

    if len(payload.samples) > MAX_INTERFACE_SAMPLES:
        raise HTTPException(
            status_code=400,
            detail=f"Maximum {MAX_INTERFACE_SAMPLES} samples allowed per request.",
        )

    base_config = PipelineConfig()
    config = PipelineConfig(
        save_to_disk=False,
        enable_live_log=False,
        enable_model_warmup=False,
        model_name=payload.model or base_config.model_name,
        ollama_options={"temperature": payload.temperature} if payload.temperature else {},
    )

    prompts = load_prompts()
    graph = build_pipeline(prompts, config.model_name, config.ollama_host, config.ollama_options)

    results: List[Dict[str, Any]] = []

    for sample in payload.samples:
        clean_text = (sample.text or "").strip()
        if not clean_text:
            results.append(
                {
                    "id": sample.id,
                    "text": sample.text,
                    "error": "Text is empty",
                }
            )
            continue

        started = time.perf_counter()
        try:
            output = graph.invoke({"clean_text": clean_text})
            elapsed_ms = (time.perf_counter() - started) * 1000
            final = output.get("final_json", {})
            results.append(
                {
                    "id": sample.id,
                    "text": clean_text,
                    "latency_ms": round(elapsed_ms, 1),
                    "final": final,
                    "agents": {
                        "A1": output.get("a1_result"),
                        "A2": output.get("a2_result"),
                        "A3": output.get("a3_result"),
                        "A4": output.get("a4_result"),
                        "A5": output.get("a5_result"),
                    },
                }
            )
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.exception("Interface test failed for sample %s", sample.id or clean_text[:32])
            results.append(
                {
                    "id": sample.id,
                    "text": clean_text,
                    "latency_ms": round(elapsed_ms, 1),
                    "error": str(exc),
                }
            )

    return {
        "model": config.model_name,
        "count": len(results),
        "results": results,
    }


@app.post("/interface-tests")
async def interface_tests(request: InterfaceTestRequest) -> Dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, lambda: _execute_interface_tests(request))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
