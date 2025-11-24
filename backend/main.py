import sys
import os
import asyncio
import json
import logging
from typing import Dict, Any, Generator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

# Add parent directory to path so we can import orchestrator and other modules
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from orchestrator import PipelineConfig, run_pipeline

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger("uvicorn")

@app.get("/health")
def health_check():
    return {"status": "ok"}

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
