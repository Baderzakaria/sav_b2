import os
from typing import Any, Dict, Optional

import requests


DEFAULT_OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL",
    "meta-llama/llama-3.2-3b-instruct:free",
)


def call_native_generate(ollama_host: str, model: str, prompt: str, options: Optional[Dict[str, Any]] = None) -> str:
    url = f"{ollama_host}/api/generate"
    payload: Dict[str, Any] = {"model": model, "prompt": prompt, "stream": False}
    if options:
        payload.update({k: v for k, v in options.items() if v is not None})
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json().get("response", "")


def call_openrouter(prompt: str, model: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    payload: Dict[str, Any] = {
        "model": model or OPENROUTER_MODEL,
        "messages": [
            {"role": "system", "content": options.get("system_message") if options else None},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    # Remove None system message entries
    payload["messages"] = [msg for msg in payload["messages"] if msg["content"]]

    if options:
        for key in ["temperature", "top_p", "max_tokens"]:
            if key in options and options[key] is not None:
                payload[key] = options[key]

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        # Optional metadata so the app shows up in leaderboards/logs:
        "HTTP-Referer": os.environ.get("OPENROUTER_REFERRER", "https://novasolve.zakaa.tech"),
        "X-Title": os.environ.get("OPENROUTER_APP_NAME", "FreeMind Orchestrator"),
    }

    resp = requests.post(f"{OPENROUTER_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def call_llm(prompt: str, model: str, *, host: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> str:
    """
    Generic LLM caller that prefers OpenRouter when an API key is provided,
    otherwise falls back to a local Ollama host.
    """

    if OPENROUTER_API_KEY:
        return call_openrouter(prompt, model=model, options=options)

    return call_native_generate(host or DEFAULT_OLLAMA_HOST, model, prompt, options=options)

