import requests
from typing import Any, Dict, Optional

def call_native_generate(
    ollama_host: str,
    model: str,
    prompt: str,
    *,
    options: Optional[Dict[str, Any]] = None,
    timeout: int = 60,
) -> str:

    generate_url = f"{ollama_host}/api/generate"
    generate_payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if options:
        generate_payload["options"] = options

    try:
        resp = requests.post(generate_url, json=generate_payload, timeout=timeout)
        if resp.status_code == 404:
            raise requests.HTTPError("Not Found", response=resp)
        resp.raise_for_status()
        data = resp.json()

        return data.get("response", data.get("message", {}).get("content", ""))
    except requests.HTTPError as http_err:

        if getattr(http_err, "response", None) is None or http_err.response.status_code != 404:
            raise

        chat_url = f"{ollama_host}/api/chat"
        chat_payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if options:
            chat_payload["options"] = options
        chat_resp = requests.post(chat_url, json=chat_payload, timeout=timeout)
        chat_resp.raise_for_status()
        chat_data = chat_resp.json()

        if isinstance(chat_data.get("message"), dict):
            return chat_data["message"].get("content", "")
        return chat_data.get("response", "")


def call_openrouter_chat(
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    *,
    timeout: int = 60,
    headers: Optional[Dict[str, str]] = None,
) -> str:
    """
    Invoke OpenRouter's OpenAI-compatible Chat Completions endpoint.
    """
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY is required for OpenRouter calls.")

    url = f"{base_url.rstrip('/')}/chat/completions"
    payload: Dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    request_headers: Dict[str, str] = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if headers:
        # remove empty header values
        request_headers.update({k: v for k, v in headers.items() if v})

    resp = requests.post(url, json=payload, headers=request_headers, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("OpenRouter response missing choices field.")
    first = choices[0] or {}
    message = first.get("message") or {}
    content = message.get("content") or first.get("text")
    if not content:
        raise RuntimeError("OpenRouter response missing message content.")
    return content

