import requests


def call_native_generate(ollama_host: str, model: str, prompt: str) -> str:
    """Call Ollama, supporting both /api/generate and /api/chat.

    Tries /api/generate first for compatibility; on 404 falls back to /api/chat.
    Returns best-effort text from either response shape.
    """

    generate_url = f"{ollama_host}/api/generate"
    generate_payload = {"model": model, "prompt": prompt, "stream": False}

    try:
        resp = requests.post(generate_url, json=generate_payload, timeout=60)
        if resp.status_code == 404:
            raise requests.HTTPError("Not Found", response=resp)
        resp.raise_for_status()
        data = resp.json()
        # Prefer classic generate shape; fallback to chat-like just in case
        return data.get("response", data.get("message", {}).get("content", ""))
    except requests.HTTPError as http_err:
        # Fallback to chat endpoint for servers without /api/generate
        if getattr(http_err, "response", None) is None or http_err.response.status_code != 404:
            raise

        chat_url = f"{ollama_host}/api/chat"
        chat_payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        chat_resp = requests.post(chat_url, json=chat_payload, timeout=60)
        chat_resp.raise_for_status()
        chat_data = chat_resp.json()
        # Chat response shape
        if isinstance(chat_data.get("message"), dict):
            return chat_data["message"].get("content", "")
        return chat_data.get("response", "")



