import requests


def call_native_generate(ollama_host: str, model: str, prompt: str) -> str:
    url = f"{ollama_host}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json().get("response", "")

