#!/usr/bin/env python3

import argparse
import json
import os
from typing import Any, Dict

import pandas as pd
from tqdm import tqdm

import requests


def build_prompt(text_value: str) -> str:
    return (
        "Analyze the sentiment of the following tweet. Return only JSON with key "
        "'sentiment' (positive/neutral/negative).\n\nTweet:\n" + text_value
    )


def extract_text_from_row(row: Dict[str, Any]) -> str:
    text_value = row.get("full_text")
    if isinstance(text_value, str) and text_value.strip():
        return text_value
    # Fallback: concatenate non-empty values
    values = [str(v) for v in row.values() if v is not None and str(v).strip()]
    return " ".join(values)


def parse_sentiment(model_output: str) -> str:
    try:
        data = json.loads(model_output)
        sentiment = data.get("sentiment")
        if isinstance(sentiment, str):
            return sentiment
    except Exception:
        pass
    # Fallback to raw string if not JSON or missing key
    return model_output


def call_native_generate(ollama_host: str, model: str, prompt: str) -> str:
    url = f"{ollama_host}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json().get("response", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Append sentiment column and save to Excel")
    parser.add_argument("--input", dest="input_csv", default=os.environ.get("CSV_PATH"), help="Path to input CSV")
    parser.add_argument("--output", dest="output_xlsx", default=None, help="Path to output XLSX")
    parser.add_argument("--model", dest="model_name", default="mistral:7b", help="Ollama model name")
    parser.add_argument(
        "--host",
        dest="ollama_host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama host URL",
    )
    args = parser.parse_args()

    if not args.input_csv:
        raise SystemExit("--input or CSV_PATH env var is required")

    # Determine default output path if not provided
    if args.output_xlsx is None:
        base, _ = os.path.splitext(args.input_csv)
        args.output_xlsx = f"{base}_with_sentiment.xlsx"

    df = pd.read_csv(args.input_csv)

    sentiments: list[str] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring sentiment"):
        row_dict = row.to_dict()
        text_value = extract_text_from_row(row_dict)
        prompt = build_prompt(text_value)
        model_output = call_native_generate(args.ollama_host, args.model_name, prompt)
        sentiments.append(parse_sentiment(model_output))

    # Append new column at the end
    df["sentiment"] = sentiments

    # Ensure the new column is the last column
    df = df[[c for c in df.columns if c != "sentiment"] + ["sentiment"]]

    # Write to Excel
    df.to_excel(args.output_xlsx, index=False)
    print(f"Saved: {args.output_xlsx}")


if __name__ == "__main__":
    main()


