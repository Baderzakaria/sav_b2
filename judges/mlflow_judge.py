from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Optional

import mlflow

try:
    from mlflow.genai.judges import make_judge as mlflow_make_judge
except ImportError:
    mlflow_make_judge = None

PROMPT_FILE = Path("prompts/freemind_prompts.json")
DEFAULT_LOG = Path("data/results/free tweet export_log.csv")
ISSUE_KEYWORDS = [
    "panne",
    "proble",
    "incident",
    "coupure",
    "rupture",
    "lent",
    "bug",
    "erreur",
    "service",
    "fail",
]


class SimpleJudge:
    """Fallback judge when MLflow GenAI judges are unavailable."""

    def __init__(self, name: str, instructions: str):
        self.name = name
        self.instructions = instructions

    @staticmethod
    def _contains_issue(text: str) -> bool:
        lower = (text or "").lower()
        return any(keyword in lower for keyword in ISSUE_KEYWORDS)

    def __call__(self, inputs: Optional[Dict] = None, outputs: Optional[Dict] = None, **_: Dict) -> Dict[str, str]:
        inputs = inputs or {}
        outputs = outputs or {}
        tweet = inputs.get("tweet", "")
        prediction = str(outputs.get("prediction", "")).lower()
        issue_present = self._contains_issue(tweet)

        if issue_present and prediction not in {"true", "problem", "issue"}:
            verdict = "fail"
            reason = "Tweet looks like an issue but classification flagged otherwise."
        elif not issue_present and prediction in {"true", "problem", "issue"}:
            verdict = "fail"
            reason = "Tweet appears promotional yet classified as an issue."
        elif not prediction or prediction == "unknown":
            verdict = "fail"
            reason = "Agent did not return a usable prediction."
        else:
            verdict = "pass"
            reason = "Classification aligns with heuristic keywords."

        return {"verdict": verdict, "reason": reason}


def make_judge(name: str, instructions: str):
    if mlflow_make_judge is not None:
        return mlflow_make_judge(name=name, instructions=instructions)
    return SimpleJudge(name=name, instructions=instructions)


def _load_judge_config(judge_key: str) -> Dict[str, str]:
    with PROMPT_FILE.open("r", encoding="utf-8") as fp:
        data = json.load(fp)
    judges = data.get("judges", {})
    judge_cfg = judges.get(judge_key)
    if not judge_cfg or "instructions" not in judge_cfg:
        raise KeyError(f"Judge '{judge_key}' not found in {PROMPT_FILE}")
    return judge_cfg


def build_judge(judge_key: str = "J_utile_consistency"):
    cfg = _load_judge_config(judge_key)
    return make_judge(
        name=cfg.get("name", judge_key),
        instructions=cfg["instructions"],
    )


def run_judge_on_log(
    log_csv: Path,
    output_csv: Optional[Path] = None,
    judge_key: str = "J_utile_consistency",
    tracking_uri: str = "file:./mlruns",
    experiment_name: str = "freemind_judges",
    start_run: bool = True,
) -> Path:
    log_csv = Path(log_csv)
    if output_csv is None:
        output_csv = log_csv.with_name(log_csv.stem.replace("_log", "_judge") + ".csv")
    else:
        output_csv = Path(output_csv)

    judge = build_judge(judge_key)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    judged_rows = []
    with log_csv.open("r", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            inputs = {
                "tweet": row.get("text_preview", ""),
                "raw_output": row.get("raw_output", ""),
            }
            outputs = {
                "prediction": row.get("mapped_utile") or row.get("extracted_value") or "unknown"
            }
            verdict = judge(inputs=inputs, outputs=outputs)
            judged_rows.append({
                **row,
                "judge_verdict": verdict,
                "judge_name": judge_key,
            })

    def _write_and_log():
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = judged_rows[0].keys() if judged_rows else []
        with output_csv.open("w", encoding="utf-8", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(judged_rows)

        mlflow.log_param("judge_log_csv", str(log_csv))
        mlflow.log_param("judge_key", judge_key)
        mlflow.log_artifact(str(output_csv), artifact_path="judge")

    if start_run or mlflow.active_run() is None:
        with mlflow.start_run(run_name=f"judge_{log_csv.stem}"):
            _write_and_log()
    else:
        _write_and_log()

    return output_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MLflow judge on pipeline logs.")
    parser.add_argument("--log-csv", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--judge-key", type=str, default="J_utile_consistency")
    parser.add_argument("--tracking-uri", type=str, default="file:./mlruns")
    parser.add_argument("--experiment-name", type=str, default="freemind_judges")
    return parser.parse_args()


def main():
    args = parse_args()
    result_path = run_judge_on_log(
        log_csv=args.log_csv,
        output_csv=args.output_csv,
        judge_key=args.judge_key,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        start_run=True,
    )
    print(f"Judge output saved to {result_path}")


if __name__ == "__main__":
    main()
"""
Utility helpers to run MLflow GenAI judges against pipeline outputs.

Usage:
    python -m judges.mlflow_judge \
        --log-csv data/results/free\ tweet\ export_log.csv \
        --output-csv data/results/free\ tweet\ export_judge.csv

Environment:
    Requires MLflow >= 3.1 and access to a GenAI provider supported by
    `mlflow.genai.judges.make_judge` (e.g., OpenAI). Configure the provider
    credentials via the usual environment variables before running.
"""


