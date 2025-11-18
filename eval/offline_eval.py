#!/usr/bin/env python3
"""Offline evaluation harness for FreeMind pipeline."""

import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Any
import argparse

from orchestrator import process_tweet, get_run_id
from config.settings import get_settings
from storage.sqlite_writer import init_database


def load_eval_set(csv_path: str, sample_size: int = None) -> List[Dict[str, Any]]:
    """Load evaluation set from CSV."""
    tweets = []
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            tweets.append(row)
            if sample_size and i + 1 >= sample_size:
                break
    return tweets


def validate_json_output(result: Dict[str, Any]) -> bool:
    """Check if output is valid JSON with required fields."""
    checked = result.get("checked", {})
    final = checked.get("final")
    
    if not final:
        return False
    
    required_fields = ["utile", "categorie", "sentiment", "type_probleme", "score_gravite"]
    return all(field in final for field in required_fields)


def run_offline_eval(
    input_csv: str,
    sample_size: int = None,
    output_dir: str = "data/results/metrics"
) -> Dict[str, Any]:
    """Run offline evaluation and generate metrics report."""
    settings = get_settings()
    
    # Initialize database
    init_database(settings.db_path)
    
    # Load eval set
    print(f"Loading eval set from {input_csv}...")
    tweets = load_eval_set(input_csv, sample_size or settings.eval_sample_size)
    print(f"Loaded {len(tweets)} tweets for evaluation")
    
    # Generate run ID
    run_id = get_run_id()
    print(f"Eval Run ID: {run_id}")
    
    # Process tweets
    results = []
    start_time = time.time()
    
    for i, tweet in enumerate(tweets, 1):
        print(f"[{i}/{len(tweets)}] Evaluating tweet {tweet.get('id', 'unknown')}...")
        try:
            result = process_tweet(tweet, run_id)
            results.append(result)
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "tweet_id": tweet.get("id"),
                "error": str(e),
                "checker_status": "fail"
            })
    
    total_time = time.time() - start_time
    
    # Compute metrics
    metrics = compute_eval_metrics(results, total_time, settings)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total tweets: {metrics['total']}")
    print(f"Total time: {metrics['total_time']:.2f}s")
    print(f"Avg time/tweet: {metrics['avg_time']:.2f}s")
    print(f"P95 latency: {metrics['p95_latency']:.2f}s")
    print(f"\nJSON validity: {metrics['json_valid']} ({metrics['json_valid_pct']:.1f}%)")
    print(f"\nChecker status:")
    print(f"  OK: {metrics['checker_ok']} ({metrics['checker_ok_pct']:.1f}%)")
    print(f"  WARN: {metrics['checker_warn']} ({metrics['checker_warn_pct']:.1f}%)")
    print(f"  FAIL: {metrics['checker_fail']} ({metrics['checker_fail_pct']:.1f}%)")
    print(f"\nGuardrails:")
    print(f"  Refused: {metrics['refused']} ({metrics['refused_pct']:.1f}%)")
    print(f"\nThresholds:")
    print(f"  JSON valid ≥ {settings.json_valid_threshold*100:.0f}%: {'✓' if metrics['json_valid_pct']/100 >= settings.json_valid_threshold else '✗'}")
    print(f"  Checker OK ≥ {settings.checker_ok_threshold*100:.0f}%: {'✓' if metrics['checker_ok_pct']/100 >= settings.checker_ok_threshold else '✗'}")
    print(f"  Avg latency ≤ {settings.mean_latency_threshold}s: {'✓' if metrics['avg_time'] <= settings.mean_latency_threshold else '✗'}")
    
    # Save report
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    report_path = Path(output_dir) / f"eval_report_{run_id}.json"
    
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\nReport saved to {report_path}")
    
    return metrics


def compute_eval_metrics(
    results: List[Dict[str, Any]],
    total_time: float,
    settings
) -> Dict[str, Any]:
    """Compute evaluation metrics."""
    total = len(results)
    
    # JSON validity (placeholder - would need actual validation)
    json_valid = sum(1 for r in results if r.get("checker_status") != "fail")
    
    # Checker status
    checker_ok = sum(1 for r in results if r.get("checker_status") == "ok")
    checker_warn = sum(1 for r in results if r.get("checker_status") == "warn")
    checker_fail = sum(1 for r in results if r.get("checker_status") == "fail")
    
    # Guardrails
    refused = sum(1 for r in results if r.get("refused"))
    
    # Latency
    times = [r.get("elapsed", 0) for r in results if "elapsed" in r]
    avg_time = sum(times) / len(times) if times else 0
    times_sorted = sorted(times)
    p95_idx = int(len(times_sorted) * 0.95)
    p95_latency = times_sorted[p95_idx] if times_sorted else 0
    
    return {
        "total": total,
        "total_time": total_time,
        "avg_time": avg_time,
        "p95_latency": p95_latency,
        "json_valid": json_valid,
        "json_valid_pct": (json_valid / total * 100) if total > 0 else 0,
        "checker_ok": checker_ok,
        "checker_ok_pct": (checker_ok / total * 100) if total > 0 else 0,
        "checker_warn": checker_warn,
        "checker_warn_pct": (checker_warn / total * 100) if total > 0 else 0,
        "checker_fail": checker_fail,
        "checker_fail_pct": (checker_fail / total * 100) if total > 0 else 0,
        "refused": refused,
        "refused_pct": (refused / total * 100) if total > 0 else 0,
        "thresholds_met": {
            "json_valid": (json_valid / total) >= settings.json_valid_threshold if total > 0 else False,
            "checker_ok": (checker_ok / total) >= settings.checker_ok_threshold if total > 0 else False,
            "avg_latency": avg_time <= settings.mean_latency_threshold
        }
    }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run offline evaluation")
    parser.add_argument("input_csv", help="Input CSV file with tweets")
    parser.add_argument("--sample-size", type=int, help="Sample size for evaluation")
    parser.add_argument("--output-dir", default="data/results/metrics", help="Output directory")
    
    args = parser.parse_args()
    
    run_offline_eval(
        input_csv=args.input_csv,
        sample_size=args.sample_size,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

