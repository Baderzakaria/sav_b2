#!/usr/bin/env python3
"""CLI runner for FreeMind pipeline."""

import csv
import argparse
import time
import json
from pathlib import Path
from typing import List, Dict, Any

from orchestrator import process_tweet, get_run_id
from config.settings import get_settings
from storage.sqlite_writer import init_database


STATUS_DESCRIPTIONS = {
    "ok": "Passed all checker rules",
    "warn": "Minor issues detected; flagged for review",
    "fail": "Checker failed or guardrails refused; human review needed",
    None: "No checker decision returned"
}

def load_tweets_from_csv(csv_path: str, max_rows: int = None) -> List[Dict[str, Any]]:
    """Load tweets from CSV file."""
    tweets = []
    # Use utf-8-sig to safely strip potential BOM characters from headers (e.g., '\ufeffid')
    with open(csv_path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            tweets.append(row)
            if max_rows and i + 1 >= max_rows:
                break
    return tweets


def run_pipeline(
    input_csv: str,
    max_rows: int = None,
    output_report: str = None
) -> Dict[str, Any]:
    """Run the full pipeline on a CSV of tweets."""
    settings = get_settings()
    
    # Initialize database
    init_database(settings.db_path)
    
    # Load tweets
    print(f"Loading tweets from {input_csv}...")
    tweets = load_tweets_from_csv(input_csv, max_rows)
    print(f"Loaded {len(tweets)} tweets")
    
    # Generate run ID
    run_id = get_run_id()
    print(f"Run ID: {run_id}")
    
    # Process tweets
    results = []
    start_time = time.time()
    
    for i, tweet in enumerate(tweets, 1):
        tweet_id = tweet.get("id") or "unknown"
        text_preview = (tweet.get("full_text") or "").strip().replace("\n", " ")
        if len(text_preview) > 80:
            text_preview = text_preview[:77] + "..."
        display_label = text_preview or tweet_id
        author = tweet.get("screen_name") or tweet.get("name") or "unknown"
        full_text = (tweet.get("full_text") or "").strip()
        print(f"\n[{i}/{len(tweets)}] Processing: {display_label}")
        print(f"  Author: {author}")
        if full_text:
            formatted_text = full_text.replace("\n", "\n    ")
            print("  Full tweet:")
            print(f"    {formatted_text}")
        try:
            result = process_tweet(tweet, run_id)
            results.append(result)
            status = result.get("checker_status")
            status_desc = STATUS_DESCRIPTIONS.get(status, STATUS_DESCRIPTIONS[None])
            print(f"  Status: {status or 'unknown'} ({status_desc})")
            print(f"  Time: {result.get('elapsed', 0):.2f}s | Refused: {result.get('refused', False)}")
            
            final_labels = result.get("final_labels")
            if final_labels:
                affect = final_labels.get("affect") or {}
                affect_display = ", ".join(f"{k}={v}" for k, v in affect.items()) if affect else "none"
                print("  Final labels:")
                print(f"    utile={final_labels.get('utile')} | categorie={final_labels.get('categorie')} | sentiment={final_labels.get('sentiment')}")
                print(f"    type_probleme={final_labels.get('type_probleme')} | score_gravite={final_labels.get('score_gravite')}")
                print(f"    affect={affect_display}")
            if result.get("checker_trace"):
                trace_preview = json.dumps(result["checker_trace"], ensure_ascii=False)
                if len(trace_preview) > 200:
                    trace_preview = trace_preview[:197] + "..."
                print(f"  Checker trace: {trace_preview}")
            if result.get("agent_results"):
                print("  Agent outputs:")
                for agent_name, agent_output in result["agent_results"].items():
                    preview = json.dumps(agent_output, ensure_ascii=False)
                    if len(preview) > 200:
                        preview = preview[:197] + "..."
                    print(f"    {agent_name}: {preview}")
            if result.get("error"):
                print(f"  Pipeline error: {result['error']}")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({
                "tweet_id": tweet.get("id"),
                "error": str(e),
                "checker_status": "fail"
            })
    
    total_time = time.time() - start_time
    
    # Compute metrics
    metrics = compute_metrics(results, total_time)
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE SUMMARY")
    print("="*60)
    print(f"Total tweets: {metrics['total']}")
    print(f"Total time: {metrics['total_time']:.2f}s")
    print(f"Avg time/tweet: {metrics['avg_time']:.2f}s")
    print(f"\nChecker status:")
    print(f"  OK: {metrics['checker_ok']} ({metrics['checker_ok_pct']:.1f}%)")
    print(f"  WARN: {metrics['checker_warn']} ({metrics['checker_warn_pct']:.1f}%)")
    print(f"  FAIL: {metrics['checker_fail']} ({metrics['checker_fail_pct']:.1f}%)")
    print(f"\nGuardrails:")
    print(f"  Refused: {metrics['refused']} ({metrics['refused_pct']:.1f}%)")
    print(f"\nErrors: {metrics['errors']}")
    
    # Save report if requested
    if output_report:
        save_report(metrics, results, output_report)
        print(f"\nReport saved to {output_report}")
    
    return metrics


def compute_metrics(results: List[Dict[str, Any]], total_time: float) -> Dict[str, Any]:
    """Compute pipeline metrics."""
    total = len(results)
    
    checker_ok = sum(1 for r in results if r.get("checker_status") == "ok")
    checker_warn = sum(1 for r in results if r.get("checker_status") == "warn")
    checker_fail = sum(1 for r in results if r.get("checker_status") == "fail")
    refused = sum(1 for r in results if r.get("refused"))
    errors = sum(1 for r in results if "error" in r)
    
    times = [r.get("elapsed", 0) for r in results if "elapsed" in r]
    avg_time = sum(times) / len(times) if times else 0
    
    return {
        "total": total,
        "total_time": total_time,
        "avg_time": avg_time,
        "checker_ok": checker_ok,
        "checker_ok_pct": (checker_ok / total * 100) if total > 0 else 0,
        "checker_warn": checker_warn,
        "checker_warn_pct": (checker_warn / total * 100) if total > 0 else 0,
        "checker_fail": checker_fail,
        "checker_fail_pct": (checker_fail / total * 100) if total > 0 else 0,
        "refused": refused,
        "refused_pct": (refused / total * 100) if total > 0 else 0,
        "errors": errors
    }


def save_report(metrics: Dict[str, Any], results: List[Dict[str, Any]], output_path: str):
    """Save metrics report to file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("FreeMind Pipeline Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total tweets: {metrics['total']}\n")
        f.write(f"Total time: {metrics['total_time']:.2f}s\n")
        f.write(f"Avg time/tweet: {metrics['avg_time']:.2f}s\n\n")
        f.write(f"Checker status:\n")
        f.write(f"  OK: {metrics['checker_ok']} ({metrics['checker_ok_pct']:.1f}%)\n")
        f.write(f"  WARN: {metrics['checker_warn']} ({metrics['checker_warn_pct']:.1f}%)\n")
        f.write(f"  FAIL: {metrics['checker_fail']} ({metrics['checker_fail_pct']:.1f}%)\n\n")
        f.write(f"Guardrails:\n")
        f.write(f"  Refused: {metrics['refused']} ({metrics['refused_pct']:.1f}%)\n\n")
        f.write(f"Errors: {metrics['errors']}\n\n")
        
        f.write("\nPer-tweet results:\n")
        f.write("-" * 60 + "\n")
        for r in results:
            f.write(f"Tweet {r.get('tweet_id')}: {r.get('checker_status', 'unknown')} "
                   f"({r.get('elapsed', 0):.2f}s)\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Run FreeMind pipeline")
    parser.add_argument("input_csv", help="Input CSV file with tweets")
    parser.add_argument("--max-rows", type=int, help="Maximum rows to process")
    parser.add_argument("--output-report", help="Output report file path")
    
    args = parser.parse_args()
    
    run_pipeline(
        input_csv=args.input_csv,
        max_rows=args.max_rows,
        output_report=args.output_report
    )


if __name__ == "__main__":
    main()

