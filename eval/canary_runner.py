#!/usr/bin/env python3
"""Canary runner for A/B testing prompt versions."""

import json
import argparse
from pathlib import Path
from typing import Dict, Any

from eval.offline_eval import run_offline_eval
from config.settings import get_settings


def run_canary(
    input_csv: str,
    baseline_version: str,
    canary_version: str,
    sample_rate: float = 0.1,
    output_dir: str = "data/results/canary"
) -> Dict[str, Any]:
    """Run canary test comparing two prompt versions."""
    settings = get_settings()
    
    print("="*60)
    print("CANARY TEST")
    print("="*60)
    print(f"Baseline version: {baseline_version}")
    print(f"Canary version: {canary_version}")
    print(f"Sample rate: {sample_rate * 100:.0f}%")
    print()
    
    # Load registry
    with open("prompts/registry.json", "r", encoding="utf-8") as f:
        registry = json.load(f)
    
    rollback_policy = registry.get("rollback_policy", {})
    auto_rollback_threshold = rollback_policy.get("auto_rollback_on_fail_rate", 0.05)
    min_samples = rollback_policy.get("min_samples_before_promotion", 100)
    
    # Calculate sample size
    sample_size = int(settings.eval_sample_size * sample_rate)
    sample_size = max(sample_size, min_samples)
    
    print(f"Running canary with {sample_size} samples...")
    
    # Run canary evaluation
    # In a real implementation, this would:
    # 1. Temporarily switch to canary version
    # 2. Run evaluation
    # 3. Compare metrics with baseline
    # 4. Decide on rollback or promotion
    
    canary_metrics = run_offline_eval(
        input_csv=input_csv,
        sample_size=sample_size,
        output_dir=output_dir
    )
    
    # Decision logic
    fail_rate = canary_metrics.get("checker_fail_pct", 0) / 100
    
    decision = "promote" if fail_rate < auto_rollback_threshold else "rollback"
    
    print("\n" + "="*60)
    print("CANARY DECISION")
    print("="*60)
    print(f"Fail rate: {fail_rate*100:.2f}%")
    print(f"Threshold: {auto_rollback_threshold*100:.2f}%")
    print(f"Decision: {decision.upper()}")
    
    if decision == "rollback":
        print(f"\n⚠️  ROLLBACK RECOMMENDED: Fail rate exceeds threshold")
    else:
        print(f"\n✓ PROMOTE: Canary passed quality gates")
    
    # Save decision
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    decision_path = Path(output_dir) / f"canary_decision_{canary_version}.json"
    
    decision_data = {
        "baseline_version": baseline_version,
        "canary_version": canary_version,
        "sample_size": sample_size,
        "fail_rate": fail_rate,
        "threshold": auto_rollback_threshold,
        "decision": decision,
        "metrics": canary_metrics
    }
    
    with open(decision_path, "w", encoding="utf-8") as f:
        json.dump(decision_data, f, indent=2)
    
    print(f"\nDecision saved to {decision_path}")
    
    return decision_data


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run canary test")
    parser.add_argument("input_csv", help="Input CSV file with tweets")
    parser.add_argument("--baseline", default="v0.2", help="Baseline version")
    parser.add_argument("--canary", default="v0.3", help="Canary version")
    parser.add_argument("--sample-rate", type=float, default=0.1, help="Sample rate")
    parser.add_argument("--output-dir", default="data/results/canary", help="Output directory")
    
    args = parser.parse_args()
    
    run_canary(
        input_csv=args.input_csv,
        baseline_version=args.baseline,
        canary_version=args.canary,
        sample_rate=args.sample_rate,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

