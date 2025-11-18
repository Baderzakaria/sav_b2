"""Feedback collector for continuous learning loop."""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional


class FeedbackCollector:
    """Collects feedback for continuous learning."""
    
    def __init__(self, output_dir: str = "data/feedback"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.training_set_path = self.output_dir / "training_set.jsonl"
    
    def collect(
        self,
        tweet_id: str,
        full_text: str,
        context: Dict[str, str],
        labels: Dict[str, Any],
        checker_status: str,
        a2a_trace: Dict[str, Any],
        guardrails: Dict[str, Any],
        source: str = "auto"
    ) -> None:
        """Collect a feedback sample."""
        
        # Curation policy: exclude refused/unsafe samples
        if guardrails.get("refused"):
            return
        
        # Only collect warn/fail cases or random ok samples for diversity
        if checker_status == "ok":
            # Could add random sampling here for diversity
            return
        
        feedback_sample = {
            "tweet_id": tweet_id,
            "full_text": full_text,
            "context": context,
            "labels": labels,
            "checker_status": checker_status,
            "a2a_trace": a2a_trace,
            "source": source,
            "timestamp": time.time()
        }
        
        # Append to training set
        with open(self.training_set_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_sample, ensure_ascii=False) + "\n")
    
    def curate_training_set(
        self,
        min_samples: int = 1000,
        output_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Curate training set for retraining readiness."""
        if not self.training_set_path.exists():
            return {"status": "no_data", "count": 0}
        
        # Load all samples
        samples = []
        seen_ids = set()
        
        with open(self.training_set_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    tweet_id = sample.get("tweet_id")
                    
                    # Deduplicate by ID
                    if tweet_id not in seen_ids:
                        samples.append(sample)
                        seen_ids.add(tweet_id)
                except Exception:
                    continue
        
        # Check if ready for retraining
        ready = len(samples) >= min_samples
        
        if ready and output_path:
            # Write curated set
            curated_path = Path(output_path)
            curated_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(curated_path, "w", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        
        return {
            "status": "ready" if ready else "collecting",
            "count": len(samples),
            "min_required": min_samples,
            "curated_path": output_path if ready else None
        }

