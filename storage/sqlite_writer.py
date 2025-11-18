"""SQLite writer with upsert for tweets_enriched and review queue."""

import sqlite3
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from models.labels import Labels, CheckerOutput


class SQLiteWriter:
    """Handles all SQLite writes for the FreeMind pipeline."""
    
    def __init__(self, db_path: str = "data/freemind.db"):
        self.db_path = db_path
        self._ensure_db_exists()
    
    def _ensure_db_exists(self):
        """Create database and tables if they don't exist."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            schema_path = Path(__file__).parent / "schema.sql"
            with open(schema_path, "r", encoding="utf-8") as f:
                conn.executescript(f.read())
    
    def upsert_tweet(
        self,
        tweet: Dict[str, Any],
        context: Dict[str, str],
        checker_output: CheckerOutput,
        meta: Dict[str, Any],
        guardrails: Optional[Dict[str, Any]] = None
    ) -> None:
        """Upsert a tweet with all labels, context, and metadata."""
        final = checker_output.final
        affect = final.affect
        
        # Prepare guardrail data
        refused = guardrails.get("refused", False) if guardrails else False
        refused_reason = guardrails.get("reason", None) if guardrails else None
        guardrails_flags_json = json.dumps(guardrails.get("flags", {})) if guardrails else None
        
        row = {
            "id": tweet.get("id"),
            "created_at": tweet.get("created_at"),
            "full_text": tweet.get("full_text"),
            "screen_name": tweet.get("screen_name"),
            "name": tweet.get("name"),
            "user_id": tweet.get("user_id"),
            "in_reply_to": tweet.get("in_reply_to"),
            "retweeted_status": tweet.get("retweeted_status"),
            "quoted_status": tweet.get("quoted_status"),
            "url": tweet.get("url"),
            
            "ctx_before": context.get("ctx_before"),
            "ctx_after": context.get("ctx_after"),
            "ctx_refs": context.get("ctx_refs"),
            
            "utile": final.utile,
            "categorie": final.categorie,
            "sentiment": final.sentiment,
            "type_probleme": final.type_probleme,
            "score_gravite": final.score_gravite,
            
            "emotion_primary": affect.emotion_primary if affect else None,
            "sarcasm": affect.sarcasm if affect else None,
            "tone_color": affect.tone_color if affect else None,
            "toxicity": affect.toxicity if affect else None,
            
            "refused_by_guardrails": refused,
            "refused_reason": refused_reason,
            "guardrails_flags": guardrails_flags_json,
            
            "checker_status": checker_output.checker_status,
            "a2a_trace": json.dumps(checker_output.a2a_trace),
            "llm_model": meta.get("model"),
            "prompt_version": meta.get("prompt_version"),
            "run_id": meta.get("run_id"),
            "created_timestamp": time.time()
        }
        
        with sqlite3.connect(self.db_path) as conn:
            placeholders = ", ".join([f":{k}" for k in row.keys()])
            columns = ", ".join(row.keys())
            update_clause = ", ".join([f"{k}=excluded.{k}" for k in row.keys() if k != "id"])
            
            sql = f"""
            INSERT INTO tweets_enriched ({columns})
            VALUES ({placeholders})
            ON CONFLICT(id) DO UPDATE SET {update_clause}
            """
            conn.execute(sql, row)
            conn.commit()
    
    def enqueue_for_review(
        self,
        tweet_id: str,
        reason: str,
        original_labels: Dict[str, Any]
    ) -> None:
        """Add a tweet to the review queue for HITL."""
        # Skip if tweet_id is None or empty
        if not tweet_id:
            return
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO review_queue (tweet_id, reason, original_labels, created_timestamp)
                VALUES (?, ?, ?, ?)
                """,
                (tweet_id, reason, json.dumps(original_labels), time.time())
            )
            conn.commit()
    
    def log_feedback(
        self,
        tweet_id: str,
        feedback_type: str,
        original_labels: Dict[str, Any],
        corrected_labels: Optional[Dict[str, Any]] = None,
        feedback_source: str = "auto",
        notes: Optional[str] = None
    ) -> None:
        """Log feedback for continuous learning."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO feedback_log 
                (tweet_id, feedback_type, original_labels, corrected_labels, 
                 feedback_source, notes, created_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tweet_id,
                    feedback_type,
                    json.dumps(original_labels),
                    json.dumps(corrected_labels) if corrected_labels else None,
                    feedback_source,
                    notes,
                    time.time()
                )
            )
            conn.commit()


def init_database(db_path: str = "data/freemind.db") -> None:
    """Initialize the database with schema."""
    writer = SQLiteWriter(db_path)
    print(f"Database initialized at {db_path}")

