-- FreeMind database schema with A2A tracing, affect, and guardrails

CREATE TABLE IF NOT EXISTS tweets_enriched (
  id TEXT PRIMARY KEY,
  created_at TEXT,
  full_text TEXT,              -- intact, source of truth
  screen_name TEXT,
  name TEXT,
  user_id TEXT,
  in_reply_to TEXT,
  retweeted_status TEXT,
  quoted_status TEXT,
  url TEXT,

  -- Context (non-destructive)
  ctx_before TEXT,
  ctx_after TEXT,
  ctx_refs TEXT,

  -- Labels finaux (checker)
  utile BOOLEAN,
  categorie TEXT,
  sentiment TEXT,
  type_probleme TEXT,
  score_gravite INTEGER,

  -- Affect (new)
  emotion_primary TEXT,
  sarcasm BOOLEAN,
  tone_color TEXT,
  toxicity TEXT,

  -- Guardrails (new)
  refused_by_guardrails BOOLEAN DEFAULT 0,
  refused_reason TEXT,
  guardrails_flags TEXT,        -- JSON

  -- Traçabilité
  checker_status TEXT,          -- ok|warn|fail
  a2a_trace TEXT,               -- JSON
  llm_model TEXT,
  prompt_version TEXT,
  run_id TEXT,
  created_timestamp REAL
);

CREATE INDEX IF NOT EXISTS idx_labels ON tweets_enriched(categorie, sentiment, type_probleme);
CREATE INDEX IF NOT EXISTS idx_checker_status ON tweets_enriched(checker_status);
CREATE INDEX IF NOT EXISTS idx_guardrails ON tweets_enriched(refused_by_guardrails);
CREATE INDEX IF NOT EXISTS idx_run_id ON tweets_enriched(run_id);

-- Review queue for HITL
CREATE TABLE IF NOT EXISTS review_queue (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tweet_id TEXT NOT NULL,
  reason TEXT,                  -- fail|warn|guardrail_refuse
  original_labels TEXT,         -- JSON
  corrected_labels TEXT,        -- JSON (filled by human)
  reviewed BOOLEAN DEFAULT 0,
  reviewer TEXT,
  review_timestamp REAL,
  created_timestamp REAL,
  FOREIGN KEY (tweet_id) REFERENCES tweets_enriched(id)
);

CREATE INDEX IF NOT EXISTS idx_review_queue_reviewed ON review_queue(reviewed);
CREATE INDEX IF NOT EXISTS idx_review_queue_tweet ON review_queue(tweet_id);

-- Feedback for continuous learning
CREATE TABLE IF NOT EXISTS feedback_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  tweet_id TEXT NOT NULL,
  feedback_type TEXT,           -- correction|validation|flag
  original_labels TEXT,         -- JSON
  corrected_labels TEXT,        -- JSON
  feedback_source TEXT,         -- hitl|auto|eval
  notes TEXT,
  created_timestamp REAL,
  FOREIGN KEY (tweet_id) REFERENCES tweets_enriched(id)
);

CREATE INDEX IF NOT EXISTS idx_feedback_tweet ON feedback_log(tweet_id);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback_log(feedback_type);

