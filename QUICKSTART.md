# FreeMind Quick Start Guide

## Prerequisites Check

```bash
# 1. Check Python version (need 3.9+)
python --version

# 2. Check Ollama is running
curl http://localhost:11434/api/tags

# 3. Verify model is available
ollama list | grep llama3.2:3b
```

If Ollama is not running or model is missing:

```bash
# Start Ollama
ollama serve &

# Pull model (if needed)
ollama pull llama3.2:3b
```

## Installation (First Time Only)

```bash
# Navigate to project
cd /home/ahmad/Desktop/sav_b2

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from storage.sqlite_writer import init_database; init_database()"

# Verify installation
python test_pipeline.py
```

Expected output: `✓ All tests passed!`

## Basic Usage

### 1. Process Tweets (Small Sample)

```bash
# Process first 20 tweets
python run_pipeline.py data/free_tweet_export.csv --max-rows 20

# With report
python run_pipeline.py data/free_tweet_export.csv \
  --max-rows 20 \
  --output-report data/results/sample_20.txt
```

### 2. Run Evaluation

```bash
# Evaluate on 50 samples
python eval/offline_eval.py data/free_tweet_export.csv --sample-size 50
```

Check metrics in: `data/results/metrics/eval_report_*.json`

### 3. Query Results

```bash
# Connect to database
sqlite3 data/freemind.db

# View summary
SELECT 
  checker_status,
  COUNT(*) as count,
  ROUND(AVG(score_gravite), 2) as avg_gravity
FROM tweets_enriched
GROUP BY checker_status;

# View recent tweets with affect
SELECT 
  id,
  sentiment,
  emotion_primary,
  sarcasm,
  tone_color,
  checker_status
FROM tweets_enriched
ORDER BY created_timestamp DESC
LIMIT 10;

# Exit
.quit
```

## Configuration

### Environment Variables (Optional)

Create a `.env` file or export:

```bash
# Model settings
export OLLAMA_HOST=http://localhost:11434
export MODEL_NAME=llama3.2:3b

# Pipeline settings
export MAX_CONCURRENCY=6
export TIMEOUT_PER_AGENT=20
export ENABLE_GUARDRAILS=true

# Prompt version
export PROMPT_VERSION=v0.3
```

### Adjust Settings

Edit `config/settings.py` to change defaults:

```python
# Reduce concurrency if slow
max_concurrency: int = 3

# Increase timeout if needed
timeout_per_agent: int = 30

# Disable guardrails for testing
enable_guardrails: bool = False
```

## Common Workflows

### Workflow 1: Process Full Dataset

```bash
# Process all tweets (no limit)
python run_pipeline.py data/free_tweet_export.csv \
  --output-report data/results/full_run.txt

# Check results
sqlite3 data/freemind.db "SELECT COUNT(*) FROM tweets_enriched;"
```

### Workflow 2: Evaluate Quality

```bash
# Run offline eval
python eval/offline_eval.py data/free_tweet_export.csv --sample-size 100

# Check if thresholds met
cat data/results/metrics/eval_report_*.json | jq '.thresholds_met'
```

### Workflow 3: Review Failed Cases

```bash
# Check review queue
sqlite3 data/freemind.db "
SELECT reason, COUNT(*) 
FROM review_queue 
WHERE reviewed=0 
GROUP BY reason;
"

# View specific cases
sqlite3 data/freemind.db "
SELECT tweet_id, reason, original_labels
FROM review_queue
WHERE reviewed=0
LIMIT 5;
"
```

### Workflow 4: Test New Prompt Version

```bash
# Run canary test
python eval/canary_runner.py data/free_tweet_export.csv \
  --baseline v0.2 \
  --canary v0.3 \
  --sample-rate 0.1

# Check decision
cat data/results/canary/canary_decision_v0.3.json | jq '.decision'
```

## Troubleshooting

### Issue: "Connection refused" to Ollama

```bash
# Check if Ollama is running
ps aux | grep ollama

# Start Ollama
ollama serve &

# Wait a few seconds, then retry
```

### Issue: Slow processing

```bash
# Reduce concurrency
export MAX_CONCURRENCY=3

# Or edit config/settings.py
# max_concurrency: int = 3
```

### Issue: High fail rate

```bash
# Check what's failing
sqlite3 data/freemind.db "
SELECT json_extract(a2a_trace, '$.rules_fired'), COUNT(*)
FROM tweets_enriched
WHERE checker_status='fail'
GROUP BY json_extract(a2a_trace, '$.rules_fired');
"

# Review prompts in prompts/freemind_prompts_v0.3.json
```

### Issue: Guardrails refusing too much

```bash
# Check refusal reasons
sqlite3 data/freemind.db "
SELECT refused_reason, COUNT(*)
FROM tweets_enriched
WHERE refused_by_guardrails=1
GROUP BY refused_reason;
"

# Temporarily disable for testing
export ENABLE_GUARDRAILS=false
```

## Understanding Output

### Pipeline Output

```
[1/20] Processing tweet 1234567890...
  Status: ok, Time: 2.34s, Refused: False
```

- **Status**: `ok` (good), `warn` (minor issues), `fail` (major issues)
- **Time**: Processing time for this tweet
- **Refused**: Whether guardrails blocked it

### Evaluation Output

```
Total tweets: 50
Avg time/tweet: 2.45s
Checker status:
  OK: 45 (90.0%)
  WARN: 4 (8.0%)
  FAIL: 1 (2.0%)
```

- **OK ≥90%**: Good quality
- **WARN ≤9%**: Acceptable
- **FAIL ≤1%**: Target threshold

### Database Schema

Key columns in `tweets_enriched`:

- `full_text`: Original tweet (never modified)
- `utile`: SAV-relevant? (bool)
- `categorie`: probleme/question/retour_client
- `sentiment`: 10-level scale
- `type_probleme`: panne/facturation/etc
- `score_gravite`: -10 to +10
- `emotion_primary`: joie/colere/tristesse/etc
- `sarcasm`: Detected? (bool)
- `tone_color`: rouge/orange/vert/etc
- `checker_status`: ok/warn/fail
- `a2a_trace`: JSON with rules fired

## Next Steps

1. **Process your data**: `python run_pipeline.py data/your_tweets.csv`
2. **Review results**: `sqlite3 data/freemind.db`
3. **Check quality**: `python eval/offline_eval.py data/your_tweets.csv`
4. **Iterate on prompts**: Edit `prompts/freemind_prompts_v0.3.json`
5. **Read docs**: `docs/runbook.md` and `docs/guardrails.md`

## Getting Help

- **Runbook**: `docs/runbook.md` - Detailed operations guide
- **Guardrails**: `docs/guardrails.md` - Safety policy
- **README**: `README_FREEMIND.md` - Full documentation
- **Tests**: `python test_pipeline.py` - Verify setup

## Example Session

```bash
# 1. Start fresh
cd /home/ahmad/Desktop/sav_b2

# 2. Process 50 tweets
python run_pipeline.py data/free_tweet_export.csv --max-rows 50

# 3. Check results
sqlite3 data/freemind.db "
SELECT checker_status, COUNT(*) 
FROM tweets_enriched 
GROUP BY checker_status;
"

# 4. View a sample
sqlite3 data/freemind.db "
SELECT full_text, sentiment, emotion_primary, score_gravite
FROM tweets_enriched
LIMIT 3;
"

# 5. Run evaluation
python eval/offline_eval.py data/free_tweet_export.csv --sample-size 50

# Done!
```

## Tips

- Start with small samples (20-50 tweets) to test
- Monitor latency - should be <3s/tweet average
- Check guardrail refusal rate - should be <2%
- Review failed cases to improve prompts
- Use canary testing before deploying new prompts
- Keep database backups: `cp data/freemind.db data/freemind_backup.db`

---

**Ready to go!** Start with: `python run_pipeline.py data/free_tweet_export.csv --max-rows 20`

