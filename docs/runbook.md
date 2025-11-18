# FreeMind Pipeline Runbook

## Quick Start

### Prerequisites

- Python 3.9+
- Ollama running locally with `llama3.2:3b` model
- Git (for run ID generation)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from storage.sqlite_writer import init_database; init_database()"

# Verify Ollama is running
curl http://localhost:11434/api/tags
```

### Basic Usage

```bash
# Run pipeline on CSV
python run_pipeline.py data/free_tweet_export.csv --max-rows 50

# Run with report
python run_pipeline.py data/free_tweet_export.csv --output-report data/results/report.txt
```

## Environment Variables

### Required

```bash
# Ollama endpoint
export OLLAMA_HOST=http://localhost:11434

# Model name
export MODEL_NAME=llama3.2:3b
```

### Optional

```bash
# Prompt version
export PROMPT_VERSION=v0.3

# Database path
export DB_PATH=data/freemind.db

# Concurrency (number of parallel agents)
export MAX_CONCURRENCY=6

# Timeout per agent (seconds)
export TIMEOUT_PER_AGENT=20

# Enable/disable guardrails
export ENABLE_GUARDRAILS=true
```

## Run ID Format

Run IDs follow the format: `YYYYMMDD-HHMM-<gitsha7>`

Example: `20251107-1430-a3f2b1c`

This ensures:
- Chronological ordering
- Git commit traceability
- Unique identification

## Prompt Versioning

### Registry

Prompt versions are tracked in `prompts/registry.json`:

```json
{
  "active_version": "v0.3",
  "versions": [
    {
      "version": "v0.3",
      "semver": "0.3.0",
      "file": "freemind_prompts.json",
      "changelog": "Added A6_affect agent...",
      "created": "2025-11-07",
      "status": "active"
    }
  ]
}
```

### Version Management

1. **Create new version**: Copy `prompts/freemind_prompts_vX.Y.json` to `vX.Y+1.json`
2. **Update registry**: Add entry to `versions` array
3. **Test**: Run offline eval on sample
4. **Canary**: Run canary test before promotion
5. **Promote**: Update `active_version` in registry

### Rollback

```bash
# Revert to previous version
# 1. Update registry.json active_version
# 2. Set environment variable
export PROMPT_VERSION=v0.2

# 3. Restart pipeline
python run_pipeline.py data/input.csv
```

## Metrics & Monitoring

### Metrics Paths

- **Eval reports**: `data/results/metrics/eval_report_<run_id>.json`
- **Canary decisions**: `data/results/canary/canary_decision_<version>.json`
- **Pipeline reports**: `data/results/report_<run_id>.txt`

### Key Metrics

| Metric | Threshold | Action if Violated |
|--------|-----------|-------------------|
| JSON valid rate | ≥95% | Review prompts, check model |
| Checker OK rate | ≥90% | Tune checker rules |
| Checker FAIL rate | ≤1% | Investigate data quality |
| Avg latency | ≤3s/tweet | Optimize prompts, check resources |
| P95 latency | ≤6s/tweet | Check for outliers |
| Guardrail refusal | ≤2% | Review input data quality |

### Monitoring Commands

```bash
# Run offline evaluation
python eval/offline_eval.py data/input.csv --sample-size 100

# Check recent metrics
ls -lt data/results/metrics/

# View latest report
cat data/results/metrics/eval_report_*.json | jq .
```

## Database Schema

### Main Tables

- **tweets_enriched**: Labeled tweets with all metadata
- **review_queue**: HITL escalation queue
- **feedback_log**: Continuous learning feedback

### Querying

```bash
# Connect to database
sqlite3 data/freemind.db

# Check recent runs
SELECT run_id, COUNT(*) as tweets, 
       SUM(CASE WHEN checker_status='ok' THEN 1 ELSE 0 END) as ok,
       SUM(CASE WHEN checker_status='warn' THEN 1 ELSE 0 END) as warn,
       SUM(CASE WHEN checker_status='fail' THEN 1 ELSE 0 END) as fail
FROM tweets_enriched
GROUP BY run_id
ORDER BY run_id DESC
LIMIT 5;

# Check review queue
SELECT COUNT(*) as pending FROM review_queue WHERE reviewed=0;
```

## Continuous Learning

### Feedback Collection

Feedback is automatically collected for:
- Checker `warn` status
- Checker `fail` status
- HITL corrections

Location: `data/feedback/training_set.jsonl`

### Curation

```bash
# Check readiness for retraining
python -c "
from feedback.collector import FeedbackCollector
fc = FeedbackCollector()
status = fc.curate_training_set(
    min_samples=1000,
    output_path='data/curated/labels_v2.jsonl'
)
print(status)
"
```

### Retraining Workflow

1. **Collect**: Minimum 1,000 samples with quality gates
2. **Curate**: Deduplicate, filter refused samples
3. **Review**: Human review of sample subset
4. **Retrain**: Update prompts or fine-tune model
5. **Eval**: Run offline eval on gold set
6. **Canary**: A/B test with 10% traffic
7. **Promote**: Update active version if successful

## Troubleshooting

### Issue: High fail rate

**Symptoms**: Checker fail rate >5%

**Diagnosis**:
```bash
# Check a2a_trace for common issues
sqlite3 data/freemind.db "
SELECT json_extract(a2a_trace, '$.rules_fired') as rules, COUNT(*) as count
FROM tweets_enriched
WHERE checker_status='fail'
GROUP BY rules
ORDER BY count DESC
LIMIT 10;
"
```

**Solutions**:
- Review and tune checker rules
- Check for data quality issues
- Verify prompt clarity

### Issue: Slow processing

**Symptoms**: Avg latency >5s/tweet

**Diagnosis**:
```bash
# Check Ollama performance
curl http://localhost:11434/api/ps

# Monitor resource usage
htop
```

**Solutions**:
- Reduce concurrency: `export MAX_CONCURRENCY=3`
- Check Ollama GPU utilization
- Optimize prompt length
- Reduce context field sizes

### Issue: Guardrail false positives

**Symptoms**: Refusal rate >5% on clean data

**Diagnosis**:
```bash
# Check refusal reasons
sqlite3 data/freemind.db "
SELECT refused_reason, COUNT(*) as count
FROM tweets_enriched
WHERE refused_by_guardrails=1
GROUP BY refused_reason;
"
```

**Solutions**:
- Review and tune guardrail patterns
- Adjust thresholds
- Add exceptions for known patterns

## Backup & Recovery

### Database Backup

```bash
# Backup database
cp data/freemind.db data/backups/freemind_$(date +%Y%m%d).db

# Restore from backup
cp data/backups/freemind_20251107.db data/freemind.db
```

### Prompt Backup

Prompts are version-controlled in Git:

```bash
# View prompt history
git log --oneline prompts/

# Restore previous version
git checkout <commit> prompts/freemind_prompts_v0.3.json
```

## Performance Tuning

### Batch Processing

For large datasets:

```bash
# Process in batches
for i in {0..10}; do
  python run_pipeline.py data/input.csv \
    --max-rows 1000 \
    --skip $((i * 1000))
done
```

### Parallel Runs

Multiple independent runs can be executed in parallel (different CSVs):

```bash
# Run in background
python run_pipeline.py data/batch1.csv &
python run_pipeline.py data/batch2.csv &
wait
```

## Maintenance

### Daily

- Monitor HITL queue size
- Check guardrail refusal rate
- Review error logs

### Weekly

- Run offline eval on sample
- Review checker metrics trends
- Check database size

### Monthly

- Evaluate retraining readiness
- Review and update guardrail patterns
- Archive old runs (>90 days)

## Support

For issues or questions:
1. Check this runbook
2. Review `docs/guardrails.md` for policy questions
3. Inspect database and logs
4. Review recent git commits for changes

