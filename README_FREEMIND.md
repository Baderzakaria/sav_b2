# FreeMind: A2A Multi-Agent Pipeline for Tweet Labeling

FreeMind is a production-grade, multi-agent pipeline for automated tweet labeling with comprehensive guardrails, A2A (agent-to-agent) validation, MLOps/LLMOps, and continuous learning.

## Features

### Core Capabilities
- **6 Specialized Agents**: A1 (utility), A2 (category), A3 (sentiment), A4 (problem type), A5 (gravity), A6 (affect/emotion)
- **A2A Checker**: Inter-agent consistency validation with conservative corrections
- **Safety Guardrails**: Prompt injection, PII, toxicity, and code injection detection
- **Expanded Affect Analysis**: Emotion, sarcasm, tone color, toxicity scoring
- **LangGraph Orchestration**: Parallel agent execution with state management
- **SQLite Persistence**: Full traceability with A2A traces and metadata
- **HITL Queue**: Human-in-the-loop escalation for edge cases
- **Continuous Learning**: Feedback collection and curation for retraining

### MLOps/LLMOps
- **Prompt Registry**: Version control with changelog and rollback support
- **Offline Evaluation**: Automated metrics on gold/adversarial sets
- **Canary Testing**: A/B testing with auto-rollback on quality degradation
- **Drift Detection**: Monitor label distribution and quality metrics
- **Run Traceability**: Git-based run IDs for reproducibility

## Quick Start

### Prerequisites

```bash
# Install Ollama and pull model
ollama pull llama3.2:3b

# Start Ollama server
ollama serve
```

### Installation

```bash
# Clone repository
cd /home/ahmad/Desktop/sav_b2

# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from storage.sqlite_writer import init_database; init_database()"
```

### Basic Usage

```bash
# Process tweets
python run_pipeline.py data/free_tweet_export.csv --max-rows 50

# Run evaluation
python eval/offline_eval.py data/free_tweet_export.csv --sample-size 100

# Run canary test
python eval/canary_runner.py data/free_tweet_export.csv --baseline v0.2 --canary v0.3
```

## Architecture

```
Input CSV
    ↓
[Preprocess] → Build context (ctx_before/after/refs)
    ↓
[Safety Gate] → Guardrails (injection, PII, toxicity)
    ↓
[Agents Parallel] → A1, A2, A3, A4, A5, A6 (concurrent)
    ↓
[Checker A2A] → Validate consistency, apply rules
    ↓
[Writer] → SQLite + HITL escalation
    ↓
[Feedback Sink] → Continuous learning
```

## Labels Schema

```python
{
  "utile": bool,  # SAV-relevant?
  "categorie": "probleme" | "question" | "retour_client",
  "sentiment": "outrage_critique" | "tres_negatif" | ... | "tres_positif" | "mixte",
  "type_probleme": "panne" | "facturation" | "abonnement" | "resiliation" | "information" | "autre",
  "score_gravite": int (-10 to +10),
  "affect": {
    "emotion_primary": "joie" | "colere" | "tristesse" | ...,
    "sarcasm": bool,
    "tone_color": "rouge" | "orange" | "jaune" | "vert" | "bleu" | "violet" | "gris",
    "toxicity": "low" | "medium" | "high" | null
  }
}
```

## Configuration

### Environment Variables

```bash
# Model
export OLLAMA_HOST=http://localhost:11434
export MODEL_NAME=llama3.2:3b

# Prompts
export PROMPT_VERSION=v0.3

# Pipeline
export MAX_CONCURRENCY=6
export TIMEOUT_PER_AGENT=20
export ENABLE_GUARDRAILS=true

# Database
export DB_PATH=data/freemind.db
```

## Project Structure

```
sav_b2/
├── models/              # Pydantic schemas
│   └── labels.py
├── guardrails/          # Safety gate
│   └── safety_gate.py
├── checker/             # A2A rules
│   └── rules.py
├── storage/             # SQLite persistence
│   ├── schema.sql
│   └── sqlite_writer.py
├── config/              # Settings
│   └── settings.py
├── eval/                # Evaluation & canary
│   ├── offline_eval.py
│   └── canary_runner.py
├── feedback/            # Continuous learning
│   └── collector.py
├── prompts/             # Prompt registry
│   ├── registry.json
│   └── freemind_prompts_v0.3.json
├── docs/                # Documentation
│   ├── guardrails.md
│   └── runbook.md
├── orchestrator.py      # LangGraph pipeline
├── run_pipeline.py      # CLI runner
└── utils_runtime.py     # Ollama client
```

## Key Metrics

| Metric | Target | Critical Threshold |
|--------|--------|-------------------|
| JSON valid rate | ≥95% | <90% |
| Checker OK rate | ≥90% | <85% |
| Checker FAIL rate | ≤1% | >5% |
| Avg latency | ≤3s/tweet | >5s |
| P95 latency | ≤6s/tweet | >10s |
| Guardrail refusal | ≤2% | >5% |

## A2A Checker Rules

1. **Utility normalization**: `utile=false` → force safe defaults
2. **Sentiment-category consistency**: Warn on `probleme` + positive sentiment
3. **Gravity clamping**: Enforce taxonomy allowed ranges
4. **Context-driven corrections**: Incident keywords → `type=panne`
5. **Emotion-gravity alignment**: `colere` → high gravity expected
6. **Sarcasm softening**: Reduce fail → warn for sarcastic tweets

## Guardrails

- **Prompt injection**: Detect override attempts → **Refuse**
- **Excessive length**: >10k chars → **Refuse**
- **PII detection**: SSN, credit cards → **Warn**
- **Safety violations**: Self-harm, explicit content → **Refuse**
- **Code injection**: Code blocks, eval() → **Refuse**

See `docs/guardrails.md` for full policy.

## Continuous Learning

1. **Feedback collection**: Auto-collect warn/fail cases
2. **Curation**: Deduplicate, exclude refused samples
3. **Retraining readiness**: ≥1k samples with quality gates
4. **Promotion**: Offline eval + canary → update active version

## Documentation

- **Runbook**: `docs/runbook.md` - Operations, troubleshooting, maintenance
- **Guardrails**: `docs/guardrails.md` - Safety policy, refusal handling
- **Plan**: `plans/freemind_plan.md` - Original architecture spec

## Database

### Main Tables

- `tweets_enriched`: Labeled tweets with full traceability
- `review_queue`: HITL escalation queue
- `feedback_log`: Continuous learning samples

### Querying

```bash
sqlite3 data/freemind.db

# Recent run summary
SELECT run_id, COUNT(*) as tweets,
       SUM(CASE WHEN checker_status='ok' THEN 1 ELSE 0 END) as ok,
       SUM(CASE WHEN checker_status='warn' THEN 1 ELSE 0 END) as warn,
       SUM(CASE WHEN checker_status='fail' THEN 1 ELSE 0 END) as fail
FROM tweets_enriched
GROUP BY run_id
ORDER BY run_id DESC
LIMIT 5;
```

## Testing

```bash
# Run 50-row smoke test
python run_pipeline.py data/free_tweet_export.csv --max-rows 50 --output-report data/results/smoke_test.txt

# Offline evaluation
python eval/offline_eval.py data/free_tweet_export.csv --sample-size 100

# Check thresholds
cat data/results/metrics/eval_report_*.json | jq '.thresholds_met'
```

## Troubleshooting

### High fail rate
- Check `a2a_trace` for common rule violations
- Review prompts for clarity
- Verify data quality

### Slow processing
- Reduce `MAX_CONCURRENCY`
- Check Ollama GPU utilization
- Optimize prompt length

### Guardrail false positives
- Review `refused_reason` distribution
- Tune patterns in `guardrails/safety_gate.py`
- Add exceptions for known safe patterns

See `docs/runbook.md` for detailed troubleshooting.

## Best Practices Applied

### A2A & Multi-Agent
✓ Specialized agents with bounded responsibilities  
✓ Standardized message protocol with tracing  
✓ Conservative A2A checker with explicit rules  
✓ Parallel execution with timeout/retry policies  
✓ HITL escalation for ambiguity and failures  

### LLMOps
✓ Prompt registry with versioning and changelog  
✓ JSON schema enforcement per agent  
✓ Offline evaluation with gold/adversarial sets  
✓ Canary testing with auto-rollback  
✓ Run IDs for reproducibility  

### MLOps
✓ Data versioning ready (DVC integration planned)  
✓ Experiment tracking (MLflow integration planned)  
✓ Drift detection on label distributions  
✓ Feedback curation for retraining  

### Guardrails
✓ Input scanning before processing  
✓ Refusal policy with safety taxonomy  
✓ PII detection and flagging  
✓ Structured logging for audit trails  

## Roadmap

- [ ] Streamlit UI for HITL review queue
- [ ] DVC integration for data versioning
- [ ] MLflow integration for experiment tracking
- [ ] Real-time monitoring dashboard
- [ ] Multi-language support (French, Arabic)
- [ ] ML-based PII detection
- [ ] Fine-tuning on curated feedback

## License

Internal project - see organization policies.

## Contact

For questions or issues, see `docs/runbook.md` or contact the team.

