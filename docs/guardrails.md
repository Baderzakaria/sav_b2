# FreeMind Guardrails & Refusal Policy

## Overview

The FreeMind pipeline implements a comprehensive guardrail system to ensure safety, quality, and compliance in automated tweet labeling.

## Safety Gate

The safety gate runs **before** agent processing and makes three types of decisions:

- **Allow**: Tweet passes all checks, proceed with labeling
- **Warn**: Minor issues detected, proceed but flag for review
- **Refuse**: Critical issues detected, skip labeling and enqueue for human review

## Guardrail Checks

### 1. Prompt Injection Detection

**Purpose**: Prevent adversarial inputs that attempt to override system instructions.

**Patterns detected**:
- "ignore previous instructions"
- "disregard rules"
- "system: you are..."
- "new instructions"
- "override mode"
- Code execution attempts (`eval()`, `__import__`, etc.)

**Action**: **Refuse** - Tweet is not processed and flagged for manual review.

### 2. Excessive Length

**Purpose**: Prevent resource exhaustion and ensure consistent processing times.

**Threshold**: 10,000 characters

**Action**: **Refuse** - Tweet exceeds reasonable length limits.

### 3. PII Detection (Basic)

**Purpose**: Identify and protect personally identifiable information.

**Patterns detected**:
- Social Security Numbers (SSN)
- Credit card numbers
- Passport-like identifiers

**Action**: **Warn** - Tweet is processed but flagged for PII redaction in exports.

**Note**: This is a basic implementation. Production systems should use specialized PII detection libraries.

### 4. Safety Violations

**Purpose**: Detect content requiring specialized handling.

**Keywords detected**:
- Self-harm indicators
- Child safety concerns
- Explicit content

**Action**: **Refuse** - Tweet is not processed and escalated to safety team.

### 5. Code Injection

**Purpose**: Prevent code execution attempts via markdown or embedded scripts.

**Patterns detected**:
- Code blocks (```)
- Direct code execution patterns

**Action**: **Refuse** - Potential security risk.

## Refusal Handling

When a tweet is refused:

1. **No labeling occurs** - Agents are not invoked
2. **Logged to database** - `refused_by_guardrails=true`, `refused_reason` set
3. **Enqueued for review** - Added to `review_queue` table
4. **Excluded from training** - Refused samples are not used for model improvement

## Curation Policy for Continuous Learning

### Inclusion Criteria

Samples are included in the training set if:
- ✓ Passed guardrails (not refused)
- ✓ Checker status is `warn` or `fail` (indicates learning opportunity)
- ✓ Unique by tweet ID (deduplicated)
- ✓ Contains complete context and labels

### Exclusion Criteria

Samples are excluded if:
- ✗ Refused by guardrails
- ✗ Contains safety violations
- ✗ Duplicate tweet ID
- ✗ Missing critical fields

### Retraining Readiness

A curated training set is ready for retraining when:
- Minimum 1,000 samples collected
- Quality gates passed (diversity, label distribution)
- Human review of sample subset completed

## Human-in-the-Loop (HITL) Escalation

### Escalation Triggers

Tweets are escalated to human review when:

1. **Guardrail refusal** - Safety or injection detected
2. **Checker fail** - Multiple validation errors
3. **Checker warn** - Significant inconsistencies (configurable threshold)
4. **Repeated errors** - Same tweet fails multiple times

### Review Queue

The `review_queue` table stores:
- Tweet ID and original text
- Reason for escalation
- Original labels (if any)
- Corrected labels (filled by human reviewer)
- Review status and timestamp

### Review Workflow

1. Reviewer accesses queue via Streamlit UI (future)
2. Reviews tweet, context, and agent outputs
3. Provides corrected labels or confirms refusal
4. Feedback is logged to `feedback_log` table
5. Corrected labels are used for continuous learning

## Monitoring & Alerts

### Key Metrics

- **Refusal rate**: Should be <2% for typical datasets
- **False positive rate**: <0.5% on gold set
- **HITL queue size**: Monitor for backlog
- **Safety violation rate**: Track trends

### Alerting Thresholds

- Refusal rate >5%: Investigate data quality or guardrail tuning
- HITL queue >100 items: Scale review capacity
- Safety violations >10/day: Escalate to safety team

## Privacy & Compliance

### Data Handling

- **PII redaction**: Detected PII is flagged but not automatically redacted (manual review required)
- **Sensitive content**: Safety violations are logged with minimal context
- **Retention**: Refused samples are retained for 90 days, then archived

### Compliance

- **GDPR**: Right to deletion supported via tweet ID
- **CCPA**: Data access and deletion requests handled via admin interface
- **Audit trail**: All guardrail decisions logged with timestamps

## Configuration

Guardrails can be configured via environment variables:

```bash
# Enable/disable guardrails
ENABLE_GUARDRAILS=true

# Adjust thresholds (future)
MAX_TEXT_LENGTH=10000
PII_DETECTION_LEVEL=basic
```

## Testing

### Unit Tests

- Test each guardrail pattern individually
- Verify action decisions (allow/warn/refuse)
- Check edge cases and false positives

### Integration Tests

- End-to-end pipeline with adversarial inputs
- Verify HITL escalation flow
- Test feedback collection

### Adversarial Eval Set

Maintain a set of adversarial examples:
- Prompt injection attempts
- Edge case inputs
- Known false positives

Run regularly to prevent regression.

## Future Enhancements

1. **ML-based PII detection**: Replace regex with trained models
2. **Toxicity scoring**: Add granular toxicity levels
3. **Multi-language support**: Extend patterns to French, Arabic, etc.
4. **Adaptive thresholds**: Learn optimal thresholds from feedback
5. **Real-time monitoring**: Dashboard for guardrail metrics

