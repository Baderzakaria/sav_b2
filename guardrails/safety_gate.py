"""Safety gate for guardrails: prompt injection, PII, toxicity, and safety checks."""

import re
from typing import Dict, Any, Literal
from dataclasses import dataclass
from models.labels import GuardrailFlags


@dataclass
class GuardrailDecision:
    """Decision from the safety gate."""
    action: Literal["allow", "warn", "refuse"]
    flags: GuardrailFlags
    reason: str = ""


class SafetyGate:
    """Implements guardrail checks for input safety and quality."""
    
    # Prompt injection patterns
    INJECTION_PATTERNS = [
        r"ignore\s+(previous|all|above|prior)",
        r"disregard\s+(instructions|rules|prompt)",
        r"system\s*:\s*you\s+are",
        r"new\s+instructions",
        r"override\s+(mode|settings|instructions)",
        r"<\s*system\s*>",
        r"```\s*(python|javascript|bash|sql)",
        r"execute\s+(code|command|script)",
        r"eval\s*\(",
        r"__import__",
    ]
    
    # PII patterns (basic)
    PII_PATTERNS = [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{16}\b",  # Credit card
        r"\b[A-Z]{2}\d{6,9}\b",  # Passport-like
    ]
    
    # Safety/toxicity keywords (basic)
    SAFETY_KEYWORDS = [
        "suicide", "self-harm", "kill myself",
        "child abuse", "sexual minor",
    ]
    
    MAX_LENGTH = 10000  # Character limit
    
    def __init__(self):
        self.injection_regex = re.compile(
            "|".join(self.INJECTION_PATTERNS),
            re.IGNORECASE
        )
        self.pii_regex = re.compile(
            "|".join(self.PII_PATTERNS),
            re.IGNORECASE
        )
    
    def check(self, text: str, context: Dict[str, str]) -> GuardrailDecision:
        """Run all guardrail checks on input text and context."""
        flags = GuardrailFlags()
        reasons = []
        action = "allow"
        
        # Length check
        if len(text) > self.MAX_LENGTH:
            flags.excessive_length = True
            reasons.append(f"Text exceeds {self.MAX_LENGTH} chars")
            action = "refuse"
        
        # Prompt injection check
        if self.injection_regex.search(text):
            flags.prompt_injection = True
            reasons.append("Potential prompt injection detected")
            action = "refuse"
        
        # Check context fields too
        for ctx_key, ctx_val in context.items():
            if ctx_val and self.injection_regex.search(ctx_val):
                flags.prompt_injection = True
                reasons.append(f"Prompt injection in {ctx_key}")
                action = "refuse"
        
        # PII detection
        if self.pii_regex.search(text):
            flags.pii_detected = True
            reasons.append("PII detected in text")
            action = "warn"  # Warn but don't refuse
        
        # Code injection (simple check for code blocks)
        if "```" in text or "eval(" in text or "__import__" in text:
            flags.code_injection = True
            reasons.append("Code injection pattern detected")
            action = "refuse"
        
        # Safety keywords
        text_lower = text.lower()
        for keyword in self.SAFETY_KEYWORDS:
            if keyword in text_lower:
                flags.safety_violation = True
                reasons.append(f"Safety keyword detected: {keyword}")
                action = "refuse"
                break
        
        flags.reasons = reasons
        reason_str = "; ".join(reasons) if reasons else ""
        
        return GuardrailDecision(
            action=action,
            flags=flags,
            reason=reason_str
        )
    
    def to_dict(self, decision: GuardrailDecision) -> Dict[str, Any]:
        """Convert decision to dict for storage."""
        return {
            "refused": decision.action == "refuse",
            "warned": decision.action == "warn",
            "reason": decision.reason,
            "flags": {
                "prompt_injection": decision.flags.prompt_injection,
                "excessive_length": decision.flags.excessive_length,
                "pii_detected": decision.flags.pii_detected,
                "safety_violation": decision.flags.safety_violation,
                "code_injection": decision.flags.code_injection,
                "reasons": decision.flags.reasons,
            }
        }

