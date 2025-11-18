"""Configuration settings for FreeMind pipeline."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class Settings:
    """Pipeline configuration settings."""
    
    # Model settings
    ollama_host: str = "http://localhost:11434"
    model_name: str = "llama3.2:3b"
    
    # Prompt settings
    prompt_version: str = "v0.3"
    prompt_file: str = "prompts/freemind_prompts_v0.3.json"
    
    # Database settings
    db_path: str = "data/freemind.db"
    
    # Pipeline settings
    max_concurrency: int = 6  # A1-A6 parallel
    timeout_per_agent: int = 20  # seconds
    max_retries: int = 1
    
    # Context limits
    ctx_max_length: int = 800  # chars per context field
    
    # Batch settings
    batch_size: int = 10
    max_rows: Optional[int] = None
    
    # Guardrails
    enable_guardrails: bool = True
    
    # Evaluation
    eval_sample_size: int = 100
    
    # Thresholds
    json_valid_threshold: float = 0.95
    checker_ok_threshold: float = 0.90
    checker_warn_threshold: float = 0.09
    checker_fail_threshold: float = 0.01
    mean_latency_threshold: float = 3.0  # seconds
    
    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        return cls(
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            model_name=os.getenv("MODEL_NAME", "llama3.2:3b"),
            prompt_version=os.getenv("PROMPT_VERSION", "v0.3"),
            db_path=os.getenv("DB_PATH", "data/freemind.db"),
            max_concurrency=int(os.getenv("MAX_CONCURRENCY", "6")),
            timeout_per_agent=int(os.getenv("TIMEOUT_PER_AGENT", "20")),
            enable_guardrails=os.getenv("ENABLE_GUARDRAILS", "true").lower() == "true",
        )


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get singleton settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings

