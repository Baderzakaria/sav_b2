from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


@lru_cache(maxsize=1)
def load_environment(dotenv_path: Optional[str] = None) -> bool:
    """
    Load environment variables from a .env file exactly once.

    Parameters
    ----------
    dotenv_path:
        Optional path to a .env file. When omitted we try the repository root.
    """
    candidate = Path(dotenv_path) if dotenv_path else Path(__file__).resolve().parent / ".env"

    # load from explicit path first; fallback to default search if nothing was loaded
    loaded = load_dotenv(candidate, override=False) if candidate.exists() else False
    if not loaded:
        loaded = load_dotenv(override=False)
    return loaded

