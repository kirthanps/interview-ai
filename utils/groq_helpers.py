"""Shared GROQ helper utilities for Module 2 and Module 4."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)

DEFAULT_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
]


def read_env_value_from_file(key: str, env_file: str = ".env") -> str:
    """Read a single environment value from a .env-style file."""
    path = Path(env_file)
    if not path.exists():
        return ""

    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            k, v = line.split("=", 1)
            if k.strip() != key:
                continue

            return v.strip().strip('"').strip("'")
    except Exception:
        logger.exception("Failed reading %s for key %s", env_file, key)

    return ""


def get_groq_api_key() -> str:
    """Resolve GROQ API key from environment or .env."""
    return os.getenv("GROQ_API_KEY") or read_env_value_from_file("GROQ_API_KEY")


def get_groq_model_candidates() -> List[str]:
    """Resolve Groq model candidates from env/.env with safe defaults."""
    explicit = os.getenv(
        "GROQ_MODEL") or read_env_value_from_file("GROQ_MODEL")
    candidates: List[str] = []

    if explicit and explicit.strip():
        candidates.append(explicit.strip())

    extra = os.getenv("GROQ_MODEL_CANDIDATES") or read_env_value_from_file(
        "GROQ_MODEL_CANDIDATES")
    if extra:
        for item in extra.split(","):
            model_name = item.strip()
            if model_name:
                candidates.append(model_name)

    candidates.extend(DEFAULT_GROQ_MODELS)

    unique: List[str] = []
    seen = set()
    for item in candidates:
        if item in seen:
            continue
        unique.append(item)
        seen.add(item)
    return unique


def is_model_decommission_error(exc: Exception) -> bool:
    """Detect known Groq model decommission errors for retry fallback."""
    text = str(exc).lower()
    return "model_decommissioned" in text or "decommissioned" in text


def build_groq_llm(model_name: str, temperature: float) -> ChatGroq:
    """Create a GROQ chat model for a specific model name."""
    from langchain_groq import ChatGroq

    api_key = get_groq_api_key()
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env/environment.")

    return ChatGroq(
        model=model_name,
        temperature=temperature,
        groq_api_key=api_key,
    )
