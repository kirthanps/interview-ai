"""Quick Groq model probe for smoke validation.

Run from project root:
    python -m tests.smoke.groq_model_probe
"""

from __future__ import annotations

import os
from pathlib import Path

from langchain_groq import ChatGroq

DEFAULT_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
]


def read_env_value_from_file(key: str, env_file: str = ".env") -> str:
    path = Path(env_file)
    if not path.exists():
        return ""

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == key:
            return v.strip().strip('"').strip("'")
    return ""


def model_candidates() -> list[str]:
    explicit = os.getenv(
        "GROQ_MODEL") or read_env_value_from_file("GROQ_MODEL")
    extra = os.getenv("GROQ_MODEL_CANDIDATES") or read_env_value_from_file(
        "GROQ_MODEL_CANDIDATES")

    candidates: list[str] = []
    if explicit:
        candidates.append(explicit.strip())

    if extra:
        candidates.extend([item.strip()
                          for item in extra.split(",") if item.strip()])

    candidates.extend(DEFAULT_MODELS)

    seen = set()
    unique = []
    for item in candidates:
        if item in seen:
            continue
        unique.append(item)
        seen.add(item)
    return unique


def main() -> None:
    api_key = os.getenv(
        "GROQ_API_KEY") or read_env_value_from_file("GROQ_API_KEY")
    if not api_key:
        print("FAIL: GROQ_API_KEY missing in env/.env")
        return

    prompt = "Reply with exactly: OK"

    for model_name in model_candidates():
        try:
            llm = ChatGroq(model=model_name, temperature=0.0,
                           groq_api_key=api_key)
            response = llm.invoke(prompt)
            content = getattr(response, "content", response)
            print(
                f"PASS model={model_name} response={str(content).strip()[:120]}")
            return
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL model={model_name} error={exc}")

    print("No working Groq models found from candidates.")


if __name__ == "__main__":
    main()
