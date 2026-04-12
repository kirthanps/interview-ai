"""Convert downloaded HR interview JSON dataset to required CSV format.

Input JSON schema expected (example):
[
  {
    "question": "...",
    "category": "Adaptability",
    "ideal_answer": "...",
    ...
  }
]

Output CSV schema:
question,answer,category
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _clean_text(value: Any) -> str:
    """Normalize text values to single-line trimmed strings."""
    if value is None:
        return ""
    text = str(value)
    text = " ".join(text.replace("\r", " ").replace("\n", " ").split())
    return text.strip()


def _normalize_category(value: Any) -> str:
    """Convert source category to a stable lowercase label."""
    raw = _clean_text(value)
    if not raw:
        return "general"
    return raw.lower()


def _extract_rows(payload: List[Dict[str, Any]]) -> Tuple[List[List[str]], Dict[str, int]]:
    """Build CSV rows from payload and return stats."""
    rows: List[List[str]] = []
    seen = set()

    stats = {
        "input_items": len(payload),
        "kept": 0,
        "dropped_missing": 0,
        "dropped_duplicate": 0,
    }

    for item in payload:
        if not isinstance(item, dict):
            stats["dropped_missing"] += 1
            continue

        question = _clean_text(item.get("question"))
        answer = _clean_text(item.get("ideal_answer") or item.get("answer"))
        category = _normalize_category(item.get("category"))

        if not question or not answer:
            stats["dropped_missing"] += 1
            continue

        key = (question.lower(), answer.lower())
        if key in seen:
            stats["dropped_duplicate"] += 1
            continue

        seen.add(key)
        rows.append([question, answer, category])
        stats["kept"] += 1

    return rows, stats


def convert_dataset(input_path: Path, output_path: Path) -> Dict[str, int]:
    """Convert JSON dataset file to required CSV output."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Input JSON must be a list of objects")

    rows, stats = _extract_rows(payload)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["question", "answer", "category"])
        writer.writerows(rows)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert HR interview JSON dataset to interview_qa.csv format")
    parser.add_argument(
        "--input",
        default="data/hr_interview_questions_dataset.json",
        help="Path to source JSON dataset",
    )
    parser.add_argument(
        "--output",
        default="data/interview_qa.csv",
        help="Path to destination CSV",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    stats = convert_dataset(input_path=input_path, output_path=output_path)

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(
        "Stats: "
        f"input_items={stats['input_items']}, "
        f"kept={stats['kept']}, "
        f"dropped_missing={stats['dropped_missing']}, "
        f"dropped_duplicate={stats['dropped_duplicate']}"
    )


if __name__ == "__main__":
    main()
