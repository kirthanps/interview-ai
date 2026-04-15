from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_json


VALID_LABELS = {"SKIPPED", "SHORT", "POOR", "AVERAGE", "GOOD", "STRONG"}


def validate_answer_file(path: Path) -> list[str]:
    errors: list[str] = []
    payload: dict[str, Any] = read_json(path)

    required_top = [
        "combo_id",
        "source_model_alias",
        "source_model_id",
        "difficulty",
        "focus_area",
        "num_questions",
        "responses",
    ]
    for key in required_top:
        if key not in payload:
            errors.append(f"Missing top-level key '{key}'")

    responses = payload.get("responses", [])
    if not isinstance(responses, list):
        errors.append("'responses' must be a list")
        return errors

    if len(responses) != 10:
        errors.append(f"Expected 10 responses, found {len(responses)}")

    for idx, row in enumerate(responses, start=1):
        for key in ["question_id", "question_text", "answer_text", "answer_label", "timestamp_utc"]:
            if key not in row:
                errors.append(f"responses[{idx}] missing '{key}'")
        label = str(row.get("answer_label", "")).upper()
        if label and label not in VALID_LABELS:
            errors.append(f"responses[{idx}] invalid answer_label '{label}'")

    return errors


def validate_270_answers(human_answers_dir: Path) -> dict[str, Any]:
    files = sorted(human_answers_dir.glob("*_answers.json"))
    all_errors: dict[str, list[str]] = {}
    total_responses = 0

    for file in files:
        errs = validate_answer_file(file)
        if errs:
            all_errors[file.name] = errs
        else:
            payload = read_json(file)
            total_responses += len(payload.get("responses", []))

    return {
        "file_count": len(files),
        "total_responses": total_responses,
        "expected_file_count": 27,
        "expected_total_responses": 270,
        "is_complete": len(files) == 27 and total_responses == 270 and not all_errors,
        "errors": all_errors,
    }
