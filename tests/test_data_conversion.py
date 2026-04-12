from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.convert_hr_json_to_interview_csv import convert_dataset  # noqa: E402


def test_convert_hr_json_to_interview_csv(tmp_path: Path) -> None:
    source = tmp_path / "hr.json"
    output = tmp_path / "interview_qa.csv"

    payload = [
        {
            "question": "Tell me about yourself",
            "category": "Career Goals",
            "ideal_answer": "I summarize my profile and outcomes.",
        },
        {
            "question": "Tell me about yourself",
            "category": "Career Goals",
            "ideal_answer": "I summarize my profile and outcomes.",
        },
        {
            "question": "",
            "category": "Team",
            "ideal_answer": "Missing question should be dropped",
        },
    ]

    source.write_text(json.dumps(payload), encoding="utf-8")

    stats = convert_dataset(input_path=source, output_path=output)

    with output.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))

    assert rows[0] == ["question", "answer", "category"]
    assert len(rows) == 2
    assert rows[1][0] == "Tell me about yourself"
    assert rows[1][1] == "I summarize my profile and outcomes."
    assert rows[1][2] == "career goals"

    assert stats["input_items"] == 3
    assert stats["kept"] == 1
    assert stats["dropped_duplicate"] == 1
    assert stats["dropped_missing"] == 1
