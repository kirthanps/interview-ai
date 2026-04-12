from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data_processing.generate_synthetic_datasets import (  # noqa: E402
    build_synthetic_jd_profiles,
    build_synthetic_qa_pairs,
    generate_default_datasets,
)


def test_synthetic_dataset_generator_meets_demo_sizes(tmp_path: Path) -> None:
    qa_path, jd_path = generate_default_datasets(output_dir=str(tmp_path))

    with qa_path.open("r", encoding="utf-8", newline="") as file:
        qa_rows = list(csv.reader(file))

    with jd_path.open("r", encoding="utf-8") as file:
        jd_rows = json.load(file)

    assert len(qa_rows) - 1 >= 500
    assert len(jd_rows) >= 30


def test_generator_helpers_return_expected_counts() -> None:
    assert len(build_synthetic_qa_pairs(100)) == 100
    assert len(build_synthetic_jd_profiles(30)) == 30
