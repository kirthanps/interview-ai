from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_json

try:
    import yaml
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "PyYAML is required for model_comparisons. Install it with: pip install pyyaml"
    ) from exc


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_experiment_config(base_dir: Path) -> dict[str, Any]:
    return load_yaml(base_dir / "config" / "experiment_config.yaml")


def load_judge_rubrics(base_dir: Path) -> dict[str, Any]:
    return load_yaml(base_dir / "config" / "judge_rubrics.yaml")


def load_combinations(base_dir: Path) -> list[dict[str, Any]]:
    return read_json(base_dir / "data" / "combinations.json")
