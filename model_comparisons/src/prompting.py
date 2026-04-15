from __future__ import annotations

from pathlib import Path

from .io_utils import read_text


def load_prompt(base_dir: Path, filename: str) -> str:
    return read_text(base_dir / "prompts" / filename)


def render_prompt(template: str, values: dict[str, str]) -> str:
    rendered = template
    for key, value in values.items():
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered
