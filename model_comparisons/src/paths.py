from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .io_utils import ensure_dir


@dataclass(frozen=True)
class RunPaths:
    run_id: str
    root: Path
    manifests: Path
    generated_questions: Path
    human_answers: Path
    model_scores: Path
    judge_question_phase: Path
    judge_scoring_phase: Path
    analysis: Path
    reports: Path


def create_run_paths(base_dir: Path, run_id: str | None = None) -> RunPaths:
    resolved_run_id = run_id or datetime.now(
        timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    root = base_dir / "runs" / resolved_run_id

    paths = RunPaths(
        run_id=resolved_run_id,
        root=root,
        manifests=root / "manifests",
        generated_questions=root / "generated_questions",
        human_answers=root / "human_answers",
        model_scores=root / "model_scores",
        judge_question_phase=root / "judge_evaluations" / "question_phase",
        judge_scoring_phase=root / "judge_evaluations" / "scoring_phase",
        analysis=root / "analysis",
        reports=root / "reports",
    )

    for path in [
        paths.root,
        paths.manifests,
        paths.generated_questions,
        paths.human_answers,
        paths.model_scores,
        paths.judge_question_phase,
        paths.judge_scoring_phase,
        paths.analysis,
        paths.reports,
        paths.reports / "figures",
    ]:
        ensure_dir(path)

    return paths
