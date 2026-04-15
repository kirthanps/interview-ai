from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io_utils import write_text


def _md_table(headers: list[str], rows: list[list[Any]]) -> str:
    header_row = "| " + " | ".join(str(h) for h in headers) + " |"
    sep_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(cell)
                              for cell in row) + " |" for row in rows]
    return "\n".join([header_row, sep_row, *body])


def build_report_markdown(summary: dict[str, Any]) -> str:
    generated_at = datetime.now(timezone.utc).isoformat()

    ranking_rows = []
    for idx, row in enumerate(summary["final_ranking"], start=1):
        ranking_rows.append(
            [
                idx,
                row["model_alias"],
                f"{row['avg_question_phase_score']:.2f}",
                f"{row['avg_scoring_phase_score']:.2f}",
                f"{row['final_score']:.2f}",
            ]
        )

    qphase_rows = []
    for row in summary["question_phase_rows"]:
        qphase_rows.append(
            [
                row["combo_id"],
                row["model_alias"],
                f"{row['qms']:.2f}",
                f"{row['jqs']:.2f}",
                f"{row['qps']:.2f}",
            ]
        )

    sphase_rows = []
    for row in summary["scoring_phase_rows"]:
        sphase_rows.append(
            [
                row["combo_id"],
                row["source_model_alias"],
                row["model_alias"],
                f"{row['sms']:.2f}",
                f"{row['jss']:.2f}",
                f"{row['sps']:.2f}",
            ]
        )

    report = f"""# InterviewAI Model Comparison Report

Generated at: {generated_at}

## 1. Executive Summary

This report compares three evaluated models (llama33, llama31_8b, qwen32b) using metric-based and judge-based methods with strict prompt fairness controls.

## 2. Fairness Controls

- Same canonical payload per combo for all evaluated models
- SHA256 payload hashes persisted
- Judge is blind to model identity (System A/B/C)
- Judge order randomized per evaluation and mapping saved
- Fixed judge prompt and deterministic low-temperature settings

## 3. Model Roles

- Evaluated models: llama-3.3-70b-versatile, llama-3.1-8b-instant, qwen/qwen3-32b
- Judge model only: openai/gpt-oss-120b

## 4. Question-Phase Results

{_md_table(["combo_id", "model", "QMS", "JQS", "QPS"], qphase_rows)}

## 5. Scoring-Phase Results

{_md_table(["combo_id", "source_set", "model", "SMS", "JSS", "SPS"], sphase_rows)}

## 6. Final Ranking

{_md_table(["rank", "model", "avg_qps", "avg_sps", "final_score"], ranking_rows)}

## 7. Statistical Significance

ANOVA:
- method: {summary['anova']['method']}
- f_stat: {summary['anova']['f_stat']}
- p_value: {summary['anova']['p_value']}

Pairwise comparisons:
{_md_table(["left", "right", "p_value", "cohen_d", "method"], [[x['left'], x['right'], x['p_value'], f"{x['cohen_d']:.4f}", x['method']] for x in summary['pairwise_tests']])}

## 8. Error Analysis

- Review low-confidence judge cases from judge evaluation JSON files.
- Inspect skipped-answer handling mismatch cases.

## 9. Recommendation

Select the highest-ranked model unless operational constraints (latency/cost) require fallback.
"""
    return report


def write_report(report_path: Path, summary: dict[str, Any]) -> None:
    write_text(report_path, build_report_markdown(summary))
