from __future__ import annotations

from datetime import datetime
from pathlib import Path

from pipeline.scorer import AnswerScore, InterviewScores
from report.report_generator import generate_pdf_report


def _sample_scores() -> InterviewScores:
    return InterviewScores(
        overall_score=7.2,
        technical_depth_score=6.8,
        communication_score=8.0,
        what_went_well=[
            "Explained API auth decisions with practical production constraints.",
            "Structured most answers using situation-task-action-result framing.",
        ],
        areas_to_improve=[
            "Expand system design tradeoff explanations with scalability choices.",
            "Include measurable outcomes for collaboration stories.",
        ],
        role_readiness_percent=72,
        role_readiness_label="Good Foundation",
        recommended_resources=[
            "AWS Well-Architected Docs", "ByteByteGo YouTube"],
        per_answer_scores=[
            AnswerScore(
                question="How do you secure APIs?",
                answer="Use token rotation and least privilege",
                question_type="technical",
                relevance_score=8,
                depth_score=7,
                clarity_score=8,
                average_score=7.67,
                feedback="Good baseline, mention key management details.",
                was_skipped=False,
            ),
            AnswerScore(
                question="Describe a conflict in a team.",
                answer="[Skipped]",
                question_type="situational",
                relevance_score=0,
                depth_score=0,
                clarity_score=0,
                average_score=0,
                feedback="Question was skipped. No score applied.",
                was_skipped=True,
            ),
        ],
    )


def test_generate_pdf_report_creates_file(tmp_path: Path) -> None:
    output_path = generate_pdf_report(
        qa_pairs=[{"question": "Q1"}, {"question": "Q2"}],
        scores=_sample_scores(),
        role="Backend Engineer",
        company="Zepto",
        candidate_name="Candidate A",
        timestamp=datetime(2026, 4, 12, 14, 30),
        output_dir=str(tmp_path),
    )

    report_path = Path(output_path)
    assert report_path.exists()
    assert report_path.suffix.lower() == ".pdf"
    assert report_path.stat().st_size > 0


def test_report_handles_model_punctuation_and_role_separators(tmp_path: Path) -> None:
    from PyPDF2 import PdfReader

    path = Path(generate_pdf_report(
        qa_pairs=[{"question": "Q1"}],
        scores=_sample_scores(),
        role="Backend Engineer / Platform – APIs",
        company="Northstar Parcel Labs",
        candidate_name="Mira “Demo” Vale",
        output_dir=str(tmp_path),
    ))

    assert path.parent == tmp_path
    text = "".join(page.extract_text() for page in PdfReader(path).pages)
    assert 'Mira "Demo" Vale' in text
    assert "Backend Engineer / Platform - APIs" in text
