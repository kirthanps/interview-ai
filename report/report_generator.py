"""PDF report generation utilities for Module 4."""

from __future__ import annotations

import logging
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fpdf import FPDF

from pipeline.scorer import InterviewScores

logger = logging.getLogger(__name__)


class InterviewReportPDF(FPDF):
    """FPDF subclass with simple branded header/footer."""

    def header(self) -> None:  # pragma: no cover
        self.set_text_color(15, 76, 129)
        self.set_font("Helvetica", "B", 16)
        self.cell(0, 8, "InterviewAI Assessment Report", new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_font("Helvetica", "", 9)
        self.cell(0, 5, "AI-Powered Mock Interview Coach", new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_draw_color(205, 220, 233)
        self.line(self.l_margin, self.get_y() + 1, self.w - self.r_margin, self.get_y() + 1)
        self.ln(5)
        self.set_text_color(30, 41, 59)

    def footer(self) -> None:  # pragma: no cover
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 5, f"Page {self.page_no()}", align="C")


def _safe_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).translate(str.maketrans({
        "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "•": "-", "…": "...",
    }))
    return " ".join(text.split()).encode("latin-1", errors="replace").decode("latin-1")


def _render_list_section(pdf: FPDF, title: str, items: List[str]) -> None:
    full_width = pdf.w - pdf.l_margin - pdf.r_margin
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)

    if not items:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(full_width, 5, "- Not available")
        pdf.ln(2)
        return

    for item in items:
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(full_width, 5, f"- {_safe_text(item)}")
    pdf.ln(2)


def generate_pdf_report(
    qa_pairs: List[Dict[str, Any]],
    scores: InterviewScores,
    role: str,
    company: str,
    candidate_name: Optional[str] = None,
    timestamp: Optional[datetime] = None,
    output_dir: str = "reports",
) -> str:
    """Generate a professional PDF interview report and return file path."""
    try:
        generated_at = timestamp or datetime.now()
        role_text = _safe_text(role) or "UnknownRole"
        company_text = _safe_text(company) or "UnknownCompany"

        pdf = InterviewReportPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        full_width = pdf.w - pdf.l_margin - pdf.r_margin

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Session Details", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)

        details = [f"Role: {role_text}", f"Company: {company_text}"]
        if candidate_name and candidate_name.strip():
            details.insert(0, f"Candidate: {_safe_text(candidate_name)}")
        for detail in details:
            pdf.multi_cell(full_width, 6, detail, new_x="LMARGIN", new_y="NEXT")
        pdf.cell(
            0,
            6,
            f"Date: {generated_at.strftime('%B %d, %Y at %H:%M')}",
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.cell(0, 6, f"Total Questions: {len(qa_pairs)}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Performance Scores", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.cell(
            0, 6, f"Overall Score: {scores.overall_score:.1f}/10", new_x="LMARGIN", new_y="NEXT")
        pdf.cell(
            0,
            6,
            f"Technical Depth: {scores.technical_depth_score:.1f}/10",
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.cell(
            0,
            6,
            f"Communication: {scores.communication_score:.1f}/10",
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.cell(
            0,
            6,
            (
                f"Role Readiness: {scores.role_readiness_percent:.0f}% "
                f"({_safe_text(scores.role_readiness_label)})"
            ),
            new_x="LMARGIN", new_y="NEXT",
        )
        pdf.ln(3)

        _render_list_section(pdf, "What Went Well", scores.what_went_well)
        _render_list_section(pdf, "Areas To Improve", scores.areas_to_improve)
        _render_list_section(pdf, "Recommended Resources",
                             scores.recommended_resources)

        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Per-Question Breakdown", new_x="LMARGIN", new_y="NEXT")

        for idx, answer_score in enumerate(scores.per_answer_scores, start=1):
            pdf.set_font("Helvetica", "B", 10)
            prefix = f"Q{idx} ({_safe_text(answer_score.question_type)})"
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(
                full_width, 5, f"{prefix}: {_safe_text(answer_score.question)}")

            pdf.set_font("Helvetica", "", 9)
            if answer_score.was_skipped:
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(
                    full_width, 5, "Skipped by candidate. Score: 0.0")
            else:
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(
                    full_width,
                    5,
                    (
                        f"Relevance: {answer_score.relevance_score:.1f} | "
                        f"Depth: {answer_score.depth_score:.1f} | "
                        f"Clarity: {answer_score.clarity_score:.1f} | "
                        f"Average: {answer_score.average_score:.1f}"
                    ),
                )
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(
                    full_width, 5, f"Feedback: {_safe_text(answer_score.feedback)}")
            pdf.ln(1)

        timestamp_tag = generated_at.strftime("%Y%m%d_%H%M%S")
        role_slug = re.sub(r"[^A-Za-z0-9_-]+", "_", role_text).strip("_")[:48] or "UnknownRole"
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        report_file = output_path / \
            f"InterviewAI_Report_{role_slug}_{timestamp_tag}_{uuid4().hex[:8]}.pdf"

        pdf.output(str(report_file))
        logger.info("PDF report generated at %s", report_file)
        return str(report_file)

    except Exception:
        logger.exception("Error generating PDF report")
        raise
