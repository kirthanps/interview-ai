"""Results page: displays scoring output and PDF download."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from pipeline.interview_conductor import get_qa_pairs_for_scoring
from pipeline.scorer import score_interview
from report.report_generator import generate_pdf_report


def _compute_scores_if_needed() -> None:
    if st.session_state.get("scores") is not None:
        return

    session = st.session_state.get("interview_session")
    if session is None:
        st.error("Interview session not found.")
        return

    qa_pairs = get_qa_pairs_for_scoring(session)
    if not qa_pairs:
        st.error("No interview answers available for scoring.")
        return

    try:
        with st.spinner("Scoring your interview..."):
            scores = score_interview(
                qa_pairs=qa_pairs,
                role=(st.session_state.get("parsed_jd")
                      or {}).get("role", "Unknown Role"),
                company=(st.session_state.get("company_profile")
                         or {}).get("company", "Unknown Company"),
            )
        st.session_state["scores"] = scores
    except Exception as exc:
        st.error(f"Error scoring interview: {exc}")


def _generate_pdf_if_needed() -> None:
    if st.session_state.get("pdf_path"):
        return

    session = st.session_state.get("interview_session")
    scores = st.session_state.get("scores")
    if session is None or scores is None:
        return

    try:
        qa_pairs = get_qa_pairs_for_scoring(session)
        pdf_path = generate_pdf_report(
            qa_pairs=qa_pairs,
            scores=scores,
            role=(st.session_state.get("parsed_jd")
                  or {}).get("role", "Unknown Role"),
            company=(st.session_state.get("company_profile")
                     or {}).get("company", "Unknown Company"),
            candidate_name=st.session_state.get("candidate_name") or None,
        )
        st.session_state["pdf_path"] = pdf_path
    except Exception as exc:
        st.error(f"Error generating PDF: {exc}")


def show_results() -> None:
    st.title("Interview Results")

    _compute_scores_if_needed()
    scores = st.session_state.get("scores")
    if scores is None:
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall", f"{scores.overall_score:.1f}/10")
    c2.metric("Technical", f"{scores.technical_depth_score:.1f}/10")
    c3.metric("Communication", f"{scores.communication_score:.1f}/10")
    c4.metric("Readiness", f"{scores.role_readiness_percent:.0f}%")

    st.info(f"Readiness Label: {scores.role_readiness_label}")

    st.subheader("What Went Well")
    for point in scores.what_went_well:
        st.success(point)

    st.subheader("Areas To Improve")
    for point in scores.areas_to_improve:
        st.warning(point)

    if scores.recommended_resources:
        st.subheader("Recommended Resources")
        for item in scores.recommended_resources:
            st.caption(f"- {item}")

    st.divider()
    st.subheader("Per-Question Breakdown")
    for idx, row in enumerate(scores.per_answer_scores, start=1):
        label = row.question[:70] + ("..." if len(row.question) > 70 else "")
        with st.expander(f"Q{idx}: {label}"):
            if row.was_skipped:
                st.error("Skipped")
            else:
                st.write(f"Relevance: {row.relevance_score:.1f}")
                st.write(f"Depth: {row.depth_score:.1f}")
                st.write(f"Clarity: {row.clarity_score:.1f}")
                st.write(f"Average: {row.average_score:.2f}")
            st.write(f"Feedback: {row.feedback}")

    _generate_pdf_if_needed()
    pdf_path = st.session_state.get("pdf_path")
    if pdf_path and Path(pdf_path).exists():
        with open(pdf_path, "rb") as file:
            st.download_button(
                label="Download PDF Report",
                data=file,
                file_name=Path(pdf_path).name,
                mime="application/pdf",
                use_container_width=True,
            )

    b1, b2 = st.columns(2)
    with b1:
        if st.button("Retry Interview", use_container_width=True):
            st.session_state["interview_session"] = None
            st.session_state["scores"] = None
            st.session_state["pdf_path"] = None
            st.session_state["page"] = "interview"
            st.rerun()
    with b2:
        if st.button("New Session", use_container_width=True):
            st.session_state["page"] = "landing"
            st.session_state["questions"] = None
            st.session_state["interview_session"] = None
            st.session_state["scores"] = None
            st.session_state["pdf_path"] = None
            st.rerun()
