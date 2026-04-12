"""Interview page: conducts multi-turn Q&A using Module 3."""

from __future__ import annotations

import streamlit as st

from pipeline.interview_conductor import (
    finalize_interview,
    get_current_question,
    initialize_interview_session,
    skip_question,
    submit_answer,
)


def _ensure_session_initialized() -> None:
    if st.session_state.get("interview_session") is not None:
        return

    st.session_state["interview_session"] = initialize_interview_session(
        session_id=st.session_state["session_id"],
        questions=st.session_state.get("questions", []),
        role=(st.session_state.get("parsed_jd")
              or {}).get("role", "Unknown Role"),
        company=(st.session_state.get("company_profile")
                 or {}).get("company", "Unknown Company"),
        resume_insights=st.session_state.get("resume_insights") or {},
        candidate_name=st.session_state.get("candidate_name") or None,
    )


def show_interview() -> None:
    st.title("Interview In Progress")

    if not st.session_state.get("questions"):
        st.error("No questions available. Go to setup first.")
        return

    _ensure_session_initialized()
    session = st.session_state["interview_session"]

    st.progress(session.get_progress_percentage() / 100.0)
    st.caption(
        f"Answered/Skipped: {len(session.qa_pairs)} / {len(session.questions)}")

    current_question = get_current_question(session)
    if current_question is None:
        finalize_interview(session)
        st.success("Interview completed.")
        if st.button("View Results", use_container_width=True):
            st.session_state["page"] = "results"
            st.rerun()
        return

    st.subheader(f"Question {session.current_question_index + 1}")
    st.write(current_question.question)
    st.caption(
        f"Type: {current_question.type} | Difficulty: {current_question.difficulty}"
    )

    with st.expander("Hints"):
        for hint in current_question.hint:
            st.caption(f"- {hint}")

    answer_key = f"answer_input_{session.current_question_index}"
    answer_text = st.text_area(
        "Your Answer",
        key=answer_key,
        height=160,
        placeholder="Provide a clear and structured answer...",
    )

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Submit Answer", use_container_width=True):
            result = submit_answer(session, answer_text)
            if result["success"]:
                st.rerun()
            else:
                st.error(result["message"])
                for err in result.get("validation_errors", []):
                    st.caption(f"- {err}")

    with c2:
        if st.button("Skip Question", use_container_width=True):
            result = skip_question(session)
            if result["success"]:
                st.rerun()
            else:
                st.error(result["message"])

    if session.qa_pairs:
        with st.expander("Previous Responses", expanded=False):
            for idx, qa in enumerate(session.qa_pairs, start=1):
                st.markdown(f"**Q{idx}:** {qa.question}")
                if qa.was_skipped:
                    st.caption("[Skipped]")
                else:
                    st.write(qa.answer)
                st.divider()
