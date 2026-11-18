"""Setup page: collect resume, JD, and generation preferences."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import streamlit as st

from pipeline.jd_parser import match_to_company_profile, parse_jd_with_llm
from pipeline.question_generator import generate_questions
from pipeline.resume_processor import (
    chunk_text,
    extract_resume_insights_with_llm,
    extract_text_from_pdf,
    generate_embeddings,
    initialize_chroma_db,
    retrieve_resume_context,
    store_jd_in_chroma,
    store_resume_in_chroma,
)

logger = logging.getLogger(__name__)


def _save_uploaded_pdf_temp(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def show_setup() -> None:
    st.title("Setup Interview")
    st.caption("Upload resume, paste JD, and choose preferences.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Resume")
        uploaded_file = st.file_uploader("Upload resume PDF", type=["pdf"])
        candidate_name = st.text_input(
            "Candidate name (optional)", value=st.session_state.get("candidate_name", ""))
        st.session_state["candidate_name"] = candidate_name

        if uploaded_file is not None:
            temp_path = _save_uploaded_pdf_temp(uploaded_file)
            try:
                with st.spinner("Extracting text from resume..."):
                    resume_text = extract_text_from_pdf(str(temp_path))
                st.session_state["resume_text"] = resume_text
                st.session_state["resume_file"] = uploaded_file.name
                st.success(f"Resume processed ({len(resume_text)} chars)")
            except Exception as exc:
                st.error(f"Failed to process resume: {exc}")
            finally:
                temp_path.unlink(missing_ok=True)

    with col2:
        st.subheader("Job Description")
        jd_text = st.text_area(
            "Paste full JD",
            value=st.session_state.get("jd_text", ""),
            height=220,
            placeholder="Include role, company, skills, and responsibilities...",
        )
        st.session_state["jd_text"] = jd_text

        if jd_text and len(jd_text.strip()) < 100:
            st.warning("JD should be at least 100 characters.")

    st.markdown("### Preferences")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=[
                                  "easy", "medium", "hard"].index(st.session_state["preferences"].get("difficulty", "medium")))
    with p_col2:
        num_questions = st.selectbox("Questions", [5, 8, 10], index=[5, 8, 10].index(
            st.session_state["preferences"].get("num_questions", 8)))
    with p_col3:
        focus_area = st.selectbox("Focus", ["technical", "hr", "mixed", "system_design"], index=[
                                  "technical", "hr", "mixed", "system_design"].index(st.session_state["preferences"].get("focus_area", "mixed")))

    st.session_state["preferences"] = {
        "difficulty": difficulty,
        "num_questions": int(num_questions),
        "focus_area": focus_area,
    }

    ready = bool(st.session_state.get("resume_text") and st.session_state.get(
        "jd_text") and len(st.session_state["jd_text"].strip()) >= 100)

    if st.button("Analyze and Generate Questions", disabled=not ready, use_container_width=True):
        try:
            with st.spinner("Running resume + JD analysis..."):
                client = initialize_chroma_db()

                chunks = chunk_text(st.session_state["resume_text"])
                embeddings, _ = generate_embeddings(chunks)
                store_resume_in_chroma(
                    chunks=chunks,
                    embeddings=embeddings,
                    session_id=st.session_state["session_id"],
                    client=client,
                )

                store_jd_in_chroma(
                    jd_text=st.session_state["jd_text"],
                    session_id=st.session_state["session_id"],
                    client=client,
                )

                resume_insights = extract_resume_insights_with_llm(
                    st.session_state["resume_text"]
                )
                st.session_state["resume_insights"] = resume_insights

                parsed_jd = parse_jd_with_llm(
                    st.session_state["jd_text"],
                    chroma_client=client,
                    k_examples=3,
                )
                st.session_state["parsed_jd"] = parsed_jd

                company_profile = match_to_company_profile(parsed_jd, client)
                st.session_state["company_profile"] = company_profile

                skill_query = " ".join(parsed_jd.get(
                    "skills", [])) or parsed_jd.get("role", "")
                resume_context = retrieve_resume_context(
                    query=skill_query,
                    session_id=st.session_state["session_id"],
                    k=5,
                    client=client,
                )

                questions = generate_questions(
                    parsed_jd=parsed_jd,
                    company_profile=company_profile,
                    resume_context=resume_context,
                    preferences=st.session_state["preferences"],
                    chroma_client=client,
                    resume_text=st.session_state["resume_text"],
                )

                st.session_state["questions"] = questions

                st.success("Questions generated successfully")
                st.session_state["page"] = "preview"
                st.rerun()

        except Exception as exc:
            logger.error(f"Error in question generation: {str(exc)}")
            st.error(
                f"❌ Error: {str(exc)}\n\nPlease check your JD and try again.")

    st.caption(
        "🔒 Your resume is processed securely and cleared after your session.")
