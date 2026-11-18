"""Setup page: collect resume, JD, and generation preferences."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from utils.groq_helpers import get_groq_api_key
from utils.pdf_input import extract_text_from_pdf


@st.cache_resource(show_spinner=False)
def _get_search_index():
    from pipeline.resume_processor import initialize_chroma_db, initialize_datasets_in_chromadb

    client = initialize_chroma_db()
    initialize_datasets_in_chromadb(client)
    return client


def _save_uploaded_pdf_temp(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def show_setup() -> None:
    st.title("Setup Interview")
    st.caption("Upload a text-based PDF, paste the job description, and choose your interview preferences.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Resume")
        uploaded_file = st.file_uploader("Upload resume PDF", type=["pdf"])
        candidate_name = st.text_input(
            "Candidate name (optional)", value=st.session_state.get("candidate_name", ""))
        st.session_state["candidate_name"] = candidate_name

        st.session_state["resume_text"] = ""
        st.session_state["resume_file"] = None
        if uploaded_file is not None:
            temp_path = _save_uploaded_pdf_temp(uploaded_file)
            try:
                with st.spinner("Extracting text from resume..."):
                    resume_text = extract_text_from_pdf(str(temp_path))
                st.session_state["resume_text"] = resume_text
                st.session_state["resume_file"] = uploaded_file.name
                st.success(f"Resume processed ({len(resume_text)} chars)")
            except Exception:
                st.error("Could not read this PDF. Upload an unlocked PDF with selectable text.")
            finally:
                temp_path.unlink(missing_ok=True)

    with col2:
        st.subheader("Job Description")
        jd_text = st.text_area(
            "Job description (minimum 100 characters)",
            value=st.session_state.get("jd_text", ""),
            height=220,
            placeholder="Include role, company, skills, and responsibilities...",
        )
        st.session_state["jd_text"] = jd_text
        company_context = st.text_area(
            "Company context (optional)",
            value=st.session_state.get("company_context", ""),
            placeholder="Add relevant details about the team, product, or interview focus.",
        )

        st.session_state["company_context"] = company_context

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

    has_key = bool(get_groq_api_key())
    if not has_key:
        st.info("Add GROQ_API_KEY to the local .env file to enable question generation.")

    if st.button("Analyze and Generate Questions", disabled=not (ready and has_key), use_container_width=True):
        try:
            from pipeline.jd_parser import match_to_company_profile, parse_jd_with_llm
            from pipeline.question_generator import generate_questions
            from pipeline.resume_processor import (
                chunk_text,
                extract_resume_insights_with_llm,
                generate_embeddings,
                retrieve_resume_context,
                store_jd_in_chroma,
                store_resume_in_chroma,
            )

            with st.spinner("Preparing retrieval and analyzing your resume and job description. The first run may download the embedding model."):
                client = _get_search_index()

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

                parsed_jd = parse_jd_with_llm(
                    st.session_state["jd_text"],
                    chroma_client=client,
                    k_examples=3,
                )

                company_profile = match_to_company_profile(parsed_jd, client)
                if company_context.strip():
                    company_profile["provided_context"] = company_context.strip()

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

                st.session_state.update(
                    questions=questions,
                    resume_insights=resume_insights,
                    parsed_jd=parsed_jd,
                    company_profile=company_profile,
                    interview_session=None,
                    scores=None,
                    pdf_path=None,
                    page="preview",
                )
                st.rerun()

        except ImportError:
            st.error("The interview dependencies are unavailable. See the local setup section in the README.")
        except FileNotFoundError:
            st.error("The local reference datasets are missing. Run the dataset generator described in the README.")
        except Exception:
            st.error("Question generation failed. Check your Groq configuration and network connection, then try again.")

    st.caption("Resume and job details are sent to Groq for analysis. Use fictional data for demos.")
