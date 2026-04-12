"""Preview page: confirm parsed insights before interview."""

from __future__ import annotations

import re
from collections import Counter

import streamlit as st


def _extract_resume_insights(resume_text: str) -> dict:
    """Heuristic resume insight extraction for preview display."""
    text = resume_text or ""
    lower_text = text.lower()

    known_skills = [
        "python",
        "java",
        "sql",
        "django",
        "flask",
        "react",
        "aws",
        "docker",
        "kubernetes",
        "pandas",
        "numpy",
        "machine learning",
    ]
    found_skills = [skill for skill in known_skills if skill in lower_text]

    years = re.findall(r"(\d+)\+?\s+years", lower_text)
    experience_level = f"{max(map(int, years))}+ years" if years else "Not explicitly stated"

    education_keywords = ["btech", "b.e", "mtech", "bsc", "msc", "mba"]
    education = next(
        (word.upper() for word in education_keywords if word in lower_text), "Not detected")

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    project_lines = [line for line in lines if any(token in line.lower(
    ) for token in ["project", "built", "developed", "implemented"])][:3]

    return {
        "skills": sorted(found_skills),
        "experience_level": experience_level,
        "education": education,
        "projects": project_lines,
    }


def _question_type_distribution(questions) -> dict:
    counts: Counter = Counter()
    for q in questions or []:
        q_type = getattr(q, "type", None)
        if not q_type and isinstance(q, dict):
            q_type = q.get("type", "technical")
        counts[str(q_type or "technical")] += 1
    return dict(counts)


def show_preview() -> None:
    st.title("Preview Interview Plan")

    if not st.session_state.get("questions"):
        st.error("No generated questions found. Complete setup first.")
        return

    parsed_jd = st.session_state.get("parsed_jd") or {}
    company_profile = st.session_state.get("company_profile") or {}

    resume_insights = _extract_resume_insights(
        st.session_state.get("resume_text", ""))
    st.session_state["resume_insights"] = resume_insights

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Resume Insights")
        st.write(
            f"Experience: {resume_insights.get('experience_level', 'Unknown')}")
        st.write(f"Education: {resume_insights.get('education', 'Unknown')}")
        st.write("Skills:")
        for skill in resume_insights.get("skills", [])[:10]:
            st.caption(f"- {skill}")

    with c2:
        st.subheader("JD Insights")
        st.write(f"Role: {parsed_jd.get('role', 'Unknown')}")
        st.write(
            f"Company: {company_profile.get('company', parsed_jd.get('company', 'Unknown'))}")
        st.write(
            f"Experience Required: {parsed_jd.get('experience_level', 'Unknown')}")
        st.write("Required skills:")
        for skill in parsed_jd.get("skills", [])[:10]:
            st.caption(f"- {skill}")

    st.divider()
    dist = _question_type_distribution(st.session_state.get("questions"))
    st.info("Question Distribution: " +
            " | ".join([f"{k}: {v}" for k, v in dist.items()]))

    st.subheader("Company Hiring Notes")
    st.write(company_profile.get("hiring_notes", "Standard interview process."))

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        if st.button("Start Interview", use_container_width=True):
            st.session_state["page"] = "interview"
            st.rerun()
    with btn_col2:
        if st.button("Back to Setup", use_container_width=True):
            st.session_state["page"] = "setup"
            st.rerun()
