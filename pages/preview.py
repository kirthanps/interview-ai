"""Preview page: show understanding of resume/JD before interview."""

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

    years = re.findall(r"\b(\d+)\+?\s+years?\b", lower_text)
    experience_level = f"{max(map(int, years))}+ years" if years else "N/A"

    education_patterns = [
        (r"\bb\.?tech\b|\bbachelor of technology\b", "BTECH"),
        (r"\bb\.?e\b|\bbachelor of engineering\b", "BE"),
        (r"\bm\.?tech\b|\bmaster of technology\b", "MTECH"),
        (r"\bmba\b|\bmaster of business administration\b", "MBA"),
        (r"\bb\.?sc\b|\bbachelor of science\b", "BSC"),
        (r"\bm\.?sc\b|\bmaster of science\b", "MSC"),
    ]
    education = "N/A"
    for pattern, label in education_patterns:
        if re.search(pattern, lower_text):
            education = label
            break

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


def show_preview():
    """Preview page - show understanding of resume/JD before interview."""

    st.title("Profile Preview")
    st.caption("Review the generated plan before starting the interview")

    if not st.session_state.get("questions"):
        st.error("No questions generated. Please go back to Setup.")
        return

    # Resume insights
    st.markdown("### Your Resume")

    resume_insights = _extract_resume_insights(st.session_state["resume_text"])
    st.session_state["resume_insights"] = resume_insights

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Experience Level**")
        st.write(resume_insights.get("experience_level", "Unknown"))

        st.markdown("**Skills Found**")
        skills = resume_insights.get("skills", [])
        if skills:
            for skill in skills[:10]:
                st.caption(f"- {skill}")
        else:
            st.caption("No specific skills detected")

    with col2:
        st.markdown("**Education**")
        st.write(resume_insights.get("education", "Unknown"))

        st.markdown("**Key Projects**")
        projects = resume_insights.get("projects", [])
        if projects:
            for proj in projects[:3]:
                st.caption(f"- {proj}")

    # JD insights
    st.divider()
    st.markdown("### Job Role")

    parsed = st.session_state["parsed_jd"]
    col1, col2 = st.columns(2)

    with col1:
        st.write(f"**Role:** {parsed.get('role', 'Unknown')}")
        st.write(f"**Company:** {parsed.get('company', 'Unknown')}")

    with col2:
        st.write(
            f"**Experience Required:** {parsed.get('experience_level', 'Unknown')}")
        st.write(
            f"**Company Focus:** {st.session_state['company_profile'].get('hiring_notes', '')[:100]}...")

    st.markdown("**Required Skills**")
    skills = parsed.get("skills", [])
    if skills:
        for skill in skills[:10]:
            st.caption(f"- {skill}")

    # Gap analysis
    st.divider()
    resume_skills = set(resume_insights.get("skills", []))
    required_skills = set(parsed.get("skills", []))
    gaps = required_skills - resume_skills

    if gaps:
        st.warning(f"⚠️ **Skill Gaps Detected**: {', '.join(list(gaps)[:5])}...\n\n"
                   f"Be prepared to discuss your approach to learning these skills!")
    else:
        st.success("✅ Your skills align well with the role requirements!")

    # Question distribution
    st.divider()
    st.markdown("### Interview Plan")

    type_dist = _question_type_distribution(st.session_state["questions"])

    dist_str = " | ".join(
        [f"{count} {qtype.title()}" for qtype, count in type_dist.items()])
    st.info(f"Distribution: {dist_str}")

    # CTA
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        if st.button("Start Interview", use_container_width=True, key="cta_start_interview"):
            st.session_state["page"] = "interview"
            st.rerun()

    if st.button("Go Back", key="go_back_setup"):
        st.session_state["page"] = "setup"
        st.rerun()
