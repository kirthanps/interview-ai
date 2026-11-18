"""Landing page for InterviewAI Streamlit app."""

from __future__ import annotations

import streamlit as st

from utils.constants import SUPPORTED_ROLES


def show_landing() -> None:
    st.markdown('<h1 class="header-title">🎤 InterviewAI</h1>',
                unsafe_allow_html=True)
    st.write("Your personal mock interview coach for the Indian job market")
    st.markdown("""
**InterviewAI** conducts personalized, AI-powered mock interviews tailored to YOUR resume 
and the job you're applying for. Get scored feedback, actionable improvement tips, and a 
professional PDF report.
    """)

    # Feature highlights
    st.markdown("### ✨ Why InterviewAI?")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="subtle-card">
        <h4>📄 Resume-Aware</h4>
        Questions draw on the experience in your resume
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="subtle-card">
        <h4>🏢 Company Context</h4>
        Practice for the role and company you are targeting
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="subtle-card">
        <h4>📊 Scored Feedback</h4>
        Specific, actionable advice with resources to improve
        </div>
        """, unsafe_allow_html=True)

    # Supported roles
    st.markdown("### 💼 Supported Roles")
    roles_str = " • ".join(SUPPORTED_ROLES[:5]) + " • ..."
    st.caption(roles_str)

    # CTA button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🚀 Start My Mock Interview", use_container_width=True,
                     key="cta_start"):
            st.session_state["page"] = "setup"
            st.rerun()

    # How it works
    with st.expander("❓ How does this work?"):
        st.markdown("""
        1. **Upload Resume**: PDF of your resume
        2. **Paste Job Description**: The role you are interviewing for
        3. **Answer Questions**: Real-time AI-tailored interview (5-10 questions)
        4. **Get Report**: Scored feedback + PDF download
        
        **Behind the scenes**: 
        - PDF extraction and vector search run locally
        - Questions are generated using AI specifically for your background
        - Answers are scored using strict rubric
        - Groq receives interview context and answers to generate feedback
        """)

    st.markdown("---")
    st.caption(
        "For portfolio demos, use a fictional resume and job description.")
