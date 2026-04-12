"""Landing page for InterviewAI Streamlit app."""

from __future__ import annotations

import streamlit as st

from utils.constants import SUPPORTED_ROLES


def show_landing() -> None:
    st.markdown('<div class="header-title">InterviewAI</div>',
                unsafe_allow_html=True)
    st.write("Personalized AI mock interviews for Indian job roles.")

    st.markdown("### Why InterviewAI")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="subtle-card"><b>Resume-Aware</b><br/>Questions are grounded in your resume context.</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="subtle-card"><b>Company Context</b><br/>Question style adapts using company profile retrieval.</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="subtle-card"><b>Scored Feedback</b><br/>Get relevance, depth, and clarity scores with tips.</div>', unsafe_allow_html=True)

    st.markdown("### Supported Roles")
    st.caption(" | ".join(SUPPORTED_ROLES[:6]) + " | ...")

    st.divider()
    if st.button("Start Mock Interview", use_container_width=True):
        st.session_state["page"] = "setup"
        st.rerun()

    with st.expander("How it works"):
        st.write("1. Upload resume PDF")
        st.write("2. Paste job description")
        st.write("3. Review generated interview plan")
        st.write("4. Complete interview and download report")
