"""InterviewAI Streamlit application entry point."""

from __future__ import annotations

import uuid

import streamlit as st
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(
    page_title="InterviewAI - Mock Interview Coach",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _render_styles() -> None:
    st.markdown(
        """
        <style>
            .header-title {
                font-size: 2.4rem;
                font-weight: 800;
                color: #0f4c81;
                margin-bottom: 8px;
                letter-spacing: 0.3px;
            }
            .subtle-card {
                background: linear-gradient(180deg, #f8fbff 0%, #eef4fb 100%);
                padding: 16px;
                border-radius: 12px;
                border: 1px solid #dce7f5;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_session_state() -> None:
    """Initialize all required keys once per user session."""
    defaults = {
        "page": "landing",
        "session_id": str(uuid.uuid4()),
        "resume_file": None,
        "resume_text": "",
        "jd_text": "",
        "preferences": {
            "difficulty": "medium",
            "num_questions": 8,
            "focus_area": "mixed",
        },
        "parsed_jd": None,
        "company_profile": None,
        "resume_insights": None,
        "questions": None,
        "interview_session": None,
        "qa_pairs": [],
        "scores": None,
        "pdf_path": None,
        "candidate_name": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _route_current_page() -> None:
    page = st.session_state["page"]
    if page == "landing":
        from pages.landing import show_landing

        show_landing()
    elif page == "setup":
        from pages.setup import show_setup

        show_setup()
    elif page == "preview":
        from pages.preview import show_preview

        show_preview()
    elif page == "interview":
        from pages.interview import show_interview

        show_interview()
    elif page == "results":
        from pages.results import show_results

        show_results()
    else:
        st.session_state["page"] = "landing"
        st.rerun()


def _render_sidebar() -> None:
    with st.sidebar:
        st.title("InterviewAI")

        page_labels = {
            "Landing": "landing",
            "Setup": "setup",
            "Preview": "preview",
            "Interview": "interview",
            "Results": "results",
        }

        for label, page_name in page_labels.items():
            if st.button(label, use_container_width=True, key=f"nav_{page_name}"):
                if page_name in {"preview", "interview", "results"} and not st.session_state.get("questions"):
                    st.warning("Complete setup before opening this page.")
                elif page_name == "results" and not (
                    (session := st.session_state.get("interview_session")) and session.is_completed()
                ):
                    st.warning("Finish the interview to view results.")
                else:
                    st.session_state["page"] = page_name
                    st.rerun()

        st.divider()
        if st.button("Restart Session", use_container_width=True):
            st.session_state.clear()
            st.rerun()


def main() -> None:
    _render_styles()
    initialize_session_state()
    _render_sidebar()
    _route_current_page()


if __name__ == "__main__":
    main()
