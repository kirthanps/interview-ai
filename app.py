"""InterviewAI Streamlit application entry point."""

from __future__ import annotations

import logging
import os
import uuid

import streamlit as st
from dotenv import load_dotenv

from pipeline.resume_processor import initialize_chroma_db, initialize_datasets_in_chromadb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        "scores": None,
        "pdf_path": None,
        "candidate_name": "",
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_resource
def get_chroma_client():
    """Initialize and cache ChromaDB client."""
    return initialize_chroma_db()


@st.cache_resource
def warmup_datasets() -> None:
    """Load shared datasets into Chroma once per process."""
    try:
        client = get_chroma_client()
        initialize_datasets_in_chromadb(client)
        logger.info("Datasets loaded into ChromaDB")
    except Exception as exc:  # pragma: no cover
        logger.warning("Dataset warmup skipped: %s", exc)


def _reset_for_new_session() -> None:
    keys_to_reset = [
        "page",
        "resume_file",
        "resume_text",
        "jd_text",
        "parsed_jd",
        "company_profile",
        "resume_insights",
        "questions",
        "interview_session",
        "scores",
        "pdf_path",
    ]
    for key in keys_to_reset:
        if key == "page":
            st.session_state[key] = "landing"
        elif key in {"resume_text", "jd_text"}:
            st.session_state[key] = ""
        else:
            st.session_state[key] = None

    st.session_state["session_id"] = str(uuid.uuid4())


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
        st.caption(f"Session: {st.session_state['session_id'][-8:]}")

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
                else:
                    st.session_state["page"] = page_name
                    st.rerun()

        st.divider()
        if st.button("Restart Session", use_container_width=True):
            _reset_for_new_session()
            st.rerun()


def main() -> None:
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not found in .env file.")
        st.stop()

    _render_styles()
    initialize_session_state()
    warmup_datasets()
    _render_sidebar()
    _route_current_page()


if __name__ == "__main__":
    main()
