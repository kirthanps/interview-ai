from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

from pages.preview import _extract_resume_insights, _question_type_distribution
from pages.setup import _save_uploaded_pdf_temp
import pages.results as results_page


class _FakeUploadedFile:
    def __init__(self, name: str, payload: bytes) -> None:
        self.name = name
        self._payload = payload

    def getbuffer(self) -> bytes:
        return self._payload


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state = {}
        self.errors = []

    def error(self, message: str) -> None:
        self.errors.append(message)

    def spinner(self, _label: str):
        return nullcontext()


def test_extract_resume_insights_detects_core_fields() -> None:
    resume = """
    BTech in Computer Science
    3 years of experience in backend systems
    Built APIs with Python, Django, SQL and AWS
    """

    insights = _extract_resume_insights(resume)

    assert insights["education"] == "BTECH"
    assert "python" in insights["skills"]
    assert "django" in insights["skills"]
    assert insights["experience_level"].startswith("3")


def test_question_type_distribution_works_for_objects_and_dicts() -> None:
    class _Q:
        def __init__(self, q_type: str) -> None:
            self.type = q_type

    dist = _question_type_distribution(
        [_Q("technical"), {"type": "hr"}, {"x": 1}])
    assert dist["technical"] == 2
    assert dist["hr"] == 1


def test_save_uploaded_pdf_temp_writes_file() -> None:
    uploaded = _FakeUploadedFile("resume.pdf", b"test-bytes")
    temp_path = _save_uploaded_pdf_temp(uploaded)

    try:
        assert temp_path.exists()
        assert temp_path.suffix == ".pdf"
        assert temp_path.read_bytes() == b"test-bytes"
    finally:
        temp_path.unlink(missing_ok=True)


def test_results_compute_scores_handles_missing_session(monkeypatch) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(results_page, "st", fake_st)

    results_page._compute_scores_if_needed()

    assert fake_st.errors
    assert "Interview session not found" in fake_st.errors[-1]


def test_results_compute_scores_success_sets_scores(monkeypatch) -> None:
    fake_st = _FakeStreamlit()
    fake_st.session_state = {
        "interview_session": object(),
        "parsed_jd": {"role": "Backend Engineer"},
        "company_profile": {"company": "Zepto"},
        "scores": None,
    }

    monkeypatch.setattr(results_page, "st", fake_st)
    monkeypatch.setattr(results_page, "get_qa_pairs_for_scoring", lambda _s: [
                        {"question": "q", "answer": "a", "was_skipped": False}])
    sentinel = object()
    monkeypatch.setattr(results_page, "score_interview",
                        lambda **_kwargs: sentinel)

    results_page._compute_scores_if_needed()

    assert fake_st.session_state["scores"] is sentinel


def test_results_generate_pdf_handles_exception(monkeypatch) -> None:
    fake_st = _FakeStreamlit()
    fake_st.session_state = {
        "interview_session": object(),
        "scores": object(),
        "parsed_jd": {"role": "Backend Engineer"},
        "company_profile": {"company": "Zepto"},
        "candidate_name": "",
        "pdf_path": None,
    }

    monkeypatch.setattr(results_page, "st", fake_st)
    monkeypatch.setattr(results_page, "get_qa_pairs_for_scoring", lambda _s: [
                        {"question": "q", "answer": "a", "was_skipped": False}])

    def _raise_pdf(**_kwargs):
        raise RuntimeError("pdf boom")

    monkeypatch.setattr(results_page, "generate_pdf_report", _raise_pdf)

    results_page._generate_pdf_if_needed()

    assert fake_st.errors
    assert "Error generating PDF" in fake_st.errors[-1]
    assert fake_st.session_state["pdf_path"] is None
