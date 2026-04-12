from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.interview_conductor import (  # noqa: E402
    InterviewQuestion,
    advance_to_next_question,
    finalize_interview,
    get_conversation_history,
    get_current_question,
    get_first_question,
    get_qa_pairs_for_scoring,
    initialize_interview_session,
    skip_question,
    submit_answer,
)


def _sample_questions() -> list[dict]:
    return [
        {
            "question": "Explain how you designed your API authentication flow.",
            "type": "technical",
            "hint": ["tokens", "rotation"],
            "difficulty": "easy",
        },
        {
            "question": "Design a scalable order service for peak traffic.",
            "type": "system_design",
            "hint": ["queues", "cache"],
            "difficulty": "medium",
        },
        {
            "question": "Tell me about a production incident you handled.",
            "type": "situational",
            "hint": ["timeline", "ownership"],
            "difficulty": "hard",
        },
    ]


def test_initialize_session_and_first_question() -> None:
    session = initialize_interview_session(
        session_id="s1",
        questions=_sample_questions(),
        role="Backend Engineer",
        company="Zepto",
        resume_insights={},
    )

    first = get_first_question(session)
    assert isinstance(first, InterviewQuestion)
    assert first.question
    assert session.current_question_index == 0


def test_submit_answer_validation_short_rejected() -> None:
    session = initialize_interview_session(
        session_id="s2",
        questions=_sample_questions(),
        role="Backend Engineer",
        company="Zepto",
        resume_insights={},
    )

    result = submit_answer(session, "too short", validate=True)
    assert result["success"] is False
    assert "Answer too short" in result["validation_errors"][0]
    assert session.current_question_index == 0


def test_submit_answer_advances_and_persists() -> None:
    session = initialize_interview_session(
        session_id="s3",
        questions=_sample_questions(),
        role="Backend Engineer",
        company="Zepto",
        resume_insights={},
    )

    answer = "I used short-lived access tokens, refresh token rotation, and revocation lists."
    result = submit_answer(session, answer, validate=True)

    assert result["success"] is True
    assert session.current_question_index == 1
    assert len(session.qa_pairs) == 1
    assert session.qa_pairs[0].answer == answer
    assert session.qa_pairs[0].was_skipped is False


def test_skip_question_marks_and_advances() -> None:
    session = initialize_interview_session(
        session_id="s4",
        questions=_sample_questions(),
        role="Backend Engineer",
        company="Zepto",
        resume_insights={},
    )

    result = skip_question(session)

    assert result["success"] is True
    assert session.current_question_index == 1
    assert len(session.qa_pairs) == 1
    assert session.qa_pairs[0].was_skipped is True
    assert session.qa_pairs[0].answer == "[SKIPPED]"


def test_full_flow_completion_and_summary() -> None:
    session = initialize_interview_session(
        session_id="s5",
        questions=_sample_questions(),
        role="Backend Engineer",
        company="Zepto",
        resume_insights={"skills": ["Python"]},
    )

    submit_answer(
        session,
        "I handled auth by splitting token scopes and enforcing rotation policies.",
        validate=True,
    )
    skip_question(session)
    submit_answer(
        session,
        "I led incident triage, rollback, and postmortem with action items.",
        validate=True,
    )

    assert get_current_question(session) is None
    assert advance_to_next_question(session) is False
    assert session.is_completed() is True
    assert session.get_progress_percentage() == 100.0

    history = get_conversation_history(session)
    assert "Interview for Backend Engineer at Zepto" in history
    assert "[Question Skipped]" in history

    scoring_payload = get_qa_pairs_for_scoring(session)
    assert len(scoring_payload) == 3
    assert any(item["was_skipped"] for item in scoring_payload)

    summary = finalize_interview(session)
    assert summary["total_questions"] == 3
    assert summary["answered_questions"] == 2
    assert summary["skipped_questions"] == 1
