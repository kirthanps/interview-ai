from __future__ import annotations

import json
from pathlib import Path

import pytest

from pipeline.scorer import (
    build_interview_scores,
    format_qa_transcript,
    score_interview,
    validate_scores_response,
)


class _FakeLLM:
    def __init__(self, payload: dict) -> None:
        self._payload = payload
        self.last_prompt = ""

    def invoke(self, prompt: str) -> str:
        self.last_prompt = prompt
        assert "INTERVIEW TRANSCRIPT" in prompt
        return json.dumps(self._payload)


def _qa_pairs() -> list[dict]:
    return [
        {
            "question": "Explain API authentication best practices",
            "answer": "Use JWT with refresh rotation and secure storage.",
            "type": "technical",
            "difficulty": "medium",
            "was_skipped": False,
        },
        {
            "question": "How do you handle disagreements in teams?",
            "answer": "[SKIPPED]",
            "type": "situational",
            "difficulty": "easy",
            "was_skipped": True,
        },
    ]


def test_format_qa_transcript_handles_skipped_questions() -> None:
    transcript = format_qa_transcript(_qa_pairs())
    assert "Q1:" in transcript
    assert "Q2:" in transcript
    assert "[Question was skipped]" in transcript


def test_validate_scores_response_rejects_mismatched_lengths() -> None:
    bad_payload = {
        "per_question_scores": [{"relevance": 8, "depth": 7, "clarity": 8}],
        "overall_score": 7,
        "technical_depth_score": 7,
        "communication_score": 8,
        "what_went_well": ["Good structure"],
        "areas_to_improve": ["Add metrics"],
        "role_readiness_percent": 70,
    }

    with pytest.raises(ValueError, match="length must match"):
        validate_scores_response(bad_payload, expected_questions=2)


def test_build_interview_scores_applies_skip_zero_logic() -> None:
    payload = {
        "per_question_scores": [
            {
                "relevance": 8,
                "depth": 7,
                "clarity": 8,
                "feedback": "Nice example.",
            },
            {
                "relevance": 0,
                "depth": 0,
                "clarity": 0,
                "feedback": "Skipped.",
            },
        ],
        "overall_score": 5.5,
        "technical_depth_score": 7.0,
        "communication_score": 4.0,
        "what_went_well": ["Used specific security controls"],
        "areas_to_improve": ["Expand teamwork examples"],
        "role_readiness_percent": 55,
    }

    scores = build_interview_scores(payload, _qa_pairs())

    assert len(scores.per_answer_scores) == 2
    assert scores.per_answer_scores[0].average_score == pytest.approx(
        (8 + 7 + 8) / 3)
    assert scores.per_answer_scores[1].was_skipped is True
    assert scores.per_answer_scores[1].average_score == 0


def test_score_interview_with_injected_llm(tmp_path: Path) -> None:
    prompt_file = tmp_path / "scorer_prompt.txt"
    prompt_file.write_text(
        (
            "Role={role}\nCompany={company}\n{transcript}\n"
            "RESUME={resume_insights}\nJD={jd_insights}\nQUESTIONS={generated_questions}"
        ),
        encoding="utf-8",
    )

    payload = {
        "per_question_scores": [
            {
                "relevance": 8,
                "depth": 7,
                "clarity": 8,
                "feedback": "Clear, add one production metric.",
            },
            {
                "relevance": 0,
                "depth": 0,
                "clarity": 0,
                "feedback": "Skipped answer.",
            },
        ],
        "overall_score": 6.0,
        "technical_depth_score": 7.0,
        "communication_score": 5.0,
        "what_went_well": ["Explained auth flow with practical tradeoffs"],
        "areas_to_improve": ["Provide conflict resolution examples with outcomes"],
        "role_readiness_percent": 60,
        "role_readiness_label": "Good Foundation",
        "recommended_resources": ["System Design Primer", "Grokking Behavioral Interviews"],
    }

    fake_llm = _FakeLLM(payload)
    scores = score_interview(
        qa_pairs=_qa_pairs(),
        role="Backend Engineer",
        company="Zepto",
        resume_insights={"skills": ["Python"]},
        jd_insights={"role": "Backend Engineer"},
        generated_questions=[{"question": "Q1"}],
        llm=fake_llm,
        prompt_path=str(prompt_file),
    )

    as_dict = scores.to_dict()
    assert as_dict["overall_score"] == 6.0
    assert as_dict["role_readiness_label"] == "Good Foundation"
    assert len(as_dict["per_answer_scores"]) == 2
    assert "RESUME=" in fake_llm.last_prompt
    assert "QUESTIONS=" in fake_llm.last_prompt
