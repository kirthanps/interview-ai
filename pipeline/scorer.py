"""Scoring and feedback utilities for Module 4."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.prompts import PromptTemplate

from utils.groq_helpers import (
    build_groq_llm,
    get_groq_model_candidates,
    is_model_decommission_error,
)

logger = logging.getLogger(__name__)

GROQ_SCORING_TEMPERATURE = 0.3
TRANSCRIPT_ANSWER_MAX_CHARS = 500


@dataclass
class AnswerScore:
    """Score payload for one question-answer pair."""

    question: str
    answer: str
    question_type: str
    relevance_score: float
    depth_score: float
    clarity_score: float
    average_score: float
    feedback: str
    was_skipped: bool = False


@dataclass
class InterviewScores:
    """Aggregated interview scoring and feedback payload."""

    overall_score: float
    technical_depth_score: float
    communication_score: float
    what_went_well: List[str]
    areas_to_improve: List[str]
    role_readiness_percent: float
    role_readiness_label: str
    per_answer_scores: List[AnswerScore] = field(default_factory=list)
    recommended_resources: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a JSON-serializable payload."""
        return {
            "overall_score": round(self.overall_score, 2),
            "technical_depth_score": round(self.technical_depth_score, 2),
            "communication_score": round(self.communication_score, 2),
            "what_went_well": list(self.what_went_well),
            "areas_to_improve": list(self.areas_to_improve),
            "role_readiness_percent": round(self.role_readiness_percent, 0),
            "role_readiness_label": self.role_readiness_label,
            "recommended_resources": list(self.recommended_resources),
            "per_answer_scores": [
                {
                    "question": item.question,
                    "answer": item.answer[:100] + "..." if len(item.answer) > 100 else item.answer,
                    "relevance": round(item.relevance_score, 1),
                    "depth": round(item.depth_score, 1),
                    "clarity": round(item.clarity_score, 1),
                    "average": round(item.average_score, 1),
                    "feedback": item.feedback,
                    "was_skipped": item.was_skipped,
                }
                for item in self.per_answer_scores
            ],
        }


def _load_prompt_template(prompt_path: str) -> str:
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return path.read_text(encoding="utf-8")


def _extract_llm_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts = [getattr(part, "text", str(part)) for part in content]
        return " ".join(parts).strip()
    return str(content).strip()


def _strip_json_code_fence(response_text: str) -> str:
    cleaned = response_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    if cleaned.lower().startswith("json"):
        cleaned = cleaned[4:].strip()
    return cleaned


def format_qa_transcript(qa_pairs: List[Dict[str, Any]]) -> str:
    """Format QA pairs for scoring prompt consumption."""
    transcript_lines = ["INTERVIEW TRANSCRIPT", "=" * 50, ""]

    for idx, qa in enumerate(qa_pairs, start=1):
        question = str(qa.get("question", "")).strip()
        question_type = str(qa.get("type", qa.get(
            "question_type", "unknown"))).strip()
        difficulty = str(
            qa.get("difficulty", qa.get("question_difficulty", "unknown"))
        ).strip()

        transcript_lines.append(f"Q{idx}: {question}")
        transcript_lines.append(
            f"Type: {question_type or 'unknown'}, Difficulty: {difficulty or 'unknown'}")

        was_skipped = bool(qa.get("was_skipped", False))
        if was_skipped:
            transcript_lines.append("A: [Question was skipped]")
        else:
            answer = str(qa.get("answer", "")).strip()
            if len(answer) > TRANSCRIPT_ANSWER_MAX_CHARS:
                answer = answer[:TRANSCRIPT_ANSWER_MAX_CHARS] + \
                    "...[truncated]"
            transcript_lines.append(f"A: {answer}")

        transcript_lines.extend(["-" * 50, ""])

    return "\n".join(transcript_lines).strip()


def validate_scores_response(scores_data: Dict[str, Any], expected_questions: int) -> None:
    """Validate LLM scoring response shape and value ranges."""
    required_fields = [
        "per_question_scores",
        "overall_score",
        "technical_depth_score",
        "communication_score",
        "what_went_well",
        "areas_to_improve",
        "role_readiness_percent",
    ]

    for field_name in required_fields:
        if field_name not in scores_data:
            raise ValueError(f"Missing required field: {field_name}")

    per_question_scores = scores_data["per_question_scores"]
    if not isinstance(per_question_scores, list):
        raise ValueError("per_question_scores must be a list")

    if len(per_question_scores) != expected_questions:
        raise ValueError(
            "per_question_scores length must match qa_pairs length "
            f"(expected {expected_questions}, got {len(per_question_scores)})"
        )

    for idx, item in enumerate(per_question_scores):
        if not isinstance(item, dict):
            raise ValueError(f"per_question_scores[{idx}] must be an object")

        for score_name in ["relevance", "depth", "clarity"]:
            if score_name not in item:
                raise ValueError(
                    f"per_question_scores[{idx}] missing {score_name}")
            score_value = float(item[score_name])
            if not 0 <= score_value <= 10:
                raise ValueError(
                    f"per_question_scores[{idx}].{score_name} must be in 0-10, got {score_value}"
                )

    for metric_name in ["overall_score", "technical_depth_score", "communication_score"]:
        value = float(scores_data[metric_name])
        if not 0 <= value <= 10:
            raise ValueError(f"{metric_name} must be in 0-10, got {value}")

    readiness = float(scores_data["role_readiness_percent"])
    if not 0 <= readiness <= 100:
        raise ValueError(
            f"role_readiness_percent must be in 0-100, got {readiness}")

    for list_name in ["what_went_well", "areas_to_improve"]:
        list_value = scores_data[list_name]
        if not isinstance(list_value, list) or not all(
            isinstance(item, str) and item.strip() for item in list_value
        ):
            raise ValueError(
                f"{list_name} must be a list of non-empty strings")

    if "recommended_resources" in scores_data:
        list_value = scores_data["recommended_resources"]
        if not isinstance(list_value, list) or not all(
            isinstance(item, str) and item.strip() for item in list_value
        ):
            raise ValueError(
                "recommended_resources must be a list of non-empty strings")


def _derive_readiness_label(overall_score: float) -> str:
    if overall_score >= 8:
        return "Strong Candidate"
    if overall_score >= 6:
        return "Good Foundation"
    if overall_score >= 4:
        return "Needs Preparation"
    return "Significant Work Needed"


def build_interview_scores(
    scores_data: Dict[str, Any],
    qa_pairs: List[Dict[str, Any]],
) -> InterviewScores:
    """Build typed scoring payload from validated LLM output."""
    per_answer_scores: List[AnswerScore] = []

    for idx, qa in enumerate(qa_pairs):
        was_skipped = bool(qa.get("was_skipped", False))
        question = str(qa.get("question", "")).strip()
        answer = str(qa.get("answer", "")).strip()
        question_type = str(qa.get("type", qa.get(
            "question_type", "unknown"))).strip()

        if was_skipped:
            per_answer_scores.append(
                AnswerScore(
                    question=question,
                    answer="[Skipped]",
                    question_type=question_type or "unknown",
                    relevance_score=0.0,
                    depth_score=0.0,
                    clarity_score=0.0,
                    average_score=0.0,
                    feedback="Question was skipped. No score applied.",
                    was_skipped=True,
                )
            )
            continue

        per_q = scores_data["per_question_scores"][idx]
        relevance = float(per_q["relevance"])
        depth = float(per_q["depth"])
        clarity = float(per_q["clarity"])
        average = (relevance + depth + clarity) / 3.0

        per_answer_scores.append(
            AnswerScore(
                question=question,
                answer=answer,
                question_type=question_type or "unknown",
                relevance_score=relevance,
                depth_score=depth,
                clarity_score=clarity,
                average_score=average,
                feedback=str(per_q.get("feedback", "")).strip(),
                was_skipped=False,
            )
        )

    overall_score = float(scores_data["overall_score"])
    readiness_label = str(scores_data.get("role_readiness_label", "")).strip()
    return InterviewScores(
        overall_score=overall_score,
        technical_depth_score=float(scores_data["technical_depth_score"]),
        communication_score=float(scores_data["communication_score"]),
        what_went_well=[str(item).strip()
                        for item in scores_data["what_went_well"]],
        areas_to_improve=[str(item).strip()
                          for item in scores_data["areas_to_improve"]],
        role_readiness_percent=float(scores_data["role_readiness_percent"]),
        role_readiness_label=readiness_label or _derive_readiness_label(
            overall_score),
        per_answer_scores=per_answer_scores,
        recommended_resources=[
            str(item).strip() for item in scores_data.get("recommended_resources", [])
        ],
    )


def generate_feedback(scores: InterviewScores) -> Dict[str, Any]:
    """Build lightweight display payload for UI integration."""
    return {
        "what_went_well": list(scores.what_went_well),
        "areas_to_improve": list(scores.areas_to_improve),
        "recommended_resources": list(scores.recommended_resources),
        "role_readiness_label": scores.role_readiness_label,
        "role_readiness_percent": round(scores.role_readiness_percent, 0),
    }


def score_interview(
    qa_pairs: List[Dict[str, Any]],
    role: str,
    company: str,
    resume_insights: Dict[str, Any] | None = None,
    jd_insights: Dict[str, Any] | None = None,
    generated_questions: List[Dict[str, Any]] | None = None,
    llm: Any | None = None,
    prompt_path: str = "prompts/scorer_prompt.txt",
) -> InterviewScores:
    """Score interview using GROQ and return validated, typed scores."""
    if not qa_pairs:
        raise ValueError("qa_pairs cannot be empty")

    prompt_template = _load_prompt_template(prompt_path)
    prompt = PromptTemplate(
        input_variables=[
            "role",
            "company",
            "transcript",
            "resume_insights",
            "jd_insights",
            "generated_questions",
        ],
        template=prompt_template,
    )

    transcript = format_qa_transcript(qa_pairs)
    prompt_text = prompt.format(
        role=role,
        company=company,
        transcript=transcript,
        resume_insights=json.dumps(resume_insights or {}, ensure_ascii=True),
        jd_insights=json.dumps(jd_insights or {}, ensure_ascii=True),
        generated_questions=json.dumps(
            generated_questions or [], ensure_ascii=True),
    )

    try:
        if llm is not None:
            logger.info("Calling injected LLM for interview scoring")
            response = llm.invoke(prompt_text)
        else:
            response = None
            last_error: Exception | None = None
            for model_name in get_groq_model_candidates():
                try:
                    model = build_groq_llm(
                        model_name=model_name,
                        temperature=GROQ_SCORING_TEMPERATURE,
                    )
                    logger.info(
                        "Calling LLM for interview scoring using model=%s", model_name
                    )
                    response = model.invoke(prompt_text)
                    break
                except Exception as exc:
                    last_error = exc
                    if is_model_decommission_error(exc):
                        logger.warning(
                            "GROQ model unavailable/decommissioned, trying next model: %s",
                            model_name,
                        )
                        continue
                    raise

            if response is None:
                if last_error is not None:
                    raise last_error
                raise ValueError("No GROQ model candidates available")

        raw_text = _extract_llm_text(response)
        scores_data = json.loads(_strip_json_code_fence(raw_text))
        if not isinstance(scores_data, dict):
            raise ValueError("Scoring response must be a JSON object")

        validate_scores_response(
            scores_data=scores_data, expected_questions=len(qa_pairs))
        scores = build_interview_scores(
            scores_data=scores_data, qa_pairs=qa_pairs)

        logger.info(
            "Interview scored: overall=%.2f, technical=%.2f, communication=%.2f",
            scores.overall_score,
            scores.technical_depth_score,
            scores.communication_score,
        )
        return scores

    except json.JSONDecodeError as exc:
        logger.error("LLM response is not valid JSON: %s", exc)
        raise ValueError("Failed to parse scoring response JSON") from exc
    except Exception:
        logger.exception("Error scoring interview")
        raise
