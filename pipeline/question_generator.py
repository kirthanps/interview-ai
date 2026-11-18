"""Question generation utilities for Module 2."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from langchain_core.prompts import PromptTemplate

from pipeline.jd_parser import (
    get_question_distribution,
    retrieve_few_shot_examples,
)
from utils.groq_helpers import (
    build_groq_llm,
    get_groq_model_candidates,
    is_model_decommission_error,
)

logger = logging.getLogger(__name__)

GROQ_QUESTION_TEMPERATURE = 0.7
ALLOWED_TYPES = {"technical", "hr", "system_design", "situational"}
ALLOWED_DIFFICULTY = {"easy", "medium", "hard"}


@dataclass
class Question:
    """Represents one generated interview question."""

    question: str
    type: str
    hint: List[str]
    difficulty: str
    focus_area: str = "mixed"
    rationale: str = ""
    is_answered: bool = False
    answer_text: str = ""
    skip_flag: bool = False


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


def _normalize_hint_list(raw_hint: Any) -> List[str]:
    if isinstance(raw_hint, list):
        return [str(item).strip() for item in raw_hint if str(item).strip()]
    if isinstance(raw_hint, str):
        parts = [item.strip() for item in raw_hint.split(";")]
        return [item for item in parts if item]
    return []


def _validate_preferences(preferences: Dict[str, Any]) -> Dict[str, Any]:
    num_questions = int(preferences.get("num_questions", 8))
    if num_questions <= 0:
        raise ValueError("preferences['num_questions'] must be > 0")

    focus_area = str(preferences.get("focus_area", "mixed")).strip().lower()
    difficulty = str(preferences.get("difficulty", "medium")).strip().lower()

    if difficulty not in ALLOWED_DIFFICULTY:
        difficulty = "medium"

    return {
        "num_questions": num_questions,
        "focus_area": focus_area,
        "difficulty": difficulty,
    }


def _normalize_question_type_for_focus(q_type: str, focus_area: str) -> str:
    if focus_area == "technical":
        return "technical"
    if focus_area == "hr":
        return "hr"
    if focus_area == "system_design":
        return "system_design"
    return q_type


def generate_questions(
    parsed_jd: Dict[str, Any],
    company_profile: Dict[str, Any],
    resume_context: List[str],
    preferences: Dict[str, Any],
    chroma_client: chromadb.ClientAPI,
    resume_text: str = "",
    llm: Any | None = None,
) -> List[Question]:
    """Generate tailored interview questions using GROQ and RAG context."""
    validated_preferences = _validate_preferences(preferences)

    prompt_template = _load_prompt_template(
        "prompts/question_generator_prompt.txt")
    prompt = PromptTemplate(
        input_variables=[
            "role",
            "company",
            "skills",
            "experience_level",
            "resume_context",
            "company_notes",
            "few_shot_examples",
            "num_questions",
            "focus_area",
            "difficulty",
            "distribution",
            "jd_json",
            "resume_text",
        ],
        template=prompt_template,
    )

    retrieval_query = " ".join(
        [
            parsed_jd.get("role", ""),
            parsed_jd.get("company", ""),
            ", ".join(parsed_jd.get("skills", [])),
            str(parsed_jd.get("experience_level", "")),
            resume_text[:300],
        ]
    ).strip()

    few_shot_examples = retrieve_few_shot_examples(
        skills=list(parsed_jd.get("skills", [])),
        chroma_client=chroma_client,
        k=3,
        query_text=retrieval_query,
    )

    resume_context_str = "\n---\n".join(
        resume_context) if resume_context else "No resume context provided."
    few_shot_str = "\n---\n".join(
        [f"Q: {ex.get('question', '')}\nA: {ex.get('answer', '')}" for ex in few_shot_examples]
    )

    distribution = get_question_distribution(
        num_questions=validated_preferences["num_questions"],
        focus_area=validated_preferences["focus_area"],
        difficulty=validated_preferences["difficulty"],
    )

    try:
        prompt_text = prompt.format(
            role=parsed_jd.get("role", "Unknown Role"),
            company=parsed_jd.get("company", "Unknown Company"),
            skills=", ".join(parsed_jd.get("skills", [])),
            experience_level=parsed_jd.get("experience_level", "Unknown"),
            resume_context=resume_context_str,
            company_notes="\n".join(filter(None, [
                company_profile.get("hiring_notes", ""),
                company_profile.get("provided_context", ""),
            ])),
            few_shot_examples=few_shot_str,
            num_questions=validated_preferences["num_questions"],
            focus_area=validated_preferences["focus_area"],
            difficulty=validated_preferences["difficulty"],
            distribution=distribution,
            jd_json=json.dumps(parsed_jd, ensure_ascii=True),
            resume_text=resume_text or "N/A",
        )

        if llm is not None:
            logger.info("Calling injected LLM for generate_questions")
            response = llm.invoke(prompt_text)
        else:
            response = None
            last_error: Exception | None = None
            for model_name in get_groq_model_candidates():
                try:
                    model = build_groq_llm(
                        model_name=model_name,
                        temperature=GROQ_QUESTION_TEMPERATURE,
                    )
                    logger.info(
                        "Calling LLM for generate_questions using model=%s", model_name)
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
        payload = json.loads(_strip_json_code_fence(raw_text))

        if isinstance(payload, dict) and "questions" in payload:
            payload = payload["questions"]

        if not isinstance(payload, list):
            raise ValueError(
                "Question generation response must be a JSON list")

        questions: List[Question] = []
        seen_questions = set()
        for item in payload:
            if not isinstance(item, dict):
                continue

            raw_question = item.get("question")
            if not isinstance(raw_question, str):
                continue
            question_text = " ".join(raw_question.split())
            if not question_text or question_text.casefold() in seen_questions:
                continue
            seen_questions.add(question_text.casefold())

            q_type = str(item.get("type", "technical")).strip().lower()
            if q_type not in ALLOWED_TYPES:
                q_type = "technical"

            q_type = _normalize_question_type_for_focus(
                q_type=q_type,
                focus_area=validated_preferences["focus_area"],
            )

            difficulty = str(
                item.get("difficulty", validated_preferences["difficulty"]))
            difficulty = difficulty.strip().lower()
            if difficulty not in ALLOWED_DIFFICULTY:
                difficulty = validated_preferences["difficulty"]

            # Honor user-selected target difficulty so preview and interview match setup.
            difficulty = validated_preferences["difficulty"]

            hints = _normalize_hint_list(item.get("hint", []))
            if not hints:
                hints = _normalize_hint_list(item.get("hints", []))
            if not hints:
                hints = [
                    "Use the candidate's stated experience to structure your answer."]

            focus_area = str(
                item.get("focus_area", validated_preferences["focus_area"])).strip().lower()
            rationale = str(item.get("rationale", "")).strip()

            questions.append(
                Question(
                    question=question_text,
                    type=q_type,
                    hint=hints,
                    difficulty=difficulty,
                    focus_area=focus_area,
                    rationale=rationale,
                )
            )

        if not questions:
            raise ValueError("No valid questions returned by LLM")

        expected = validated_preferences["num_questions"]
        if len(questions) < expected:
            raise ValueError(f"Expected {expected} distinct questions, received {len(questions)}")

        return questions[:expected]

    except json.JSONDecodeError as exc:
        logger.error(
            "Invalid JSON response while generating questions: %s", exc)
        raise ValueError("Failed to parse generated questions JSON") from exc
    except Exception:
        logger.exception("Error while generating questions")
        raise
