"""Question generation utilities for Module 2."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from pipeline.jd_parser import (
    get_question_distribution,
    retrieve_few_shot_examples,
)

logger = logging.getLogger(__name__)

GROQ_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
]
GROQ_QUESTION_TEMPERATURE = 0.7
ALLOWED_TYPES = {"technical", "hr", "system_design", "situational"}
ALLOWED_DIFFICULTY = {"easy", "medium", "hard"}


def _read_env_value_from_file(key: str, env_file: str = ".env") -> str:
    """Read a single environment value from a .env-style file."""
    path = Path(env_file)
    if not path.exists():
        return ""

    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            k, v = line.split("=", 1)
            if k.strip() != key:
                continue

            return v.strip().strip('"').strip("'")
    except Exception:
        logger.exception("Failed reading %s for key %s", env_file, key)

    return ""


@dataclass
class Question:
    """Represents one generated interview question."""

    question: str
    type: str
    hint: List[str]
    difficulty: str
    is_answered: bool = False
    answer_text: str = ""
    skip_flag: bool = False


def _load_prompt_template(prompt_path: str) -> str:
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return path.read_text(encoding="utf-8")


def _build_groq_llm(temperature: float) -> ChatGroq:
    return _build_groq_llm_for_model(
        model_name=_get_groq_model_candidates()[0],
        temperature=temperature,
    )


def _get_groq_model_candidates() -> List[str]:
    """Resolve Groq model candidates from env/.env with safe defaults."""
    explicit = os.getenv(
        "GROQ_MODEL") or _read_env_value_from_file("GROQ_MODEL")
    candidates = []

    if explicit and explicit.strip():
        candidates.append(explicit.strip())

    extra = os.getenv("GROQ_MODEL_CANDIDATES") or _read_env_value_from_file(
        "GROQ_MODEL_CANDIDATES")
    if extra:
        for item in extra.split(","):
            model_name = item.strip()
            if model_name:
                candidates.append(model_name)

    candidates.extend(GROQ_FALLBACK_MODELS)

    unique: List[str] = []
    seen = set()
    for item in candidates:
        if item in seen:
            continue
        unique.append(item)
        seen.add(item)
    return unique


def _is_model_decommission_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "model_decommissioned" in text or "decommissioned" in text


def _build_groq_llm_for_model(model_name: str, temperature: float) -> ChatGroq:
    api_key = os.getenv(
        "GROQ_API_KEY") or _read_env_value_from_file("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to your .env/environment.")

    return ChatGroq(
        model=model_name,
        temperature=temperature,
        groq_api_key=api_key,
    )


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


def generate_questions(
    parsed_jd: Dict[str, Any],
    company_profile: Dict[str, Any],
    resume_context: List[str],
    preferences: Dict[str, Any],
    chroma_client: chromadb.ClientAPI,
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
        ],
        template=prompt_template,
    )

    few_shot_examples = retrieve_few_shot_examples(
        skills=list(parsed_jd.get("skills", [])),
        chroma_client=chroma_client,
        k=3,
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
            company=company_profile.get(
                "company", parsed_jd.get("company", "Unknown Company")),
            skills=", ".join(parsed_jd.get("skills", [])),
            experience_level=parsed_jd.get("experience_level", "Unknown"),
            resume_context=resume_context_str,
            company_notes=company_profile.get("hiring_notes", ""),
            few_shot_examples=few_shot_str,
            num_questions=validated_preferences["num_questions"],
            focus_area=validated_preferences["focus_area"],
            difficulty=validated_preferences["difficulty"],
            distribution=distribution,
        )

        if llm is not None:
            logger.info("Calling injected LLM for generate_questions")
            response = llm.invoke(prompt_text)
        else:
            response = None
            last_error: Exception | None = None
            for model_name in _get_groq_model_candidates():
                try:
                    model = _build_groq_llm_for_model(
                        model_name=model_name,
                        temperature=GROQ_QUESTION_TEMPERATURE,
                    )
                    logger.info(
                        "Calling LLM for generate_questions using model=%s", model_name)
                    response = model.invoke(prompt_text)
                    break
                except Exception as exc:
                    last_error = exc
                    if _is_model_decommission_error(exc):
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
        for item in payload:
            if not isinstance(item, dict):
                continue

            question_text = str(item.get("question", "")).strip()
            if not question_text:
                continue

            q_type = str(item.get("type", "technical")).strip().lower()
            if q_type not in ALLOWED_TYPES:
                q_type = "technical"

            difficulty = str(
                item.get("difficulty", validated_preferences["difficulty"]))
            difficulty = difficulty.strip().lower()
            if difficulty not in ALLOWED_DIFFICULTY:
                difficulty = validated_preferences["difficulty"]

            hints = _normalize_hint_list(item.get("hint", []))
            if not hints:
                hints = [
                    "Use the candidate's stated experience to structure your answer."]

            questions.append(
                Question(
                    question=question_text,
                    type=q_type,
                    hint=hints,
                    difficulty=difficulty,
                )
            )

        if not questions:
            raise ValueError("No valid questions returned by LLM")

        expected = validated_preferences["num_questions"]
        if len(questions) != expected:
            logger.warning("Expected %s questions, received %s",
                           expected, len(questions))

        return questions

    except json.JSONDecodeError as exc:
        logger.error(
            "Invalid JSON response while generating questions: %s", exc)
        raise ValueError("Failed to parse generated questions JSON") from exc
    except Exception:
        logger.exception("Error while generating questions")
        raise
