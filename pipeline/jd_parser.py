"""JD parsing and company-context retrieval utilities for Module 2."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from langchain_core.prompts import PromptTemplate

from utils.constants import (
    INDIAN_JD_COLLECTION,
    INTERVIEW_QA_COLLECTION,
)
from utils.embedding_provider import embed_query_text
from utils.groq_helpers import (
    build_groq_llm,
    get_groq_model_candidates,
    is_model_decommission_error,
)

logger = logging.getLogger(__name__)

GROQ_PARSE_TEMPERATURE = 0.5


def _load_prompt_template(prompt_path: str) -> str:
    """Load a prompt template file from disk."""
    path = Path(prompt_path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return path.read_text(encoding="utf-8")


def _strip_json_code_fence(response_text: str) -> str:
    """Strip markdown code fences around JSON payloads when present."""
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


def _extract_llm_text(response: Any) -> str:
    """Extract text content from an LLM response object or a raw string."""
    if isinstance(response, str):
        return response
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts = [getattr(part, "text", str(part)) for part in content]
        return " ".join(parts).strip()
    return str(content).strip()


def _normalize_string_list(value: Any) -> List[str]:
    """Normalize model output into a clean list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        parts = [item.strip() for item in value.split(",")]
        return [item for item in parts if item]
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _retrieve_jd_examples(
    jd_text: str,
    chroma_client: chromadb.ClientAPI,
    k: int,
) -> List[str]:
    if k <= 0:
        return []

    try:
        collection = chroma_client.get_collection(name=INDIAN_JD_COLLECTION)
        query_embedding = embed_query_text(jd_text)
        results = collection.query(
            query_embeddings=[query_embedding], n_results=k)
        documents = (results or {}).get("documents") or []
        if not documents or not documents[0]:
            return []
        return [str(item) for item in documents[0] if str(item).strip()]
    except Exception:
        logger.exception("Failed retrieving JD few-shot examples")
        return []


def _normalize_jd_payload(parsed: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {
        "role": str(parsed.get("role", "Unknown Role")).strip() or "Unknown Role",
        "company": str(parsed.get("company", "Unknown Company")).strip() or "Unknown Company",
        "employment_type": str(parsed.get("employment_type", "N/A")).strip() or "N/A",
        "experience_level": str(parsed.get("experience_level", "Unknown")).strip() or "Unknown",
        "location": str(parsed.get("location", "N/A")).strip() or "N/A",
        "team": str(parsed.get("team", "N/A")).strip() or "N/A",
        "skills": _normalize_string_list(parsed.get("skills", [])),
        "tools": _normalize_string_list(parsed.get("tools", [])),
        "responsibilities": _normalize_string_list(parsed.get("responsibilities", [])),
        "qualifications": _normalize_string_list(parsed.get("qualifications", [])),
        "interview_focus": _normalize_string_list(parsed.get("interview_focus", [])),
        "company_signals": _normalize_string_list(parsed.get("company_signals", [])),
        "missing_info": _normalize_string_list(parsed.get("missing_info", ["N/A"])) or ["N/A"],
    }
    return normalized


def parse_jd_with_llm(
    jd_text: str,
    llm: Any | None = None,
    chroma_client: chromadb.ClientAPI | None = None,
    k_examples: int = 3,
) -> Dict[str, Any]:
    """Parse job-description text into structured fields using GROQ.

    Args:
        jd_text: Raw JD content entered by the user.
        llm: Optional injected LLM (used by tests/mocks).

    Returns:
        Parsed dictionary with role/company/skills/experience_level/tools.

    Raises:
        ValueError: If input is too short or response cannot be parsed.
    """
    if not jd_text or len(jd_text.strip()) < 100:
        raise ValueError("JD too short (minimum 100 characters)")

    prompt_template = _load_prompt_template("prompts/jd_parser_prompt.txt")
    prompt = PromptTemplate(
        input_variables=["jd_text", "jd_examples"], template=prompt_template)

    try:
        jd_examples = []
        if chroma_client is not None:
            jd_examples = _retrieve_jd_examples(
                jd_text=jd_text,
                chroma_client=chroma_client,
                k=k_examples,
            )
        examples_block = "\n---\n".join(jd_examples) if jd_examples else "N/A"

        prompt_text = prompt.format(
            jd_text=jd_text, jd_examples=examples_block)

        if llm is not None:
            logger.info("Calling injected LLM for parse_jd_with_llm")
            response = llm.invoke(prompt_text)
        else:
            response = None
            last_error: Exception | None = None
            for model_name in get_groq_model_candidates():
                try:
                    model = build_groq_llm(
                        model_name=model_name,
                        temperature=GROQ_PARSE_TEMPERATURE,
                    )
                    logger.info(
                        "Calling LLM for parse_jd_with_llm using model=%s", model_name)
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
        parsed = json.loads(_strip_json_code_fence(raw_text))

        if not isinstance(parsed, dict):
            raise ValueError("Parsed JD payload must be a JSON object")

        normalized = _normalize_jd_payload(parsed)

        logger.info(
            "Parsed JD result role=%s company=%s",
            normalized["role"],
            normalized["company"],
        )
        return normalized

    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON response while parsing JD: %s", exc)
        raise ValueError("Failed to parse JD response JSON") from exc
    except Exception:
        logger.exception("Error while parsing JD")
        raise


def match_to_company_profile(
    parsed_jd: Dict[str, Any],
    chroma_client: chromadb.ClientAPI,
) -> Dict[str, Any]:
    """Find the closest company profile from the Indian JD dataset."""
    role = str(parsed_jd.get("role", "Unknown Role")).strip()
    company = str(parsed_jd.get("company", "Unknown Company")).strip()
    skills = _normalize_string_list(parsed_jd.get("skills", []))

    fallback_profile = {
        "company": company or "Unknown Company",
        "role": role or "Unknown Role",
        "hiring_notes": "Standard interview process with technical and behavioral rounds.",
    }

    try:
        collection = chroma_client.get_collection(name=INDIAN_JD_COLLECTION)
        query_text = f"{role} at {company} requires {', '.join(skills)}"
        query_embedding = embed_query_text(query_text)
        results = collection.query(
            query_embeddings=[query_embedding], n_results=1)

        documents = (results or {}).get("documents") or []
        if not documents or not documents[0]:
            logger.warning(
                "No company profile match found; using fallback profile")
            return fallback_profile

        matched_doc = str(documents[0][0])
        metadata = ((results or {}).get("metadatas") or [[{}]])[0][0] or {}

        try:
            parsed_doc = json.loads(matched_doc)
            if isinstance(parsed_doc, dict):
                result = {**fallback_profile, **parsed_doc}
                return result
        except json.JSONDecodeError:
            pass

        # Handle Module 1 text format:
        # Company: X\nRole: Y\nSkills: ...\nExperience Level: ...\nHiring Notes: ...
        extracted: Dict[str, Any] = dict(fallback_profile)
        for line in matched_doc.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            label = key.strip().lower().replace(" ", "_")
            extracted[label] = value.strip()

        if isinstance(metadata, dict):
            extracted["company"] = str(metadata.get("company", extracted.get(
                "company", ""))).strip() or extracted.get("company")
            extracted["role"] = str(metadata.get("role", extracted.get(
                "role", ""))).strip() or extracted.get("role")

        if "hiring_notes" not in extracted:
            extracted["hiring_notes"] = fallback_profile["hiring_notes"]

        return extracted

    except Exception:
        logger.exception("Error while matching company profile")
        return fallback_profile


def retrieve_few_shot_examples(
    skills: List[str],
    chroma_client: chromadb.ClientAPI,
    k: int = 3,
    query_text: str | None = None,
) -> List[Dict[str, str]]:
    """Retrieve relevant Q&A examples from the shared interview dataset."""
    if k <= 0:
        return []

    query = (query_text or "").strip()
    if not query:
        query = " ".join(_normalize_string_list(skills)).strip()
    if not query:
        query = "general interview questions"

    try:
        collection = chroma_client.get_collection(name=INTERVIEW_QA_COLLECTION)
        query_embedding = embed_query_text(query)
        results = collection.query(
            query_embeddings=[query_embedding], n_results=k)

        documents = (results or {}).get("documents") or []
        if not documents:
            return []

        examples: List[Dict[str, str]] = []
        for doc in documents[0]:
            text = str(doc)
            question = ""
            answer = ""

            for line in text.splitlines():
                lowered = line.lower().strip()
                if lowered.startswith("question:"):
                    question = line.split(":", 1)[1].strip()
                elif lowered.startswith("q:"):
                    question = line.split(":", 1)[1].strip()
                elif lowered.startswith("answer:"):
                    answer = line.split(":", 1)[1].strip()
                elif lowered.startswith("a:"):
                    answer = line.split(":", 1)[1].strip()

            if question or answer:
                examples.append({"question": question, "answer": answer})

        return examples

    except Exception:
        logger.exception("Error while retrieving few-shot examples")
        return []


def get_question_distribution(num_questions: int, focus_area: str, difficulty: str) -> str:
    """Build a readable question distribution hint for prompt conditioning."""
    lookup = {
        ("technical", "easy"): "5 HR + 2 Basic Technical + 1 Situational",
        ("technical", "medium"): "2 HR + 3 Technical + 2 System Design + 1 Situational",
        ("technical", "hard"): "1 HR + 2 Technical (DSA) + 3 System Design + 2 Complex Problem-Solving",
        ("hr", "easy"): "6 HR + 2 Situational",
        ("hr", "medium"): "5 HR + 2 Situational + 1 Basic Technical",
        ("hr", "hard"): "4 HR (behavioral depth) + 3 Situational (STAR)",
        ("mixed", "easy"): "3 HR + 3 Technical + 2 Situational",
        ("mixed", "medium"): "2 HR + 3 Technical + 2 System Design + 1 Situational",
        ("mixed", "hard"): "1 HR + 2 Technical (DSA) + 3 System Design + 2 Complex scenarios",
        ("system_design", "medium"): "1 HR + 2 Intro to System Design + 3 Full System Design + 2 Trade-offs",
        ("system_design", "hard"): "1 HR + 1 Intro + 4 Full System Design + 2 Advanced",
    }

    base = lookup.get((focus_area, difficulty),
                      "2-3 HR + 3-4 Technical + 2-3 Problem-solving")
    return f"Target count: {num_questions}. Suggested mix: {base}."
