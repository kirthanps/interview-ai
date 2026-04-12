"""JD parsing and company-context retrieval utilities for Module 2."""

from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from utils.constants import (
    EMBEDDING_MODEL,
    INDIAN_JD_COLLECTION,
    INTERVIEW_QA_COLLECTION,
    MODEL_CACHE_DIR,
)

logger = logging.getLogger(__name__)

GROQ_FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "llama3-70b-8192",
]
GROQ_PARSE_TEMPERATURE = 0.5


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

            value = v.strip().strip('"').strip("'")
            return value
    except Exception:
        logger.exception("Failed reading %s for key %s", env_file, key)

    return ""


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


@lru_cache(maxsize=1)
def _load_sentence_transformer() -> Any:
    """Load shared sentence-transformer model with project-local cache."""
    from sentence_transformers import SentenceTransformer

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    return SentenceTransformer(EMBEDDING_MODEL, cache_folder=MODEL_CACHE_DIR)


def _embed_query_text(query_text: str) -> List[float]:
    """Embed one query string for Chroma query_embeddings calls."""
    model = _load_sentence_transformer()
    vector = model.encode([query_text], show_progress_bar=False)[0]
    return vector.tolist() if hasattr(vector, "tolist") else list(vector)


def _build_groq_llm(temperature: float) -> ChatGroq:
    """Create a GROQ chat model from environment configuration."""
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

    # Keep order while removing duplicates.
    unique: List[str] = []
    seen = set()
    for item in candidates:
        if item in seen:
            continue
        unique.append(item)
        seen.add(item)

    return unique


def _is_model_decommission_error(exc: Exception) -> bool:
    """Detect known Groq model decommission errors for retry fallback."""
    text = str(exc).lower()
    return "model_decommissioned" in text or "decommissioned" in text


def _build_groq_llm_for_model(model_name: str, temperature: float) -> ChatGroq:
    """Create a GROQ chat model for a specific model name."""
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


def parse_jd_with_llm(jd_text: str, llm: Any | None = None) -> Dict[str, Any]:
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
        input_variables=["jd_text"], template=prompt_template)

    try:
        prompt_text = prompt.format(jd_text=jd_text)

        if llm is not None:
            logger.info("Calling injected LLM for parse_jd_with_llm")
            response = llm.invoke(prompt_text)
        else:
            response = None
            last_error: Exception | None = None
            for model_name in _get_groq_model_candidates():
                try:
                    model = _build_groq_llm_for_model(
                        model_name=model_name,
                        temperature=GROQ_PARSE_TEMPERATURE,
                    )
                    logger.info(
                        "Calling LLM for parse_jd_with_llm using model=%s", model_name)
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
        parsed = json.loads(_strip_json_code_fence(raw_text))

        if not isinstance(parsed, dict):
            raise ValueError("Parsed JD payload must be a JSON object")

        normalized: Dict[str, Any] = {
            "role": str(parsed.get("role", "Unknown Role")).strip() or "Unknown Role",
            "company": str(parsed.get("company", "Unknown Company")).strip()
            or "Unknown Company",
            "skills": _normalize_string_list(parsed.get("skills", [])),
            "experience_level": str(parsed.get("experience_level", "Unknown")).strip()
            or "Unknown",
            "tools": _normalize_string_list(parsed.get("tools", [])),
        }

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
        query_embedding = _embed_query_text(query_text)
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
) -> List[Dict[str, str]]:
    """Retrieve relevant Q&A examples from the shared interview dataset."""
    if k <= 0:
        return []

    query = " ".join(_normalize_string_list(skills)).strip()
    if not query:
        query = "general interview questions"

    try:
        collection = chroma_client.get_collection(name=INTERVIEW_QA_COLLECTION)
        query_embedding = _embed_query_text(query)
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
