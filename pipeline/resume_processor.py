"""Resume ingestion and RAG pipeline utilities for Module 1."""

from __future__ import annotations

import logging
import os
import re
import hashlib
import json
from pathlib import Path
from typing import Any, List, Optional, Tuple

import chromadb
import numpy as np
import pandas as pd
import PyPDF2
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.constants import (
    CHROMA_STORE_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    INDIAN_JD_COLLECTION,
    INTERVIEW_QA_COLLECTION,
    MODEL_CACHE_DIR,
)
from utils.embedding_provider import get_embedding_model
from utils.groq_helpers import (
    build_groq_llm,
    get_groq_model_candidates,
    is_model_decommission_error,
)

logger = logging.getLogger(__name__)
INGESTION_STATE_FILE = Path(CHROMA_STORE_DIR) / "dataset_ingestion_state.json"
GROQ_RESUME_INSIGHTS_TEMPERATURE = 0.3


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


def _ensure_list(value: Any) -> List[Any]:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _normalize_resume_insights_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "candidate_name": str(payload.get("candidate_name", "N/A")).strip() or "N/A",
        "headline": str(payload.get("headline", "N/A")).strip() or "N/A",
        "experience_level": str(payload.get("experience_level", "N/A")).strip() or "N/A",
        "education": _ensure_list(payload.get("education", [])) or ["N/A"],
        "skills": [str(item).strip() for item in _ensure_list(payload.get("skills", [])) if str(item).strip()],
        "tools": [str(item).strip() for item in _ensure_list(payload.get("tools", [])) if str(item).strip()],
        "projects": _ensure_list(payload.get("projects", [])) or ["N/A"],
        "experience": _ensure_list(payload.get("experience", [])) or ["N/A"],
        "certifications": [str(item).strip() for item in _ensure_list(payload.get("certifications", [])) if str(item).strip()] or ["N/A"],
        "achievements": [str(item).strip() for item in _ensure_list(payload.get("achievements", [])) if str(item).strip()] or ["N/A"],
        "summary": str(payload.get("summary", "N/A")).strip() or "N/A",
        "missing_info": [str(item).strip() for item in _ensure_list(payload.get("missing_info", [])) if str(item).strip()] or ["N/A"],
    }


def _load_embedding_model() -> Any:
    """Load sentence-transformers model with project-local cache."""
    return get_embedding_model()


def _encode_texts(texts: List[str], show_progress_bar: bool = False) -> np.ndarray:
    """Encode a list of texts into embeddings."""
    if not texts:
        raise ValueError("texts cannot be empty")

    model = _load_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=show_progress_bar)
    array = np.array(embeddings)

    if array.ndim != 2 or array.shape[1] != 384:
        raise ValueError(
            f"Expected 2D embeddings with 384 dimensions, got shape {array.shape}"
        )

    return array


def _sanitize_collection_name(name: str) -> str:
    """Make collection names safe for ChromaDB naming constraints."""
    lowered = name.lower()
    cleaned = re.sub(r"[^a-z0-9_-]", "_", lowered)
    if len(cleaned) < 3:
        cleaned = cleaned.ljust(3, "_")
    return cleaned[:63]


def extract_text_from_pdf(file_path: str) -> str:
    """Extract raw text from a PDF file.

    Args:
        file_path: Path to PDF file.

    Returns:
        Concatenated text from all pages.

    Raises:
        FileNotFoundError: If PDF path does not exist.
        ValueError: If no extractable text is found.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {file_path}")

    try:
        text_parts: List[str] = []
        with path.open("rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            num_pages = len(pdf_reader.pages)
            logger.info("Extracting text from PDF with %s pages", num_pages)

            for page_number, page in enumerate(pdf_reader.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)
                logger.debug("Processed page %s/%s", page_number, num_pages)

        text = "\n".join(text_parts).strip()
        if not text:
            raise ValueError(
                "No text found in PDF. The file may be image-based and require OCR."
            )

        logger.info("Successfully extracted %s characters from PDF", len(text))
        return text

    except Exception as exc:
        logger.error("Error extracting text from PDF %s: %s", file_path, exc)
        raise


def chunk_text(
    raw_text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP
) -> List[str]:
    """Split resume text into overlapping chunks.

    Args:
        raw_text: Full resume text.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        List of text chunks.

    Raises:
        ValueError: If input text is empty.
    """
    if not raw_text or not raw_text.strip():
        raise ValueError("raw_text cannot be empty")

    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = [chunk.strip()
              for chunk in splitter.split_text(raw_text) if chunk.strip()]
    logger.info("Created %s chunks from text", len(chunks))
    return chunks


def generate_embeddings(chunks: List[str]) -> Tuple[np.ndarray, Any]:
    """Generate embeddings for text chunks using MiniLM.

    Args:
        chunks: List of resume chunks.

    Returns:
        Tuple of embeddings array and loaded model.

    Raises:
        ValueError: If chunks are empty or invalid.
    """
    if not chunks:
        raise ValueError("chunks cannot be empty")

    model = _load_embedding_model()
    logger.info("Generating embeddings for %s chunks", len(chunks))
    embeddings = np.array(model.encode(chunks, show_progress_bar=True))

    if embeddings.ndim != 2 or embeddings.shape[1] != 384:
        raise ValueError(
            f"Expected 2D embeddings with 384 dimensions, got shape {embeddings.shape}"
        )

    logger.info(
        "Generated %s embeddings of dimension %s", embeddings.shape[0], embeddings.shape[1]
    )
    return embeddings, model


def initialize_chroma_db() -> chromadb.ClientAPI:
    """Initialize ChromaDB with persistent local storage."""
    os.makedirs(CHROMA_STORE_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=CHROMA_STORE_DIR)
    logger.info("ChromaDB initialized at %s", CHROMA_STORE_DIR)
    return client


def _session_collection_name(session_id: str) -> str:
    """Build a session-scoped collection name."""
    if not session_id or not session_id.strip():
        raise ValueError("session_id cannot be empty")
    return _sanitize_collection_name(f"resume_{session_id}")


def _jd_session_collection_name(session_id: str) -> str:
    if not session_id or not session_id.strip():
        raise ValueError("session_id cannot be empty")
    return _sanitize_collection_name(f"jd_{session_id}")


def store_resume_in_chroma(
    chunks: List[str],
    embeddings: np.ndarray,
    session_id: str,
    client: chromadb.ClientAPI,
) -> str:
    """Store resume chunks and embeddings in a session-specific Chroma collection.

    Args:
        chunks: Text chunks.
        embeddings: Embeddings aligned with chunks.
        session_id: Session identifier.
        client: Initialized Chroma client.

    Returns:
        Collection name used for storage.

    Raises:
        ValueError: If chunk and embedding counts do not match.
    """
    if not chunks:
        raise ValueError("chunks cannot be empty")

    if embeddings.shape[0] != len(chunks):
        raise ValueError("embeddings row count must match number of chunks")

    collection_name = _session_collection_name(session_id)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0:
        existing = collection.get(include=[])
        existing_ids = existing.get("ids", [])
        if existing_ids:
            collection.delete(ids=existing_ids)

    ids = [f"chunk_{idx}" for idx in range(len(chunks))]
    metadatas = [{"chunk_index": idx, "source": "resume"}
                 for idx in range(len(chunks))]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=chunks,
        metadatas=metadatas,
    )

    logger.info("Stored %s resume chunks in collection %s",
                len(chunks), collection_name)
    return collection_name


def retrieve_resume_context(
    query: str,
    session_id: str,
    k: int = 5,
    client: Optional[chromadb.ClientAPI] = None,
) -> List[str]:
    """Retrieve top-k most relevant chunks for a query from a session collection.

    Args:
        query: Query text.
        session_id: Session identifier.
        k: Number of relevant chunks to fetch.
        client: Optional Chroma client.

    Returns:
        List of retrieved chunks, best-first.
    """
    if not query or not query.strip():
        return []

    if k <= 0:
        return []

    local_client = client or initialize_chroma_db()
    collection_name = _session_collection_name(session_id)

    try:
        collection = local_client.get_collection(name=collection_name)
    except Exception:
        logger.warning("Collection %s not found for session %s",
                       collection_name, session_id)
        return []

    count = collection.count()
    if count == 0:
        return []

    query_embedding = _encode_texts([query], show_progress_bar=False)
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(k, count),
    )

    documents = results.get("documents", []) if results else []
    if not documents:
        return []

    retrieved_chunks = [chunk for chunk in documents[0]
                        if isinstance(chunk, str)]
    logger.info("Retrieved %s chunks for query", len(retrieved_chunks))
    return retrieved_chunks


def store_jd_in_chroma(
    jd_text: str,
    session_id: str,
    client: chromadb.ClientAPI,
) -> str:
    """Store user JD text in a session-specific Chroma collection."""
    if not jd_text or not jd_text.strip():
        raise ValueError("jd_text cannot be empty")

    collection_name = _jd_session_collection_name(session_id)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    if collection.count() > 0:
        existing = collection.get(include=[])
        existing_ids = existing.get("ids", [])
        if existing_ids:
            collection.delete(ids=existing_ids)

    embedding = _encode_texts([jd_text], show_progress_bar=False)
    collection.add(
        ids=["jd_0"],
        embeddings=embedding.tolist(),
        documents=[jd_text],
        metadatas=[{"source": "jd", "session_id": session_id}],
    )
    return collection_name


def extract_resume_insights_with_llm(
    resume_text: str,
    llm: Any | None = None,
    prompt_path: str = "prompts/resume_insights_prompt.txt",
) -> Dict[str, Any]:
    """Extract rich structured resume insights using GROQ."""
    if not resume_text or not resume_text.strip():
        raise ValueError("resume_text cannot be empty")

    prompt_template = _load_prompt_template(prompt_path)
    prompt = PromptTemplate(
        input_variables=["resume_text"],
        template=prompt_template,
    )
    prompt_text = prompt.format(resume_text=resume_text)

    try:
        if llm is not None:
            response = llm.invoke(prompt_text)
        else:
            response = None
            last_error: Exception | None = None
            for model_name in get_groq_model_candidates():
                try:
                    model = build_groq_llm(
                        model_name=model_name,
                        temperature=GROQ_RESUME_INSIGHTS_TEMPERATURE,
                    )
                    response = model.invoke(prompt_text)
                    break
                except Exception as exc:
                    last_error = exc
                    if is_model_decommission_error(exc):
                        continue
                    raise

            if response is None:
                if last_error is not None:
                    raise last_error
                raise ValueError("No GROQ model candidates available")

        payload = json.loads(_strip_json_code_fence(
            _extract_llm_text(response)))
        if not isinstance(payload, dict):
            raise ValueError("Resume insights response must be a JSON object")
        return _normalize_resume_insights_payload(payload)

    except json.JSONDecodeError as exc:
        raise ValueError("Failed to parse resume insights JSON") from exc


def preprocess_kaggle_dataset(csv_path: str) -> List[Document]:
    """Load and convert Kaggle HR Q&A rows into LangChain Document objects.

    Args:
        csv_path: Path to interview_qa CSV file.

    Returns:
        List of documents containing Q&A text plus metadata.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Kaggle dataset not found: {csv_path}")

    df = pd.read_csv(path)
    required_cols = {"question", "answer"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV must contain question and answer columns")

    df = df.dropna(subset=["question", "answer"])

    docs: List[Document] = []
    for index, row in df.iterrows():
        text = f"Question: {row['question']}\nAnswer: {row['answer']}"
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": "kaggle_interview_qa",
                    "row": int(index),
                    "question_category": str(row.get("category", "general")),
                },
            )
        )

    logger.info("Prepared %s Kaggle interview documents", len(docs))
    return docs


def preprocess_indian_jd_dataset(json_path: str) -> List[Document]:
    """Load synthetic Indian JD dataset JSON into LangChain Document objects."""
    import json

    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Indian JD dataset not found: {json_path}")

    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Indian JD dataset must be a list of objects")

    docs: List[Document] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            continue
        company = str(item.get("company", "unknown"))
        role = str(item.get("role", "unknown"))
        skills = item.get("skills", [])
        skills_text = ", ".join(map(str, skills)) if isinstance(
            skills, list) else str(skills)
        experience_level = str(item.get("experience_level", "unknown"))
        hiring_notes = str(item.get("hiring_notes", ""))

        text = (
            f"Company: {company}\n"
            f"Role: {role}\n"
            f"Skills: {skills_text}\n"
            f"Experience Level: {experience_level}\n"
            f"Hiring Notes: {hiring_notes}"
        )

        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": "indian_jd_dataset",
                    "row": index,
                    "company": company,
                    "role": role,
                },
            )
        )

    logger.info("Prepared %s Indian JD documents", len(docs))
    return docs


def initialize_datasets_in_chromadb(client: chromadb.ClientAPI) -> None:
    """Initialize global reusable Chroma collections for Kaggle Q&A and Indian JD data.

    Existing non-empty collections are left unchanged to avoid duplicate ingestion.
    """
    kaggle_path = "data/interview_qa.csv"
    kaggle_docs = preprocess_kaggle_dataset(kaggle_path)
    _store_documents_collection(
        client,
        INTERVIEW_QA_COLLECTION,
        kaggle_docs,
        source_key=INTERVIEW_QA_COLLECTION,
        source_signature=_compute_file_signature(kaggle_path),
    )

    jd_path = "data/indian_jd_dataset.json"
    jd_docs = preprocess_indian_jd_dataset(jd_path)
    _store_documents_collection(
        client,
        INDIAN_JD_COLLECTION,
        jd_docs,
        source_key=INDIAN_JD_COLLECTION,
        source_signature=_compute_file_signature(jd_path),
    )


def _compute_file_signature(file_path: str) -> str:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    digest.update(str(path.stat().st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def _load_ingestion_state() -> dict:
    if not INGESTION_STATE_FILE.exists():
        return {}
    try:
        return json.loads(INGESTION_STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_ingestion_state(state: dict) -> None:
    INGESTION_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    INGESTION_STATE_FILE.write_text(
        json.dumps(state, indent=2), encoding="utf-8")


def _store_documents_collection(
    client: chromadb.ClientAPI,
    collection_name: str,
    docs: List[Document],
    source_key: str,
    source_signature: str,
) -> None:
    """Persist a list of Documents into a collection with generated embeddings."""
    name = _sanitize_collection_name(collection_name)
    collection = client.get_or_create_collection(
        name=name, metadata={"hnsw:space": "cosine"})

    state = _load_ingestion_state()
    previous_signature = state.get(source_key)
    has_data = collection.count() > 0

    if has_data and previous_signature == source_signature:
        logger.info(
            "Collection %s already up to date; skipping ingestion", name)
        return

    if has_data and previous_signature != source_signature:
        existing = collection.get(include=[])
        existing_ids = existing.get("ids", [])
        if existing_ids:
            collection.delete(ids=existing_ids)
        logger.info("Dataset changed for %s. Re-indexing collection.", name)

    texts = [doc.page_content for doc in docs]
    embeddings = _encode_texts(texts, show_progress_bar=False)

    ids = [f"{name}_{idx}" for idx in range(len(docs))]
    metadatas = [doc.metadata for doc in docs]

    collection.add(
        ids=ids,
        embeddings=embeddings.tolist(),
        documents=texts,
        metadatas=metadatas,
    )

    state[source_key] = source_signature
    _save_ingestion_state(state)
    logger.info("Initialized collection %s with %s documents", name, len(docs))
