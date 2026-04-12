from __future__ import annotations
from pipeline import resume_processor as rp

import shutil
import sys
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
from fpdf import FPDF

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


@pytest.fixture
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "chroma_store").mkdir(exist_ok=True)
    return tmp_path


def _make_test_pdf(path: Path, text: str) -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    pdf.multi_cell(0, 10, text=text)
    pdf.output(str(path))


class DummyModel:
    def encode(self, texts, show_progress_bar=False):
        vectors = []
        for idx, text in enumerate(texts):
            base = float((sum(ord(c) for c in text) % 97) + idx)
            vector = [base + i for i in range(384)]
            vectors.append(vector)
        return np.array(vectors, dtype=float)


def test_extract_text_from_pdf(isolated_cwd: Path) -> None:
    pdf_path = isolated_cwd / "sample_resume.pdf"
    expected = "Python Django AWS Experience"
    _make_test_pdf(pdf_path, expected)

    extracted = rp.extract_text_from_pdf(str(pdf_path))
    assert "Python" in extracted
    assert "Django" in extracted


def test_chunk_text() -> None:
    raw = "A " * 400 + "B " * 400
    chunks = rp.chunk_text(raw, chunk_size=200, chunk_overlap=20)
    assert len(chunks) >= 2
    assert all(isinstance(chunk, str) and chunk for chunk in chunks)


def test_store_and_retrieve_resume_context(
    isolated_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rp, "_load_embedding_model", lambda: DummyModel())

    text = (
        "Backend engineer with Python Django AWS and PostgreSQL. "
        "Built scalable APIs and async services. "
        "Strong in SQL and system design."
    )
    chunks = rp.chunk_text(text, chunk_size=80, chunk_overlap=10)
    embeddings, _ = rp.generate_embeddings(chunks)

    client = rp.initialize_chroma_db()
    collection_name = rp.store_resume_in_chroma(
        chunks=chunks,
        embeddings=embeddings,
        session_id="test_session_123",
        client=client,
    )

    assert collection_name.startswith("resume_")

    retrieved = rp.retrieve_resume_context(
        query="Python AWS backend APIs",
        session_id="test_session_123",
        k=5,
        client=client,
    )
    assert 1 <= len(retrieved) <= 5
    assert all(isinstance(item, str) for item in retrieved)


def test_preprocess_and_initialize_datasets(
    isolated_cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(rp, "_load_embedding_model", lambda: DummyModel())

    data_dir = isolated_cwd / "data"
    data_dir.mkdir(exist_ok=True)

    (data_dir / "interview_qa.csv").write_text(
        "question,answer,category\n"
        "Tell me about yourself,I am a software engineer,general\n"
        "What is Python?,A programming language,technical\n",
        encoding="utf-8",
    )

    (data_dir / "indian_jd_dataset.json").write_text(
        "["
        "{\"company\":\"Zepto\",\"role\":\"SDE\",\"skills\":[\"Python\",\"SQL\"],\"experience_level\":\"2-4 years\",\"hiring_notes\":\"Strong DSA\"},"
        "{\"company\":\"Infosys\",\"role\":\"SE\",\"skills\":[\"Java\"],\"experience_level\":\"0-2 years\",\"hiring_notes\":\"Communication focus\"}"
        "]",
        encoding="utf-8",
    )

    client = rp.initialize_chroma_db()
    rp.initialize_datasets_in_chromadb(client)

    qa = client.get_collection("interview_qa_dataset")
    jd = client.get_collection("indian_jd_dataset")

    assert qa.count() == 2
    assert jd.count() == 2


@pytest.fixture(autouse=True)
def cleanup_chroma() -> Generator[None, None, None]:
    yield
    shutil.rmtree("chroma_store", ignore_errors=True)
