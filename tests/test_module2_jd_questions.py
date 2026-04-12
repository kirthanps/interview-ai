from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pipeline.jd_parser import (  # noqa: E402
    get_question_distribution,
    match_to_company_profile,
    parse_jd_with_llm,
    retrieve_few_shot_examples,
)
from pipeline.question_generator import Question, generate_questions  # noqa: E402


class DummyResponse:
    def __init__(self, content: str) -> None:
        self.content = content


class DummyLLM:
    def __init__(self, content: str) -> None:
        self._content = content

    def invoke(self, _: str) -> DummyResponse:
        return DummyResponse(self._content)


class FakeCollection:
    def __init__(self, documents, metadatas=None) -> None:
        self._documents = documents
        self._metadatas = metadatas or [{} for _ in documents]

    def query(self, query_texts=None, query_embeddings=None, n_results=1, **kwargs):
        docs = self._documents[:n_results]
        metas = self._metadatas[:n_results]
        return {"documents": [docs], "metadatas": [metas]}


class FakeChromaClient:
    def __init__(self, collections) -> None:
        self._collections = collections

    def get_collection(self, name: str):
        return self._collections[name]


def test_parse_jd_with_llm_success() -> None:
    jd_text = (
        "Senior Backend Engineer at Zepto with 3-5 years of experience. "
        "Required skills include Python, Django, PostgreSQL, AWS, Docker, and Kubernetes. "
        "The role includes designing scalable APIs and distributed systems in Bangalore."
    )

    llm = DummyLLM(
        """
```json
{
  "role": "Senior Backend Engineer",
  "company": "Zepto",
  "skills": ["Python", "Django", "AWS"],
  "experience_level": "3-5 years",
  "tools": ["Docker", "Kubernetes"]
}
```
"""
    )

    parsed = parse_jd_with_llm(jd_text=jd_text, llm=llm)

    assert parsed["role"] == "Senior Backend Engineer"
    assert parsed["company"] == "Zepto"
    assert "Python" in parsed["skills"]
    assert parsed["experience_level"] == "3-5 years"


def test_parse_jd_with_llm_short_jd_raises() -> None:
    with pytest.raises(ValueError, match="JD too short"):
        parse_jd_with_llm(jd_text="short jd", llm=DummyLLM("{}"))


def test_match_to_company_profile_text_doc_format() -> None:
    profile_doc = (
        "Company: Zepto\n"
        "Role: Backend Engineer\n"
        "Skills: Python, Django, AWS\n"
        "Experience Level: 3-5 years\n"
        "Hiring Notes: Focus on high-throughput systems and DSA"
    )
    client = FakeChromaClient(
        {
            "indian_jd_dataset": FakeCollection(
                [profile_doc],
                metadatas=[{"company": "Zepto", "role": "Backend Engineer"}],
            )
        }
    )

    parsed_jd = {
        "role": "Senior Backend Engineer",
        "company": "Zepto",
        "skills": ["Python", "Django"],
    }

    profile = match_to_company_profile(parsed_jd, chroma_client=client)

    assert profile["company"] == "Zepto"
    assert profile["role"] == "Backend Engineer"
    assert "high-throughput" in profile["hiring_notes"].lower()


def test_retrieve_few_shot_examples_parses_documents() -> None:
    qa_docs = [
        "Question: What is Python?\nAnswer: A programming language.",
        "Q: Explain REST APIs.\nA: APIs that use HTTP methods.",
    ]
    client = FakeChromaClient(
        {"interview_qa_dataset": FakeCollection(qa_docs)})

    examples = retrieve_few_shot_examples(
        ["Python", "REST"], chroma_client=client, k=2)

    assert len(examples) == 2
    assert examples[0]["question"]
    assert examples[0]["answer"]


def test_generate_questions_success() -> None:
    parsed_jd = {
        "role": "Backend Engineer",
        "company": "Zepto",
        "skills": ["Python", "Django", "AWS"],
        "experience_level": "3-5 years",
    }
    company_profile = {
        "company": "Zepto",
        "hiring_notes": "Prioritize system design, scale, and clean API design",
    }
    preferences = {"num_questions": 2,
                   "difficulty": "medium", "focus_area": "mixed"}
    resume_context = [
        "Built a fintech API with JWT auth and token rotation.",
        "Improved latency by optimizing PostgreSQL queries.",
    ]

    llm = DummyLLM(
        """
[
  {
    "question": "In your fintech API project, how did you secure refresh tokens?",
    "type": "technical",
    "hint": ["token rotation", "revocation strategy"],
    "difficulty": "medium"
  },
  {
    "question": "Design a high-throughput order service for rapid delivery.",
    "type": "system_design",
    "hint": ["read/write paths", "scaling and caching"],
    "difficulty": "hard"
  }
]
"""
    )

    qa_docs = [
        "Question: Explain CAP theorem.\nAnswer: Trade-offs in distributed systems.",
        "Question: What is indexing in SQL?\nAnswer: Improves query performance.",
    ]
    client = FakeChromaClient(
        {"interview_qa_dataset": FakeCollection(qa_docs)})

    questions = generate_questions(
        parsed_jd=parsed_jd,
        company_profile=company_profile,
        resume_context=resume_context,
        preferences=preferences,
        chroma_client=client,
        llm=llm,
    )

    assert len(questions) == 2
    assert all(isinstance(item, Question) for item in questions)
    assert questions[0].question
    assert questions[0].hint


def test_get_question_distribution_contains_target_count() -> None:
    result = get_question_distribution(8, "mixed", "medium")
    assert "Target count: 8" in result
