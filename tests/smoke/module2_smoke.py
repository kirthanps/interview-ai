"""Module 2 integration smoke script.

Run from project root:
    python -m tests.smoke.module2_smoke
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from pipeline.jd_parser import match_to_company_profile, parse_jd_with_llm
from pipeline.question_generator import generate_questions
from pipeline.resume_processor import initialize_chroma_db, initialize_datasets_in_chromadb


def _write_seed_qa_csv(csv_path: Path) -> None:
    rows = [
        ["question", "answer", "category"],
        [
            "What is Python?",
            "A high-level programming language used in backend and data systems.",
            "technical",
        ],
        [
            "How do you handle API failures?",
            "Use retries, timeouts, circuit breakers, and observability.",
            "technical",
        ],
        [
            "Describe a challenge you solved",
            "Explain context, action, and measurable outcome.",
            "hr",
        ],
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def _is_valid_qa_csv(csv_path: Path) -> bool:
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as file:
            reader = list(csv.reader(file))
    except Exception:
        return False

    if not reader:
        return False

    header = [cell.strip().lower() for cell in reader[0]]
    if header != ["question", "answer", "category"]:
        return False

    for row in reader[1:]:
        if len(row) != 3:
            return False

    return len(reader) >= 2


def _ensure_minimal_datasets() -> None:
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)

    qa_csv = data_dir / "interview_qa.csv"
    if not qa_csv.exists() or not _is_valid_qa_csv(qa_csv):
        _write_seed_qa_csv(qa_csv)

    jd_json = data_dir / "indian_jd_dataset.json"
    if not jd_json.exists():
        jd_payload = [
            {
                "company": "Zepto",
                "role": "Senior Backend Engineer",
                "skills": ["Python", "Django", "PostgreSQL", "AWS", "Docker"],
                "experience_level": "3-5 years",
                "hiring_notes": "Focus on high-throughput backend design, SQL tuning, and pragmatic system design.",
            },
            {
                "company": "Infosys",
                "role": "Software Engineer",
                "skills": ["Java", "SQL", "OOP"],
                "experience_level": "1-2 years",
                "hiring_notes": "Balanced focus on fundamentals, communication, and clean coding.",
            },
        ]
        jd_json.write_text(json.dumps(
            jd_payload, ensure_ascii=True, indent=2), encoding="utf-8")


def main() -> None:
    client = initialize_chroma_db()
    _ensure_minimal_datasets()
    initialize_datasets_in_chromadb(client)

    jd_text = (
        "Senior Backend Engineer at Zepto with 3-5 years experience. "
        "Required: Python, Django, PostgreSQL, AWS, Docker, Kubernetes. "
        "You will build scalable APIs, optimize high-throughput systems, "
        "and work on distributed backend architecture."
    )

    resume_context = [
        "Built fintech backend APIs in Python and Django with JWT auth and token rotation.",
        "Improved PostgreSQL query latency by introducing indexing and query plan tuning.",
        "Designed async workflows and worker queues for high request throughput.",
        "Deployed services on AWS EC2, RDS, and S3 with Docker-based CI/CD.",
        "Worked on system design trade-offs for read-heavy microservices.",
    ]

    preferences = {
        "num_questions": 6,
        "difficulty": "medium",
        "focus_area": "mixed",
    }

    print("\n[1] Parsing JD...")
    parsed_jd = parse_jd_with_llm(jd_text)
    print(parsed_jd)

    print("\n[2] Matching company profile...")
    company_profile = match_to_company_profile(parsed_jd, client)
    print(company_profile)

    print("\n[3] Generating questions...")
    questions = generate_questions(
        parsed_jd=parsed_jd,
        company_profile=company_profile,
        resume_context=resume_context,
        preferences=preferences,
        chroma_client=client,
    )

    print(f"Generated {len(questions)} questions:\n")
    for index, q in enumerate(questions, start=1):
        qd = asdict(q)
        print(f"Q{index}: {qd['question']}")
        print(f"  type={qd['type']} difficulty={qd['difficulty']}")
        print(f"  hints={qd['hint']}")
        print()


if __name__ == "__main__":
    main()
