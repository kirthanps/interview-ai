"""Synthetic dataset generator for InterviewAI.

Creates demo-ready datasets:
- interview_qa.csv with question, answer, category
- indian_jd_dataset.json with company role profiles
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Tuple


QA_TEMPLATES = [
    ("Tell me about yourself",
     "Provide a concise profile summary with relevant achievements.", "hr"),
    ("Why do you want this role?",
     "Connect your skills and goals to the role requirements.", "hr"),
    ("How do you handle tight deadlines?",
     "Prioritize tasks, communicate risks, and deliver iteratively.", "situational"),
    ("Describe a challenging bug you fixed.",
     "Explain diagnosis steps, root cause, and final resolution.", "technical"),
    ("How do you ensure code quality?",
     "Use tests, reviews, static checks, and clear documentation.", "technical"),
    ("How do you resolve team conflict?",
     "Listen actively, align on goals, and agree on actionable steps.", "behavioral"),
]

JD_BASE = [
    {
        "company": "TCS",
        "role": "Software Engineer",
        "skills": ["Java", "SQL", "OOP"],
        "experience_level": "1-2 years",
        "hiring_notes": "Strong fundamentals and communication are important.",
    },
    {
        "company": "Infosys",
        "role": "Backend Developer",
        "skills": ["Python", "Django", "SQL"],
        "experience_level": "2-3 years",
        "hiring_notes": "Focus on API design and maintainable backend practices.",
    },
    {
        "company": "Wipro",
        "role": "Application Developer",
        "skills": ["Java", "Spring Boot", "REST"],
        "experience_level": "2-4 years",
        "hiring_notes": "Look for practical implementation depth and debugging approach.",
    },
    {
        "company": "Zepto",
        "role": "Backend Engineer",
        "skills": ["Python", "Microservices", "Redis"],
        "experience_level": "3-5 years",
        "hiring_notes": "Scalability and low-latency system design are valuable.",
    },
    {
        "company": "CRED",
        "role": "Software Development Engineer",
        "skills": ["Kotlin", "System Design", "PostgreSQL"],
        "experience_level": "3-5 years",
        "hiring_notes": "Strong product thinking and robust engineering choices are expected.",
    },
]


def build_synthetic_qa_pairs(target_count: int = 500) -> List[dict]:
    """Build deterministic synthetic QA records."""
    if target_count <= 0:
        raise ValueError("target_count must be > 0")

    rows: List[dict] = []
    for idx in range(target_count):
        question, answer, category = QA_TEMPLATES[idx % len(QA_TEMPLATES)]
        rows.append(
            {
                "question": f"{question} (variation {idx + 1})",
                "answer": answer,
                "category": category,
            }
        )
    return rows


def build_synthetic_jd_profiles(target_count: int = 30) -> List[dict]:
    """Build deterministic synthetic JD profiles."""
    if target_count <= 0:
        raise ValueError("target_count must be > 0")

    profiles: List[dict] = []
    for idx in range(target_count):
        base = JD_BASE[idx % len(JD_BASE)]
        profile = dict(base)
        profile["company"] = f"{base['company']} {idx + 1}"
        profiles.append(profile)
    return profiles


def _write_qa_csv(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["question", "answer", "category"])
        for row in rows:
            writer.writerow([row["question"], row["answer"], row["category"]])


def _write_jd_json(rows: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)


def generate_default_datasets(output_dir: str = "data") -> Tuple[Path, Path]:
    """Generate default dataset files and return their paths."""
    out_dir = Path(output_dir)
    qa_path = out_dir / "interview_qa.csv"
    jd_path = out_dir / "indian_jd_dataset.json"

    _write_qa_csv(build_synthetic_qa_pairs(500), qa_path)
    _write_jd_json(build_synthetic_jd_profiles(30), jd_path)

    return qa_path, jd_path


if __name__ == "__main__":
    qa_out, jd_out = generate_default_datasets("data")
    print(f"Generated: {qa_out}")
    print(f"Generated: {jd_out}")
