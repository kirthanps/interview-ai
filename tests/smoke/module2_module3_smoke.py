"""Module 2 -> Module 3 integration smoke script.

Run from project root:
    python -m tests.smoke.module2_module3_smoke
"""

from __future__ import annotations

from dataclasses import asdict

from pipeline.interview_conductor import (
    finalize_interview,
    get_conversation_history,
    get_current_question,
    get_qa_pairs_for_scoring,
    initialize_interview_session,
    skip_question,
    submit_answer,
)
from tests.smoke.module2_smoke import _ensure_minimal_datasets
from pipeline.jd_parser import match_to_company_profile, parse_jd_with_llm
from pipeline.question_generator import generate_questions
from pipeline.resume_processor import initialize_chroma_db, initialize_datasets_in_chromadb


def main() -> None:
    client = initialize_chroma_db()
    _ensure_minimal_datasets()
    initialize_datasets_in_chromadb(client)

    jd_text = (
        "Senior Backend Engineer at Zepto with 3-5 years experience. "
        "Required: Python, Django, PostgreSQL, AWS, Docker, Kubernetes."
    )

    resume_context = [
        "Built fintech backend APIs in Python and Django with JWT auth and token rotation.",
        "Improved PostgreSQL query latency by introducing indexing and query plan tuning.",
        "Designed async workflows and worker queues for high request throughput.",
    ]

    preferences = {"num_questions": 4,
                   "difficulty": "medium", "focus_area": "mixed"}

    parsed_jd = parse_jd_with_llm(jd_text)
    company_profile = match_to_company_profile(parsed_jd, client)
    questions = generate_questions(
        parsed_jd=parsed_jd,
        company_profile=company_profile,
        resume_context=resume_context,
        preferences=preferences,
        chroma_client=client,
    )

    session = initialize_interview_session(
        session_id="smoke_m2_m3",
        questions=[asdict(q) for q in questions],
        role=parsed_jd.get("role", "Unknown Role"),
        company=company_profile.get("company", "Unknown Company"),
        resume_insights={"skills": parsed_jd.get("skills", [])},
        candidate_name="Smoke Candidate",
    )

    print(f"Session started with {len(session.questions)} questions")

    step = 1
    while True:
        current_q = get_current_question(session)
        if current_q is None:
            break

        print(f"\n[Q{step}] {current_q.question}")
        if step == 2:
            print("Action: skip")
            result = skip_question(session)
        else:
            answer = (
                "I would structure the solution with clear trade-offs, monitoring, "
                "and production-safe rollout practices based on prior backend experience."
            )
            print("Action: submit answer")
            result = submit_answer(session, answer)

        print(
            f"Result: success={result['success']} complete={result['is_interview_complete']}"
        )
        step += 1

    print("\nConversation history:\n")
    print(get_conversation_history(session))

    scoring_payload = get_qa_pairs_for_scoring(session)
    print(f"\nScoring payload size: {len(scoring_payload)}")

    summary = finalize_interview(session)
    print("\nFinal summary:")
    print({
        "session_id": summary["session_id"],
        "total_questions": summary["total_questions"],
        "answered_questions": summary["answered_questions"],
        "skipped_questions": summary["skipped_questions"],
        "duration_seconds": summary["duration_seconds"],
    })


if __name__ == "__main__":
    main()
