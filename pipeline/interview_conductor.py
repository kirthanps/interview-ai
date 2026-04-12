"""Interview conductor and session memory utilities for Module 3."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from pipeline.question_generator import Question as GeneratedQuestion

logger = logging.getLogger(__name__)

MIN_ANSWER_LENGTH = 20

try:
    from langchain.memory import ConversationBufferMemory  # type: ignore
except Exception:  # pragma: no cover
    class ConversationBufferMemory:  # type: ignore[override]
        """Small fallback memory when langchain package is unavailable."""

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.records: List[Dict[str, Any]] = []

        def save_context(self, inputs: Dict[str, str], outputs: Dict[str, str]) -> None:
            self.records.append({"inputs": inputs, "outputs": outputs})


@dataclass
class InterviewQuestion:
    """Question state used during interview progression."""

    question: str
    type: str
    hint: List[str]
    difficulty: str
    is_answered: bool = False
    answer_text: str = ""
    skip_flag: bool = False
    answer_timestamp: Optional[datetime] = None


@dataclass
class QAPair:
    """Question/answer event captured during the session."""

    question: str
    answer: str
    question_type: str
    question_difficulty: str
    was_skipped: bool = False
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class InterviewSession:
    """Mutable interview session state for a single candidate."""

    session_id: str
    role: str
    company: str
    candidate_name: Optional[str]
    resume_insights: Dict[str, Any]
    questions: List[InterviewQuestion]
    current_question_index: int = 0
    qa_pairs: List[QAPair] = field(default_factory=list)
    interview_started_at: datetime = field(default_factory=datetime.now)
    interview_ended_at: Optional[datetime] = None
    conversation_memory: Optional[ConversationBufferMemory] = None

    def get_progress_percentage(self) -> float:
        if not self.questions:
            return 0.0
        return (len(self.qa_pairs) / len(self.questions)) * 100.0

    def is_completed(self) -> bool:
        return self.current_question_index >= len(self.questions)

    def get_total_answered(self) -> int:
        return sum(1 for qa in self.qa_pairs if not qa.was_skipped)


def _to_interview_question(item: Any) -> InterviewQuestion:
    """Normalize question payload from Module 2 output or dict input."""
    if isinstance(item, InterviewQuestion):
        return item

    if isinstance(item, GeneratedQuestion):
        return InterviewQuestion(
            question=item.question,
            type=item.type,
            hint=list(item.hint),
            difficulty=item.difficulty,
        )

    if isinstance(item, dict):
        return InterviewQuestion(
            question=str(item.get("question", "")).strip(),
            type=str(item.get("type", "technical")).strip() or "technical",
            hint=list(item.get("hint", [])) if isinstance(
                item.get("hint", []), list) else [],
            difficulty=str(item.get("difficulty", "medium")
                           ).strip() or "medium",
        )

    raise TypeError("Unsupported question payload type")


def initialize_interview_session(
    session_id: str,
    questions: List[Any],
    role: str,
    company: str,
    resume_insights: Dict[str, Any],
    candidate_name: Optional[str] = None,
) -> InterviewSession:
    """Create interview session state and initialize conversation memory."""
    if not session_id or not session_id.strip():
        raise ValueError("session_id is required")

    question_objects: List[InterviewQuestion] = []
    for raw_question in questions:
        q = _to_interview_question(raw_question)
        if q.question:
            question_objects.append(q)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        human_prefix="Candidate",
        ai_prefix="Interviewer",
    )

    session = InterviewSession(
        session_id=session_id,
        role=role,
        company=company,
        candidate_name=candidate_name,
        resume_insights=resume_insights,
        questions=question_objects,
        conversation_memory=memory,
    )

    logger.info(
        "Initialized interview session %s with %s questions",
        session_id,
        len(question_objects),
    )
    return session


def get_first_question(session: InterviewSession) -> InterviewQuestion:
    """Return the first question in a session."""
    if not session.questions:
        raise ValueError("No questions available in session")
    return session.questions[0]


def get_current_question(session: InterviewSession) -> Optional[InterviewQuestion]:
    """Return current question or None if session is complete."""
    if session.is_completed():
        return None
    if session.current_question_index < 0:
        return None
    if session.current_question_index >= len(session.questions):
        return None
    return session.questions[session.current_question_index]


def get_next_question(session: InterviewSession) -> Optional[InterviewQuestion]:
    """Peek next question without mutating session state."""
    next_idx = session.current_question_index + 1
    if next_idx >= len(session.questions):
        return None
    return session.questions[next_idx]


def advance_to_next_question(session: InterviewSession) -> bool:
    """Advance pointer; return False when no more questions remain."""
    next_idx = session.current_question_index + 1
    if next_idx >= len(session.questions):
        logger.info("Interview completed for session %s", session.session_id)
        session.current_question_index = len(session.questions)
        return False

    session.current_question_index = next_idx
    return True


def submit_answer(
    session: InterviewSession,
    answer_text: str,
    validate: bool = True,
) -> Dict[str, Any]:
    """Submit answer for current question and advance state."""
    validation_errors: List[str] = []
    normalized_answer = answer_text.strip()

    if validate and len(normalized_answer) < MIN_ANSWER_LENGTH:
        validation_errors.append(
            f"Answer too short (minimum {MIN_ANSWER_LENGTH} characters)")

    if validation_errors:
        return {
            "success": False,
            "message": "Please provide a longer answer with more detail",
            "validation_errors": validation_errors,
            "next_question": None,
            "is_interview_complete": False,
        }

    current_q = get_current_question(session)
    if current_q is None:
        return {
            "success": False,
            "message": "No current question available",
            "validation_errors": ["Session error"],
            "next_question": None,
            "is_interview_complete": True,
        }

    current_q.is_answered = True
    current_q.answer_text = normalized_answer
    current_q.answer_timestamp = datetime.now()

    if session.conversation_memory is not None:
        session.conversation_memory.save_context(
            {"input": current_q.question},
            {"output": normalized_answer},
        )

    session.qa_pairs.append(
        QAPair(
            question=current_q.question,
            answer=normalized_answer,
            question_type=current_q.type,
            question_difficulty=current_q.difficulty,
            was_skipped=False,
        )
    )

    interview_complete = not advance_to_next_question(session)
    next_q = None if interview_complete else get_current_question(session)

    return {
        "success": True,
        "message": "Answer submitted successfully",
        "validation_errors": [],
        "next_question": next_q,
        "is_interview_complete": interview_complete,
    }


def skip_question(session: InterviewSession) -> Dict[str, Any]:
    """Skip current question and move to the next one."""
    current_q = get_current_question(session)
    if current_q is None:
        return {
            "success": False,
            "message": "No current question to skip",
            "next_question": None,
            "is_interview_complete": True,
        }

    current_q.skip_flag = True
    current_q.is_answered = False

    session.qa_pairs.append(
        QAPair(
            question=current_q.question,
            answer="[SKIPPED]",
            question_type=current_q.type,
            question_difficulty=current_q.difficulty,
            was_skipped=True,
        )
    )

    if session.conversation_memory is not None:
        session.conversation_memory.save_context(
            {"input": current_q.question},
            {"output": "[Question was skipped by candidate]"},
        )

    interview_complete = not advance_to_next_question(session)
    next_q = None if interview_complete else get_current_question(session)

    return {
        "success": True,
        "message": "Question skipped. Skipped questions will receive 0 score.",
        "next_question": next_q,
        "is_interview_complete": interview_complete,
    }


def get_conversation_history(session: InterviewSession) -> str:
    """Build formatted conversation transcript from recorded Q&A pairs."""
    history = [
        f"Interview for {session.role} at {session.company}", "=" * 50, ""]

    for qa in session.qa_pairs:
        history.append(f"Q: {qa.question}")
        if qa.was_skipped:
            history.append("A: [Question Skipped]")
        else:
            history.append(f"A: {qa.answer}")
        history.append("-" * 40)

    return "\n".join(history).strip()


def get_qa_pairs_for_scoring(session: InterviewSession) -> List[Dict[str, Any]]:
    """Return scoring-engine friendly QA payload."""
    return [
        {
            "question": qa.question,
            "answer": qa.answer,
            "type": qa.question_type,
            "difficulty": qa.question_difficulty,
            "was_skipped": qa.was_skipped,
        }
        for qa in session.qa_pairs
    ]


def finalize_interview(session: InterviewSession) -> Dict[str, Any]:
    """Mark interview end time and return summary metrics."""
    session.interview_ended_at = datetime.now()
    duration_seconds = int(
        (session.interview_ended_at - session.interview_started_at).total_seconds()
    )

    answered = session.get_total_answered()
    total = len(session.questions)

    return {
        "session_id": session.session_id,
        "role": session.role,
        "company": session.company,
        "total_questions": total,
        "answered_questions": answered,
        "skipped_questions": total - answered,
        "duration_seconds": duration_seconds,
        "qa_pairs": session.qa_pairs,
    }
