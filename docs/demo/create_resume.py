"""Create a contact-free fictional resume for the InterviewAI walkthrough."""

from pathlib import Path

from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", "B", 24)
pdf.cell(0, 14, "Mira Vale", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", size=12)
pdf.multi_cell(
    0,
    8,
    "Backend Engineer | Fictional demo candidate\n\n"
    "Three years of experience building Python services and PostgreSQL-backed APIs.\n\n"
    "Parcel event simulator: Built a FastAPI service that ingests delivery events, "
    "uses a unique event ID to handle retries, and tracks failures in a dead-letter queue.\n\n"
    "Inventory API: Added pytest integration tests for transactional updates and "
    "wrote runbooks for failed background jobs.\n\n"
    "Skills: Python, FastAPI, PostgreSQL, SQL, pytest, Docker.\n"
    "Education: Bachelor of Science in Computer Science.",
)
pdf.output(str(Path(__file__).with_name("mira-vale-resume.pdf")))
