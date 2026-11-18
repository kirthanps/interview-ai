# InterviewAI

InterviewAI is an AI-powered mock interview coach.

You upload your resume, paste a target job description, answer interview questions, and get a scored feedback report with a downloadable PDF.

The app is designed for an Indian job-market context and uses:
- resume-aware questioning
- company/JD-aware context
- structured LLM outputs for preview, questions, and scoring

## What This Project Does

In simple terms, the app does this:
1. Reads your resume PDF.
2. Understands the job description you are targeting.
3. Generates tailored interview questions.
4. Runs a mock interview flow.
5. Scores your answers and gives practical feedback.

## User Story Workflow (Simple End-to-End)

1. Open app and start a session.
2. Upload resume PDF and paste JD.
3. Pick preferences: difficulty, question count, and focus area.
4. App analyzes resume and JD, then generates questions.
5. Preview page shows what the system understood.
6. You answer questions one by one.
7. Results page shows per-question and overall scores.
8. You download the final PDF report.

## How Data Flows Through The System

### 1) Resume Ingestion
- Input: resume PDF.
- Processing: PDF text extraction, chunking, embedding.
- Storage: session-specific ChromaDB collection (resume vectors).

### 2) JD Ingestion
- Input: pasted JD text.
- Processing: JD text embedding.
- Storage: session-specific ChromaDB collection (JD vectors).

### 3) Shared Dataset Warmup
At startup, shared datasets are loaded into persistent ChromaDB collections:
- interview Q&A dataset from data/interview_qa.csv
- Indian JD dataset from data/indian_jd_dataset.json

### 4) LLM Calls and What Is Sent

Resume insights call:
- Sent to LLM: extracted resume text.
- Returned: structured resume insights JSON.

JD parsing call:
- Sent to LLM: user JD text + top-k matched JD examples from vector DB.
- Returned: structured JD JSON.

Question generation call:
- Sent to LLM:
  - resume context (retrieved chunks)
  - full resume text
  - parsed JD JSON
  - company profile context
  - top-k few-shot interview examples from Q&A dataset
  - user preferences (difficulty, focus, question count)
- Returned: structured questions JSON with type, difficulty, hints, rationale.

Scoring call:
- Sent to LLM:
  - full interview transcript
  - resume insights JSON
  - JD insights JSON
  - generated questions JSON
- Returned: structured scoring JSON (per-question scores, overall metrics, readiness, feedback, resources).

### 5) What Is Stored Where

Stored in ChromaDB:
- session resume chunks/vectors
- session JD vectors
- shared interview Q&A vectors
- shared Indian JD vectors

Stored in session state:
- raw resume text
- raw JD text
- parsed resume insights
- parsed JD insights
- generated questions
- interview Q/A pairs
- scoring output
- report path

## Project Structure

- app.py: Streamlit entry point.
- pages/: landing, setup, preview, interview, results pages.
- pipeline/: core backend logic (RAG, parsing, generation, scoring).
- prompts/: LLM prompt templates.
- report/: PDF report generation.
- data/: input datasets.
- tests/: unit tests.

## Setup and Run

## Prerequisites

- Python 3.11+ (current workspace is running 3.14).
- A GROQ API key.

## 1) Create and activate virtual environment

Windows CMD:

```bat
python -m venv venv
venv\Scripts\activate
```

PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## 2) Install dependencies

```bat
pip install -r requirements.txt
```

## 3) Configure environment

Create a .env file in project root with:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## 4) Ensure datasets exist

Required files:
- data/interview_qa.csv
- data/indian_jd_dataset.json

If missing, generate them with:

```bat
python data_processing/generate_synthetic_datasets.py
```

## 5) Run the app

```bat
streamlit run app.py
```

## 6) Run tests

```bat
pytest -q
```

## Notes

- The app caches embeddings and vector collections locally.
- First run may take longer due to model and dataset warmup.
- Some dependency deprecation warnings can appear on Python 3.14; tests still pass.

## Documentation

Detailed backend contract and rich JSON schemas are documented in:
- ai_files/docs and plans/rich_insights_backend_spec.md
