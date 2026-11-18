# InterviewAI | Resume-aware mock interview coach

**Business problem.** Generic question banks give candidates little practice connecting their own project experience to a specific job opening.

**Solution.** InterviewAI takes a resume PDF, a job description, and optional company context. It pulls relevant examples from a vector index, builds a tailored practice interview, and returns scored feedback, improvement suggestions, and a downloadable PDF.

**Stack.** Python, Streamlit, LangChain, Groq, ChromaDB, Sentence Transformers, PyPDF2, fpdf2, and pytest.

**My role.** I developed the application across the Streamlit interface, document ingestion, retrieval and prompts, interview state, scoring, and PDF delivery.

**Technical challenges.** The retrieval pipeline embeds resume chunks and locates the closest matches for the target role. The question generator checks whether the model returned enough distinct questions. The scoring pipeline checks per-answer coverage and numeric bounds. The UI guards against scoring an unfinished interview and resets previous results when a new question set is generated.

**What clients can assess.** The repository shows my ability to translate a document-heavy workflow into a usable application, integrate an LLM with structured validation, and deliver results in a portable format.

## Upwork description

I built InterviewAI, a Python and Streamlit application that helps candidates practice for a specific job opening. A candidate uploads a resume PDF, pastes the job description, and can add context about the team or role. The application extracts resume text, uses ChromaDB and sentence embeddings to retrieve relevant context, and calls Groq through LangChain to generate a tailored set of interview questions. Candidates answer one question at a time and receive rubric-based feedback on relevance, depth, and clarity, along with suggestions for improvement and a downloadable PDF. The project demonstrates how I build document-driven AI workflows, validate structured model responses, manage multi-step application state, and turn generated feedback into an artifact people can use after the session.
