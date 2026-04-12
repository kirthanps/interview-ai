"""Project-wide constants used across InterviewAI modules."""

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K_RETRIEVAL = 5
MIN_ANSWER_LENGTH = 20

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_CACHE_DIR = "models"
CHROMA_STORE_DIR = "chroma_store"

INTERVIEW_QA_COLLECTION = "interview_qa_dataset"
INDIAN_JD_COLLECTION = "indian_jd_dataset"

SUPPORTED_ROLES = [
    "Software Engineer",
    "Backend Engineer",
    "Frontend Engineer",
    "Full Stack Developer",
    "Data Analyst",
    "Data Scientist",
    "Machine Learning Engineer",
    "DevOps Engineer",
    "SRE",
    "Product Manager",
]

DIFFICULTY_LEVELS = ["easy", "medium", "hard"]
FOCUS_AREAS = ["technical", "hr", "mixed", "system_design"]
