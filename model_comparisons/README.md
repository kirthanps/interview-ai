# InterviewAI Model Comparisons

This folder contains model comparison execution with normal-session pipeline parity.
All artifacts are kept strictly inside this folder.

Generation and scoring now reuse the same core flow as the main interview app:
- resume ingestion + chunking + embeddings + Chroma retrieval
- shared dataset initialization (interview_qa + indian_jd)
- JD parsing with retrieved examples
- few-shot example retrieval for question generation
- prompt contracts from the main prompts/ files
- scoring flow and validation from main scorer pipeline

## Evaluated vs Judge Models

- Evaluated models:
  - llama-3.3-70b-versatile
  - llama-3.1-8b-instant
  - qwen/qwen3-32b
- Judge model (not ranked):
  - openai/gpt-oss-120b

## Folder Layout

- config/
  - experiment_config.yaml
  - judge_rubrics.yaml
- data/
  - resumes/
  - jds/
  - combinations.json
- prompts/
  - question_generation_prompt.txt
  - scoring_prompt.txt
  - judge_question_prompt.txt
  - judge_scoring_prompt.txt
- templates/
  - human_answers_template.json
  - human_answers_schema.md
- src/
  - pipeline implementation
- runs/
  - generated run artifacts

## Setup

Set API keys:
- GROQ_API_KEY
- OPENAI_API_KEY

Install dependencies (example):
- requests
- pyyaml
- PyPDF2
- scipy (optional, for p-values in stats)

Also ensure main app dependencies are available because model comparison reuses pipeline modules.

## Execution

From this folder:

- Prepare run skeleton:
  - python run_experiment.py --stage prepare
- Generate questions (27 calls):
  - python run_experiment.py --stage generate
- Scaffold 27 human answer files:
  - python run_experiment.py --stage scaffold_answers
- Validate answer collection (27 files, 270 responses):
  - python run_experiment.py --stage validate_answers
- Run scoring (81 calls):
  - python run_experiment.py --stage score
- Judge question phase:
  - python run_experiment.py --stage judge_questions
- Judge scoring phase:
  - python run_experiment.py --stage judge_scoring
- Analyze and build report:
  - python run_experiment.py --stage analyze
- Full pipeline:
  - python run_experiment.py --stage all

Optional run id:
- python run_experiment.py --run-id run_20260415_120000 --stage prepare

## Human Answers (270)

Answer format details are in:
- templates/human_answers_schema.md

Use one file per combo+source-model set in:
- runs/<run_id>/human_answers/

Pattern:
- <combo_id>_<source_model_alias>_answers.json

There are 27 files x 10 responses each = 270 answers.
