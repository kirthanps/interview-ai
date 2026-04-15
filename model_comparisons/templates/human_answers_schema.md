# Human Answer Format (270 Answers)

Store one JSON file per question set in:
- runs/<run_id>/human_answers/

File pattern:
- <combo_id>_<source_model_alias>_answers.json

There are 27 question sets total (9 combos x 3 source models).
Each file contains 10 responses.
Total responses = 270.

## Required JSON shape

```json
{
  "combo_id": "combo_01",
  "source_model_alias": "llama33",
  "source_model_id": "llama-3.3-70b-versatile",
  "difficulty": "hard",
  "focus_area": "mixed",
  "num_questions": 10,
  "responses": [
    {
      "question_id": "combo_01_llama33_q01",
      "question_text": "exact question",
      "answer_text": "exact raw human answer",
      "answer_label": "SKIPPED|SHORT|POOR|AVERAGE|GOOD|STRONG",
      "timestamp_utc": "YYYY-MM-DDTHH:MM:SSZ"
    }
  ]
}
```

## Rules

- Keep answer_text raw; no rewriting/cleanup.
- Keep question_text exactly as generated.
- Use UTC timestamps.
- answer_label must be one of:
  - SKIPPED
  - SHORT
  - POOR
  - AVERAGE
  - GOOD
  - STRONG
