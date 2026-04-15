import json
import os
from pathlib import Path

import requests


def main() -> None:
    api_key = os.getenv("GROQ_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("Set GROQ_API_KEY before running this script.")

    url = "https://api.groq.com/openai/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.get(url, headers=headers, timeout=60)
    response.raise_for_status()
    models = response.json()

    output_dir = Path(__file__).resolve().parent / "runs" / "model_probe"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "available_groq_models.json"
    output_file.write_text(json.dumps(
        models, indent=2, ensure_ascii=True), encoding="utf-8")

    for model in models.get("data", []):
        print(f"Model ID: {model.get('id', '')}")

    print(f"Saved model list to: {output_file}")


if __name__ == "__main__":
    main()
