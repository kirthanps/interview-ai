from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key_env: str


@dataclass(frozen=True)
class ModelConfig:
    model_id: str
    alias: str
    provider: str


class OpenAICompatibleClient:
    def __init__(
        self,
        providers: dict[str, ProviderConfig],
        request_delay_seconds: float,
        max_retries: int,
        backoff_base_seconds: float,
        temperature: float = 0.2,
        max_tokens: int = 3000,
    ) -> None:
        self.providers = providers
        self.request_delay_seconds = request_delay_seconds
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _endpoint_for(self, provider: ProviderConfig) -> str:
        return provider.base_url.rstrip("/") + "/chat/completions"

    def _api_key_for(self, provider: ProviderConfig) -> str:
        key = os.getenv(provider.api_key_env, "").strip()
        if not key:
            raise RuntimeError(
                f"Missing API key env var: {provider.api_key_env}")
        return key

    def completion_text(
        self,
        model: ModelConfig,
        prompt: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        system_prompt: str = "Return strict JSON only.",
    ) -> str:
        provider = self.providers[model.provider]
        api_key = self._api_key_for(provider)

        payload = {
            "model": model.model_id,
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": self.max_tokens if max_tokens is None else max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        }

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        endpoint = self._endpoint_for(provider)
        for attempt in range(1, self.max_retries + 1):
            try:
                response = requests.post(
                    endpoint, headers=headers, json=payload, timeout=120)
                if response.status_code == 429:
                    raise RuntimeError("rate_limited")
                response.raise_for_status()
                body = response.json()
                content = body["choices"][0]["message"]["content"]
                time.sleep(self.request_delay_seconds)
                return str(content)
            except Exception:
                if attempt == self.max_retries:
                    raise
                time.sleep(self.backoff_base_seconds * (2 ** (attempt - 1)))

        raise RuntimeError("Failed to obtain model completion after retries")

    def completion_json(self, model: ModelConfig, prompt: str) -> dict[str, Any]:
        return json.loads(self.completion_text(model, prompt))
