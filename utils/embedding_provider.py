"""Shared sentence-transformer loader utilities."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, List

from utils.constants import EMBEDDING_MODEL, MODEL_CACHE_DIR


@lru_cache(maxsize=1)
def get_embedding_model() -> Any:
    """Load and cache the embedding model with project-local cache folder."""
    from sentence_transformers import SentenceTransformer

    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    return SentenceTransformer(EMBEDDING_MODEL, cache_folder=MODEL_CACHE_DIR)


def embed_query_text(query_text: str) -> List[float]:
    """Embed one query string into a list[float] vector."""
    model = get_embedding_model()
    vector = model.encode([query_text], show_progress_bar=False)[0]
    return vector.tolist() if hasattr(vector, "tolist") else list(vector)
