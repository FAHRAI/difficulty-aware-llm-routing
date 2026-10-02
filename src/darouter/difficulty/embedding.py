"""Sentence embeddings of router inputs (all-MiniLM-L6-v2, 256 word pieces, L2-normalised)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from darouter.difficulty.features import head_tail

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_SEQ_LENGTH = 256


def load_encoder(device: str | None = None):
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    encoder = SentenceTransformer(MODEL_NAME, device=device)
    encoder.max_seq_length = MAX_SEQ_LENGTH
    return encoder


def embed(prompts: Sequence[str], encoder=None, batch_size: int = 128) -> np.ndarray:
    encoder = encoder or load_encoder()
    vectors = encoder.encode(
        [head_tail(t) for t in prompts],
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    return vectors.astype(np.float32)
