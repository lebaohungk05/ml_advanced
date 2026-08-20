"""Dependency-free stand-in embedder so the whole frame is runnable and testable.

No torch, no GPU, no downloads. It is deterministic across processes and
machines: every token is hashed to a fixed pseudo-random unit vector and a text
is the L2-normalized sum of its token vectors (the classic hashing trick). That
gives real lexical-overlap behaviour — "váy hai dây đi biển" scores highest on
products whose text shares those tokens — which is enough to demo and test the
pipeline end to end. It is *not* semantic: swapping in a real backbone
(SigLIP 2, CLIP) is the point of the Embedder port.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import unicodedata
from collections.abc import Sequence

from src.core.models import Matrix, Vector

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
DEFAULT_DIM = 64


class FakeEmbedder:
    """Hash-based deterministic Embedder. Satisfies the full Embedder port."""

    def __init__(self, dim: int = DEFAULT_DIM, name: str = "fake") -> None:
        if dim < 2:
            raise ValueError("dim must be >= 2")
        self._dim = dim
        self._name = name
        self._token_cache: dict[str, Vector] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    def encode_text(self, texts: Sequence[str]) -> Matrix:
        return [self._encode_one_text(text) for text in texts]

    def encode_image(self, images: Sequence[bytes]) -> Matrix:
        # Identical bytes always give the identical vector, so an image query
        # using a catalog image retrieves that exact product first.
        return [self._seeded_unit_vector(hashlib.sha256(data).hexdigest()) for data in images]

    def _encode_one_text(self, text: str) -> Vector:
        tokens = _tokenize(text)
        if not tokens:
            return self._seeded_unit_vector(f"__empty__{text}")
        summed = [0.0] * self._dim
        for token in tokens:
            for i, value in enumerate(self._token_vector(token)):
                summed[i] += value
        return _l2_normalize(summed) or self._seeded_unit_vector(f"__degenerate__{text}")

    def _token_vector(self, token: str) -> Vector:
        cached = self._token_cache.get(token)
        if cached is None:
            cached = self._seeded_unit_vector(token)
            self._token_cache[token] = cached
        return cached

    def _seeded_unit_vector(self, key: str) -> Vector:
        digest = hashlib.sha256(f"{self._name}:{self._dim}:{key}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        raw = [rng.gauss(0.0, 1.0) for _ in range(self._dim)]
        return _l2_normalize(raw) or [1.0] + [0.0] * (self._dim - 1)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(unicodedata.normalize("NFC", text).lower())


def _l2_normalize(vector: Vector) -> Vector:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return []
    return [value / norm for value in vector]
