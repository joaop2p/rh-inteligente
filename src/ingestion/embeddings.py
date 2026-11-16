"""Geração e validação de embeddings para chunks documentais."""

from __future__ import annotations

import math
from typing import Protocol, Sequence


class EmbeddingModel(Protocol):
    """Interface mínima compatível com SentenceTransformer."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> Sequence[Sequence[float]]: ...


def generate_embeddings(
    chunks: list[dict[str, object]],
    model: EmbeddingModel,
    embedding_dimension: int = 1024,
    batch_size: int = 32,
) -> list[dict[str, object]]:
    """Gera vetores normalizados e os associa aos chunks de origem."""
    if embedding_dimension <= 0:
        raise ValueError("embedding_dimension deve ser maior que zero.")
    if batch_size <= 0:
        raise ValueError("batch_size deve ser maior que zero.")
    if not chunks:
        return []

    texts: list[str] = []
    for chunk in chunks:
        text = chunk.get("texto")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Cada chunk deve possuir um texto não vazio.")
        texts.append(text)

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    if len(vectors) != len(chunks):
        raise ValueError("O modelo retornou uma quantidade de vetores diferente da de chunks.")

    chunks_with_embeddings: list[dict[str, object]] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        embedding = [float(value) for value in vector]
        if len(embedding) != embedding_dimension:
            raise ValueError(
                f"{chunk.get('chunk_id')}: embedding possui {len(embedding)} dimensões; "
                f"esperado: {embedding_dimension}."
            )
        if not all(math.isfinite(value) for value in embedding):
            raise ValueError(f"{chunk.get('chunk_id')}: embedding contém valor não finito.")
        chunks_with_embeddings.append({**chunk, "embedding": embedding})

    return chunks_with_embeddings
