from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from src.retrieval.vector_search import (
    EMBEDDING_DIMENSION,
    VECTOR_INDEX_NAME,
    VECTOR_SEARCH_QUERY,
    RetrievedChunk,
    VectorSearch,
)


class FakeEmbeddingModel:
    def __init__(self, vectors: Sequence[Sequence[float]]) -> None:
        self.vectors = vectors
        self.sentences: list[str] | None = None
        self.options: dict[str, object] | None = None

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> Sequence[Sequence[float]]:
        self.sentences = sentences
        self.options = {
            "batch_size": batch_size,
            "normalize_embeddings": normalize_embeddings,
            "show_progress_bar": show_progress_bar,
        }
        return self.vectors


class FakeDatabase:
    def __init__(self, records: list[dict[str, Any]]) -> None:
        self.records = records
        self.query: str | None = None
        self.parameters: dict[str, object] | None = None

    def execute_read(
        self,
        query: str,
        parameters: dict[str, object],
    ) -> list[dict[str, Any]]:
        self.query = query
        self.parameters = parameters
        return self.records


def test_search_returns_chunks_with_document_provenance_and_score() -> None:
    embedding = [0.0] * EMBEDDING_DIMENSION
    model = FakeEmbeddingModel([embedding])
    database = FakeDatabase(
        [
            {
                "chunk_id": "CHUNK-023",
                "doc_id": "DOC-012",
                "texto": "Condições previstas na política de férias.",
                "score": 0.87,
            }
        ]
    )

    results = VectorSearch(database, model).search("  política de férias  ")

    assert results == [
        RetrievedChunk(
            chunk_id="CHUNK-023",
            doc_id="DOC-012",
            texto="Condições previstas na política de férias.",
            score=0.87,
        )
    ]
    assert model.sentences == ["política de férias"]
    assert model.options == {
        "batch_size": 1,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }
    assert database.query == VECTOR_SEARCH_QUERY
    assert database.parameters is not None
    assert database.parameters["index_name"] == VECTOR_INDEX_NAME
    assert database.parameters["top_k"] == 5
    assert database.parameters["embedding"] == embedding


def test_search_returns_empty_list_when_index_has_no_matches() -> None:
    model = FakeEmbeddingModel([[0.0] * EMBEDDING_DIMENSION])
    database = FakeDatabase([])

    assert VectorSearch(database, model).search("pergunta sem resultados") == []


@pytest.mark.parametrize(
    ("question", "top_k", "error"),
    [
        ("  ", 5, "pergunta"),
        ("consulta válida", 0, "top_k"),
        ("consulta válida", True, "top_k"),
    ],
)
def test_search_rejects_invalid_input(
    question: str,
    top_k: int,
    error: str,
) -> None:
    model = FakeEmbeddingModel([[0.0] * EMBEDDING_DIMENSION])
    database = FakeDatabase([])

    with pytest.raises(ValueError, match=error):
        VectorSearch(database, model).search(question, top_k)

    assert database.query is None


def test_search_rejects_embedding_with_wrong_dimension() -> None:
    model = FakeEmbeddingModel([[0.0] * (EMBEDDING_DIMENSION - 1)])
    database = FakeDatabase([])

    with pytest.raises(ValueError, match="dimensões"):
        VectorSearch(database, model).search("consulta válida")

    assert database.query is None


def test_search_rejects_result_without_provenance() -> None:
    model = FakeEmbeddingModel([[0.0] * EMBEDDING_DIMENSION])
    database = FakeDatabase(
        [{"chunk_id": "CHUNK-023", "texto": "Trecho sem doc_id", "score": 0.8}]
    )

    with pytest.raises(ValueError, match="doc_id"):
        VectorSearch(database, model).search("consulta válida")
