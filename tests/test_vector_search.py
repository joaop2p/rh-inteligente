from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest

from src.retrieval.vector_search import (
    EMBEDDING_DIMENSION,
    FULLTEXT_SEARCH_CLAUSE_QUERY,
    FULLTEXT_SEARCH_QUERY,
    VECTOR_SEARCH_CLAUSE_QUERY,
    VECTOR_INDEX_NAME,
    VECTOR_SEARCH_QUERY,
    RetrievedChunk,
    SearchClauseVectorSearch,
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
    def __init__(
        self,
        vector_records: list[dict[str, Any]],
        fulltext_records: list[dict[str, Any]] | None = None,
    ) -> None:
        self.vector_records = vector_records
        self.fulltext_records = fulltext_records or []
        self.calls: list[tuple[str, dict[str, object]]] = []

    def execute_read(
        self,
        query: str,
        parameters: dict[str, object],
    ) -> list[dict[str, Any]]:
        self.calls.append((query, parameters))
        if query in {VECTOR_SEARCH_QUERY, VECTOR_SEARCH_CLAUSE_QUERY}:
            return self.vector_records
        if query in {FULLTEXT_SEARCH_QUERY, FULLTEXT_SEARCH_CLAUSE_QUERY}:
            return self.fulltext_records
        raise AssertionError(f"Consulta não esperada: {query}")


def test_search_returns_chunks_with_document_provenance_and_score() -> None:
    embedding = [0.0] * EMBEDDING_DIMENSION
    model = FakeEmbeddingModel([embedding])
    database = FakeDatabase(
        [
            {
                "chunk_id": "CHUNK-023",
                "doc_id": "DOC-012",
                "texto": "Condições previstas na política de férias.",
                "vector_score": 0.87,
            }
        ],
        [
            {
                "chunk_id": "CHUNK-023",
                "doc_id": "DOC-012",
                "texto": "Condições previstas na política de férias.",
                "fulltext_score": 2.4,
            }
        ],
    )

    results = VectorSearch(database, model).search("  política de férias  ")

    assert results == [
        RetrievedChunk(
            chunk_id="CHUNK-023",
            doc_id="DOC-012",
            texto="Condições previstas na política de férias.",
            score=0.87,
            rank_score=2 / 61,
            fulltext_score=2.4,
        )
    ]
    assert model.sentences == ["política de férias"]
    assert model.options == {
        "batch_size": 1,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }
    assert [query for query, _ in database.calls] == [
        VECTOR_SEARCH_QUERY,
        FULLTEXT_SEARCH_QUERY,
    ]
    assert database.calls[0][1]["index_name"] == VECTOR_INDEX_NAME
    assert database.calls[0][1]["candidate_count"] == 250
    assert database.calls[0][1]["document_scope"] is None
    assert database.calls[0][1]["embedding"] == embedding
    assert database.calls[1][1]["query"] == '"política" OR "de" OR "férias"'


def test_search_returns_empty_list_when_index_has_no_matches() -> None:
    model = FakeEmbeddingModel([[0.0] * EMBEDDING_DIMENSION])
    database = FakeDatabase([])

    assert VectorSearch(database, model).search("pergunta sem resultados") == []


def test_search_clause_backend_uses_cypher_25_and_preserves_hybrid_results() -> None:
    model = FakeEmbeddingModel([[0.0] * EMBEDDING_DIMENSION])
    database = FakeDatabase(
        [
            {
                "chunk_id": "CHUNK-023",
                "doc_id": "DOC-012",
                "texto": "Condições previstas na política de férias.",
                "vector_score": 0.87,
            }
        ],
        [
            {
                "chunk_id": "CHUNK-023",
                "doc_id": "DOC-012",
                "texto": "Condições previstas na política de férias.",
                "fulltext_score": 2.4,
            }
        ],
    )

    results = SearchClauseVectorSearch(database, model).search(
        "política de férias",
        document_scope="ORGANIZACAO",
    )

    assert results[0].chunk_id == "CHUNK-023"
    assert results[0].score == 0.87
    assert results[0].fulltext_score == 2.4
    assert database.calls[0][0] == VECTOR_SEARCH_CLAUSE_QUERY
    assert database.calls[1][0] == FULLTEXT_SEARCH_CLAUSE_QUERY
    assert "CYPHER 25" in database.calls[0][0]
    assert "CYPHER 25" in database.calls[1][0]
    assert "db.index.vector.queryNodes" not in database.calls[0][0]
    assert "db.index.fulltext.queryNodes" not in database.calls[1][0]
    assert database.calls[0][1]["document_scope"] == "ORGANIZACAO"


@pytest.mark.parametrize(
    ("question", "top_k", "error"),
    [
        ("  ", 5, "pergunta"),
        ("consulta válida", 0, "top_k"),
        ("consulta válida", True, "top_k"),
        ("consulta válida", 5, "document_scope"),
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
        if error == "document_scope":
            VectorSearch(database, model).search(question, top_k, "INVALIDO")
        else:
            VectorSearch(database, model).search(question, top_k)

    assert database.calls == []


def test_search_rejects_embedding_with_wrong_dimension() -> None:
    model = FakeEmbeddingModel([[0.0] * (EMBEDDING_DIMENSION - 1)])
    database = FakeDatabase([])

    with pytest.raises(ValueError, match="dimensões"):
        VectorSearch(database, model).search("consulta válida")

    assert database.calls == []


def test_search_rejects_result_without_provenance() -> None:
    model = FakeEmbeddingModel([[0.0] * EMBEDDING_DIMENSION])
    database = FakeDatabase(
        [{"chunk_id": "CHUNK-023", "texto": "Trecho sem doc_id", "score": 0.8}]
    )

    with pytest.raises(ValueError, match="doc_id"):
        VectorSearch(database, model).search("consulta válida")


def test_search_fuses_vector_and_fulltext_by_rank_without_comparing_scores() -> None:
    model = FakeEmbeddingModel([[0.0] * EMBEDDING_DIMENSION])
    database = FakeDatabase(
        [
            {
                "chunk_id": "VECTOR-ONLY",
                "doc_id": "DOC-001",
                "texto": "Achado vetorial.",
                "vector_score": 0.99,
            },
            {
                "chunk_id": "BOTH",
                "doc_id": "DOC-002",
                "texto": "Achado nas duas buscas.",
                "vector_score": 0.75,
            },
        ],
        [
            {
                "chunk_id": "BOTH",
                "doc_id": "DOC-002",
                "texto": "Achado nas duas buscas.",
                "fulltext_score": 1.2,
            },
            {
                "chunk_id": "LEXICAL-ONLY",
                "doc_id": "DOC-003",
                "texto": "Achado lexical.",
                "fulltext_score": 8.0,
            },
        ],
    )

    results = VectorSearch(database, model).search(
        "consulta da organização",
        top_k=3,
        document_scope="ORGANIZACAO",
    )

    assert [item.chunk_id for item in results] == [
        "BOTH",
        "VECTOR-ONLY",
        "LEXICAL-ONLY",
    ]
    assert results[0].score == 0.75
    assert results[0].fulltext_score == 1.2
    assert results[0].rank_score == 1 / 62 + 1 / 61
    assert results[1].score == 0.99
    assert results[1].fulltext_score is None
    assert results[1].rank_score == 1 / 61
    assert results[2].score is None
    assert results[2].fulltext_score == 8.0
    assert results[2].rank_score == 1 / 62
    assert database.calls[0][1]["document_scope"] == "ORGANIZACAO"
    assert database.calls[1][1]["document_scope"] == "ORGANIZACAO"
