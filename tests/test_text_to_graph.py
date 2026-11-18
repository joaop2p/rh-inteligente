from __future__ import annotations

from typing import Any

import pytest

from src.graph.text_to_graph import (
    CANONICAL_ENTITIES_QUERY,
    CHUNKS_QUERY,
    MENTION_WRITE_QUERIES,
    CanonicalEntity,
    TextChunk,
    TextToGraphPipeline,
    extract_mentions,
    parse_canonical_entities,
)


class FakeGraphDatabase:
    def __init__(
        self,
        entities: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
    ) -> None:
        self.entities = entities
        self.chunks = chunks
        self.read_calls: list[str] = []
        self.write_calls: list[tuple[str, dict[str, object]]] = []

    def execute_read(
        self,
        query: str,
        parameters: dict[str, object] | None = None,
    ) -> list[dict[str, Any]]:
        self.read_calls.append(query)
        if query == CANONICAL_ENTITIES_QUERY:
            return self.entities
        if query == CHUNKS_QUERY:
            return self.chunks
        raise AssertionError(f"Consulta de leitura inesperada: {query}")

    def execute_write(
        self,
        query: str,
        parameters: dict[str, object] | None = None,
    ) -> list[dict[str, Any]]:
        if parameters is None:
            raise AssertionError("A consulta de escrita precisa de parâmetros.")
        self.write_calls.append((query, parameters))
        return [{"applied": len(parameters["rows"])}]


def test_extract_mentions_matches_full_names_and_preserves_source_spans() -> None:
    candidates = extract_mentions(
        [TextChunk("C-1", "D-1", "ANA\nDA SILVA consta aqui; Ana da Silvação, não.")],
        [CanonicalEntity("Pessoa", "P-1", "Ana da Silva")],
    )

    assert len(candidates) == 1
    mention = candidates[0]
    assert mention.surface_form == "ANA\nDA SILVA"
    assert mention.start_char == 0
    assert mention.end_char == len("ANA\nDA SILVA")
    assert mention.as_row()["properties"] == {
        "surface_form": "ANA\nDA SILVA",
        "start_char": 0,
        "end_char": len("ANA\nDA SILVA"),
        "extraction_method": "canonical_name_match",
        "review_status": "AUTO_EXACT",
    }


def test_extract_mentions_skips_ambiguous_canonical_names() -> None:
    candidates = extract_mentions(
        [TextChunk("C-1", "D-1", "Ana da Silva consta no documento.")],
        [
            CanonicalEntity("Pessoa", "P-1", "Ana da Silva"),
            CanonicalEntity("Pessoa", "P-2", "ANA DA SILVA"),
        ],
    )

    assert candidates == []


def test_pipeline_dry_run_and_apply_use_allowlisted_parameterized_queries() -> None:
    database = FakeGraphDatabase(
        entities=[
            {
                "entity_type": "Pessoa",
                "entity_id": "P-1",
                "canonical_name": "Ana da Silva",
            }
        ],
        chunks=[
            {
                "chunk_id": "C-1",
                "doc_id": "D-1",
                "text": "Ana da Silva está no documento.",
            }
        ],
    )
    pipeline = TextToGraphPipeline(database)

    candidates = pipeline.extract()

    assert len(candidates) == 1
    assert database.read_calls == [CANONICAL_ENTITIES_QUERY, CHUNKS_QUERY]
    assert database.write_calls == []
    assert pipeline.apply(candidates + candidates) == 1
    assert len(database.write_calls) == 1
    query, parameters = database.write_calls[0]
    assert query == MENTION_WRITE_QUERIES["Pessoa"]
    assert "MATCH (e:Pessoa {funcionario_id: row.entity_id})" in query
    assert "Ana da Silva" not in query
    assert parameters["rows"][0]["entity_id"] == "P-1"


def test_pipeline_returns_zero_when_no_canonical_mention_is_found() -> None:
    database = FakeGraphDatabase(
        entities=[
            {
                "entity_type": "Pessoa",
                "entity_id": "P-1",
                "canonical_name": "Ana da Silva",
            }
        ],
        chunks=[
            {
                "chunk_id": "C-1",
                "doc_id": "D-1",
                "text": "Pessoa não identificada.",
            }
        ],
    )

    pipeline = TextToGraphPipeline(database)
    candidates = pipeline.extract()

    assert candidates == []
    assert pipeline.apply(candidates) == 0
    assert database.write_calls == []


def test_parse_canonical_entities_rejects_unknown_labels() -> None:
    with pytest.raises(ValueError, match="Label canônico não permitido"):
        parse_canonical_entities(
            [
                {
                    "entity_type": "Unknown",
                    "entity_id": "X-1",
                    "canonical_name": "Nome",
                }
            ]
        )
