from __future__ import annotations

from collections.abc import Mapping
from datetime import date, datetime
from typing import Any

import pytest

from src.retrieval.graph_search import (
    CARGO_OCCUPANTS_QUERY,
    PERSON_BY_EXACT_NAME_QUERY,
    PERSON_BY_ID_QUERY,
    PEOPLE_DIRECTORY_QUERY,
    REPORTING_CHAIN_QUERY,
    GraphEntityAmbiguousError,
    GraphEntityNotFoundError,
    GraphCargoOccupant,
    GraphPerson,
    GraphSearch,
    GraphSearchDataError,
)


class FakeGraphDatabase:
    def __init__(
        self,
        *,
        name_matches: list[dict[str, Any]] | None = None,
        manager_matches: list[dict[str, Any]] | None = None,
        role_matches: list[dict[str, Any]] | None = None,
        people: list[dict[str, Any]] | None = None,
        path_records: list[dict[str, Any]] | None = None,
    ) -> None:
        self.name_matches = name_matches or []
        self.manager_matches = manager_matches or []
        self.role_matches = role_matches or []
        self.people = people or []
        self.path_records = path_records or []
        self.calls: list[tuple[str, Mapping[str, object]]] = []

    def execute_read(
        self,
        query: str,
        parameters: Mapping[str, object] | None = None,
    ) -> list[dict[str, Any]]:
        if parameters is None:
            raise AssertionError("A consulta precisa de parâmetros explícitos.")
        self.calls.append((query, parameters))
        if query == CARGO_OCCUPANTS_QUERY:
            return self.role_matches
        if query == PERSON_BY_EXACT_NAME_QUERY:
            return self.name_matches
        if query == PERSON_BY_ID_QUERY:
            return self.manager_matches
        if query == PEOPLE_DIRECTORY_QUERY:
            return self.people
        if query == REPORTING_CHAIN_QUERY:
            return self.path_records
        raise AssertionError(f"Consulta não permitida: {query}")


def _path_record() -> dict[str, Any]:
    return {
        "path_nodes": [
            {"funcionario_id": "E-015", "nome": "Pessoa subordinada"},
            {"funcionario_id": "E-013", "nome": "Gestora intermediária"},
            {"funcionario_id": "E-007", "nome": "Gestora responsável"},
        ],
        "path_relationships": [
            {
                "relacao_id": "REL-15-13",
                "source_id": "E-015",
                "target_id": "E-013",
                "valid_from": "2024-01-01",
                "valid_to": None,
                "valid_from_fonte": "DOC-015",
                "valid_to_fonte": ["DOC-020"],
            },
            {
                "relacao_id": "REL-13-07",
                "source_id": "E-013",
                "target_id": "E-007",
                "valid_from": date(2023, 5, 1),
                "valid_to": date(2025, 12, 31),
                "valid_from_fonte": None,
                "valid_to_fonte": None,
            },
        ],
        "evidence": [
            {
                "doc_id": "DOC-015",
                "chunk_id": "CHUNK-015-1",
                "text": "Trecho fonte relacionado à pessoa.",
                "funcionario_id": "E-015",
            }
        ],
    }


def test_resolve_person_name_requires_one_exact_canonical_match() -> None:
    database = FakeGraphDatabase(
        name_matches=[
            {"funcionario_id": "E-007", "nome": "Gerente de Tecnologia"}
        ]
    )

    person = GraphSearch(database).resolve_person_name(" Gerente de Tecnologia ")

    assert person is not None
    assert person.funcionario_id == "E-007"
    assert database.calls == [
        (PERSON_BY_EXACT_NAME_QUERY, {"name": "Gerente de Tecnologia"})
    ]


def test_resolve_person_name_reports_ambiguous_names() -> None:
    database = FakeGraphDatabase(
        name_matches=[
            {"funcionario_id": "E-001", "nome": "Ana Silva"},
            {"funcionario_id": "E-002", "nome": "Ana Silva"},
        ]
    )

    with pytest.raises(GraphEntityAmbiguousError, match="ID canônico"):
        GraphSearch(database).resolve_person_name("Ana Silva")


def test_people_mentioned_in_resolves_only_full_canonical_names() -> None:
    database = FakeGraphDatabase(
        people=[
            {"funcionario_id": "E-001", "nome": "Ana da Silva"},
            {"funcionario_id": "E-002", "nome": "João Pereira"},
        ]
    )

    people = GraphSearch(database).people_mentioned_in(
        "ANA da Silva aparece, mas Ana da Silvação e João não."
    )

    assert [person.funcionario_id for person in people] == ["E-001"]
    assert database.calls == [(PEOPLE_DIRECTORY_QUERY, {})]


def test_people_occupying_cargo_uses_exact_titles_and_inclusive_dates() -> None:
    database = FakeGraphDatabase(
        role_matches=[
            {
                "funcionario_id": "E-001",
                "nome": "Ana Silva",
                "cargo_canonico": ["Chief Executive Officer", "CEO"],
            }
        ]
    )

    occupants = GraphSearch(database).people_occupying_cargo(
        ("CEO", "Chief Executive Officer"),
        organization_id="ORG-001",
        as_of=date(2025, 1, 1),
    )

    assert occupants == (
        GraphCargoOccupant(
            person=GraphPerson("E-001", "Ana Silva"),
            cargo_canonico=("CEO", "Chief Executive Officer"),
        ),
    )
    query, parameters = database.calls[0]
    assert query == CARGO_OCCUPANTS_QUERY
    assert "position.valid_from <= date($as_of)" in query
    assert "position.valid_to >= date($as_of)" in query
    assert "employment.data_inicio <= date($as_of)" in query
    assert "employment.data_fim >= date($as_of)" in query
    assert "organization.organizacao_id = $organization_id" in query
    assert parameters == {
        "cargo_titles": ["ceo", "chief executive officer"],
        "organization_id": "ORG-001",
        "as_of": "2025-01-01",
    }


def test_reporting_chain_returns_multihop_temporal_path_and_chunk_provenance() -> None:
    database = FakeGraphDatabase(
        manager_matches=[
            {"funcionario_id": "E-007", "nome": "Gerente de Tecnologia"}
        ],
        path_records=[_path_record()],
    )

    result = GraphSearch(database).reporting_chain(
        "E-007",
        as_of=date(2025, 1, 1),
        max_paths=7,
    )

    assert len(result.paths) == 1
    path = result.paths[0]
    assert [person.funcionario_id for person in path.people] == [
        "E-015",
        "E-013",
        "E-007",
    ]
    assert [relationship.relacao_id for relationship in path.relationships] == [
        "REL-15-13",
        "REL-13-07",
    ]
    assert path.relationships[0].source_references == ("DOC-015", "DOC-020")
    assert path.relationships[1].valid_to == date(2025, 12, 31)
    assert result.evidence[0].doc_id == "DOC-015"
    assert result.evidence[0].chunk_id == "CHUNK-015-1"
    assert result.as_of == date(2025, 1, 1)
    assert "REPORTS_TO*1..5" in result.cypher
    assert "ORDER BY length(path), toLower(subordinate.nome)" in result.cypher
    assert "subordinate.funcionario_id <> manager.funcionario_id" in result.cypher
    assert "path_key" in result.cypher
    assert "relation.valid_from <= date($as_of)" in result.cypher
    assert "relation.valid_to >= date($as_of)" in result.cypher
    assert database.calls[-1] == (
        REPORTING_CHAIN_QUERY,
        {
            "manager_id": "E-007",
            "as_of": "2025-01-01",
            "max_paths": 7,
            "document_scope": None,
        },
    )


def test_reporting_chain_preserves_chunk_text_whitespace() -> None:
    record = _path_record()
    raw_text = "  Trecho fonte com indentação."
    record["evidence"][0]["text"] = raw_text
    database = FakeGraphDatabase(
        manager_matches=[
            {"funcionario_id": "E-007", "nome": "Gerente de Tecnologia"}
        ],
        path_records=[record],
    )

    result = GraphSearch(database).reporting_chain(
        "E-007",
        as_of=date(2025, 1, 1),
    )

    assert result.evidence[0].text == raw_text


def test_reporting_chain_returns_empty_when_no_valid_path_exists() -> None:
    database = FakeGraphDatabase(
        manager_matches=[
            {"funcionario_id": "E-007", "nome": "Gerente de Tecnologia"}
        ]
    )

    result = GraphSearch(database).reporting_chain(
        "E-007",
        as_of=date(2025, 1, 1),
    )

    assert result.paths == ()
    assert result.evidence == ()


def test_reporting_chain_applies_document_scope_to_chunk_evidence() -> None:
    database = FakeGraphDatabase(
        manager_matches=[
            {"funcionario_id": "E-007", "nome": "Gerente de Tecnologia"}
        ]
    )

    GraphSearch(database).reporting_chain(
        "E-007",
        as_of=date(2025, 1, 1),
        document_scope="PESSOA",
    )

    query, parameters = database.calls[-1]
    assert "document.escopo = $document_scope" in query
    assert parameters["document_scope"] == "PESSOA"


def test_reporting_chain_rejects_unknown_manager_before_traversal() -> None:
    database = FakeGraphDatabase()

    with pytest.raises(GraphEntityNotFoundError, match="não foi encontrada"):
        GraphSearch(database).reporting_chain("E-999", as_of=date(2025, 1, 1))

    assert database.calls == [(PERSON_BY_ID_QUERY, {"funcionario_id": "E-999"})]


@pytest.mark.parametrize(
    ("manager_id", "as_of", "max_paths", "message"),
    [
        (" ", date(2025, 1, 1), 10, "manager_id"),
        ("E-007", datetime(2025, 1, 1), 10, "as_of"),
        ("E-007", date(2025, 1, 1), 0, "max_paths"),
        ("E-007", date(2025, 1, 1), True, "max_paths"),
        ("E-007", date(2025, 1, 1), 10, "document_scope"),
    ],
)
def test_reporting_chain_rejects_invalid_inputs(
    manager_id: str,
    as_of: date,
    max_paths: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        GraphSearch(FakeGraphDatabase()).reporting_chain(
            manager_id,
            as_of=as_of,
            max_paths=max_paths,
            document_scope="INVALID" if message == "document_scope" else None,
        )


def test_reporting_chain_rejects_paths_with_inconsistent_edge_direction() -> None:
    invalid_path = _path_record()
    invalid_path["path_relationships"][0]["target_id"] = "E-007"
    database = FakeGraphDatabase(
        manager_matches=[
            {"funcionario_id": "E-007", "nome": "Gerente de Tecnologia"}
        ],
        path_records=[invalid_path],
    )

    with pytest.raises(GraphSearchDataError, match="direção"):
        GraphSearch(database).reporting_chain(
            "E-007",
            as_of=date(2025, 1, 1),
        )
