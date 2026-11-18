from __future__ import annotations

import json
from datetime import date
from typing import Any, Literal, Sequence

import pytest

from src.generation.llm import ChatMessage
from src.rag.kag_pipeline import KAGPipeline
from src.rag.graphrag_pipeline import (
    GraphRAGPipeline,
    GraphRAGResponseError,
)
from src.retrieval.graph_search import (
    GraphCargoOccupant,
    GraphEvidence,
    GraphPath,
    GraphPerson,
    GraphRelationship,
    GraphSearchResult,
)
from src.retrieval.vector_search import RetrievedChunk
from src.retrieval.graph_search import DocumentEvidence


class FakeSearch:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.calls: list[tuple[str, int, str | None, float | None]] = []

    def search(
        self,
        question: str,
        top_k: int = 5,
        document_scope: str | None = None,
        min_vector_score: float | None = None,
    ) -> list[RetrievedChunk]:
        self.calls.append((question, top_k, document_scope, min_vector_score))
        return self.chunks


class FakeGraphSearch:
    def __init__(
        self,
        people: tuple[GraphPerson, ...],
        result: GraphSearchResult,
        role_people: tuple[GraphCargoOccupant, ...] = (),
        document_evidence: tuple[DocumentEvidence, ...] = (),
    ) -> None:
        self.people = people
        self.result = result
        self.role_people = role_people
        self.people_calls: list[str] = []
        self.role_calls: list[tuple[tuple[str, ...], str, date]] = []
        self.reporting_calls: list[tuple[str, date, str | None, int]] = []
        self.document_evidence = document_evidence
        self.upward_calls: list[str] = []
        self.path_calls: list[tuple[str, str]] = []

    def people_mentioned_in(self, text: str) -> tuple[GraphPerson, ...]:
        self.people_calls.append(text)
        return self.people

    def people_occupying_cargo(
        self,
        cargo_titles: tuple[str, ...],
        *,
        organization_id: str,
        as_of: date,
    ) -> tuple[GraphCargoOccupant, ...]:
        self.role_calls.append((cargo_titles, organization_id, as_of))
        return self.role_people

    def search_documents(
        self,
        text: str,
        *,
        document_scope: str | None = None,
    ) -> tuple[DocumentEvidence, ...]:
        return self.document_evidence

    def reporting_chain(
        self,
        manager_id: str,
        *,
        as_of: date,
        max_paths: int = 100,
        document_scope: str | None = None,
    ) -> GraphSearchResult:
        self.reporting_calls.append((manager_id, as_of, document_scope, max_paths))
        return self.result

    def management_chain(
        self,
        person_id: str,
        *,
        as_of: date,
        document_scope: str | None = None,
    ) -> GraphSearchResult:
        self.upward_calls.append(person_id)
        return self.result

    def shortest_path(
        self,
        start_id: str,
        end_id: str,
        *,
        as_of: date,
        document_scope: str | None = None,
    ) -> GraphSearchResult:
        self.path_calls.append((start_id, end_id))
        return self.result

    def cargo_evidence(
        self,
        person_id: str,
        cargo_titles: tuple[str, ...],
        *,
        as_of: date,
        document_scope: str | None = None,
    ) -> tuple[GraphEvidence, ...]:
        return (
            GraphEvidence(
                doc_id="DOC-CARGO",
                chunk_id="CHUNK-CARGO",
                text="Contrato do CEO.",
                funcionario_id=person_id,
            ),
        )


class FakeLLM:
    def __init__(self, payload: dict[str, Any] | None = None) -> None:
        self.payload = payload or {
            "status": "answered",
            "answer": "Ana Silva tem uma pessoa subordinada.",
            "citations": [{"doc_id": "DOC-1", "chunk_id": "CHUNK-1"}],
        }
        self.messages: Sequence[ChatMessage] | None = None

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        format: Literal["json"] | None = None,
    ) -> str:
        self.messages = messages
        assert temperature == 0.0
        assert format == "json"
        return json.dumps(self.payload)


def _graph_result(
    *,
    paths: tuple[GraphPath, ...] | None = None,
    evidence: tuple[GraphEvidence, ...] | None = None,
) -> GraphSearchResult:
    if paths is None:
        paths = (
            GraphPath(
                people=(
                    GraphPerson("E-002", "Pessoa subordinada"),
                    GraphPerson("E-001", "Ana Silva"),
                ),
                relationships=(
                    GraphRelationship(
                        relacao_id="REL-2-1",
                        source_id="E-002",
                        target_id="E-001",
                        valid_from=date(2024, 1, 1),
                        valid_to=None,
                        source_references=("DOC-HIER-1",),
                    ),
                ),
            ),
        )
    if evidence is None:
        evidence = (
            GraphEvidence(
                doc_id="DOC-1",
                chunk_id="CHUNK-1",
                text="Chunk ligado à pessoa do caminho.",
                funcionario_id="E-002",
            ),
        )
    return GraphSearchResult(
        paths=paths,
        evidence=evidence,
        cypher="MATCH path=(p:Pessoa)-[:REPORTS_TO*1..5]->(g:Pessoa)",
        as_of=date(2025, 1, 1),
    )


def _reporting_path(
    subordinate_id: str,
    subordinate_name: str,
    distance: int,
) -> GraphPath:
    intermediate = tuple(
        GraphPerson(
            funcionario_id=f"{subordinate_id}-M{index}",
            nome=f"Intermediário {subordinate_id}-{index}",
        )
        for index in range(1, distance)
    )
    people = (
        GraphPerson(subordinate_id, subordinate_name),
        *intermediate,
        GraphPerson("E-001", "CEO"),
    )
    relationships = tuple(
        GraphRelationship(
            relacao_id=f"REL-{subordinate_id}-{index}",
            source_id=people[index].funcionario_id,
            target_id=people[index + 1].funcionario_id,
            valid_from=date(2024, 1, 1),
            valid_to=None,
            source_references=(),
        )
        for index in range(distance)
    )
    return GraphPath(people=people, relationships=relationships)


def test_kag_answers_basic_document_questions_when_graph_intent_is_absent() -> None:
    chunk = RetrievedChunk(
        chunk_id="CHUNK-FERIAS",
        doc_id="DOC-FERIAS",
        texto="O colaborador tem direito a férias conforme a política interna.",
        score=0.91,
        rank_score=0.04,
        fulltext_score=2.0,
    )
    search = FakeSearch([chunk])
    graph_search = FakeGraphSearch((), _graph_result(paths=(), evidence=()))
    llm = FakeLLM(
        {
            "status": "answered",
            "answer": "A política prevê férias ao colaborador.",
            "citations": [
                {"doc_id": "DOC-FERIAS", "chunk_id": "CHUNK-FERIAS"}
            ],
        }
    )

    response = KAGPipeline(search, graph_search, llm).answer(
        "Qual é a política de férias?",
        as_of=date(2025, 1, 1),
        top_k=5,
        document_scope="ORGANIZACAO",
    )

    assert response.status == "answered"
    assert response.sources[0].origins == ("vector",)
    assert response.knowledge_package is not None
    assert response.knowledge_package["document_chunks"][0]["chunk_id"] == (
        "CHUNK-FERIAS"
    )
    assert search.calls == [
        ("Qual é a política de férias?", 12, "ORGANIZACAO", None)
    ]


def test_kag_uses_document_metadata_when_vector_search_is_empty() -> None:
    search = FakeSearch([])
    evidence = (
        DocumentEvidence(
            doc_id="DOC-055",
            chunk_id="DOC-055_chunk_001",
            text="Aviso de férias de 15 dias.",
            matched_terms=("ferias", "afastamento"),
        ),
    )
    graph_search = FakeGraphSearch(
        (),
        _graph_result(paths=(), evidence=()),
        document_evidence=evidence,
    )
    llm = FakeLLM(
        {
            "status": "answered",
            "answer": "Há um aviso de férias de 15 dias.",
            "citations": [
                {"doc_id": "DOC-055", "chunk_id": "DOC-055_chunk_001"}
            ],
        }
    )

    response = KAGPipeline(search, graph_search, llm).answer(
        "Qual é a política de férias?",
        as_of=date(2025, 1, 1),
        document_scope="ORGANIZACAO",
    )

    assert response.status == "answered"
    assert response.sources[0].origins == ("graph",)
    assert response.knowledge_package is not None
    assert response.knowledge_package["document_chunks"][0]["doc_id"] == "DOC-055"


def test_kag_prefers_rescission_document_over_contract_for_desligamentos() -> None:
    search = FakeSearch([])
    evidence = (
        DocumentEvidence(
            doc_id="DOC-041",
            chunk_id="DOC-041_chunk_001",
            text=(
                "TERMO DE RESCISÃO – DISPENSA SEM JUSTA CAUSA\n"
                "FUNCIONÁRIO: Lucas Henrique Pinto (ID E012)"
            ),
            matched_terms=("desligamento", "rescisao"),
            title="Termo de Rescisão",
            document_type="Termo de Rescisão (TRCT)",
            category="Rescisão",
        ),
        DocumentEvidence(
            doc_id="DOC-047",
            chunk_id="DOC-047_chunk_001",
            text=(
                "TERMO DE RESCISÃO – JUSTA CAUSA\n"
                "FUNCIONÁRIO: Diego Ramos Falcão (ID E018)"
            ),
            matched_terms=("desligamento", "rescisao"),
            title="Termo de Rescisão",
            document_type="Termo de Rescisão (TRCT)",
            category="Rescisão",
        ),
        DocumentEvidence(
            doc_id="DOC-049",
            chunk_id="DOC-049_chunk_001",
            text=(
                "TERMO DE RESCISÃO – PEDIDO DE DEMISSÃO\n"
                "FUNCIONÁRIO: Natália Cordeiro Peixoto (ID E027)"
            ),
            matched_terms=("desligamento", "demissao"),
            title="Termo de Demissão",
            document_type="Termo de Rescisão (TRCT)",
            category="Rescisão",
        ),
    )
    graph_search = FakeGraphSearch(
        (),
        _graph_result(paths=(), evidence=()),
        document_evidence=evidence,
    )
    llm = FakeLLM(
        {
            "status": "insufficient",
            "answer": "sem evidências",
            "citations": [],
        }
    )

    response = KAGPipeline(search, graph_search, llm).answer(
        "Quais desligamentos foram registrados?",
        as_of=date(2025, 1, 1),
        document_scope="ORGANIZACAO",
    )

    assert response.status == "answered"
    assert response.sources[0].doc_id == "DOC-041"
    assert "Lucas Henrique Pinto" in response.answer
    assert "Diego Ramos Falcão" in response.answer
    assert "Natália Cordeiro Peixoto" in response.answer
    assert "Juliana Barbosa Reis" not in response.answer


def test_graphrag_answers_direct_subordinates_deterministically() -> None:
    graph_result = _graph_result(
        paths=(
            GraphPath(
                people=(
                    GraphPerson("E-006", "Fernando Gomes Bastos"),
                    GraphPerson("E-002", "Marcos Vinícius Tavares"),
                ),
                relationships=(
                    GraphRelationship(
                        relacao_id="REL-006-002",
                        source_id="E-006",
                        target_id="E-002",
                        valid_from=date(2024, 1, 1),
                        valid_to=None,
                        source_references=("DOC-006",),
                    ),
                ),
            ),
            GraphPath(
                people=(
                    GraphPerson("E-007", "Camila Rocha Medeiros"),
                    GraphPerson("E-002", "Marcos Vinícius Tavares"),
                ),
                relationships=(
                    GraphRelationship(
                        relacao_id="REL-007-002",
                        source_id="E-007",
                        target_id="E-002",
                        valid_from=date(2024, 1, 1),
                        valid_to=None,
                        source_references=("DOC-007",),
                    ),
                ),
            ),
        )
    )
    graph_search = FakeGraphSearch(
        (GraphPerson("E-002", "Marcos Vinícius Tavares"),),
        graph_result,
    )
    llm = FakeLLM()

    response = GraphRAGPipeline(FakeSearch([]), graph_search, llm).answer(
        "Quais são os subordinados diretos de Marcos Vinícius Tavares?",
        as_of=date(2025, 1, 1),
        document_scope="ORGANIZACAO",
    )

    assert response.status == "answered"
    assert "Fernando Gomes Bastos" in response.answer
    assert "Camila Rocha Medeiros" in response.answer
    assert llm.messages is None


def test_pipeline_fuses_graph_and_vector_chunks_and_validates_citations() -> None:
    chunk_text = "  Chunk ligado à pessoa do caminho."
    graph_result = _graph_result(
        evidence=(
            GraphEvidence(
                doc_id="DOC-1",
                chunk_id="CHUNK-1",
                text=chunk_text,
                funcionario_id="E-002",
            ),
        )
    )
    vector_chunk = RetrievedChunk(
        chunk_id="CHUNK-1",
        doc_id="DOC-1",
        texto=chunk_text,
        score=0.82,
        rank_score=0.03,
        fulltext_score=1.5,
    )
    search = FakeSearch([vector_chunk])
    graph_search = FakeGraphSearch((GraphPerson("E-001", "Ana Silva"),), graph_result)
    llm = FakeLLM()

    response = GraphRAGPipeline(search, graph_search, llm).answer(
        "Quais são os subordinados de Ana Silva?",
        as_of=date(2025, 1, 1),
        document_scope="ORGANIZACAO",
    )

    assert response.status == "answered"
    assert response.knowledge_package is not None
    assert response.knowledge_package["as_of"] == "2025-01-01"
    assert response.knowledge_package["document_chunks"][0]["chunk_id"] == "CHUNK-1"
    assert response.sources[0].origins == ("graph", "vector")
    assert response.sources[0].vector_score == 0.82
    assert response.paths == graph_result.paths
    assert response.cypher == graph_result.cypher
    assert graph_search.reporting_calls == [
        ("E-001", date(2025, 1, 1), "ORGANIZACAO", 100)
    ]
    assert search.calls == [
        (
            "Quais são os subordinados de Ana Silva?",
            5,
            "ORGANIZACAO",
            None,
        )
    ]
    assert llm.messages is None
    assert "Pessoa subordinada" in response.answer


def test_pipeline_rejects_unsupported_intent_without_search_or_generation() -> None:
    graph_search = FakeGraphSearch(
        (GraphPerson("E-001", "Ana Silva"),),
        _graph_result(),
    )
    search = FakeSearch([])
    llm = FakeLLM()

    response = GraphRAGPipeline(search, graph_search, llm).answer(
        "Quais são os critérios de férias?",
        as_of=date(2025, 1, 1),
    )

    assert response.status == "insufficient"
    assert response.sources == []
    assert graph_search.people_calls == []
    assert graph_search.reporting_calls == []
    assert search.calls == []
    assert llm.messages is None


def test_pipeline_requires_one_canonical_person_mention() -> None:
    graph_search = FakeGraphSearch((), _graph_result())
    search = FakeSearch([])
    llm = FakeLLM()

    response = GraphRAGPipeline(search, graph_search, llm).answer(
        "Quais são os subordinados de alguém?",
        as_of=date(2025, 1, 1),
    )

    assert response.status == "insufficient"
    assert graph_search.people_calls
    assert graph_search.reporting_calls == []
    assert search.calls == []
    assert llm.messages is None


def test_pipeline_resolves_ceo_and_limits_to_three_nearest_subordinates() -> None:
    graph_result = _graph_result()
    graph_search = FakeGraphSearch(
        (),
        graph_result,
        role_people=(
            GraphCargoOccupant(
                GraphPerson("E-001", "Ana Silva"),
                ("CEO",),
            ),
        ),
    )
    search = FakeSearch(
        [
            RetrievedChunk(
                chunk_id="CHUNK-1",
                doc_id="DOC-1",
                texto="Chunk ligado à pessoa do caminho.",
                score=0.82,
                rank_score=0.03,
            )
        ]
    )
    llm = FakeLLM()

    response = GraphRAGPipeline(search, graph_search, llm).answer(
        "Quem é o CEO da empresa e quais são os seus 3 subordinados mais próximos?",
        as_of=date(2025, 1, 1),
    )

    assert response.status == "answered"
    assert graph_search.role_calls == [
        (
            ("CEO", "Chief Executive Officer"),
            "ORG-001",
            date(2025, 1, 1),
        )
    ]
    assert graph_search.reporting_calls == [
        ("E-001", date(2025, 1, 1), None, 3)
    ]
    assert "OCUPA_CARGO" in response.cypher
    assert "REPORTS_TO*1..5" in response.cypher
    assert llm.messages is None
    assert "Ana Silva" in response.answer


def test_pipeline_truncates_and_ranks_nearest_three_paths_from_backend() -> None:
    paths = (
        _reporting_path("E-005", "Carlos", 3),
        _reporting_path("E-004", "Zoe", 1),
        _reporting_path("E-003", "Bruno", 2),
        _reporting_path("E-002", "ana", 1),
    )
    evidence = tuple(
        GraphEvidence(
            doc_id=f"DOC-{path.people[0].funcionario_id}",
            chunk_id=f"CHUNK-{path.people[0].funcionario_id}",
            text=f"Evidência de {path.people[0].nome}.",
            funcionario_id=path.people[0].funcionario_id,
        )
        for path in paths
    )
    graph_search = FakeGraphSearch(
        (),
        _graph_result(paths=paths, evidence=evidence),
        role_people=(
            GraphCargoOccupant(
                GraphPerson("E-001", "CEO"),
                ("CEO",),
            ),
        ),
    )
    llm = FakeLLM(
        {
            "status": "answered",
            "answer": "Os três mais próximos.",
            "citations": [{"doc_id": "DOC-E-002", "chunk_id": "CHUNK-E-002"}],
        }
    )

    response = GraphRAGPipeline(FakeSearch([]), graph_search, llm).answer(
        "Quais são os 3 subordinados mais próximos do CEO?",
        as_of=date(2025, 1, 1),
    )

    assert response.status == "answered"
    assert [
        path.people[0].funcionario_id
        for path in response.paths
    ] == ["E-002", "E-004", "E-003"]
    assert llm.messages is None
    assert "ana" in response.answer
    assert "Zoe" in response.answer
    assert "Bruno" in response.answer
    assert {
        source.chunk_id
        for source in response.sources
    } == {"CHUNK-E-002", "CHUNK-E-003", "CHUNK-E-004", "CHUNK-CARGO"}


@pytest.mark.parametrize(
    "role_people",
    [
        (),
        (
            GraphCargoOccupant(GraphPerson("E-001", "Ana Silva"), ("CEO",)),
            GraphCargoOccupant(GraphPerson("E-002", "Bruno Lima"), ("CEO",)),
        ),
    ],
)
def test_pipeline_requires_unique_temporal_ceo_occupant(
    role_people: tuple[GraphCargoOccupant, ...],
) -> None:
    graph_search = FakeGraphSearch((), _graph_result(), role_people=role_people)
    search = FakeSearch([])
    llm = FakeLLM()

    response = GraphRAGPipeline(search, graph_search, llm).answer(
        "Quais são os subordinados do CEO?",
        as_of=date(2025, 1, 1),
    )

    assert response.status == "insufficient"
    assert graph_search.role_calls == [
        (
            ("CEO", "Chief Executive Officer"),
            "ORG-001",
            date(2025, 1, 1),
        )
    ]
    assert graph_search.reporting_calls == []
    assert search.calls == []
    assert llm.messages is None


def test_pipeline_returns_insufficient_when_no_temporal_path_exists() -> None:
    empty_graph = _graph_result(paths=(), evidence=())
    graph_search = FakeGraphSearch((GraphPerson("E-001", "Ana Silva"),), empty_graph)
    search = FakeSearch([])
    llm = FakeLLM()

    response = GraphRAGPipeline(search, graph_search, llm).answer(
        "Quais são os subordinados de Ana Silva?",
        as_of=date(2025, 1, 1),
    )

    assert response.status == "insufficient"
    assert response.paths == ()
    assert response.cypher == empty_graph.cypher
    assert search.calls == []
    assert llm.messages is None


def test_pipeline_does_not_call_llm_without_document_chunks() -> None:
    graph_result = _graph_result(evidence=())
    graph_search = FakeGraphSearch((GraphPerson("E-001", "Ana Silva"),), graph_result)
    search = FakeSearch([])
    llm = FakeLLM()

    response = GraphRAGPipeline(search, graph_search, llm).answer(
        "Quais são os subordinados de Ana Silva?",
        as_of=date(2025, 1, 1),
    )

    assert response.status == "insufficient"
    assert response.paths == graph_result.paths
    assert response.sources == []
    assert llm.messages is None


def test_pipeline_filters_low_vector_scores_before_generation() -> None:
    graph_result = _graph_result(evidence=())
    search = FakeSearch(
        [
            RetrievedChunk(
                chunk_id="CHUNK-LOW",
                doc_id="DOC-LOW",
                texto="Trecho abaixo do limiar.",
                score=0.2,
                rank_score=0.03,
            )
        ]
    )
    llm = FakeLLM()
    pipeline = GraphRAGPipeline(
        search,
        FakeGraphSearch((GraphPerson("E-001", "Ana Silva"),), graph_result),
        llm,
        min_vector_score=0.5,
    )

    response = pipeline.answer(
        "Quais são os subordinados de Ana Silva?",
        as_of=date(2025, 1, 1),
    )

    assert response.status == "insufficient"
    assert search.calls[0][3] == 0.5
    assert response.paths == graph_result.paths
    assert llm.messages is None


def test_pipeline_ignores_untrusted_llm_citation_for_hierarchy() -> None:
    llm = FakeLLM(
        {
            "status": "answered",
            "answer": "Resposta sem suporte.",
            "citations": [{"doc_id": "DOC-X", "chunk_id": "CHUNK-X"}],
        }
    )
    pipeline = GraphRAGPipeline(
        FakeSearch([]),
        FakeGraphSearch(
            (GraphPerson("E-001", "Ana Silva"),),
            _graph_result(),
        ),
        llm,
    )

    response = pipeline.answer(
        "Quais são os subordinados de Ana Silva?",
        as_of=date(2025, 1, 1),
    )

    assert response.status == "answered"
    assert "Pessoa subordinada" in response.answer
    assert llm.messages is None


def test_pipeline_preserves_graph_answer_when_llm_returns_insufficient() -> None:
    graph_result = _graph_result()
    llm = FakeLLM({"status": "insufficient", "citations": []})
    pipeline = GraphRAGPipeline(
        FakeSearch([]),
        FakeGraphSearch(
            (GraphPerson("E-001", "Ana Silva"),),
            graph_result,
        ),
        llm,
    )

    response = pipeline.answer(
        "Quais são os subordinados de Ana Silva?",
        as_of=date(2025, 1, 1),
    )

    assert response.status == "answered"
    assert response.sources
    assert response.paths == graph_result.paths
    assert "Pessoa subordinada" in response.answer


def test_pipeline_rejects_inconsistent_text_for_same_chunk() -> None:
    graph_result = _graph_result()
    search = FakeSearch(
        [
            RetrievedChunk(
                chunk_id="CHUNK-1",
                doc_id="DOC-1",
                texto="Texto divergente.",
                score=0.8,
                rank_score=0.03,
            )
        ]
    )
    pipeline = GraphRAGPipeline(
        search,
        FakeGraphSearch((GraphPerson("E-001", "Ana Silva"),), graph_result),
        FakeLLM(),
    )

    with pytest.raises(GraphRAGResponseError, match="textos diferentes"):
        pipeline.answer(
            "Quais são os subordinados de Ana Silva?",
            as_of=date(2025, 1, 1),
        )


def test_graphrag_answers_direct_manager_with_upward_chain_and_graph_citations() -> None:
    graph_search = FakeGraphSearch(
        (GraphPerson("E-002", "Pessoa subordinada"),),
        _graph_result(),
    )
    llm = FakeLLM()
    search = FakeSearch(
        [RetrievedChunk("OUTRO", "DOC-X", "Texto vetorial irrelevante", 0.9, 0.1, None)]
    )

    response = GraphRAGPipeline(search, graph_search, llm).answer(
        "Quem é o gestor direto de Pessoa subordinada?",
        as_of=date(2025, 1, 1),
    )

    assert response.status == "answered"
    assert "gestor direto de Pessoa subordinada é Ana Silva" in response.answer
    assert graph_search.upward_calls == ["E-002"]
    assert [(s.doc_id, s.chunk_id) for s in response.sources] == [("DOC-1", "CHUNK-1")]
    assert response.paths and llm.messages is None


def test_graphrag_resolves_shortest_path_between_person_and_ceo() -> None:
    graph_search = FakeGraphSearch(
        (GraphPerson("E-002", "Pessoa subordinada"),),
        _graph_result(),
        role_people=(
            GraphCargoOccupant(GraphPerson("E-001", "Ana Silva"), ("CEO",)),
        ),
    )

    response = GraphRAGPipeline(FakeSearch([]), graph_search, FakeLLM()).answer(
        "Qual a cadeia hierárquica entre Pessoa subordinada e o CEO?",
        as_of=date(2025, 1, 1),
    )

    assert response.status == "answered"
    assert graph_search.path_calls == [("E-002", "E-001")]
    assert "1 salto(s)" in response.answer
    assert "Pessoa subordinada reporta a Ana Silva" in response.answer
    assert {s.doc_id for s in response.sources} >= {"DOC-1", "DOC-CARGO"}


def test_graphrag_answers_who_is_ceo_from_cargo_evidence() -> None:
    graph_search = FakeGraphSearch(
        (),
        _graph_result(),
        role_people=(
            GraphCargoOccupant(GraphPerson("E-001", "Ana Silva"), ("CEO",)),
        ),
    )

    response = GraphRAGPipeline(FakeSearch([]), graph_search, FakeLLM()).answer(
        "Quem é o CEO?",
        as_of=date(2025, 1, 1),
    )

    assert response.status == "answered"
    assert "Ana Silva ocupa o cargo de CEO" in response.answer
    assert [s.chunk_id for s in response.sources] == ["CHUNK-CARGO"]
    assert graph_search.reporting_calls == []


def test_graphrag_ceo_with_direct_subordinates_names_role_holder_and_cites_graph() -> None:
    graph_search = FakeGraphSearch(
        (),
        _graph_result(),
        role_people=(
            GraphCargoOccupant(GraphPerson("E-001", "Ana Silva"), ("CEO",)),
        ),
    )
    search = FakeSearch(
        [RetrievedChunk("OUTRO", "DOC-X", "Texto vetorial irrelevante", 0.9, 0.1, None)]
    )

    response = GraphRAGPipeline(search, graph_search, FakeLLM()).answer(
        "Quem é o CEO e quais são seus subordinados diretos?",
        as_of=date(2025, 1, 1),
    )

    assert response.status == "answered"
    assert response.answer.startswith("Ana Silva ocupa o cargo de CEO")
    assert "Pessoa subordinada" in response.answer
    assert {s.doc_id for s in response.sources} == {"DOC-1", "DOC-CARGO"}


def test_graphrag_answers_from_path_when_scope_hides_source_chunks() -> None:
    graph_search = FakeGraphSearch(
        (GraphPerson("E-002", "Pessoa subordinada"),),
        _graph_result(evidence=()),
    )

    response = GraphRAGPipeline(FakeSearch([]), graph_search, FakeLLM()).answer(
        "Quem é o gestor direto de Pessoa subordinada?",
        as_of=date(2025, 1, 1),
        document_scope="ORGANIZACAO",
    )

    assert response.status == "answered"
    assert response.sources == []
    assert response.paths


def test_graphrag_does_not_treat_generic_between_as_path_request() -> None:
    graph_search = FakeGraphSearch(
        (GraphPerson("E-001", "Ana Silva"), GraphPerson("E-002", "Bruno Lima")),
        _graph_result(),
    )

    response = GraphRAGPipeline(FakeSearch([]), graph_search, FakeLLM()).answer(
        "Quem é o gestor de Ana Silva e Bruno Lima entre 2023 e 2024?",
        as_of=date(2025, 1, 1),
    )

    assert graph_search.path_calls == []
    assert response.status == "insufficient"
