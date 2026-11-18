"""Teste da orquestração RAG com recuperação Neo4j real e LLM simulado."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence

import pytest

from src.database.neo4j import Neo4jDatabase
from src.generation.llm import ChatMessage
from src.rag.pipeline import RAGPipeline
from src.retrieval.vector_search import SearchClauseVectorSearch, load_embedding_model


pytestmark = pytest.mark.skipif(
    os.getenv("RH_RUN_NEO4J_INTEGRATION") != "1",
    reason="Defina RH_RUN_NEO4J_INTEGRATION=1 para usar Neo4j e BGE-M3 reais.",
)


class CitingLLM:
    def __init__(self) -> None:
        self.messages: Sequence[ChatMessage] | None = None

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        format: str | None = None,
    ) -> str:
        self.messages = messages
        assert temperature == 0.0
        assert format == "json"
        return json.dumps(
            {
                "status": "answered",
                "answer": "As promoções ocorrem no ciclo anual, com aprovação do gestor e do RH.",
                "citations": [
                    {"doc_id": "DOC-058", "chunk_id": "DOC-058_chunk_002"}
                ],
            }
        )


def test_pipeline_uses_live_retrieval_and_returns_cited_source() -> None:
    llm = CitingLLM()
    with Neo4jDatabase.from_env() as database:
        database.verify_connectivity()
        search = SearchClauseVectorSearch(database, load_embedding_model())
        response = RAGPipeline(search, llm).answer(
            "Quais são os critérios de promoção no plano de cargos?",
            document_scope="ORGANIZACAO",
        )

    assert response.status == "answered"
    assert [source.chunk_id for source in response.sources] == [
        "DOC-058_chunk_002"
    ]
    assert "ciclo anual" in response.sources[0].quote
    assert llm.messages is not None
    assert "DOC-058_chunk_002" in llm.messages[1]["content"]
