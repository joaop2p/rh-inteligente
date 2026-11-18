from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from src.generation.llm import ChatMessage
from src.rag.pipeline import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    RAGPipeline,
    RAGResponseError,
)
from src.retrieval.vector_search import RetrievedChunk


class FakeSearch:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self.chunks = chunks
        self.arguments: tuple[str, int, str | None] | None = None

    def search(
        self,
        question: str,
        top_k: int = 5,
        document_scope: str | None = None,
    ) -> list[RetrievedChunk]:
        self.arguments = (question, top_k, document_scope)
        return self.chunks


class FakeLLM:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages: Sequence[ChatMessage] | None = None
        self.temperature: float | None = None
        self.response_format: str | None = None

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        format: str | None = None,
    ) -> str:
        self.messages = messages
        self.temperature = temperature
        self.response_format = format
        return self.response


def retrieved_chunk() -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id="DOC-058_chunk_002",
        doc_id="DOC-058",
        texto="Promoções ocorrem no ciclo anual, com aprovação do gestor e do RH.",
        score=0.82,
        rank_score=0.03,
        fulltext_score=1.5,
    )


def test_pipeline_sends_only_current_question_and_evidence_to_llm() -> None:
    search = FakeSearch([retrieved_chunk()])
    llm = FakeLLM(
        json.dumps(
            {
                "status": "answered",
                "answer": "As promoções ocorrem no ciclo anual.",
                "citations": [
                    {"doc_id": "DOC-058", "chunk_id": "DOC-058_chunk_002"}
                ],
            }
        )
    )

    result = RAGPipeline(search, llm).answer(
        "Como ocorrem as promoções?",
        top_k=3,
        document_scope="ORGANIZACAO",
    )

    assert result.status == "answered"
    assert result.answer == "As promoções ocorrem no ciclo anual."
    assert result.sources[0].doc_id == "DOC-058"
    assert result.sources[0].chunk_id == "DOC-058_chunk_002"
    assert result.sources[0].quote == retrieved_chunk().texto
    assert result.sources[0].vector_score == 0.82
    assert result.sources[0].fulltext_score == 1.5
    assert result.sources[0].rank_score == 0.03
    assert search.arguments == (
        "Como ocorrem as promoções?",
        3,
        "ORGANIZACAO",
    )
    assert llm.temperature == 0.0
    assert llm.response_format == "json"
    assert llm.messages is not None
    assert len(llm.messages) == 2
    assert llm.messages[0]["role"] == "system"
    assert "somente sobre RH da NovaTech" in llm.messages[0]["content"]
    assert "não use conhecimento geral" in llm.messages[0]["content"].casefold()
    assert "histórico da conversa" in llm.messages[0]["content"].casefold()
    assert "DOC-058_chunk_002" in llm.messages[1]["content"]


def test_pipeline_returns_insufficient_without_calling_llm_if_search_is_empty() -> None:
    search = FakeSearch([])
    llm = FakeLLM("not called")

    result = RAGPipeline(search, llm).answer("Pergunta sem evidências.")

    assert result.status == "insufficient"
    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.sources == []
    assert llm.messages is None


def test_pipeline_discards_candidates_when_llm_reports_insufficient() -> None:
    search = FakeSearch([retrieved_chunk()])
    llm = FakeLLM(
        json.dumps(
            {"status": "insufficient", "answer": "Sem evidência.", "citations": []}
        )
    )

    result = RAGPipeline(search, llm).answer("Pergunta sem resposta nos trechos.")

    assert result.status == "insufficient"
    assert result.answer == INSUFFICIENT_EVIDENCE_ANSWER
    assert result.sources == []


@pytest.mark.parametrize(
    "response",
    [
        "texto sem JSON",
        json.dumps(
            {
                "status": "answered",
                "answer": "Resposta sem citação.",
                "citations": [],
            }
        ),
        json.dumps(
            {
                "status": "answered",
                "answer": "Citação fabricada.",
                "citations": [
                    {"doc_id": "DOC-999", "chunk_id": "DOC-999_chunk_001"}
                ],
            }
        ),
        json.dumps(
            {
                "status": "insufficient",
                "answer": "Insuficiente.",
                "citations": [{"doc_id": "DOC-058", "chunk_id": "DOC-058_chunk_002"}],
            }
        ),
    ],
)
def test_pipeline_rejects_malformed_or_ungrounded_llm_responses(
    response: str,
) -> None:
    pipeline = RAGPipeline(FakeSearch([retrieved_chunk()]), FakeLLM(response))

    with pytest.raises(RAGResponseError):
        pipeline.answer("Como ocorrem as promoções?")


def test_pipeline_rejects_empty_question_before_search() -> None:
    search = FakeSearch([retrieved_chunk()])
    llm = FakeLLM("{}")

    with pytest.raises(ValueError, match="pergunta"):
        RAGPipeline(search, llm).answer(" ")

    assert search.arguments is None
    assert llm.messages is None
