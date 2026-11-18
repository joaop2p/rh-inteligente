"""KAG: monta um pacote explícito de conhecimento antes da geração."""

from __future__ import annotations

from datetime import date

from src.rag.graphrag_pipeline import (
    GraphRAGPipeline,
    GraphRAGResponse,
    GraphRAGResponseError,
    GraphRAGSource,
)
from src.generation.llm import ChatMessage
from src.rag.pipeline import RAGPipeline, RAGResponse, RAGResponseError
from src.retrieval.graph_search import DocumentEvidence
from src.retrieval.vector_search import RetrievedChunk


KAG_SYSTEM_MESSAGE: ChatMessage = {
    "role": "system",
    "content": (
        "Você é o assistente de conhecimento estruturado da NovaTech. Construa "
        "a resposta exclusivamente a partir do pacote KAG fornecido: fatos, "
        "relações temporais, caminhos do grafo e chunks documentais. Preserve "
        "datas inclusivas, direção REPORTS_TO e a distinção entre CLT, estágio "
        "e prestador PJ. Não use conhecimento geral, histórico da conversa ou "
        "informações não presentes no pacote. Se o pacote não sustentar a "
        "resposta, retorne status 'insufficient'. Retorne somente JSON "
        '{"status":"answered|insufficient","answer":"texto",'
        '"citations":[{"doc_id":"...","chunk_id":"..."}]}.'
    ),
}


class _EvidenceSearch:
    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def search(
        self,
        question: str,
        top_k: int = 5,
        document_scope: str | None = None,
        min_vector_score: float | None = None,
    ) -> list[RetrievedChunk]:
        return self._chunks[:top_k]


class KAGPipeline:
    """Expõe a recuperação híbrida como contexto estruturado para o LLM."""

    def __init__(self, search: object, graph_search: object, llm: object) -> None:
        self._pipeline = GraphRAGPipeline(
            search,
            graph_search,
            llm,
            system_message=KAG_SYSTEM_MESSAGE,
        )
        self._rag_pipeline = RAGPipeline(
            search,
            llm,
            system_message=KAG_SYSTEM_MESSAGE,
        )
        self._graph_search = graph_search
        self._llm = llm

    def answer(
        self,
        question: str,
        *,
        as_of: date,
        top_k: int = 5,
        document_scope: str | None = None,
    ) -> GraphRAGResponse:
        response = self._pipeline.answer(
            question,
            as_of=as_of,
            top_k=top_k,
            document_scope=document_scope,
        )
        if response.status == "insufficient":
            metadata_evidence = (
                self._graph_search.search_documents(
                    question,
                    document_scope=None,
                )
                if hasattr(self._graph_search, "search_documents")
                else ()
            )
            try:
                rag_response = self._rag_pipeline.answer(
                    question,
                    top_k=max(top_k, 12),
                    document_scope=None,
                )
            except RAGResponseError:
                rag_response = RAGResponse(
                    answer=(
                        "Não encontrei evidências documentais suficientes "
                        "para responder a essa pergunta."
                    ),
                    sources=[],
                    status="insufficient",
                )
            sources = [
                GraphRAGSource(
                    doc_id=source.doc_id,
                    chunk_id=source.chunk_id,
                    text=source.text,
                    origins=("vector",),
                    vector_score=source.vector_score,
                    fulltext_score=source.fulltext_score,
                    rank_score=source.rank_score,
                )
                for source in rag_response.sources
            ]
            sources_by_key = {
                (source.doc_id, source.chunk_id): source
                for source in sources
            }
            combined_chunks: list[RetrievedChunk] = [
                RetrievedChunk(
                    chunk_id=source.chunk_id,
                    doc_id=source.doc_id,
                    texto=source.text,
                    score=source.vector_score,
                    rank_score=source.rank_score or 0.0,
                    fulltext_score=source.fulltext_score,
                )
                for source in sources
            ]
            for evidence in metadata_evidence:
                if not isinstance(evidence, DocumentEvidence):
                    continue
                key = (evidence.doc_id, evidence.chunk_id)
                existing = sources_by_key.get(key)
                if existing is not None:
                    origins = (
                        ("graph", "vector")
                        if "vector" in existing.origins
                        else ("graph",)
                    )
                    sources_by_key[key] = GraphRAGSource(
                        doc_id=existing.doc_id,
                        chunk_id=existing.chunk_id,
                        text=existing.text,
                        origins=origins,
                        vector_score=existing.vector_score,
                        fulltext_score=existing.fulltext_score,
                        rank_score=existing.rank_score,
                    )
                    continue
                sources_by_key[key] = GraphRAGSource(
                    doc_id=evidence.doc_id,
                    chunk_id=evidence.chunk_id,
                    text=evidence.text,
                    origins=("graph",),
                    vector_score=None,
                    fulltext_score=None,
                    rank_score=None,
                )
                combined_chunks.insert(
                    0,
                    RetrievedChunk(
                        chunk_id=evidence.chunk_id,
                        doc_id=evidence.doc_id,
                        texto=evidence.text,
                        score=None,
                        rank_score=1.0,
                        fulltext_score=None,
                    ),
                )
            sources = list(sources_by_key.values())
            if metadata_evidence:
                evidence_pipeline = RAGPipeline(
                    _EvidenceSearch(combined_chunks),
                    self._llm,
                    system_message=KAG_SYSTEM_MESSAGE,
                )
                rag_response = evidence_pipeline.answer(
                    question,
                    top_k=max(top_k, 12),
                    document_scope=None,
                )
            if rag_response.status == "answered":
                cited_keys = {
                    (source.doc_id, source.chunk_id)
                    for source in rag_response.sources
                }
                sources = [
                    sources_by_key[key]
                    for key in cited_keys
                    if key in sources_by_key
                ]
                for source in rag_response.sources:
                    key = (source.doc_id, source.chunk_id)
                    if key not in sources_by_key:
                        continue
                    existing = sources_by_key[key]
                    if "graph" not in existing.origins:
                        continue
                    sources_by_key[key] = existing
            elif metadata_evidence:
                relevant = [
                    evidence
                    for evidence in metadata_evidence
                    if evidence.chunk_id
                ]
                if relevant:
                    first = relevant[0]
                    key = (first.doc_id, first.chunk_id)
                    if key in sources_by_key:
                        sources = [sources_by_key[key]]
                    else:
                        sources = [
                            GraphRAGSource(
                                doc_id=first.doc_id,
                                chunk_id=first.chunk_id,
                                text=first.text,
                                origins=("graph",),
                                vector_score=None,
                                fulltext_score=None,
                                rank_score=None,
                            )
                        ]
                    rag_response = RAGResponse(
                        answer=(
                            "Evidência KAG recuperada — "
                            f"{first.doc_id}/{first.chunk_id}: "
                            + first.text
                        ),
                        sources=(),
                        status="answered",
                    )
            response = GraphRAGResponse(
                answer=rag_response.answer,
                sources=sources,
                paths=response.paths,
                status=rag_response.status,
                cypher=response.cypher,
                as_of=as_of,
                knowledge_package={
                    "as_of": as_of.isoformat(),
                    "facts": [],
                    "relationships": [],
                    "document_chunks": [
                        {
                            "doc_id": source.doc_id,
                            "chunk_id": source.chunk_id,
                            "text": source.text,
                            "origins": list(source.origins),
                        }
                        for source in sources
                    ],
                },
            )
        if response.knowledge_package is None:
            raise GraphRAGResponseError(
                "O pacote KAG não foi construído a partir das evidências recuperadas."
            )
        return response


__all__ = ["KAGPipeline"]
