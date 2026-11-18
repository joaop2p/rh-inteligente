"""KAG: monta um pacote explícito de conhecimento antes da geração."""

from __future__ import annotations

from datetime import date

from src.rag.graphrag_pipeline import (
    GraphRAGPipeline,
    GraphRAGResponse,
    GraphRAGResponseError,
)
from src.generation.llm import ChatMessage


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


class KAGPipeline:
    """Expõe a recuperação híbrida como contexto estruturado para o LLM."""

    def __init__(self, search: object, graph_search: object, llm: object) -> None:
        self._pipeline = GraphRAGPipeline(
            search,
            graph_search,
            llm,
            system_message=KAG_SYSTEM_MESSAGE,
        )

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
        if response.knowledge_package is None:
            raise GraphRAGResponseError(
                "O pacote KAG não foi construído a partir das evidências recuperadas."
            )
        return response


__all__ = ["KAGPipeline"]
