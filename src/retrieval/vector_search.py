"""Busca de chunks documentais pelo índice vetorial do Neo4j."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable, Protocol


EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024
VECTOR_INDEX_NAME = "chunk_embedding"
DEFAULT_TOP_K = 5

VECTOR_SEARCH_QUERY = """
    CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
    YIELD node, score
    RETURN node.chunk_id AS chunk_id,
           node.doc_id AS doc_id,
           node.texto AS texto,
           score
    ORDER BY score DESC
"""


class EmbeddingModel(Protocol):
    """Interface mínima para codificar uma pergunta com SentenceTransformer."""

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> object: ...


class ReadDatabase(Protocol):
    """Interface de leitura necessária para consultar o Neo4j."""

    def execute_read(
        self,
        query: str,
        parameters: dict[str, object],
    ) -> list[dict[str, Any]]: ...


class _EmbeddingModelAdapter:
    def __init__(self, encode: Callable[..., object]) -> None:
        self._encode = encode

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> object:
        return self._encode(
            sentences,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )


@dataclass(frozen=True)
class RetrievedChunk:
    """Trecho recuperado com identificadores de origem e score vetorial."""

    chunk_id: str
    doc_id: str
    texto: str
    score: float


def load_embedding_model() -> EmbeddingModel:
    """Carrega o modelo usado na ingestão dos embeddings documentais."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Dependências de embeddings não instaladas. Execute: pip install -r requirements.txt"
        ) from exc

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _EmbeddingModelAdapter(model.encode)


class VectorSearch:
    """Gera embeddings de perguntas e recupera chunks no índice do Neo4j."""

    def __init__(
        self,
        database: ReadDatabase,
        embedding_model: EmbeddingModel,
    ) -> None:
        self._database = database
        self._embedding_model = embedding_model

    def search(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
    ) -> list[RetrievedChunk]:
        """Retorna até ``top_k`` chunks ordenados por similaridade vetorial."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A pergunta deve ser um texto não vazio.")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k deve ser um inteiro maior que zero.")

        encoded = self._embedding_model.encode(
            [question.strip()],
            batch_size=1,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        if not isinstance(encoded, Iterable):
            raise ValueError("O modelo retornou embeddings em formato inválido.")
        vectors = list(encoded)
        if len(vectors) != 1:
            raise ValueError("O modelo deve retornar exatamente um embedding para a pergunta.")

        vector = vectors[0]
        if not isinstance(vector, Iterable):
            raise ValueError("O modelo retornou um embedding em formato inválido.")
        embedding = [float(value) for value in vector]
        if len(embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Embedding da pergunta possui {len(embedding)} dimensões; "
                f"esperado: {EMBEDDING_DIMENSION}."
            )
        if not all(math.isfinite(value) for value in embedding):
            raise ValueError("Embedding da pergunta contém valor não finito.")

        records = self._database.execute_read(
            VECTOR_SEARCH_QUERY,
            {
                "index_name": VECTOR_INDEX_NAME,
                "top_k": top_k,
                "embedding": embedding,
            },
        )
        return [self._parse_record(record) for record in records]

    @staticmethod
    def _parse_record(record: dict[str, Any]) -> RetrievedChunk:
        chunk_id = record.get("chunk_id")
        doc_id = record.get("doc_id")
        texto = record.get("texto")
        score = record.get("score")

        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("Resultado vetorial sem chunk_id válido.")
        if not isinstance(doc_id, str) or not doc_id:
            raise ValueError(f"{chunk_id}: resultado vetorial sem doc_id válido.")
        if not isinstance(texto, str) or not texto.strip():
            raise ValueError(f"{chunk_id}: resultado vetorial sem texto válido.")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"{chunk_id}: score vetorial inválido.")

        numeric_score = float(score)
        if not math.isfinite(numeric_score):
            raise ValueError(f"{chunk_id}: score vetorial não finito.")

        return RetrievedChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            texto=texto,
            score=numeric_score,
        )