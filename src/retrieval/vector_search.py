"""Busca de chunks documentais pelo índice vetorial do Neo4j."""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Callable, Protocol


EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIMENSION = 1024
VECTOR_INDEX_NAME = "chunk_embedding"
FULLTEXT_INDEX_NAME = "chunk_fulltext"
DEFAULT_TOP_K = 5
RRF_CONSTANT = 60
MAX_CANDIDATES = 500
DOCUMENT_SCOPES = frozenset({"ORGANIZACAO", "PESSOA"})

VECTOR_SEARCH_QUERY = """
    CALL db.index.vector.queryNodes($index_name, $candidate_count, $embedding)
    YIELD node, score
    MATCH (d:Documento)-[:TEM_TRECHO]->(node)
    WHERE $document_scope IS NULL OR d.escopo = $document_scope
    RETURN node.chunk_id AS chunk_id,
           node.doc_id AS doc_id,
           node.texto AS texto,
           score AS vector_score
    ORDER BY vector_score DESC
"""

FULLTEXT_SEARCH_QUERY = """
    CALL db.index.fulltext.queryNodes($index_name, $query, {limit: $candidate_count})
    YIELD node, score
    MATCH (d:Documento)-[:TEM_TRECHO]->(node)
    WHERE $document_scope IS NULL OR d.escopo = $document_scope
    RETURN node.chunk_id AS chunk_id,
           node.doc_id AS doc_id,
           node.texto AS texto,
           score AS fulltext_score
    ORDER BY fulltext_score DESC
"""

VECTOR_SEARCH_CLAUSE_QUERY = """
CYPHER 25
MATCH (node:Chunk)
SEARCH node IN (
    VECTOR INDEX chunk_embedding
    FOR $embedding
    LIMIT $candidate_count
) SCORE AS vector_score
MATCH (d:Documento)-[:TEM_TRECHO]->(node)
WHERE $document_scope IS NULL OR d.escopo = $document_scope
RETURN node.chunk_id AS chunk_id,
       node.doc_id AS doc_id,
       node.texto AS texto,
       vector_score
ORDER BY vector_score DESC
"""

FULLTEXT_SEARCH_CLAUSE_QUERY = """
CYPHER 25
MATCH (node:Chunk)
SEARCH node IN (
    FULLTEXT INDEX chunk_fulltext
    FOR $query
    LIMIT $candidate_count
) SCORE AS fulltext_score
MATCH (d:Documento)-[:TEM_TRECHO]->(node)
WHERE $document_scope IS NULL OR d.escopo = $document_scope
RETURN node.chunk_id AS chunk_id,
       node.doc_id AS doc_id,
       node.texto AS texto,
       fulltext_score
ORDER BY fulltext_score DESC
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
    """Trecho com proveniência e sinais separados de cada fonte de recuperação."""

    chunk_id: str
    doc_id: str
    texto: str
    score: float | None
    rank_score: float
    fulltext_score: float | None = None


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
    """Combina recuperação vetorial e lexical pelo ranking recíproco."""

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
        document_scope: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retorna chunks fundidos por ranking; escopo limita os documentos elegíveis."""
        return self._search(
            question,
            top_k,
            document_scope,
            VECTOR_SEARCH_QUERY,
            FULLTEXT_SEARCH_QUERY,
        )

    def _search(
        self,
        question: str,
        top_k: int,
        document_scope: str | None,
        vector_query: str,
        fulltext_query_template: str,
    ) -> list[RetrievedChunk]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("A pergunta deve ser um texto não vazio.")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0:
            raise ValueError("top_k deve ser um inteiro maior que zero.")
        if document_scope is not None and document_scope not in DOCUMENT_SCOPES:
            raise ValueError(
                "document_scope deve ser ORGANIZACAO, PESSOA ou None."
            )

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

        candidate_count = min(MAX_CANDIDATES, max(top_k * 50, 250))
        shared_parameters: dict[str, object] = {
            "index_name": VECTOR_INDEX_NAME,
            "candidate_count": candidate_count,
            "embedding": embedding,
            "document_scope": document_scope,
        }
        vector_records = self._database.execute_read(
            vector_query,
            shared_parameters,
        )
        vector_chunks = [self._parse_record(record, "vector_score") for record in vector_records]

        lucene_query = self._to_fulltext_query(question)
        if not lucene_query:
            return self._fuse(vector_chunks, [], top_k)

        fulltext_records = self._database.execute_read(
            fulltext_query_template,
            {
                "index_name": FULLTEXT_INDEX_NAME,
                "candidate_count": candidate_count,
                "query": lucene_query,
                "document_scope": document_scope,
            },
        )
        fulltext_chunks = [
            self._parse_record(record, "fulltext_score")
            for record in fulltext_records
        ]
        return self._fuse(vector_chunks, fulltext_chunks, top_k)
    @staticmethod
    def _to_fulltext_query(question: str) -> str:
        terms = dict.fromkeys(re.findall(r"[\w]+", question, flags=re.UNICODE))
        return " OR ".join(f'"{term}"' for term in terms)

    @staticmethod
    def _parse_record(record: dict[str, Any], score_field: str) -> RetrievedChunk:
        chunk_id = record.get("chunk_id")
        doc_id = record.get("doc_id")
        texto = record.get("texto")
        score = record.get(score_field)

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
            raise ValueError(f"{chunk_id}: score de recuperação não finito.")

        return RetrievedChunk(
            chunk_id=chunk_id,
            doc_id=doc_id,
            texto=texto,
            score=numeric_score if score_field == "vector_score" else None,
            rank_score=0.0,
            fulltext_score=(
                numeric_score if score_field == "fulltext_score" else None
            ),
        )

    @staticmethod
    def _fuse(
        vector_chunks: list[RetrievedChunk],
        fulltext_chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        merged: dict[str, dict[str, Any]] = {}
        for rank, chunk in enumerate(vector_chunks, start=1):
            merged[chunk.chunk_id] = {
                "chunk": chunk,
                "vector_rank": rank,
                "fulltext_rank": None,
                "fulltext_score": None,
            }
        for rank, chunk in enumerate(fulltext_chunks, start=1):
            item = merged.setdefault(
                chunk.chunk_id,
                {
                    "chunk": chunk,
                    "vector_rank": None,
                    "fulltext_rank": None,
                    "fulltext_score": None,
                },
            )
            item["fulltext_rank"] = rank
            item["fulltext_score"] = chunk.fulltext_score

        results: list[RetrievedChunk] = []
        for item in merged.values():
            rank_score = sum(
                1.0 / (RRF_CONSTANT + rank)
                for rank in (item["vector_rank"], item["fulltext_rank"])
                if rank is not None
            )
            chunk = item["chunk"]
            results.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    texto=chunk.texto,
                    score=chunk.score,
                    rank_score=rank_score,
                    fulltext_score=item["fulltext_score"],
                )
            )
        results.sort(
            key=lambda chunk: (
                chunk.rank_score,
                chunk.score if chunk.score is not None else -1.0,
                chunk.chunk_id,
            ),
            reverse=True,
        )
        return results[:top_k]


class SearchClauseVectorSearch(VectorSearch):
    """Busca híbrida usando SEARCH (Cypher 25), sem procedures obsoletas."""

    def search(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        document_scope: str | None = None,
    ) -> list[RetrievedChunk]:
        return self._search(
            question,
            top_k,
            document_scope,
            VECTOR_SEARCH_CLAUSE_QUERY,
            FULLTEXT_SEARCH_CLAUSE_QUERY,
        )
