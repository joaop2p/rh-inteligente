#!/usr/bin/env python3
"""Gera e importa chunks documentais com embeddings no Neo4j."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Protocol

from src.ingestion.chunking import Tokenizer, build_chunks
from src.ingestion.embeddings import EmbeddingModel, generate_embeddings


CHUNK_IMPORT_QUERY = """
    UNWIND $rows AS row
    MATCH (d:Documento {doc_id: row.doc_id})
    MERGE (c:Chunk {chunk_id: row.chunk_id})
    SET c.texto = row.texto,
        c.ordem = row.ordem,
        c.token_count = row.token_count,
        c.doc_id = row.doc_id,
        c.embedding = row.embedding
    MERGE (d)-[r:TEM_TRECHO]->(c)
    SET r.ordem = row.ordem
"""

DOCUMENTS_QUERY = """
    MATCH (d:Documento)
    WHERE d.doc_id IN $doc_ids
    RETURN collect(d.doc_id) AS doc_ids
"""


class Neo4jSession(Protocol):
    """Operações mínimas da sessão Neo4j usadas pelo importador."""

    def run(self, query: str, **parameters: object) -> object: ...


def validate_chunk_rows(chunks: list[dict[str, object]]) -> None:
    """Valida o contrato dos chunks antes de qualquer escrita no grafo."""
    seen_chunk_ids: set[str] = set()
    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        doc_id = chunk.get("doc_id")
        text = chunk.get("texto")
        order = chunk.get("ordem")
        token_count = chunk.get("token_count")
        embedding = chunk.get("embedding")

        if not isinstance(chunk_id, str) or not chunk_id:
            raise ValueError("Chunk sem chunk_id válido.")
        if chunk_id in seen_chunk_ids:
            raise ValueError(f"chunk_id duplicado: {chunk_id}.")
        seen_chunk_ids.add(chunk_id)
        if not isinstance(doc_id, str) or not doc_id.startswith("DOC-"):
            raise ValueError(f"{chunk_id}: doc_id canônico inválido.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{chunk_id}: texto vazio.")
        if not isinstance(order, int) or order <= 0:
            raise ValueError(f"{chunk_id}: ordem inválida.")
        if not isinstance(token_count, int) or token_count <= 0:
            raise ValueError(f"{chunk_id}: token_count inválido.")
        if not isinstance(embedding, list) or not embedding:
            raise ValueError(f"{chunk_id}: embedding ausente.")


def generate_chunk_rows(
    data_dir: Path,
    tokenizer: Tokenizer,
    embedding_model: EmbeddingModel,
    max_tokens: int = 350,
    overlap_tokens: int = 50,
    embedding_dimension: int = 1024,
    embedding_batch_size: int = 32,
    limit_documents: int | None = None,
) -> list[dict[str, object]]:
    """Gera chunks sanitizados e seus vetores a partir do índice documental."""
    chunks = build_chunks(
        index_path=data_dir / "documentos_index.csv",
        data_dir=data_dir,
        tokenizer=tokenizer,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        limit_documents=limit_documents,
    )
    chunks_with_embeddings = generate_embeddings(
        chunks,
        embedding_model,
        embedding_dimension=embedding_dimension,
        batch_size=embedding_batch_size,
    )
    validate_chunk_rows(chunks_with_embeddings)
    return chunks_with_embeddings


def verify_documents_exist(session: Neo4jSession, doc_ids: set[str]) -> None:
    """Impede chunks órfãos quando os metadados Documento não foram importados."""
    result = session.run(DOCUMENTS_QUERY, doc_ids=sorted(doc_ids))
    record = result.single()  # type: ignore[union-attr]
    found_ids = set(record["doc_ids"]) if record else set()
    missing_ids = sorted(doc_ids - found_ids)
    if missing_ids:
        raise ValueError(
            "Nós Documento ausentes no Neo4j: " + ", ".join(missing_ids) + "."
        )


def import_chunks(
    uri: str,
    username: str,
    password: str,
    database: str,
    chunks: list[dict[str, object]],
    write_batch_size: int = 100,
) -> None:
    """Grava chunks e relacionamentos documentais em transações parametrizadas."""
    if write_batch_size <= 0:
        raise ValueError("write_batch_size deve ser maior que zero.")
    validate_chunk_rows(chunks)

    try:
        from neo4j import GraphDatabase
    except ImportError as exc:
        raise RuntimeError(
            "Dependência neo4j não instalada. Execute: pip install -r requirements.txt"
        ) from exc

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            verify_documents_exist(
                session,
                {str(chunk["doc_id"]) for chunk in chunks},
            )
            for start in range(0, len(chunks), write_batch_size):
                batch = chunks[start : start + write_batch_size]
                session.execute_write(
                    lambda tx, rows=batch: tx.run(CHUNK_IMPORT_QUERY, rows=rows).consume()
                )
    finally:
        driver.close()


def load_models(model_name: str) -> tuple[Tokenizer, EmbeddingModel]:
    """Carrega o tokenizer e o modelo de embeddings compatíveis com o schema."""
    try:
        from sentence_transformers import SentenceTransformer
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "Dependências de embeddings não instaladas. Execute: pip install -r requirements.txt"
        ) from exc
    return AutoTokenizer.from_pretrained(model_name), SentenceTransformer(model_name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gera/importa chunks documentais e embeddings no Neo4j."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Pasta que contém documentos_index.csv e documentos/.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Grava Chunk e TEM_TRECHO no Neo4j; sem esta opção, executa dry-run.",
    )
    parser.add_argument(
        "--model-name",
        default="BAAI/bge-m3",
        help="Modelo SentenceTransformer; deve produzir vetores de 1024 dimensões.",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Quantidade de textos por lote de embeddings.",
    )
    parser.add_argument(
        "--limit-documents",
        type=int,
        help="Processa somente os primeiros N documentos do índice.",
    )
    parser.add_argument(
        "--write-batch-size",
        type=int,
        default=100,
        help="Quantidade de chunks por transação Neo4j.",
    )
    args = parser.parse_args()

    try:
        data_dir = args.data_dir.expanduser().resolve()
        tokenizer, embedding_model = load_models(args.model_name)
        chunks = generate_chunk_rows(
            data_dir,
            tokenizer,
            embedding_model,
            embedding_batch_size=args.embedding_batch_size,
            limit_documents=args.limit_documents,
        )
        print(
            f"Validação local concluída: {len(chunks)} chunks de "
            f"{len({chunk['doc_id'] for chunk in chunks})} documentos."
        )

        if not args.apply:
            print("Dry-run: nenhum dado foi gravado no Neo4j.")
            return 0

        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise RuntimeError(
                "Dependência python-dotenv não instalada. Execute: pip install -r requirements.txt"
            ) from exc
        load_dotenv()
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv("NEO4J_DATABASE", "neo4j")
        if not uri or not password:
            raise ValueError("Defina NEO4J_URI e NEO4J_PASSWORD no arquivo .env.")

        import_chunks(
            uri,
            username,
            password,
            database,
            chunks,
            write_batch_size=args.write_batch_size,
        )
        print("Importação de chunks concluída no Neo4j.")
        return 0
    except Exception as exc:  # CLI: apresenta erro operacional sem ocultar a causa.
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
