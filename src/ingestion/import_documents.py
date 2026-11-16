#!/usr/bin/env python3
"""Importa metadados de documentos e suas relações no Neo4j.

O conteúdo dos TXT não é carregado nesta etapa. A criação de Chunk e embeddings
fica para o pipeline documental posterior.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any



DOCUMENT_PROPERTIES = (
    "titulo",
    "tipo",
    "categoria",
    "data_documento",
    "escopo",
    "resumo",
    "arquivo",
    "tipo_vinculo_pessoa",
)

# Cada relação tem uma consulta fixa: nenhum label/tipo vindo do CSV é interpolado.
RELATION_QUERIES = {
    "REFERE_SE_A": """
        UNWIND $rows AS row
        MATCH (d:Documento {doc_id: row.doc_id})
        MATCH (p:Pessoa {funcionario_id: row.target_id})
        MERGE (d)-[:REFERE_SE_A]->(p)
    """,
    "ASSINADO_POR": """
        UNWIND $rows AS row
        MATCH (d:Documento {doc_id: row.doc_id})
        MATCH (p:Pessoa {funcionario_id: row.target_id})
        MERGE (d)-[:ASSINADO_POR]->(p)
    """,
    "ENVOLVE": """
        UNWIND $rows AS row
        MATCH (d:Documento {doc_id: row.doc_id})
        MATCH (p:Pessoa {funcionario_id: row.target_id})
        MERGE (d)-[:ENVOLVE]->(p)
    """,
    "EMITIDO_POR": """
        UNWIND $rows AS row
        MATCH (d:Documento {doc_id: row.doc_id})
        MATCH (o:Organizacao {organizacao_id: row.target_id})
        MERGE (d)-[:EMITIDO_POR]->(o)
    """,
    "ENVOLVE_ORGANIZACAO": """
        UNWIND $rows AS row
        MATCH (d:Documento {doc_id: row.doc_id})
        MATCH (o:Organizacao {organizacao_id: row.target_id})
        MERGE (d)-[:ENVOLVE_ORGANIZACAO]->(o)
    """,
    "RELACIONADO_A": """
        UNWIND $rows AS row
        MATCH (d:Documento {doc_id: row.doc_id})
        MATCH (outro:Documento {doc_id: row.target_id})
        MERGE (d)-[:RELACIONADO_A]->(outro)
    """,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"Arquivo CSV não encontrado: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return [
            {key: (value or "").strip() for key, value in row.items() if key is not None}
            for row in csv.DictReader(file)
        ]


def split_ids(value: str) -> list[str]:
    """O dataset usa | para separar IDs múltiplos."""
    return [item.strip() for item in value.split("|") if item.strip()]


def parse_document_date(value: str, doc_id: str) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"{doc_id}: data_documento inválida ({value!r}); esperado AAAA-MM-DD."
        ) from exc


def safe_document_path(data_dir: Path, relative_path: str, doc_id: str) -> Path:
    if not relative_path:
        raise ValueError(f"{doc_id}: campo 'arquivo' vazio.")
    base = data_dir.resolve()
    candidate = (base / relative_path).resolve()
    if candidate != base and base not in candidate.parents:
        raise ValueError(f"{doc_id}: caminho de arquivo sai da pasta do dataset.")
    if not candidate.is_file():
        raise ValueError(f"{doc_id}: TXT não encontrado: {relative_path}")
    return candidate


def build_import(data_dir: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, str]]]]:
    documents = read_csv(data_dir / "documentos_index.csv")
    people = read_csv(data_dir / "funcionarios.csv")
    organizations = read_csv(data_dir / "organizacoes.csv")

    person_ids = {row["funcionario_id"] for row in people if row.get("funcionario_id")}
    organization_ids = {
        row["organizacao_id"] for row in organizations if row.get("organizacao_id")
    }
    document_ids = {row["doc_id"] for row in documents if row.get("doc_id")}

    if len(document_ids) != len(documents):
        raise ValueError("documentos_index.csv contém doc_id vazio ou duplicado.")
    if len(person_ids) != len(people):
        raise ValueError("funcionarios.csv contém funcionario_id vazio ou duplicado.")
    if len(organization_ids) != len(organizations):
        raise ValueError("organizacoes.csv contém organizacao_id vazio ou duplicado.")

    nodes: list[dict[str, Any]] = []
    relationships: dict[str, list[dict[str, str]]] = {
        relation_type: [] for relation_type in RELATION_QUERIES
    }

    for row in documents:
        doc_id = row.get("doc_id", "")
        if not doc_id:
            raise ValueError("Documento encontrado sem doc_id.")
        for field in ("titulo", "tipo", "categoria", "escopo", "arquivo"):
            if not row.get(field):
                raise ValueError(f"{doc_id}: campo obrigatório '{field}' vazio.")

        safe_document_path(data_dir, row["arquivo"], doc_id)
        properties: dict[str, Any] = {
            field: (row.get(field) or None) for field in DOCUMENT_PROPERTIES
        }
        properties["data_documento"] = parse_document_date(
            row.get("data_documento", ""), doc_id
        )
        nodes.append({"doc_id": doc_id, "properties": properties})

        def add_people_relation(relation_type: str, ids: list[str]) -> None:
            for target_id in ids:
                if target_id not in person_ids:
                    raise ValueError(
                        f"{doc_id}: {relation_type} referencia pessoa inexistente: {target_id}"
                    )
                relationships[relation_type].append(
                    {"doc_id": doc_id, "target_id": target_id}
                )

        def add_organization_relation(relation_type: str, ids: list[str]) -> None:
            for target_id in ids:
                if target_id not in organization_ids:
                    raise ValueError(
                        f"{doc_id}: {relation_type} referencia organização inexistente: {target_id}"
                    )
                relationships[relation_type].append(
                    {"doc_id": doc_id, "target_id": target_id}
                )

        if row.get("funcionario_id"):
            add_people_relation("REFERE_SE_A", [row["funcionario_id"]])
        add_people_relation("ASSINADO_POR", split_ids(row.get("signatarios_ids", "")))
        add_people_relation("ENVOLVE", split_ids(row.get("outros_envolvidos_ids", "")))

        issuer_id = row.get("organizacao_emissora_id", "")
        if issuer_id:
            add_organization_relation("EMITIDO_POR", [issuer_id])
        add_organization_relation(
            "ENVOLVE_ORGANIZACAO",
            split_ids(row.get("organizacoes_envolvidas_ids", "")),
        )

        related_ids = split_ids(row.get("docs_relacionados", ""))
        for target_id in related_ids:
            if target_id not in document_ids:
                raise ValueError(
                    f"{doc_id}: RELACIONADO_A referencia documento inexistente: {target_id}"
                )
            if target_id == doc_id:
                raise ValueError(f"{doc_id}: documento não pode relacionar-se a si próprio.")
            relationships["RELACIONADO_A"].append(
                {"doc_id": doc_id, "target_id": target_id}
            )

    return nodes, relationships


def verify_existing_entities(session: Any, people: list[dict[str, str]], organizations: list[dict[str, str]]) -> None:
    expected = {
        "Pessoa": [row["funcionario_id"] for row in people],
        "Organizacao": [row["organizacao_id"] for row in organizations],
    }
    checks = (
        ("Pessoa", "funcionario_id"),
        ("Organizacao", "organizacao_id"),
    )
    for label, key in checks:
        query = (
            f"UNWIND $ids AS id MATCH (n:{label} {{{key}: id}}) "
            "RETURN count(DISTINCT n) AS encontrados"
        )
        result = session.run(query, ids=expected[label]).single()
        found = result["encontrados"] if result else 0
        if found != len(expected[label]):
            raise ValueError(
                f"Neo4j possui {found}/{len(expected[label])} nós {label}. "
                "Importe/valide os nós-base antes dos documentos."
            )


def apply_import(
    uri: str,
    username: str,
    password: str,
    database: str,
    nodes: list[dict[str, Any]],
    relationships: dict[str, list[dict[str, str]]],
    people: list[dict[str, str]],
    organizations: list[dict[str, str]],
) -> None:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            verify_existing_entities(session, people, organizations)
            session.execute_write(
                lambda tx: tx.run(
                    "UNWIND $rows AS row "
                    "MERGE (d:Documento {doc_id: row.doc_id}) "
                    "SET d += row.properties",
                    rows=nodes,
                ).consume()
            )
            for relation_type, rows in relationships.items():
                if not rows:
                    continue
                query = RELATION_QUERIES[relation_type]
                session.execute_write(
                    lambda tx, q=query, batch=rows: tx.run(q, rows=batch).consume()
                )
    finally:
        driver.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida e importa metadados de documentos no Neo4j."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Pasta que contém documentos_index.csv, funcionarios.csv, organizacoes.csv e documentos/.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Grava os nós Documento e seus relacionamentos. Sem esta opção, executa dry-run.",
    )
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    try:
        nodes, relationships = build_import(data_dir)
        print("Validação local concluída:")
        print(f"  Documento: {len(nodes)}")
        for relation_type, rows in relationships.items():
            print(f"  {relation_type}: {len(rows)}")
        print(f"  Total de relacionamentos documentais: {sum(map(len, relationships.values()))}")

        if not args.apply:
            print("\nDry-run: nenhum dado foi gravado no Neo4j.")
            return 0

        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise ValueError("Instale as dependências do projeto: pip install -r requirements.txt") from exc
        load_dotenv()
        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv("NEO4J_DATABASE", "neo4j")
        if not uri or not password:
            raise ValueError("Defina NEO4J_URI e NEO4J_PASSWORD no arquivo .env.")

        people = read_csv(data_dir / "funcionarios.csv")
        organizations = read_csv(data_dir / "organizacoes.csv")
        apply_import(
            uri, username, password, database,
            nodes, relationships, people, organizations,
        )
        print("\nImportação concluída no Neo4j.")
        return 0
    except Exception as exc:  # CLI: apresenta erro operacional sem esconder a causa.
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
