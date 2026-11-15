"""Importa os relacionamentos estruturais do cadastro mestre no Neo4j.

Etapas cobertas:
- Pessoa -> Departamento: ALOCADO_EM (fotografia atual)
- Pessoa -> Cargo: OCUPA_CARGO (histórico temporal)
- Pessoa -> Pessoa: REPORTS_TO (histórico temporal)
- Relações funcionais: INTERFACE_TECNICA e SUPERVISIONA_ESTAGIO
- Pessoa -> Organização: EMPREGADO_DE, ESTAGIARIO_DE e
  REPRESENTANTE_DE_CONTRATADA

Por segurança, a execução padrão é dry-run. Use --apply para gravar.

Variáveis de ambiente: NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD.
Opcional: NEO4J_DATABASE (padrão: neo4j).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


DATE_FIELDS = {
    "historico_cargos.csv": ("valid_from", "valid_to"),
    "hierarquia.csv": ("valid_from", "valid_to"),
    "relacoes_funcionais.csv": ("valid_from", "valid_to"),
    "vinculos.csv": ("data_inicio", "data_fim", "data_fim_prevista"),
}

ALLOWED_FUNCTIONAL_TYPES = {"INTERFACE_TECNICA", "SUPERVISIONA_ESTAGIO"}
ALLOWED_EMPLOYMENT_TYPES = {
    "EMPREGADO_DE",
    "ESTAGIARIO_DE",
    "REPRESENTANTE_DE_CONTRATADA",
}

QUERIES = {
    "ALOCADO_EM": """
        UNWIND $rows AS row
        MATCH (p:Pessoa {funcionario_id: row.funcionario_id})
        OPTIONAL MATCH (p)-[old:ALOCADO_EM]->(:Departamento)
        DELETE old
        WITH p, row
        MATCH (d:Departamento {departamento_id: row.departamento_id})
        MERGE (p)-[:ALOCADO_EM]->(d)
    """,
    "OCUPA_CARGO": """
        UNWIND $rows AS row
        MATCH (p:Pessoa {funcionario_id: row.funcionario_id})
        MATCH (c:Cargo {cargo_id: row.cargo_id})
        MERGE (p)-[r:OCUPA_CARGO {historico_id: row.historico_id}]->(c)
        SET r += row.properties
    """,
    "REPORTS_TO": """
        UNWIND $rows AS row
        MATCH (p:Pessoa {funcionario_id: row.subordinado_id})
        MATCH (g:Pessoa {funcionario_id: row.gestor_id})
        MERGE (p)-[r:REPORTS_TO {relacao_id: row.relacao_id}]->(g)
        SET r += row.properties
    """,
    "FUNCTIONAL": """
        UNWIND $rows AS row
        MATCH (origem:Pessoa {funcionario_id: row.origem_id})
        MATCH (destino:Pessoa {funcionario_id: row.destino_id})
        CALL {
            WITH origem, destino, row
            WITH origem, destino, row WHERE row.tipo_relacao = 'INTERFACE_TECNICA'
            MERGE (origem)-[r:INTERFACE_TECNICA {relacao_id: row.relacao_id}]->(destino)
            SET r += row.properties
            RETURN count(*) AS imported
            UNION
            WITH origem, destino, row
            WITH origem, destino, row WHERE row.tipo_relacao = 'SUPERVISIONA_ESTAGIO'
            MERGE (origem)-[r:SUPERVISIONA_ESTAGIO {relacao_id: row.relacao_id}]->(destino)
            SET r += row.properties
            RETURN count(*) AS imported
        }
        RETURN sum(imported) AS imported
    """,
    "EMPLOYMENT": """
        UNWIND $rows AS row
        MATCH (p:Pessoa {funcionario_id: row.funcionario_id})
        MATCH (o:Organizacao {organizacao_id: row.organizacao_id})
        CALL {
            WITH p, o, row
            WITH p, o, row WHERE row.tipo_relacao = 'EMPREGADO_DE'
            MERGE (p)-[r:EMPREGADO_DE {vinculo_id: row.vinculo_id}]->(o)
            SET r += row.properties
            RETURN count(*) AS imported
            UNION
            WITH p, o, row
            WITH p, o, row WHERE row.tipo_relacao = 'ESTAGIARIO_DE'
            MERGE (p)-[r:ESTAGIARIO_DE {vinculo_id: row.vinculo_id}]->(o)
            SET r += row.properties
            RETURN count(*) AS imported
            UNION
            WITH p, o, row
            WITH p, o, row WHERE row.tipo_relacao = 'REPRESENTANTE_DE_CONTRATADA'
            MERGE (p)-[r:REPRESENTANTE_DE_CONTRATADA {vinculo_id: row.vinculo_id}]->(o)
            SET r += row.properties
            RETURN count(*) AS imported
        }
        RETURN sum(imported) AS imported
    """,
}


def clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def read_csv(data_dir: Path, filename: str) -> list[dict[str, str | None]]:
    path = data_dir / filename
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError(f"CSV sem cabeçalho: {filename}")
        return [
            {key: clean(value) for key, value in row.items() if key}
            for row in reader
        ]


def as_date(value: str | None, field: str, record_id: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"Data inválida em {field}, registro {record_id}: {value!r}. "
            "Esperado AAAA-MM-DD."
        ) from exc


def split_ids(value: str | None) -> list[str]:
    return [part.strip() for part in value.split("|") if part.strip()] if value else []


def require(row: dict[str, Any], fields: tuple[str, ...], filename: str, line: int) -> None:
    missing = [field for field in fields if not row.get(field)]
    if missing:
        raise ValueError(
            f"{filename}, linha {line}: campo(s) obrigatório(s) vazio(s): "
            + ", ".join(missing)
        )


def unique_ids(rows: list[dict[str, Any]], field: str, filename: str) -> None:
    seen: set[str] = set()
    for row in rows:
        value = row[field]
        if value in seen:
            raise ValueError(f"{filename}: ID duplicado {field}={value!r}.")
        seen.add(value)


def prepare(data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    people_rows = read_csv(data_dir, "funcionarios.csv")
    department_rows = read_csv(data_dir, "departamentos.csv")
    cargo_rows = read_csv(data_dir, "cargos.csv")
    org_rows = read_csv(data_dir, "organizacoes.csv")

    person_ids = {row["funcionario_id"] for row in people_rows if row.get("funcionario_id")}
    department_ids = {row["departamento_id"] for row in department_rows if row.get("departamento_id")}
    cargo_ids = {row["cargo_id"] for row in cargo_rows if row.get("cargo_id")}
    org_ids = {row["organizacao_id"] for row in org_rows if row.get("organizacao_id")}

    # 1) Lotação atual: não representa histórico de departamento.
    allocated: list[dict[str, Any]] = []
    for line, row in enumerate(people_rows, start=2):
        require(row, ("funcionario_id",), "funcionarios.csv", line)
        department_id = row.get("departamento_id")
        if department_id:
            if row["funcionario_id"] not in person_ids or department_id not in department_ids:
                raise ValueError(f"funcionarios.csv, linha {line}: referência de pessoa/departamento inválida.")
            allocated.append({"funcionario_id": row["funcionario_id"], "departamento_id": department_id})

    # 2) Histórico de cargos.
    history_rows = read_csv(data_dir, "historico_cargos.csv")
    unique_ids(history_rows, "historico_id", "historico_cargos.csv")
    occupies: list[dict[str, Any]] = []
    for line, row in enumerate(history_rows, start=2):
        require(row, ("historico_id", "funcionario_id", "cargo_id", "valid_from"), "historico_cargos.csv", line)
        if row["funcionario_id"] not in person_ids or row["cargo_id"] not in cargo_ids:
            raise ValueError(f"historico_cargos.csv, linha {line}: pessoa ou cargo inexistente.")
        start = as_date(row.get("valid_from"), "valid_from", row["historico_id"])
        end = as_date(row.get("valid_to"), "valid_to", row["historico_id"])
        if end and start and end < start:
            raise ValueError(f"historico_cargos.csv, linha {line}: valid_to anterior a valid_from.")
        occupies.append({
            "historico_id": row["historico_id"],
            "funcionario_id": row["funcionario_id"],
            "cargo_id": row["cargo_id"],
            "properties": {
                "historico_id": row["historico_id"],
                "valid_from": start,
                "valid_to": end,
                "tipo_relacao": row.get("tipo_relacao"),
                "cargo_original_no_periodo": row.get("cargo_original_no_periodo"),
                "fonte_doc_ids": split_ids(row.get("fonte_doc_ids")),
            },
        })

    # 3) Hierarquia temporal.
    hierarchy_rows = read_csv(data_dir, "hierarquia.csv")
    unique_ids(hierarchy_rows, "relacao_id", "hierarquia.csv")
    reports: list[dict[str, Any]] = []
    for line, row in enumerate(hierarchy_rows, start=2):
        require(row, ("relacao_id", "subordinado_id", "gestor_id", "valid_from", "status"), "hierarquia.csv", line)
        if row["subordinado_id"] not in person_ids or row["gestor_id"] not in person_ids:
            raise ValueError(f"hierarquia.csv, linha {line}: subordinado ou gestor inexistente.")
        start = as_date(row.get("valid_from"), "valid_from", row["relacao_id"])
        end = as_date(row.get("valid_to"), "valid_to", row["relacao_id"])
        if end and start and end < start:
            raise ValueError(f"hierarquia.csv, linha {line}: valid_to anterior a valid_from.")
        reports.append({
            "relacao_id": row["relacao_id"],
            "subordinado_id": row["subordinado_id"],
            "gestor_id": row["gestor_id"],
            "properties": {
                "relacao_id": row["relacao_id"],
                "valid_from": start,
                "valid_to": end,
                "status": row.get("status"),
                "valid_from_fonte": row.get("valid_from_fonte"),
                "valid_to_fonte": split_ids(row.get("valid_to_fonte")),
                "valid_to_regra": row.get("valid_to_regra"),
            },
        })

    # 4) Relações funcionais: tipo e direção vêm diretamente do CSV auditado.
    functional_rows = read_csv(data_dir, "relacoes_funcionais.csv")
    unique_ids(functional_rows, "relacao_id", "relacoes_funcionais.csv")
    functional: list[dict[str, Any]] = []
    for line, row in enumerate(functional_rows, start=2):
        require(row, ("relacao_id", "origem_id", "destino_id", "tipo_relacao", "valid_from"), "relacoes_funcionais.csv", line)
        if row["tipo_relacao"] not in ALLOWED_FUNCTIONAL_TYPES:
            raise ValueError(f"relacoes_funcionais.csv, linha {line}: tipo não permitido {row['tipo_relacao']!r}.")
        if row["origem_id"] not in person_ids or row["destino_id"] not in person_ids:
            raise ValueError(f"relacoes_funcionais.csv, linha {line}: pessoa de origem/destino inexistente.")
        start = as_date(row.get("valid_from"), "valid_from", row["relacao_id"])
        end = as_date(row.get("valid_to"), "valid_to", row["relacao_id"])
        functional.append({
            "relacao_id": row["relacao_id"],
            "origem_id": row["origem_id"],
            "destino_id": row["destino_id"],
            "tipo_relacao": row["tipo_relacao"],
            "properties": {
                "relacao_id": row["relacao_id"],
                "valid_from": start,
                "valid_to": end,
                "fonte_doc_ids": split_ids(row.get("fonte_doc_ids")),
            },
        })

    # 5) Vínculos pessoais. Não se cria aqui PRESTA_SERVICOS_PARA:
    # o registro PJ menciona renovação em 2026, mas há lacuna documental;
    # inferir uma relação contínua seria incorreto.
    link_rows = read_csv(data_dir, "vinculos.csv")
    unique_ids(link_rows, "vinculo_id", "vinculos.csv")
    employment: list[dict[str, Any]] = []
    for line, row in enumerate(link_rows, start=2):
        require(row, ("vinculo_id", "funcionario_id", "relacao_neo4j"), "vinculos.csv", line)
        relation_type = row["relacao_neo4j"]
        if relation_type not in ALLOWED_EMPLOYMENT_TYPES:
            raise ValueError(f"vinculos.csv, linha {line}: relação não permitida {relation_type!r}.")
        if row["funcionario_id"] not in person_ids:
            raise ValueError(f"vinculos.csv, linha {line}: pessoa inexistente {row['funcionario_id']!r}.")
        org_field = "organizacao_contratada_id" if relation_type == "REPRESENTANTE_DE_CONTRATADA" else "organizacao_principal_id"
        org_id = row.get(org_field)
        if not org_id or org_id not in org_ids:
            raise ValueError(f"vinculos.csv, linha {line}: organização inválida em {org_field}.")
        start = as_date(row.get("data_inicio"), "data_inicio", row["vinculo_id"])
        end = as_date(row.get("data_fim"), "data_fim", row["vinculo_id"])
        expected_type = {
            "EMPREGADO_DE": "CLT",
            "ESTAGIARIO_DE": "ESTAGIO",
            "REPRESENTANTE_DE_CONTRATADA": "PJ",
        }[relation_type]
        if row.get("tipo_vinculo") != expected_type:
            raise ValueError(
                f"vinculos.csv, linha {line}: {relation_type} incompatível com "
                f"tipo_vinculo={row.get('tipo_vinculo')!r}."
            )
        props = {
            "vinculo_id": row["vinculo_id"],
            "data_inicio": start,
            "data_fim": end,
            "doc_contrato_id": row.get("doc_contrato_id"),
        }
        if relation_type == "ESTAGIARIO_DE":
            props["data_fim_prevista"] = as_date(
                row.get("data_fim_prevista"), "data_fim_prevista", row["vinculo_id"]
            )
        employment.append({
            "vinculo_id": row["vinculo_id"],
            "funcionario_id": row["funcionario_id"],
            "organizacao_id": org_id,
            "tipo_relacao": relation_type,
            "properties": props,
        })

    return {
        "ALOCADO_EM": allocated,
        "OCUPA_CARGO": occupies,
        "REPORTS_TO": reports,
        "FUNCTIONAL": functional,
        "EMPLOYMENT": employment,
    }


def import_relationships(uri: str, username: str, password: str, database: str,
                         dataset: dict[str, list[dict[str, Any]]]) -> None:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(uri, auth=(username, password))
    try:
        driver.verify_connectivity()
        with driver.session(database=database) as session:
            for key in ("ALOCADO_EM", "OCUPA_CARGO", "REPORTS_TO", "FUNCTIONAL", "EMPLOYMENT"):
                rows = dataset[key]
                if not rows:
                    print(f"{key}: 0 relações")
                    continue
                result = session.run(QUERIES[key], rows=rows)
                summary = result.consume()
                print(f"{key}: {len(rows)} relações processadas; {summary.counters.relationships_created} criadas; {summary.counters.properties_set} propriedades definidas.")
    finally:
        driver.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Valida/importa relacionamentos do grafo RH.")
    parser.add_argument("--data-dir", required=True, type=Path,
                        help="Pasta que contém os CSVs saneados usados na carga inicial.")
    parser.add_argument("--apply", action="store_true",
                        help="Grava os relacionamentos no Neo4j; sem isso, apenas valida.")
    args = parser.parse_args()
    load_dotenv()

    try:
        dataset = prepare(args.data_dir.expanduser().resolve())
        print("Validação dos relacionamentos concluída:")
        for key in ("ALOCADO_EM", "OCUPA_CARGO", "REPORTS_TO", "FUNCTIONAL", "EMPLOYMENT"):
            print(f"  {key}: {len(dataset[key])}")
        total = sum(len(rows) for rows in dataset.values())
        print(f"  Total: {total} relações previstas.")
        print("  PRESTA_SERVICOS_PARA: adiada para não inferir continuidade contratual do PJ E026.")

        if not args.apply:
            print("\nDry-run: nenhum dado foi gravado.")
            return 0

        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv("NEO4J_DATABASE", "neo4j")
        missing = [name for name, value in (("NEO4J_URI", uri), ("NEO4J_PASSWORD", password)) if not value]
        if missing:
            raise ValueError("Variáveis ausentes no ambiente/.env: " + ", ".join(missing))
        assert uri is not None and password is not None
        import_relationships(uri, username, password, database, dataset)
        print("\nImportação dos relacionamentos concluída.")
        return 0
    except Exception as exc:
        print(f"\nERRO: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
