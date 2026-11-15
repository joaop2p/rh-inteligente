"""
Importação inicial do cadastro mestre para o Neo4j.

Carrega somente:
- Organizacao      <- organizacoes.csv
- Departamento    <- departamentos.csv
- Cargo           <- cargos.csv
- Pessoa          <- funcionarios.csv

Por segurança, a execução padrão é apenas uma validação (dry-run).
Use --apply para gravar os nós no Neo4j.

Variáveis de ambiente necessárias para --apply:
    NEO4J_URI
    NEO4J_USERNAME
    NEO4J_PASSWORD

Opcional:
    NEO4J_DATABASE (padrão: neo4j)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any



# Consultas fixas: não interpolamos labels ou nomes de propriedades recebidos
# do CSV, evitando montar Cypher a partir de dados externos.
IMPORT_QUERIES = {
    "Organizacao": """
        UNWIND $rows AS row
        MERGE (n:Organizacao {organizacao_id: row.organizacao_id})
        SET n += row.properties
    """,
    "Departamento": """
        UNWIND $rows AS row
        MERGE (n:Departamento {departamento_id: row.departamento_id})
        SET n += row.properties
    """,
    "Cargo": """
        UNWIND $rows AS row
        MERGE (n:Cargo {cargo_id: row.cargo_id})
        SET n += row.properties
    """,
    "Pessoa": """
        UNWIND $rows AS row
        MERGE (n:Pessoa {funcionario_id: row.funcionario_id})
        SET n += row.properties
    """,
}

# Propriedades alinhadas ao documento MODELAGEM_LPG_NOVATECH.md.
# CPF, salários, jornada e escolaridade não são importados nesta etapa.
NODE_CONFIG = {
    "organizacoes.csv": {
        "label": "Organizacao",
        "id_field": "organizacao_id",
        "required": ("organizacao_id", "nome"),
        "properties": (
            "organizacao_id",
            "nome",
            "tipo_organizacao",
            "relacao_com_novatech",
            "cnpj",
        ),
    },
    "departamentos.csv": {
        "label": "Departamento",
        "id_field": "departamento_id",
        "required": ("departamento_id", "nome"),
        "properties": ("departamento_id", "nome"),
    },
    "cargos.csv": {
        "label": "Cargo",
        "id_field": "cargo_id",
        "required": ("cargo_id", "cargo_canonico"),
        "properties": (
            "cargo_id",
            "cargo_canonico",
            "funcao",
            "senioridade",
            "camada_gestao",
            "tipo_posicao",
        ),
    },
    "funcionarios.csv": {
        "label": "Pessoa",
        "id_field": "funcionario_id",
        "required": ("funcionario_id", "nome", "categoria_pessoa"),
        "properties": (
            "funcionario_id",
            "nome",
            "email",
            "categoria_pessoa",
            "tipo_vinculo",
            "status",
            "data_admissao",
            "data_desligamento",
            "departamento_id",
            "cargo_atual_id",
            "remuneracao_tipo",
        ),
    },
}


def clean_value(value: str | None) -> str | None:
    """Remove espaços e transforma strings vazias em None."""
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def read_csv(path: Path) -> list[dict[str, str | None]]:
    if not path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        if not reader.fieldnames:
            raise ValueError(f"CSV sem cabeçalho: {path.name}")

        return [
            {key: clean_value(value) for key, value in row.items() if key}
            for row in reader
        ]


def parse_optional_date(value: str | None, field: str, row_id: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"Data inválida em {field} do registro {row_id}: {value!r}. "
            "Use o formato AAAA-MM-DD."
        ) from exc


def load_dataset(data_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Lê os quatro CSVs, valida campos/IDs/referências e prepara propriedades."""
    loaded: dict[str, list[dict[str, Any]]] = {}

    for filename, config in NODE_CONFIG.items():
        rows = read_csv(data_dir / filename)
        id_field = config["id_field"]
        required = config["required"]
        seen_ids: set[str] = set()
        prepared: list[dict[str, Any]] = []

        for line_number, row in enumerate(rows, start=2):
            for field in required:
                if not row.get(field):
                    raise ValueError(
                        f"{filename}, linha {line_number}: campo obrigatório "
                        f"{field!r} está vazio."
                    )

            record_id = row[id_field]
            assert record_id is not None

            if record_id in seen_ids:
                raise ValueError(
                    f"{filename}: ID duplicado {id_field}={record_id!r}."
                )
            seen_ids.add(record_id)

            properties: dict[str, Any] = {
                field: row.get(field) for field in config["properties"]
            }

            if config["label"] == "Pessoa":
                properties["data_admissao"] = parse_optional_date(
                    row.get("data_admissao"), "data_admissao", record_id
                )
                properties["data_desligamento"] = parse_optional_date(
                    row.get("data_desligamento"), "data_desligamento", record_id
                )

            prepared.append(
                {
                    id_field: record_id,
                    "properties": properties,
                }
            )

        loaded[config["label"]] = prepared

    # Valida as referências cadastrais antes de qualquer gravação.
    department_ids = {
        row["departamento_id"] for row in loaded["Departamento"]
    }
    cargo_ids = {row["cargo_id"] for row in loaded["Cargo"]}

    for person in loaded["Pessoa"]:
        person_id = person["funcionario_id"]
        props = person["properties"]

        department_id = props.get("departamento_id")
        if department_id and department_id not in department_ids:
            raise ValueError(
                f"Pessoa {person_id}: departamento inexistente {department_id!r}."
            )

        # Cargo pode estar vazio para pessoas cuja relação não representa
        # ocupação de cargo. Quando informado, precisa existir no cadastro.
        for field in ("cargo_atual_id",):
            cargo_id = props.get(field)
            if cargo_id and cargo_id not in cargo_ids:
                raise ValueError(
                    f"Pessoa {person_id}: {field} referencia cargo "
                    f"inexistente {cargo_id!r}."
                )

    return loaded


def import_nodes(
    uri: str,
    username: str,
    password: str,
    database: str,
    dataset: dict[str, list[dict[str, Any]]],
) -> None:
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
            # Primeiro cadastros de referência; Pessoa por último.
            for label in ("Organizacao", "Departamento", "Cargo", "Pessoa"):
                rows = dataset[label]
                if not rows:
                    print(f"{label}: nenhum registro para importar.")
                    continue

                result = session.run(IMPORT_QUERIES[label], rows=rows) # type: ignore
                summary = result.consume()
                counters = summary.counters
                print(
                    f"{label}: {len(rows)} registros processados | "
                    f"{counters.nodes_created} nós criados | "
                    f"{counters.properties_set} propriedades definidas."
                )
    finally:
        driver.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Valida/importa os cadastros mestre no Neo4j."
    )
    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Pasta que contém os quatro CSVs do cadastro mestre.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Grava os dados no Neo4j. Sem esta opção, executa somente validação.",
    )
    args = parser.parse_args()

    # python-dotenv é usado apenas para ler o arquivo .env local.
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    data_dir = args.data_dir.expanduser().resolve()

    try:
        dataset = load_dataset(data_dir)
        print("Validação concluída:")
        for label in ("Organizacao", "Departamento", "Cargo", "Pessoa"):
            print(f"  {label}: {len(dataset[label])}")

        total = sum(len(rows) for rows in dataset.values())
        print(f"  Total: {total} nós previstos.")

        if not args.apply:
            print("\nDry-run: nenhum dado foi gravado.")
            print("Se os números estiverem corretos, execute novamente com --apply.")
            return 0

        uri = os.getenv("NEO4J_URI")
        username = os.getenv("NEO4J_USERNAME", "neo4j")
        password = os.getenv("NEO4J_PASSWORD")
        database = os.getenv("NEO4J_DATABASE", "neo4j")

        missing = [
            name
            for name, value in (
                ("NEO4J_URI", uri),
                ("NEO4J_USERNAME", username),
                ("NEO4J_PASSWORD", password),
                ("NEO4J_DATABASE", database),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "Variáveis ausentes no ambiente/.env: " + ", ".join(missing)
            )

        import_nodes(uri, username, password, database, dataset)
        print("\nImportação concluída.")
        return 0

    except Exception as exc:
        print(f"\nERRO: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
