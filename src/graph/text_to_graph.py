"""Extração determinística de menções ligadas a entidades canônicas do LPG."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol


CANONICAL_ENTITIES_QUERY = """
MATCH (e:Pessoa)
RETURN 'Pessoa' AS entity_type, e.funcionario_id AS entity_id, e.nome AS canonical_name
UNION ALL
MATCH (e:Organizacao)
RETURN 'Organizacao' AS entity_type, e.organizacao_id AS entity_id, e.nome AS canonical_name
UNION ALL
MATCH (e:Departamento)
RETURN 'Departamento' AS entity_type, e.departamento_id AS entity_id, e.nome AS canonical_name
UNION ALL
MATCH (e:Cargo)
RETURN 'Cargo' AS entity_type, e.cargo_id AS entity_id, e.cargo_canonico AS canonical_name
"""

CHUNKS_QUERY = """
MATCH (c:Chunk)
RETURN c.chunk_id AS chunk_id, c.doc_id AS doc_id, c.texto AS text
ORDER BY c.doc_id, c.chunk_id
"""

MENTION_WRITE_QUERIES = {
    "Pessoa": """
        UNWIND $rows AS row
        MATCH (c:Chunk {chunk_id: row.chunk_id, doc_id: row.doc_id})
        MATCH (e:Pessoa {funcionario_id: row.entity_id})
        MERGE (c)-[m:MENCIONA {mention_id: row.mention_id}]->(e)
        SET m += row.properties
        RETURN count(m) AS applied
    """,
    "Organizacao": """
        UNWIND $rows AS row
        MATCH (c:Chunk {chunk_id: row.chunk_id, doc_id: row.doc_id})
        MATCH (e:Organizacao {organizacao_id: row.entity_id})
        MERGE (c)-[m:MENCIONA {mention_id: row.mention_id}]->(e)
        SET m += row.properties
        RETURN count(m) AS applied
    """,
    "Departamento": """
        UNWIND $rows AS row
        MATCH (c:Chunk {chunk_id: row.chunk_id, doc_id: row.doc_id})
        MATCH (e:Departamento {departamento_id: row.entity_id})
        MERGE (c)-[m:MENCIONA {mention_id: row.mention_id}]->(e)
        SET m += row.properties
        RETURN count(m) AS applied
    """,
    "Cargo": """
        UNWIND $rows AS row
        MATCH (c:Chunk {chunk_id: row.chunk_id, doc_id: row.doc_id})
        MATCH (e:Cargo {cargo_id: row.entity_id})
        MERGE (c)-[m:MENCIONA {mention_id: row.mention_id}]->(e)
        SET m += row.properties
        RETURN count(m) AS applied
    """,
}


class GraphDatabase(Protocol):
    def execute_read(
        self,
        query: str,
        parameters: dict[str, object] | None = None,
    ) -> list[dict[str, Any]]: ...

    def execute_write(
        self,
        query: str,
        parameters: dict[str, object] | None = None,
    ) -> list[dict[str, Any]]: ...


@dataclass(frozen=True)
class CanonicalEntity:
    entity_type: str
    entity_id: str
    canonical_name: str


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    doc_id: str
    text: str


@dataclass(frozen=True)
class MentionCandidate:
    mention_id: str
    chunk_id: str
    doc_id: str
    entity_type: str
    entity_id: str
    canonical_name: str
    surface_form: str
    start_char: int
    end_char: int

    def as_row(self) -> dict[str, object]:
        return {
            "mention_id": self.mention_id,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "entity_id": self.entity_id,
            "properties": {
                "surface_form": self.surface_form,
                "start_char": self.start_char,
                "end_char": self.end_char,
                "extraction_method": "canonical_name_match",
                "review_status": "AUTO_EXACT",
            },
        }


def _required_string(record: dict[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Registro do grafo sem {field} válido.")
    return value.strip()


def parse_canonical_entities(
    records: list[dict[str, Any]],
) -> list[CanonicalEntity]:
    entities: list[CanonicalEntity] = []
    for record in records:
        entity_type = _required_string(record, "entity_type")
        if entity_type not in MENTION_WRITE_QUERIES:
            raise ValueError(f"Label canônico não permitido: {entity_type}.")
        entities.append(
            CanonicalEntity(
                entity_type=entity_type,
                entity_id=_required_string(record, "entity_id"),
                canonical_name=_required_string(record, "canonical_name"),
            )
        )
    return entities


def parse_chunks(records: list[dict[str, Any]]) -> list[TextChunk]:
    chunks: list[TextChunk] = []
    for record in records:
        chunks.append(
            TextChunk(
                chunk_id=_required_string(record, "chunk_id"),
                doc_id=_required_string(record, "doc_id"),
                text=_required_string(record, "text"),
            )
        )
    return chunks


def extract_mentions(
    chunks: list[TextChunk],
    entities: list[CanonicalEntity],
) -> list[MentionCandidate]:
    """Localiza nomes canônicos completos; não infere entidades nem relações."""
    candidates: list[MentionCandidate] = []
    entities_by_name: dict[str, dict[tuple[str, str], CanonicalEntity]] = {}
    for entity in entities:
        normalized_name = " ".join(entity.canonical_name.casefold().split())
        entities_by_name.setdefault(normalized_name, {})[
            (entity.entity_type, entity.entity_id)
        ] = entity

    for chunk in chunks:
        for same_name_entities in entities_by_name.values():
            if len(same_name_entities) != 1:
                continue
            entity = next(iter(same_name_entities.values()))
            name_parts = entity.canonical_name.split()
            if not name_parts:
                continue
            name_pattern = r"\s+".join(re.escape(part) for part in name_parts)
            pattern = re.compile(
                rf"(?<!\w){name_pattern}(?!\w)",
                flags=re.IGNORECASE | re.UNICODE,
            )
            for match in pattern.finditer(chunk.text):
                mention_id = (
                    f"{chunk.chunk_id}|{entity.entity_type}|{entity.entity_id}|"
                    f"{match.start()}"
                )
                candidates.append(
                    MentionCandidate(
                        mention_id=mention_id,
                        chunk_id=chunk.chunk_id,
                        doc_id=chunk.doc_id,
                        entity_type=entity.entity_type,
                        entity_id=entity.entity_id,
                        canonical_name=entity.canonical_name,
                        surface_form=match.group(),
                        start_char=match.start(),
                        end_char=match.end(),
                    )
                )
    return candidates


class TextToGraphPipeline:
    """Produz candidatos reproduzíveis e grava apenas menções canônicas."""

    def __init__(self, database: GraphDatabase) -> None:
        self._database = database

    def extract(self) -> list[MentionCandidate]:
        entities = parse_canonical_entities(
            self._database.execute_read(CANONICAL_ENTITIES_QUERY)
        )
        chunks = parse_chunks(self._database.execute_read(CHUNKS_QUERY))
        return extract_mentions(chunks, entities)

    def apply(self, candidates: list[MentionCandidate]) -> int:
        grouped: dict[str, list[dict[str, object]]] = {
            entity_type: []
            for entity_type in MENTION_WRITE_QUERIES
        }
        seen: set[str] = set()
        for candidate in candidates:
            if candidate.entity_type not in MENTION_WRITE_QUERIES:
                raise ValueError(
                    f"Label de entidade não permitido: {candidate.entity_type}."
                )
            if candidate.mention_id in seen:
                continue
            seen.add(candidate.mention_id)
            grouped[candidate.entity_type].append(candidate.as_row())

        applied = 0
        for entity_type, rows in grouped.items():
            if rows:
                result = self._database.execute_write(
                    MENTION_WRITE_QUERIES[entity_type],
                    {"rows": rows},
                )
                if (
                    len(result) != 1
                    or not isinstance(result[0].get("applied"), int)
                    or isinstance(result[0].get("applied"), bool)
                ):
                    raise RuntimeError(
                        "A gravação de menções retornou uma contagem inválida."
                    )
                applied += result[0]["applied"]
        return applied


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Extrai menções exatas a entidades canônicas dos chunks Neo4j."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Grava relações MENCIONA; sem essa opção, executa somente dry-run.",
    )
    args = parser.parse_args(argv)

    from src.database.neo4j import Neo4jDatabase

    with Neo4jDatabase.from_env() as database:
        database.verify_connectivity()
        pipeline = TextToGraphPipeline(database)
        candidates = pipeline.extract()
        counts = Counter(candidate.entity_type for candidate in candidates)
        if args.apply:
            applied = pipeline.apply(candidates)
            print(f"Relações MENCIONA gravadas/atualizadas: {applied}")
        else:
            print(f"Dry-run: {len(candidates)} menções canônicas encontradas.")
            for entity_type, count in sorted(counts.items()):
                print(f"{entity_type}: {count}")
            print("Nenhuma alteração foi gravada; use --apply para persistir.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
