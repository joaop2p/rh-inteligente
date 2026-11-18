"""Consultas GraphRAG seguras sobre relações hierárquicas temporais."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Protocol

from src.retrieval.vector_search import DOCUMENT_SCOPES


MAX_HOPS = 5
MAX_PATHS = 100

PERSON_BY_EXACT_NAME_QUERY = """
MATCH (p:Pessoa)
WHERE toLower(trim(p.nome)) = toLower(trim($name))
RETURN p.funcionario_id AS funcionario_id, p.nome AS nome
LIMIT 2
"""

PERSON_BY_ID_QUERY = """
MATCH (p:Pessoa {funcionario_id: $funcionario_id})
RETURN p.funcionario_id AS funcionario_id, p.nome AS nome
LIMIT 1
"""

PEOPLE_DIRECTORY_QUERY = """
MATCH (p:Pessoa)
RETURN p.funcionario_id AS funcionario_id, p.nome AS nome
ORDER BY p.nome, p.funcionario_id
"""

CARGO_OCCUPANTS_QUERY = """
MATCH (p:Pessoa)-[position:OCUPA_CARGO]->(cargo:Cargo)
MATCH (p)-[employment:EMPREGADO_DE]->(organization:Organizacao)
WHERE toLower(trim(cargo.cargo_canonico)) IN $cargo_titles
  AND organization.organizacao_id = $organization_id
  AND position.valid_from <= date($as_of)
  AND (position.valid_to IS NULL OR position.valid_to >= date($as_of))
  AND employment.data_inicio <= date($as_of)
  AND (employment.data_fim IS NULL OR employment.data_fim >= date($as_of))
WITH p, collect(DISTINCT cargo.cargo_canonico) AS canonical_titles
RETURN
    p.funcionario_id AS funcionario_id,
    p.nome AS nome,
    canonical_titles AS cargo_canonico
ORDER BY funcionario_id
LIMIT 2
"""

REPORTING_CHAIN_QUERY = """
MATCH (manager:Pessoa {funcionario_id: $manager_id})
MATCH path=(subordinate:Pessoa)-[reports:REPORTS_TO*1..__MAX_HOPS__]->(manager)
WHERE all(
    relation IN reports
    WHERE relation.valid_from <= date($as_of)
      AND (relation.valid_to IS NULL OR relation.valid_to >= date($as_of))
)
  AND subordinate.funcionario_id <> manager.funcionario_id
WITH
    subordinate,
    path,
    reduce(
        path_key = '',
        relation IN relationships(path) |
        path_key + coalesce(relation.relacao_id, '') + '|'
    ) AS path_key
ORDER BY subordinate.funcionario_id, length(path), path_key
WITH subordinate, collect(path)[0] AS path
ORDER BY length(path), toLower(subordinate.nome), subordinate.funcionario_id
LIMIT $max_paths
WITH
    path,
    subordinate,
    nodes(path) AS path_nodes,
    relationships(path) AS path_relationships
UNWIND path_nodes AS path_person
OPTIONAL MATCH (document:Documento)-[:TEM_TRECHO]->(chunk:Chunk)
OPTIONAL MATCH
    (document)-[:REFERE_SE_A|ASSINADO_POR|ENVOLVE]->(path_person)
WHERE
    chunk IS NOT NULL
    AND ($document_scope IS NULL OR document.escopo = $document_scope)
WITH
    path,
    path_nodes,
    path_relationships,
    subordinate,
    collect(DISTINCT CASE
        WHEN chunk IS NULL THEN null
        ELSE {
            doc_id: chunk.doc_id,
            chunk_id: chunk.chunk_id,
            text: chunk.texto,
            funcionario_id: path_person.funcionario_id
        }
    END) AS evidence
RETURN
    [person IN path_nodes | {
        funcionario_id: person.funcionario_id,
        nome: person.nome
    }] AS path_nodes,
    [relation IN path_relationships | {
        relacao_id: relation.relacao_id,
        source_id: startNode(relation).funcionario_id,
        target_id: endNode(relation).funcionario_id,
        valid_from: toString(relation.valid_from),
        valid_to: toString(relation.valid_to),
        valid_from_fonte: relation.valid_from_fonte,
        valid_to_fonte: relation.valid_to_fonte
    }] AS path_relationships,
    evidence
ORDER BY length(path), subordinate.nome, subordinate.funcionario_id
""".replace("__MAX_HOPS__", str(MAX_HOPS))

DOCUMENT_METADATA_SEARCH_QUERY = """
MATCH (document:Documento)-[:TEM_TRECHO]->(chunk:Chunk)
WHERE $document_scope IS NULL OR document.escopo = $document_scope
WITH document, chunk,
     toLower(trim(coalesce(document.titulo, '') + ' ' +
                  coalesce(document.tipo, '') + ' ' +
                  coalesce(document.categoria, '') + ' ' +
                  coalesce(document.resumo, '') + ' ' +
                  coalesce(chunk.texto, ''))) AS raw_searchable
WITH document, chunk,
     reduce(
       normalized = raw_searchable,
       pair IN [
         ['á','a'],['à','a'],['ã','a'],['â','a'],
         ['é','e'],['ê','e'],['í','i'],['ó','o'],
         ['õ','o'],['ô','o'],['ú','u'],['ç','c']
       ] |
       replace(normalized, pair[0], pair[1])
     ) AS searchable
WHERE any(term IN $terms WHERE searchable CONTAINS term)
RETURN
    document.doc_id AS doc_id,
    chunk.chunk_id AS chunk_id,
    chunk.texto AS text,
    searchable
LIMIT 30
"""


class GraphDatabase(Protocol):
    def execute_read(
        self,
        query: str,
        parameters: Mapping[str, object] | None = None,
    ) -> list[dict[str, Any]]: ...


class GraphSearchError(RuntimeError):
    """Falha ao resolver ou interpretar evidências do grafo."""


class GraphEntityAmbiguousError(GraphSearchError):
    """O nome corresponde a mais de uma pessoa canônica."""


class GraphEntityNotFoundError(GraphSearchError):
    """A pessoa canônica solicitada não existe no grafo."""


class GraphSearchDataError(GraphSearchError):
    """O Neo4j retornou dados incompatíveis com o contrato esperado."""


@dataclass(frozen=True)
class GraphPerson:
    funcionario_id: str
    nome: str


@dataclass(frozen=True)
class GraphCargoOccupant:
    person: GraphPerson
    cargo_canonico: tuple[str, ...]


@dataclass(frozen=True)
class GraphRelationship:
    relacao_id: str
    source_id: str
    target_id: str
    valid_from: date
    valid_to: date | None
    source_references: tuple[str, ...]


@dataclass(frozen=True)
class GraphPath:
    """Caminho na ordem subordinado -> gestor, conforme a direção do LPG."""

    people: tuple[GraphPerson, ...]
    relationships: tuple[GraphRelationship, ...]


@dataclass(frozen=True)
class GraphEvidence:
    doc_id: str
    chunk_id: str
    text: str
    funcionario_id: str


@dataclass(frozen=True)
class DocumentEvidence:
    doc_id: str
    chunk_id: str
    text: str
    matched_terms: tuple[str, ...]


@dataclass(frozen=True)
class GraphSearchResult:
    paths: tuple[GraphPath, ...]
    evidence: tuple[GraphEvidence, ...]
    cypher: str
    as_of: date


def _required_string(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise GraphSearchDataError(
            f"Resultado do Neo4j sem o campo textual obrigatório {field}."
        )
    return value.strip()


def _required_text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise GraphSearchDataError(
            f"Resultado do Neo4j sem o campo textual obrigatório {field}."
        )
    return value


def _person(record: Mapping[str, Any]) -> GraphPerson:
    return GraphPerson(
        funcionario_id=_required_string(record, "funcionario_id"),
        nome=_required_string(record, "nome"),
    )


def _date_value(value: object, field: str, *, optional: bool = False) -> date | None:
    if value is None and optional:
        return None
    if isinstance(value, datetime):
        raise GraphSearchDataError(f"{field} deve ser uma data, não datetime.")
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError as exc:
            raise GraphSearchDataError(f"Data inválida em {field}.") from exc
    raise GraphSearchDataError(f"Data ausente ou inválida em {field}.")


def _source_references(record: Mapping[str, Any]) -> tuple[str, ...]:
    references: list[str] = []
    for field in ("valid_from_fonte", "valid_to_fonte"):
        value = record.get(field)
        if value is None:
            continue
        values = value if isinstance(value, (list, tuple)) else [value]
        for item in values:
            if not isinstance(item, str):
                raise GraphSearchDataError(
                    f"Referência de origem inválida no campo {field}."
                )
            if item.strip():
                references.append(item.strip())
    return tuple(dict.fromkeys(references))


def _parse_relationship(record: Mapping[str, Any]) -> GraphRelationship:
    valid_from = _date_value(record.get("valid_from"), "valid_from")
    valid_to = _date_value(record.get("valid_to"), "valid_to", optional=True)
    if valid_from is None:
        raise GraphSearchDataError("A relação hierárquica não tem valid_from.")
    if valid_to is not None and valid_to < valid_from:
        raise GraphSearchDataError("A relação hierárquica tem período inválido.")
    return GraphRelationship(
        relacao_id=_required_string(record, "relacao_id"),
        source_id=_required_string(record, "source_id"),
        target_id=_required_string(record, "target_id"),
        valid_from=valid_from,
        valid_to=valid_to,
        source_references=_source_references(record),
    )


def _parse_path(record: Mapping[str, Any]) -> GraphPath:
    raw_people = record.get("path_nodes")
    raw_relationships = record.get("path_relationships")
    if not isinstance(raw_people, list) or not isinstance(raw_relationships, list):
        raise GraphSearchDataError("O caminho retornado tem formato inválido.")
    people = tuple(
        _person(person)
        for person in raw_people
        if isinstance(person, Mapping)
    )
    relationships = tuple(
        _parse_relationship(relationship)
        for relationship in raw_relationships
        if isinstance(relationship, Mapping)
    )
    if (
        len(people) != len(raw_people)
        or len(relationships) != len(raw_relationships)
        or len(people) < 2
        or len(relationships) != len(people) - 1
    ):
        raise GraphSearchDataError(
            "O caminho não contém pessoas e relações alinhadas."
        )
    for index, relationship in enumerate(relationships):
        if (
            relationship.source_id != people[index].funcionario_id
            or relationship.target_id != people[index + 1].funcionario_id
        ):
            raise GraphSearchDataError(
                "A direção de uma relação não corresponde ao caminho retornado."
            )
    return GraphPath(people=people, relationships=relationships)


def _parse_evidence(records: object) -> tuple[GraphEvidence, ...]:
    if not isinstance(records, list):
        raise GraphSearchDataError("A lista de chunks de evidência é inválida.")
    evidence: dict[tuple[str, str, str], GraphEvidence] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise GraphSearchDataError("Um chunk de evidência tem formato inválido.")
        if record.get("doc_id") is None or record.get("chunk_id") is None:
            continue
        item = GraphEvidence(
            doc_id=_required_string(record, "doc_id"),
            chunk_id=_required_string(record, "chunk_id"),
            text=_required_text(record, "text"),
            funcionario_id=_required_string(record, "funcionario_id"),
        )
        evidence[(item.doc_id, item.chunk_id, item.funcionario_id)] = item
    return tuple(
        evidence[key]
        for key in sorted(evidence)
    )


def _normalize_search_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.casefold())
    return "".join(
        character
        for character in normalized
        if not unicodedata.combining(character)
    )


_DOCUMENT_TOPIC_ALIASES = {
    "ferias": ("ferias", "férias", "afastamento"),
    "ferias vencidas": ("ferias", "vencidas"),
    "promocao": ("promocao", "promocoes", "promoção", "promoções"),
    "desligamento": (
        "desligamento",
        "rescisao",
        "demissao",
        "dispensa",
        "trct",
    ),
    "beneficios": ("beneficios", "benefício", "beneficios", "plano"),
    "avaliacao": ("avaliacao", "desempenho", "melhoria"),
}


def _document_search_terms(text: str) -> tuple[str, ...]:
    normalized = _normalize_search_text(text)
    raw_terms = re.findall(r"[a-z0-9]+", normalized)
    stopwords = {
        "a", "as", "o", "os", "da", "das", "de", "do", "dos",
        "e", "qual", "quais", "como", "sobre", "da", "nova",
        "novatech", "documento", "documentos", "fala", "falam",
        "empresa", "politica", "processo",
    }
    terms = {
        term
        for term in raw_terms
        if len(term) >= 3 and term not in stopwords
    }
    for key, aliases in _DOCUMENT_TOPIC_ALIASES.items():
        if key in normalized:
            terms.update(_normalize_search_text(alias) for alias in aliases)
    for alias_group in _DOCUMENT_TOPIC_ALIASES.values():
        if any(alias in normalized for alias in alias_group):
            terms.update(_normalize_search_text(alias) for alias in alias_group)
    return tuple(sorted(term for term in terms if term))


class GraphSearch:
    """Executa travessias permitidas e retorna caminhos com proveniência."""

    def __init__(self, database: GraphDatabase) -> None:
        self._database = database

    def search_documents(
        self,
        text: str,
        *,
        document_scope: str | None = None,
    ) -> tuple[DocumentEvidence, ...]:
        terms = _document_search_terms(text)
        if not terms:
            return ()
        records = self._database.execute_read(
            DOCUMENT_METADATA_SEARCH_QUERY,
            {
                "terms": terms,
                "document_scope": document_scope,
            },
        )
        evidence: dict[tuple[str, str], DocumentEvidence] = {}
        for record in records:
            doc_id = record.get("doc_id")
            chunk_id = record.get("chunk_id")
            chunk_text = record.get("text")
            searchable = record.get("searchable")
            if not isinstance(doc_id, str) or not isinstance(chunk_id, str):
                continue
            if not isinstance(chunk_text, str) or not chunk_text.strip():
                continue
            searchable_text = searchable if isinstance(searchable, str) else ""
            matches = tuple(
                term for term in terms if term in searchable_text
            )
            evidence[(doc_id, chunk_id)] = DocumentEvidence(
                doc_id=doc_id,
                chunk_id=chunk_id,
                text=chunk_text,
                matched_terms=matches,
            )
        return tuple(
            sorted(
                evidence.values(),
                key=lambda item: (
                    -len(item.matched_terms),
                    item.doc_id,
                    item.chunk_id,
                ),
            )
        )

    def resolve_person_name(self, name: str) -> GraphPerson | None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("O nome da pessoa deve ser um texto não vazio.")
        matches = self._database.execute_read(
            PERSON_BY_EXACT_NAME_QUERY,
            {"name": name.strip()},
        )
        if len(matches) > 1:
            raise GraphEntityAmbiguousError(
                "O nome corresponde a mais de uma pessoa; use o ID canônico."
            )
        return _person(matches[0]) if matches else None

    def people_mentioned_in(self, text: str) -> tuple[GraphPerson, ...]:
        """Resolve apenas nomes canônicos completos presentes literalmente no texto."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("O texto deve ser não vazio.")
        people = self._database.execute_read(PEOPLE_DIRECTORY_QUERY, {})
        matches: dict[str, GraphPerson] = {}
        for record in people:
            person = _person(record)
            name_parts = person.nome.split()
            if not name_parts:
                continue
            name_pattern = r"\s+".join(
                re.escape(part)
                for part in name_parts
            )
            if re.search(
                rf"(?<!\w){name_pattern}(?!\w)",
                text,
                flags=re.IGNORECASE | re.UNICODE,
            ):
                matches[person.funcionario_id] = person
        return tuple(matches[key] for key in sorted(matches))

    def people_occupying_cargo(
        self,
        cargo_titles: tuple[str, ...],
        *,
        organization_id: str,
        as_of: date,
    ) -> tuple[GraphCargoOccupant, ...]:
        if (
            isinstance(cargo_titles, str)
            or not cargo_titles
            or any(
                not isinstance(title, str) or not title.strip()
                for title in cargo_titles
            )
        ):
            raise ValueError("cargo_titles deve conter títulos não vazios.")
        if not isinstance(organization_id, str) or not organization_id.strip():
            raise ValueError("organization_id deve ser um ID canônico não vazio.")
        if not isinstance(as_of, date) or isinstance(as_of, datetime):
            raise ValueError("as_of deve ser datetime.date.")

        normalized_titles = tuple(
            dict.fromkeys(title.strip().casefold() for title in cargo_titles)
        )
        records = self._database.execute_read(
            CARGO_OCCUPANTS_QUERY,
            {
                "cargo_titles": list(normalized_titles),
                "organization_id": organization_id.strip(),
                "as_of": as_of.isoformat(),
            },
        )
        occupants: list[GraphCargoOccupant] = []
        for record in records:
            raw_titles = record.get("cargo_canonico")
            if not isinstance(raw_titles, list) or not raw_titles:
                raise GraphSearchDataError(
                    "O cargo canônico retornado pelo Neo4j é inválido."
                )
            if any(not isinstance(title, str) or not title.strip() for title in raw_titles):
                raise GraphSearchDataError(
                    "O cargo canônico retornado pelo Neo4j é inválido."
                )
            canonical_titles = tuple(
                sorted(
                    {title.strip() for title in raw_titles},
                    key=lambda title: (title.casefold(), title),
                )
            )
            occupants.append(
                GraphCargoOccupant(
                    person=_person(record),
                    cargo_canonico=canonical_titles,
                )
            )
        return tuple(occupants)

    def reporting_chain(
        self,
        manager_id: str,
        *,
        as_of: date,
        max_paths: int = 100,
        document_scope: str | None = None,
    ) -> GraphSearchResult:
        if not isinstance(manager_id, str) or not manager_id.strip():
            raise ValueError("manager_id deve ser um ID canônico não vazio.")
        if not isinstance(as_of, date) or isinstance(as_of, datetime):
            raise ValueError("as_of deve ser datetime.date.")
        if (
            isinstance(max_paths, bool)
            or not isinstance(max_paths, int)
            or not 1 <= max_paths <= MAX_PATHS
        ):
            raise ValueError(f"max_paths deve estar entre 1 e {MAX_PATHS}.")
        if document_scope is not None and document_scope not in DOCUMENT_SCOPES:
            raise ValueError("document_scope deve ser ORGANIZACAO, PESSOA ou None.")

        normalized_manager_id = manager_id.strip()
        manager = self._database.execute_read(
            PERSON_BY_ID_QUERY,
            {"funcionario_id": normalized_manager_id},
        )
        if not manager:
            raise GraphEntityNotFoundError(
                "A pessoa gerente não foi encontrada pelo ID canônico."
            )
        _person(manager[0])

        records = self._database.execute_read(
            REPORTING_CHAIN_QUERY,
            {
                "manager_id": normalized_manager_id,
                "as_of": as_of.isoformat(),
                "max_paths": max_paths,
                "document_scope": document_scope,
            },
        )
        paths = tuple(_parse_path(record) for record in records)
        evidence_by_key: dict[tuple[str, str, str], GraphEvidence] = {}
        for record in records:
            for item in _parse_evidence(record.get("evidence")):
                evidence_by_key[
                    (item.doc_id, item.chunk_id, item.funcionario_id)
                ] = item
        evidence = tuple(
            evidence_by_key[key]
            for key in sorted(evidence_by_key)
        )
        return GraphSearchResult(
            paths=paths,
            evidence=evidence,
            cypher=REPORTING_CHAIN_QUERY,
            as_of=as_of,
        )


__all__ = [
    "CARGO_OCCUPANTS_QUERY",
    "DocumentEvidence",
    "DOCUMENT_METADATA_SEARCH_QUERY",
    "GraphCargoOccupant",
    "GraphEntityAmbiguousError",
    "GraphEntityNotFoundError",
    "GraphEvidence",
    "GraphPath",
    "GraphPerson",
    "GraphRelationship",
    "GraphSearch",
    "GraphSearchDataError",
    "GraphSearchError",
    "GraphSearchResult",
    "MAX_HOPS",
    "MAX_PATHS",
    "PEOPLE_DIRECTORY_QUERY",
]
