"""Teste opcional de recuperação contra dados e serviços configurados."""

from __future__ import annotations

import os
import unicodedata
from collections.abc import Iterator

import pytest

from src.database.neo4j import Neo4jDatabase
from src.retrieval.vector_search import SearchClauseVectorSearch, load_embedding_model


pytestmark = pytest.mark.skipif(
    os.getenv("RH_RUN_NEO4J_INTEGRATION") != "1",
    reason="Defina RH_RUN_NEO4J_INTEGRATION=1 para testar o Neo4j e o BGE-M3 reais.",
)


@pytest.fixture(scope="module")
def search() -> Iterator[SearchClauseVectorSearch]:
    with Neo4jDatabase.from_env() as database:
        database.verify_connectivity()
        yield SearchClauseVectorSearch(database, load_embedding_model())


@pytest.mark.parametrize(
    ("question", "expected_doc", "expected_chunk", "evidence_term", "document_scope"),
    [
        (
            "Quais são os critérios de promoção no plano de cargos?",
            "DOC-058",
            "DOC-058_chunk_002",
            "promocoes",
            "ORGANIZACAO",
        ),
        (
            "Qual é a hierarquia de níveis no plano de cargos da empresa?",
            "DOC-058",
            "DOC-058_chunk_002",
            "hierarquia",
            "ORGANIZACAO",
        ),
        (
            "Como funciona a gradação de medidas disciplinares?",
            "DOC-059",
            "DOC-059_chunk_001",
            "gradacao",
            "ORGANIZACAO",
        ),
        (
            "Quais são os princípios da política disciplinar?",
            "DOC-059",
            "DOC-059_chunk_001",
            "principios",
            "ORGANIZACAO",
        ),
        (
            "Onde podem ser feitas denúncias de conduta?",
            "DOC-059",
            "DOC-059_chunk_001",
            "canal de etica",
            "ORGANIZACAO",
        ),
        (
            "Qual documento apresenta o organograma oficial da empresa?",
            "DOC-060",
            "DOC-060_chunk_001",
            "organograma oficial",
            "ORGANIZACAO",
        ),
        (
            "O contrato de prestação de serviços PJ prevê subordinação ou vínculo empregatício?",
            "DOC-026",
            "DOC-026_chunk_001",
            "sem vinculo empregaticio",
            "PESSOA",
        ),
        (
            "Qual é a instituição de ensino indicada no termo de compromisso de estágio?",
            "DOC-015",
            "DOC-015_chunk_001",
            "instituicao de ensino",
            "PESSOA",
        ),
        (
            "Qual é a jornada semanal definida no termo de estágio e em seu aditivo?",
            "DOC-015",
            "DOC-015_chunk_002",
            "30 horas semanais",
            "PESSOA",
        ),
        (
            "O que prevê o aditivo que promoveu a coordenador de backend?",
            "DOC-029",
            "DOC-029_chunk_001",
            "coordenador de backend",
            "PESSOA",
        ),
    ],
)
def test_real_search_finds_reviewed_evidence(
    search: SearchClauseVectorSearch,
    question: str,
    expected_doc: str,
    expected_chunk: str,
    evidence_term: str,
    document_scope: str,
) -> None:
    results = search.search(question, top_k=5, document_scope=document_scope)

    evidence = next((item for item in results if item.chunk_id == expected_chunk), None)
    assert evidence is not None, f"Trecho esperado ausente: {expected_chunk}"
    assert evidence.doc_id == expected_doc
    normalized_text = "".join(
        char
        for char in unicodedata.normalize("NFKD", evidence.texto.casefold())
        if not unicodedata.combining(char)
    )
    assert evidence_term in normalized_text
