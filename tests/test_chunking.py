from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.ingestion.chunking import (
    build_chunks,
    chunk_text,
    load_document_paths,
    normalize_text,
    sanitize_for_indexing,
    split_semantic_units,
)
from src.ingestion.embeddings import generate_embeddings
from src.ingestion.import_chunks import generate_chunk_rows, validate_chunk_rows


class WordTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        return list(range(len(text.split())))

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class FakeEmbeddingModel:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = vectors
        self.call_arguments: dict[str, object] | None = None

    def encode(
        self,
        sentences: list[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
    ) -> list[list[float]]:
        self.call_arguments = {
            "sentences": sentences,
            "batch_size": batch_size,
            "normalize_embeddings": normalize_embeddings,
            "show_progress_bar": show_progress_bar,
        }
        return self.vectors


def test_chunking_preserves_sentence_boundaries_and_canonical_metadata() -> None:
    tokenizer = WordTokenizer()
    text = "Frase um. Frase dois. Frase três."

    chunks = chunk_text(
        text,
        "DOC-001",
        tokenizer,
        max_tokens=5,
        overlap_tokens=2,
    )

    assert [chunk["chunk_id"] for chunk in chunks] == [
        "DOC-001_chunk_001",
        "DOC-001_chunk_002",
    ]
    assert all(chunk["doc_id"] == "DOC-001" for chunk in chunks)
    assert all(chunk["texto"].endswith(".") for chunk in chunks) # type: ignore
    assert all(chunk["token_count"] <= 5 for chunk in chunks) # type: ignore


def test_chunking_rejects_invalid_limits() -> None:
    tokenizer = WordTokenizer()

    with pytest.raises(ValueError, match="maior que zero"):
        chunk_text("Uma frase.", "DOC-001", tokenizer, max_tokens=0)
    with pytest.raises(ValueError, match="menor que max_tokens"):
        chunk_text("Uma frase.", "DOC-001", tokenizer, max_tokens=3, overlap_tokens=3)


def test_chunking_splits_sentence_that_exceeds_limit_without_cutting_words() -> None:
    tokenizer = WordTokenizer()

    chunks = chunk_text(
        "Uma frase muito longa sem encerramento.",
        "DOC-001",
        tokenizer,
        max_tokens=3,
        overlap_tokens=0,
    )

    assert [chunk["texto"] for chunk in chunks] == [
        "Uma frase muito",
        "longa sem encerramento.",
    ]
    assert all(chunk["token_count"] <= 3 for chunk in chunks) # type: ignore


def test_chunking_preserves_list_items_and_indentation() -> None:
    text = normalize_text("Responsabilidades:\n- Equipe\n    1. Relatórios\n* Indicadores")
    assert split_semantic_units(text) == [
        "Responsabilidades:", "- Equipe", "    1. Relatórios", "* Indicadores"
    ]
    chunks = chunk_text(text, "DOC-001", WordTokenizer(), max_tokens=8, overlap_tokens=0)
    assert chunks[0]["texto"] == text


def test_chunking_counts_concatenated_text_and_has_no_zero_overlap() -> None:
    class ContextTokenizer(WordTokenizer):
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            return list(range(len(text.split()) + text.count("\n")))

    tokenizer = ContextTokenizer()
    chunks = chunk_text("Um dois. Três quatro.", "DOC-001", tokenizer, max_tokens=4, overlap_tokens=0)
    assert [chunk["texto"] for chunk in chunks] == ["Um dois.", "Três quatro."]
    assert all(
        chunk["token_count"] == len(tokenizer.encode(str(chunk["texto"]), add_special_tokens=False))
        for chunk in chunks
    )


def test_chunking_rejects_oversized_word_after_valid_word() -> None:
    class LongWordTokenizer(WordTokenizer):
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            return list(range(sum(4 if word == "gigante" else 1 for word in text.split())))

    with pytest.raises(ValueError, match="palavra maior"):
        chunk_text("breve gigante", "DOC-001", LongWordTokenizer(), max_tokens=3, overlap_tokens=0)


def test_document_index_rejects_duplicate_ids_and_external_paths(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("a", encoding="utf-8")
    (tmp_path / "b.txt").write_text("b", encoding="utf-8")
    index = tmp_path / "documentos_index.csv"
    index.write_text("doc_id,arquivo\nDOC-001,a.txt\nDOC-001,b.txt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="doc_id duplicado"):
        load_document_paths(index, tmp_path)
    index.write_text("doc_id,arquivo\nDOC-001,../fora.txt\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fora do diretório"):
        load_document_paths(index, tmp_path)


def test_limit_documents_applies_to_chunk_and_embedding_pipeline(tmp_path: Path) -> None:
    index = tmp_path / "documentos_index.csv"
    index.write_text("doc_id,arquivo\nDOC-001,a.txt\nDOC-002,b.txt\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text("Primeiro documento.", encoding="utf-8")
    (tmp_path / "b.txt").write_text("Segundo documento.", encoding="utf-8")
    model = FakeEmbeddingModel([[1.0, 0.0]])
    chunks = generate_chunk_rows(
        tmp_path, WordTokenizer(), model, embedding_dimension=2, limit_documents=1
    )
    assert len(chunks) == 1
    assert chunks[0]["doc_id"] == "DOC-001"
    with pytest.raises(ValueError, match="limit_documents"):
        build_chunks(index, tmp_path, WordTokenizer(), limit_documents=0)


def test_build_chunks_rejects_empty_document(tmp_path: Path) -> None:
    index = tmp_path / "documentos_index.csv"
    index.write_text("doc_id,arquivo\nDOC-001,a.txt\n", encoding="utf-8")
    (tmp_path / "a.txt").write_text(" \n", encoding="utf-8")
    with pytest.raises(ValueError, match="DOC-001: documento vazio"):
        build_chunks(index, tmp_path, WordTokenizer())


def test_sanitize_for_indexing_masks_cpf_and_remuneration() -> None:
    sanitized = sanitize_for_indexing("CPF 123.456.789-10 e salário de R$ 12.345,67.")

    assert "123.456.789-10" not in sanitized
    assert "R$ 12.345,67" not in sanitized
    assert re.search(r"\[CPF_REDACTED\]", sanitized)
    assert re.search(r"\[REMUNERACAO_REDACTED\]", sanitized)


@pytest.mark.parametrize("value", ["R$ 1000,00", "R$ 100", "R$ 1.234,56", "R$ -50,00"])
def test_sanitize_for_indexing_masks_other_monetary_formats(value: str) -> None:
    assert sanitize_for_indexing(f"Valor: {value}.") == "Valor: [REMUNERACAO_REDACTED]."


def test_generate_embeddings_associates_normalized_vectors_to_chunks() -> None:
    model = FakeEmbeddingModel([[0.6, 0.8], [1.0, 0.0]])
    chunks = [
        {"chunk_id": "DOC-001_chunk_001", "texto": "Primeiro trecho."},
        {"chunk_id": "DOC-001_chunk_002", "texto": "Segundo trecho."},
    ]

    chunks_with_embeddings = generate_embeddings(
        chunks, # type: ignore
        model,
        embedding_dimension=2,
        batch_size=8,
    )

    assert model.call_arguments == {
        "sentences": ["Primeiro trecho.", "Segundo trecho."],
        "batch_size": 8,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }
    assert chunks_with_embeddings[0]["embedding"] == [0.6, 0.8]
    assert "embedding" not in chunks[0]


def test_generate_embeddings_rejects_vectors_with_wrong_dimension() -> None:
    model = FakeEmbeddingModel([[1.0]])

    with pytest.raises(ValueError, match="possui 1 dimensões"):
        generate_embeddings(
            [{"chunk_id": "DOC-001_chunk_001", "texto": "Trecho."}],
            model,
            embedding_dimension=2,
        )


def test_validate_chunk_rows_rejects_duplicate_chunk_id() -> None:
    chunks = [
        {
            "chunk_id": "DOC-001_chunk_001",
            "doc_id": "DOC-001",
            "texto": "Primeiro trecho.",
            "ordem": 1,
            "token_count": 2,
            "embedding": [1.0, 0.0],
        },
        {
            "chunk_id": "DOC-001_chunk_001",
            "doc_id": "DOC-001",
            "texto": "Segundo trecho.",
            "ordem": 2,
            "token_count": 2,
            "embedding": [0.0, 1.0],
        },
    ]

    with pytest.raises(ValueError, match="chunk_id duplicado"):
        validate_chunk_rows(chunks)
