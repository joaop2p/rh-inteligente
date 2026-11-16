"""Chunking seguro e rastreável para os documentos do dataset."""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Protocol


CPF_PATTERN = re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")
REMUNERATION_PATTERN = re.compile(r"R\$\s*[-+]?\s*\d(?:[\d.,]*\d)?")
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+")
CLAUSE_BOUNDARY_PATTERN = re.compile(r"(?<=[,;:])\s+")
LIST_ITEM_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


class Tokenizer(Protocol):
    """Interface mínima usada pelos tokenizers do Transformers."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool) -> str: ...


def normalize_text(text: str) -> str:
    """Normaliza espaços sem alterar o conteúdo lógico do documento."""
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").replace("\t", "    ")
    lines = [line.rstrip() for line in normalized.split("\n")]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip("\n")


def sanitize_for_indexing(text: str) -> str:
    """Remove identificadores e valores financeiros do texto indexável."""
    sanitized = CPF_PATTERN.sub("[CPF_REDACTED]", text)
    return REMUNERATION_PATTERN.sub("[REMUNERACAO_REDACTED]", sanitized)


def load_document_paths(index_path: Path, data_dir: Path) -> dict[str, Path]:
    """Mapeia cada arquivo do índice ao seu ID canônico e valida sua existência."""
    if not index_path.is_file():
        raise ValueError(f"Índice de documentos não encontrado: {index_path}")

    documents: dict[str, Path] = {}
    data_root = data_dir.resolve()
    with index_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            doc_id = (row.get("doc_id") or "").strip()
            relative_path = (row.get("arquivo") or "").strip()
            path = (data_root / relative_path).resolve()

            if not path.is_relative_to(data_root):
                raise ValueError(
                    f"{doc_id}: caminho fora do diretório do dataset."
                )
            if not doc_id or not relative_path:
                raise ValueError("documentos_index.csv contém doc_id ou arquivo vazio.")

            if path in documents.values():
                raise ValueError(f"Arquivo duplicado no índice: {relative_path}")
            if not path.is_file():
                raise ValueError(f"{doc_id}: documento não encontrado: {relative_path}")
            if doc_id in documents:
                raise ValueError(f"doc_id duplicado no índice: {doc_id}")
            documents[doc_id] = path

    if not documents:
        raise ValueError("documentos_index.csv não contém documentos.")
    return documents


def split_semantic_units(text: str) -> list[str]:
    """Divide frases e mantém itens de lista como unidades separadas."""
    units: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            units.extend(
                sentence.strip()
                for sentence in SENTENCE_BOUNDARY_PATTERN.split(" ".join(paragraph))
                if sentence.strip()
            )
            paragraph.clear()

    for line in text.splitlines():
        if not line.strip():
            flush_paragraph()
        elif LIST_ITEM_PATTERN.match(line):
            flush_paragraph()
            units.append(line.rstrip())
        else:
            paragraph.append(line.strip())
    flush_paragraph()
    return units


def split_oversized_unit(unit: str, tokenizer: Tokenizer, max_tokens: int) -> list[str]:
    """Divide uma unidade longa por cláusulas e, em último caso, por palavras."""
    if len(tokenizer.encode(unit, add_special_tokens=False)) <= max_tokens:
        return [unit]
    clauses = [clause.rstrip() for clause in CLAUSE_BOUNDARY_PATTERN.split(unit) if clause.strip()]
    units: list[str] = []
    for clause in clauses:
        if len(tokenizer.encode(clause, add_special_tokens=False)) <= max_tokens:
            units.append(clause)
            continue

        indent = clause[:len(clause) - len(clause.lstrip())]
        words = clause.split()
        current_words: list[str] = []
        for word in words:
            candidate_words = [*current_words, word]
            candidate = (indent if not units else "") + " ".join(candidate_words)
            if len(tokenizer.encode(candidate, add_special_tokens=False)) <= max_tokens:
                current_words = candidate_words
                continue
            if not current_words:
                raise ValueError(f"Unidade contém palavra maior que {max_tokens} tokens: {word!r}")
            units.append((indent if not units else "") + " ".join(current_words))
            if len(tokenizer.encode(word, add_special_tokens=False)) > max_tokens:
                raise ValueError(f"Unidade contém palavra maior que {max_tokens} tokens: {word!r}")
            current_words = [word]
        if current_words:
            units.append((indent if not units else "") + " ".join(current_words))
    return units



def chunk_text(
    text: str,
    doc_id: str,
    tokenizer: Tokenizer,
    max_tokens: int = 350,
    overlap_tokens: int = 50,
) -> list[dict[str, str | int]]:
    """Cria chunks por unidades semânticas, respeitando o limite de tokens."""

    if max_tokens <= 0:
        raise ValueError("max_tokens deve ser maior que zero.")

    if not 0 <= overlap_tokens < max_tokens:
        raise ValueError(
            "overlap_tokens deve ser maior ou igual a zero "
            "e menor que max_tokens."
        )

    units = [
        split_unit
        for unit in split_semantic_units(text)
        for split_unit in split_oversized_unit(
            unit, tokenizer, max_tokens
        )
    ]

    chunks: list[dict[str, str | int]] = []
    current: list[str] = []

    def count_tokens(units_to_count: list[str]) -> int:
        combined = "\n".join(units_to_count)
        return len(
            tokenizer.encode(
                combined,
                add_special_tokens=False,
            )
        )

    def append_chunk(units_to_append: list[str]) -> None:
        content = "\n".join(units_to_append)
        token_count = count_tokens(units_to_append)

        if token_count > max_tokens:
            raise ValueError(
                f"{doc_id}: chunk excedeu o limite de tokens."
            )

        chunks.append({
            "chunk_id": f"{doc_id}_chunk_{len(chunks) + 1:03d}",
            "doc_id": doc_id,
            "ordem": len(chunks) + 1,
            "texto": content,
            "token_count": token_count,
        })

    for unit in units:
        candidate = [*current, unit]

        if not current or count_tokens(candidate) <= max_tokens:
            current.append(unit)
            continue

        append_chunk(current)

        # Sem overlap, inicia o próximo chunk diretamente.
        if overlap_tokens == 0:
            current = [unit]
            continue

        # Recupera o sufixo do chunk anterior sem exceder o limite.
        overlap: list[str] = []

        for previous_unit in reversed(current):
            candidate_overlap = [
                previous_unit,
                *overlap,
                unit,
            ]

            if count_tokens(candidate_overlap) > max_tokens:
                break

            overlap.insert(0, previous_unit)

            if count_tokens(overlap) >= overlap_tokens:
                break

        current = [*overlap, unit]

    if current:
        append_chunk(current)

    return chunks



def build_chunks(
    index_path: Path,
    data_dir: Path,
    tokenizer: Tokenizer,
    max_tokens: int = 350,
    overlap_tokens: int = 50,
    limit_documents: int | None = None,
) -> list[dict[str, str | int]]:
    """Gera chunks sanitizados para os documentos referenciados pelo índice."""
    if limit_documents is not None and limit_documents <= 0:
        raise ValueError("limit_documents deve ser maior que zero.")
    chunks: list[dict[str, str | int]] = []
    documents = list(load_document_paths(index_path, data_dir).items())
    for doc_id, path in documents[:limit_documents]:
        document_text = sanitize_for_indexing(normalize_text(path.read_text(encoding="utf-8")))
        document_chunks = chunk_text(
            document_text,
            doc_id,
            tokenizer,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )
        if not document_chunks:
            raise ValueError(f"{doc_id}: documento vazio.")
        chunks.extend(document_chunks)
    return chunks
