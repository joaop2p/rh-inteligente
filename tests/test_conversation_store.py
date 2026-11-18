from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.fernet import Fernet

from src.persistence.conversations import (
    Conversation,
    ConversationStore,
    ConversationStoreConfigurationError,
    ConversationStoreCorruptError,
    StoredMessage,
    StoredSource,
)


@pytest.fixture
def encryption_key() -> bytes:
    return Fernet.generate_key()


@pytest.fixture
def store(tmp_path: Path, encryption_key: bytes) -> ConversationStore:
    return ConversationStore(tmp_path / ".tmp" / "conversations", encryption_key)


def test_round_trip_encrypts_unicode_messages_and_sources(
    store: ConversationStore,
    encryption_key: bytes,
) -> None:
    source = StoredSource(
        doc_id="DOC-058",
        chunk_id="DOC-058_chunk_002",
        quote="Promoções — ciclo anual ```",
        vector_score=0.82,
        fulltext_score=1.5,
        rank_score=0.03,
    )
    saved = store.save_exchange(
        None,
        "RAG",
        "Quais são os critérios de promoção?",
        "O ciclo é anual.",
        sources=(source,),
        status="answered",
    )
    path = store._conversation_path(saved.conversation_id)
    encrypted = path.read_bytes()

    assert path.name == f"{saved.conversation_id}.md.enc"
    assert path.stat().st_mode & 0o777 == 0o600
    assert path.parent.stat().st_mode & 0o777 == 0o700
    assert b"Quais s\xc3\xa3o os crit\xc3\xa9rios" not in encrypted
    plaintext = Fernet(encryption_key).decrypt(encrypted).decode("utf-8")
    assert plaintext.startswith(f"# Conversa {saved.conversation_id}")
    assert store.load_conversation(saved.conversation_id) == saved
    assert store.list_conversations() == [saved]


def test_save_appends_exchange_without_crossing_modes(
    store: ConversationStore,
) -> None:
    direct = store.save_exchange(None, "direct", "Oi", "Olá")
    direct_again = store.save_exchange(
        direct.conversation_id,
        "direct",
        "Outra pergunta",
        "Outra resposta",
    )
    rag = store.save_exchange(
        None,
        "RAG",
        "Pergunta RAG",
        "Resposta RAG",
        sources=(
            StoredSource("DOC-1", "DOC-1_chunk_001", "Trecho", None, 1.0, 0.02),
        ),
        status="answered",
    )

    assert len(direct_again.messages) == 4
    assert all(message.mode == "direct" for message in direct_again.messages)
    assert rag.conversation_id != direct.conversation_id
    assert all(message.mode == "RAG" for message in rag.messages)


def test_wrong_key_and_invalid_rag_contract_are_reported(
    store: ConversationStore,
    tmp_path: Path,
) -> None:
    saved = store.save_exchange(None, "direct", "Oi", "Olá")
    wrong_key_store = ConversationStore(
        tmp_path / ".tmp" / "conversations",
        Fernet.generate_key(),
    )

    with pytest.raises(ConversationStoreCorruptError, match="verifique a chave"):
        wrong_key_store.load_conversation(saved.conversation_id)
    with pytest.raises(ValueError, match="precisa de fontes"):
        store.save_exchange(
            None,
            "RAG",
            "Pergunta",
            "Resposta",
            status="answered",
        )


def test_expiration_uses_last_activity_and_exact_30_day_boundary(
    store: ConversationStore,
) -> None:
    saved = store.save_exchange(None, "direct", "Oi", "Olá")
    exact_expiration = saved.updated_at + timedelta(days=30)

    assert store.purge_expired(now=exact_expiration) == 0
    assert store.purge_expired(
        now=exact_expiration + timedelta(microseconds=1)
    ) == 1
    assert store.list_conversations() == []


def test_delete_and_invalid_conversation_id(store: ConversationStore) -> None:
    saved = store.save_exchange(None, "direct", "Oi", "Olá")

    with pytest.raises(ValueError, match="UUID válido"):
        store.load_conversation("../../outside")
    store.delete_conversation(saved.conversation_id)
    with pytest.raises(FileNotFoundError):
        store.load_conversation(saved.conversation_id)


def test_from_env_requires_a_fernet_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RH_CONVERSATION_ENCRYPTION_KEY", raising=False)

    with pytest.raises(ConversationStoreConfigurationError, match="Defina"):
        ConversationStore.from_env()


def test_expired_conversation_is_purged_but_recent_one_remains(
    store: ConversationStore,
) -> None:
    recent = store.save_exchange(None, "direct", "Recente", "Resposta recente")
    old_id = store.save_exchange(None, "direct", "Antiga", "Resposta antiga")
    old = store.load_conversation(old_id.conversation_id)
    stale = Conversation(
        conversation_id=old.conversation_id,
        created_at=old.created_at,
        updated_at=datetime.now(timezone.utc) - timedelta(days=31),
        messages=(
            StoredMessage(mode="direct", role="user", content="Antiga"),
            StoredMessage(mode="direct", role="assistant", content="Resposta antiga"),
        ),
    )
    store._write_conversation(stale)

    assert store.purge_expired() == 1
    assert [conversation.conversation_id for conversation in store.list_conversations()] == [
        recent.conversation_id
    ]


def test_invalid_fernet_key_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConversationStoreConfigurationError):
        ConversationStore(tmp_path, b"not-a-fernet-key")


def test_ciphertext_authentication_failure_is_not_silenced(
    store: ConversationStore,
) -> None:
    saved = store.save_exchange(None, "direct", "Conteúdo", "Resposta")
    path = store._conversation_path(saved.conversation_id)
    ciphertext = bytearray(path.read_bytes())
    ciphertext[-1] ^= 1
    path.write_bytes(ciphertext)

    with pytest.raises(ConversationStoreCorruptError):
        store.load_conversation(saved.conversation_id)

