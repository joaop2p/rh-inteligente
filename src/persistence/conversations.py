"""Armazenamento local criptografado de conversas."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from cryptography.fernet import Fernet, InvalidToken


ConversationMode = Literal["direct", "RAG"]
MessageRole = Literal["user", "assistant"]
RAGStatus = Literal["answered", "insufficient"]
RETENTION_PERIOD = timedelta(days=30)
_MARKDOWN_JSON_START = "```json\n"


class ConversationStoreError(RuntimeError):
    """Falha lendo ou escrevendo uma conversa criptografada."""


class ConversationStoreConfigurationError(ConversationStoreError):
    """Chave de criptografia ausente ou inválida."""


class ConversationStoreCorruptError(ConversationStoreError):
    """Arquivo de conversa inválido ou não descriptografável."""


@dataclass(frozen=True)
class StoredSource:
    doc_id: str
    chunk_id: str
    quote: str
    vector_score: float | None
    fulltext_score: float | None
    rank_score: float


@dataclass(frozen=True)
class StoredMessage:
    mode: ConversationMode
    role: MessageRole
    content: str
    sources: tuple[StoredSource, ...] = ()
    status: RAGStatus | None = None


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    created_at: datetime
    updated_at: datetime
    messages: tuple[StoredMessage, ...]

    @property
    def title(self) -> str:
        first_question = next(
            (
                message.content.strip().replace("\n", " ")
                for message in self.messages
                if message.role == "user"
            ),
            "Nova conversa",
        )
        return (
            first_question[:57] + "..."
            if len(first_question) > 60
            else first_question
        )


class ConversationStore:
    """Persiste cada sessão como Markdown UTF-8 criptografado com Fernet."""

    def __init__(self, directory: Path, encryption_key: bytes) -> None:
        try:
            self._fernet = Fernet(encryption_key)
        except (TypeError, ValueError) as exc:
            raise ConversationStoreConfigurationError(
                "RH_CONVERSATION_ENCRYPTION_KEY não é uma chave Fernet válida."
            ) from exc
        self._directory = directory

    @classmethod
    def from_env(cls) -> ConversationStore:
        """Carrega a chave local e usa `.tmp/conversations` dentro do projeto."""
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env", override=False)
        key_value = os.getenv("RH_CONVERSATION_ENCRYPTION_KEY")
        if not key_value:
            raise ConversationStoreConfigurationError(
                "Defina RH_CONVERSATION_ENCRYPTION_KEY no arquivo .env."
            )
        try:
            key = key_value.strip().encode("ascii")
        except UnicodeEncodeError as exc:
            raise ConversationStoreConfigurationError(
                "RH_CONVERSATION_ENCRYPTION_KEY deve ser uma chave Fernet ASCII."
            ) from exc
        directory = Path(__file__).resolve().parents[2] / ".tmp" / "conversations"
        return cls(directory, key)

    def save_exchange(
        self,
        conversation_id: str | None,
        mode: ConversationMode,
        user_content: str,
        assistant_content: str,
        *,
        sources: tuple[StoredSource, ...] = (),
        status: RAGStatus | None = None,
    ) -> Conversation:
        if mode not in {"direct", "RAG"}:
            raise ValueError("mode deve ser direct ou RAG.")
        if not user_content.strip() or not assistant_content.strip():
            raise ValueError("As mensagens da conversa não podem ser vazias.")
        if mode == "direct" and (sources or status is not None):
            raise ValueError("Conversa direta não pode armazenar fontes RAG.")
        if mode == "RAG" and status not in {"answered", "insufficient"}:
            raise ValueError("Resposta RAG precisa de status válido.")
        if mode == "RAG" and status == "answered" and not sources:
            raise ValueError("Resposta RAG fundamentada precisa de fontes.")
        if mode == "RAG" and status == "insufficient" and sources:
            raise ValueError("Resposta insuficiente não pode ter fontes.")

        now = datetime.now(timezone.utc)
        if conversation_id is None:
            conversation_id = str(uuid4())
            created_at = now
            messages: list[StoredMessage] = []
        else:
            conversation = self.load_conversation(conversation_id)
            created_at = conversation.created_at
            messages = list(conversation.messages)

        messages.extend(
            [
                StoredMessage(mode=mode, role="user", content=user_content),
                StoredMessage(
                    mode=mode,
                    role="assistant",
                    content=assistant_content,
                    sources=sources,
                    status=status,
                ),
            ]
        )
        result = Conversation(
            conversation_id=conversation_id,
            created_at=created_at,
            updated_at=now,
            messages=tuple(messages),
        )
        self._write_conversation(result)
        return result

    def list_conversations(self) -> list[Conversation]:
        if not self._directory.exists():
            return []
        conversations = [
            self._read_conversation(path)
            for path in self._directory.glob("*.md.enc")
        ]
        return sorted(
            conversations,
            key=lambda conversation: conversation.updated_at,
            reverse=True,
        )

    def load_conversation(self, conversation_id: str) -> Conversation:
        path = self._conversation_path(conversation_id)
        if not path.is_file():
            raise FileNotFoundError("A conversa solicitada não existe.")
        return self._read_conversation(path)

    def delete_conversation(self, conversation_id: str) -> None:
        path = self._conversation_path(conversation_id)
        if not path.is_file():
            raise FileNotFoundError("A conversa solicitada não existe.")
        path.unlink()

    def purge_expired(self, now: datetime | None = None) -> int:
        if not self._directory.exists():
            return 0
        current_time = now or datetime.now(timezone.utc)
        if current_time.tzinfo is None:
            raise ValueError("now deve incluir fuso horário.")

        expired = [
            conversation
            for conversation in self.list_conversations()
            if current_time - conversation.updated_at > RETENTION_PERIOD
        ]
        for conversation in expired:
            self._conversation_path(conversation.conversation_id).unlink()
        return len(expired)

    def _conversation_path(self, conversation_id: str) -> Path:
        try:
            parsed_id = UUID(conversation_id)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError("conversation_id deve ser um UUID válido.") from exc
        if str(parsed_id) != conversation_id:
            raise ValueError("conversation_id deve estar no formato UUID canônico.")
        return self._directory / f"{conversation_id}.md.enc"

    def _write_conversation(self, conversation: Conversation) -> None:
        self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._directory.chmod(0o700)
        path = self._conversation_path(conversation.conversation_id)
        temporary_path = self._directory / f".{uuid4()}.tmp"
        markdown = self._serialize_markdown(conversation)
        encrypted = self._fernet.encrypt(markdown.encode("utf-8"))
        try:
            with temporary_path.open("xb") as file:
                file.write(encrypted)
                file.flush()
                os.fsync(file.fileno())
            temporary_path.chmod(0o600)
            temporary_path.replace(path)
            path.chmod(0o600)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _read_conversation(self, path: Path) -> Conversation:
        if path.is_symlink():
            raise ConversationStoreError("Links simbólicos não são aceitos.")
        try:
            plaintext = self._fernet.decrypt(path.read_bytes()).decode("utf-8")
        except InvalidToken as exc:
            raise ConversationStoreCorruptError(
                f"Não foi possível descriptografar {path.name}; verifique a chave."
            ) from exc
        except UnicodeDecodeError as exc:
            raise ConversationStoreCorruptError(
                f"O conteúdo de {path.name} não está codificado em UTF-8."
            ) from exc
        try:
            conversation = self._parse_markdown(plaintext)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ConversationStoreCorruptError(
                f"O conteúdo de {path.name} não segue o formato de conversa."
            ) from exc
        expected_id = path.name.removesuffix(".md.enc")
        if conversation.conversation_id != expected_id:
            raise ConversationStoreCorruptError(
                f"O UUID interno de {path.name} não corresponde ao nome do arquivo."
            )
        return conversation

    @staticmethod
    def _serialize_markdown(conversation: Conversation) -> str:
        payload = {
            "conversation_id": conversation.conversation_id,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "messages": [
                {
                    "mode": message.mode,
                    "role": message.role,
                    "content": message.content,
                    "sources": [
                        {
                            "doc_id": source.doc_id,
                            "chunk_id": source.chunk_id,
                            "quote": source.quote,
                            "vector_score": source.vector_score,
                            "fulltext_score": source.fulltext_score,
                            "rank_score": source.rank_score,
                        }
                        for source in message.sources
                    ],
                    "status": message.status,
                }
                for message in conversation.messages
            ],
        }
        json_payload = json.dumps(payload, ensure_ascii=False, indent=2).replace(
            "`", "\\u0060"
        )
        return (
            f"# Conversa {conversation.conversation_id}\n\n"
            f"- Criada em: `{conversation.created_at.isoformat()}`\n"
            f"- Atualizada em: `{conversation.updated_at.isoformat()}`\n\n"
            f"{_MARKDOWN_JSON_START}{json_payload}\n```\n"
        )

    @staticmethod
    def _parse_markdown(markdown: str) -> Conversation:
        start = markdown.find(_MARKDOWN_JSON_START)
        if start < 0:
            raise ValueError("Bloco JSON ausente.")
        payload_start = start + len(_MARKDOWN_JSON_START)
        payload_end = markdown.find("\n```", payload_start)
        if payload_end < 0 or markdown[payload_end + 4 :].strip():
            raise ValueError("Bloco JSON incompleto.")
        payload = json.loads(markdown[payload_start:payload_end])
        if not isinstance(payload, dict):
            raise ValueError("Payload deve ser um objeto.")

        conversation_id = payload["conversation_id"]
        created_at = ConversationStore._parse_datetime(payload["created_at"])
        updated_at = ConversationStore._parse_datetime(payload["updated_at"])
        raw_messages = payload["messages"]
        if not isinstance(conversation_id, str) or not isinstance(raw_messages, list):
            raise ValueError("Metadados de conversa inválidos.")

        messages = tuple(
            ConversationStore._parse_message(raw_message)
            for raw_message in raw_messages
        )
        return Conversation(
            conversation_id=conversation_id,
            created_at=created_at,
            updated_at=updated_at,
            messages=messages,
        )

    @staticmethod
    def _parse_datetime(value: object) -> datetime:
        if not isinstance(value, str):
            raise ValueError("Data inválida.")
        parsed = datetime.fromisoformat(value)
        if parsed.tzinfo is None:
            raise ValueError("Data sem fuso horário.")
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _parse_message(raw_message: object) -> StoredMessage:
        if not isinstance(raw_message, dict):
            raise ValueError("Mensagem inválida.")
        mode = raw_message["mode"]
        role = raw_message["role"]
        content = raw_message["content"]
        status = raw_message["status"]
        raw_sources = raw_message["sources"]
        if mode not in {"direct", "RAG"} or role not in {"user", "assistant"}:
            raise ValueError("Modo ou papel inválido.")
        if not isinstance(content, str) or not isinstance(raw_sources, list):
            raise ValueError("Conteúdo ou fontes inválidas.")
        if status not in {None, "answered", "insufficient"}:
            raise ValueError("Status inválido.")
        return StoredMessage(
            mode=mode,
            role=role,
            content=content,
            sources=tuple(
                ConversationStore._parse_source(raw_source)
                for raw_source in raw_sources
            ),
            status=status,
        )

    @staticmethod
    def _parse_source(raw_source: object) -> StoredSource:
        if not isinstance(raw_source, dict):
            raise ValueError("Fonte inválida.")
        doc_id = raw_source["doc_id"]
        chunk_id = raw_source["chunk_id"]
        quote = raw_source["quote"]
        vector_score = ConversationStore._optional_score(
            raw_source["vector_score"]
        )
        fulltext_score = ConversationStore._optional_score(
            raw_source["fulltext_score"]
        )
        rank_score = raw_source["rank_score"]
        if (
            not isinstance(doc_id, str)
            or not isinstance(chunk_id, str)
            or not isinstance(quote, str)
            or isinstance(rank_score, bool)
            or not isinstance(rank_score, (int, float))
        ):
            raise ValueError("Campos de fonte inválidos.")
        return StoredSource(
            doc_id=doc_id,
            chunk_id=chunk_id,
            quote=quote,
            vector_score=vector_score,
            fulltext_score=fulltext_score,
            rank_score=float(rank_score),
        )

    @staticmethod
    def _optional_score(value: object) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Score inválido.")
        return float(value)


__all__ = [
    "Conversation",
    "ConversationMode",
    "ConversationStore",
    "ConversationStoreConfigurationError",
    "ConversationStoreCorruptError",
    "ConversationStoreError",
    "RAGStatus",
    "StoredMessage",
    "StoredSource",
]
