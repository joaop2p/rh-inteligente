from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Sequence

from cryptography.fernet import Fernet
import pytest
from streamlit.proto.SelectWidgetFilterMode_pb2 import FILTER_MODE_NONE
from streamlit.testing.v1 import AppTest

from src.database.neo4j import Neo4jDatabase
from src.generation.llm import ChatMessage, LLMRequestError, OllamaClient
from src.persistence.conversations import ConversationStore
from src.retrieval.vector_search import RetrievedChunk, SearchClauseVectorSearch
import src.retrieval.vector_search as vector_search


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


class FakeDatabase:
    def __init__(self) -> None:
        self.verified = False
        self.closed = False

    def verify_connectivity(self) -> None:
        self.verified = True

    def __enter__(self) -> FakeDatabase:
        return self

    def __exit__(self, *args: object) -> None:
        self.closed = True


class FakeRAGLLM:
    def __init__(self) -> None:
        self.messages: Sequence[ChatMessage] | None = None

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        format: Literal["json"] | None = None,
    ) -> str:
        self.messages = messages
        assert temperature == 0.0
        assert format == "json"
        return json.dumps(
            {
                "status": "answered",
                "answer": "Promoções seguem o ciclo anual.",
                "citations": [
                    {"doc_id": "DOC-058", "chunk_id": "DOC-058_chunk_002"}
                ],
            }
        )


def test_direct_chat_preserves_history_and_can_be_cleared(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[ChatMessage]] = []

    def fake_chat(
        self: OllamaClient, messages: Sequence[ChatMessage], **kwargs: object
    ) -> str:
        calls.append(list(messages))
        return f"Resposta {len(calls)}"

    monkeypatch.setattr(OllamaClient, "from_env", lambda: OllamaClient("localhost"))
    monkeypatch.setattr(OllamaClient, "chat", fake_chat)
    app = AppTest.from_file(APP_PATH).run()
    app.chat_input[0].set_value("Olá").run()
    app.chat_input[0].set_value("Outra pergunta").run()

    assert not app.exception
    assert [message["role"] for message in calls[1]] == [
        "system", "user", "assistant", "user"
    ]
    assert calls[1][2]["content"] == "Resposta 1"
    assert len(app.chat_message) == 4
    assert [element.type for element in app if element.type in ("chat_message", "chat_input")] == [
        "chat_message", "chat_message", "chat_message", "chat_message", "chat_input"
    ]
    app.button[0].click().run()
    assert len(app.chat_message) == 0


def test_search_selector_is_not_editable_and_model_selector_is_absent() -> None:
    app = AppTest.from_file(APP_PATH).run()

    assert app.selectbox[0].proto.filter_mode == FILTER_MODE_NONE
    assert not app.selectbox[0].proto.accept_new_options
    assert not app.text_input


@pytest.mark.parametrize("mode", ["GraphRAG", "KAG"])
def test_unimplemented_search_modes_do_not_enable_chat(mode: str) -> None:
    app = AppTest.from_file(APP_PATH).run()
    app.selectbox[0].set_value(mode).run()

    assert not app.exception
    assert app.chat_input[0].disabled
    assert any("ainda não está disponível" in warning.value for warning in app.warning)


def test_rag_mode_requires_remote_consent() -> None:
    app = AppTest.from_file(APP_PATH).run()
    app.selectbox[0].set_value("RAG").run()

    assert not app.exception
    assert app.chat_input[0].disabled
    assert app.checkbox[0].value is False
    assert len(app.selectbox) == 2


def test_rag_mode_queries_pipeline_and_displays_sources(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    database = FakeDatabase()
    llm = FakeRAGLLM()
    store = ConversationStore(tmp_path / "conversations", Fernet.generate_key())
    search_calls: list[tuple[str, int, str | None]] = []
    retrieved = RetrievedChunk(
        chunk_id="DOC-058_chunk_002",
        doc_id="DOC-058",
        texto="Promoções ocorrem no ciclo anual, com aprovação do gestor e do RH.",
        score=0.82,
        rank_score=0.03,
        fulltext_score=1.5,
    )

    def fake_search(
        self: SearchClauseVectorSearch,
        question: str,
        top_k: int = 5,
        document_scope: str | None = None,
    ) -> list[RetrievedChunk]:
        search_calls.append((question, top_k, document_scope))
        return [retrieved]

    monkeypatch.setattr(
        Neo4jDatabase,
        "from_env",
        classmethod(lambda cls: database),
    )
    monkeypatch.setattr(vector_search, "load_embedding_model", lambda: object())
    monkeypatch.setattr(SearchClauseVectorSearch, "search", fake_search)
    monkeypatch.setattr(
        OllamaClient,
        "from_env",
        classmethod(lambda cls: llm),
    )
    monkeypatch.setattr(
        ConversationStore,
        "from_env",
        classmethod(lambda cls: store),
    )
    app = AppTest.from_file(APP_PATH).run()
    app.session_state.messages = [
        {"role": "user", "content": "pergunta direta antiga"},
        {"role": "assistant", "content": "resposta direta antiga"},
    ]
    app.selectbox[0].set_value("RAG").run()
    app.selectbox[1].set_value("Somente pessoas").run()
    app.checkbox[0].check().run()
    app.chat_input[0].set_value("Quais são os critérios de promoção?").run()

    assert not app.exception
    assert database.verified
    assert database.closed
    assert search_calls == [
        ("Quais são os critérios de promoção?", 5, "PESSOA")
    ]
    assert llm.messages is not None
    assert len(llm.messages) == 2
    assert "pergunta direta antiga" not in llm.messages[1]["content"]
    assert app.session_state.messages[0]["content"] == "pergunta direta antiga"
    assert len(app.session_state.rag_messages) == 2
    assert app.session_state.rag_status == "answered"
    assert [source.chunk_id for source in app.session_state.rag_sources] == [
        "DOC-058_chunk_002"
    ]
    assert any("DOC-058" in item.value for item in app.markdown)
    saved = store.load_conversation(app.session_state.rag_conversation_id)
    assert saved.messages[-1].sources[0].doc_id == "DOC-058"
    assert saved.messages[-1].sources[0].chunk_id == "DOC-058_chunk_002"


def test_direct_conversation_can_be_resumed_and_deleted(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "conversations", Fernet.generate_key())

    def fake_chat(
        self: OllamaClient, messages: Sequence[ChatMessage], **kwargs: object
    ) -> str:
        return "Resposta persistida."

    monkeypatch.setattr(
        ConversationStore,
        "from_env",
        classmethod(lambda cls: store),
    )
    monkeypatch.setattr(OllamaClient, "from_env", lambda: OllamaClient("localhost"))
    monkeypatch.setattr(OllamaClient, "chat", fake_chat)

    first_app = AppTest.from_file(APP_PATH).run()
    first_app.chat_input[0].set_value("Pergunta persistida").run()
    assert not first_app.exception
    conversation_id = first_app.session_state.direct_conversation_id
    assert store.load_conversation(conversation_id).messages[-1].content == (
        "Resposta persistida."
    )

    resumed_app = AppTest.from_file(APP_PATH).run()
    assert not resumed_app.exception
    assert any("Pergunta persistida" in item.value for item in resumed_app.markdown)
    assert any("Resposta persistida." in item.value for item in resumed_app.markdown)
    delete_button = next(
        button for button in resumed_app.button
        if button.label == "Excluir conversa atual"
    )
    delete_button.click().run()
    assert not resumed_app.exception
    assert store.list_conversations() == []


def test_theme_selection_and_llm_error_are_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(
        self: OllamaClient, messages: Sequence[ChatMessage], **kwargs: object
    ) -> str:
        raise LLMRequestError("Servidor indisponível.")

    monkeypatch.setattr(OllamaClient, "from_env", lambda: OllamaClient("localhost"))
    monkeypatch.setattr(OllamaClient, "chat", fail)
    app = AppTest.from_file(APP_PATH).run()
    app.radio[0].set_value("Escuro").run()
    assert any("#111827" in item.value for item in app.markdown)
    app.chat_input[0].set_value("Teste").run()
    assert any("Servidor indisponível" in error.value for error in app.error)
    assert app.session_state.messages == []
