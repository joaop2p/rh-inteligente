from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pytest
from streamlit.proto.SelectWidgetFilterMode_pb2 import FILTER_MODE_NONE
from streamlit.testing.v1 import AppTest

from src.generation.llm import ChatMessage, LLMRequestError, OllamaClient


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


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


@pytest.mark.parametrize("mode", ["RAG", "GraphRAG", "KAG"])
def test_unimplemented_search_modes_do_not_enable_chat(mode: str) -> None:
    app = AppTest.from_file(APP_PATH).run()
    app.selectbox[0].set_value(mode).run()

    assert not app.exception
    assert app.chat_input[0].disabled
    assert any("ainda não está disponível" in warning.value for warning in app.warning)


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
