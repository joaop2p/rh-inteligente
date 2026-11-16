from __future__ import annotations

import json
from email.message import Message
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from src.generation.llm import (
    LLMConfigurationError,
    LLMRequestError,
    OllamaClient,
)
from src.generation import chat_cli


class FakeResponse:
    def __init__(self, payload: object) -> None:
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_chat_discovers_single_model_and_sends_non_streaming_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[Request] = []
    responses = iter(
        [
            {"models": [{"name": "local-model:latest"}]},
            {"message": {"content": "Resposta fundamentada."}},
        ]
    )

    def fake_urlopen(
        request: Request, *, timeout: float
    ) -> FakeResponse:
        requests.append(request)
        assert timeout == 120.0
        return FakeResponse(next(responses))

    monkeypatch.setattr("src.generation.llm.urlopen", fake_urlopen)
    client = OllamaClient("100.64.0.1:11434")

    answer = client.chat(
        [{"role": "user", "content": "Responda apenas com uma saudação."}],
        temperature=0.1,
        max_tokens=20,
    )

    assert answer == "Resposta fundamentada."
    assert client.model_name == "local-model:latest"
    assert requests[0].full_url == "http://100.64.0.1:11434/api/tags"
    assert requests[1].full_url == "http://100.64.0.1:11434/api/chat"
    request_data = requests[1].data
    assert isinstance(request_data, bytes)
    body = json.loads(request_data)
    assert body == {
        "model": "local-model:latest",
        "messages": [{"role": "user", "content": "Responda apenas com uma saudação."}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.1, "num_predict": 20},
    }


def test_model_must_be_explicit_when_server_has_multiple_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.generation.llm.urlopen",
        lambda *args, **kwargs: FakeResponse(
            {"models": [{"name": "one"}, {"name": "two"}]}
        ),
    )
    client = OllamaClient("http://localhost:11434")

    with pytest.raises(LLMConfigurationError, match="model_name explicitamente"):
        client.chat([{"role": "user", "content": "Olá"}])


def test_invalid_host_is_rejected() -> None:
    with pytest.raises(LLMConfigurationError, match="URL HTTP"):
        OllamaClient("ftp://localhost:11434")


def test_chat_rejects_empty_or_invalid_messages() -> None:
    client = OllamaClient("localhost:11434", model_name="model")
    with pytest.raises(ValueError, match="não pode estar vazio"):
        client.chat([])
    with pytest.raises(ValueError, match="role"):
        client.chat([{"role": "tool", "content": "resultado"}])  # type: ignore[list-item]
    with pytest.raises(ValueError, match="conteúdo não vazio"):
        client.chat([{"role": "user", "content": " "}])
    with pytest.raises(ValueError, match="temperature"):
        client.chat([{"role": "user", "content": "Oi"}], temperature=3)


def test_http_errors_do_not_include_configured_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject_request(
        request: Request, *, timeout: float
    ) -> FakeResponse:
        raise HTTPError(
            request.full_url,
            503,
            "Unavailable",
            hdrs=Message(),
            fp=None,
        )

    monkeypatch.setattr("src.generation.llm.urlopen", reject_request)
    client = OllamaClient("http://private-host.invalid:11434", model_name="model")

    with pytest.raises(LLMRequestError, match="HTTP 503") as error:
        client.chat([{"role": "user", "content": "Oi"}])
    assert "private-host.invalid" not in str(error.value)


def test_list_models_accepts_ollama_model_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = iter(
        [
            {"models": [{"model": "legacy-model"}, {"name": "named-model"}]},
        ]
    )

    def fake_urlopen(
        request: Request, *, timeout: float
    ) -> FakeResponse:
        return FakeResponse(next(responses))

    monkeypatch.setattr("src.generation.llm.urlopen", fake_urlopen)
    assert OllamaClient("localhost:11434").list_models() == [
        "legacy-model",
        "named-model",
    ]


def test_interactive_chat_preserves_history_and_exits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    requests: list[list[dict[str, str]]] = []

    class FakeClient:
        model_name: str | None = None

        def chat(self, messages: list[dict[str, str]]) -> str:
            requests.append([message.copy() for message in messages])
            return f"Resposta {len(requests)}"

    inputs = iter(["Olá", "Como vai?", "/sair"])
    monkeypatch.setattr("sys.argv", ["chat_cli"])
    monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
    monkeypatch.setattr(chat_cli.OllamaClient, "from_env", lambda: FakeClient())

    assert chat_cli.main() == 0
    assert [message["role"] for message in requests[1]] == [
        "system", "user", "assistant", "user"
    ]
    assert requests[1][2]["content"] == "Resposta 1"
    assert "LLM: Resposta 2" in capsys.readouterr().out


def test_interactive_chat_surfaces_connection_errors(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail() -> OllamaClient:
        raise LLMRequestError("Servidor indisponível.")

    monkeypatch.setattr("sys.argv", ["chat_cli"])
    monkeypatch.setattr(chat_cli.OllamaClient, "from_env", fail)
    assert chat_cli.main() == 1
    assert "Servidor indisponível" in capsys.readouterr().err
