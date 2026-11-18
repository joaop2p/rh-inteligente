"""Cliente Ollama para o servidor LLM configurado via Tailscale."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypedDict
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMConfigurationError(ValueError):
    """Configuração ausente ou incompatível com o servidor Ollama."""


class LLMRequestError(RuntimeError):
    """Falha ao enviar uma solicitação ao servidor Ollama."""


class OllamaClient:
    """Cliente mínimo para a API nativa de chat do Ollama."""

    def __init__(
        self,
        base_url: str,
        model_name: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout deve ser maior que zero.")
        self.base_url = self._normalize_base_url(base_url)
        self.model_name = model_name.strip() if model_name else None
        self.timeout = timeout

    @staticmethod
    def _normalize_base_url(value: str) -> str:
        value = value.strip()
        if not value:
            raise LLMConfigurationError("TAILSCALE_MODEL_HOST está vazio.")
        if "://" not in value:
            value = "http://" + value
        parsed = urlsplit(value)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise LLMConfigurationError(
                "TAILSCALE_MODEL_HOST deve ser um host ou URL HTTP(S) válido."
            )
        return value.rstrip("/")

    @classmethod
    def from_env(cls, timeout: float = 120.0) -> OllamaClient:
        """Cria o cliente a partir de TAILSCALE_MODEL_HOST no .env."""
        from dotenv import load_dotenv

        load_dotenv(Path.cwd() / ".env", override=False)
        host = os.getenv("TAILSCALE_MODEL_HOST")
        if not host:
            raise LLMConfigurationError(
                "Defina TAILSCALE_MODEL_HOST no ambiente ou no arquivo .env."
            )
        return cls(host, timeout=timeout)

    def _request(
        self, path: str, body: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        data = (
            json.dumps(body, ensure_ascii=False).encode("utf-8")
            if body is not None
            else None
        )
        request = Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                response_body = response.read()
        except HTTPError as exc:
            raise LLMRequestError(
                f"O servidor Ollama respondeu com HTTP {exc.code}."
            ) from exc
        except (URLError, TimeoutError) as exc:
            raise LLMRequestError(
                "Não foi possível conectar ao servidor LLM configurado."
            ) from exc

        try:
            payload = json.loads(response_body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise LLMRequestError(
                "O servidor Ollama retornou uma resposta JSON inválida."
            ) from exc
        if not isinstance(payload, dict):
            raise LLMRequestError(
                "O servidor Ollama retornou um formato de resposta inesperado."
            )
        if "error" in payload:
            raise LLMRequestError("O servidor Ollama informou falha na solicitação.")
        return payload

    def list_models(self) -> list[str]:
        """Lista identificadores locais de modelos disponíveis no Ollama."""
        payload = self._request("/api/tags")
        models = payload.get("models")
        if not isinstance(models, list):
            raise LLMRequestError("Resposta /api/tags sem a lista 'models'.")

        names: list[str] = []
        for model in models:
            if not isinstance(model, dict):
                raise LLMRequestError("Entrada inválida na lista de modelos Ollama.")
            name = model.get("name") or model.get("model")
            if not isinstance(name, str) or not name.strip():
                raise LLMRequestError("Modelo Ollama sem identificador válido.")
            names.append(name.strip())
        return names

    def _resolve_model(self) -> str:
        if self.model_name:
            return self.model_name
        models = self.list_models()
        if len(models) != 1:
            raise LLMConfigurationError(
                "Configure model_name explicitamente quando o servidor Ollama "
                "não tiver exatamente um modelo disponível."
            )
        self.model_name = models[0]
        return self.model_name

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        format: Literal["json"] | None = None,
    ) -> str:
        """Gera texto a partir de mensagens; não envia dados ao Ollama além delas."""
        if not messages:
            raise ValueError("messages não pode estar vazio.")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature deve estar entre 0 e 2.")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens deve ser maior que zero.")
        if format not in {None, "json"}:
            raise ValueError("format deve ser 'json' ou None.")
        for message in messages:
            if message.get("role") not in {"system", "user", "assistant"}:
                raise ValueError("Cada mensagem deve ter role system, user ou assistant.")
            if not isinstance(message.get("content"), str) or not message["content"].strip():
                raise ValueError("Cada mensagem deve ter conteúdo não vazio.")

        options: dict[str, int | float] = {"temperature": temperature}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        body: dict[str, Any] = {
            "model": self._resolve_model(),
            "messages": list(messages),
            "stream": False,
            "think": False,
            "options": options,
        }
        if format is not None:
            body["format"] = format
        payload = self._request("/api/chat", body)
        response_message = payload.get("message")
        if not isinstance(response_message, dict):
            raise LLMRequestError("Resposta do Ollama sem objeto 'message'.")
        content = response_message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise LLMRequestError("Resposta do Ollama sem conteúdo textual.")
        return content.strip()


__all__ = [
    "ChatMessage",
    "LLMConfigurationError",
    "LLMRequestError",
    "OllamaClient",
]