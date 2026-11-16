"""Teste interativo de conversa com o Ollama configurado."""

from __future__ import annotations

import argparse
import sys

from src.generation.llm import (
    ChatMessage,
    LLMConfigurationError,
    LLMRequestError,
    OllamaClient,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Conversa de teste com o Ollama.")
    parser.add_argument("--model", help="Nome do modelo, se houver mais de um.")
    args = parser.parse_args()

    try:
        client = OllamaClient.from_env()
        if args.model is not None:
            if not args.model.strip():
                raise LLMConfigurationError("--model não pode estar vazio.")
            client.model_name = args.model.strip()

        messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": (
                    "Esta é uma conversa de teste sem acesso a documentos ou ao "
                    "Neo4j. Não afirme fatos sobre os dados da organização sem "
                    "evidências; informe quando não houver informação suficiente."
                ),
            }
        ]
        print(
            "Teste do LLM sem acesso ao Neo4j ou RAG. "
            "Digite /sair para encerrar; não envie dados sensíveis."
        )
        while True:
            try:
                text = input("\nVocê: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nConversa encerrada.")
                return 0
            if text.lower() == "/sair":
                print("Conversa encerrada.")
                return 0
            if not text:
                continue

            messages.append({"role": "user", "content": text})
            answer = client.chat(messages)
            messages.append({"role": "assistant", "content": answer})
            print(f"\nLLM: {answer}")
    except (LLMConfigurationError, LLMRequestError) as exc:
        print(f"Erro no teste do LLM: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
