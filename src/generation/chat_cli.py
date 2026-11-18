"""Teste interativo de conversa com o Ollama configurado."""

from __future__ import annotations

import argparse
import sys
from datetime import date

from src.generation.llm import (
    ChatMessage,
    LLMConfigurationError,
    LLMRequestError,
    OllamaClient,
)
from src.database.neo4j import Neo4jConfigurationError, Neo4jDatabase
from src.rag.kag_pipeline import KAGPipeline
from src.rag.graphrag_pipeline import GraphRAGPipeline
from src.rag.pipeline import RAGPipeline
from src.retrieval.graph_search import GraphSearch, GraphSearchError
from src.rag.graphrag_pipeline import GraphRAGResponseError
from src.retrieval.vector_search import (
    SearchClauseVectorSearch,
    VectorSearch,
    load_embedding_model,
)
from neo4j.exceptions import CypherSyntaxError


def main() -> int:
    parser = argparse.ArgumentParser(description="Conversa de teste com o Ollama.")
    parser.add_argument("--model", help="Nome do modelo, se houver mais de um.")
    parser.add_argument(
        "--mode",
        choices=("direct", "rag", "graphrag", "kag"),
        default="direct",
    )
    parser.add_argument("--question", help="Pergunta única para execução não interativa.")
    parser.add_argument(
        "--scope",
        choices=("ORGANIZACAO", "PESSOA"),
        default="ORGANIZACAO",
    )
    parser.add_argument("--as-of", dest="as_of", default=date.today().isoformat())
    parser.add_argument(
        "--database",
        action="store_true",
        help="No modo direct, habilita a consulta Neo4j pelo fluxo KAG.",
    )
    args = parser.parse_args()

    try:
        if args.question is None and (
            args.mode != "direct" or args.database
        ):
            raise LLMConfigurationError(
                "Modo interativo no CLI suportado somente em direct; use "
                "--question para rag, graphrag, kag ou direct --database."
            )
        client = OllamaClient.from_env()
        if args.model is not None:
            if not args.model.strip():
                raise LLMConfigurationError("--model não pode estar vazio.")
            client.model_name = args.model.strip()

        if args.question:
            if args.mode == "direct" and not args.database:
                answer = client.chat(
                    [
                        {
                            "role": "system",
                            "content": (
                                "Conversa de teste sem acesso ao Neo4j ou a documentos. "
                                "Não afirme fatos do dataset."
                            ),
                        },
                        {"role": "user", "content": args.question},
                    ]
                )
                print(answer)
                return 0

            with Neo4jDatabase.from_env() as database:
                database.verify_connectivity()
                embedding = load_embedding_model()
                search_clause = SearchClauseVectorSearch(database, embedding)
                graph_search = GraphSearch(database)
                try:
                    as_of = date.fromisoformat(args.as_of)
                    if args.mode in {"kag", "direct"}:
                        response = KAGPipeline(
                            search_clause, graph_search, client
                        ).answer(
                            args.question,
                            as_of=as_of,
                            document_scope=args.scope,
                        )
                    elif args.mode == "graphrag":
                        response = GraphRAGPipeline(
                            search_clause,
                            graph_search,
                            client,
                        ).answer(
                            args.question,
                            as_of=as_of,
                            document_scope=args.scope,
                        )
                    else:
                        response = RAGPipeline(search_clause, client).answer(
                            args.question,
                            document_scope=args.scope,
                        )
                except CypherSyntaxError:
                    search_procedures = VectorSearch(database, embedding)
                    if args.mode in {"kag", "direct"}:
                        response = KAGPipeline(
                            search_procedures, graph_search, client
                        ).answer(
                            args.question,
                            as_of=date.fromisoformat(args.as_of),
                            document_scope=args.scope,
                        )
                    elif args.mode == "graphrag":
                        response = GraphRAGPipeline(
                            search_procedures,
                            graph_search,
                            client,
                        ).answer(
                            args.question,
                            as_of=date.fromisoformat(args.as_of),
                            document_scope=args.scope,
                        )
                    else:
                        response = RAGPipeline(search_procedures, client).answer(
                            args.question,
                            document_scope=args.scope,
                        )

            if args.mode == "rag":
                print(response.answer)
                for source in response.sources:
                    print(f"- {source.doc_id} · {source.chunk_id}")
            else:
                print(response.answer)
                if getattr(response, "knowledge_package", None) is not None:
                    print("Pacote KAG:")
                    import json

                    print(json.dumps(response.knowledge_package, ensure_ascii=False))
                for path in response.paths:
                    print(
                        "Caminho:",
                        " -> ".join(
                            f"{person.nome} ({person.funcionario_id})"
                            for person in path.people
                        ),
                    )
            return 0

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
    except (
        LLMConfigurationError,
        LLMRequestError,
        Neo4jConfigurationError,
        GraphSearchError,
        GraphRAGResponseError,
        ValueError,
    ) as exc:
        print(f"Erro no teste do LLM: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
