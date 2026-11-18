"""Interface Streamlit do Sistema de RH Inteligente."""

from __future__ import annotations

import logging
from datetime import date

import streamlit as st
from neo4j.exceptions import CypherSyntaxError, Neo4jError

from src.database.neo4j import Neo4jConfigurationError, Neo4jDatabase
from src.generation.llm import (
    ChatMessage,
    LLMConfigurationError,
    LLMRequestError,
    OllamaClient,
)
from src.persistence.conversations import (
    Conversation,
    ConversationStore,
    ConversationStoreError,
    StoredSource,
)
from src.rag.graphrag_pipeline import (
    GraphRAGPipeline,
    GraphRAGResponse,
    GraphRAGResponseError,
)
from src.rag.kag_pipeline import KAGPipeline
from src.rag.pipeline import RAGPipeline, RAGResponse, RAGResponseError
from src.retrieval.graph_search import GraphSearch, GraphSearchError
from src.retrieval.vector_search import (
    EmbeddingModel,
    SearchClauseVectorSearch,
    VectorSearch,
    load_embedding_model,
)
logging.getLogger("streamlit").setLevel(logging.CRITICAL)

LOGGER = logging.getLogger(__name__)


SYSTEM_MESSAGE: ChatMessage = {
    "role": "system",
    "content": (
        "Esta é uma conversa de teste sem acesso a documentos ou ao Neo4j. "
        "Não afirme fatos sobre os dados da organização sem evidências; "
        "informe quando não houver informação suficiente."
    ),
}

THEMES = {
    "Claro": {
        "background": "#f5f7fb",
        "surface": "#ffffff",
        "sidebar": "#edf2fa",
        "text": "#17233b",
        "muted": "#53627b",
        "border": "#d9e2ef",
        "accent": "#315edb",
    },
    "Escuro": {
        "background": "#111827",
        "surface": "#1e293b",
        "sidebar": "#172033",
        "text": "#e9eff9",
        "muted": "#becbe0",
        "border": "#40516b",
        "accent": "#91acff",
    },
}


@st.cache_resource
def _cached_embedding_model() -> EmbeddingModel:
    return load_embedding_model()


def answer_with_rag(question: str, document_scope: str | None) -> RAGResponse:
    with Neo4jDatabase.from_env() as database:
        database.verify_connectivity()
        embedding_model = _cached_embedding_model()
        llm = OllamaClient.from_env()
        search_clause = SearchClauseVectorSearch(database, embedding_model)
        try:
            response = RAGPipeline(search_clause, llm).answer(
                question,
                document_scope=document_scope,
            )
        except CypherSyntaxError as exc:
            LOGGER.warning(
                "SEARCH do Neo4j indisponível (%s); usando as procedures clássicas.",
                type(exc).__name__,
            )
            search_procedures = VectorSearch(database, embedding_model)
            response = RAGPipeline(search_procedures, llm).answer(
                question,
                document_scope=document_scope,
            )
            LOGGER.info("Recuperação RAG concluída pelo caminho de procedures.")
            return response

        LOGGER.info("Recuperação RAG concluída pelo caminho Cypher 25 SEARCH.")
        return response


def answer_with_graphrag(
    question: str,
    document_scope: str | None,
    as_of: date,
) -> GraphRAGResponse:
    with Neo4jDatabase.from_env() as database:
        database.verify_connectivity()
        embedding_model = _cached_embedding_model()
        llm = OllamaClient.from_env()
        graph_search = GraphSearch(database)
        search_clause = SearchClauseVectorSearch(database, embedding_model)
        try:
            response = GraphRAGPipeline(
                search_clause,
                graph_search,
                llm,
            ).answer(
                question,
                document_scope=document_scope,
                as_of=as_of,
            )
        except CypherSyntaxError as exc:
            LOGGER.warning(
                "SEARCH do Neo4j indisponível (%s); usando as procedures clássicas.",
                type(exc).__name__,
            )
            search_procedures = VectorSearch(database, embedding_model)
            response = GraphRAGPipeline(
                search_procedures,
                graph_search,
                llm,
            ).answer(
                question,
                document_scope=document_scope,
                as_of=as_of,
            )
            LOGGER.info("Recuperação GraphRAG concluída pelo caminho de procedures.")
            return response

        LOGGER.info("Recuperação GraphRAG concluída pelo caminho Cypher 25 SEARCH.")
        return response


def answer_with_kag(
    question: str,
    document_scope: str | None,
    as_of: date,
) -> GraphRAGResponse:
    """Executa o mesmo núcleo seguro do GraphRAG e retorna seu pacote KAG."""
    with Neo4jDatabase.from_env() as database:
        database.verify_connectivity()
        embedding_model = _cached_embedding_model()
        llm = OllamaClient.from_env()
        graph_search = GraphSearch(database)
        search_clause = SearchClauseVectorSearch(database, embedding_model)
        try:
            response = KAGPipeline(
                search_clause, graph_search, llm
            ).answer(
                question,
                document_scope=document_scope,
                as_of=as_of,
            )
        except CypherSyntaxError as exc:
            LOGGER.warning(
                "SEARCH do Neo4j indisponível (%s); usando as procedures clássicas.",
                type(exc).__name__,
            )
            search_procedures = VectorSearch(database, embedding_model)
            response = KAGPipeline(
                search_procedures, graph_search, llm
            ).answer(
                question,
                document_scope=document_scope,
                as_of=as_of,
            )
        return response


def restore_conversation(mode: str, saved: Conversation) -> None:
    message_key = "messages" if mode == "direct" else "rag_messages"
    restored = [
        {"role": message.role, "content": message.content}
        for message in saved.messages
        if message.mode == mode
    ]
    st.session_state[message_key] = restored
    if mode == "RAG":
        latest_answer = next(
            (
                message
                for message in reversed(saved.messages)
                if message.mode == "RAG" and message.role == "assistant"
            ),
            None,
        )
        st.session_state.rag_sources = (
            list(latest_answer.sources) if latest_answer is not None else []
        )
        st.session_state.rag_status = (
            latest_answer.status if latest_answer is not None else None
        )


def apply_theme(theme: str) -> None:
    colors = THEMES[theme]
    st.markdown(
        f"""
        <style>
        .stApp, [data-testid="stAppViewContainer"] {{
            background: {colors["background"]};
            color: {colors["text"]};
        }}
        [data-testid="stHeader"] {{ background: {colors["background"]}; }}
        [data-testid="stSidebar"], [data-testid="stSidebarContent"] {{
            background: {colors["sidebar"]};
        }}
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] h1,
        [data-testid="stAppViewContainer"] h2,
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label {{
            color: {colors["text"]};
        }}
        [data-testid="stCaptionContainer"] p {{ color: {colors["muted"]}; }}
        [data-testid="stChatMessage"], [data-testid="stVerticalBlockBorderWrapper"],
        [data-baseweb="select"] > div, [data-testid="stChatInput"] textarea,
        [data-testid="stTextInput"] input {{
            background: {colors["surface"]};
            color: {colors["text"]};
            border-color: {colors["border"]};
        }}
        [data-testid="stChatInput"] {{ background: {colors["background"]}; }}
        [data-testid="stChatInput"] div,
        [data-testid="stChatInput"] button,
        [data-testid="stAlert"] > div {{
            background: {colors["surface"]};
            color: {colors["text"]};
        }}
        [data-testid="stAlert"] > div {{ border: 1px solid {colors["border"]}; }}
        [data-testid="stChatInput"] textarea::placeholder {{
            color: {colors["muted"]};
            opacity: 1;
        }}
        [data-testid="stBaseButton-secondary"] {{
            color: {colors["text"]};
            background: {colors["surface"]};
            border-color: {colors["border"]};
        }}
        a {{ color: {colors["accent"]}; }}
        .block-container {{ padding-top: 2rem; max-width: 1500px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="RH Inteligente", page_icon="💬", layout="wide")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "direct_conversation_id" not in st.session_state:
        st.session_state.direct_conversation_id = None
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = []
    if "rag_conversation_id" not in st.session_state:
        st.session_state.rag_conversation_id = None
    if "rag_sources" not in st.session_state:
        st.session_state.rag_sources = []
    if "rag_status" not in st.session_state:
        st.session_state.rag_status = None
    if "graphrag_messages" not in st.session_state:
        st.session_state.graphrag_messages = []
    if "graphrag_sources" not in st.session_state:
        st.session_state.graphrag_sources = []
    if "graphrag_paths" not in st.session_state:
        st.session_state.graphrag_paths = []
    if "graphrag_cypher" not in st.session_state:
        st.session_state.graphrag_cypher = ""
    if "graphrag_status" not in st.session_state:
        st.session_state.graphrag_status = None
    if "graphrag_result_as_of" not in st.session_state:
        st.session_state.graphrag_result_as_of = None
    if "direct_database_consultation" not in st.session_state:
        st.session_state.direct_database_consultation = False

    store: ConversationStore | None = None
    store_error: str | None = None
    try:
        store = ConversationStore.from_env()
        store.purge_expired()
    except (ConversationStoreError, OSError) as exc:
        store = None
        store_error = str(exc)

    with st.sidebar:
        st.markdown("### ◈ RH Inteligente")
        st.caption("NovaTech · assistente de conhecimento")
        st.divider()
        st.markdown("**Área de trabalho**")
        st.caption("Conversa")
        mode = st.selectbox(
            "Modo de busca",
            ("Conversa direta", "RAG", "GraphRAG", "KAG"),
            help="RAG consulta documentos; GraphRAG e KAG combinam grafo hierárquico e documentos.",
            filter_mode=None,
        )
        document_scope: str | None = "ORGANIZACAO"
        rag_authorized = False
        if mode in {"RAG", "GraphRAG", "KAG"}:
            scope_options = {
                "Somente organização": "ORGANIZACAO",
                "Somente pessoas": "PESSOA",
                "Todos os documentos": None,
            }
            scope_label = st.selectbox(
                "Escopo documental",
                tuple(scope_options),
                help="O escopo é escolhido explicitamente e não é inferido da pergunta.",
            )
            document_scope = scope_options[scope_label]
            if mode == "RAG":
                st.warning(
                    "RAG envia sua pergunta, doc_id/chunk_id e trechos recuperados "
                    "ao Ollama via Tailscale. Use somente dados saneados; no escopo "
                    "PESSOA, confirme que o envio está autorizado."
                )
            elif mode == "GraphRAG":
                st.warning(
                    "GraphRAG envia a pergunta, caminho hierárquico e chunks "
                    "recuperados ao Ollama via Tailscale. A busca atual exige "
                    "uma pergunta hierárquica sobre pessoa canônica ou CEO. "
                    "A hierarquia é consultada na data atual. Use dados saneados "
                    "e autorize o envio."
                )
                st.caption(
                    "O histórico GraphRAG não é persistido e fica somente nesta sessão."
                )
            else:
                st.warning(
                    "KAG monta fatos, relações temporais, caminhos e chunks antes "
                    "de enviar um pacote de conhecimento ao Ollama. Use somente "
                    "dados saneados e autorize explicitamente o envio."
                )
            rag_authorized = st.checkbox(
                "Entendi e autorizo o envio da pergunta, IDs, caminho e trechos recuperados.",
                key="remote_evidence_consent",
            )
        if mode == "Conversa direta":
            st.checkbox(
                "Consultar o banco Neo4j nesta conversa",
                key="direct_database_consultation",
                help="Usa o fluxo KAG parametrizado para perguntas hierárquicas.",
            )
            if st.session_state.direct_database_consultation:
                st.warning(
                    "A consulta envia pergunta, IDs, relações temporais e chunks "
                    "ao Ollama via Tailscale. Autorize o envio explicitamente."
                )
                rag_authorized = st.checkbox(
                    "Entendi e autorizo o envio das evidências recuperadas.",
                    key="direct_database_consent",
                )
            else:
                rag_authorized = True
        history_mode = (
            "direct" if mode == "Conversa direta" else "RAG" if mode == "RAG" else None
        )
        id_key: str | None = None
        message_key: str | None = None
        if history_mode is not None:
            id_key = (
                "direct_conversation_id"
                if history_mode == "direct"
                else "rag_conversation_id"
            )
            message_key = "messages" if history_mode == "direct" else "rag_messages"
            history_key = f"history_selection_{history_mode}"
            if store is not None:
                try:
                    saved_conversations = store.list_conversations()
                except (ConversationStoreError, OSError) as exc:
                    store = None
                    store_error = str(exc)
                    saved_conversations = []
            else:
                saved_conversations = []

            mode_conversations = [
                saved
                for saved in saved_conversations
                if any(message.mode == history_mode for message in saved.messages)
            ]
            if st.session_state[id_key] is None:
                latest = mode_conversations[0] if mode_conversations else None
                st.session_state[id_key] = (
                    latest.conversation_id if latest is not None else ""
                )
                if latest is not None:
                    restore_conversation(history_mode, latest)
            options = [""] + [saved.conversation_id for saved in mode_conversations]
            if st.session_state[id_key] not in options:
                st.session_state[id_key] = ""
                st.session_state[message_key] = []
                if history_mode == "RAG":
                    st.session_state.rag_sources = []
                    st.session_state.rag_status = None
            if store is not None:
                reset_requested = st.session_state.pop(
                    f"new_conversation_requested_{history_mode}", False
                )
                selection_sync_requested = st.session_state.pop(
                    f"history_selection_sync_{history_mode}", False
                )
                if reset_requested or selection_sync_requested:
                    st.session_state[history_key] = st.session_state[id_key]
                if st.session_state.get(history_key) not in options:
                    st.session_state[history_key] = st.session_state[id_key]
                labels = {
                    saved.conversation_id: (
                        f"{saved.title} · "
                        f"{saved.updated_at.astimezone().strftime('%d/%m %H:%M')}"
                    )
                    for saved in mode_conversations
                }
                selected_id = st.selectbox(
                    "Conversas salvas",
                    options,
                    key=history_key,
                    format_func=lambda value: labels.get(value, "Nova conversa"),
                )
                if selected_id != st.session_state[id_key]:
                    st.session_state[id_key] = selected_id
                    if selected_id:
                        selected_conversation = next(
                            saved
                            for saved in mode_conversations
                            if saved.conversation_id == selected_id
                        )
                        restore_conversation(history_mode, selected_conversation)
                    else:
                        st.session_state[message_key] = []
                        if history_mode == "RAG":
                            st.session_state.rag_sources = []
                            st.session_state.rag_status = None
            elif store_error:
                st.warning(
                    "Persistência local indisponível. Esta conversa ficará "
                    f"somente nesta sessão: {store_error}"
                )
        theme = st.radio("Tema", ("Claro", "Escuro"), horizontal=True)
        st.divider()
        if st.button("Nova conversa", use_container_width=True):
            if history_mode == "direct":
                st.session_state.direct_conversation_id = ""
                st.session_state.messages = []
            elif history_mode == "RAG":
                st.session_state.rag_conversation_id = ""
                st.session_state.rag_messages = []
                st.session_state.rag_sources = []
                st.session_state.rag_status = None
            elif mode == "GraphRAG":
                st.session_state.graphrag_messages = []
                st.session_state.graphrag_sources = []
                st.session_state.graphrag_paths = []
                st.session_state.graphrag_cypher = ""
                st.session_state.graphrag_status = None
                st.session_state.graphrag_result_as_of = None
            if history_mode is not None:
                st.session_state[f"new_conversation_requested_{history_mode}"] = True
            st.rerun()
        if (
            history_mode is not None
            and store is not None
            and id_key is not None
            and message_key is not None
            and st.session_state[id_key]
            and st.button("Excluir conversa atual", use_container_width=True)
        ):
            try:
                store.delete_conversation(st.session_state[id_key])
            except (ConversationStoreError, OSError) as exc:
                st.error(f"Não foi possível excluir a conversa: {exc}")
            else:
                st.session_state[id_key] = ""
                st.session_state[message_key] = []
                if history_mode == "RAG":
                    st.session_state.rag_sources = []
                    st.session_state.rag_status = None
                st.rerun()
        st.caption("Não envie dados pessoais ou informações sensíveis.")

    apply_theme(theme)
    conversation, evidence = st.columns([2.4, 1], gap="large")

    with conversation:
        st.title("Converse com a IA")
        st.caption("Assistente de conhecimento de RH da NovaTech.")
        if mode == "Conversa direta":
            if st.session_state.direct_database_consultation:
                st.info(
                    "Conversa direta com KAG: a resposta usa exclusivamente fatos, "
                    "relações temporais e chunks recuperados do Neo4j."
                )
            else:
                st.info(
                    "Conversa de teste: sem consulta ao Neo4j ou aos documentos. "
                    "As respostas não são fundamentadas no dataset."
                )
            visible_messages = st.session_state.messages
        elif mode == "RAG":
            st.info(
                "RAG responde somente com evidências documentais recuperadas. "
                "O histórico não é enviado como evidência."
            )
            visible_messages = st.session_state.rag_messages
        elif mode == "GraphRAG":
            st.info(
                "GraphRAG percorre relações hierárquicas válidas na data atual "
                "e combina o caminho com chunks recuperados. O histórico não é evidência."
            )
            visible_messages = st.session_state.graphrag_messages
        else:
            st.info(
                "KAG constrói um pacote explícito de fatos, relações temporais "
                "e trechos documentais antes da geração."
            )
            visible_messages = st.session_state.graphrag_messages

        history = st.container(height=400, border=False)
        with history:
            welcome = st.empty()
            if not visible_messages:
                with welcome.container(border=True):
                    st.subheader("Como posso ajudar?")
                    if mode == "RAG":
                        st.write("Faça uma pergunta sobre os documentos de RH da NovaTech.")
                    elif mode == "GraphRAG":
                        st.write(
                            "Pergunte pelos subordinados de uma pessoa usando o nome "
                            "canônico completo."
                        )
                    else:
                        st.write("Faça uma pergunta para testar a conversa.")

            for message in visible_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        prompt = st.chat_input(
            "Escreva sua mensagem",
            disabled=(
                (
                    mode in {"RAG", "GraphRAG", "KAG"}
                    or (
                        mode == "Conversa direta"
                        and st.session_state.direct_database_consultation
                    )
                )
                and not rag_authorized
            ),
        )
        if prompt:
            welcome.empty()
            user_message: ChatMessage = {"role": "user", "content": prompt}
            with history:
                with st.chat_message("user"):
                    st.markdown(prompt)
                if mode == "Conversa direta":
                    if st.session_state.direct_database_consultation:
                        try:
                            with st.spinner(
                                "Consultando KAG no Neo4j e preparando evidências..."
                            ):
                                response = answer_with_kag(
                                    prompt,
                                    document_scope="ORGANIZACAO",
                                    as_of=date.today(),
                                )
                        except (LLMConfigurationError, LLMRequestError) as exc:
                            st.error(f"Não foi possível consultar o Ollama: {exc}")
                        except Neo4jConfigurationError as exc:
                            st.error(f"Não foi possível configurar o Neo4j: {exc}")
                        except Neo4jError as exc:
                            st.error(f"Falha ao consultar o Neo4j: {exc}")
                        except GraphSearchError as exc:
                            st.error(f"Falha ao resolver ou percorrer o grafo: {exc}")
                        except GraphRAGResponseError as exc:
                            st.error(f"A resposta do KAG foi inválida: {exc}")
                        else:
                            answer = response.answer
                            st.session_state.graphrag_sources = response.sources
                            st.session_state.graphrag_paths = response.paths
                            st.session_state.graphrag_cypher = response.cypher
                            st.session_state.graphrag_status = response.status
                            st.session_state.graphrag_result_as_of = response.as_of
                    else:
                        try:
                            with st.spinner("Aguardando resposta do modelo..."):
                                client = OllamaClient.from_env()
                                answer = client.chat(
                                    [
                                        SYSTEM_MESSAGE,
                                        *st.session_state.messages,
                                        user_message,
                                    ]
                                )
                        except (LLMConfigurationError, LLMRequestError) as exc:
                            st.error(f"Não foi possível obter uma resposta: {exc}")
                            answer = ""
                    if "answer" in locals() and answer: # type: ignore
                        assistant_message: ChatMessage = {
                            "role": "assistant",
                            "content": answer,
                        }
                        if store is not None:
                            try:
                                saved = store.save_exchange(
                                    st.session_state.direct_conversation_id or None,
                                    "direct",
                                    prompt,
                                    answer,
                                )
                            except (ConversationStoreError, OSError) as exc:
                                st.error(
                                    "A resposta foi recebida, mas não foi possível "
                                    f"salvar a conversa: {exc}"
                                )
                            else:
                                st.session_state.direct_conversation_id = (
                                    saved.conversation_id
                                )
                                st.session_state.history_selection_sync_direct = True
                        st.session_state.messages.extend(
                            [user_message, assistant_message]
                        )
                        with st.chat_message("assistant"):
                            st.markdown(answer)
                elif mode == "RAG":
                    st.session_state.rag_sources = []
                    st.session_state.rag_status = None
                    try:
                        with st.spinner("Buscando evidências e consultando o modelo..."):
                            response = answer_with_rag(prompt, document_scope)
                    except (LLMConfigurationError, LLMRequestError) as exc:
                        st.error(f"Não foi possível consultar o Ollama: {exc}")
                    except Neo4jConfigurationError as exc:
                        st.error(f"Não foi possível configurar o Neo4j: {exc}")
                    except Neo4jError as exc:
                        st.error(f"Falha ao consultar o Neo4j: {exc}")
                    except RAGResponseError as exc:
                        st.error(f"A resposta do RAG foi inválida: {exc}")
                    except RuntimeError as exc:
                        st.error(f"Não foi possível executar o pipeline RAG: {exc}")
                    else:
                        assistant_message = {
                            "role": "assistant",
                            "content": response.answer,
                        }
                        if store is not None:
                            try:
                                saved = store.save_exchange(
                                    st.session_state.rag_conversation_id or None,
                                    "RAG",
                                    prompt,
                                    response.answer,
                                    sources=tuple(
                                        StoredSource(
                                            doc_id=source.doc_id,
                                            chunk_id=source.chunk_id,
                                            quote=source.text,
                                            vector_score=source.vector_score,
                                            fulltext_score=source.fulltext_score,
                                            rank_score=source.rank_score,
                                        )
                                        for source in response.sources
                                    ),
                                    status=response.status,
                                )
                            except (ConversationStoreError, OSError) as exc:
                                st.error(
                                    "A resposta foi recebida, mas não foi possível "
                                    f"salvar a conversa: {exc}"
                                )
                            else:
                                st.session_state.rag_conversation_id = (
                                    saved.conversation_id
                                )
                                st.session_state.history_selection_sync_RAG = True
                        st.session_state.rag_messages.extend(
                            [user_message, assistant_message]
                        )
                        st.session_state.rag_sources = response.sources
                        st.session_state.rag_status = response.status
                        with st.chat_message("assistant"):
                            st.markdown(response.answer)
                elif mode == "GraphRAG":
                    st.session_state.graphrag_sources = []
                    st.session_state.graphrag_paths = []
                    st.session_state.graphrag_cypher = ""
                    st.session_state.graphrag_status = None
                    st.session_state.graphrag_result_as_of = None
                    try:
                        with st.spinner(
                            "Percorrendo o grafo e buscando evidências documentais..."
                        ):
                            response = answer_with_graphrag(
                                prompt,
                                document_scope,
                                date.today(),
                            )
                    except (LLMConfigurationError, LLMRequestError) as exc:
                        st.error(f"Não foi possível consultar o Ollama: {exc}")
                    except Neo4jConfigurationError as exc:
                        st.error(f"Não foi possível configurar o Neo4j: {exc}")
                    except Neo4jError as exc:
                        st.error(f"Falha ao consultar o Neo4j: {exc}")
                    except GraphSearchError as exc:
                        st.error(f"Falha ao resolver ou percorrer o grafo: {exc}")
                    except GraphRAGResponseError as exc:
                        st.error(f"A resposta do GraphRAG foi inválida: {exc}")
                    except RuntimeError as exc:
                        st.error(f"Não foi possível executar o GraphRAG: {exc}")
                    else:
                        assistant_message = {
                            "role": "assistant",
                            "content": response.answer,
                        }
                        st.session_state.graphrag_messages.extend(
                            [user_message, assistant_message]
                        )
                        st.session_state.graphrag_sources = response.sources
                        st.session_state.graphrag_paths = response.paths
                        st.session_state.graphrag_cypher = response.cypher
                        st.session_state.graphrag_status = response.status
                        st.session_state.graphrag_result_as_of = response.as_of
                        with st.chat_message("assistant"):
                            st.markdown(response.answer)
                elif mode == "KAG":
                    st.session_state.graphrag_sources = []
                    st.session_state.graphrag_paths = []
                    st.session_state.graphrag_cypher = ""
                    st.session_state.graphrag_status = None
                    st.session_state.graphrag_result_as_of = None
                    try:
                        with st.spinner(
                            "Montando pacote KAG e consultando evidências..."
                        ):
                            response = answer_with_kag(
                                prompt,
                                document_scope,
                                date.today(),
                            )
                    except (LLMConfigurationError, LLMRequestError) as exc:
                        st.error(f"Não foi possível consultar o Ollama: {exc}")
                    except Neo4jConfigurationError as exc:
                        st.error(f"Não foi possível configurar o Neo4j: {exc}")
                    except Neo4jError as exc:
                        st.error(f"Falha ao consultar o Neo4j: {exc}")
                    except GraphSearchError as exc:
                        st.error(f"Falha ao resolver ou percorrer o grafo: {exc}")
                    except GraphRAGResponseError as exc:
                        st.error(f"A resposta do KAG foi inválida: {exc}")
                    except RuntimeError as exc:
                        st.error(f"Não foi possível executar o KAG: {exc}")
                    else:
                        assistant_message = {
                            "role": "assistant",
                            "content": response.answer,
                        }
                        st.session_state.graphrag_messages.extend(
                            [user_message, assistant_message]
                        )
                        st.session_state.graphrag_sources = response.sources
                        st.session_state.graphrag_paths = response.paths
                        st.session_state.graphrag_cypher = response.cypher
                        st.session_state.graphrag_status = response.status
                        st.session_state.graphrag_result_as_of = response.as_of
                        with st.chat_message("assistant"):
                            st.markdown(response.answer)

    with evidence:
        st.subheader("Documentos e referências")
        if mode == "RAG":
            st.caption("Fontes citadas pela resposta RAG mais recente")
            if st.session_state.rag_sources:
                for source in st.session_state.rag_sources:
                    with st.container(border=True):
                        st.markdown(f"**{source.doc_id} · {source.chunk_id}**")
                        st.caption("Chunk recuperado completo")
                        st.write(source.text)
                        with st.expander("Sinais de recuperação"):
                            st.write(f"Similaridade vetorial: {source.vector_score}")
                            st.write(f"Score full-text: {source.fulltext_score}")
                            st.write(f"RRF rank score: {source.rank_score:.6f}")
            elif st.session_state.rag_status == "insufficient":
                st.info("A resposta mais recente não citou fontes: evidência insuficiente.")
            else:
                st.write("Nenhuma busca RAG foi concluída nesta conversa.")
        elif mode in {"GraphRAG", "KAG"} or (
            mode == "Conversa direta"
            and st.session_state.direct_database_consultation
        ):
            st.caption("Chunks citados pela resposta KAG/GraphRAG mais recente")
            if st.session_state.graphrag_sources:
                for source in st.session_state.graphrag_sources:
                    with st.container(border=True):
                        st.markdown(f"**{source.doc_id} · {source.chunk_id}**")
                        st.caption(
                            "Origem: " + ", ".join(source.origins)
                        )
                        st.write(source.text)
                        with st.expander("Sinais de recuperação"):
                            st.write(f"Similaridade vetorial: {source.vector_score}")
                            st.write(f"Score full-text: {source.fulltext_score}")
                            st.write(f"RRF rank score: {source.rank_score}")
            elif st.session_state.graphrag_status == "insufficient":
                st.info(
                    "A resposta GraphRAG mais recente não citou chunks documentais."
                )
            else:
                st.write("Nenhuma busca GraphRAG foi concluída nesta sessão.")
        else:
            with st.container(border=True):
                st.write("Nenhuma busca documental foi realizada.")
                st.caption(
                    "Selecione RAG para consultar documentos e exibir as fontes."
                )
        st.subheader("Caminho no grafo")
        if (
            mode in {"GraphRAG", "KAG"}
            or (
                mode == "Conversa direta"
                and st.session_state.direct_database_consultation
            )
        ) and st.session_state.graphrag_paths:
            for path_number, path in enumerate(
                st.session_state.graphrag_paths,
                start=1,
            ):
                with st.container(border=True):
                    st.markdown(f"**Caminho {path_number}**")
                    st.write(
                        " → ".join(
                            f"{person.nome} ({person.funcionario_id})"
                            for person in path.people
                        )
                    )
                    for relationship in path.relationships:
                        valid_to = (
                            relationship.valid_to.isoformat()
                            if relationship.valid_to is not None
                            else "sem data final"
                        )
                        st.caption(
                            f"{relationship.source_id} -REPORTS_TO→ "
                            f"{relationship.target_id} · {relationship.relacao_id} · "
                            f"{relationship.valid_from.isoformat()} a {valid_to}"
                        )
                        if relationship.source_references:
                            st.caption(
                                "Referências: "
                                + ", ".join(relationship.source_references)
                            )
        else:
            with st.container(border=True):
                st.write("Nenhum caminho GraphRAG foi recuperado.")
                st.caption(
                    "A busca estruturada hierárquica está disponível no modo "
                    "GraphRAG; o RAG documental não percorre relações."
                )
        with st.expander("Consulta e detalhes da busca"):
            if mode == "RAG":
                scope_label = (
                    "todos os documentos"
                    if document_scope is None
                    else document_scope
                )
                st.caption(
                    "Recuperação híbrida pelos índices vetorial e full-text "
                    "com Cypher 25 SEARCH. "
                    f"Escopo explícito: {scope_label}. "
                    "Scores são sinais de recuperação, não confiança."
                )
            elif mode in {"GraphRAG", "KAG"} or (
                mode == "Conversa direta"
                and st.session_state.direct_database_consultation
            ):
                result_as_of = (
                    st.session_state.graphrag_result_as_of
                    or date.today()
                )
                st.caption(
                    "Travessia hierárquica temporal e pacote KAG explícito. A data inclusiva usada "
                    f"foi {result_as_of.isoformat()}. "
                    "Os resultados não incluem score de confiança."
                )
                if st.session_state.graphrag_cypher:
                    st.code(st.session_state.graphrag_cypher, language="cypher")
            else:
                st.caption(
                    "RAG e KAG usam consultas de recuperação validadas. "
                    "Nenhum Cypher é gerada pelo modelo."
                )


if __name__ == "__main__":
    main()
