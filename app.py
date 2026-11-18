"""Interface Streamlit do Sistema de RH Inteligente."""

from __future__ import annotations

import streamlit as st
from neo4j.exceptions import Neo4jError

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
from src.rag.pipeline import RAGPipeline, RAGResponse, RAGResponseError
from src.retrieval.vector_search import (
    EmbeddingModel,
    SearchClauseVectorSearch,
    load_embedding_model,
)


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
        search = SearchClauseVectorSearch(database, _cached_embedding_model())
        return RAGPipeline(search, OllamaClient.from_env()).answer(
            question,
            document_scope=document_scope,
        )


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
            help="RAG está disponível; GraphRAG e KAG ainda estão em desenvolvimento.",
            filter_mode=None,
        )
        document_scope: str | None = "ORGANIZACAO"
        rag_authorized = False
        if mode == "RAG":
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
            st.warning(
                "RAG envia sua pergunta, doc_id/chunk_id e trechos recuperados "
                "ao Ollama via Tailscale. Use somente dados saneados; no escopo "
                "PESSOA, confirme que o envio está autorizado."
            )
            rag_authorized = st.checkbox(
                "Entendi e autorizo o envio da pergunta, IDs e trechos recuperados.",
                key="rag_remote_consent",
            )
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
        else:
            st.warning(
                f"{mode} ainda não está disponível; nenhuma busca será executada."
            )
            visible_messages = []

        history = st.container(height=400, border=False)
        with history:
            welcome = st.empty()
            if not visible_messages:
                with welcome.container(border=True):
                    st.subheader("Como posso ajudar?")
                    if mode == "RAG":
                        st.write("Faça uma pergunta sobre os documentos de RH da NovaTech.")
                    else:
                        st.write("Faça uma pergunta para testar a conversa.")

            for message in visible_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        prompt = st.chat_input(
            "Escreva sua mensagem",
            disabled=(
                mode in {"GraphRAG", "KAG"}
                or (mode == "RAG" and not rag_authorized)
            ),
        )
        if prompt:
            welcome.empty()
            user_message: ChatMessage = {"role": "user", "content": prompt}
            with history:
                with st.chat_message("user"):
                    st.markdown(prompt)
                if mode == "Conversa direta":
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
                    else:
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
                                            quote=source.quote,
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

    with evidence:
        st.subheader("Documentos e referências")
        if mode == "RAG":
            st.caption("Fontes citadas pela resposta RAG mais recente")
            if st.session_state.rag_sources:
                for source in st.session_state.rag_sources:
                    with st.container(border=True):
                        st.markdown(f"**{source.doc_id} · {source.chunk_id}**")
                        st.write(source.quote)
                        with st.expander("Sinais de recuperação"):
                            st.write(f"Similaridade vetorial: {source.vector_score}")
                            st.write(f"Score full-text: {source.fulltext_score}")
                            st.write(f"RRF rank score: {source.rank_score:.6f}")
            elif st.session_state.rag_status == "insufficient":
                st.info("A resposta mais recente não citou fontes: evidência insuficiente.")
            else:
                st.write("Nenhuma busca RAG foi concluída nesta conversa.")
        else:
            with st.container(border=True):
                st.write("Nenhuma busca documental foi realizada.")
                st.caption(
                    "Selecione RAG para consultar documentos e exibir as fontes."
                )
        st.subheader("Caminho no grafo")
        with st.container(border=True):
            st.write("Visualização do caminho indisponível.")
            st.caption(
                "O RAG recupera trechos documentais, mas não percorre relações. "
                "GraphRAG/KAG ainda não estão disponíveis."
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
            else:
                st.caption(
                    "GraphRAG/KAG e consultas estruturadas de relações "
                    "ainda não estão disponíveis."
                )


if __name__ == "__main__":
    main()
