"""Interface Streamlit do Sistema de RH Inteligente."""

from __future__ import annotations

import streamlit as st

from src.generation.llm import (
    ChatMessage,
    LLMConfigurationError,
    LLMRequestError,
    OllamaClient,
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

    with st.sidebar:
        st.markdown("### ◈ RH Inteligente")
        st.caption("NovaTech · assistente de conhecimento")
        st.divider()
        st.markdown("**Área de trabalho**")
        st.caption("Conversa")
        mode = st.selectbox(
            "Modo de busca",
            ("Conversa direta", "RAG", "GraphRAG", "KAG"),
            help="Os modos de recuperação ainda não estão implementados.",
            filter_mode=None,
        )
        theme = st.radio("Tema", ("Claro", "Escuro"), horizontal=True)
        st.divider()
        if st.button("Nova conversa", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        st.caption("Não envie dados pessoais ou informações sensíveis.")

    apply_theme(theme)
    conversation, evidence = st.columns([2.4, 1], gap="large")

    with conversation:
        st.title("Converse com a IA")
        st.caption("Um espaço para testar a interação com o modelo local.")
        if mode == "Conversa direta":
            st.info(
                "Conversa de teste: sem consulta ao Neo4j ou aos documentos. "
                "As respostas não são fundamentadas no dataset."
            )
        else:
            st.warning(
                f"{mode} ainda não está disponível. Selecione Conversa direta "
                "para falar com o modelo; nenhuma busca será executada."
            )

        history = st.container(height=400, border=False)
        with history:
            welcome = st.empty()
            if not st.session_state.messages:
                with welcome.container(border=True):
                    st.subheader("Como posso ajudar?")
                    st.write(
                        "Faça uma pergunta para testar a conversa. Quando a busca "
                        "estiver integrada, as fontes aparecerão ao lado."
                    )

            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        prompt = st.chat_input(
            "Escreva sua mensagem",
            disabled=mode != "Conversa direta",
        )
        if prompt:
            welcome.empty()
            user_message: ChatMessage = {"role": "user", "content": prompt}
            with history:
                with st.chat_message("user"):
                    st.markdown(prompt)
                try:
                    with st.spinner("Aguardando resposta do modelo..."):
                        client = OllamaClient.from_env()
                        answer = client.chat(
                            [SYSTEM_MESSAGE, *st.session_state.messages, user_message]
                        )
                except (LLMConfigurationError, LLMRequestError) as exc:
                    st.error(f"Não foi possível obter uma resposta: {exc}")
                else:
                    assistant_message: ChatMessage = {
                        "role": "assistant",
                        "content": answer,
                    }
                    st.session_state.messages.extend([user_message, assistant_message])
                    with st.chat_message("assistant"):
                        st.markdown(answer)

    with evidence:
        st.subheader("Documentos e referências")
        st.caption("Evidências da última busca")
        with st.container(border=True):
            st.write("Nenhuma busca documental foi realizada.")
            st.caption(
                "Quando a recuperação estiver disponível, documentos, "
                "doc_id e chunk_id aparecerão aqui."
            )
        st.subheader("Caminho no grafo")
        with st.container(border=True):
            st.write("Visualização do caminho indisponível.")
            st.caption(
                "Espaço reservado para nós e relações recuperados pelo "
                "GraphRAG/KAG. Nenhuma consulta ao grafo foi executada."
            )
        with st.expander("Consulta e detalhes da busca"):
            st.caption(
                "Espaço reservado para a consulta validada e os detalhes da "
                "recuperação. Nenhuma consulta está disponível nesta etapa."
            )


if __name__ == "__main__":
    main()
