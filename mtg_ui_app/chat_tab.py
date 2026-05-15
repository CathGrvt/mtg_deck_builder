from __future__ import annotations

import os
import uuid
from typing import Optional

import streamlit as st

from research_pipeline.retrieval.index import HybridRetrievalIndex

from mtg_ui_app.backend_client import (
    build_chat_backend_payload,
    call_backend_json,
)
from mtg_ui_app.shared import (
    build_research_index_with_feedback,
    ensure_research_paths,
    generate_chatbot_answer,
    parse_path_list,
    render_chat_evidence,
)


def render_chat_tab() -> None:
    st.subheader("RAG Chatbot")
    st.caption("Ask MTG questions grounded in your local corpus (decks + cards + meta JSON).")

    chat_settings_col1, chat_settings_col2 = st.columns(2)
    with chat_settings_col1:
        chat_cards_path = st.text_input(
            "Cards CSV",
            value="data/commander_cards.csv",
            key="chat_cards_path",
        )
        chat_decks_path = st.text_input(
            "Deck Directory",
            value="current_commander_decks",
            key="chat_decks_path",
        )
        chat_meta_paths_raw = st.text_area(
            "Meta JSON Paths (optional)",
            value="",
            key="chat_meta_paths",
            help="Comma or newline separated file paths. Leave empty to auto-discover json_outputs/*.json.",
            height=80,
        )

    with chat_settings_col2:
        chat_top_k = st.number_input(
            "Retriever Top-K",
            min_value=1,
            max_value=20,
            value=6,
            step=1,
            key="chat_top_k",
        )
        chat_enable_semantic = st.checkbox(
            "Enable Semantic Retrieval",
            value=False,
            key="chat_enable_semantic",
        )
        chat_lexical_weight = st.slider(
            "Lexical Retrieval Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            key="chat_lexical_weight",
        )
        chat_semantic_weight = st.slider(
            "Semantic Retrieval Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            key="chat_semantic_weight",
        )
        chat_use_backend = st.checkbox(
            "Use Deployed Agent Backend",
            value=False,
            key="chat_use_backend",
            help="Calls backend chat endpoint before local pipeline fallback.",
        )
        chat_backend_url = st.text_input(
            "Backend Chat URL",
            value=os.getenv("MTG_GCP_CHAT_URL", "http://localhost:8080/v1/chat/respond"),
            key="chat_backend_url",
            disabled=not chat_use_backend,
        )
        chat_backend_timeout = st.number_input(
            "Backend Timeout Seconds",
            min_value=5,
            max_value=240,
            value=60,
            step=1,
            key="chat_backend_timeout",
            disabled=not chat_use_backend,
        )

    with st.expander("Chat LLM Settings (Optional)", expanded=False):
        cc1, cc2 = st.columns(2)
        with cc1:
            chat_use_llm = st.checkbox(
                "Use OpenAI synthesis",
                value=True,
                key="chat_use_llm",
                help="If disabled or no API key is present, chatbot uses rule-based synthesis.",
            )
            chat_llm_model = st.text_input(
                "LLM Model",
                value="gpt-4o-mini",
                key="chat_llm_model",
            )
            chat_llm_api_key_env = st.text_input(
                "API Key Env Var",
                value="OPENAI_API_KEY",
                key="chat_llm_api_key_env",
            )
        with cc2:
            chat_llm_base_url = st.text_input(
                "Base URL",
                value="https://api.openai.com/v1",
                key="chat_llm_base_url",
            )
            chat_llm_timeout = st.number_input(
                "Timeout Seconds",
                min_value=5,
                max_value=180,
                value=45,
                step=1,
                key="chat_llm_timeout",
            )
            chat_llm_temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.2,
                step=0.05,
                key="chat_llm_temperature",
            )

    clear_chat = st.button("Clear Chat History", key="clear_chat")

    if "mtg_chat_messages" not in st.session_state:
        st.session_state["mtg_chat_messages"] = []
    if clear_chat:
        st.session_state["mtg_chat_messages"] = []
        st.session_state.pop("mtg_chat_session_id", None)

    chat_index: Optional[HybridRetrievalIndex] = None
    parsed_chat_meta_paths = parse_path_list(chat_meta_paths_raw)

    if ensure_research_paths(chat_cards_path, chat_decks_path, emit_error=False):
        chat_index_bundle = build_research_index_with_feedback(
            cards_path=chat_cards_path,
            decks_path=chat_decks_path,
            meta_paths=parsed_chat_meta_paths,
            enable_semantic=chat_enable_semantic,
            lexical_weight=float(chat_lexical_weight),
            semantic_weight=float(chat_semantic_weight),
            spinner_message="Preparing chatbot index...",
        )
        if chat_index_bundle is not None:
            chat_index, chat_chunk_count, chat_source_count = chat_index_bundle
            st.caption(f"Corpus loaded: {chat_chunk_count} chunks from {chat_source_count} sources.")

    for msg in st.session_state["mtg_chat_messages"]:
        with st.chat_message(msg.get("role", "assistant")):
            st.markdown(msg.get("content", ""))
            evidence = msg.get("evidence", [])
            if evidence:
                with st.expander("Evidence", expanded=False):
                    render_chat_evidence(evidence=evidence, max_items=8)

    user_prompt = st.chat_input("Ask a question about MTG decks, cards, or meta trends...")
    if not user_prompt:
        return

    st.session_state["mtg_chat_messages"].append(
        {
            "role": "user",
            "content": user_prompt,
        }
    )

    history_before_answer = st.session_state["mtg_chat_messages"][:-1]

    if chat_use_backend:
        backend_payload = build_chat_backend_payload(
            session_id=st.session_state.get("mtg_chat_session_id", f"chat-{uuid.uuid4().hex[:12]}"),
            question=user_prompt,
            history=history_before_answer,
            top_k=int(chat_top_k),
        )
        st.session_state["mtg_chat_session_id"] = backend_payload["session_id"]
        try:
            with st.spinner("Calling deployed backend chat endpoint..."):
                backend_response = call_backend_json(
                    backend_url=chat_backend_url,
                    payload=backend_payload,
                    timeout_sec=int(chat_backend_timeout),
                )
        except Exception as e:
            st.warning(f"Backend chat call failed ({e}). Falling back to local chat pipeline.")
        else:
            answer_text = str(backend_response.get("answer", "")).strip() or "(No answer)"
            evidence = backend_response.get("evidence", [])
            st.session_state["mtg_chat_messages"].append(
                {
                    "role": "assistant",
                    "content": answer_text,
                    "evidence": evidence if isinstance(evidence, list) else [],
                }
            )
            st.rerun()

    if chat_index is None:
        st.error("Chatbot index is not available. Fix paths/settings first.")
        st.stop()

    with st.spinner("Generating answer..."):
        answer_text, evidence = generate_chatbot_answer(
            question=user_prompt,
            index=chat_index,
            top_k=int(chat_top_k),
            history=history_before_answer,
            use_llm=bool(chat_use_llm),
            llm_model=chat_llm_model,
            llm_api_key_env=chat_llm_api_key_env,
            llm_base_url=chat_llm_base_url,
            llm_timeout_sec=int(chat_llm_timeout),
            llm_temperature=float(chat_llm_temperature),
        )

    st.session_state["mtg_chat_messages"].append(
        {
            "role": "assistant",
            "content": answer_text,
            "evidence": evidence,
        }
    )
    st.rerun()
