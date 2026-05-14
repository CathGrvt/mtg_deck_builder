from __future__ import annotations

import os
import uuid
from collections import Counter
from typing import Dict, List

import streamlit as st

from ai_deck_generator import (
    DeckSpec,
    LLMRerankConfig,
    format_decklist,
    generate_deck_from_meta,
    parse_colors,
    summarize_deck,
)
from ui_helpers import parse_card_list

from mtg_ui_app.shared import (
    call_deployed_recommendation,
    cached_load_card_db,
    cached_load_cluster_model,
    cached_load_decklists,
    model_is_available,
)


def render_generate_tab() -> None:
    with st.sidebar:
        st.header("Generation Inputs")
        cards_path = st.text_input("Cards CSV", value="data/commander_cards.csv")
        decks_path = st.text_input("Training Deck Directory", value="current_commander_decks")
        cluster_model_prefix = st.text_input("Cluster Model Prefix", value="models/deck_kmeans")
        use_cluster_model = st.checkbox(
            "Use Cluster Model",
            value=True,
            help="Requires <prefix>.npz and <prefix>_meta.json.",
        )
        use_deployed_backend = st.checkbox(
            "Use Deployed Agent Backend",
            value=False,
            help="Calls backend adapter API contract before local generation fallback.",
        )
        backend_url = st.text_input(
            "Backend Recommendation URL",
            value=os.getenv("MTG_GCP_BACKEND_URL", "http://localhost:8080/v1/deck/recommend"),
            disabled=not use_deployed_backend,
        )
        backend_timeout = st.number_input(
            "Backend Timeout Seconds",
            min_value=5,
            max_value=180,
            value=60,
            step=1,
            disabled=not use_deployed_backend,
        )
        save_output = st.checkbox("Save Generated Decklist", value=False)
        output_path = st.text_input("Output File", value="generated_decks/ui_generated_deck.txt")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Deck Spec")
        deck_format = st.selectbox(
            "Format",
            options=["commander", "standard", "pioneer", "modern", "legacy", "vintage"],
            index=0,
        )
        color_input = st.text_input("Colors", value="WR", help="Examples: WR, W,U, W U. Leave blank for any.")
        archetype = st.selectbox(
            "Archetype Hint",
            options=["", "aggro", "midrange", "control", "tempo", "combo"],
            index=0,
        )
        target_size = st.number_input(
            "Target Size",
            min_value=40,
            max_value=250,
            value=100,
            step=1,
        )
        land_ratio = st.slider(
            "Land Ratio",
            min_value=0.20,
            max_value=0.60,
            value=0.42,
            step=0.01,
        )
        recommendation_goal = st.text_area(
            "Recommendation Goal (for backend mode)",
            value="Build a high-synergy and resilient deck recommendation with grounded rationale.",
            height=100,
        )
        seed_enabled = st.checkbox("Set Random Seed", value=True)
        seed = st.number_input(
            "Seed",
            min_value=0,
            max_value=1_000_000,
            value=42,
            step=1,
            disabled=not seed_enabled,
        )

    with col2:
        st.subheader("Card Controls")
        include_raw = st.text_area(
            "Include Cards",
            value="",
            help="Comma or newline separated.",
            placeholder="Sol Ring\nArcane Signet",
            height=120,
        )
        exclude_raw = st.text_area(
            "Exclude Cards",
            value="",
            help="Comma or newline separated.",
            placeholder="Mana Crypt",
            height=120,
        )

        st.subheader("Model Weights")
        cluster_strength = st.slider("Cluster Strength", min_value=0.0, max_value=4.0, value=2.0, step=0.1)
        semantic_strength = st.slider("Semantic Strength", min_value=0.0, max_value=4.0, value=1.0, step=0.1)

        with st.expander("LLM Rerank (Optional)", expanded=False):
            enable_llm = st.checkbox("Enable LLM Rerank", value=False)
            llm_top_k = st.number_input(
                "Top-K Candidates",
                min_value=1,
                max_value=200,
                value=20,
                step=1,
                disabled=not enable_llm,
            )
            llm_strength = st.slider(
                "LLM Strength",
                min_value=0.0,
                max_value=3.0,
                value=0.8,
                step=0.1,
                disabled=not enable_llm,
            )
            llm_model = st.text_input("LLM Model", value="gpt-4o-mini", disabled=not enable_llm)
            llm_api_key_env = st.text_input("API Key Env Var", value="OPENAI_API_KEY", disabled=not enable_llm)
            llm_base_url = st.text_input("Base URL", value="https://api.openai.com/v1", disabled=not enable_llm)
            llm_timeout = st.number_input(
                "Timeout Seconds",
                min_value=5,
                max_value=180,
                value=45,
                step=1,
                disabled=not enable_llm,
            )

    generate_clicked = st.button("Generate Deck", type="primary")

    if not generate_clicked:
        return

    include_cards = parse_card_list(include_raw)
    exclude_cards = parse_card_list(exclude_raw)
    colors = parse_colors(color_input)

    if not os.path.isfile(cards_path):
        st.error(f"Cards CSV not found: {cards_path}")
        st.stop()

    with st.spinner("Loading card database..."):
        card_db = cached_load_card_db(cards_path)

    decklists: Dict[str, List[str]] = {}
    if os.path.isdir(decks_path):
        with st.spinner("Loading training decklists..."):
            decklists = cached_load_decklists(decks_path)
    else:
        st.warning(f"Training deck directory not found: {decks_path}. Continuing without training decks.")

    cluster_model = None
    if use_cluster_model:
        if not model_is_available(cluster_model_prefix):
            st.warning(
                "Cluster model files were not found. Expected:\n"
                f"- {cluster_model_prefix}.npz\n"
                f"- {cluster_model_prefix}_meta.json\n"
                "Continuing without cluster model."
            )
        else:
            with st.spinner("Loading cluster model..."):
                cluster_model = cached_load_cluster_model(cluster_model_prefix)
            if cluster_model is None:
                st.warning("Cluster model failed to load. Continuing without cluster model.")

    llm_cfg = None
    if enable_llm and llm_strength > 0.0:
        if not os.getenv(llm_api_key_env):
            st.warning(
                f"Env var '{llm_api_key_env}' is not set. "
                "LLM rerank disabled for this run."
            )
        else:
            llm_cfg = LLMRerankConfig(
                top_k=int(llm_top_k),
                strength=float(llm_strength),
                model=llm_model,
                api_key_env=llm_api_key_env,
                base_url=llm_base_url,
                timeout_sec=int(llm_timeout),
            )

    spec = DeckSpec(
        format=deck_format,
        colors=colors,
        archetype=archetype or None,
        target_size=int(target_size),
        land_ratio=float(land_ratio),
        include_cards=include_cards,
        exclude_cards=exclude_cards,
    )

    if use_deployed_backend:
        payload = {
            "session_id": f"ui-{uuid.uuid4().hex[:12]}",
            "user_query": recommendation_goal.strip() or "Generate a deck recommendation.",
            "format": deck_format,
            "colors": colors,
            "archetype_hint": archetype or None,
            "must_include": include_cards,
            "must_exclude": exclude_cards,
            "mode": "deck_recommendation",
        }
        try:
            with st.spinner("Calling deployed backend..."):
                backend_response = call_deployed_recommendation(
                    backend_url=backend_url,
                    payload=payload,
                    timeout_sec=int(backend_timeout),
                )
        except Exception as e:
            st.warning(f"Backend recommendation call failed ({e}). Falling back to local generation.")
        else:
            safety = backend_response.get("safety_verdict", {})
            if bool(safety.get("blocked", False)):
                st.error("Request blocked by backend safety policy.")
                st.json(backend_response, expanded=False)
                st.stop()

            deck = [str(item) for item in backend_response.get("recommended_decklist", []) if str(item).strip()]
            if deck:
                deck_counts = Counter(deck)
                deck_text = "\n".join(f"{count} {name}" for name, count in sorted(deck_counts.items()))
            else:
                deck_text = ""
            summary = str(backend_response.get("summary", "")).strip()
            key_claims = [str(item).strip() for item in backend_response.get("key_claims", []) if str(item).strip()]
            citations = backend_response.get("citations", [])

            st.success("Deck recommendation generated from deployed backend.")
            st.subheader("Summary")
            st.write(summary or "(No summary)")

            if key_claims:
                st.subheader("Key Claims")
                for idx, claim in enumerate(key_claims, start=1):
                    st.markdown(f"{idx}. {claim}")

            if citations:
                st.subheader("Citations")
                for citation in citations[:8]:
                    doc_id = citation.get("doc_id", "")
                    chunk_id = citation.get("chunk_id", "")
                    source = citation.get("source", "")
                    title = citation.get("title", "")
                    st.markdown(f"- `{doc_id}::{chunk_id}` ({source}) {title}")

            st.subheader("Decklist")
            st.code(deck_text or "(No decklist returned)", language="text")

            st.download_button(
                label="Download Decklist",
                data=(deck_text + "\n"),
                file_name="generated_deck.txt",
                mime="text/plain",
                key="download_backend_decklist",
            )
            if save_output and deck_text:
                out_dir = os.path.dirname(output_path)
                if out_dir:
                    os.makedirs(out_dir, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(deck_text + "\n")
                st.info(f"Saved to {output_path}")
            st.stop()

    try:
        with st.spinner("Generating deck..."):
            deck = generate_deck_from_meta(
                card_db=card_db,
                decklists=decklists,
                spec=spec,
                cluster_model=cluster_model,
                cluster_strength=float(cluster_strength),
                semantic_strength=float(semantic_strength),
                llm_rerank_config=llm_cfg,
                seed=int(seed) if seed_enabled else None,
            )
    except Exception as e:
        st.error(f"Generation failed: {e}")
        st.stop()

    summary = summarize_deck(deck, card_db)
    deck_text = format_decklist(deck, card_db)

    st.success("Deck generated.")
    st.subheader("Summary")
    st.text(summary)

    st.subheader("Decklist")
    st.code(deck_text, language="text")

    st.download_button(
        label="Download Decklist",
        data=(deck_text + "\n"),
        file_name="generated_deck.txt",
        mime="text/plain",
    )

    if save_output:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(deck_text + "\n")
        st.info(f"Saved to {output_path}")
