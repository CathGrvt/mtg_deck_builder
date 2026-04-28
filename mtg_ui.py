import json
import os
import shlex
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from ai_deck_generator import (
    DeckSpec,
    LLMRerankConfig,
    format_decklist,
    generate_deck_from_meta,
    load_cluster_model,
    parse_colors,
    summarize_deck,
)
from mtg_io import load_card_database, load_decklists_from_directory
from ui_helpers import parse_card_list


st.set_page_config(page_title="MTG Deck Builder UI", page_icon=":mage:", layout="wide")


@st.cache_data(show_spinner=False)
def cached_load_card_db(path: str) -> pd.DataFrame:
    return load_card_database(path)


@st.cache_data(show_spinner=False)
def cached_load_decklists(path: str) -> Dict[str, List[str]]:
    return load_decklists_from_directory(path, include_command_zone=True)


@st.cache_resource(show_spinner=False)
def cached_load_cluster_model(prefix: str) -> Optional[Dict]:
    return load_cluster_model(prefix)


def model_is_available(prefix: str) -> bool:
    return os.path.isfile(prefix + ".npz") and os.path.isfile(prefix + "_meta.json")


def run_cli_command(cmd: List[str]) -> Tuple[int, str]:
    """
    Run a CLI command and return (return_code, combined_output).
    """
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    output = ""
    if proc.stdout:
        output += proc.stdout
    if proc.stderr:
        output += ("\n" if output else "") + proc.stderr
    return proc.returncode, output.strip()


def render_command_result(title: str, cmd: List[str], return_code: int, output: str) -> None:
    st.markdown(f"**{title} Command**")
    st.code(" ".join(shlex.quote(part) for part in cmd), language="bash")
    st.markdown(f"**{title} Output**")
    st.code(output or "(no output)")
    if return_code == 0:
        st.success(f"{title} completed successfully.")
    else:
        st.error(f"{title} failed (exit code {return_code}).")


st.title("MTG AI Deck Builder")
st.caption(
    "Local UI for generation and training "
    "(frequency + cluster + semantics + optional LLM rerank)."
)

tab_generate, tab_train = st.tabs(["Generate Deck", "Train Model"])

with tab_generate:
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

    if generate_clicked:
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

with tab_train:
    st.subheader("Corpus Builder")
    tcol1, tcol2 = st.columns(2)

    with tcol1:
        train_cards_path = st.text_input("Cards CSV", value="data/commander_cards.csv", key="train_cards_path")
        train_decks_path = st.text_input("Deck Directory", value="current_commander_decks", key="train_decks_path")
        corpus_output_prefix = st.text_input(
            "Corpus Output Prefix",
            value="data/deck_corpus",
            key="corpus_output_prefix",
            help="Creates <prefix>.npz, <prefix>_cards.json, <prefix>_meta.json",
        )
    with tcol2:
        min_card_frequency = st.number_input(
            "Min Card Frequency",
            min_value=1,
            max_value=100,
            value=1,
            step=1,
            key="min_card_frequency",
        )
        max_vocab_size = st.number_input(
            "Max Vocab Size (0 = unlimited)",
            min_value=0,
            max_value=500000,
            value=0,
            step=100,
            key="max_vocab_size",
        )

    st.subheader("Model Trainer")
    mcol1, mcol2 = st.columns(2)
    with mcol1:
        corpus_prefix_for_train = st.text_input(
            "Corpus Prefix For Training",
            value="data/deck_corpus",
            key="corpus_prefix_for_train",
        )
        model_output_prefix = st.text_input(
            "Model Output Prefix",
            value="models/deck_kmeans",
            key="model_output_prefix",
            help="Creates <prefix>.npz and <prefix>_meta.json",
        )
    with mcol2:
        clusters = st.number_input("Clusters", min_value=1, max_value=512, value=16, step=1, key="clusters")
        train_seed = st.number_input("Seed", min_value=0, max_value=1_000_000, value=42, step=1, key="train_seed")
        semantic_dim = st.number_input(
            "Semantic Dim",
            min_value=2,
            max_value=512,
            value=64,
            step=1,
            key="semantic_dim",
        )

    build_before_train = st.checkbox(
        "Build corpus before training",
        value=True,
        help="If enabled, training will rebuild corpus first using settings above.",
    )

    bcol1, bcol2 = st.columns(2)
    build_corpus_clicked = bcol1.button("Build Corpus", type="secondary")
    train_model_clicked = bcol2.button("Train Model", type="primary")

    def run_build_corpus() -> bool:
        if not os.path.isfile(train_cards_path):
            st.error(f"Cards CSV not found: {train_cards_path}")
            return False
        if not os.path.isdir(train_decks_path):
            st.error(f"Deck directory not found: {train_decks_path}")
            return False

        cmd = [
            sys.executable,
            "deck_corpus_builder.py",
            "--cards",
            train_cards_path,
            "--decks",
            train_decks_path,
            "--output-prefix",
            corpus_output_prefix,
            "--min-card-frequency",
            str(int(min_card_frequency)),
            "--max-vocab-size",
            str(int(max_vocab_size)),
        ]
        with st.spinner("Building corpus..."):
            return_code, output = run_cli_command(cmd)
        render_command_result("Build Corpus", cmd, return_code, output)
        return return_code == 0

    def run_train_model() -> bool:
        cmd = [
            sys.executable,
            "train_deck_generator.py",
            "--corpus-prefix",
            corpus_prefix_for_train,
            "--clusters",
            str(int(clusters)),
            "--seed",
            str(int(train_seed)),
            "--output-prefix",
            model_output_prefix,
            "--cards",
            train_cards_path,
            "--semantic-dim",
            str(int(semantic_dim)),
        ]
        with st.spinner("Training model..."):
            return_code, output = run_cli_command(cmd)
        render_command_result("Train Model", cmd, return_code, output)
        if return_code == 0:
            cached_load_cluster_model.clear()
        return return_code == 0

    if build_corpus_clicked:
        ok = run_build_corpus()
        meta_path = corpus_output_prefix + "_meta.json"
        if ok and os.path.isfile(meta_path):
            st.markdown("**Corpus Metadata**")
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    st.json(json.load(f), expanded=False)
            except Exception as e:
                st.warning(f"Failed to read corpus metadata: {e}")

    if train_model_clicked:
        if build_before_train:
            build_ok = run_build_corpus()
            if not build_ok:
                st.stop()

        train_ok = run_train_model()
        model_meta_path = model_output_prefix + "_meta.json"
        if train_ok and os.path.isfile(model_meta_path):
            st.markdown("**Model Metadata**")
            try:
                with open(model_meta_path, "r", encoding="utf-8") as f:
                    st.json(json.load(f), expanded=False)
            except Exception as e:
                st.warning(f"Failed to read model metadata: {e}")
