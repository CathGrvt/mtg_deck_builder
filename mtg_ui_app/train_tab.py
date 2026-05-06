from __future__ import annotations

import json
import os
import sys

import streamlit as st

from mtg_ui_app.shared import (
    cached_load_cluster_model,
    render_command_result,
    run_cli_command,
)


def render_train_tab() -> None:
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
