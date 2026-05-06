import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import requests
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
from research_pipeline.graph import ResearchPipeline
from research_pipeline.llm import RuleBasedLLM, build_default_llm
from research_pipeline.retrieval.corpus import build_domain_corpus
from research_pipeline.retrieval.index import HybridRetrievalIndex
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


def parse_path_list(raw: str) -> Tuple[str, ...]:
    if not raw.strip():
        return tuple()

    paths: List[str] = []
    seen = set()
    for part in raw.replace(",", "\n").splitlines():
        candidate = part.strip()
        if not candidate:
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        paths.append(candidate)

    return tuple(paths)


@st.cache_resource(show_spinner=False)
def cached_build_research_index(
    cards_path: str,
    decks_path: str,
    meta_paths: Tuple[str, ...],
    enable_semantic: bool,
    lexical_weight: float,
    semantic_weight: float,
) -> Tuple[HybridRetrievalIndex, int, int]:
    chunks = build_domain_corpus(
        cards_csv=cards_path,
        decks_dir=decks_path,
        meta_json_paths=list(meta_paths) if meta_paths else None,
    )
    if not chunks:
        raise ValueError("No corpus chunks were built from the provided paths.")

    index = HybridRetrievalIndex(
        chunks=chunks,
        lexical_weight=float(lexical_weight),
        semantic_weight=float(semantic_weight),
        enable_semantic=bool(enable_semantic),
    )
    source_count = len({chunk.doc_id for chunk in chunks})
    return index, len(chunks), source_count


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


def format_structured_report_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = [
        f"# Research Report: {report.get('topic', '')}",
        "",
        "## Summary",
        str(report.get("summary", "")),
        "",
        "## Claims",
    ]

    claims = report.get("claims", [])
    if not claims:
        lines.append("- No claims produced.")
    else:
        for claim in claims:
            claim_text = str(claim.get("claim", ""))
            confidence = claim.get("confidence", 0.0)
            citations = claim.get("citations", [])
            citation_text = ", ".join(
                [f"{item.get('doc_id')}::{item.get('chunk_id')}" for item in citations]
            )
            lines.append(
                f"- {claim_text} (confidence={confidence}, citations=[{citation_text}])"
            )

    lines.extend(["", "## Open Questions"])
    for question in report.get("open_questions", []):
        lines.append(f"- {question}")

    lines.extend(
        [
            "",
            "## Validation",
            "```json",
            json.dumps(report.get("validation", {}), indent=2),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def _call_openai_chat_completion(
    messages: Sequence[Dict[str, str]],
    model: str,
    api_key_env: str,
    base_url: str,
    timeout_sec: int,
    temperature: float,
) -> str:
    api_key = os.getenv(api_key_env, "")
    if not api_key:
        raise ValueError(f"Environment variable '{api_key_env}' is not set.")

    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": float(temperature),
            "messages": list(messages),
        },
        timeout=max(5, int(timeout_sec)),
    )
    response.raise_for_status()
    payload = response.json()
    choices = payload.get("choices") or []
    if not choices:
        return "No response was returned by the model."

    message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
    return str(message.get("content", "")).strip()


def generate_chatbot_answer(
    question: str,
    index: HybridRetrievalIndex,
    top_k: int,
    history: Sequence[Dict[str, Any]],
    use_llm: bool,
    llm_model: str,
    llm_api_key_env: str,
    llm_base_url: str,
    llm_timeout_sec: int,
    llm_temperature: float,
) -> Tuple[str, List[Dict[str, Any]]]:
    retrieved = index.search(question, top_k=max(1, int(top_k)))
    evidence = [item.to_dict() for item in retrieved]

    if not retrieved:
        return (
            "I could not find relevant corpus evidence for that question. Try rephrasing with a card, archetype, or format keyword.",
            evidence,
        )

    if use_llm and os.getenv(llm_api_key_env, ""):
        context_chunks = [
            {
                "doc_id": item.doc_id,
                "chunk_id": item.chunk_id,
                "source": item.source,
                "title": item.title,
                "score": round(float(item.score), 4),
                "text": item.text[:420],
            }
            for item in retrieved[:8]
        ]

        system_prompt = (
            "You are an MTG research assistant. Answer using only the provided context. "
            "If uncertain, say so. End with a short 'Sources' line using provided chunk ids."
        )
        messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

        for turn in history[-6:]:
            role = "assistant" if str(turn.get("role")) == "assistant" else "user"
            content = str(turn.get("content", "")).strip()
            if content:
                messages.append({"role": role, "content": content[:1200]})

        user_payload = {
            "question": question,
            "context_chunks": context_chunks,
            "instructions": [
                "Ground claims in context chunks.",
                "Do not invent citations.",
                "Keep answer concise and actionable.",
            ],
        }
        messages.append({"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)})

        try:
            answer = _call_openai_chat_completion(
                messages=messages,
                model=llm_model,
                api_key_env=llm_api_key_env,
                base_url=llm_base_url,
                timeout_sec=llm_timeout_sec,
                temperature=llm_temperature,
            )
            if answer:
                return answer, evidence
        except Exception as e:
            fallback_msg = f"LLM synthesis failed ({e}). Falling back to rule-based answer."
            st.warning(fallback_msg)

    rule_llm = RuleBasedLLM()
    report = rule_llm.write_report(topic=question, retrieved_chunks=retrieved, gaps=[])

    lines: List[str] = [report.summary, "", "Evidence-backed points:"]
    for claim in report.claims[:4]:
        citations = ", ".join(f"{item.doc_id}::{item.chunk_id}" for item in claim.citations)
        lines.append(f"- {claim.claim} [{citations}]")

    return "\n".join(lines).strip(), evidence


def save_agentic_run_artifacts(
    output: Dict[str, Any],
    out_dir: str,
    run_dir: Optional[str] = None,
    trace_path: Optional[str] = None,
) -> Dict[str, str]:
    if run_dir is None:
        os.makedirs(out_dir, exist_ok=True)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = os.path.join(out_dir, run_id)
    os.makedirs(run_dir, exist_ok=True)

    report_json_path = os.path.join(run_dir, "report.json")
    state_json_path = os.path.join(run_dir, "state.json")
    report_md_path = os.path.join(run_dir, "report.md")
    resolved_trace_path = trace_path or os.path.join(run_dir, "trace.jsonl")

    with open(report_json_path, "w", encoding="utf-8") as handle:
        json.dump(output["report"], handle, indent=2)

    with open(state_json_path, "w", encoding="utf-8") as handle:
        json.dump(output["state"], handle, indent=2)

    with open(report_md_path, "w", encoding="utf-8") as handle:
        handle.write(format_structured_report_markdown(output["report"]))

    return {
        "run_dir": run_dir,
        "report_json": report_json_path,
        "state_json": state_json_path,
        "report_md": report_md_path,
        "trace_jsonl": resolved_trace_path,
    }


st.title("MTG AI Deck Builder")
st.caption(
    "Local UI for deck generation, model training, agentic research, and RAG chatbot workflows."
)

(tab_generate, tab_train, tab_agentic, tab_chat) = st.tabs(
    ["Generate Deck", "Train Model", "Agentic Research", "Chatbot"]
)

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

with tab_agentic:
    st.subheader("Agentic Research")
    st.caption("Planner -> Retriever -> Critic -> Writer -> Validator with structured, cited output.")

    acol1, acol2 = st.columns(2)

    with acol1:
        agent_cards_path = st.text_input(
            "Cards CSV",
            value="data/commander_cards.csv",
            key="agent_cards_path",
        )
        agent_decks_path = st.text_input(
            "Deck Directory",
            value="current_commander_decks",
            key="agent_decks_path",
        )
        agent_meta_paths_raw = st.text_area(
            "Meta JSON Paths (optional)",
            value="",
            key="agent_meta_paths",
            help="Comma or newline separated file paths. Leave empty to auto-discover json_outputs/*.json.",
            height=90,
        )
        agent_topic = st.text_area(
            "Research Topic",
            value="",
            key="agent_topic",
            placeholder="Example: What card advantage patterns appear most often in successful Boros commander decks?",
            height=120,
        )

    with acol2:
        agent_max_iterations = st.number_input(
            "Max Critic Iterations",
            min_value=1,
            max_value=8,
            value=3,
            step=1,
            key="agent_max_iterations",
        )
        agent_max_questions = st.number_input(
            "Max Planner Questions",
            min_value=1,
            max_value=12,
            value=5,
            step=1,
            key="agent_max_questions",
        )
        agent_top_k = st.number_input(
            "Retriever Top-K Per Query",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
            key="agent_top_k",
        )
        agent_enable_semantic = st.checkbox(
            "Enable Semantic Retrieval",
            value=False,
            key="agent_enable_semantic",
            help="Enables sentence-transformer embeddings if available.",
        )
        agent_use_langgraph = st.checkbox(
            "Use LangGraph Runtime",
            value=True,
            key="agent_use_langgraph",
            help="Falls back to manual loop if LangGraph is unavailable.",
        )
        agent_lexical_weight = st.slider(
            "Lexical Retrieval Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            key="agent_lexical_weight",
        )
        agent_semantic_weight = st.slider(
            "Semantic Retrieval Weight",
            min_value=0.0,
            max_value=1.0,
            value=0.4,
            step=0.05,
            key="agent_semantic_weight",
        )

    with st.expander("LLM Settings (Optional)", expanded=False):
        ac1, ac2 = st.columns(2)
        with ac1:
            agent_llm_model = st.text_input(
                "LLM Model",
                value="gpt-4o-mini",
                key="agent_llm_model",
            )
            agent_llm_api_key_env = st.text_input(
                "API Key Env Var",
                value="OPENAI_API_KEY",
                key="agent_llm_api_key_env",
            )
        with ac2:
            agent_llm_base_url = st.text_input(
                "Base URL",
                value="https://api.openai.com/v1",
                key="agent_llm_base_url",
            )
            agent_llm_timeout = st.number_input(
                "Timeout Seconds",
                min_value=5,
                max_value=180,
                value=45,
                step=1,
                key="agent_llm_timeout",
            )

    save_agentic_run = st.checkbox("Save Artifacts To Disk", value=True, key="save_agentic_run")
    agentic_out_dir = st.text_input("Output Directory", value="runs", key="agentic_out_dir")

    run_agentic = st.button("Run Agentic Research", type="primary", key="run_agentic")

    if run_agentic:
        topic = agent_topic.strip()
        if not topic:
            st.error("Please enter a research topic.")
            st.stop()
        if not os.path.isfile(agent_cards_path):
            st.error(f"Cards CSV not found: {agent_cards_path}")
            st.stop()
        if not os.path.isdir(agent_decks_path):
            st.error(f"Deck directory not found: {agent_decks_path}")
            st.stop()

        parsed_meta_paths = parse_path_list(agent_meta_paths_raw)

        try:
            with st.spinner("Building retrieval index..."):
                index, chunk_count, source_count = cached_build_research_index(
                    cards_path=agent_cards_path,
                    decks_path=agent_decks_path,
                    meta_paths=parsed_meta_paths,
                    enable_semantic=agent_enable_semantic,
                    lexical_weight=float(agent_lexical_weight),
                    semantic_weight=float(agent_semantic_weight),
                )
        except Exception as e:
            st.error(f"Failed to build retrieval index: {e}")
            st.stop()

        st.info(f"Corpus ready: {chunk_count} chunks from {source_count} sources.")

        trace_path = None
        run_dir = None
        if save_agentic_run:
            os.makedirs(agentic_out_dir, exist_ok=True)
            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            run_dir = os.path.join(agentic_out_dir, run_id)
            os.makedirs(run_dir, exist_ok=True)
            trace_path = os.path.join(run_dir, "trace.jsonl")

        llm = build_default_llm(
            model=agent_llm_model,
            api_key_env=agent_llm_api_key_env,
            base_url=agent_llm_base_url,
            timeout_sec=int(agent_llm_timeout),
        )

        pipeline = ResearchPipeline(
            index=index,
            llm=llm,
            max_iterations=int(agent_max_iterations),
            max_questions=int(agent_max_questions),
            top_k_per_query=int(agent_top_k),
            use_langgraph=agent_use_langgraph,
            trace_path=trace_path,
        )

        try:
            with st.spinner("Running agentic pipeline..."):
                output = pipeline.run(topic)
        except Exception as e:
            st.error(f"Agentic pipeline failed: {e}")
            st.stop()

        report = output["report"]
        st.success("Agentic report generated.")

        st.markdown("**Summary**")
        st.write(report.get("summary", ""))

        st.markdown("**Claims**")
        claims = report.get("claims", [])
        if not claims:
            st.write("No claims were produced.")
        else:
            for idx, claim in enumerate(claims, start=1):
                claim_text = str(claim.get("claim", ""))
                confidence = claim.get("confidence", 0.0)
                citations = claim.get("citations", [])
                citation_text = ", ".join(
                    [f"{item.get('doc_id')}::{item.get('chunk_id')}" for item in citations]
                )
                st.markdown(
                    f"{idx}. {claim_text}  \n"
                    f"`confidence={confidence}`  \n"
                    f"`citations: {citation_text or 'none'}`"
                )

        st.markdown("**Open Questions**")
        open_questions = report.get("open_questions", [])
        if not open_questions:
            st.write("None")
        else:
            for question in open_questions:
                st.markdown(f"- {question}")

        st.markdown("**Validation**")
        st.json(report.get("validation", {}), expanded=False)

        report_json_text = json.dumps(report, indent=2)
        report_md_text = format_structured_report_markdown(report)

        st.download_button(
            label="Download Report JSON",
            data=report_json_text,
            file_name="research_report.json",
            mime="application/json",
            key="download_agent_report_json",
        )
        st.download_button(
            label="Download Report Markdown",
            data=report_md_text,
            file_name="research_report.md",
            mime="text/markdown",
            key="download_agent_report_md",
        )

        if save_agentic_run and run_dir is not None:
            artifacts = save_agentic_run_artifacts(
                output=output,
                out_dir=agentic_out_dir,
                run_dir=run_dir,
                trace_path=trace_path,
            )
            st.info(
                "Artifacts saved:\n"
                f"- run_dir: {artifacts['run_dir']}\n"
                f"- report_json: {artifacts['report_json']}\n"
                f"- report_md: {artifacts['report_md']}\n"
                f"- state_json: {artifacts['state_json']}\n"
                f"- trace_jsonl: {artifacts['trace_jsonl']}"
            )

with tab_chat:
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

    chat_index: Optional[HybridRetrievalIndex] = None
    parsed_chat_meta_paths = parse_path_list(chat_meta_paths_raw)

    if not os.path.isfile(chat_cards_path):
        st.warning(f"Cards CSV not found: {chat_cards_path}")
    elif not os.path.isdir(chat_decks_path):
        st.warning(f"Deck directory not found: {chat_decks_path}")
    else:
        try:
            with st.spinner("Preparing chatbot index..."):
                chat_index, chat_chunk_count, chat_source_count = cached_build_research_index(
                    cards_path=chat_cards_path,
                    decks_path=chat_decks_path,
                    meta_paths=parsed_chat_meta_paths,
                    enable_semantic=chat_enable_semantic,
                    lexical_weight=float(chat_lexical_weight),
                    semantic_weight=float(chat_semantic_weight),
                )
            st.caption(f"Corpus loaded: {chat_chunk_count} chunks from {chat_source_count} sources.")
        except Exception as e:
            st.error(f"Failed to initialize chatbot index: {e}")

    for msg in st.session_state["mtg_chat_messages"]:
        with st.chat_message(msg.get("role", "assistant")):
            st.markdown(msg.get("content", ""))
            evidence = msg.get("evidence", [])
            if evidence:
                with st.expander("Evidence", expanded=False):
                    for item in evidence[:8]:
                        title = item.get("title", "")
                        chunk_id = item.get("chunk_id", "")
                        source = item.get("source", "")
                        score = item.get("score", 0.0)
                        text = str(item.get("text", ""))
                        if len(text) > 280:
                            text = text[:280].rstrip() + "..."
                        st.markdown(
                            f"- `{chunk_id}` | {title} ({source}, score={float(score):.3f})\n\n"
                            f"  {text}"
                        )

    user_prompt = st.chat_input("Ask a question about MTG decks, cards, or meta trends...")
    if user_prompt:
        if chat_index is None:
            st.error("Chatbot index is not available. Fix paths/settings first.")
            st.stop()

        st.session_state["mtg_chat_messages"].append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )

        history_before_answer = st.session_state["mtg_chat_messages"][:-1]
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
