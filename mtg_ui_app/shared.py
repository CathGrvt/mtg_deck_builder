from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mtg_shared.openai_api import chat_completion_content
from mtg_shared.text import dedupe_preserving_order, keyword_tokens
import pandas as pd
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

from mtg_io import load_card_database, load_decklists_from_directory
from mtg_ui_app.backend_client import call_backend_json
from research_pipeline.llm import RuleBasedLLM
from research_pipeline.models import RetrievedChunk
from research_pipeline.reporting import report_to_markdown
from research_pipeline.retrieval.corpus import build_domain_corpus
from research_pipeline.retrieval.index import HybridRetrievalIndex
from research_pipeline.set_aliases import extract_set_codes_from_text


@st.cache_data(show_spinner=False)
def cached_load_card_db(path: str) -> pd.DataFrame:
    return load_card_database(path)


@st.cache_data(show_spinner=False)
def cached_load_decklists(path: str) -> Dict[str, List[str]]:
    return load_decklists_from_directory(path, include_command_zone=True)


@st.cache_resource(show_spinner=False)
def cached_load_cluster_model(prefix: str) -> Optional[Dict]:
    from ai_deck_generator import load_cluster_model

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


def ensure_research_paths(cards_path: str, decks_path: str, emit_error: bool = True) -> bool:
    show = st.error if emit_error else st.warning
    if not os.path.isfile(cards_path):
        show(f"Cards CSV not found: {cards_path}")
        return False
    if not os.path.isdir(decks_path):
        show(f"Deck directory not found: {decks_path}")
        return False
    return True


def build_research_index_with_feedback(
    cards_path: str,
    decks_path: str,
    meta_paths: Tuple[str, ...],
    enable_semantic: bool,
    lexical_weight: float,
    semantic_weight: float,
    spinner_message: str,
) -> Optional[Tuple[HybridRetrievalIndex, int, int]]:
    try:
        with st.spinner(spinner_message):
            index, chunk_count, source_count = cached_build_research_index(
                cards_path=cards_path,
                decks_path=decks_path,
                meta_paths=meta_paths,
                enable_semantic=enable_semantic,
                lexical_weight=float(lexical_weight),
                semantic_weight=float(semantic_weight),
            )
    except Exception as e:
        st.error(f"Failed to build retrieval index: {e}")
        return None
    return index, chunk_count, source_count


def call_deployed_recommendation(
    backend_url: str,
    payload: Dict[str, Any],
    timeout_sec: int = 60,
) -> Dict[str, Any]:
    """
    Call deployed backend adapter endpoint that proxies to Agent Engine.
    """
    return call_backend_json(
        backend_url=backend_url,
        payload=payload,
        timeout_sec=timeout_sec,
    )


def render_chat_evidence(evidence: Sequence[Dict[str, Any]], max_items: int = 8) -> None:
    for item in evidence[:max_items]:
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
    return chat_completion_content(
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout_sec=timeout_sec,
        temperature=float(temperature),
        messages=messages,
        empty_message="No response was returned by the model.",
    )


def _augment_with_set_filtered_cards(
    index: HybridRetrievalIndex,
    query: str,
    base_results: Sequence[RetrievedChunk],
    set_codes: Sequence[str],
    max_results: int,
) -> List[RetrievedChunk]:
    if not set_codes:
        return list(base_results)

    wanted_codes = {code.lower() for code in set_codes if code}
    if not wanted_codes:
        return list(base_results)

    query_vec = index.vectorizer.transform([query])
    lexical_scores = cosine_similarity(query_vec, index.lexical_matrix).flatten()

    candidates: List[tuple[int, float]] = []
    for idx, chunk in enumerate(index.chunks):
        if chunk.source != "card_db":
            continue
        chunk_set = str(chunk.metadata.get("set", "")).strip().lower()
        if chunk_set and chunk_set in wanted_codes:
            candidates.append((idx, float(lexical_scores[idx])))

    candidates.sort(key=lambda item: item[1], reverse=True)

    set_ranked_results: List[RetrievedChunk] = []
    seen_ids = set()
    for idx, score in candidates:
        chunk = index.chunks[idx]
        if chunk.chunk_id in seen_ids:
            continue
        # Explicit user set constraints should dominate ranking for chat contexts.
        boosted_score = 1.0 + float(score)
        set_ranked_results.append(RetrievedChunk.from_chunk(chunk=chunk, score=boosted_score))
        seen_ids.add(chunk.chunk_id)
        if len(set_ranked_results) >= max_results:
            break

    merged: List[RetrievedChunk] = list(set_ranked_results)
    for item in base_results:
        if item.chunk_id in seen_ids:
            continue
        merged.append(item)
        seen_ids.add(item.chunk_id)
        if len(merged) >= max_results:
            break

    return merged


def _extract_focus_terms(text: str) -> List[str]:
    tokens = keyword_tokens(
        text,
        stopwords={
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "into",
            "your",
            "their",
            "about",
            "what",
            "which",
            "when",
            "where",
            "how",
            "why",
            "some",
            "good",
            "cards",
            "commander",
            "deck",
        },
    )
    return dedupe_preserving_order(tokens)[:8]


def _find_explicit_card_mentions(index: HybridRetrievalIndex, question: str) -> List[RetrievedChunk]:
    question_lower = str(question or "").lower()
    if not question_lower.strip():
        return []

    matches: List[RetrievedChunk] = []
    seen = set()
    for chunk in index.chunks:
        if chunk.source != "card_db":
            continue

        title = str(chunk.title or "").strip().lower()
        if not title:
            continue

        short_title = title.split(",")[0].strip()
        direct_match = title in question_lower
        short_match = bool(
            short_title
            and len(short_title) >= 4
            and re.search(rf"\b{re.escape(short_title)}\b", question_lower)
        )
        if not (direct_match or short_match):
            continue

        chunk_id = str(chunk.chunk_id)
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        matches.append(RetrievedChunk.from_chunk(chunk=chunk, score=2.0))

    return matches


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
    available_set_codes = {
        str(chunk.metadata.get("set", "")).strip().lower()
        for chunk in index.chunks
        if chunk.source == "card_db" and str(chunk.metadata.get("set", "")).strip()
    }
    set_codes = extract_set_codes_from_text(question, valid_codes=available_set_codes)

    query_terms = [question]
    explicit_commander_mentions = _find_explicit_card_mentions(index=index, question=question)
    if explicit_commander_mentions:
        query_terms.extend(_extract_focus_terms(explicit_commander_mentions[0].text))
    expanded_query = " ".join(part for part in query_terms if part.strip())

    retrieved = index.search(expanded_query, top_k=max(1, int(top_k)))
    if explicit_commander_mentions:
        # Force explicit commander context into the final evidence pool.
        retrieved = explicit_commander_mentions + [
            item for item in retrieved if item.chunk_id != explicit_commander_mentions[0].chunk_id
        ]
    if set_codes:
        retrieved = _augment_with_set_filtered_cards(
            index=index,
            query=expanded_query,
            base_results=retrieved,
            set_codes=set_codes,
            max_results=max(12, int(top_k) * 2),
        )
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
        handle.write(report_to_markdown(output["report"]))

    return {
        "run_dir": run_dir,
        "report_json": report_json_path,
        "state_json": state_json_path,
        "report_md": report_md_path,
        "trace_jsonl": resolved_trace_path,
    }
