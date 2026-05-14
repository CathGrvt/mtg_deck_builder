from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

import streamlit as st

from research_pipeline.graph import ResearchPipeline
from research_pipeline.llm import build_default_llm
from research_pipeline.reporting import report_to_markdown

from mtg_ui_app.backend_client import (
    build_research_backend_payload,
    call_backend_json,
)
from mtg_ui_app.shared import (
    build_research_index_with_feedback,
    ensure_research_paths,
    parse_path_list,
    save_agentic_run_artifacts,
)


def render_agentic_tab() -> None:
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
        use_deployed_backend = st.checkbox(
            "Use Deployed Agent Backend",
            value=False,
            key="agent_use_deployed_backend",
            help="Calls backend research endpoint before local pipeline fallback.",
        )
        backend_research_url = st.text_input(
            "Backend Research URL",
            value=os.getenv("MTG_GCP_RESEARCH_URL", "http://localhost:8080/v1/research/run"),
            key="agent_backend_research_url",
            disabled=not use_deployed_backend,
        )
        backend_timeout = st.number_input(
            "Backend Timeout Seconds",
            min_value=5,
            max_value=300,
            value=90,
            step=1,
            key="agent_backend_timeout",
            disabled=not use_deployed_backend,
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

    if not run_agentic:
        return

    topic = agent_topic.strip()
    if not topic:
        st.error("Please enter a research topic.")
        st.stop()

    def _render_report(report: dict) -> None:
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
        report_md_text = report_to_markdown(report)

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

    if use_deployed_backend:
        backend_payload = build_research_backend_payload(
            session_id=f"agentic-{uuid.uuid4().hex[:12]}",
            topic=topic,
            max_iterations=int(agent_max_iterations),
            max_questions=int(agent_max_questions),
            top_k_per_query=int(agent_top_k),
            enable_semantic=bool(agent_enable_semantic),
            use_langgraph=bool(agent_use_langgraph),
        )
        try:
            with st.spinner("Calling deployed backend research endpoint..."):
                backend_response = call_backend_json(
                    backend_url=backend_research_url,
                    payload=backend_payload,
                    timeout_sec=int(backend_timeout),
                )
        except Exception as e:
            st.warning(f"Backend research call failed ({e}). Falling back to local pipeline.")
        else:
            report = backend_response.get("report", {})
            if isinstance(report, dict) and "validation" not in report:
                report = dict(report)
                report["validation"] = backend_response.get("validation", {})
            if not isinstance(report, dict):
                st.error("Backend response did not include a valid report object.")
                st.stop()
            _render_report(report)
            st.info(
                "Backend metadata:\n"
                f"- latency_ms: {backend_response.get('latency_ms', 'n/a')}\n"
                f"- model_used: {backend_response.get('model_used', 'n/a')}\n"
                f"- trace_id: {backend_response.get('trace_id', 'n/a')}"
            )
            if save_agentic_run:
                os.makedirs(agentic_out_dir, exist_ok=True)
                run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                run_dir = os.path.join(agentic_out_dir, run_id)
                os.makedirs(run_dir, exist_ok=True)
                trace_path = os.path.join(run_dir, "trace.jsonl")
                with open(trace_path, "a", encoding="utf-8"):
                    pass
                artifacts = save_agentic_run_artifacts(
                    output={
                        "report": report,
                        "state": {"backend_response": backend_response},
                    },
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
            st.stop()

    if not ensure_research_paths(agent_cards_path, agent_decks_path):
        st.stop()

    parsed_meta_paths = parse_path_list(agent_meta_paths_raw)
    index_bundle = build_research_index_with_feedback(
        cards_path=agent_cards_path,
        decks_path=agent_decks_path,
        meta_paths=parsed_meta_paths,
        enable_semantic=agent_enable_semantic,
        lexical_weight=float(agent_lexical_weight),
        semantic_weight=float(agent_semantic_weight),
        spinner_message="Building retrieval index...",
    )
    if index_bundle is None:
        st.stop()
    index, chunk_count, source_count = index_bundle

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
    _render_report(report)

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
