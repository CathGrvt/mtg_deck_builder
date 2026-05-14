from __future__ import annotations

import argparse
import json
import os
from typing import List

from gcp_agent_runtime.adk_app import build_agent_engine_app
from gcp_agent_runtime.telemetry import TelemetryConfig, build_agent_engine_env_vars


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy MTG ADK agent to Vertex AI Agent Engine.")
    parser.add_argument("--project", required=True, help="GCP project ID.")
    parser.add_argument("--location", default="us-central1", help="Vertex location.")
    parser.add_argument("--staging-bucket", required=True, help="GCS staging bucket URI.")
    parser.add_argument("--display-name", default="mtg-deck-builder-agent", help="Agent display name.")
    parser.add_argument(
        "--requirements",
        nargs="*",
        default=[
            "google-cloud-aiplatform[adk,agent_engines]>=1.144",
            "google-adk",
            "numpy>=2.0,<3",
            "pandas>=2.2,<3",
            "requests>=2.31,<3",
            "scikit-learn>=1.5,<2",
        ],
        help="Extra runtime requirements packaged into Agent Engine deployment.",
    )
    parser.add_argument(
        "--langsmith-fanout",
        action="store_true",
        help="Enable LangSmith OpenTelemetry fan-out env vars.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print deployment payload without deploying.",
    )
    return parser.parse_args()


def _collect_agent_runtime_env() -> dict:
    keys = [
        "OPENAI_API_KEY",
        "GOOGLE_CLOUD_PROJECT",
        "GOOGLE_CLOUD_LOCATION",
        "MTG_LLM_PROVIDER",
        "MTG_OPENAI_MODEL",
        "MTG_OPENAI_API_KEY_ENV",
        "MTG_OPENAI_BASE_URL",
        "MTG_VERTEX_MODEL",
        "MTG_LLM_TIMEOUT_SEC",
        "MTG_CHAT_ENABLE_CLARIFICATION",
        "MTG_CHAT_MAX_CLARIFICATION_TURNS",
        "MTG_RAG_CORPUS_URI",
        "MTG_LOCAL_RETRIEVER_CARDS_CSV",
        "MTG_LOCAL_RETRIEVER_DECKS_DIR",
        "MTG_LOCAL_RETRIEVER_META_JSON_PATHS",
        "MTG_LOCAL_RETRIEVER_ENABLE_SEMANTIC",
        "MTG_LOCAL_RETRIEVER_LEXICAL_WEIGHT",
        "MTG_LOCAL_RETRIEVER_SEMANTIC_WEIGHT",
    ]
    result = {}
    for key in keys:
        value = os.getenv(key, "")
        if str(value).strip():
            result[key] = str(value)
    return result


def deploy(
    project: str,
    location: str,
    staging_bucket: str,
    display_name: str,
    requirements: List[str],
    langsmith_fanout: bool = False,
    dry_run: bool = False,
) -> dict:
    try:
        import vertexai
        from vertexai import agent_engines
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "google-cloud-aiplatform is required for deployment. Install requirements-gcp.txt."
        ) from exc

    env_vars = build_agent_engine_env_vars(
        TelemetryConfig(
            enable_vertex_telemetry=True,
            enable_langsmith_fanout=langsmith_fanout,
        )
    )
    env_vars.update(_collect_agent_runtime_env())
    payload = {
        "project": project,
        "location": location,
        "staging_bucket": staging_bucket,
        "display_name": display_name,
        "requirements": requirements,
        "env_vars": env_vars,
    }

    if dry_run:
        return {
            "status": "dry_run",
            "payload": payload,
        }

    vertexai.init(
        project=project,
        location=location,
        staging_bucket=staging_bucket,
    )

    app = build_agent_engine_app()
    remote_agent = agent_engines.create(
        agent_engine=app,
        requirements=requirements,
        display_name=display_name,
        env_vars=env_vars,
    )
    return {
        "status": "deployed",
        "resource_name": remote_agent.resource_name,
        "payload": payload,
    }


def main() -> None:
    args = parse_args()
    result = deploy(
        project=args.project,
        location=args.location,
        staging_bucket=args.staging_bucket,
        display_name=args.display_name,
        requirements=list(args.requirements),
        langsmith_fanout=bool(args.langsmith_fanout),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
