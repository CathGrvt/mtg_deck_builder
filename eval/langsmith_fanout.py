from __future__ import annotations

import argparse
import json
from typing import Dict


def build_langsmith_otel_env(
    api_key: str,
    project: str,
    endpoint: str = "https://api.smith.langchain.com/otel",
) -> Dict[str, str]:
    return {
        "LANGSMITH_OTEL_ENABLED": "true",
        "LANGSMITH_API_KEY": api_key,
        "LANGSMITH_PROJECT": project,
        "OTEL_EXPORTER_OTLP_ENDPOINT": endpoint,
        "OTEL_EXPORTER_OTLP_HEADERS": f"x-api-key={api_key}",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate env vars for LangSmith OTEL fan-out.")
    parser.add_argument("--api-key", required=True, help="LangSmith API key")
    parser.add_argument("--project", default="mtg-deck-builder")
    parser.add_argument("--endpoint", default="https://api.smith.langchain.com/otel")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = build_langsmith_otel_env(
        api_key=args.api_key,
        project=args.project,
        endpoint=args.endpoint,
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
