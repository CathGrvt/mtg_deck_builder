from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class TelemetryConfig:
    enable_vertex_telemetry: bool = True
    enable_langsmith_fanout: bool = False
    langsmith_project: str = "mtg-deck-builder"


def build_agent_engine_env_vars(config: TelemetryConfig | None = None) -> Dict[str, str]:
    cfg = config or TelemetryConfig()
    env_vars: Dict[str, str] = {}

    if cfg.enable_vertex_telemetry:
        env_vars.update(
            {
                "GOOGLE_CLOUD_AGENT_ENGINE_ENABLE_TELEMETRY": "true",
                "OTEL_SEMCONV_STABILITY_OPT_IN": "gen_ai_latest_experimental",
                "OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT": "EVENT_ONLY",
            }
        )

    if cfg.enable_langsmith_fanout:
        env_vars.update(
            {
                "LANGSMITH_OTEL_ENABLED": "true",
                "LANGSMITH_PROJECT": cfg.langsmith_project,
            }
        )

    return env_vars
