from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Mapping


@dataclass(frozen=True)
class RuntimeEnvVar:
    name: str
    default: str = ""
    backend: bool = True
    agent_engine: bool = True


RUNTIME_ENV_VARS: List[RuntimeEnvVar] = [
    RuntimeEnvVar("GOOGLE_CLOUD_PROJECT"),
    RuntimeEnvVar("GOOGLE_CLOUD_LOCATION", default="us-central1"),
    RuntimeEnvVar("MTG_BACKEND_MODE", default="local", agent_engine=False),
    RuntimeEnvVar("MTG_LLM_PROVIDER", default="openai"),
    RuntimeEnvVar("MTG_OPENAI_MODEL", default="gpt-4o-mini"),
    RuntimeEnvVar("MTG_OPENAI_API_KEY_ENV", default="OPENAI_API_KEY"),
    RuntimeEnvVar("MTG_OPENAI_API_KEY_SECRET_RESOURCE"),
    RuntimeEnvVar("MTG_OPENAI_API_KEY_SECRET"),
    RuntimeEnvVar("MTG_OPENAI_BASE_URL", default="https://api.openai.com/v1"),
    RuntimeEnvVar("MTG_VERTEX_MODEL", default="gemini-2.5-flash"),
    RuntimeEnvVar("MTG_LLM_TIMEOUT_SEC", default="45"),
    RuntimeEnvVar("MTG_VERTEX_PROXY_RESEARCH", default="false", agent_engine=False),
    RuntimeEnvVar("MTG_VERTEX_PROXY_CHAT", default="false", agent_engine=False),
    RuntimeEnvVar("MTG_VERTEX_FALLBACK_TO_LOCAL", default="true", agent_engine=False),
    RuntimeEnvVar("MTG_VERTEX_AGENT_ENGINE_RESOURCE"),
    RuntimeEnvVar("MTG_VERTEX_AGENT_ENGINE_TIMEOUT_SEC", default="60"),
    RuntimeEnvVar("MTG_CHAT_ENABLE_CLARIFICATION", default="true"),
    RuntimeEnvVar("MTG_CHAT_MAX_CLARIFICATION_TURNS", default="1"),
    RuntimeEnvVar("LANGSMITH_API_KEY_SECRET_RESOURCE"),
    RuntimeEnvVar("LANGSMITH_API_KEY_SECRET"),
    RuntimeEnvVar("MTG_RAG_CORPUS_URI"),
    RuntimeEnvVar("MTG_LOCAL_RETRIEVER_CARDS_CSV"),
    RuntimeEnvVar("MTG_LOCAL_RETRIEVER_DECKS_DIR"),
    RuntimeEnvVar("MTG_LOCAL_RETRIEVER_META_JSON_PATHS"),
    RuntimeEnvVar("MTG_LOCAL_RETRIEVER_ENABLE_SEMANTIC"),
    RuntimeEnvVar("MTG_LOCAL_RETRIEVER_LEXICAL_WEIGHT"),
    RuntimeEnvVar("MTG_LOCAL_RETRIEVER_SEMANTIC_WEIGHT"),
]


RUNTIME_ENV_DEFAULTS: Dict[str, str] = {item.name: item.default for item in RUNTIME_ENV_VARS}


def _target_enabled(spec: RuntimeEnvVar, target: str) -> bool:
    normalized = str(target or "all").strip().lower()
    if normalized == "backend":
        return spec.backend
    if normalized in {"agent", "agent-engine", "agent_engine"}:
        return spec.agent_engine
    return True


def runtime_env_specs(target: str = "all") -> List[RuntimeEnvVar]:
    return [item for item in RUNTIME_ENV_VARS if _target_enabled(item, target)]


def runtime_env_names(target: str = "all") -> List[str]:
    return [item.name for item in runtime_env_specs(target)]


def runtime_env_default(name: str, fallback: str = "") -> str:
    value = RUNTIME_ENV_DEFAULTS.get(str(name), fallback)
    return str(value or fallback)


def collect_runtime_env_values(
    *,
    target: str = "all",
    include_defaults: bool = False,
    environ: Mapping[str, str] | None = None,
) -> Dict[str, str]:
    env = environ if environ is not None else os.environ
    result: Dict[str, str] = {}

    for spec in runtime_env_specs(target):
        raw = str(env.get(spec.name, "")).strip()
        if raw:
            result[spec.name] = raw
            continue
        if include_defaults and str(spec.default).strip():
            result[spec.name] = str(spec.default).strip()

    return result


def _escape_gcloud_value(value: str) -> str:
    escaped = str(value)
    escaped = escaped.replace("\\", "\\\\")
    escaped = escaped.replace(",", "\\,")
    escaped = escaped.replace("=", "\\=")
    return escaped


def to_gcloud_set_env_vars(values: Mapping[str, str]) -> str:
    parts: List[str] = []
    for key in sorted(values):
        val = _escape_gcloud_value(values[key])
        parts.append(f"{key}={val}")
    return ",".join(parts)


def build_gcloud_env_arg(
    *,
    target: str = "backend",
    include_defaults: bool = True,
    environ: Mapping[str, str] | None = None,
) -> str:
    values = collect_runtime_env_values(
        target=target,
        include_defaults=include_defaults,
        environ=environ,
    )
    return to_gcloud_set_env_vars(values)
