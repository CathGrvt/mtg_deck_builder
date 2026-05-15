from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict

from mtg_shared.env import parse_int_value
from mtg_shared.runtime_env import runtime_env_default


@dataclass
class VertexAgentEngineConfig:
    resource_name: str = ""
    timeout_sec: int = 60
    project: str = ""
    location: str = "us-central1"

    @classmethod
    def from_env(cls) -> "VertexAgentEngineConfig":
        return cls(
            resource_name=os.getenv("MTG_VERTEX_AGENT_ENGINE_RESOURCE", "").strip(),
            timeout_sec=parse_int_value(
                os.getenv(
                    "MTG_VERTEX_AGENT_ENGINE_TIMEOUT_SEC",
                    runtime_env_default("MTG_VERTEX_AGENT_ENGINE_TIMEOUT_SEC", "60"),
                ),
                default=int(runtime_env_default("MTG_VERTEX_AGENT_ENGINE_TIMEOUT_SEC", "60")),
                minimum=5,
            ),
            project=os.getenv("GOOGLE_CLOUD_PROJECT", "").strip(),
            location=(
                os.getenv(
                    "GOOGLE_CLOUD_LOCATION",
                    runtime_env_default("GOOGLE_CLOUD_LOCATION", "us-central1"),
                ).strip()
                or runtime_env_default("GOOGLE_CLOUD_LOCATION", "us-central1")
            ),
        )


class VertexAgentEngineClient:
    def __init__(self, config: VertexAgentEngineConfig | None = None):
        self.config = config or VertexAgentEngineConfig.from_env()

    def recommend(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._invoke_operation(operation="deck_recommendation", payload=payload)

    def run_research(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._invoke_operation(operation="research_run", payload=payload)

    def run_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._invoke_operation(operation="chat_respond", payload=payload)

    def _invoke_operation(self, operation: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.config.resource_name:
            raise ValueError("MTG_VERTEX_AGENT_ENGINE_RESOURCE is required for vertex backend mode.")

        try:
            import vertexai
            from vertexai import agent_engines
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Vertex backend mode requested but Vertex SDK is unavailable. Install requirements-gcp.txt."
            ) from exc

        init_kwargs = {}
        if self.config.project:
            init_kwargs["project"] = self.config.project
        if self.config.location:
            init_kwargs["location"] = self.config.location
        if init_kwargs:
            vertexai.init(**init_kwargs)

        remote = agent_engines.get(self.config.resource_name)
        envelopes = [
            {"operation": operation, "payload": dict(payload)},
            dict(payload, _operation=operation),
            dict(payload),
        ]
        last_error: Exception | None = None
        raw = None
        for envelope in envelopes:
            try:
                raw = self._invoke_remote(remote=remote, payload=envelope)
                break
            except Exception as exc:
                last_error = exc
                continue
        if raw is None and last_error is not None:
            raise last_error
        normalized = self._normalize_response(raw)
        if not isinstance(normalized, dict):
            raise RuntimeError("Vertex Agent Engine response could not be converted to JSON object.")
        return normalized

    def _invoke_remote(self, remote, payload: Dict[str, Any]) -> Any:
        timeout = int(self.config.timeout_sec)
        attempts = [
            ("query", (), {"input": payload, "timeout": timeout}),
            ("query", (payload,), {}),
            ("run", (), {"input": payload, "timeout": timeout}),
            ("run", (payload,), {}),
            ("invoke", (payload,), {"timeout": timeout}),
            ("invoke", (payload,), {}),
            ("chat", (), {"message": payload, "timeout": timeout}),
            ("chat", (payload,), {}),
        ]

        last_error: Exception | None = None
        for method_name, args, kwargs in attempts:
            method = getattr(remote, method_name, None)
            if method is None:
                continue
            try:
                return method(*args, **kwargs)
            except TypeError as exc:
                last_error = exc
                continue
            except Exception as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise RuntimeError(f"Failed to invoke Vertex Agent Engine: {last_error}") from last_error
        raise RuntimeError("No compatible invocation method found on Vertex Agent Engine client.")

    @staticmethod
    def _normalize_response(response: Any) -> Dict[str, Any]:
        if isinstance(response, dict):
            return response

        if hasattr(response, "to_dict"):
            payload = response.to_dict()
            if isinstance(payload, dict):
                return payload

        if hasattr(response, "model_dump"):
            payload = response.model_dump()
            if isinstance(payload, dict):
                return payload

        maybe_output = getattr(response, "output", None)
        if isinstance(maybe_output, dict):
            return maybe_output

        if isinstance(response, str):
            text = response.strip()
            try:
                payload = json.loads(text)
                if isinstance(payload, dict):
                    return payload
            except json.JSONDecodeError:
                pass
            return {"summary": text}

        return {}
