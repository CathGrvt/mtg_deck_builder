from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from gcp_agent_runtime.backend_services import ChatBackendService, ResearchBackendService
from gcp_agent_runtime.contracts import DeckRecommendationRequest
from gcp_agent_runtime.coordinator import RootCoordinatorAgent
from gcp_agent_runtime.vertex_agent_engine import VertexAgentEngineClient


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name, "").strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


@dataclass
class AdapterSettings:
    backend_mode: str = "local"
    vertex_fallback_to_local: bool = True
    vertex_proxy_research: bool = False
    vertex_proxy_chat: bool = False

    @classmethod
    def from_env(cls) -> "AdapterSettings":
        mode = os.getenv("MTG_BACKEND_MODE", "local").strip().lower() or "local"
        if mode not in {"local", "vertex"}:
            raise ValueError("MTG_BACKEND_MODE must be either 'local' or 'vertex'.")
        return cls(
            backend_mode=mode,
            vertex_fallback_to_local=_bool_env("MTG_VERTEX_FALLBACK_TO_LOCAL", True),
            vertex_proxy_research=_bool_env("MTG_VERTEX_PROXY_RESEARCH", False),
            vertex_proxy_chat=_bool_env("MTG_VERTEX_PROXY_CHAT", False),
        )


class CloudRunAgentAdapter:
    """
    API adapter used by Cloud Run service to serve deck recommendation,
    research, and chat endpoints.
    """

    def __init__(
        self,
        coordinator: Optional[RootCoordinatorAgent] = None,
        settings: Optional[AdapterSettings] = None,
        vertex_client: Optional[VertexAgentEngineClient] = None,
        research_service: Optional[ResearchBackendService] = None,
        chat_service: Optional[ChatBackendService] = None,
    ):
        self.settings = settings or AdapterSettings.from_env()
        self.coordinator = coordinator or RootCoordinatorAgent()
        self.vertex_client = vertex_client or VertexAgentEngineClient()
        self.research_service = research_service or ResearchBackendService()
        self.chat_service = chat_service or ChatBackendService()

    def _handle_deck_local(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = DeckRecommendationRequest.from_dict(dict(payload))
        response = self.coordinator.run(request)
        return response.to_dict()

    def handle_recommendation(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        validated = DeckRecommendationRequest.from_dict(dict(payload))

        if self.settings.backend_mode == "vertex":
            try:
                result = self.vertex_client.recommend(validated.to_dict())
                if isinstance(result, dict) and result:
                    return result
                raise RuntimeError("Vertex Agent Engine returned an empty response.")
            except Exception:
                if not self.settings.vertex_fallback_to_local:
                    raise
                return self._handle_deck_local(validated.to_dict())

        return self._handle_deck_local(validated.to_dict())

    def handle_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Backward-compatible alias used by existing callers/tests.
        return self.handle_recommendation(payload)

    def handle_research(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.settings.backend_mode == "vertex" and self.settings.vertex_proxy_research:
            try:
                return self.vertex_client.run_research(dict(payload))
            except Exception:
                if not self.settings.vertex_fallback_to_local:
                    raise
        return self.research_service.run(dict(payload))

    def handle_chat(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self.settings.backend_mode == "vertex" and self.settings.vertex_proxy_chat:
            try:
                return self.vertex_client.run_chat(dict(payload))
            except Exception:
                if not self.settings.vertex_fallback_to_local:
                    raise
        return self.chat_service.run(dict(payload))


def create_fastapi_app(adapter: Optional[CloudRunAgentAdapter] = None):
    """
    Optional FastAPI factory for Cloud Run.
    Import errors are deferred so local tests can run without FastAPI installed.
    """
    try:
        from fastapi import FastAPI, HTTPException
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "FastAPI is not installed. Install requirements-gcp.txt to run the backend adapter service."
        ) from exc

    app = FastAPI(title="MTG Deck Builder Adapter", version="1.1.0")
    resolved = adapter or CloudRunAgentAdapter()

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/deck/recommend")
    def recommend(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return resolved.handle_recommendation(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc

    @app.post("/v1/research/run")
    def research(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return resolved.handle_research(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc

    @app.post("/v1/chat/respond")
    def chat(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return resolved.handle_chat(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc

    return app
