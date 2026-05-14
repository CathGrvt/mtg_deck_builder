from __future__ import annotations

from typing import Any, Dict, Optional

from gcp_agent_runtime.contracts import DeckRecommendationRequest
from gcp_agent_runtime.coordinator import RootCoordinatorAgent


class CloudRunAgentAdapter:
    """
    Thin API adapter used by Cloud Run service to proxy requests to Agent Engine
    (or local fallback coordinator during development).
    """

    def __init__(self, coordinator: Optional[RootCoordinatorAgent] = None):
        self.coordinator = coordinator or RootCoordinatorAgent()

    def handle_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        request = DeckRecommendationRequest.from_dict(dict(payload))
        response = self.coordinator.run(request)
        return response.to_dict()


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

    app = FastAPI(title="MTG Deck Builder Adapter", version="1.0.0")
    resolved = adapter or CloudRunAgentAdapter()

    @app.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/deck/recommend")
    def recommend(payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return resolved.handle_request(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc

    return app
