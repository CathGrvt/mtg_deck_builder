from __future__ import annotations

import os
from typing import Dict

_SECRET_CACHE: Dict[str, str] = {}


def _project_id_from_env() -> str:
    for name in ("GOOGLE_CLOUD_PROJECT", "GCP_PROJECT", "GCLOUD_PROJECT"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return ""


def _normalize_secret_resource(secret_ref: str, project_id: str) -> str:
    raw = str(secret_ref or "").strip()
    if not raw:
        return ""

    if raw.startswith("projects/"):
        if "/versions/" in raw:
            return raw
        return raw.rstrip("/") + "/versions/latest"

    if raw.startswith("secrets/"):
        if "/versions/" in raw:
            return f"projects/{project_id}/{raw}"
        return f"projects/{project_id}/{raw.rstrip('/')}/versions/latest"

    if not project_id:
        raise ValueError("GOOGLE_CLOUD_PROJECT is required when secret ref is not fully-qualified.")
    return f"projects/{project_id}/secrets/{raw}/versions/latest"


def _access_secret_version(resource_name: str) -> str:
    try:
        from google.cloud import secretmanager
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "google-cloud-secret-manager is required for Secret Manager key resolution."
        ) from exc

    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(request={"name": resource_name})
    payload = getattr(response, "payload", None)
    data = getattr(payload, "data", b"")
    if isinstance(data, bytes):
        return data.decode("utf-8")
    return str(data or "")


def resolve_secret_value(secret_ref: str, project_id: str = "") -> str:
    resolved_project = str(project_id or "").strip() or _project_id_from_env()
    resource_name = _normalize_secret_resource(secret_ref=secret_ref, project_id=resolved_project)
    if not resource_name:
        return ""

    cached = _SECRET_CACHE.get(resource_name)
    if cached:
        return cached

    value = _access_secret_version(resource_name).strip()
    if value:
        _SECRET_CACHE[resource_name] = value
    return value


def resolve_openai_api_key(api_key_env: str = "OPENAI_API_KEY") -> str:
    existing = os.getenv(api_key_env, "").strip()
    if existing:
        return existing

    secret_ref = os.getenv("MTG_OPENAI_API_KEY_SECRET_RESOURCE", "").strip()
    if not secret_ref:
        secret_ref = os.getenv("MTG_OPENAI_API_KEY_SECRET", "").strip()
    if not secret_ref:
        return ""

    try:
        resolved = resolve_secret_value(secret_ref=secret_ref)
    except Exception:
        return ""

    if resolved:
        os.environ[api_key_env] = resolved
    return resolved
