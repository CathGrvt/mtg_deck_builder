from __future__ import annotations

import hashlib
import os
from typing import Optional, Tuple


def is_gcs_uri(path_or_uri: str) -> bool:
    return str(path_or_uri or "").strip().lower().startswith("gs://")


def parse_gcs_uri(uri: str) -> Tuple[str, str]:
    raw = str(uri or "").strip()
    if not raw.startswith("gs://"):
        raise ValueError(f"Not a GCS URI: {uri}")
    remainder = raw[5:]
    bucket, _, blob = remainder.partition("/")
    if not bucket:
        raise ValueError("GCS URI must include bucket name.")
    if not blob:
        raise ValueError("GCS URI must include object path.")
    return bucket, blob


def _resolve_cache_dir(cache_dir: Optional[str]) -> str:
    if cache_dir and str(cache_dir).strip():
        return str(cache_dir)
    return os.getenv("MTG_GCS_CACHE_DIR", os.path.join(".cache", "mtg_gcs"))


def resolve_uri_to_local_path(
    path_or_uri: str,
    cache_dir: Optional[str] = None,
    storage_client=None,
) -> str:
    value = str(path_or_uri or "").strip()
    if not value:
        raise ValueError("path_or_uri is required.")

    if not is_gcs_uri(value):
        return value

    bucket_name, blob_name = parse_gcs_uri(value)
    resolved_cache = _resolve_cache_dir(cache_dir)
    os.makedirs(resolved_cache, exist_ok=True)

    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    filename = os.path.basename(blob_name) or "artifact"
    local_path = os.path.join(resolved_cache, f"{digest}_{filename}")
    if os.path.isfile(local_path):
        return local_path

    client = storage_client
    if client is None:
        try:
            from google.cloud import storage
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "google-cloud-storage is required for gs:// inputs. Install requirements-gcp.txt."
            ) from exc
        client = storage.Client()

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)
    return local_path
