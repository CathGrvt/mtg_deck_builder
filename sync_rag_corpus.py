from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Optional

from research_pipeline.retrieval.corpus import build_domain_corpus


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and optionally upload MTG corpus snapshot for RAG ingestion."
    )
    parser.add_argument("--cards", default="data/commander_cards.csv")
    parser.add_argument("--decks", default="current_commander_decks")
    parser.add_argument("--meta-json", nargs="*", default=None)
    parser.add_argument("--out-dir", default="rag_exports")
    parser.add_argument(
        "--gcs-uri",
        default="",
        help="Optional destination prefix, e.g. gs://my-bucket/mtg-rag/",
    )
    return parser.parse_args()


def _upload_to_gcs(local_path: str, gcs_uri: str) -> Optional[str]:
    if not gcs_uri.strip():
        return None
    if not gcs_uri.startswith("gs://"):
        raise ValueError("gcs-uri must start with gs://")
    try:
        from google.cloud import storage
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "google-cloud-storage is required for GCS upload. Install requirements-gcp.txt."
        ) from exc

    path = gcs_uri[5:]
    bucket_name, _, blob_prefix = path.partition("/")
    if not bucket_name:
        raise ValueError("gcs-uri must include a bucket name.")
    blob_name = "/".join(part for part in [blob_prefix.rstrip("/"), os.path.basename(local_path)] if part)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket_name}/{blob_name}"


def main() -> None:
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    export_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    export_path = os.path.join(args.out_dir, f"rag_corpus_{export_id}.jsonl")

    chunks = build_domain_corpus(
        cards_csv=args.cards,
        decks_dir=args.decks,
        meta_json_paths=args.meta_json,
    )
    if not chunks:
        raise ValueError("No chunks built. Check cards/decks/meta paths.")

    with open(export_path, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    uploaded_uri = _upload_to_gcs(export_path, args.gcs_uri)

    print("RAG corpus export complete.")
    print(f"chunks: {len(chunks)}")
    print(f"local_path: {export_path}")
    if uploaded_uri:
        print(f"gcs_path: {uploaded_uri}")


if __name__ == "__main__":
    main()
