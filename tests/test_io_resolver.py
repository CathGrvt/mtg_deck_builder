import tempfile
import unittest
from pathlib import Path

from research_pipeline.eval.dataset import load_eval_cases
from research_pipeline.io_resolver import parse_gcs_uri, resolve_uri_to_local_path


class FakeBlob:
    def __init__(self, content: str):
        self.content = content
        self.download_count = 0

    def download_to_filename(self, path: str) -> None:
        self.download_count += 1
        Path(path).write_text(self.content, encoding="utf-8")


class FakeBucket:
    def __init__(self, blob_map):
        self.blob_map = blob_map

    def blob(self, name: str):
        return self.blob_map[name]


class FakeStorageClient:
    def __init__(self, bucket_map):
        self.bucket_map = bucket_map

    def bucket(self, name: str):
        return self.bucket_map[name]


class IoResolverTests(unittest.TestCase):
    def test_parse_gcs_uri(self):
        bucket, blob = parse_gcs_uri("gs://my-bucket/path/to/file.jsonl")
        self.assertEqual(bucket, "my-bucket")
        self.assertEqual(blob, "path/to/file.jsonl")

    def test_resolve_local_path_passthrough(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = str(Path(tmp_dir) / "sample.jsonl")
            resolved = resolve_uri_to_local_path(path)
            self.assertEqual(resolved, path)

    def test_resolve_gcs_downloads_and_reuses_cache(self):
        blob = FakeBlob('{"id":"case-a","topic":"Topic A"}\n')
        client = FakeStorageClient(
            {"bucket-a": FakeBucket({"datasets/topics.jsonl": blob})}
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            uri = "gs://bucket-a/datasets/topics.jsonl"
            first = resolve_uri_to_local_path(uri, cache_dir=tmp_dir, storage_client=client)
            second = resolve_uri_to_local_path(uri, cache_dir=tmp_dir, storage_client=client)

            self.assertEqual(first, second)
            self.assertTrue(Path(first).is_file())
            self.assertEqual(blob.download_count, 1)

    def test_eval_loader_reads_gcs_uri(self):
        blob = FakeBlob('{"id":"case-a","topic":"Topic A"}\n{"id":"case-b","topic":"Topic B"}\n')
        client = FakeStorageClient(
            {"bucket-a": FakeBucket({"eval/topics.jsonl": blob})}
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            cases = load_eval_cases(
                "gs://bucket-a/eval/topics.jsonl",
                cache_dir=tmp_dir,
                storage_client=client,
            )
            self.assertEqual(len(cases), 2)
            self.assertEqual(cases[0].id, "case-a")


if __name__ == "__main__":
    unittest.main()
