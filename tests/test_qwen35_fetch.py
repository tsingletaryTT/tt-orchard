"""scripts/qwen35/fetch_snapshot.py: when a cached blob is reused, and what it writes under refs/. No network."""
import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location("q35_fetch_snapshot", ROOT / "scripts" / "qwen35" / "fetch_snapshot.py")
fs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fs)

DATA = b"x" * 1000
GOOD = hashlib.sha256(DATA).hexdigest()


def test_an_lfs_blob_of_the_right_size_is_reused_only_when_its_sha256_matches(tmp_path):
    blob = tmp_path / "b"
    blob.write_bytes(b"y" * 1000)                     # same size, other bytes: a truncated-then-padded or corrupted download
    assert fs.cached_blob_ok(blob, 1000, GOOD) is False
    blob.write_bytes(DATA)
    assert fs.cached_blob_ok(blob, 1000, GOOD) is True


def test_a_blob_of_the_wrong_size_or_a_missing_one_is_not_reused(tmp_path):
    blob = tmp_path / "b"
    assert fs.cached_blob_ok(blob, 1000, GOOD) is False
    blob.write_bytes(DATA[:10])
    assert fs.cached_blob_ok(blob, 1000, GOOD) is False


def test_a_small_file_without_an_lfs_hash_is_reused_by_size(tmp_path):
    blob = tmp_path / "b"
    blob.write_bytes(b"{}")
    assert fs.cached_blob_ok(blob, 2, None) is True


def test_a_pinned_commit_does_not_move_refs_main(tmp_path):
    (tmp_path / "refs").mkdir()
    (tmp_path / "refs" / "main").write_text("a" * 40)
    fs.record_ref(tmp_path, "b" * 40, pinned=True)
    assert (tmp_path / "refs" / "main").read_text() == "a" * 40


def test_an_unpinned_download_records_the_branch(tmp_path):
    fs.record_ref(tmp_path, "b" * 40, pinned=False)
    assert (tmp_path / "refs" / "main").read_text() == "b" * 40


def test_a_blob_of_the_wrong_size_is_not_reused_even_without_a_hash(tmp_path):
    blob = tmp_path / "b"
    blob.write_bytes(b"{}")
    assert fs.cached_blob_ok(blob, 3, None) is False


class _Info:
    def __init__(self, name, data):
        self.rfilename, self.size, self.blob_id = name, len(data), "b" * 40
        self.lfs = type("L", (), {"sha256": hashlib.sha256(data).hexdigest()})()


def _fake_hub(monkeypatch, files: dict):
    import sys
    import types

    class Api:
        def model_info(self, repo, revision=None, files_metadata=False):
            return types.SimpleNamespace(siblings=[_Info(n, d) for n, d in files.items()])

    class File:
        def __init__(self, data):
            self.data, self.pos = data, 0

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def seek(self, n):
            self.pos = n

        def read(self, n):
            return self.data[self.pos:self.pos + n]

    class FS:
        def open(self, remote, mode):
            return File(files[remote.split("/", 1)[1].split("/", 1)[1]])

    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(HfApi=Api, HfFileSystem=FS))


def test_main_refetches_a_cached_blob_with_the_right_size_and_other_bytes_and_leaves_refs_alone(tmp_path, monkeypatch):
    rev = "c" * 40
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    _fake_hub(monkeypatch, {"model.safetensors": DATA})
    root = tmp_path / "models--org--name"
    (root / "blobs").mkdir(parents=True)
    (root / "refs").mkdir()
    (root / "refs" / "main").write_text("a" * 40)
    (root / "blobs" / GOOD).write_bytes(b"y" * 1000)           # right name, right size, wrong bytes
    assert fs.main(["x", "org/name", rev]) == 0
    assert (root / "blobs" / GOOD).read_bytes() == DATA
    assert (root / "snapshots" / rev / "model.safetensors").read_bytes() == DATA
    assert (root / "refs" / "main").read_text() == "a" * 40
