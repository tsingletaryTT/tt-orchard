# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Guards of scripts/acestep/acestep_run.py for the two ways a run can touch the wrong files (issues 270 and 264). No chip, no ACE-Step code, no weights."""
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "acestep"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


run = _load("acestep_run")


# ---- issue 270: the model code files the run uses must be the recorded ones and real files ----
def _model_code_tree(tmp_path, files=None):
    code = tmp_path / "code"
    d = code / "acestep" / "models" / "turbo"
    d.mkdir(parents=True)
    (d / "__init__.py").write_text("")
    files = files or {"a.py": "x = 1\n", "b.py": "y = 2\n"}
    for n, t in files.items():
        (d / n).write_text(t)
    pinned = {"turbo": {n: hashlib.sha256(t.encode()).hexdigest() for n, t in files.items()}}
    return code, pinned


def test_model_code_check_passes_on_the_pinned_files(tmp_path):
    code, pinned = _model_code_tree(tmp_path)
    assert run.model_code_problems(code, pinned) == []


def test_model_code_check_names_a_changed_missing_and_extra_file(tmp_path):
    code, pinned = _model_code_tree(tmp_path)
    d = code / "acestep" / "models" / "turbo"
    (d / "a.py").write_text("x = 3\n")
    (d / "b.py").unlink()
    (d / "c.py").write_text("z = 0\n")
    got = " | ".join(run.model_code_problems(code, pinned))
    assert "turbo/a.py" in got and "differs" in got
    assert "turbo/b.py" in got and "missing" in got
    assert "turbo/c.py" in got and "not recorded" in got


def test_model_code_check_refuses_links_even_when_the_content_matches(tmp_path):
    code, pinned = _model_code_tree(tmp_path)
    d = code / "acestep" / "models" / "turbo"
    blob = tmp_path / "blob"
    blob.write_text("x = 1\n")
    (d / "a.py").unlink()
    (d / "a.py").symlink_to(blob)
    assert any("link" in x and "turbo/a.py" in x for x in run.model_code_problems(code, pinned))


def test_model_code_check_refuses_a_code_directory_that_is_a_link(tmp_path):
    real, pinned = _model_code_tree(tmp_path / "snap")
    code = tmp_path / "code"
    code.symlink_to(real)
    assert any("link" in x for x in run.model_code_problems(code, pinned))


def test_the_recorded_model_code_hashes_cover_the_turbo_files():
    assert set(run.MODEL_CODE_SHA256["turbo"]) == {"configuration_acestep_v15.py", "modeling_acestep_v15_turbo.py"}
    assert all(len(h) == 64 for h in run.MODEL_CODE_SHA256["turbo"].values())


def test_session_runs_the_model_code_gate_before_anything_else(tmp_path, monkeypatch):
    """The wiring: Session.__init__ must call the gate and stop on a problem (a bundle directory with different model code)."""
    code, _ = _model_code_tree(tmp_path)  # real files, but not the recorded content
    snap = tmp_path / "snap"
    snap.mkdir()
    with pytest.raises(SystemExit) as e:
        run.Session(device="cpu", bundle_code=code, snapshot=snap, root=tmp_path / "root")
    assert "model code" in str(e.value) and "not recorded" in str(e.value)  # the fixture's files are not in the recorded list


# ---- issue 264: a directory ACE-Step may write into must not lead into the Hugging Face cache ----
def test_cache_write_hazards_flag_a_root_inside_the_cache_and_small_files_linked_to_it(tmp_path):
    cache = tmp_path / "hub"
    snap = cache / "models--x--y" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    (snap / "w.safetensors").write_bytes(b"0" * 10)
    assert any("inside the Hugging Face cache" in x for x in run.cache_write_hazards(snap, [cache]))
    lk = tmp_path / "lk"
    lk.symlink_to(snap)
    assert any("inside the Hugging Face cache" in x for x in run.cache_write_hazards(lk, [cache]))
    priv = tmp_path / "priv"
    priv.mkdir()
    (priv / "config.json").write_text("{}")
    (priv / "w.safetensors").symlink_to(snap / "w.safetensors")
    assert run.cache_write_hazards(priv, [cache]) == []  # a copied config and a linked weight file are fine
    (priv / "config.json").unlink()
    (priv / "config.json").symlink_to(snap / "config.json")
    assert any("config.json" in x for x in run.cache_write_hazards(priv, [cache]))  # a linked small file would be rewritten in the cache


def test_hf_cache_dirs_follow_the_hub_variables(monkeypatch, tmp_path):
    for k in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "h"))
    assert (tmp_path / "h" / "hub").resolve() in run.hf_cache_dirs()
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "c"))
    assert (tmp_path / "c").resolve() in run.hf_cache_dirs()


def test_session_refuses_a_project_root_inside_the_cache(tmp_path, monkeypatch):
    """The wiring for 264: the root ACE-Step writes under is checked against the cache before the private tree is built."""
    cache = tmp_path / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    root = cache / "models--x--y" / "snapshots" / "abc"
    root.mkdir(parents=True)
    monkeypatch.setattr(run, "model_code_problems", lambda *a, **k: [])
    with pytest.raises(SystemExit) as e:
        run.Session(device="cpu", bundle_code=tmp_path, snapshot=tmp_path, root=root)
    assert "Hugging Face cache" in str(e.value)


def test_a_stale_checkpoints_link_into_the_cache_under_the_root_is_not_a_root_hazard(tmp_path):
    """An older run left <root>/checkpoints as a link to a cache snapshot; Session replaces it, so judging the root alone must not refuse it."""
    cache = tmp_path / "hub"
    snap = cache / "models--x--y" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    root = tmp_path / "root"
    root.mkdir()
    (root / "checkpoints").symlink_to(snap)
    assert run.cache_write_hazards(root, [cache], links=False) == []
    assert run.cache_write_hazards(root, [cache])  # the full scan does see it


def test_the_recorded_hashes_match_the_pinned_bundle_when_it_is_on_this_machine():
    """Against a real copy of the pinned bundle, when ORCHARD_ACESTEP_BUNDLE_CODE names one (skipped otherwise)."""
    import os
    d = os.environ.get("ORCHARD_ACESTEP_BUNDLE_CODE")
    if not d:
        pytest.skip("ORCHARD_ACESTEP_BUNDLE_CODE is not set")
    assert run.model_code_problems(d) == []


def test_private_tree_copies_everything_but_large_weights_and_replaces_an_old_link(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    snap = cache / "models--x--y" / "snapshots" / "abc"
    (snap / "m").mkdir(parents=True)
    (snap / "m" / "config.json").write_text("{}")
    (snap / "m" / "tokenizer.json").write_bytes(b"0" * ((8 << 20) + 1))   # large, but not a weight: copied
    (snap / "m" / "w.safetensors").write_bytes(b"0" * ((8 << 20) + 1))     # large weight: linked
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    root = tmp_path / "root"
    root.mkdir()
    (root / "checkpoints").symlink_to(snap)                                  # what the bundle's own server leaves behind
    run.build_private_tree(snap, root / "checkpoints")
    t = root / "checkpoints" / "m"
    assert not (root / "checkpoints").is_symlink()
    assert not (t / "config.json").is_symlink() and not (t / "tokenizer.json").is_symlink()
    assert (t / "w.safetensors").is_symlink()
    (t / "config.json").write_text("changed")                                # what the handler does
    assert (snap / "m" / "config.json").read_text() == "{}"                  # the cache is untouched


def test_private_tree_stops_when_the_result_would_still_lead_into_the_cache(tmp_path, monkeypatch):
    cache = tmp_path / "hub"
    snap = cache / "models--x--y" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    with pytest.raises(SystemExit):
        run.build_private_tree(snap, cache / "models--x--y" / "private")     # a destination inside the cache


def test_a_destination_inside_the_cache_is_refused_before_anything_is_deleted_or_written(tmp_path, monkeypatch):
    """Review finding: the guard used to run after the delete and copy, so the refusal had already damaged the cache."""
    cache = tmp_path / "hub"
    snap = cache / "models--x--y" / "snapshots" / "abc"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    dest = cache / "models--x--y" / "private"
    dest.mkdir()
    (dest / "keep.json").write_text("original")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    before = sorted(p.relative_to(cache).as_posix() for p in cache.rglob("*"))
    with pytest.raises(SystemExit):
        run.build_private_tree(snap, dest)
    assert (dest / "keep.json").read_text() == "original"
    assert sorted(p.relative_to(cache).as_posix() for p in cache.rglob("*")) == before
