# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 1 tries a model that ships its own code instead of refusing it.

The operator's rule for stage 1 (2026-11-08): "goonies never say die", "i'll try anything once (or twice)", "why not?".
Spark-X2.6-4B-Loop-mHC carries `modeling_spark.py` (auto_map), transformers refuses to load it without trust_remote_code, and the gate used to
stop there with a failed `loads` check. Now the gate catches that one refusal, scrubs the credentials it can scrub, turns the hub and
transformers offline, lists the hash of every .py file it is about to let run, and loads again with trust_remote_code=True. The
report says so. This is not a sandbox: the code runs with the user's file permissions and may use the network.
Fakes only (no torch, no transformers needed)."""
import importlib.util
import os
from pathlib import Path

import pytest

TEMPLATE = Path(__file__).resolve().parent.parent / "orchard" / "skills" / "reference-gate-templates" / "reference_gate.py"
spec = importlib.util.spec_from_file_location("reference_gate_remote_under_test", TEMPLATE)
rg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rg)

REFUSAL = ("ValueError: The repository /x contains custom code which must be executed to correctly load the model. "
           "Please pass the argument `trust_remote_code=True` to allow custom code to be run.")


@pytest.fixture(autouse=True)
def clean_state(monkeypatch, tmp_path):
    """allow_remote_code changes os.environ directly, so the whole environment is saved and restored, and the scratch home and the evidence
    directory live under tmp_path, never beside the shipped template (review of PR 597)."""
    saved = dict(os.environ)
    rg.REMOTE_CODE.clear()
    rg.REMOTE_CODE.update({"used": False, "files": {}, "scrubbed": []})
    monkeypatch.setattr(rg, "HOME_SCRATCH", tmp_path / "default-home")
    monkeypatch.setattr(rg, "EVIDENCE", tmp_path / "evidence")
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "GH_TOKEN", "SOME_API_KEY", "KEEP_ME"):
        monkeypatch.delenv(k, raising=False)
    yield
    os.environ.clear()
    os.environ.update(saved)


def snapshot_with_code(tmp_path):
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "modeling_spark.py").write_text("class Spark: pass\n")
    (snap / "config.json").write_text("{}")
    return snap


def test_a_refusal_is_retried_once_with_trust_remote_code(tmp_path):
    snap = snapshot_with_code(tmp_path)
    calls = []

    def loader(path, **kw):
        calls.append(dict(kw))
        if not kw.get("trust_remote_code"):
            raise ValueError(REFUSAL)
        return "loaded"

    assert rg.with_remote_code(loader, str(snap), snapshot=snap) == "loaded"
    assert [bool(c.get("trust_remote_code")) for c in calls] == [False, True]
    assert rg.REMOTE_CODE["used"] is True


def test_once_allowed_later_calls_go_straight_to_trust_remote_code(tmp_path):
    snap = snapshot_with_code(tmp_path)
    calls = []

    def loader(path, **kw):
        calls.append(kw.get("trust_remote_code"))
        return "ok"

    rg.with_remote_code(lambda p, **kw: (_ for _ in ()).throw(ValueError(REFUSAL)) if not kw.get("trust_remote_code") else "ok",
                        str(snap), snapshot=snap)
    rg.with_remote_code(loader, str(snap), snapshot=snap)
    assert calls == [True]


def test_another_error_is_not_swallowed(tmp_path):
    snap = snapshot_with_code(tmp_path)

    def loader(path, **kw):
        raise ValueError("shapes do not match")

    with pytest.raises(ValueError, match="shapes do not match"):
        rg.with_remote_code(loader, str(snap), snapshot=snap)
    assert rg.REMOTE_CODE["used"] is False


def test_credentials_in_the_environment_are_scrubbed_and_named(tmp_path, monkeypatch):
    snap = snapshot_with_code(tmp_path)
    monkeypatch.setenv("HF_TOKEN", "hf_SECRET")
    monkeypatch.setenv("GH_TOKEN", "gh_SECRET")
    monkeypatch.setenv("SOME_API_KEY", "k")
    monkeypatch.setenv("KEEP_ME", "yes")
    rg.allow_remote_code(snap, scratch_home=tmp_path / "home")
    assert "HF_TOKEN" not in os.environ and "GH_TOKEN" not in os.environ and "SOME_API_KEY" not in os.environ
    assert os.environ["KEEP_ME"] == "yes"
    assert set(rg.REMOTE_CODE["scrubbed"]) >= {"HF_TOKEN", "GH_TOKEN", "SOME_API_KEY"}
    assert "hf_SECRET" not in repr(rg.REMOTE_CODE)           # names only, never values


def test_the_hub_and_transformers_go_offline_and_home_moves_to_scratch(tmp_path):
    snap = snapshot_with_code(tmp_path)
    rg.allow_remote_code(snap, scratch_home=tmp_path / "home")
    assert os.environ["HF_HUB_OFFLINE"] == "1" and os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HOME"] == str(tmp_path / "home") and (tmp_path / "home").is_dir()


def test_every_py_file_that_may_run_is_listed_with_its_sha256(tmp_path):
    snap = snapshot_with_code(tmp_path)
    (snap / "sub").mkdir()
    (snap / "sub" / "extra.py").write_text("x = 1\n")
    rg.allow_remote_code(snap, scratch_home=tmp_path / "home")
    files = rg.REMOTE_CODE["files"]
    assert set(files) == {"modeling_spark.py", "sub/extra.py"} and all(len(v) == 64 for v in files.values())


def test_load_model_passes_trust_remote_code_to_the_config_and_the_class(tmp_path, monkeypatch):
    """The wiring at the layer that used to fail: a config that raises the refusal gets loaded with trust_remote_code, and so does the
    class's from_pretrained."""
    snap = snapshot_with_code(tmp_path)
    seen = {}

    class Cfg:
        architectures = ["SparkForCausalLM"]
        model_type = "spark"

    class Stop(Exception):
        pass

    class Cls:
        @staticmethod
        def from_pretrained(path, **kw):
            seen["class_kw"] = kw
            raise Stop

    class AutoConfig:
        @staticmethod
        def from_pretrained(path, **kw):
            seen["config_kw"] = kw
            if not kw.get("trust_remote_code"):
                raise ValueError(REFUSAL)
            return Cfg()

    T = type("T", (), {"__version__": "4.57.0", "AutoConfig": AutoConfig, "SparkForCausalLM": Cls})

    class Torch:
        bfloat16 = "bf16"

    with pytest.raises(Stop):
        rg.load_model(Torch, T, snap)
    assert seen["config_kw"].get("trust_remote_code") is True and seen["class_kw"].get("trust_remote_code") is True


def test_a_link_or_a_pipe_named_py_is_listed_but_never_followed(tmp_path):
    snap = snapshot_with_code(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("do not hash me")
    os.symlink(secret, snap / "linked.py")
    os.mkfifo(snap / "pipe.py")
    rg.allow_remote_code(snap, scratch_home=tmp_path / "home")      # a FIFO that was opened for reading would hang here
    files = rg.REMOTE_CODE["files"]
    assert files["linked.py"] == "unreadable" and files["pipe.py"] == "not a regular file"
    assert len(files["modeling_spark.py"]) == 64


def test_the_offline_flags_reach_the_modules_that_were_imported_before(tmp_path, monkeypatch):
    """Review of PR 597: huggingface_hub and transformers 4.x read the environment when they are imported, so setting it afterwards does
    nothing. The gate sets the modules' own flags too."""
    import sys
    import types
    hub = types.SimpleNamespace(HF_HUB_OFFLINE=False)
    thub = types.SimpleNamespace(_is_offline_mode=False)
    monkeypatch.setitem(sys.modules, "huggingface_hub.constants", hub)
    monkeypatch.setitem(sys.modules, "transformers.utils.hub", thub)
    rg.allow_remote_code(snapshot_with_code(tmp_path), scratch_home=tmp_path / "home")
    assert hub.HF_HUB_OFFLINE is True and thub._is_offline_mode is True


def test_a_builtin_config_with_a_custom_model_class_is_also_retried(tmp_path):
    """Review of PR 597: only the model class asks for trust_remote_code. The class load goes through the same retry."""
    snap = snapshot_with_code(tmp_path)
    seen = []

    class Cfg:
        architectures = ["SparkForCausalLM"]
        model_type = "spark"

    class Stop(Exception):
        pass

    class Cls:
        @staticmethod
        def from_pretrained(path, **kw):
            seen.append(kw.get("trust_remote_code"))
            if not kw.get("trust_remote_code"):
                raise ValueError(REFUSAL)
            raise Stop

    AutoConfig = type("AutoConfig", (), {"from_pretrained": staticmethod(lambda path, **kw: Cfg())})
    T = type("T", (), {"__version__": "4.57.0", "AutoConfig": AutoConfig, "SparkForCausalLM": Cls})
    with pytest.raises(Stop):
        rg.load_model(type("Torch", (), {"bfloat16": "bf16"}), T, snap)
    assert seen == [None, True] and rg.REMOTE_CODE["used"] is True


def test_the_audit_is_written_before_the_trusted_retry_runs(tmp_path):
    snap = snapshot_with_code(tmp_path)
    audit_seen = []

    def loader(path, **kw):
        if not kw.get("trust_remote_code"):
            raise ValueError(REFUSAL)
        audit_seen.append((tmp_path / "evidence" / "remote-code-audit.json").is_file())
        raise RuntimeError("the code failed")

    with pytest.raises(RuntimeError):
        rg.with_remote_code(loader, str(snap), snapshot=snap)
    assert audit_seen == [True]
    import json
    audit = json.loads((tmp_path / "evidence" / "remote-code-audit.json").read_text())
    assert "modeling_spark.py" in audit["files"] and "scrubbed" in audit


# ---- issue 602: a built-in architecture name with an auto_map must load the shipped code ------------------------------------------

def _auto_map_fakes(auto_map):
    """The Auto classes load the LOCAL implementation without complaint when trust_remote_code is omitted (a built-in config such as LlamaConfig
    does that), so only an explicit trust_remote_code=True loads the shipped code. A refusal would hide that case (review of PR 607)."""
    seen = {"builtin": 0, "auto": [], "which": []}

    class Stop(Exception):
        pass

    class Builtin:
        @staticmethod
        def from_pretrained(path, **kw):
            seen["builtin"] += 1
            raise AssertionError("the built-in class was loaded and the shipped code skipped")

    def auto(name):
        class Auto:
            @staticmethod
            def from_pretrained(path, **kw):
                seen["auto"].append(kw.get("trust_remote_code"))
                seen["which"].append(name)
                if not kw.get("trust_remote_code"):
                    raise AssertionError("the local implementation loaded without trust_remote_code: the shipped code was skipped")
                raise Stop
        Auto.__name__ = name
        return Auto

    class Cfg:
        architectures = ["LlamaForCausalLM"]
        model_type = "llama"

    cfg = Cfg()
    if auto_map is not None:
        cfg.auto_map = auto_map
    AutoConfig = type("AutoConfig", (), {"from_pretrained": staticmethod(lambda path, **kw: cfg)})
    T = type("T", (), {"__version__": "4.57.0", "AutoConfig": AutoConfig, "LlamaForCausalLM": Builtin,
                       "AutoModelForCausalLM": auto("AutoModelForCausalLM"), "AutoModel": auto("AutoModel"),
                       "AutoModelForSequenceClassification": auto("AutoModelForSequenceClassification")})
    return T, seen, Stop


TORCH = type("Torch", (), {"bfloat16": "bf16"})


def test_a_built_in_architecture_with_an_auto_map_is_audited_and_loaded_with_trust_before_any_load(tmp_path):
    snap = snapshot_with_code(tmp_path)
    T, seen, Stop = _auto_map_fakes({"AutoModelForCausalLM": "modeling_spark.SparkForCausalLM"})
    with pytest.raises(Stop):
        rg.load_model(TORCH, T, snap)
    assert seen["builtin"] == 0
    assert seen["auto"] == [True]                              # the very first load already carries trust: no refusal is waited for
    assert rg.REMOTE_CODE["used"] is True and "modeling_spark.py" in rg.REMOTE_CODE["files"]
    assert (tmp_path / "evidence" / "remote-code-audit.json").is_file()


def test_without_an_auto_map_the_built_in_class_is_still_chosen(tmp_path):
    snap = snapshot_with_code(tmp_path)
    T, seen, _ = _auto_map_fakes(None)
    with pytest.raises(AssertionError, match="built-in class"):
        rg.load_model(TORCH, T, snap)
    assert seen["auto"] == [] and seen["builtin"] == 1 and rg.REMOTE_CODE["used"] is False


def test_an_auto_map_with_only_config_or_tokenizer_entries_keeps_the_built_in_class(tmp_path):
    snap = snapshot_with_code(tmp_path)
    T, seen, _ = _auto_map_fakes({"AutoConfig": "configuration_x.XConfig", "AutoTokenizer": ["tok.X", None]})
    with pytest.raises(AssertionError, match="built-in class"):
        rg.load_model(TORCH, T, snap)
    assert seen["auto"] == [] and rg.REMOTE_CODE["used"] is False


def test_json_key_order_does_not_decide_which_auto_class_loads(tmp_path):
    """A head architecture with no AutoModelForCausalLM entry takes AutoModel before any other model entry, whatever the order of the keys."""
    snap = snapshot_with_code(tmp_path)
    T, seen, Stop = _auto_map_fakes({"AutoModelForSequenceClassification": "m.Seq", "AutoModel": "m.Base"})
    with pytest.raises(Stop):
        rg.load_model(TORCH, T, snap)
    assert seen["which"] == ["AutoModel"]


def test_a_model_mapping_the_installed_transformers_cannot_dispatch_fails_instead_of_loading_the_built_in_class(tmp_path):
    """Review of PR 607: the only model mapping names an Auto class this transformers lacks. That is not a config with no model mapping:
    loading the built-in LlamaForCausalLM would skip the shipped code. The load fails, naming the class and the version."""
    snap = snapshot_with_code(tmp_path)
    T, seen, _ = _auto_map_fakes({"AutoModelForVideoGeneration": "modeling_x.X"})
    with pytest.raises(rg.UnsupportedDispatcher) as err:
        rg.load_model(TORCH, T, snap)
    assert "AutoModelForVideoGeneration" in str(err.value) and "4.57.0" in str(err.value)
    assert seen["builtin"] == 0 and seen["auto"] == []


def test_the_unsupported_dispatcher_error_reaches_the_load_report_text(tmp_path):
    """main() writes `f"{type(exc).__name__}: {exc}"` into the load report for any load failure; this is what it would say."""
    snap = snapshot_with_code(tmp_path)
    T, _, _ = _auto_map_fakes({"AutoModelForVideoGeneration": "modeling_x.X"})
    with pytest.raises(Exception) as err:
        rg.load_model(TORCH, T, snap)
    text = f"{type(err.value).__name__}: {err.value}"
    assert text.startswith("UnsupportedDispatcher: ") and "AutoModelForVideoGeneration" in text


def test_a_mapping_with_one_resolvable_entry_among_unresolvable_ones_still_loads(tmp_path):
    snap = snapshot_with_code(tmp_path)
    T, seen, Stop = _auto_map_fakes({"AutoModelForVideoGeneration": "m.X", "AutoModel": "m.Base"})
    with pytest.raises(Stop):
        rg.load_model(TORCH, T, snap)
    assert seen["which"] == ["AutoModel"]
def test_a_hub_cache_snapshot_file_that_links_into_its_own_blobs_directory_is_hashed(tmp_path):
    """Found on the Spark run 2 (2026-11-08): every file in a Hugging Face cache snapshot is a link to ../../blobs/<hash>, and the first
    version of the hash read refused all links, so the audit listed every file as unreadable. A link whose target stays inside the model's
    cache directory (the parent of snapshots/ and blobs/) is followed; a link that leaves it is still refused."""
    import hashlib
    model_dir = tmp_path / "models--org--name"
    blobs = model_dir / "blobs"
    snap = model_dir / "snapshots" / ("a" * 40)
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    (blobs / "abc123").write_bytes(b"class Spark: pass\n")
    os.symlink("../../blobs/abc123", snap / "modeling_spark.py")
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("not in the cache")
    os.symlink(outside, snap / "escape.py")
    rg.allow_remote_code(snap, scratch_home=tmp_path / "home")
    files = rg.REMOTE_CODE["files"]
    assert files["modeling_spark.py"] == hashlib.sha256(b"class Spark: pass\n").hexdigest()
    assert files["escape.py"] == "unreadable"


def test_a_directory_that_is_not_a_cache_snapshot_follows_no_links_at_all(tmp_path):
    """Review of PR 608: for a plain checkout the cache root fell back to the snapshot itself, so a .py link to another file inside the
    checkout was followed. A directory that is not <model dir>/snapshots/<sha> follows no link."""
    snap = tmp_path / "checkout"
    snap.mkdir()
    (snap / "real.py").write_text("x = 1\n")
    os.symlink(snap / "real.py", snap / "alias.py")
    rg.allow_remote_code(snap, scratch_home=tmp_path / "home")
    files = rg.REMOTE_CODE["files"]
    assert len(files["real.py"]) == 64 and files["alias.py"] == "unreadable"
