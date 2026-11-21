# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Reference excerpts for a model that ships its own modeling file (issue 781).

`microsoft/Aion-1.0-Instruct-2.3B` has no module under the installed transformers, and its config says
`auto_map: {"AutoModelForCausalLM": "modeling_aion_instruct.AionInstructForCausalLM"}`. The excerpts step used to look only in the
installed transformers, wrote nothing, and the ledger said "the reference modeling module was not found or could not be read".

The fix reads the file the config names from the snapshot. The file is a third party's code, so it is parsed with `ast` and never
imported or run, the name is checked before any path is made from it, and a link that leads outside the snapshot's cache is refused.
Fakes only: a tiny snapshot built in a temporary directory."""
import json
import os
from types import SimpleNamespace

import pytest

from orchard import port_excerpts, supervisor

MODELING = '''
import os
from pathlib import Path
Path(os.environ.get("ORCHARD_TEST_MARKER", "/nonexistent-marker-dir/x")).write_text("the module was run")


class TinyRemoteRMSNorm:
    def __init__(self, hidden_size, eps=1e-6):
        self.eps = eps

    def forward(self, x):
        return x


class TinyRemoteAttention:
    def __init__(self, config, layer_idx):
        self.layer_idx = layer_idx

    def forward(self, hidden_states):
        return hidden_states


class TinyRemoteMLP:
    def __init__(self, config):
        pass

    def forward(self, x):
        return x


class TinyRemoteDecoderLayer:
    def __init__(self, config, layer_idx):
        pass

    def forward(self, hidden_states):
        return hidden_states


class TinyRemoteModel:
    def __init__(self, config):
        pass

    def forward(self, input_ids):
        return input_ids


class TinyRemoteForCausalLM:
    def __init__(self, config):
        pass

    def forward(self, input_ids):
        return input_ids
'''

CONFIG = {"architectures": ["TinyRemoteForCausalLM"], "model_type": "tiny_remote", "layer_types": ["full_attention"],
          "auto_map": {"AutoConfig": "configuration_tiny_remote.TinyRemoteConfig",
                       "AutoModelForCausalLM": "modeling_tiny_remote.TinyRemoteForCausalLM"}}


def snapshot(tmp_path, config=None, modeling=MODELING, name="modeling_tiny_remote.py"):
    snap = tmp_path / "snap"
    snap.mkdir(parents=True, exist_ok=True)
    (snap / "config.json").write_text(json.dumps(config or CONFIG))
    if modeling is not None:
        (snap / name).write_text(modeling)
    return snap


def transformers_root(tmp_path):
    """An installed-transformers directory that has no module for `tiny_remote`."""
    root = tmp_path / "transformers"
    (root / "models").mkdir(parents=True, exist_ok=True)
    return root


def test_a_model_with_its_own_modeling_file_gets_excerpts_from_that_file(tmp_path):
    snap = snapshot(tmp_path)
    got = port_excerpts.for_root(transformers_root(tmp_path), CONFIG, snapshot=snap)
    assert got.text, got.reason
    assert "TinyRemoteAttention" in got.classes and "TinyRemoteDecoderLayer" in got.classes
    assert "class TinyRemoteAttention" in got.text
    assert got.source == "modeling_tiny_remote.py"
    assert "modeling_tiny_remote.py" in got.text.split("\n## ", 1)[0]          # the header says where the code came from


def test_without_a_snapshot_the_old_empty_result_and_reason_stay(tmp_path):
    got = port_excerpts.for_root(transformers_root(tmp_path), CONFIG)
    assert got.text == "" and "not found or could not be read" in got.reason


def test_the_installed_transformers_module_is_used_first_when_it_has_one(tmp_path):
    root = transformers_root(tmp_path)
    (root / "models" / "tiny_remote").mkdir()
    (root / "models" / "tiny_remote" / "modeling_tiny_remote.py").write_text(MODELING.replace("return hidden_states", "return 'installed'"))
    snap = snapshot(tmp_path)
    got = port_excerpts.for_root(root, CONFIG, snapshot=snap)
    assert "return 'installed'" in got.text and got.source == "transformers"


def test_the_modeling_file_is_parsed_and_never_run(tmp_path, monkeypatch):
    marker = tmp_path / "marker"
    monkeypatch.setenv("ORCHARD_TEST_MARKER", str(marker))
    snap = snapshot(tmp_path)
    got = port_excerpts.for_root(transformers_root(tmp_path), CONFIG, snapshot=snap)
    assert got.text
    assert not marker.exists()


@pytest.mark.parametrize("target", [
    "other-org/other-repo--modeling_tiny_remote.TinyRemoteForCausalLM",     # code that lives in another repository
    "../modeling_tiny_remote.TinyRemoteForCausalLM",
    "sub/modeling_tiny_remote.TinyRemoteForCausalLM",
    "modeling tiny.TinyRemoteForCausalLM",
    "modeling_tiny_remote",                                                   # no class part
    "x" * 300 + ".TinyRemoteForCausalLM",
    ".TinyRemoteForCausalLM",
    42,
])
def test_an_auto_map_entry_that_is_not_a_plain_module_in_the_snapshot_is_refused_with_a_fixed_reason(tmp_path, target):
    config = dict(CONFIG, auto_map={"AutoModelForCausalLM": target})
    snap = snapshot(tmp_path, config=config)
    got = port_excerpts.for_root(transformers_root(tmp_path), config, snapshot=snap)
    assert got.text == ""
    assert "auto_map" in got.reason
    assert str(target)[:30] not in got.reason                  # the reason is a fixed sentence and never quotes the config


def test_a_modeling_file_that_is_a_link_leading_outside_the_snapshot_is_refused(tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text(MODELING)
    snap = snapshot(tmp_path, modeling=None)
    os.symlink(outside, snap / "modeling_tiny_remote.py")
    got = port_excerpts.for_root(transformers_root(tmp_path), CONFIG, snapshot=snap)
    assert got.text == ""
    assert "auto_map" in got.reason


def test_a_snapshot_link_into_the_caches_blobs_directory_is_read(tmp_path):
    model_dir = tmp_path / "models--x--y"
    blobs, snap = model_dir / "blobs", model_dir / "snapshots" / "abc"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    (blobs / "h1").write_text(MODELING)
    os.symlink(blobs / "h1", snap / "modeling_tiny_remote.py")
    (snap / "config.json").write_text(json.dumps(CONFIG))
    got = port_excerpts.for_root(transformers_root(tmp_path), CONFIG, snapshot=snap)
    assert got.text, got.reason


def test_a_missing_or_unparsable_own_file_gives_a_named_reason_and_no_file(tmp_path):
    snap = snapshot(tmp_path, modeling=None)
    assert "auto_map" in port_excerpts.for_root(transformers_root(tmp_path), CONFIG, snapshot=snap).reason
    snap2 = snapshot(tmp_path / "b", modeling="def broken(:\n")
    got = port_excerpts.for_root(transformers_root(tmp_path), CONFIG, snapshot=snap2)
    assert got.text == "" and "could not be parsed" in got.reason


# ---- the supervisor passes the snapshot -----------------------------------------------------------------------------------------------

class Ledger:
    def __init__(self):
        self.entries = []

    def append(self, event, stage, **kw):
        self.entries.append((event, kw))


def test_the_supervisor_writes_the_file_for_a_model_with_its_own_code_and_says_where_it_came_from(tmp_path):
    snap = snapshot(tmp_path)
    stage = tmp_path / "stage"
    stage.mkdir()
    ledger = Ledger()
    written, summary = supervisor.write_reference_excerpts(ledger, tmp_path, SimpleNamespace(number=2), stage, snap, transformers_root(tmp_path))
    assert written is True and (stage / port_excerpts.FILE_NAME).is_file()
    decision = next(k for e, k in ledger.entries if k.get("decision") == "reference excerpts written")
    assert decision["source"] == "modeling_tiny_remote.py"
    assert summary["source"] == "modeling_tiny_remote.py"


def test_the_supervisors_not_written_decision_carries_a_fixed_reason_for_a_bad_auto_map(tmp_path):
    config = dict(CONFIG, auto_map={"AutoModelForCausalLM": "a/b--modeling_tiny_remote.TinyRemoteForCausalLM"})
    snap = snapshot(tmp_path, config=config)
    stage = tmp_path / "stage"
    stage.mkdir()
    ledger = Ledger()
    written, _ = supervisor.write_reference_excerpts(ledger, tmp_path, SimpleNamespace(number=2), stage, snap, transformers_root(tmp_path))
    assert written is False
    problem = next(k for e, k in ledger.entries if k.get("decision") == "reference excerpts not written")["problem"]
    assert "auto_map" in problem and "a/b--" not in problem
