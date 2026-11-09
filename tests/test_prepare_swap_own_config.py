# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 2 serves the new model's own config.json for a bundle that ships orchard-capabilities.json (Track C, item C2).

Such a bundle's code reads the model's configuration at run time (docs/bundle-capabilities.md), and stage 0 has checked every
differing key against the bundle's declarations. Serving the nearest model's config would then hide the very differences stage 0
allowed (a 64-layer release served with a 20-layer config). Only `architectures` is replaced by the served base config's, because the
bundle's TT model class is registered under that name. Without the file the swap serves the nearest model's config as before."""
import json
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_prepare_swap_current_form import cur  # noqa: E402,F401  (the current-form fixture)
from test_weights_swap_templates import NEAREST, prep, put, run_prepare  # noqa: E402,F401

CAPS = json.dumps({"schema": 1, "config_keys": {"num_hidden_layers": {"type": "int"}}})
NEW_CONFIG = {"architectures": ["Qwen4ForCausalLM"], "model_type": "qwen4_text", "num_hidden_layers": 64, "rope_parameters": {"rope_theta": 5e6},
              "eos_token_id": 9}


def served(stage: Path) -> dict:
    return json.loads((stage / "model-dir" / "config.json").read_text())


def sources(stage: Path) -> dict:
    return json.loads((stage / "model-dir-sources.json").read_text())


@pytest.fixture
def older(prep):
    """The older-form fixture with a base config that names its architecture and a new config that differs."""
    put(prep["base"], "config.json", json.dumps({"architectures": ["Qwen4ExpForCausalLM"], "model_type": "qwen4_exp_text", "num_hidden_layers": 20}))
    put(prep["new"], "config.json", json.dumps(NEW_CONFIG))
    return prep


def test_without_the_file_the_nearest_models_config_is_served(older):
    r = run_prepare(older["stage"])
    assert r.returncode == 0, r.stderr
    assert served(older["stage"])["num_hidden_layers"] == 20
    assert sources(older["stage"])["config_json"]["source"] == "base"


def test_with_the_file_a_planted_difference_reaches_the_served_model_dir(older):
    (older["bundle"] / "orchard-capabilities.json").write_text(CAPS)
    r = run_prepare(older["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    md = older["stage"] / "model-dir"
    assert (md / "config.json").is_file() and not (md / "config.json").is_symlink()       # stage 7 packages real files only
    got = served(older["stage"])
    assert got == {**NEW_CONFIG, "architectures": ["Qwen4ExpForCausalLM"]}               # only architectures changed
    rec = sources(older["stage"])
    assert rec["config_json"] == {"source": "model", "architectures": ["Qwen4ExpForCausalLM"], "model_architectures": ["Qwen4ForCausalLM"],
                                  "architectures_from": "nearest model's snapshot"}
    entry = next(f for f in rec["files"] if f["name"] == "config.json")
    assert entry["source"] == "model" and entry["how"] == "copy"
    # the other configuration files are still the nearest model's
    assert (md / "preprocessor_config.json").read_text() == (older["base"] / "preprocessor_config.json").read_text()
    assert 9 in rec["eos_token_ids"]                                                     # the model's own config is read for stop tokens
    assert "own config.json" in r.stdout


def test_with_the_file_a_current_form_bundle_takes_architectures_from_base_config(cur):
    (cur["bundle"] / "orchard-capabilities.json").write_text(CAPS)
    put(cur["new"], "config.json", json.dumps(NEW_CONFIG))
    r = run_prepare(cur["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    got = served(cur["stage"])
    assert got["architectures"] == ["BundleArch"] and got["num_hidden_layers"] == 64
    assert sources(cur["stage"])["config_json"]["architectures_from"] == "bundle base_config/"
    # base_config/'s other files are still copied
    md = cur["stage"] / "model-dir"
    assert json.loads((md / "preprocessor_config.json").read_text()) == {"bundle": "pre"}


def test_a_new_config_without_architectures_gets_the_bases(older):
    (older["bundle"] / "orchard-capabilities.json").write_text(CAPS)
    put(older["new"], "config.json", json.dumps({"num_hidden_layers": 64}))
    r = run_prepare(older["stage"])
    assert r.returncode == 0, r.stderr
    assert served(older["stage"]) == {"num_hidden_layers": 64, "architectures": ["Qwen4ExpForCausalLM"]}
    assert sources(older["stage"])["config_json"]["model_architectures"] is None


@pytest.mark.parametrize("text", ["{not json", "[]", '"x"', '{"a": NaN}', '{"a": 1, "a": 2}'])
def test_an_unusable_new_config_stops_the_swap(older, text):
    (older["bundle"] / "orchard-capabilities.json").write_text(CAPS)
    put(older["new"], "config.json", text)
    r = run_prepare(older["stage"])
    assert r.returncode == 2
    assert "config.json" in r.stderr
    assert not (older["stage"] / "run.sh").exists()


def test_a_missing_new_config_stops_the_swap(older):
    (older["bundle"] / "orchard-capabilities.json").write_text(CAPS)
    (older["new"] / "config.json").unlink()
    r = run_prepare(older["stage"])
    assert r.returncode == 2 and "config.json" in r.stderr


@pytest.mark.parametrize("arch", [None, [], "Qwen4ExpForCausalLM", [3], [""]])
def test_a_base_config_without_a_usable_architectures_list_stops_the_swap(older, arch):
    (older["bundle"] / "orchard-capabilities.json").write_text(CAPS)
    base = {"model_type": "qwen4_exp_text"}
    if arch is not None:
        base["architectures"] = arch
    put(older["base"], "config.json", json.dumps(base))
    r = run_prepare(older["stage"])
    assert r.returncode == 2 and "architectures" in r.stderr


def test_a_capabilities_file_that_is_a_link_still_counts_as_declared(older, tmp_path):
    """Stage 0 refuses a linked file and calls the run a full port, so stage 2 does not normally run. If it does, the bundle still
    reads its config at run time, so the model's own config is the one to serve."""
    real = tmp_path / "caps.json"
    real.write_text(CAPS)
    (older["bundle"] / "orchard-capabilities.json").symlink_to(real)
    r = run_prepare(older["stage"])
    assert r.returncode == 0, r.stderr
    assert served(older["stage"])["num_hidden_layers"] == 64


def test_the_vocab_trim_rewrite_leaves_the_models_own_config_alone(older):
    (older["bundle"] / "orchard-capabilities.json").write_text(CAPS)
    put(older["base"], "config.json", json.dumps({"architectures": ["Qwen4ExpForCausalLM"], "vocab_size": 200}))
    put(older["new"], "config.json", json.dumps({**NEW_CONFIG, "vocab_size": 100}))
    r = run_prepare(older["stage"])
    assert r.returncode == 0, r.stderr
    assert served(older["stage"])["vocab_size"] == 100


# ---- stage 7: the package serves the config stage 2 served, and its card says whose it is --------------------------------------

def test_stage_7_reads_the_own_config_record(tmp_path):
    from orchard.package import stage_2_own_config
    run = tmp_path / "run"
    (run / "stages" / "2").mkdir(parents=True)
    rec = run / "stages" / "2" / "model-dir-sources.json"
    assert stage_2_own_config(run) is False                                    # no record
    for value, want in (({"source": "model"}, True), ({"source": "base"}, False), ("model", False), (None, False), ([], False)):
        rec.write_text(json.dumps({"config_json": value}))
        assert stage_2_own_config(run) is want, value
    rec.write_text('{"config_json": {"source": "model"}, "x": NaN}')
    assert stage_2_own_config(run) is False                                    # refused by the safe reader


def test_the_card_names_the_models_own_config_and_still_passes_its_check(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_package_card import facts, run_with_evidence
    from orchard.package_card import card_problems, render_card
    base_files = ("config.json", "preprocessor_config.json")
    own = render_card(facts(base_files=base_files, own_config=True, license_id="apache-2.0"))
    plain = render_card(facts(base_files=base_files, license_id="apache-2.0"))
    assert "own `config.json` with `architectures` set to Qwen/Qwen3.8-27B's" in own
    assert "orchard-capabilities.json" in own and "same text architecture" not in own
    assert "same text architecture" in plain and "own `config.json`" not in plain
    run = run_with_evidence(tmp_path)
    for card in (own, plain):
        assert card_problems(card, license_id="apache-2.0", run_dir=run, base_files=base_files) == []


def test_a_new_config_that_is_a_fifo_stops_the_swap_without_blocking(older):
    import os
    (older["bundle"] / "orchard-capabilities.json").write_text(CAPS)
    blob = (older["new"] / "config.json").resolve()
    blob.unlink()
    os.mkfifo(blob)
    r = run_prepare(older["stage"])
    assert r.returncode == 2 and "regular file" in r.stderr


def test_an_oversized_new_config_stops_the_swap(older):
    (older["bundle"] / "orchard-capabilities.json").write_text(CAPS)
    put(older["new"], "config.json", json.dumps({"pad": "x" * (4 * 1024 * 1024)}))
    r = run_prepare(older["stage"])
    assert r.returncode == 2 and "larger" in r.stderr
