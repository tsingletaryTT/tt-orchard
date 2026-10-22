# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 7's spec on each path, and gate_package checking the staged package from disk (plan 5)."""
import json
import os
import re
import socket
import sys
from pathlib import Path

import pytest

from orchard.defaults import SWAP_TOP1_MIN
from orchard.package import finish, stage_all
from orchard.stages import (PACKAGE_STAGE_7, STAGES, gate_package, package_format,
                            package_options, spec_for, validate_table)
from package_fakes import fake_bin, fake_boot_result, make_run, make_source, write

FAKE_SERVER = Path(__file__).resolve().with_name("fake_swap_server.py")


def test_stage_7_packages_only_on_the_weights_only_path_with_a_format():
    assert spec_for(7, "weights-only", "v6") is PACKAGE_STAGE_7
    for path, fmt in (("weights-only", None), ("full-port", "v6"), (None, "v6"), ("weights-only", "v5.1")):
        assert spec_for(7, path, fmt) is STAGES[7] and STAGES[7].skip
    s = PACKAGE_STAGE_7
    assert (s.harness, s.boards, s.gate_file, s.gate, s.marker, s.skip, s.skill) == (
        True, 1, "package.json", gate_package, "test-result.json", None, "")
    validate_table(tuple(s if t.number == 7 else t for t in STAGES))


def test_a_skill_less_stage_that_is_not_harness_code_is_refused():
    import dataclasses
    bad = dataclasses.replace(PACKAGE_STAGE_7, harness=False)
    with pytest.raises(ValueError, match="stage 7 needs a skill"):
        validate_table(tuple(bad if t.number == 7 else t for t in STAGES))


def test_the_package_options_come_from_run_start_or_a_later_decision():
    start = {"seq": 1, "event": "run_start", "stage": None, "data": {"package": {"format": "v6"}}}
    assert package_format([start]) == "v6"
    assert package_format([{**start, "data": {}}]) is None
    assert package_format([]) is None
    later = {"seq": 2, "event": "decision", "stage": None,
             "data": {"decision": "package options set", "package": {"format": "v6", "namespace": "n"}}}
    assert package_options([{**start, "data": {"package": None}}, later]) == {"format": "v6",
                                                                             "namespace": "n"}


@pytest.fixture
def stage7(tmp_path, monkeypatch, stub_tools):
    """A stage 7 directory whose package was staged, boot-checked (faked) and finished."""
    models = tmp_path / "models"
    r = make_run(tmp_path, make_source(models))
    fb, _ = fake_bin(tmp_path)
    monkeypatch.setenv("PATH", f"{fb}:{os.environ['PATH']}")
    env = dict(os.environ, FAKE_TT_MODEL_PYTHON=sys.executable, FAKE_SWAP_SERVER=str(FAKE_SERVER),
               FAKE_SWAP_CONFIG=str(tmp_path / "server.json"))
    stage = r["run"] / "stages/7"
    stage.mkdir(parents=True)
    stage_all(r["run"], stage, namespace="episod", models_root=models, env=env)
    fake_boot_result(stage)
    finish(r["run"], stage)
    return {**r, "stage": stage, "pkg": stage / "package/hemmingway-1-p300"}


def reasons(s) -> list[str]:
    return list(gate_package(s["stage"], s["run"]).reasons)


def test_a_staged_and_boot_checked_package_passes(stage7):
    g = gate_package(stage7["stage"], stage7["run"])
    assert g.ok, g.reasons
    assert "stages/7/verify/evidence/verify.json" in g.evidence


def test_a_manifest_at_another_revision_than_stage_0s_fails(stage7):
    # Stage 0 names <repo>@<revision>; the manifest keeps the two apart and both must match.
    m = stage7["pkg"] / "tt_kernel_manifest.json"
    data = json.loads(m.read_text())
    data["weights"]["revision"] = "0" * 40
    write(m, data)
    assert any("the manifest's weights are Altworld/Hemmingway-1@" + "0" * 40 in r
               for r in reasons(stage7))


def test_a_run_sh_edited_back_to_the_base_weights_fails(stage7):
    run_sh = stage7["pkg"] / "run.sh"
    run_sh.write_text(run_sh.read_text().replace('export MODEL_WEIGHTS_DIR="$HERE/model-dir"\n', ""))
    assert any("MODEL_WEIGHTS_DIR" in r for r in reasons(stage7))


def test_a_hostname_or_a_cache_in_the_staged_package_fails(stage7):
    (stage7["pkg"] / "NOTES.txt").write_text(f"staged on {socket.gethostname()}\n")
    (stage7["pkg"] / ".tt_cache").mkdir()
    rs = reasons(stage7)
    assert any("scrub: NOTES.txt: the hostname" in r for r in rs), rs
    assert any("scrub: .tt_cache: a tensor cache directory" in r for r in rs), rs


def test_a_card_whose_license_was_changed_fails(stage7):
    card = stage7["pkg"] / "README.md"
    card.write_text(card.read_text().replace("license: cc-by-nc-4.0", "license: apache-2.0"))
    assert any("card: the card's license is apache-2.0" in r for r in reasons(stage7))


def test_a_model_whose_license_cannot_be_read_fails(stage7):
    (stage7["snapshot"] / "README.md").unlink()
    assert any("license cannot be read" in r for r in reasons(stage7))


def test_a_boot_check_below_the_bar_fails(stage7):
    v = stage7["stage"] / "verify/evidence/verify.json"
    write(v, {**json.loads(v.read_text()), "top1_agreement": 0.78})
    assert any(f"below {SWAP_TOP1_MIN}" in r for r in reasons(stage7))


def test_a_boot_check_against_another_server_fails(stage7):
    v = stage7["stage"] / "verify/evidence/verify.json"
    data = json.loads(v.read_text())
    data["server_weights_env"]["MODEL_WEIGHTS_DIR"] = "/elsewhere/Qwen3.8-27B"
    write(v, data)
    assert any("did not serve its own model-dir" in r for r in reasons(stage7))


def test_a_package_with_no_boot_checked_profile_fails(stage7):
    p = stage7["stage"] / "package.json"
    data = json.loads(p.read_text())
    for prof in data["profiles"]:
        prof["verified"] = False
    write(p, data)
    assert any("no required profile passed its boot check" in r for r in reasons(stage7))


def test_a_public_publish_command_fails(stage7):
    f = stage7["stage"] / "PUBLISH_COMMANDS.txt"
    f.write_text(f.read_text().replace(" .\n", " . --public\n"))
    assert any("--public" in r for r in reasons(stage7))


def test_a_missing_package_json_fails(tmp_path):
    g = gate_package(tmp_path, tmp_path)
    assert not g.ok and g.reasons == ("package.json is missing",)


@pytest.fixture
def stage7_borrowed(tmp_path, monkeypatch, stub_tools):
    """Like stage7, for a model that ships no generation_config.json: stage 2's staging copied the nearest
    model's as a real file, so the package carries it in base_config/."""
    models = tmp_path / "models"
    r = make_run(tmp_path, make_source(models))
    write(r["run"] / "stages/2/model-dir/generation_config.json", '{"eos_token_id": [248046, 248044]}')
    fb, _ = fake_bin(tmp_path)
    monkeypatch.setenv("PATH", f"{fb}:{os.environ['PATH']}")
    env = dict(os.environ, FAKE_TT_MODEL_PYTHON=sys.executable, FAKE_SWAP_SERVER=str(FAKE_SERVER),
               FAKE_SWAP_CONFIG=str(tmp_path / "server.json"))
    stage = r["run"] / "stages/7"
    stage.mkdir(parents=True)
    stage_all(r["run"], stage, namespace="episod", models_root=models, env=env)
    fake_boot_result(stage)
    finish(r["run"], stage)
    return {**r, "stage": stage, "pkg": stage / "package/hemmingway-1-p300"}


def test_the_card_of_a_package_that_borrows_the_generation_config_says_so_and_passes(stage7_borrowed):
    card = (stage7_borrowed["pkg"] / "README.md").read_text()
    assert "ships no generation_config.json" in card
    assert "`generation_config.json`" in card and "shipped in `base_config/`" in card
    g = gate_package(stage7_borrowed["stage"], stage7_borrowed["run"])
    assert g.ok, g.reasons


def test_the_card_of_a_package_with_the_models_own_generation_config_does_not_say_so(stage7):
    card = (stage7["pkg"] / "README.md").read_text()
    assert "ships no generation_config.json" not in card and "`generation_config.json`" not in card


def test_a_card_that_dropped_the_borrowed_sampling_statement_fails_the_gate(stage7_borrowed):
    card = stage7_borrowed["pkg"] / "README.md"
    kept = [ln for ln in card.read_text().splitlines() if "ships no generation_config.json" not in ln]
    card.write_text("\n".join(kept) + "\n")
    assert any("card:" in r and "generation_config.json" in r and "sampling defaults" in r
               for r in reasons(stage7_borrowed))


def _attribution_fails(world, edit):
    card = world["pkg"] / "README.md"
    card.write_text(edit(card.read_text()))
    return [r for r in reasons(world) if "card:" in r and "generation_config.json" in r]


def test_a_card_that_keeps_the_absence_sentence_but_drops_whose_defaults_are_used_fails_the_gate(stage7_borrowed):
    """The bullet says the model ships no generation config; the gate must also see the sentence that says the stop
    tokens and sampling defaults are the nearest model's."""
    def edit(text):
        return "\n".join(ln.split(", so the stop tokens")[0] + "." if "ships no generation_config.json" in ln else ln
                         for ln in text.splitlines()) + "\n"
    assert _attribution_fails(stage7_borrowed, edit)


def test_a_card_that_names_another_model_as_the_source_of_the_defaults_fails_the_gate(stage7_borrowed):
    card = (stage7_borrowed["pkg"] / "README.md").read_text()
    m = re.search(r"are those of (\S+)\.", card)
    assert m, card
    assert _attribution_fails(stage7_borrowed, lambda t: t.replace(f"are those of {m.group(1)}.", "are those of other/model."))


def test_a_card_that_drops_the_sampling_limitation_fails_the_gate(stage7_borrowed):
    def edit(text):
        return "\n".join(ln for ln in text.splitlines() if not ln.startswith("- Default sampling settings")) + "\n"
    assert _attribution_fails(stage7_borrowed, edit)


def test_a_card_that_does_not_name_a_base_config_file_fails_the_gate(stage7):
    card = stage7["pkg"] / "README.md"
    card.write_text(card.read_text().replace("`preprocessor_config.json`, ", ""))
    assert any("card must name `preprocessor_config.json`" in r for r in reasons(stage7))
