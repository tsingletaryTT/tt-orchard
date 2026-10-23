# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 7 follows stage 2's graft decision (orchard/graft_trial.py): a package is grafted when stage 2 chose the graft, with no option, and
the card shows what the trial measured. `--graft-mtp` still forces a graft and `--no-graft-mtp` refuses one the decision made."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_package_graft import SHARD, SHARD_FILE, stage, world  # noqa: E402,F401  (the fixture)
from package_fakes import fake_boot_result, write  # noqa: E402

from orchard.graft_trial import DECISION_FILE, TRIAL_DIR, TRIAL_RESULT, decide, write_decision  # noqa: E402
from orchard.package import PackageError, finish, read_run  # noqa: E402
from orchard.stages import gate_package  # noqa: E402

CANDIDATE = {"base_tensors": 3, "base_revision": "a" * 40, "base_shards": [SHARD_FILE], "dtypes": ["BF16"], "hidden_size": 4,
             "shapes_match": True, "graftable": True}


def trial(*, safe=20.0, variant=60.0, top1=0.96875):
    return {"label": "measured", "steps": [], "safe": {"server_ready_s": 100.0, "decode_tok_s": safe, "ttft_s": 0.3},
            "variant": {"served": True, "coherent": True, "top1_agreement": top1, "n_tokens": 32, "server_ready_s": 240.0,
                        "decode_tok_s": variant, "ttft_s": 0.2, "failure": None}}


def decided(w, **kw):
    """Stage 2's decision files, as the supervisor writes them, for a trial with these numbers (0.94 is the fake run's stage 2 result)."""
    s2 = w["run"] / "stages" / "2"
    t = trial(**kw)
    write(s2 / TRIAL_DIR / TRIAL_RESULT, t)
    d = decide(t, safe_top1=0.94)
    write_decision(s2, d, candidate=CANDIDATE)
    return d["choice"]


def finished_auto(w):
    stage_dir, package = stage(w)
    fake_boot_result(stage_dir)
    finish(w["run"], stage_dir)
    return stage_dir, package, w["run"] / package["profiles"][0]["dir"]


# ---- the decision sets the package -------------------------------------------------------------------------------------------------

def test_a_graft_decision_gives_a_grafted_package_with_no_option(world):
    assert decided(world) == "graft"
    _, package = stage(world)
    pkg = world["run"] / package["profiles"][0]["dir"]
    graft = json.loads((pkg / "graft.json").read_text())
    assert package["mtp_graft"] == graft and graft["file"] == SHARD_FILE and graft["shard"] == SHARD
    assert "sample_on_device_mode" in (pkg / "run.sh").read_text()


def test_a_safe_decision_gives_the_drafter_off_package(world):
    assert decided(world, variant=21.0) == "safe"
    _, package = stage(world)
    pkg = world["run"] / package["profiles"][0]["dir"]
    assert not (pkg / "graft.json").exists() and "mtp_graft" not in package
    assert json.loads((pkg / "tt_kernel_manifest.json").read_text())["env"]["QWEN36_DRAFTER"] == ""


def test_no_decision_gives_the_drafter_off_package_as_before(world):
    _, package = stage(world)
    assert "mtp_graft" not in package


def test_graft_mtp_still_forces_a_graft_over_a_safe_decision(world):
    assert decided(world, variant=21.0) == "safe"
    _, package = stage(world, graft_mtp=True)
    assert "mtp_graft" in package


def test_a_malformed_decision_file_is_not_a_graft_and_the_gate_in_read_run_says_so(world):
    (world["run"] / "stages/2" / DECISION_FILE).write_text("{")
    with pytest.raises(PackageError, match=DECISION_FILE):
        read_run(world["run"])


def test_stage_7_refuses_a_graft_decision_the_trial_does_not_support(world):
    decided(world)
    write(world["run"] / "stages/2" / TRIAL_DIR / TRIAL_RESULT, trial(variant=19.0))
    with pytest.raises(PackageError, match="do not support"):
        read_run(world["run"])


# ---- --no-graft-mtp ---------------------------------------------------------------------------------------------------------------------

def test_no_graft_mtp_refuses_a_package_that_ignores_a_graft_decision(world):
    decided(world)
    with pytest.raises(PackageError) as exc:
        read_run(world["run"], no_graft_mtp=True)
    assert "--no-graft-trial" in str(exc.value) and "stage 4" in str(exc.value)


def test_no_graft_mtp_with_a_safe_decision_or_none_changes_nothing(world):
    assert read_run(world["run"], no_graft_mtp=True).graft is None
    decided(world, variant=21.0)
    assert read_run(world["run"], no_graft_mtp=True).graft is None


def test_graft_mtp_and_no_graft_mtp_together_are_refused(world):
    with pytest.raises(PackageError, match="both"):
        read_run(world["run"], graft_mtp=True, no_graft_mtp=True)


def test_no_graft_mtp_refuses_when_stage_2_itself_served_with_the_graft(world):
    s2 = world["run"] / "stages/2"
    (s2 / "run.sh").write_text((world["src"] / "run.sh").read_text())
    write(s2 / "model-dir-sources.json", {"schema": 1, "mtp_graft": {"from": "x", "file": SHARD}})
    with pytest.raises(PackageError, match="stage 2"):
        read_run(world["run"], no_graft_mtp=True)


# ---- the card ---------------------------------------------------------------------------------------------------------------------------

def rows(card: str) -> dict:
    out = {}
    for line in card.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("| ") and len(cells) == 4 and cells[2] == "measured":
            out[cells[0]] = (cells[1], cells[3])
    return out


GRAFT_ROWS = ("decode speed, safe configuration (graft trial, median of four prompts)",
              "decode speed, grafted drafter (graft trial, median of four prompts)",
              "time to first token, safe configuration (graft trial, median of four prompts)",
              "time to first token, grafted drafter (graft trial, median of four prompts)",
              "top1 agreement with the CPU reference, grafted drafter (graft trial)")


def test_the_card_shows_the_trials_numbers_with_their_evidence(world):
    decided(world)
    _, _, pkg = finished_auto(world)
    r = rows((pkg / "README.md").read_text())
    assert r[GRAFT_ROWS[0]][0] == "20.0 tokens/s" and r[GRAFT_ROWS[1]][0] == "60.0 tokens/s"
    assert r[GRAFT_ROWS[2]][0] == "0.3 s" and r[GRAFT_ROWS[3]][0] == "0.2 s"
    assert r[GRAFT_ROWS[4]][0] == "0.96875 fraction"
    for name in GRAFT_ROWS:
        assert r[name][1] == f"`stages/2/{TRIAL_DIR}/{TRIAL_RESULT}`, `stages/2/{DECISION_FILE}`"
    assert (pkg / "evidence" / "stages" / "2" / TRIAL_DIR / TRIAL_RESULT).is_file()
    assert (pkg / "evidence" / "stages" / "2" / DECISION_FILE).is_file()


def test_the_gate_passes_a_package_built_from_a_graft_decision(world):
    decided(world)
    stage_dir, _, _ = finished_auto(world)
    g = gate_package(stage_dir, world["run"])
    assert g.ok, g.reasons


def test_a_kept_safe_decision_still_shows_what_the_trial_measured(world):
    decided(world, variant=21.0)
    _, _, pkg = finished_auto(world)
    r = rows((pkg / "README.md").read_text())
    assert r[GRAFT_ROWS[1]][0] == "21.0 tokens/s" and r[GRAFT_ROWS[0]][0] == "20.0 tokens/s"
    flat = " ".join((pkg / "README.md").read_text().split())
    assert "graft trial" in flat and "kept the safe configuration" in flat


def test_a_run_with_no_trial_has_no_trial_rows(world):
    _, _, pkg = finished_auto(world)
    r = rows((pkg / "README.md").read_text())
    assert not [n for n in r if "graft trial" in n]


def test_a_skipped_trial_has_no_trial_rows(world):
    s2 = world["run"] / "stages/2"
    write_decision(s2, {"choice": "safe", "reason": "the graft trial was skipped: --no-graft-trial", "numbers": {}, "thresholds": {}},
                   candidate=CANDIDATE, trial=False)
    _, _, pkg = finished_auto(world)
    assert not [n for n in rows((pkg / "README.md").read_text()) if "graft trial" in n]


def test_a_card_for_a_graft_decision_says_how_the_graft_was_chosen_and_which_numbers_came_from_where(world):
    decided(world)
    _, _, pkg = finished_auto(world)
    flat = " ".join((pkg / "README.md").read_text().split())
    assert "chose the graft" in flat and "graft trial" in flat
    limits = flat.split("## Limitations")[1].split("## Risks")[0]
    assert "stage 2" in limits and "without the graft" in limits
    assert "stage 4 numbers were measured without the graft" not in flat       # stage 4's bundle tests used the graft


def test_a_forced_graft_over_no_trial_keeps_the_old_limitation_text(world):
    stage_dir, package = stage(world, graft_mtp=True)
    fake_boot_result(stage_dir)
    finish(world["run"], stage_dir)
    flat = " ".join((world["run"] / package["profiles"][0]["dir"] / "README.md").read_text().split())
    assert "stage 2 and stage 4 numbers were measured without the graft" in flat


def test_the_gate_refuses_a_card_that_lost_the_trial_rows(world):
    decided(world)
    stage_dir, _, pkg = finished_auto(world)
    card = (pkg / "README.md").read_text()
    kept = [ln for ln in card.splitlines() if "graft trial" not in ln or "chose the graft" in ln]
    (pkg / "README.md").write_text("\n".join(kept) + "\n")
    assert not gate_package(stage_dir, world["run"]).ok


def test_the_gate_refuses_a_package_without_graft_json_when_stage_2_chose_the_graft(world):
    """A package that was edited to drop the graft, record and all, no longer matches the decision."""
    decided(world)
    stage_dir, _, pkg = finished_auto(world)
    (pkg / "graft.json").unlink()
    package = json.loads((stage_dir / "package.json").read_text())
    package.pop("mtp_graft", None)
    for p in package["profiles"]:
        p.pop("mtp_graft", None)
    (stage_dir / "package.json").write_text(json.dumps(package))
    reasons = gate_package(stage_dir, world["run"]).reasons
    assert any("graft trial chose the graft" in r and "graft.json" in r for r in reasons), reasons


def test_no_graft_mtp_leaves_a_package_staged_with_the_graft_flag_alone_out_of_the_new_facts(world):
    stage(world, graft_mtp=True)                         # a grafted package is on disk now
    assert read_run(world["run"]).graft is not None
    assert read_run(world["run"], no_graft_mtp=True).graft is None


def test_a_skipped_trial_does_not_put_trial_rows_in_the_card_even_if_a_result_file_is_lying_around(world):
    s2 = world["run"] / "stages/2"
    write(s2 / TRIAL_DIR / TRIAL_RESULT, trial())
    write_decision(s2, {"choice": "safe", "reason": "the graft trial was skipped: --no-graft-trial", "numbers": {}, "thresholds": {}},
                   candidate=CANDIDATE, trial=False)
    _, _, pkg = finished_auto(world)
    card = (pkg / "README.md").read_text()
    assert not [n for n in rows(card) if "graft trial" in n]
    assert "kept the safe configuration" not in " ".join(card.split())            # a skipped trial tried nothing


def test_a_trial_result_with_no_decision_file_shows_no_rows(world):
    write(world["run"] / "stages/2" / TRIAL_DIR / TRIAL_RESULT, trial())
    _, _, pkg = finished_auto(world)
    assert not [n for n in rows((pkg / "README.md").read_text()) if "graft trial" in n]


def test_a_trial_value_that_is_not_a_positive_number_is_left_out_of_the_card(world):
    s2 = world["run"] / "stages/2"
    t = trial()
    t["safe"]["decode_tok_s"] = True
    t["variant"]["top1_agreement"] = 0
    write(s2 / TRIAL_DIR / TRIAL_RESULT, t)
    d = decide(trial(), safe_top1=0.94)                   # the decision file is the supervisor's; the result file is the one edited
    write_decision(s2, {**d, "choice": "safe"}, candidate=CANDIDATE)
    _, _, pkg = finished_auto(world)
    names = set(rows((pkg / "README.md").read_text()))
    assert GRAFT_ROWS[0] not in names and GRAFT_ROWS[4] not in names
    assert GRAFT_ROWS[1] in names


@pytest.mark.parametrize("variant, said", [(60.0, "chose the graft"), (21.0, "kept the safe configuration")])
def test_the_gate_refuses_a_card_that_leaves_out_what_stage_2_decided(world, variant, said):
    decided(world, variant=variant)
    stage_dir, _, pkg = finished_auto(world)
    assert gate_package(stage_dir, world["run"]).ok
    card = (pkg / "README.md").read_text()
    lines = card.splitlines()
    start = next(i for i, ln in enumerate(lines) if "graft trial" in ln and ln.startswith("- Stage 2"))
    end = start + 1
    while end < len(lines) and lines[end].startswith("  "):
        end += 1
    del lines[start:end]
    (pkg / "README.md").write_text("\n".join(lines) + "\n")
    reasons = gate_package(stage_dir, world["run"]).reasons
    assert any(said in r for r in reasons), reasons


# ---- an optional profile has no trial rows (review finding on PR 47) ---------------------------------------------------------------
# card_numbers() gives an optional profile one TODO row, because the run's measurements were taken on the required profile's chip count.
# The gate must not ask that card for the trial's rows either: they would be another chip count's numbers.

def with_optional_profile(w):
    from package_fakes import make_source
    make_source(w["models"], name="qwen3.8-27b-dflash2-p150", chips=1, mesh="P150",
                env={"ARCH_NAME": "blackhole", "QWEN36_DRAFTER": "dflash2", "QWEN36_DFLASH_TP": "1"})


@pytest.mark.parametrize("variant", [60.0, 21.0])
def test_an_optional_profiles_card_is_not_asked_for_the_trials_rows(world, variant):
    decided(world, variant=variant)                        # graft, then kept safe
    with_optional_profile(world)
    stage_dir, package = stage(world)
    fake_boot_result(stage_dir)
    finish(world["run"], stage_dir)
    required = [p for p in package["profiles"] if p["required"]]
    optional = [p for p in package["profiles"] if not p["required"]]
    assert len(required) == 1 and len(optional) == 1
    opt_card = (world["run"] / optional[0]["dir"] / "README.md").read_text()
    assert "graft trial" not in opt_card or "TODO" in opt_card         # the optional card shows no trial measurement
    g = gate_package(stage_dir, world["run"])
    assert g.ok, g.reasons


def test_the_required_profiles_card_still_has_to_show_the_trials_rows_next_to_an_optional_one(world):
    decided(world)
    with_optional_profile(world)
    stage_dir, package = stage(world)
    fake_boot_result(stage_dir)
    finish(world["run"], stage_dir)
    req = next(p for p in package["profiles"] if p["required"])
    card_path = world["run"] / req["dir"] / "README.md"
    kept = [ln for ln in card_path.read_text().splitlines() if "graft trial" not in ln or "chose the graft" in ln]
    card_path.write_text("\n".join(kept) + "\n")
    g = gate_package(stage_dir, world["run"])
    assert not g.ok and any("graft trial measured it" in r for r in g.reasons)


def test_a_required_profile_that_was_not_boot_checked_still_has_to_show_the_trials_rows(world):
    """The rule follows `required`, as card_numbers does, and not whether the boot check ran."""
    decided(world)
    stage_dir, package = stage(world)                           # staged, never boot-checked
    req = next(p for p in package["profiles"] if p["required"])
    card_path = world["run"] / req["dir"] / "README.md"
    kept = [ln for ln in card_path.read_text().splitlines() if "graft trial" not in ln or "chose the graft" in ln]
    card_path.write_text("\n".join(kept) + "\n")
    g = gate_package(stage_dir, world["run"])
    assert any("graft trial measured it" in r for r in g.reasons), g.reasons
