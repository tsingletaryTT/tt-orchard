"""Stage 7, part 3: stage every profile, write the cards, the record and the publish commands (plan 5)."""
import json
import os
import sys
from pathlib import Path

import pytest

from orchard.handoff import NOTE_KEYS
from orchard.package import (PackageError, finish, publish_commands, publish_problems, read_run,
                             stage_all)
from orchard.package_card import card_problems
from orchard.scrub import scrub_package
from package_fakes import (SOURCE_ENV, calls, fake_bin, fake_boot_result, make_run, make_source,
                           write)

HOST = "quietbox-test"
FAKE_SERVER = Path(__file__).resolve().with_name("fake_swap_server.py")
pytestmark = pytest.mark.usefixtures("stub_tools")


@pytest.fixture
def world(tmp_path, monkeypatch, stub_tools):
    models = tmp_path / "models"
    src2 = make_source(models)
    make_source(models, name="qwen3.8-27b-dflash2-p150", chips=1, mesh="P150")
    r = make_run(tmp_path, src2)
    fb, log = fake_bin(tmp_path)
    monkeypatch.setenv("PATH", f"{fb}:{os.environ['PATH']}")
    env = dict(os.environ, FAKE_TT_MODEL_PYTHON=sys.executable, FAKE_SWAP_SERVER=str(FAKE_SERVER),
               FAKE_SWAP_CONFIG=str(tmp_path / "server.json"))
    return {**r, "models": models, "src2": src2, "log": log, "env": env, "stubs": stub_tools,
            "stage": r["run"] / "stages/7"}


def staged(world):
    world["stage"].mkdir(parents=True, exist_ok=True)
    return stage_all(world["run"], world["stage"], namespace="episod", models_root=world["models"],
                     env=world["env"], hostname=HOST)


def test_publish_commands_upload_privately_and_only_boot_checked_packages_get_a_live_line():
    profiles = [{"chips": 2, "name": "hemmingway-1-p300", "mesh": "P150x2", "verified": True},
                {"chips": 1, "name": "hemmingway-1-p150", "mesh": "P150", "verified": False}]
    text = publish_commands(profiles, namespace="episod", license_id="cc-by-nc-4.0")
    live = [ln for ln in text.splitlines() if ln and not ln.startswith("#")]
    assert live == ["hf upload --repo-type model --private episod/hemmingway-1-p300 "
                    "stages/7/package/hemmingway-1-p300 ."]
    assert "# hf upload --repo-type model --private episod/hemmingway-1-p150" in text
    assert "NOT boot-checked" in text and "cc-by-nc-4.0 (non-commercial)" in text
    assert publish_problems(text) == []


@pytest.mark.parametrize("line, words", [
    ("hf upload --repo-type model --private episod/x stages/7/package/x . --public",
     "PUBLISH_COMMANDS.txt uses --public"),
    ("tt-model push stages/7/package/x", "not a private hf upload"),
    ("tt-model package-thin episod/x --publish", "PUBLISH_COMMANDS.txt uses --publish"),
    ("hf upload --repo-type model episod/x stages/7/package/x .", "not a private hf upload"),
])
def test_a_publish_line_that_is_not_a_private_upload_is_refused(line, words):
    assert any(words in p for p in publish_problems(f"# header\n{line}\n"))


def test_a_publish_file_with_no_live_line_is_refused():
    assert any("no command" in p for p in publish_problems("# hf upload ...\n"))


def test_stage_all_stages_every_profile_and_lays_out_the_boot_check(world):
    pkg = staged(world)
    assert [(p["chips"], p["name"], p["required"], p["verified"]) for p in pkg["profiles"]] == [
        (1, "hemmingway-1-p150", False, False), (2, "hemmingway-1-p300", True, False)]
    assert pkg["skipped_profiles"] == [
        {"chips": 4, "reason": "no v6 bundle of Qwen/Qwen3.8-27B for 4 chips is installed"}]
    assert (pkg["license"], pkg["non_commercial"], pkg["format"]) == ("cc-by-nc-4.0", True, "v6")
    assert json.loads((world["stage"] / "package.json").read_text()) == pkg
    for p in pkg["profiles"]:
        out = world["run"] / p["dir"]
        card = (out / "README.md").read_text()
        assert card_problems(card, license_id="cc-by-nc-4.0", run_dir=world["run"]) == []
        assert scrub_package(out, hostname=HOST, namespace="episod") == []
    test = json.loads((world["stage"] / "hw_test.json").read_text())
    assert test["command"].endswith("stages/7/verify/verify_bundle.py")
    note = json.loads((world["stage"] / "handoff.json").read_text())
    assert all(note.get(k) not in (None, "") for k in NOTE_KEYS)
    assert (world["stage"] / "verify" / "bundle" / "venv" / "bin" / "python").is_file()
    # Two package-thin calls with --out, nothing else; no stub ran.
    assert [c[0] for c in calls(world["log"])] == ["package-thin", "package-thin"]
    assert all("--out" in c for c in calls(world["log"])) and world["stubs"].calls() == []
    # Every absolute path in package.json lives outside it: the operator bundle copies it.
    assert "/home/" not in (world["stage"] / "package.json").read_text()


def test_a_scrub_hit_stops_the_stage_before_anything_is_installed(world):
    m = json.loads((world["src2"] / "tt_kernel_manifest.json").read_text())
    m["env"] = {**SOURCE_ENV, "BUILD_HOST": HOST}
    write(world["src2"] / "tt_kernel_manifest.json", m)
    with pytest.raises(PackageError, match=f"scrub of hemmingway-1-p300: .*{HOST}"):
        staged(world)
    assert not (world["stage"] / "verify").exists()
    assert not (world["stage"] / "hw_test.json").exists()


def test_finish_records_a_passing_boot_check_on_the_required_profile_only(world):
    staged(world)
    fake_boot_result(world["stage"])
    pkg = finish(world["run"], world["stage"], hostname=HOST)
    req = next(p for p in pkg["profiles"] if p["required"])
    opt = next(p for p in pkg["profiles"] if not p["required"])
    assert req["verified"] is True and opt["verified"] is False
    assert req["verify"]["top1_agreement"] == 0.94 and req["scrub"] == [] and opt["scrub"] == []
    card = (world["run"] / req["dir"] / "README.md").read_text()
    assert ("| top1 agreement with the CPU reference, this package (stage 7) | 0.94 fraction | measured |"
            in card)
    assert card_problems(card, license_id="cc-by-nc-4.0", run_dir=world["run"]) == []
    opt_card = (world["run"] / opt["dir"] / "README.md").read_text()
    assert "stage 2" not in opt_card.split("## Expected performance")[1].split("## ")[0]       # no borrowed numbers
    text = (world["stage"] / "PUBLISH_COMMANDS.txt").read_text()
    assert publish_problems(text) == []
    assert "private episod/hemmingway-1-p300 stages/7/package/hemmingway-1-p300 .\n" in text


def test_finish_after_a_failed_boot_check_publishes_nothing(world):
    staged(world)
    fake_boot_result(world["stage"], returncode=4)
    pkg = finish(world["run"], world["stage"], hostname=HOST)
    req = next(p for p in pkg["profiles"] if p["required"])
    assert req["verified"] is False and "exited 4" in req["verify"]["failed"]
    assert any("no command" in p for p in
               publish_problems((world["stage"] / "PUBLISH_COMMANDS.txt").read_text()))


def test_read_run_still_works_after_staging(world):
    staged(world)
    assert read_run(world["run"]).model_id == "Altworld/Hemmingway-1"
