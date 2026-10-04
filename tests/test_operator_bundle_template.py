"""The stage 8 template the operator-bundle skill copies: build_bundle.py.

Every test builds a fake run directory in tmp: a ledger written by the real orchard.ledger.Ledger
(so it has the seq, ts and prev hash-chain fields), stage 0 and 1 files with evidence, a stage 2
weights-swap result, a stage 4 result with three chip configurations, and stage 7's package.json,
publish commands, boot-check verify.json and one staged package folder per profile. The package
cards are rendered by the real orchard.package_card.render_card, so the "Not measured" list has the
form stage 7 writes. Tensor caches are tmp directories with or without the `.orchard-model` marker.

The script is copied into stages/8/ and run as a subprocess from the run directory, the way the
agent runs it. Its bundle is then checked with the real stage 8 gate (after the supervisor's own
copy step) and the real bundle scrub. Nothing here touches the network or a device.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard.ledger import Ledger
from orchard.package import publish_commands
from orchard.package_card import CardFacts, Number, render_card
from orchard.scrub import scrub_bundle
from orchard.stages import (SKIP_3_WEIGHTS_ONLY, SKIP_5_WEIGHTS_ONLY, SKIP_6_WEIGHTS_ONLY, SKIP_7,
                            gate_bundle)
from orchard.supervisor import Supervisor

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "orchard" / "skills" / "operator-bundle-templates" / "build_bundle.py"
REV = "1a5f363a3dd2d1cc456c28b8abbb403b9555efaf"
BASE_REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MODEL = f"Altworld/Hemmingway-1@{REV}"
NEAREST = f"Qwen/Qwen3.8-27B@{BASE_REV}"
TOKEN = "hf_" + "A1b2C3d4E5" * 4          # matches the scrub's Hugging Face token pattern


def template_module():
    spec = importlib.util.spec_from_file_location("build_bundle_under_test", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- the fake run ---------------------------------------------------------------------------------

def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_json(path: Path, data) -> Path:
    return write(path, json.dumps(data, indent=2) + "\n")


def make_cache(path: Path, *, marker: str | None = MODEL, empty: bool = False) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if not empty:
        if marker is not None:
            (path / ".orchard-model").write_text(marker, encoding="utf-8")
        (path / "layers.0.weight.tensorbin").write_bytes(b"\0" * 16)
    return path


def card(name: str, chips: int, mesh: str, *, verified: bool) -> str:
    numbers = ((Number("top1 agreement with the CPU reference, this package (stage 7)", 0.9375,
                       "fraction", "measured", ("stages/7/verify/evidence/verify.json",)),)
               if verified else
               (Number("top1 agreement with the CPU reference, this package", None, "fraction",
                       "TODO", ()),))
    return render_card(CardFacts(
        name=name, namespace="episod", model_id="Altworld/Hemmingway-1", revision=REV,
        nearest_model="Qwen/Qwen3.8-27B", source_name=f"qwen-{name}", license_id="cc-by-nc-4.0",
        chips=chips, mesh=mesh, arch="p150", max_model_len=16384, max_num_seqs=2,
        drafter="incoai/Qwen3.8-27B-DFlash2", verified=verified, numbers=numbers,
        not_measured=("the drafter's acceptance rate on this model",
                      "a download of the weights through `tt-model pull`, and a boot of the "
                      "package from the Hub")))


def make_run(tmp_path: Path, *, shared_cache=False, marker=MODEL, stage7=True, publish=True,
             nearest_cache=False) -> Path:
    """A run directory that has finished stages 0 to 7 on the weights-only path."""
    run = tmp_path / "run"
    op = tmp_path / "op"
    cache_root = tmp_path / "cache"
    s = run / "stages"
    # stage 0
    for name in ("tensor-compare.json", "config-compare.json", "genconfig-license.json",
                 "disk-free.json", "tokenizer-compare.json"):
        write_json(s / "0" / "evidence" / name, {"fake": name})
    write_json(s / "0" / "delta.json", {
        "model": MODEL, "nearest_model": NEAREST, "path": "weights-only", "path_reasons": [],
        "differences": [
            {"area": "tokenizer", "finding": "6 of 206 strings encode differently (Thai, Devanagari).",
             "evidence": ["stages/0/evidence/tokenizer-compare.json"]},
            {"area": "license", "finding": "cc-by-nc-4.0 against apache-2.0. Read at "
                                           "/home/someone/.cache/huggingface/x/README.md.",
             "evidence": ["stages/0/evidence/genconfig-license.json"]}],
        "hazards": [
            {"area": "tensor_cache", "finding": "The nearest model's cache at "
                                                f"{op}/.cache/src-build/tensors is keyed by layer name.",
             "evidence": ["stages/0/evidence/tensor-compare.json"]},
            {"area": "drafter", "finding": "The drafter was trained on the nearest model.",
             "evidence": ["stages/0/evidence/config-compare.json"]},
            {"area": "disk", "finding": "120 GiB free.", "evidence": ["stages/0/evidence/disk-free.json"]},
            {"area": "license", "finding": "The license changed to cc-by-nc-4.0.",
             "evidence": ["stages/0/evidence/genconfig-license.json"]}]})
    # stage 1
    for name in ("load-report.json", "decode.txt", "card-check.txt"):
        write(s / "1" / "evidence" / name, "fake\n")
    write_json(s / "1" / "reference.json", {
        "verdict": "pass", "environment": {"python": "3.12", "torch": "2.11", "transformers": "5.17"},
        "checks": [
            {"name": "loads", "pass": True, "note": "0 missing keys.",
             "evidence": ["stages/1/evidence/load-report.json"]},
            {"name": "decodes forward", "pass": True, "note": "32 tokens.",
             "evidence": ["stages/1/evidence/decode.txt"]},
            {"name": "matches the card", "pass": True, "note": "This is a form check only.",
             "evidence": ["stages/1/evidence/card-check.txt"]}]})
    # stage 2
    c2 = make_cache(cache_root / "hemmingway-1" / "tt_cache", marker=marker)
    if nearest_cache:
        c2 = make_cache(op / ".cache" / "src-build" / "tensors", marker=marker)
    write(s / "2" / "evidence" / "server.log", "ready\n")
    write_json(s / "2" / "evidence" / "swap-check.json", {"result_draft": {"top1_agreement": 0.9375}})
    write_json(s / "2" / "swap_config.json", {"tt_cache": str(c2), "bundle_dir": str(op / "bundle-p300"),
                                              "new_model_id": MODEL})
    write_json(s / "2" / "result.json", {
        "serves": True, "server_ready_s": 728.1, "coherent": True, "free_run_text": "Let me work.",
        "top1_agreement": 0.9375, "n_tokens": 32, "cache_dir": str(c2),
        "evidence": ["stages/2/evidence/swap-check.json", "stages/2/evidence/server.log"]})
    # stage 4
    configs = []
    for chips, top1 in ((2, 0.9375), (4, 0.90625)):
        cache = c2 if (shared_cache and chips == 2) else make_cache(
            cache_root / "hemmingway-1" / f"{chips}chip" / "tt_cache", marker=marker)
        ev = s / "4" / "configs" / str(chips) / "evidence"
        write_json(ev / "swap-check.json", {"result_draft": {"top1_agreement": top1}})
        write(s / "4" / "tests" / str(chips) / "output.txt", "ok\n")
        configs.append({"chips": chips, "pass": True, "kind": "bundle", "package": f"episod/p{chips}",
                        "serves": True, "server_ready_s": 150.0 + chips, "coherent": True,
                        "free_run_text": "Let me work.", "top1_agreement": top1, "n_tokens": 32,
                        "cache_dir": str(cache),
                        "evidence": [f"stages/4/configs/{chips}/evidence/swap-check.json",
                                     f"stages/4/tests/{chips}/output.txt"]})
    write(s / "4" / "tests" / "1" / "output.txt", "health timeout\n")
    configs.append({"chips": 1, "pass": False, "reason": "the server did not become healthy",
                    "evidence": ["stages/4/tests/1/output.txt"]})
    write_json(s / "4" / "result.json", {"configs": configs})
    # stage 7
    if stage7:
        profiles = []
        for chips, name, mesh, required in ((1, "hemmingway-1-p150", "P150", False),
                                            (2, "hemmingway-1-p300", "P150x2", True)):
            pkg = s / "7" / "package" / name
            write(pkg / "README.md", card(name, chips, mesh, verified=required))
            write(pkg / "run.sh", "#!/bin/bash\nexec vllm\n")
            write_json(pkg / "tt_kernel_manifest.json",
                       {"name": name, "deps": {"wheels": ["wheels/tt_metal.whl"]}})
            (pkg / "wheels").mkdir()
            (pkg / "wheels" / "tt_metal.whl").write_bytes(b"PK\x03\x04 shipped wheel")
            prof = {"chips": chips, "name": name, "required": required, "source": f"qwen-{name}",
                    "mesh": mesh, "dir": f"stages/7/package/{name}", "verified": required, "scrub": []}
            if required:
                prof["verify"] = {"top1_agreement": 0.9375, "coherent": True, "n_tokens": 32,
                                  "server_ready_s": 2034.4,
                                  "evidence": ["stages/7/verify/evidence/verify.json"]}
            profiles.append(prof)
        write_json(s / "7" / "verify" / "evidence" / "verify.json", {
            "top1_agreement": 0.9375, "coherent": True, "n_tokens": 32, "server_ready_s": 2034.4,
            "evidence": ["stages/7/verify/evidence/verify.json"]})
        write(s / "7" / "package" / "hemmingway-1-p300.package-thin.log", "log\n")
        write_json(s / "7" / "package.json", {
            "format": "v6", "namespace": "episod", "model": "Altworld/Hemmingway-1", "revision": REV,
            "nearest_model": "Qwen/Qwen3.8-27B", "license": "cc-by-nc-4.0", "non_commercial": True,
            "profiles": profiles,
            "skipped_profiles": [{"chips": 4, "reason": "no v6 bundle for 4 chips is installed"}],
            "hf_linked": ["Altworld/Hemmingway-1"], "hf_missing": [],
            "publish_commands": "stages/7/PUBLISH_COMMANDS.txt"})
        if publish:
            write(s / "7" / "PUBLISH_COMMANDS.txt",
                  publish_commands(profiles, namespace="episod", license_id="cc-by-nc-4.0"))
    # the ledger
    with Ledger(run / "ledger.jsonl") as led:
        led.append("run_start", None, model=MODEL, required_chips=[2, 4],
                   package=({"format": "v6", "namespace": "episod",
                             "models_root": str(op / ".cache" / "tt-model" / "models")}
                            if stage7 else None),
                   paths={"orchard_dir": str(REPO), "hf_home": str(op / ".cache" / "huggingface"),
                          "operator_home": str(op),
                          "tt_model_root": str(op / ".cache" / "tt-model" / "models"),
                          "cache_root": str(cache_root)})
        led.append("stage_start", 0, escalated=False, resumed=False)
        led.append("stage_end", 0, result="pass", path="weights-only", evidence=[])
        led.append("stage_start", 1, escalated=False, resumed=False)
        led.append("retry", 1, what="model request", phase="run", attempt=2)
        led.append("stage_end", 1, result="pass", evidence=[])
        led.append("stage_start", 2, escalated=False, resumed=False)
        led.append("escalate", 2, by="stage machine", reasons=["result.json is missing"])
        led.append("stage_end", 2, result="escalate", reasons=["result.json is missing"])
        led.append("stage_start", 2, escalated=True, resumed=False)
        led.append("stage_end", 2, result="pass", evidence=[])
        led.append("stage_start", 3, skip=True)
        led.append("stage_end", 3, result="skipped", reason=SKIP_3_WEIGHTS_ONLY)
        led.append("stage_start", 4, escalated=False, resumed=False)
        led.append("decision", 4, decision="pause", reason="blocked: a test")
        led.append("decision", None, decision="resume", by="operator")
        led.append("stage_end", 4, result="pass", evidence=[])
        led.append("stage_start", 5, skip=True)
        led.append("stage_end", 5, result="skipped", reason=SKIP_5_WEIGHTS_ONLY)
        led.append("stage_start", 6, skip=True)
        led.append("stage_end", 6, result="skipped", reason=SKIP_6_WEIGHTS_ONLY)
        if stage7:
            led.append("stage_start", 7, escalated=False, resumed=False)
            led.append("stage_end", 7, result="pass", evidence=[])
        else:
            led.append("stage_start", 7, skip=True)
            led.append("stage_end", 7, result="skipped", reason=SKIP_7)
        led.append("stage_start", 8, escalated=False, resumed=False)
    # the agent's config and its copy of the template
    write_json(s / "8" / "bundle_config.json", {"run_dir": str(run)})
    shutil.copyfile(TEMPLATE, s / "8" / "build_bundle.py")
    return run


def build(run: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "stages/8/build_bundle.py"], cwd=run, capture_output=True,
                          text=True, timeout=120)


def bundle_of(run: Path) -> Path:
    return run / "stages" / "8" / "bundle"


def section(text: str, heading: str) -> str:
    """The text under `heading` up to the next heading of the same or a higher level."""
    level = heading.split(" ", 1)[0]
    after = text.split(heading + "\n", 1)[1]
    out = []
    for line in after.splitlines():
        if line.startswith("#") and len(line.split(" ", 1)[0]) <= len(level):
            break
        out.append(line)
    return "\n".join(out)


def hazard_block(risks: str, area: str) -> str:
    hazards = section(risks, "## Stage 0 hazards")
    return hazards.split(f"### {area}\n", 1)[1].split("\n### ", 1)[0]


# ---- the bundle and the gate ----------------------------------------------------------------------

def test_the_built_bundle_passes_the_real_stage_8_gate_and_the_scrub(tmp_path):
    run = make_run(tmp_path)
    out = build(run)
    assert out.returncode == 0, out.stderr + out.stdout
    bundle = bundle_of(run)
    for name in ("RESULTS.md", "RISKS.md", "card.md", "PUBLISH_COMMANDS.txt", "ledger.jsonl"):
        assert (bundle / name).is_file(), name
    # The supervisor copies the ledger and stage 7's record and cards before it runs the gate.
    shutil.copyfile(run / "ledger.jsonl", bundle / "ledger.jsonl")
    Supervisor._copy_package(SimpleNamespace(run_dir=run), bundle)
    gate = gate_bundle(run / "stages" / "8", run)
    assert gate.ok, gate.reasons
    assert scrub_bundle(bundle) == []
    assert "scrub: no findings" in out.stdout


def test_results_names_every_stage_with_its_result(tmp_path):
    run = make_run(tmp_path)
    assert build(run).returncode == 0
    results = (bundle_of(run) / "RESULTS.md").read_text()
    for n in range(9):
        assert f"## Stage {n}:" in results, n
    assert "Result: pass" in section(results, "## Stage 2: weights swap check on one board")


def test_the_skipped_stages_show_their_ledger_reasons(tmp_path):
    run = make_run(tmp_path)
    assert build(run).returncode == 0
    results = (bundle_of(run) / "RESULTS.md").read_text()
    flat = " ".join(results.split())
    for n, reason in ((3, SKIP_3_WEIGHTS_ONLY), (5, SKIP_5_WEIGHTS_ONLY), (6, SKIP_6_WEIGHTS_ONLY)):
        assert f"Result: skipped. Reason (from the ledger): {reason}" in flat, n


def test_results_carry_measured_numbers_with_labels_and_evidence(tmp_path):
    run = make_run(tmp_path)
    assert build(run).returncode == 0
    results = (bundle_of(run) / "RESULTS.md").read_text()
    s2 = section(results, "## Stage 2: weights swap check on one board")
    assert "| top1_agreement | 0.9375 | measured | `stages/2/result.json`" in s2
    s4 = section(results, "## Stage 4: weights swap on each chip configuration")
    assert "| 4 | pass | yes | 154.0 (measured) | yes | 0.90625 (measured) |" in s4
    assert "`stages/4/configs/4/evidence/swap-check.json`" in s4
    assert "the server did not become healthy" in s4
    s7 = section(results, "## Stage 7: package (v6 thin bundle) and boot check")
    assert "hemmingway-1-p300" in s7 and "2034.4 (measured)" in s7
    assert "`stages/7/verify/evidence/verify.json`" in s7
    assert "no v6 bundle for 4 chips is installed" in s7
    s0 = section(results, "## Stage 0: intake and delta triage")
    assert "weights-only" in s0 and "`stages/0/evidence/tokenizer-compare.json`" in s0
    s1 = section(results, "## Stage 1: environment and CPU reference")
    assert "matches the card" in s1 and "`stages/1/evidence/card-check.txt`" in s1
    assert "ledger.jsonl" in section(results, "## Stage 8: operator bundle")
    assert "not for publishing" in results


def test_results_count_the_run_from_the_ledger(tmp_path):
    run = make_run(tmp_path)
    assert build(run).returncode == 0
    run_section = section((bundle_of(run) / "RESULTS.md").read_text(), "## The run")
    for line in ("| Escalations | 1 |", "| Retries | 1 |", "| Pauses | 1 |", "| Operator commands | 1 |"):
        assert line in run_section, line
    assert "| Wall time" in run_section and "measured" in run_section


def test_free_text_copied_from_a_result_file_loses_home_paths(tmp_path):
    run = make_run(tmp_path)
    assert build(run).returncode == 0
    for name in ("RESULTS.md", "RISKS.md", "card.md"):
        text = (bundle_of(run) / name).read_text()
        assert "/home/someone" not in text and str(tmp_path / "op") not in text, name


# ---- RISKS.md -------------------------------------------------------------------------------------

def test_a_profile_that_was_not_boot_checked_is_a_todo_risk(tmp_path):
    run = make_run(tmp_path)
    assert build(run).returncode == 0
    todo = section((bundle_of(run) / "RISKS.md").read_text(), "## TODO numbers")
    assert "hemmingway-1-p150" in todo and "verified false" in todo
    assert "hemmingway-1-p300 was not boot-checked" not in todo


def test_what_the_cards_list_under_not_measured_is_a_todo_risk(tmp_path):
    run = make_run(tmp_path)
    assert build(run).returncode == 0
    todo = section((bundle_of(run) / "RISKS.md").read_text(), "## TODO numbers")
    assert "the drafter's acceptance rate on this model" in todo
    assert "a boot of the package from the Hub" in todo
    assert "the 1-chip configuration" in todo           # stage 4's optional config that failed


def test_the_tensor_cache_hazard_is_not_shown_when_two_stages_share_a_cache(tmp_path):
    run = make_run(tmp_path, shared_cache=True)
    assert build(run).returncode == 0
    block = hazard_block((bundle_of(run) / "RISKS.md").read_text(), "tensor_cache")
    assert "Dealt with: Not shown" in block and "share" in block


def test_the_tensor_cache_hazard_is_dealt_with_when_caches_are_distinct_and_marked(tmp_path):
    run = make_run(tmp_path)
    assert build(run).returncode == 0
    block = hazard_block((bundle_of(run) / "RISKS.md").read_text(), "tensor_cache")
    assert "Dealt with: yes" in block


def test_the_tensor_cache_hazard_is_not_shown_when_a_cache_has_no_marker(tmp_path):
    run = make_run(tmp_path, marker=None)
    assert build(run).returncode == 0
    block = hazard_block((bundle_of(run) / "RISKS.md").read_text(), "tensor_cache")
    assert "Dealt with: Not shown" in block and ".orchard-model" in block


def test_the_tensor_cache_hazard_is_not_shown_when_a_cache_is_under_the_nearest_models_cache(tmp_path):
    run = make_run(tmp_path, nearest_cache=True)
    assert build(run).returncode == 0
    block = hazard_block((bundle_of(run) / "RISKS.md").read_text(), "tensor_cache")
    assert "Dealt with: Not shown" in block and "nearest model" in block


def test_hazards_the_evidence_cannot_show_say_not_shown(tmp_path):
    run = make_run(tmp_path)
    assert build(run).returncode == 0
    risks = (bundle_of(run) / "RISKS.md").read_text()
    for area in ("drafter", "disk", "license"):
        assert "Dealt with: Not shown" in hazard_block(risks, area), area
    assert "cc-by-nc-4.0" in section(risks, "## License")
    assert "non-commercial" in section(risks, "## License")


# ---- card.md and the publish commands -------------------------------------------------------------

def test_the_model_card_copies_the_package_cards_and_says_non_commercial(tmp_path):
    run = make_run(tmp_path)
    assert build(run).returncode == 0
    text = (bundle_of(run) / "card.md").read_text()
    assert f"Altworld/Hemmingway-1" in text and REV in text
    assert "Non-commercial use only." in text
    assert "| hemmingway-1-p300 | 2 | P150x2 | passed in stage 7 |" in text
    assert "| hemmingway-1-p150 | 1 | P150 | not boot-checked |" in text
    pkg_card = (run / "stages/7/package/hemmingway-1-p300/README.md").read_text()
    body = pkg_card.split("\n---\n", 1)[1]
    for line in body.splitlines():
        if line and not line.startswith("#"):
            assert line in text, line


def test_the_publish_commands_are_copied_with_their_header(tmp_path):
    run = make_run(tmp_path)
    assert build(run).returncode == 0
    assert (bundle_of(run) / "PUBLISH_COMMANDS.txt").read_text() == \
        (run / "stages/7/PUBLISH_COMMANDS.txt").read_text()


def test_a_missing_publish_commands_file_exits_2_and_names_it(tmp_path):
    run = make_run(tmp_path, publish=False)
    out = build(run)
    assert out.returncode == 2
    assert "stages/7/PUBLISH_COMMANDS.txt" in out.stderr


def test_a_missing_delta_exits_2_and_names_it(tmp_path):
    run = make_run(tmp_path)
    (run / "stages/0/delta.json").unlink()
    out = build(run)
    assert out.returncode == 2 and "stages/0/delta.json" in out.stderr


def test_a_run_whose_stage_7_was_skipped_gets_a_bundle_that_says_so(tmp_path):
    run = make_run(tmp_path, stage7=False)
    out = build(run)
    assert out.returncode == 0, out.stderr
    bundle = bundle_of(run)
    publish = (bundle / "PUBLISH_COMMANDS.txt").read_text()
    assert all(ln.startswith("#") for ln in publish.splitlines() if ln.strip())
    assert "no package" in publish.lower()
    results = (bundle / "RESULTS.md").read_text()
    assert " ".join(SKIP_7.split()) in " ".join(results.split())
    shutil.copyfile(run / "ledger.jsonl", bundle / "ledger.jsonl")
    assert gate_bundle(run / "stages" / "8", run).ok


# ---- the package copy -----------------------------------------------------------------------------

def test_the_package_copy_leaves_out_venvs_caches_and_unshipped_wheels(tmp_path):
    run = make_run(tmp_path)
    pkg = run / "stages/7/package/hemmingway-1-p300"
    write(pkg / "venv" / "bin" / "python", "binary\n")
    write(pkg / ".tt_cache" / "x.tensorbin", "cache\n")
    write(pkg / "model-dir" / "config.json", "{}\n")
    (pkg / "wheels" / "stray.whl").write_bytes(b"PK stray")
    assert build(run).returncode == 0
    copy = bundle_of(run) / "package" / "hemmingway-1-p300"
    assert (copy / "README.md").is_file() and (copy / "run.sh").is_file()
    assert (copy / "wheels" / "tt_metal.whl").is_file()
    for absent in ("venv", ".tt_cache", "model-dir", "wheels/stray.whl"):
        assert not (copy / absent).exists(), absent
    assert not (bundle_of(run) / "package" / "hemmingway-1-p300.package-thin.log").exists()


def test_the_copy_leaves_out_everything_the_package_scrub_forbids():
    from orchard.scrub import FORBIDDEN_DIRS, FORBIDDEN_SUFFIXES
    mod = template_module()
    assert set(FORBIDDEN_DIRS) <= mod.SKIP_DIRS
    assert set(FORBIDDEN_SUFFIXES) <= mod.SKIP_SUFFIXES


def test_a_file_larger_than_the_limit_is_skipped_and_reported(tmp_path):
    mod = template_module()
    src = tmp_path / "src"
    write(src / "small.txt", "ok\n")
    write(src / "big.txt", "x" * 100)
    skipped = []
    written = mod.copy_package_tree(src, tmp_path / "dst", shipped_wheels=set(), skipped=skipped,
                                    max_bytes=50)
    assert (tmp_path / "dst" / "small.txt").is_file() and not (tmp_path / "dst" / "big.txt").exists()
    assert len(skipped) == 1 and skipped[0]["path"].endswith("big.txt")
    assert "larger than" in skipped[0]["reason"]
    assert [p.name for p in written] == ["small.txt"]


# ---- the scrub and the record ----------------------------------------------------------------------

def test_a_hostname_or_token_planted_in_a_package_file_is_reported_by_the_scrub(tmp_path):
    run = make_run(tmp_path)
    host = socket.gethostname()
    write(run / "stages/7/package/hemmingway-1-p300/notes.txt", f"built on {host}\ntoken {TOKEN}\n")
    out = build(run)
    assert out.returncode == 0, out.stderr     # the gate fails the bundle; the script reports it
    assert "package/hemmingway-1-p300/notes.txt: a Hugging Face token" in out.stdout
    if host and host != "localhost":
        assert f"package/hemmingway-1-p300/notes.txt: the hostname {host!r}" in out.stdout
    record = json.loads((run / "stages/8/evidence/bundle-build.json").read_text())
    assert any("a Hugging Face token" in hit for hit in record["scrub"])


def snapshot(root: Path) -> dict:
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file() and not p.is_symlink():
            out[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out


def test_the_script_writes_nothing_outside_stages_8(tmp_path):
    run = make_run(tmp_path)
    before = snapshot(tmp_path)
    assert build(run).returncode == 0
    after = snapshot(tmp_path)
    changed = {p for p in set(before) | set(after) if before.get(p) != after.get(p)}
    stage8 = str(run / "stages" / "8") + "/"
    assert changed and all(p.startswith(stage8) for p in changed), sorted(changed)
    assert all(p.startswith(stage8 + "bundle/") or p.startswith(stage8 + "evidence/")
               for p in changed), sorted(changed)


def test_the_build_record_lists_every_file_written_with_its_sha256(tmp_path):
    run = make_run(tmp_path)
    assert build(run).returncode == 0
    record = json.loads((run / "stages/8/evidence/bundle-build.json").read_text())
    listed = {f["path"]: f["sha256"] for f in record["files"]}
    bundle = bundle_of(run)
    on_disk = {f"stages/8/bundle/{p.relative_to(bundle).as_posix()}":
               hashlib.sha256(p.read_bytes()).hexdigest()
               for p in bundle.rglob("*") if p.is_file()}
    assert listed == on_disk
