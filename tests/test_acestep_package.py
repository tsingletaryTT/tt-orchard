# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/acestep_package.py: stage a package from synthetic evidence, a tiny bundle and tiny weight files, and check that the gate catches each damage.
No chip, no model, no network."""
import json
import shutil
from pathlib import Path

import pytest

from orchard import acestep_package as pk

REPO = Path(__file__).resolve().parents[1]
BARS = REPO / "docs" / "run-logs" / "evidence" / "acestep-step-bars"


def _sweep():
    run = lambda t: {"generate_and_decode_s": t, "vae_s": t / 8, "dit_s": t / 9, "dit_calls": 8}
    return {"device": "chip", "seed": 42, "load": {"chip_open_upload_vae_warm_s": 7.9},
            "rows": [{"duration_s": d, "warmup": 1, "runs": [run(d / 9), run(d / 8.5), run(d / 8)]} for d in (30.0, 60.0)]}


def _evidence(tmp_path):
    ev = tmp_path / "ev"
    ev.mkdir()
    (ev / "sweep.json").write_text(json.dumps(_sweep()))
    shutil.copy(BARS / "lofi-bars.json", ev / "bars-lofi.json")
    shutil.copy(BARS / "folk-bars.json", ev / "bars-folk.json")
    agreement = {"passed": 17, "arrays": 21, "failed": ["step_denoised/v4"], "latent_pcc": 0.98, "latent_rel_l2": 0.19, "waveform_pcc": 0.94, "waveform_rel_l2": 0.35}
    boot = {"ok": True, "ready_s": 16.0, "request_s": 3.3, "duration_s": 30.0, "sample_rate": 48000, "channels": 2, "clip_duration_s": 30.0, "identical_to_recorded_clip": True}
    (ev / "agreement.json").write_text(json.dumps(agreement))
    (ev / "boot.json").write_text(json.dumps(boot))
    summary = pk.make_summary(_sweep(), [json.loads((ev / "bars-lofi.json").read_text()), json.loads((ev / "bars-folk.json").read_text())], agreement, boot)
    (ev / "summary.json").write_text(json.dumps(summary))
    return ev


@pytest.fixture
def staged(tmp_path):
    code = tmp_path / "code"
    (code / "acestep").mkdir(parents=True)
    (code / "acestep" / "a.py").write_text("x = 1\n")
    snap = tmp_path / "snap" / pk.REVISION
    for rel in pk.WEIGHT_FILES:
        (snap / rel).parent.mkdir(parents=True, exist_ok=True)
        (snap / rel).write_bytes(b"w" + rel.encode())
    out = pk.stage(_evidence(tmp_path), code, snap, tmp_path / "pkg")
    return out


def test_a_freshly_staged_package_passes_the_gate(staged):
    assert pk.package_problems(staged) == []


def test_stage_refuses_a_snapshot_that_is_not_the_pinned_revision(tmp_path):
    snap = tmp_path / "other"
    for rel in pk.WEIGHT_FILES:
        (snap / rel).parent.mkdir(parents=True, exist_ok=True)
        (snap / rel).write_bytes(b"w")
    code = tmp_path / "code"
    code.mkdir()
    with pytest.raises(SystemExit):
        pk.stage(_evidence(tmp_path), code, snap, tmp_path / "pkg")


def _problems_after(staged, edit):
    edit(staged)
    return pk.package_problems(staged)


def test_a_weight_file_is_caught(staged):
    p = _problems_after(staged, lambda d: (d / "model.safetensors").write_bytes(b"w"))
    assert any("weight" in x.lower() or "safetensors" in x for x in p), p


def test_an_edited_card_number_is_caught(staged):
    p = _problems_after(staged, lambda d: (d / "README.md").write_text((d / "README.md").read_text().replace("17 of 21", "21 of 21")))
    assert any("a number differs" in x for x in p), p


def test_a_changed_runtime_file_is_caught_by_the_record_and_the_sums(staged):
    p = _problems_after(staged, lambda d: (d / "runtime" / "acestep_serve.py").write_text((d / "runtime" / "acestep_serve.py").read_text() + "\n# x\n"))
    assert any("changed after it was staged" in x for x in p) and any("SHA256SUMS" in x for x in p), p


def test_changed_pins_are_caught(staged):
    def edit(d):
        pins = json.loads((d / "pins.json").read_text())
        pins["bundle_revision"] = "0" * 40
        (d / "pins.json").write_text(json.dumps(pins))
    p = _problems_after(staged, edit)
    assert any("bundle_revision" in x for x in p), p


def test_a_wrong_license_front_matter_is_caught(staged):
    p = _problems_after(staged, lambda d: (d / "README.md").write_text((d / "README.md").read_text().replace("license: apache-2.0", "license: mit", 1)))
    assert any("front matter: license" in x for x in p), p


def test_a_home_path_and_a_key_file_name_are_caught(staged):
    def edit(d):
        (d / "tools" / "extra.py").write_text("P = '/home/someone/x'\nK = '.tt_ai_key'\n")
    p = _problems_after(staged, edit)
    assert any("scrub" in x for x in p) and any(".tt_ai_key" in x for x in p), p


def test_an_overclaim_in_the_card_is_caught(staged):
    p = _problems_after(staged, lambda d: (d / "README.md").write_text((d / "README.md").read_text() + "\nThis is production-ready.\n"))
    assert any("overclaim" in x for x in p), p


def test_a_failed_boot_or_refused_bars_are_caught(staged):
    def edit(d):
        s = json.loads((d / "evidence" / "summary.json").read_text())
        s["boot"]["ok"] = False
        s["step_bars"][0]["usable"] = False
        (d / "evidence" / "summary.json").write_text(json.dumps(s))
    p = _problems_after(staged, edit)
    assert any("boot check did not pass" in x for x in p) and any("refused" in x for x in p), p


def test_a_link_in_the_package_is_caught(staged):
    def edit(d):
        (d / "evidence" / "link.json").symlink_to(d / "pins.json")
    assert _problems_after(staged, edit), "a link must produce a problem"


def test_the_manifest_ready_line_must_be_in_the_server(staged):
    p = _problems_after(staged, lambda d: (d / "runtime" / "acestep_serve.py").write_text("print('hi')\n"))
    assert any("ready line marker" in x for x in p), p


def test_run_sh_is_valid_shell_and_names_both_pins(staged):
    text = (staged / "run.sh").read_text()
    assert pk.REVISION in text and pk.BUNDLE_REVISION in text and "--local-dir" in text


def test_the_checked_in_evidence_directory_can_stage_a_package(tmp_path):
    code = tmp_path / "code"
    code.mkdir()
    (code / "a.py").write_text("x = 1\n")
    snap = tmp_path / "snap" / pk.REVISION
    for rel in pk.WEIGHT_FILES:
        (snap / rel).parent.mkdir(parents=True, exist_ok=True)
        (snap / rel).write_bytes(b"w")
    out = pk.stage(REPO / "docs" / "run-logs" / "evidence" / "acestep-package", code, snap, tmp_path / "pkg")
    assert pk.package_problems(out) == []


def test_run_sh_honours_hf_hub_cache(staged):
    assert "HF_HUB_CACHE" in (staged / "run.sh").read_text()


@pytest.mark.parametrize("name,content", [("package-record.json", "[]"), ("pins.json", "[]"), ("pins.json", '{"weights": 5}'), ("evidence/summary.json", "[]"),
                                          ("tt-model.yaml", "- a\n- b\n"), ("tt-model.yaml", "runtime: 5\nweights: 5\n")])
def test_malformed_metadata_gives_problems_not_a_traceback(staged, name, content):
    (staged / name).write_text(content)
    assert pk.package_problems(staged), "a malformed file must be reported"


def test_the_manifest_command_names_the_paths_the_server_requires(staged):
    text = (staged / "tt-model.yaml").read_text()
    assert all(f in text for f in ("--bundle-code", "--snapshot", "--root"))
    (staged / "tt-model.yaml").write_text(text.replace("--root, /work, ", ""))
    assert any("lacks --root" in x for x in pk.package_problems(staged))


def test_run_sh_honours_xdg_cache_home(staged):
    assert "XDG_CACHE_HOME" in (staged / "run.sh").read_text()


def test_every_acestep_module_the_staged_runtime_imports_is_staged(staged):
    # a runtime file that imports a sibling the stager does not copy fails only in the packaged copy
    import re
    names = set()
    for f in (staged / "runtime").glob("*.py"):
        names |= set(re.findall(r"^\s*import (acestep_\w+)", f.read_text(), re.M)) | set(re.findall(r"^\s*from (acestep_\w+) import", f.read_text(), re.M))
    assert names, "the runtime imports no acestep_* sibling; the test is not looking at anything"
    assert all((staged / "runtime" / f"{n}.py").is_file() for n in names), sorted(n for n in names if not (staged / "runtime" / f"{n}.py").is_file())
