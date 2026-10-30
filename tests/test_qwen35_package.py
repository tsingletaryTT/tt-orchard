# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/qwen35_package.py: the staged recipe package for Qwen3.5-35B-A3B, its gate, its comparison script and its launcher."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from orchard import qwen35_package as qp

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "package_src"
FIXCFG = ROOT / "tests" / "fixtures" / "model-configs" / "qwen3.5-35b-a3b.json"


def _top(ids, lps):
    return {"ids": ids, "logprobs": lps}


def make_evidence(d: Path, *, agree=False):
    d.mkdir(parents=True)
    runs = [{"tokens": 128, "ttft_s": t, "decode_tok_s": v, "load1": 5.0 + i / 10} for i, (t, v) in enumerate([(0.11, 126.8), (0.10, 126.9), (0.12, 126.7)])]
    bench = {"runs": runs, "min": 126.7, "median": 126.8, "max": 126.9}
    (d / "bench_package.log").write_text("BENCH " + json.dumps(bench) + "\n")
    for name, med in (("bench_40layer_grid105.log", 126.84), ("bench_40layer_rk72.log", 126.78)):
        (d / name).write_text("BENCH " + json.dumps(dict(bench, median=med)) + "\n")
    (d / "boot.txt").write_text("BOOT cold api_up_s=727 first_answer_s=728\nBOOT warm api_up_s=308 first_answer_s=309\n")
    ref, dev = [], []
    for i, p in enumerate(["alpha prompt", "beta prompt"]):
        toks = [10, 11, 12, 13]
        ref.append({"prompt": p, "prompt_tokens": 3, "tokens": toks, "top5": [_top([t, 99, 98, 97, 96], [-0.5, -1.0, -2.0, -3.0, -4.0]) for t in toks], "text": "x"})
        dt = list(toks) if (agree or i == 1) else [10, 11, 99, 13]
        dev.append({"prompt": p, "tokens": dt, "top5": [], "text": "y"})
    (d / "cpu_reference.json").write_text(json.dumps(ref))
    (d / "device_quality.json").write_text(json.dumps({
        "completions": dev,
        "chat": {"default": {"text": "A card. " * 30, "finish": "length"}, "thinking_off": {"text": "Red, blue, yellow.", "finish": "stop"}},
        "long_prompt": {"prompt_tokens": 3100, "text": " PELICAN.\nmore"}}))
    shutil.copy(FIXCFG, d / "config.json")
    return d


@pytest.fixture
def ev(tmp_path):
    return make_evidence(tmp_path / "ev")


@pytest.fixture
def pkg(tmp_path, ev):
    out = tmp_path / "pkg"
    qp.stage_package(SRC, ev, out)
    return out


def gate(p):
    return qp.package_problems(p, hostname="testhost", home="/home/tester")


def rewrite(path: Path, old: str, new: str):
    t = path.read_text()
    assert old in t, old
    path.write_text(t.replace(old, new))


# ---- stage and gate ---------------------------------------------------------------------------------------------------

def test_a_staged_package_passes_the_gate(pkg):
    assert gate(pkg) == []


def test_staging_refuses_an_existing_directory(pkg, ev):
    with pytest.raises(qp.PackageError):
        qp.stage_package(SRC, ev, pkg)


def test_a_missing_evidence_file_stops_staging(tmp_path, ev):
    (ev / "boot.txt").unlink()
    with pytest.raises(qp.PackageError, match="boot.txt"):
        qp.stage_package(SRC, ev, tmp_path / "p")


def test_an_all_agree_run_stages_and_says_all_tokens_matched(tmp_path):
    ev = make_evidence(tmp_path / "ev2", agree=True)       # no differences at all: max() of nothing used to raise
    out = tmp_path / "p"
    qp.stage_package(SRC, ev, out)
    card = (out / "README.md").read_text()
    assert "All tokens of every prompt matched the reference." in card and "2 of 2 prompts matched in full" in card
    assert gate(out) == []


def test_weights_file_fails_and_nothing_else_is_read(pkg):
    (pkg / "model.safetensors").write_bytes(b"x")
    (pkg / "README.md").write_text("changed after staging")      # would be a record problem if the gate read on
    p = gate(pkg)
    assert len(p) == 1 and "weights" in p[0]


def test_oversized_file_returns_at_once(pkg):
    (pkg / "data.txt").write_text("a" * (qp.MAX_FILE_BYTES + 1))
    (pkg / "run.sh").write_text("changed")
    p = gate(pkg)
    assert len(p) == 1 and "larger than" in p[0]


def test_home_path_fails(pkg):
    (pkg / "docs" / "running.md").write_text("see /home/tester/x\n")
    assert any("home path" in p for p in gate(pkg))


def test_credential_file_returns_at_once_and_litellm_text_fails(pkg):
    (pkg / "notes.txt").write_text("uses litellm\n")
    assert any("litellm" in x for x in gate(pkg))
    (pkg / ".tt_ai_key").write_text("k")
    p = gate(pkg)
    assert len(p) == 1 and "credential" in p[0]


def test_token_fails(pkg):
    (pkg / "notes.txt").write_text("hf_" + "a" * 34)
    assert any("token" in p for p in gate(pkg))


def test_edited_file_after_staging_fails_the_record(pkg):
    (pkg / "run.sh").write_text((pkg / "run.sh").read_text() + "# x\n")
    assert any("run.sh was changed" in p for p in gate(pkg))


def test_unrecorded_file_fails(pkg):
    (pkg / "extra.txt").write_text("x")
    assert any("not in the record" in p for p in gate(pkg))


def test_license_must_be_apache(pkg):
    (pkg / "LICENSE").write_text("Some other license\n")
    assert any("not the Apache" in p for p in gate(pkg))


def test_a_different_revision_in_the_manifest_fails(pkg):
    rewrite(pkg / "tt-model.yaml", qp.REVISION, "0" * 40)
    assert any("tt-model.yaml does not pin" in p for p in gate(pkg))


def test_pins_must_match_the_module(pkg):
    rewrite(pkg / "pins.env", qp.PATCHED_SHA, "0" * 64)
    assert any("pins.env PATCHED_SHA" in p for p in gate(pkg))


def test_symlink_fails(pkg, tmp_path):
    (pkg / "link").symlink_to(tmp_path)
    assert any("is a link" in p for p in gate(pkg))


def test_executable_bit_is_required_on_run_sh(pkg):
    (pkg / "run.sh").chmod(0o644)
    assert any("run.sh is not executable" in p for p in gate(pkg))


# ---- the card must be a rendering of the evidence ---------------------------------------------------------------------

def test_cold_start_edited_in_the_card_fails(pkg):
    rewrite(pkg / "README.md", "728 seconds", "900 seconds")
    assert any("not the card rendered from the evidence" in p for p in gate(pkg))


def test_cold_start_edited_in_facts_json_fails(pkg):
    f = json.loads((pkg / "evidence" / "facts.json").read_text())
    f["cold_start_s"] = 900
    (pkg / "evidence" / "facts.json").write_text(json.dumps(f))
    assert any("facts.json differs" in p and "cold_start_s" in p for p in gate(pkg))


def test_cold_start_edited_in_the_raw_boot_file_fails(pkg):
    rewrite(pkg / "evidence" / "boot.txt", "first_answer_s=728", "first_answer_s=900")
    p = gate(pkg)
    assert any("facts.json differs" in x for x in p) and any("not the card rendered" in x for x in p)


@pytest.mark.parametrize("old,new", [("| 110 ms |", "| 90 ms |"), ("309 seconds on the boot check", "270 seconds on the boot check"),
                                     ("2 of 4", "3 of 4"), ("126.80 tok/s", "130.00 tok/s"), ("(largest gap", "(largest gap 9")])
def test_every_measurement_field_is_checked(pkg, old, new):
    text = (pkg / "README.md").read_text()
    if old not in text:
        pytest.skip(f"{old!r} is not in this fixture's card")
    rewrite(pkg / "README.md", old, new)
    assert any("not the card rendered" in p for p in gate(pkg))


def test_ttft_warm_and_quality_values_appear_in_the_card(pkg):
    card = (pkg / "README.md").read_text()
    assert "110 ms" in card and "309 seconds" in card and "2 of 4" in card


def test_a_raw_quality_edit_fails(pkg):
    f = pkg / "evidence" / "device_quality.json"
    d = json.loads(f.read_text())
    d["completions"][0]["tokens"] = [10, 11, 12, 13]
    f.write_text(json.dumps(d))
    p = gate(pkg)
    assert any("was changed after staging" in x for x in p) and any("not the card rendered" in x for x in p)


def test_publish_text_is_text_with_the_operator_caveat():
    t = qp.publish_text("ns")
    assert "has not approved" in t and "hf upload" in t and t.startswith("#")


def test_fetch_script_pins_the_revision():
    assert qp.REVISION in (SRC / "fetch_weights.py").read_text()


# ---- check/compare.py -------------------------------------------------------------------------------------------------

def compare_mod():
    spec = importlib.util.spec_from_file_location("cmp", SRC / "check" / "compare.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _r(p, toks):
    return {"prompt": p, "tokens": toks, "top5": [_top([t, 9, 8, 7, 6], [-0.1, -1, -2, -3, -4]) for t in toks]}


def test_compare_rejects_unequal_prompt_counts():
    with pytest.raises(ValueError, match="prompts"):
        compare_mod().compare_rows([_r("a", [1]), _r("b", [1])], [_r("a", [1])])


def test_compare_rejects_unequal_token_counts():
    with pytest.raises(ValueError, match="tokens"):
        compare_mod().compare_rows([_r("a", [1, 2, 3])], [_r("a", [1, 2])])


def test_compare_rejects_different_prompts_and_empty_token_lists():
    with pytest.raises(ValueError, match="differs"):
        compare_mod().compare_rows([_r("a", [1])], [_r("b", [1])])
    with pytest.raises(ValueError, match="tokens"):
        compare_mod().compare_rows([_r("a", [])], [_r("a", [])])


def test_compare_leading_agree_is_the_full_length_when_nothing_differs():
    row = compare_mod().compare_rows([_r("a", [1, 2, 3])], [_r("a", [1, 2, 3])])[0]
    assert row["leading_agree"] == 3 and "first_diff" not in row and row["same_position_share"] == 1.0


def test_compare_reports_the_first_difference():
    row = compare_mod().compare_rows([_r("a", [1, 2, 3])], [_r("a", [1, 9, 3])])[0]
    assert row["leading_agree"] == 1 and row["first_diff"]["index"] == 1 and row["first_diff"]["device_token_in_ref_top5"] is True


# ---- run.sh -----------------------------------------------------------------------------------------------------------

ORIG = "original loader\n"
PATCH_TEXT = "--- a/tt/weights.py\n+++ b/tt/weights.py\n@@ -1 +1,2 @@\n original loader\n+patched line\n"


@pytest.fixture
def launcher(tmp_path):
    """run.sh in a temporary directory with its own pins, a stub `docker` that logs its arguments, a fake sysfs and fake device nodes."""
    d = tmp_path / "pkgdir"
    d.mkdir()
    shutil.copy(SRC / "run.sh", d / "run.sh")
    (d / "weights_qwen35_moe.patch").write_text(PATCH_TEXT)
    sha = lambda b: hashlib.sha256(b).hexdigest()
    patched = "original loader\npatched line\n"
    (d / "pins.env").write_text(f"IMAGE_TAG=img:tag\nIMAGE_ID=sha256:rec\nLOADER=/x/weights.py\nORIG_SHA={sha(ORIG.encode())}\nPATCHED_SHA={sha(patched.encode())}\n")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "docker").write_text("""#!/bin/bash
if [ "$1" = image ]; then [ -n "$FAKE_MISSING" ] && exit 1; [ "$4" = --format ] && echo "$FAKE_ID"; exit 0; fi
if [ "$1" = run ] && [ "$4" = cat ]; then printf '%s' "$FAKE_LOADER"; exit 0; fi
echo "$@" > "$FAKE_LOG"; exit 0
""")
    (bin_ / "docker").chmod(0o755)
    sysfs = tmp_path / "pci"
    for bdf, node in (("0000:01:00.0", 0), ("0000:02:00.0", 1), ("0000:03:00.0", 2), ("0000:04:00.0", 3)):
        (sysfs / bdf / "tenstorrent" / f"tenstorrent!{node}").mkdir(parents=True)
    devs = tmp_path / "dev"
    devs.mkdir()
    for n in range(4):
        (devs / str(n)).write_text("")
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_text("")
    env = {"PATH": f"{bin_}:{os.environ['PATH']}", "FAKE_ID": "sha256:rec", "FAKE_LOADER": ORIG, "FAKE_LOG": str(tmp_path / "docker.log"),
           "SYSFS_PCI": str(sysfs), "DEV_ROOT": str(devs), "MODEL_DIR": str(model), "WORK": str(tmp_path / "work"), "HOME": str(tmp_path)}

    def run(**over):
        e = dict(env, **over)
        for k in [k for k, v in e.items() if v is None]:
            del e[k]
        r = subprocess.run(["bash", str(d / "run.sh")], env=e, capture_output=True, text=True)
        log = Path(env["FAKE_LOG"])
        return r, (log.read_text() if log.exists() else "")
    run.dir, run.work = d, tmp_path / "work"
    return run


def test_run_sh_maps_a_lease_to_its_device_nodes(launcher):
    r, log = launcher(TT_VISIBLE_DEVICES="0000:03:00.0,0000:04:00.0")
    assert r.returncode == 0, r.stderr
    assert "--device /dev/tenstorrent/2" in log and "--device /dev/tenstorrent/3" in log and "/dev/tenstorrent/0" not in log
    assert "TT_VISIBLE_DEVICES=0,1" in log            # inside the container the two dies are 0 and 1


def test_run_sh_refuses_devices_outside_the_lease(launcher):
    r, log = launcher(TT_VISIBLE_DEVICES="0000:03:00.0,0000:04:00.0", DEVICES="0 1")
    assert r.returncode == 2 and "not the leased set" in r.stderr and log == ""


def test_run_sh_accepts_devices_that_equal_the_lease(launcher):
    r, log = launcher(TT_VISIBLE_DEVICES="0000:03:00.0,0000:04:00.0", DEVICES="3 2")
    assert r.returncode == 0 and "/dev/tenstorrent/2" in log


def test_run_sh_refuses_with_no_lease_and_no_devices(launcher):
    r, _ = launcher(TT_VISIBLE_DEVICES=None)
    assert r.returncode == 2 and "TT_VISIBLE_DEVICES" in r.stderr


def test_run_sh_takes_explicit_devices_when_there_is_no_lease(launcher):
    r, log = launcher(TT_VISIBLE_DEVICES=None, DEVICES="0 1")
    assert r.returncode == 0 and "/dev/tenstorrent/0" in log


def test_run_sh_refuses_an_unknown_pci_address_and_a_wrong_count(launcher):
    assert launcher(TT_VISIBLE_DEVICES="0000:09:00.0,0000:04:00.0")[0].returncode == 2
    r, _ = launcher(TT_VISIBLE_DEVICES="0000:03:00.0")
    assert r.returncode == 2 and "exactly two dies" in r.stderr


def test_run_sh_refuses_an_image_id_mismatch(launcher):
    r, log = launcher(TT_VISIBLE_DEVICES="0,1", FAKE_ID="sha256:other")
    assert r.returncode == 2 and "differs from the recorded" in r.stderr and log == ""


def test_run_sh_accepts_another_tag_that_resolves_to_the_recorded_id(launcher):
    r, _ = launcher(TT_VISIBLE_DEVICES="0,1", IMAGE="other/name:latest")
    assert r.returncode == 0, r.stderr


def test_run_sh_refuses_an_image_that_is_not_present(launcher):
    r, _ = launcher(TT_VISIBLE_DEVICES="0,1", FAKE_MISSING="1")
    assert r.returncode == 2 and "not on this machine" in r.stderr


def test_run_sh_refuses_a_loader_that_is_not_the_patched_one_on_every_launch(launcher):
    assert launcher(TT_VISIBLE_DEVICES="0,1")[0].returncode == 0
    (launcher.work.parent / "docker.log").unlink()
    (launcher.work / "weights.py").write_text("tampered\n")          # left over from an earlier launch, edited since
    r, log = launcher(TT_VISIBLE_DEVICES="0,1")
    assert r.returncode == 2 and "does not match the measured patched loader" in r.stderr and log == ""


def test_run_sh_refuses_an_image_loader_that_the_patch_was_not_made_for(launcher):
    r, _ = launcher(TT_VISIBLE_DEVICES="0,1", FAKE_LOADER="some other file\n")
    assert r.returncode == 2 and "not the file the patch was made for" in r.stderr


# ---- review round 2: untrusted package content ------------------------------------------------------------------------

def test_the_gate_never_executes_code_in_the_package(pkg, tmp_path):
    marker = tmp_path / "ran"
    f = pkg / "check" / "compare.py"
    f.write_text(f"open({str(marker)!r}, 'w').write('x')\n" + f.read_text())
    p = gate(pkg)
    assert not marker.exists()
    assert any("check/compare.py was changed after staging" in x for x in p)


def test_staging_does_not_execute_the_source_compare_script_either(tmp_path, ev):
    src = tmp_path / "src"
    shutil.copytree(SRC, src)
    marker = tmp_path / "ran2"
    f = src / "check" / "compare.py"
    f.write_text(f"open({str(marker)!r}, 'w').write('x')\n" + f.read_text())
    qp.stage_package(src, ev, tmp_path / "p")
    assert not marker.exists()


def test_the_shipped_compare_script_and_the_trusted_comparison_agree(ev):
    ref = json.loads((ev / "cpu_reference.json").read_text())
    dev = json.loads((ev / "device_quality.json").read_text())["completions"]
    assert compare_mod().compare_rows(ref, dev) == qp.compare_rows(ref, dev)


def _set_record(pkg, value):
    (pkg / "package-record.json").write_text(value if isinstance(value, str) else json.dumps(value))


@pytest.mark.parametrize("bad", ["../outside.safetensors", "/etc/hostname", "a/../../outside.txt", "", "evidence//x", "evidence/./facts.json"])
def test_record_paths_that_escape_or_are_not_plain_are_rejected_without_reading(pkg, tmp_path, bad):
    outside = tmp_path / "outside.safetensors"
    outside.write_text("big model")
    rec = json.loads((pkg / "package-record.json").read_text())
    rec["files"][bad] = "0" * 64
    _set_record(pkg, rec)
    p = gate(pkg)
    assert any("record path" in x and "not a plain path inside the package" in x for x in p), p


def test_a_record_path_through_a_link_is_rejected(pkg, tmp_path):
    (tmp_path / "target.txt").write_text("x")
    # a directory link inside the package would be reported as a link first; a record naming a path under one must not be read either
    rec = json.loads((pkg / "package-record.json").read_text())
    rec["files"]["docs/../README.md"] = "0" * 64
    _set_record(pkg, rec)
    assert any("not a plain path inside the package" in x for x in gate(pkg))


@pytest.mark.parametrize("bad", ["{", "[]", "null", '{"files": null}', '{"files": []}', '{"files": {"README.md": 5}}', '{"files": {"README.md": "xyz"}}', '{"files": {}}'])
def test_a_malformed_record_is_a_problem_not_a_crash(pkg, bad):
    _set_record(pkg, bad)
    p = gate(pkg)
    assert any("package-record.json" in x for x in p), p


@pytest.mark.parametrize("bad", ["{", "[]", "null", '"text"', "5"])
def test_a_malformed_facts_file_is_a_problem_not_a_crash(pkg, bad):
    (pkg / "evidence" / "facts.json").write_text(bad)
    p = gate(pkg)
    assert any("facts.json" in x for x in p), p


@pytest.mark.parametrize("name,content", [("device_quality.json", '{"completions": [1, 2]}'), ("device_quality.json", "[]"),
                                          ("cpu_reference.json", '{"a": 1}'), ("bench_package.log", "BENCH [1]"), ("bench_package.log", 'BENCH {"runs": [3]}'),
                                          ("boot.txt", "BOOT cold api_up_s=1 first_answer_s=2"), ("config.json", "[]")])
def test_malformed_evidence_is_a_named_problem_and_staging_error(pkg, tmp_path, name, content):
    (pkg / "evidence" / name).write_text(content)
    p = gate(pkg)
    assert p and not any("Traceback" in x for x in p)
    bad_ev = make_evidence(tmp_path / "bad_ev")
    (bad_ev / name).write_text(content)
    with pytest.raises(qp.PackageError):
        qp.stage_package(SRC, bad_ev, tmp_path / "never")


def test_a_non_utf8_package_file_is_a_problem_not_a_crash(pkg):
    (pkg / "README.md").write_bytes(b"\xff\xfe\x00bad")
    p = gate(pkg)
    assert any("README.md" in x for x in p), p
