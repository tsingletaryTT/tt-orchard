# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The text package builder (orchard/text_package.py): it stages from a finished fake run, the gate passes the staged package, and each rule turns red when its
subject is damaged."""
import json
import shutil
import sys
import types
from pathlib import Path

import pytest

from orchard import text_package as tp, text_runtime as rt
from test_text_stages import done  # noqa: F401  (the fixture: a finished fake run)
from test_supervisor_text_runtime import trig  # noqa: F401

M = rt.get("minicpm3")
REV = "a" * 40            # the fake run's snapshot directory is named for this revision
TERMS = "The card says weights follow the MiniCPM Model License; the link was not reachable when this was staged."


@pytest.fixture
def staged(done, tmp_path):
    lic = tmp_path / "LICENSE-src"
    lic.write_text("Apache License\nVersion 2.0, January 2004\n")
    ctx = tmp_path / "ctx.json"
    ctx.write_text(json.dumps({"tp": 1, "windows": [{"start_position": 32, "steps": 32, "seconds": 1.0, "tok_s": 33.0},
                                                    {"start_position": 4000, "steps": 32, "seconds": 1.0, "tok_s": 30.5}]}))
    out = tmp_path / "pkgs"
    out.mkdir()
    rec = tp.stage_package(done.run_dir, rtm=M, namespace="example-org", revision=REV, license_file=lic, context_file=ctx, out=out, weights_terms=TERMS)
    pkg = out / M.package_name
    return types.SimpleNamespace(run=done.run_dir, pkg=pkg, out=out, rec=rec, lic=lic, ctx=ctx)


def gate(s, **kw):
    return tp.package_problems(s.pkg, rtm=M, hostname="examplehost", home="/home/example", **kw)


def edit(path, fn):
    p = Path(path)
    p.write_text(fn(p.read_text()))


def test_a_staged_package_passes_the_gate_and_holds_no_weights(staged):
    assert gate(staged) == []
    names = {p.relative_to(staged.pkg).as_posix() for p in staged.pkg.rglob("*") if p.is_file()}
    assert {"README.md", "LICENSE", "NOTICE", "tt-model.yaml", "run.sh", "package-record.json", "runtime/scripts/minicpm3/serve.py",
            "runtime/scripts/minicpm3/chip_model.py", "evidence/text_bench.json", "evidence/context-bench.json"} <= names
    assert not [n for n in names if n.endswith((".safetensors", ".bin", ".pt"))]


def test_the_runtime_imports_are_all_in_the_package(staged):
    for f in (staged.pkg / "runtime" / "scripts" / "minicpm3").glob("*.py"):
        for m in tp.orchard_imports(f.read_text()):
            assert (staged.pkg / "runtime" / "orchard" / f"{m}.py").is_file(), (f.name, m)


def test_the_card_names_the_numbers_and_the_limits(staged):
    card = (staged.pkg / "README.md").read_text()
    assert "| decode speed, bf16 | 33" in card and "teacher-forced next-token agreement, bf16 | 4 of 4" in card
    assert "## Not tested" in card and "## License" in card and "MiniCPM Model License" in card and "decode speed at the deepest position" in card
    assert card.startswith("---\nlicense: apache-2.0\n")


def test_the_card_says_what_the_model_is_for_and_where_it_is_not_a_fit(staged):
    card = (staged.pkg / "README.md").read_text()
    assert "## What it is for" in card and card.index("## What it is for") < card.index("## Numbers")
    section = card.split("## What it is for")[1].split("\n## ")[0]
    assert "Chinese and English" in section and "chat" in section
    assert "authors say" in section.lower() and "not tested here" in section.lower()
    assert "greedy" in section and "2048" in section


def test_the_publish_commands_are_text_only(staged):
    text = tp.publish_commands(M, "example-org", "x/pkg")
    assert "hf upload --repo-type model --private example-org/minicpm3-4b-blackhole x/pkg ." in text and "#    hf repos settings" in text


def test_staging_refuses_an_existing_directory_a_bad_revision_and_a_failed_stage(done, staged, tmp_path):
    with pytest.raises(tp.TextPackageError, match="not overwritten|never overwritten"):
        tp.stage_package(done.run_dir, rtm=M, namespace="o", revision=REV, license_file=staged.lic, out=staged.out, weights_terms=TERMS)
    with pytest.raises(tp.TextPackageError, match="40-character"):
        tp.stage_package(done.run_dir, rtm=M, namespace="o", revision="abc", license_file=staged.lic, out=tmp_path / "x", weights_terms=TERMS)
    ledger = done.run_dir / "ledger.jsonl"
    saved = ledger.read_text()
    ledger.write_text("\n".join(l for l in saved.splitlines() if not ('"stage_end"' in l and '"stage":6' in l.replace(" ", ""))) + "\n")
    with pytest.raises(tp.TextPackageError, match="stage 6"):
        tp.stage_package(done.run_dir, rtm=M, namespace="o", revision=REV, license_file=staged.lic, out=tmp_path / "y", weights_terms=TERMS)


def test_a_license_file_that_is_not_apache_is_refused(done, tmp_path):
    bad = tmp_path / "bad"
    bad.write_text("All rights reserved\n")
    with pytest.raises(tp.TextPackageError, match="Apache License 2.0"):
        tp.stage_package(done.run_dir, rtm=M, namespace="o", revision=REV, license_file=bad, out=tmp_path / "z", weights_terms=TERMS)


# ---- the gate: each rule turns red ------------------------------------------------------------------------------------------

def test_a_weights_file_is_refused(staged):
    (staged.pkg / "model.safetensors").write_bytes(b"x")
    assert any("weights" in p for p in gate(staged))


def test_a_home_path_a_host_name_and_a_forbidden_word_are_refused(staged):
    edit(staged.pkg / "docs" / "running.md", lambda t: t + "\n/home/example/x examplehost .tt_ai_key litellm\n")
    text = " ".join(gate(staged))
    assert "scrub" in text and ".tt_ai_key" in text and "litellm" in text


def test_a_card_number_that_differs_from_the_evidence_is_refused(staged):
    edit(staged.pkg / "README.md", lambda t: t.replace("| decode speed, bf16 | 33", "| decode speed, bf16 | 99"))
    assert any("Numbers table" in p for p in gate(staged))


def test_a_speed_in_prose_is_refused(staged):
    edit(staged.pkg / "README.md", lambda t: t.replace("## Running it", "## Running it\n\nIt reaches 40 tok/s.\n"))
    assert any("outside its Numbers table" in p for p in gate(staged))


def test_a_commercial_use_claim_is_refused(staged):
    edit(staged.pkg / "README.md", lambda t: t.replace("## Not tested", "Commercial use is allowed.\n\n## Not tested"))
    assert any("commercial-use" in p for p in gate(staged))


def test_a_card_without_the_not_tested_section_is_refused(staged):
    edit(staged.pkg / "README.md", lambda t: t.replace("## Not tested", "## Other"))
    assert any("Not tested" in p for p in gate(staged))


def test_a_card_license_that_is_not_the_adapters_is_refused(staged):
    edit(staged.pkg / "README.md", lambda t: t.replace("license: apache-2.0", "license: mit", 1))
    assert any("card's license" in p for p in gate(staged))


def test_a_changed_license_file_is_refused(staged):
    (staged.pkg / "LICENSE").write_text("Apache License\nVersion 2.0, edited\n")
    problems = gate(staged)
    assert any("LICENSE" in p and "differs" in p for p in problems) and any("not the license text" in p for p in problems)


def rerecord(staged, rel):
    """Make the record agree with an edited file, so that only the rule under test can object."""
    rec_path = staged.pkg / tp.RECORD_FILE
    rec = json.loads(rec_path.read_text())
    for f in rec["files"]:
        if f["path"] == rel:
            f["sha256"] = tp.sha((staged.pkg / rel).read_bytes())
    rec_path.write_text(json.dumps(rec))


def test_a_launcher_that_does_not_check_the_configuration_is_refused(staged):
    edit(staged.pkg / "run.sh", lambda t: t.replace("sha256sum", "true"))
    rerecord(staged, "run.sh")
    assert any("run.sh must carry the recorded configuration sha256" in p for p in gate(staged))


def test_a_card_that_does_not_name_the_revision_is_refused(staged):
    edit(staged.pkg / "README.md", lambda t: t.replace(f"`{REV}`", "`unknown`"))
    rerecord(staged, "README.md")
    assert any("must name the recorded model revision" in p for p in gate(staged))


def test_a_run_whose_default_variant_missed_the_bar_is_not_packaged(done, tmp_path):
    p = done.run_dir / "stages" / "2" / "text_result.json"
    old = tp.sha(p.read_bytes())
    edit(p, lambda t: t.replace('"ok": true', '"ok": false'))
    edit(done.run_dir / "ledger.jsonl", lambda t: t.replace(old, tp.sha(p.read_bytes())))      # as if the gate had accepted this file
    lic = tmp_path / "L"
    lic.write_text("Apache License\nVersion 2.0\n")
    with pytest.raises(tp.TextPackageError, match="default variant did not meet"):
        tp.stage_package(done.run_dir, rtm=M, namespace="o", revision=REV, license_file=lic, out=tmp_path / "w", weights_terms=TERMS)


def test_a_manifest_that_does_not_pin_the_revision_is_refused(staged):
    edit(staged.pkg / "tt-model.yaml", lambda t: t.replace(REV, "e" * 40))
    assert any("pin the recorded model revision" in p for p in gate(staged))


def test_a_file_not_in_the_record_and_a_file_changed_after_staging_are_refused(staged):
    (staged.pkg / "extra.txt").write_text("x")
    edit(staged.pkg / "runtime" / "scripts" / "minicpm3" / "serve.py", lambda t: t + "\n# edited\n")
    text = " ".join(gate(staged))
    assert "extra.txt: is in the package and not in" in text and "serve.py: differs from the hash" in text


def test_requirements_must_name_what_the_scripts_import(staged):
    edit(staged.pkg / "requirements.txt", lambda t: t.replace("safetensors", ""))
    assert any("safetensors" in p for p in gate(staged))


def test_a_link_in_the_package_is_refused(staged):
    (staged.pkg / "link").symlink_to(staged.pkg / "README.md")
    assert any("symbolic link" in p for p in gate(staged))


def test_evidence_that_cannot_back_the_card_is_refused(staged):
    (staged.pkg / "evidence" / "text_bench.json").write_text("{}")
    assert any("Numbers table" in p or "evidence cannot back the card" in p for p in gate(staged))
    (staged.pkg / "evidence" / "text_bench.json").write_text('{"variants": {"bf16": {}}}')
    assert any("evidence cannot back the card" in p for p in gate(staged))


def test_the_manifest_checker_hook_is_applied(staged):
    assert any("tt-model.yaml: broken" in p for p in gate(staged, validate_manifest=lambda path: ["broken"]))


# ---- the command ------------------------------------------------------------------------------------------------------------

def test_the_command_stages_checks_and_writes_the_commands_text(done, tmp_path, capsys):
    import io
    from types import SimpleNamespace
    lic = tmp_path / "L"
    lic.write_text("Apache License\nVersion 2.0\n")
    args = SimpleNamespace(run_dir=str(done.run_dir), namespace="example-org", revision=REV, license_file=str(lic), weights_terms=TERMS, context=None,
                           out=str(tmp_path / "o"), check=False)
    (tmp_path / "o").mkdir()
    buf = io.StringIO()
    assert tp.main(args, out=buf, hostname="examplehost") == 0 and "passes the gate" in buf.getvalue()
    assert (tmp_path / "o" / "TEXT_PUBLISH_COMMANDS.txt").read_text().count("hf upload") == 1
    args.check = True
    shutil.rmtree(tmp_path / "o" / M.package_name / "evidence")
    assert tp.main(args, out=io.StringIO(), hostname="examplehost") == 1


# ---- serve.py takes the MLP flags ---------------------------------------------------------------------------------------------

def test_serve_passes_the_mlp_flags_to_the_model(monkeypatch, tmp_path):
    seen = {}

    class Model:
        def __init__(self, dev, mdir, weights, **kw):
            seen.update(kw)

        def enable_trace(self):
            pass
    fake_ttnn = types.SimpleNamespace(open_device=lambda **k: object(), close_device=lambda d: None)
    monkeypatch.setitem(sys.modules, "ttnn", fake_ttnn)
    monkeypatch.setitem(sys.modules, "chip_model", types.SimpleNamespace(ChipMiniCPM3=Model))
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda d: object())))
    import importlib.util
    spec = importlib.util.spec_from_file_location("_t_serve2", M.scripts_dir / "serve.py")
    serve = importlib.util.module_from_spec(spec)
    monkeypatch.syspath_prepend(str(M.scripts_dir))
    spec.loader.exec_module(serve)

    class Srv:
        def __init__(self, *a, **k):
            pass

        def serve_forever(self):
            raise KeyboardInterrupt
    monkeypatch.setattr(serve, "ThreadingHTTPServer", Srv)
    monkeypatch.setattr(sys, "argv", ["serve.py", "--model-dir", str(tmp_path), "--weights", "w", "--mlp-dtype", "bfp8", "--mlp-fidelity", "lofi", "--batched-prefill"])
    with pytest.raises(KeyboardInterrupt):
        serve.main()
    assert seen["mlp_dtype"] == "bfp8" and seen["mlp_fidelity"] == "lofi" and serve.STATE["batched_prefill"] is True


def test_a_variant_that_missed_the_bar_is_labelled_so_on_the_card(done, tmp_path):
    res = json.loads((done.run_dir / "stages" / "2" / "text_result.json").read_text())
    bench = json.loads((done.run_dir / "stages" / "6" / "text_bench.json").read_text())
    res["variants"]["mlp8"]["ok"] = False
    rows = tp.number_rows(res, bench, None)
    by = {r[0]: r[2] for r in rows}
    assert by["teacher-forced next-token agreement, mlp8"] == "measured (the stage 2 bar was not met)"
    assert by["teacher-forced next-token agreement, bf16"] == "measured (provisional bar met)"
    assert by["perplexity on 767 tokens, mlp8"].endswith("not met)") and by["worst block error over the CPU bfloat16 error, mlp8"].endswith("not met)")


def test_the_card_says_when_a_variant_missed_the_bar(done, tmp_path):
    res = json.loads((done.run_dir / "stages" / "2" / "text_result.json").read_text())
    bench = json.loads((done.run_dir / "stages" / "6" / "text_bench.json").read_text())
    assert "did not meet the numeric bar" not in tp.render_card(M, res, bench, None, revision=REV, namespace="o", license_name="x", weights_terms="t")
    res["variants"]["mlp8"]["ok"] = False
    card = tp.render_card(M, res, bench, None, revision=REV, namespace="o", license_name="x", weights_terms="t")
    assert "The `mlp8` variant did not meet the stage 2 bar" in card and "`bf16` variant did not" not in card


# ---- the Copilot review of PR 247 ------------------------------------------------------------------------------------------------

def stage(done, tmp_path, name="out", **kw):
    lic = tmp_path / "L"
    lic.write_text("Apache License\nVersion 2.0\n")
    args = dict(rtm=M, namespace="o", revision=REV, license_file=lic, out=tmp_path / name, weights_terms=TERMS)
    args.update(kw)
    (tmp_path / name).mkdir(exist_ok=True)
    return tp.stage_package(done.run_dir, **args)


def test_the_launcher_loads_the_directory_it_checked_even_when_given_another(staged, tmp_path):
    import subprocess
    fake = tmp_path / "fakepython"
    fake.write_text("#!/bin/sh\nfor a in \"$@\"; do echo \"$a\"; done\n")
    fake.chmod(0o755)
    checked = tmp_path / "checked"
    checked.mkdir()
    shutil.copy(staged.run / "stages" / "1" / "text_config.json", tmp_path / "unused")
    (checked / "config.json").write_text(json.dumps({"x": 1}))
    rec = json.loads((staged.pkg / tp.RECORD_FILE).read_text())
    # the pinned hash is the fake run's; use a directory whose config has that hash
    snap = json.loads((staged.run / "stages" / "1" / "text_config.json").read_text())["model_dir"]
    env = {**__import__("os").environ, "MINICPM3_DIR": snap, "MINICPM3_WEIGHTS": "/w.safetensors", "ORCHARD_TEXT_PYTHON": str(fake)}
    out = subprocess.run(["bash", str(staged.pkg / "run.sh"), "--model-dir", "/elsewhere", "--weights", "/other"], env=env, capture_output=True, text=True)
    args = out.stdout.split()
    assert out.returncode == 0, out.stderr
    assert args[len(args) - 1 - args[::-1].index("--model-dir") + 1] == snap and args[len(args) - 1 - args[::-1].index("--weights") + 1] == "/w.safetensors"


def test_a_revision_that_is_not_the_runs_snapshot_is_refused(done, tmp_path):
    with pytest.raises(tp.TextPackageError, match="snapshot"):
        stage(done, tmp_path, revision="b" * 40)


def triage_without_snapshot(done):
    p = done.run_dir / "stages" / "0" / "evidence" / "triage.json"
    edit(p, lambda t: json.dumps({k: v for k, v in json.loads(t).items() if k != "snapshot"}))


def test_a_run_with_no_snapshot_revision_needs_the_hub_hash_of_its_configuration(done, tmp_path):
    triage_without_snapshot(done)
    with pytest.raises(tp.TextPackageError, match="cannot be verified"):
        stage(done, tmp_path)
    with pytest.raises(tp.TextPackageError, match="not the configuration"):
        stage(done, tmp_path, name="o2", config_sha256_at_revision="0" * 64)
    ref = json.loads((done.run_dir / "stages" / "1" / "text_reference.json").read_text())
    rec = stage(done, tmp_path, name="o3", config_sha256_at_revision=ref["config_sha256"])
    assert rec["revision_verified_by"] == "the hash of the configuration at that revision"


def test_a_context_record_for_two_chips_or_with_no_chip_count_is_refused(tmp_path):
    for body in ({"tp": 2, "windows": [{"start_position": 1, "steps": 1, "tok_s": 1.0}]}, {"windows": [{"start_position": 1, "steps": 1, "tok_s": 1.0}]}):
        f = tmp_path / "c.json"
        f.write_text(json.dumps(body))
        with pytest.raises(tp.TextPackageError, match="one-chip"):
            tp.read_context(f)


def test_results_edited_after_their_gate_passed_are_not_packaged(done, tmp_path):
    edit(done.run_dir / "stages" / "6" / "text_bench.json", lambda t: t.replace('"label": "benchmark"', '"label": "benchmark "', 1))
    with pytest.raises(tp.TextPackageError, match="text_bench.json"):
        stage(done, tmp_path)


def test_scripts_that_changed_since_the_run_are_not_packaged_but_the_server_may_differ(done, tmp_path):
    scripts = tmp_path / "scripts"
    shutil.copytree(M.scripts_dir, scripts)

    class Copy(type(M)):
        scripts_dir = scripts
    rtm = Copy()
    (scripts / "serve.py").write_text((scripts / "serve.py").read_text() + "\n# newer\n")
    stage(done, tmp_path, name="ok", rtm=rtm)                                    # serve.py is not run by the stages
    (scripts / "chip_model.py").write_text((scripts / "chip_model.py").read_text() + "\n# newer\n")
    with pytest.raises(tp.TextPackageError, match="chip_model.py"):
        stage(done, tmp_path, name="bad", rtm=rtm)


def test_the_card_gives_the_floor_in_tokens_per_second(staged):
    assert "% of" not in (staged.pkg / "README.md").read_text()
    row = next(l for l in (staged.pkg / "README.md").read_text().splitlines() if l.startswith("| share of the decode floor, bf16"))
    floor = json.loads((staged.pkg / "evidence" / "text_bench.json").read_text())["variants"]["bf16"]["floor"]["floor_tok_s"]
    assert "percent of" in row and f"{floor:.1f} tok/s" in row


def test_a_launcher_that_lets_a_forwarded_argument_override_the_checked_directory_is_refused(staged):
    edit(staged.pkg / "run.sh", lambda t: t.replace(' --model-dir "$MINICPM3_DIR" --weights "$MINICPM3_WEIGHTS"', ''))
    rec_path = staged.pkg / tp.RECORD_FILE
    rec = json.loads(rec_path.read_text())
    for f in rec["files"]:
        if f["path"] == "run.sh":
            f["sha256"] = tp.sha((staged.pkg / "run.sh").read_bytes())
    rec_path.write_text(json.dumps(rec))
    assert any("after the forwarded arguments" in p for p in gate(staged))


def test_the_card_shows_the_host_load_the_benchmark_ran_under(staged):
    card = (staged.pkg / "README.md").read_text()
    row = next(l for l in card.splitlines() if l.startswith("| decode speed, bf16"))
    assert "host load average 1.0 to 1.0 while it ran" in row
    bench = json.loads((staged.pkg / "evidence" / "text_bench.json").read_text())
    for v in bench["variants"].values():
        v["summary"].pop("host_loadavg_1m", None)
    res = json.loads((staged.pkg / "evidence" / "text_result.json").read_text())
    assert "host load not recorded" in next(r[1] for r in tp.number_rows(res, bench, None) if r[0] == "decode speed, bf16")


# ---- the second review of PR 247 ----------------------------------------------------------------------------------------------------------------

def test_the_card_does_not_say_which_check_a_variant_failed(done, tmp_path):
    res = json.loads((done.run_dir / "stages" / "2" / "text_result.json").read_text())
    bench = json.loads((done.run_dir / "stages" / "6" / "text_bench.json").read_text())
    res["variants"]["mlp8"]["ok"] = False
    card = tp.render_card(M, res, bench, None, revision=REV, namespace="o", license_name="x", weights_terms="t")
    sentence = next(l for l in card.splitlines() if "mlp8" in l and "did not meet" in l)
    assert "block" not in sentence and "text_result.json" in sentence


def test_the_latest_runtime_selection_in_the_ledger_decides_the_script_hashes(done, tmp_path):
    ledger = done.run_dir / "ledger.jsonl"
    lines = ledger.read_text().splitlines()
    i = next(n for n, l in enumerate(lines) if rt.TEXT_RUNTIME_SELECTED in l)
    stale = json.loads(lines[i])
    stale["data"]["scripts"] = {k: "0" * 64 for k in stale["data"]["scripts"]}
    lines.insert(i, json.dumps(stale))                 # a failed first attempt wrote another selection with other hashes
    ledger.write_text("\n".join(lines) + "\n")
    stage(done, tmp_path)                              # the later selection is the one that counts


def test_a_package_record_with_a_malformed_file_list_is_a_failed_gate_not_a_crash(staged):
    for bad in (None, [3], [{"path": "x"}], "text"):
        rec_path = staged.pkg / tp.RECORD_FILE
        rec = json.loads(rec_path.read_text())
        rec["files"] = bad
        rec_path.write_text(json.dumps(rec))
        assert any("file list" in p for p in gate(staged)), bad
