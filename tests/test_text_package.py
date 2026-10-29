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
REV = "d" * 40
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
    edit(p, lambda t: t.replace('"ok": true', '"ok": false'))
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
    monkeypatch.setattr(sys, "argv", ["serve.py", "--model-dir", str(tmp_path), "--weights", "w", "--mlp-dtype", "bfp8", "--mlp-fidelity", "lofi"])
    with pytest.raises(KeyboardInterrupt):
        serve.main()
    assert seen["mlp_dtype"] == "bfp8" and seen["mlp_fidelity"] == "lofi"
