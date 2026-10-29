"""The Granite package's staging and gate (orchard/granite_package.py), with a made-up summary: no chip, no weights, no network."""
import json
import shutil
from pathlib import Path

import pytest

from orchard import granite_package as gp

SUMMARY = {
    "floor_tok_s": 175.2, "decode": {"mean": 34.0, "min": 33.0, "max": 35.0}, "decode_label": "benchmark: 5 repeats after a warm-up",
    "decode_untraced": {"mean": 20.0, "min": 19.0, "max": 21.0}, "decode_untraced_label": "benchmark: 5 repeats after a warm-up", "context": 2048,
    "first_token_s_12": 0.3, "prefill": [{"n": 128, "tok_s": 40.0, "label": "median of 3"}], "context_sweep": [{"max_len": 512, "mean": 35.0, "label": "5 repeats"}],
    "boot": {"first_token_s": 0.4, "decode_tok_s": 33.0, "label": "one request"}, "numbers_note": "Test note.",
    "agreement": {"floor": {"rel_l2": [0.0047, 0.0121, 0.0043, 0.0111], "top1": [1.0, 1.0, 0.875, 1.0], "prefix": [3, 24, 8, 24]},
                  "chip": {"rel_l2": [0.0066, 0.018, 0.0076, 0.0184], "top1": [1.0, 1.0, 0.917, 1.0], "prefix": [3, 24, 13, 24]}},
    "agreement_note": "Four prompts only.", "long": {"tokens": 2048, "chip": {"rel_l2_mean": 0.0134, "rel_l2_max": 0.16, "top1": 0.978}, "floor": {"rel_l2_mean": 0.0084, "rel_l2_max": 0.097, "top1": 0.973}},
    "long_note": "One sequence.", "prefill_note": "Prefill is a loop of decode steps.", "limitations": ["Limit one.", "Limit two."], "not_tested": ["Not tested one."],
    "kernel_modes": {"text": "Optional kernel modes exist; see the run log."},
}


@pytest.fixture()
def staged(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps(SUMMARY))
    ev = tmp_path / "ev"
    ev.mkdir()
    (ev / "bench.json").write_text('{"x": 1}')
    out = tmp_path / "pkg"
    gp.stage(summary, ev, out)
    return out


def problems(pkg):
    return gp.package_problems(pkg, hostname="no-such-host", home="/home/nobody")


def test_staged_package_passes_the_gate(staged):
    assert problems(staged) == []
    assert (staged / "run.sh").stat().st_mode & 0o111
    assert not list(staged.rglob("*.safetensors"))


def test_refuses_to_stage_into_a_used_directory(tmp_path, staged):
    with pytest.raises(SystemExit):
        gp.stage(staged / "evidence" / "summary.json", tmp_path / "ev", staged)


@pytest.mark.parametrize("name,mutate,fragment", [
    ("weights", lambda p: (p / "runtime" / "w.safetensors").write_bytes(b"x"), "weights"),
    ("big file", lambda p: (p / "runtime" / "blob.txt").write_bytes(b"x" * (21 * 1024 * 1024)), "bytes"),
    ("home path", lambda p: (p / "run.sh").write_text((p / "run.sh").read_text() + "\n# /home/someone/x\n"), "home path"),
    ("key file", lambda p: (p / "run.sh").write_text((p / "run.sh").read_text() + "\n# reads ~/.tt_ai_key\n"), ".tt_ai_key"),
    ("litellm", lambda p: (p / "server" / "serve.py").write_text((p / "server" / "serve.py").read_text() + "\n# litellm\n"), "litellm"),
    ("license edited", lambda p: (p / "LICENSE").write_text("MIT"), "LICENSE"),
    ("card number edited", lambda p: (p / "README.md").write_text((p / "README.md").read_text().replace("34.0 (33.0 to 35.0)", "44.0 (33.0 to 35.0)")), "number row"),
    ("card section removed", lambda p: (p / "README.md").write_text((p / "README.md").read_text().replace("## Not tested", "## Other")), "Not tested"),
    ("card overclaim", lambda p: (p / "README.md").write_text((p / "README.md").read_text() + "\nThis is production-ready.\n"), "overclaim"),
    ("wrong license front matter", lambda p: (p / "README.md").write_text((p / "README.md").read_text().replace("license: apache-2.0", "license: mit")), "apache-2.0"),
    ("manifest kind", lambda p: (p / "tt-model.yaml").write_text((p / "tt-model.yaml").read_text().replace("kind: http-server", "kind: vllm-plugin")), "kind"),
    ("manifest missing script", lambda p: (p / "tt-model.yaml").write_text((p / "tt-model.yaml").read_text().replace("server/serve.py", "server/gone.py")), "gone.py"),
    ("bad python", lambda p: (p / "runtime" / "granite_chip.py").write_text("def ("), "does not compile"),
    ("bad shell", lambda p: (p / "run.sh").write_text("if then\n"), "bash -n"),
    ("missing file", lambda p: (p / "NOTICE").unlink(), "NOTICE"),
    ("kernel changed", lambda p: (p / "runtime" / "kernels" / "state.py").write_text("# changed\n"), "changed after"),
])
def test_gate_catches(staged, name, mutate, fragment):
    mutate(staged)
    out = problems(staged)
    assert any(fragment in x for x in out), (name, out)


def test_link_is_refused(staged):
    (staged / "link").symlink_to(staged / "README.md")
    assert any("symbolic link" in x for x in problems(staged))


def test_stage_copies_the_kernels_unchanged(staged):
    for src, dst in gp.KERNEL_FILES:
        assert (staged / dst).read_bytes() == (gp.ROOT / src).read_bytes()


# ---- review fixes (PR 244) ------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("root", ["", "- a\n- b\n", "just a string\n", "42\n"])
def test_manifest_whose_root_is_not_a_mapping_is_a_named_problem(staged, root):
    (staged / "tt-model.yaml").write_text(root)
    out = problems(staged)
    assert any("tt-model.yaml" in x and "mapping" in x for x in out), out


def test_manifest_ready_line_the_server_never_prints_fails_the_gate(staged):
    p = staged / "tt-model.yaml"
    p.write_text(p.read_text().replace("""ready_line: '"event": "ready"'""", "ready_line: NEVER_READY"))
    assert "NEVER_READY" in p.read_text()
    assert any("ready line" in x for x in problems(staged))


def test_the_good_manifest_ready_line_passes(staged):
    assert not any("ready line" in x for x in problems(staged))


def test_boot_script_does_not_hang_or_leave_the_server_when_it_never_becomes_ready(tmp_path):
    """boot.sh must stop the process it launched even when the ready line never appears (the pid in the ready line does not exist then)."""
    import subprocess
    import time
    pkg, out, venv = tmp_path / "pkg", tmp_path / "out", tmp_path / "venv"
    (pkg / "tools").mkdir(parents=True)
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text("")
    (pkg / "run.sh").write_text("echo $$ > %s/server.pid\nexec sleep 300\n" % tmp_path)
    t0 = time.time()
    r = subprocess.run(["bash", str(gp.ROOT / "docs/run-logs/evidence/granite-package/scripts/boot.sh"), str(pkg), str(out)],
                       env={**__import__("os").environ, "TT_METAL_HOME": "/nonexistent", "GRANITE_VENV": str(venv), "READY_TRIES": "2", "READY_SLEEP": "1"}, capture_output=True, text=True, timeout=60)
    pid = int((tmp_path / "server.pid").read_text())
    time.sleep(0.5)
    alive = Path(f"/proc/{pid}").exists() and "sleep" in (Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="ignore") if Path(f"/proc/{pid}/cmdline").exists() else "")
    assert not alive, "the launched server is still running"
    assert time.time() - t0 < 40 and "did not become ready" in r.stdout


def test_the_ready_line_and_the_card_report_the_effective_precision():
    import importlib.util
    import sys
    import types
    from unittest import mock
    sys.modules.setdefault("ttnn", mock.MagicMock())
    root = gp.ROOT
    s1 = importlib.util.spec_from_file_location("granite_chip_t", root / "pkgsrc" / "runtime" / "granite_chip.py")
    chip = importlib.util.module_from_spec(s1)
    s1.loader.exec_module(chip)
    s2 = importlib.util.spec_from_file_location("granite_serve_t", root / "pkgsrc" / "server" / "serve.py")
    srv = importlib.util.module_from_spec(s2)
    s2.loader.exec_module(srv)
    eff = chip.GraniteChip.effective_precision
    assert eff("composed", "fp32") == "fp32" and eff("composed", "bf16") == "bf16"
    assert eff("state_full_ro", "fp32") == "bf16" and eff("bcast", "fp32") == "bf16"      # the kernel modes run in bf16 whatever was asked
    fake = types.SimpleNamespace(f32=False, L=2048)
    ev = srv.ready_event(fake, types.SimpleNamespace(host="127.0.0.1", port=1, ssd="state_full_ro", ssd_precision="fp32", no_trace=False), 1.0)
    assert ev["ssd_precision"] == "bf16"                              # the constructed model's precision, not the requested one
    card = gp.render_card(SUMMARY, "")
    assert "kept in float32" in card and eff("composed", "fp32") == "fp32"
    assert "run with a bf16 state whatever `--ssd-precision` says" in card and eff("state_full_ro", "fp32") == "bf16"
    assert 'res["ssd_precision"] = "fp32" if m.f32 else "bf16"' in (root / "pkgsrc" / "tools" / "evaluate.py").read_text()
