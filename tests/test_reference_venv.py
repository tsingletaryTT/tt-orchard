# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Building a second reference interpreter for a model type the installed transformers lacks (scripts/reference_venv.py).

Agents may not install packages, so on release day the operator runs this script when `tt-orchard drop-check` says the release
needs a newer transformers. It builds a new venv beside the existing one and never touches an existing directory. Every command
is injected, so no test installs anything."""
import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "reference_venv.py"
spec = importlib.util.spec_from_file_location("reference_venv", SCRIPT)
rv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv)


class Recorder:
    def __init__(self, fail_at=None, outputs=None):
        self.calls, self.fail_at, self.outputs = [], fail_at, outputs or {}

    def __call__(self, argv):
        self.calls.append(argv)
        if self.fail_at is not None and len(self.calls) - 1 == self.fail_at:
            return 1, "", "boom"
        return 0, self.outputs.get(argv[-1] if argv else "", "5.23.0\n"), ""


@pytest.mark.parametrize("good", ["transformers==5.23.0", "transformers>=5.23", "transformers @ git+https://github.com/huggingface/transformers@0123456789abcdef0123456789abcdef01234567"])
def test_accepted_specs(good):
    assert rv.spec_problem(good) is None


@pytest.mark.parametrize("bad", ["", "torch==2.0", "transformers; rm -rf /", "transformers==5.23.0 --index-url http://x", "transformers @ git+https://evil.example/t",
                                 "transformers\x00", "-e .", "transformers @ file:///tmp/x"])
def test_refused_specs(bad):
    assert rv.spec_problem(bad)


def test_dry_run_prints_the_commands_and_runs_nothing(tmp_path, capsys):
    run = Recorder()
    code = rv.main(["transformers==5.23.0", "--venv-dir", str(tmp_path / "v"), "--dry-run", "--uv", "/usr/bin/uv"], run=run)
    out = capsys.readouterr().out
    assert code == 0 and run.calls == []
    assert "uv venv" in out and "--index-url https://download.pytorch.org/whl/cpu torch" in out and "transformers==5.23.0" in out


def test_build_runs_venv_torch_then_transformers_and_prints_the_config_line(tmp_path, capsys):
    run = Recorder()
    v = tmp_path / "v"
    code = rv.main(["transformers==5.23.0", "--venv-dir", str(v), "--uv", "/usr/bin/uv", "--arch", "Qwen4ForCausalLM"], run=run)
    out = capsys.readouterr().out
    assert code == 0
    assert run.calls[0][:3] == ["/usr/bin/uv", "venv", str(v)]
    assert "torch" in run.calls[1] and "https://download.pytorch.org/whl/cpu" in run.calls[1]
    assert "transformers==5.23.0" in run.calls[2]
    assert any("Qwen4ForCausalLM" in " ".join(c) for c in run.calls[3:])
    assert f'reference_python = "{v}/bin/python"' in out


def test_an_existing_directory_is_never_touched(tmp_path, capsys):
    v = tmp_path / "v"
    v.mkdir()
    run = Recorder()
    assert rv.main(["transformers==5.23.0", "--venv-dir", str(v), "--uv", "/usr/bin/uv"], run=run) == 2
    assert run.calls == []


def test_a_failed_step_stops_and_says_which(tmp_path, capsys):
    run = Recorder(fail_at=1)
    code = rv.main(["transformers==5.23.0", "--venv-dir", str(tmp_path / "v"), "--uv", "/usr/bin/uv"], run=run)
    assert code == 1 and len(run.calls) == 2
    assert "torch" in capsys.readouterr().err


def test_a_missing_class_fails_the_check(tmp_path, capsys):
    def run(argv):
        if any("getattr" in a for a in argv):
            return 1, "", "AttributeError"
        return 0, "5.23.0\n", ""
    code = rv.main(["transformers==5.23.0", "--venv-dir", str(tmp_path / "v"), "--uv", "/usr/bin/uv", "--arch", "Qwen4ForCausalLM"],
                   run=run)
    assert code == 1 and "Qwen4ForCausalLM" in capsys.readouterr().err


def test_the_default_directory_sits_beside_the_existing_reference_venv(tmp_path):
    d = rv.default_venv_dir("transformers==5.23.0", home=tmp_path)
    assert d.parent == tmp_path / ".local" / "share" / "tt-orchard" / "venvs" and d.name.startswith("reference-")
    assert d.name != "reference"
