# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard demo-env` on the command line (issue 92)."""
from pathlib import Path

from orchard import cli
from orchard import demo_env as de
from test_cli import Pipe


def run(argv, env=None, monkeypatch=None, results=None):
    seen = {}

    def fake_ensure(denv, *, check_only, say=lambda s: None):
        seen["env"], seen["check_only"] = denv, check_only
        for r in results or []:
            say(f"{r.status:5s} {r.name}: {r.detail}")
        return results or []
    monkeypatch.setattr(de, "ensure", fake_ensure)
    out = Pipe()
    code = cli.main(argv, env=env or {}, stdout=out)
    return code, out.getvalue() if hasattr(out, "getvalue") else str(out), seen


def test_the_subcommand_exists_with_its_four_options():
    parser = cli._parser()
    args = parser.parse_args(["demo-env", "--check", "--dry-run", "--venv-dir", "/v", "--tt-metal-dir", "/t"])
    assert (args.cmd, args.check, args.dry_run, args.venv_dir, args.tt_metal_dir) == ("demo-env", True, True, "/v", "/t")
    plain = parser.parse_args(["demo-env"])
    assert plain.check is False and plain.dry_run is False and plain.venv_dir is None


def test_it_needs_no_bringup_config_and_builds_under_the_home_by_default(tmp_path, monkeypatch):
    code, text, seen = run(["demo-env"], env={"HOME": str(tmp_path)}, monkeypatch=monkeypatch, results=[de.Result("venv", de.OK, "x")])
    assert code == 0 and "venv" in text
    assert seen["env"].venv_dir == tmp_path / "orchard-venvs" / "tt-demo" and seen["check_only"] is False
    assert seen["env"].shared_venv == tmp_path / ".tenstorrent-venv" and seen["env"].tt_metal_dir == tmp_path / "tt-metal"


def test_check_and_dry_run_change_nothing_and_the_exit_follows_the_results(tmp_path, monkeypatch):
    bad = [de.Result("numpy", de.FAIL, "numpy 2.5.3")]
    code, _, seen = run(["demo-env", "--check"], env={"HOME": str(tmp_path)}, monkeypatch=monkeypatch, results=bad)
    assert code == 2 and seen["check_only"] is True
    code, _, seen = run(["demo-env", "--dry-run"], env={"HOME": str(tmp_path)}, monkeypatch=monkeypatch,
                        results=[de.Result("venv", de.TODO, "would run: uv venv")])
    assert code == 0 and seen["check_only"] is True


def test_the_environment_variables_and_options_pick_the_directories(tmp_path, monkeypatch):
    env = {"HOME": str(tmp_path), "ORCHARD_DEMO_VENV": "/x/venv", "ORCHARD_TT_METAL_DIR": "/x/tree"}
    _, _, seen = run(["demo-env", "--check"], env=env, monkeypatch=monkeypatch, results=[])
    assert seen["env"].venv_dir == Path("/x/venv") and seen["env"].tt_metal_dir == Path("/x/tree")
    _, _, seen = run(["demo-env", "--check", "--venv-dir", "/y/v"], env=env, monkeypatch=monkeypatch, results=[])
    assert seen["env"].venv_dir == Path("/y/v")


def test_the_checkout_is_the_repository_the_command_refuses_to_build_in(tmp_path, monkeypatch):
    _, _, seen = run(["demo-env", "--check"], env={"HOME": str(tmp_path)}, monkeypatch=monkeypatch, results=[])
    assert seen["env"].repo_dir == cli.CHECKOUT
