# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard audio-reference-env` on the command line."""
from orchard import audio_reference_env as ae
from orchard import cli
from test_cli import Pipe


def run(argv, env, monkeypatch, results):
    seen = {}

    def fake_main(args, *, env=None, out=None):
        seen["env"], seen["args"] = env, args
        return 0 if all(r.status == ae.OK for r in results) else 2
    monkeypatch.setattr(ae, "main", fake_main)
    out = Pipe()
    return cli.main(argv, env=env, stdout=out), seen


def test_the_subcommand_has_its_four_options():
    parser = cli._parser()
    args = parser.parse_args(["audio-reference-env", "--check", "--dry-run", "--venv-dir", "/v", "--src-dir", "/s"])
    assert (args.cmd, args.check, args.dry_run, args.venv_dir, args.src_dir) == ("audio-reference-env", True, True, "/v", "/s")
    plain = parser.parse_args(["audio-reference-env"])
    assert plain.check is False and plain.dry_run is False and plain.venv_dir is None and plain.src_dir is None


def test_it_needs_no_bringup_config_and_uses_the_home_by_default(tmp_path, monkeypatch):
    code, seen = run(["audio-reference-env"], {"HOME": str(tmp_path)}, monkeypatch, [ae.Result("venv", ae.OK, "x")])
    assert code == 0
    assert seen["env"].venv_dir == tmp_path / "orchard-venvs" / "stable-audio" and seen["env"].src_dir == tmp_path / "orchard-src" / "stable-audio-3"
    assert seen["env"].shared_venv == tmp_path / ".tenstorrent-venv"


def test_the_exit_code_follows_the_result(tmp_path, monkeypatch):
    code, _ = run(["audio-reference-env", "--check"], {"HOME": str(tmp_path)}, monkeypatch, [ae.Result("import", ae.FAIL, "x")])
    assert code == 2
