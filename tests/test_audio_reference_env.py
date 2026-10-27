# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard audio-reference-env` (orchard/audio_reference_env.py): the interpreter and source checkout the CPU reference script needs. Every
command runs through a fake runner that records it; nothing is installed and no network is used."""
import io
import json
from pathlib import Path

import pytest

from orchard import audio_reference_env as ae

PIN = ae.LIBRARY_COMMIT


class Fake:
    """A runner that records each command and answers from a table keyed by the command's first words."""

    def __init__(self, tmp_path, head=PIN, import_ok=True, cuda="False"):
        self.calls, self.tmp, self.head, self.import_ok, self.cuda = [], tmp_path, head, import_ok, cuda

    def __call__(self, argv, timeout, env=None):
        argv = [str(a) for a in argv]
        self.calls.append((argv, env))
        if argv[:2] == ["git", "init"]:
            Path(argv[-1]).mkdir(parents=True, exist_ok=True)
            (Path(argv[-1]) / ".git").mkdir(exist_ok=True)
            return 0, "", ""
        if "rev-parse" in argv:
            return 0, self.head + "\n", ""
        if argv[:2] == ["uv", "venv"]:
            py = Path(argv[-1]) / "bin" / "python"
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("")
            return 0, "", ""
        if argv and argv[0].endswith("bin/python"):
            code = argv[-1]
            if "stable_audio_3" in code:
                return (0, "ok\n", "") if self.import_ok else (1, "", "ModuleNotFoundError: No module named 'stable_audio_3'")
            if "torch.version" in code or "torch.cuda" in code:
                return 0, f"2.7.1+cpu {self.cuda}\n", ""
            if "importlib.metadata" in code:
                return 0, json.dumps({"torch": "2.7.1+cpu", "transformers": "5.19.0"}), ""
        return 0, "", ""


@pytest.fixture
def env(tmp_path):
    home = tmp_path / "home"
    (home / ".tenstorrent-venv").mkdir(parents=True)
    run = Fake(tmp_path)
    e = ae.env_from_args(type("A", (), {"venv_dir": None, "src_dir": None})(), environ={"HF_TOKEN": "secret", "UV_INDEX_URL": "x", "PATH": "/bin"},
                         home=home, repo_dir=tmp_path / "repo", run=run)
    return e


def statuses(results):
    return {r.name: r.status for r in results}


def test_the_defaults_are_under_orchard_venvs_and_orchard_src(env, tmp_path):
    assert env.venv_dir == (tmp_path / "home" / "orchard-venvs" / "stable-audio").resolve()
    assert env.src_dir == (tmp_path / "home" / "orchard-src" / "stable-audio-3").resolve()


def test_the_environment_variables_name_other_places(tmp_path):
    e = ae.env_from_args(type("A", (), {"venv_dir": None, "src_dir": None})(),
                         environ={"ORCHARD_AUDIO_VENV": str(tmp_path / "v"), "ORCHARD_AUDIO_SRC": str(tmp_path / "s")}, home=tmp_path / "h",
                         repo_dir=tmp_path / "repo")
    assert e.venv_dir == (tmp_path / "v").resolve() and e.src_dir == (tmp_path / "s").resolve()


def test_a_dry_run_changes_nothing_and_lists_what_it_would_do(env):
    results = ae.ensure(env, check_only=True)
    assert env.run.calls == [] or all(c[0][0] != "uv" or c[0][1] not in ("venv", "pip") for c in env.run.calls)
    assert not env.venv_dir.exists() and not env.src_dir.exists()
    assert statuses(results)["source"] == ae.TODO and statuses(results)["venv"] == ae.TODO


def test_a_build_fetches_the_pinned_commit_makes_the_venv_and_installs(env):
    results = ae.ensure(env, check_only=False)
    cmds = [" ".join(c[0]) for c in env.run.calls]
    assert any(c.startswith("git init") for c in cmds)
    assert any("fetch" in c and PIN in c and "--depth 1" in c for c in cmds)
    assert any("checkout" in c and PIN in c for c in cmds)
    assert any(c.startswith("uv venv") for c in cmds)
    assert statuses(results) == {"venv-dir": ae.OK, "source": ae.OK, "venv": ae.OK, "torch": ae.OK, "packages": ae.OK, "library": ae.OK,
                                 "import": ae.OK, "cpu-only": ae.OK, "record": ae.OK} or all(s == ae.OK for s in statuses(results).values())


def test_torch_comes_from_the_cpu_index_alone_and_nothing_else_does(env):
    ae.ensure(env, check_only=False)
    installs = [c[0] for c in env.run.calls if c[0][:3] == ["uv", "pip", "install"]]
    torch_calls = [a for a in installs if any(x.startswith("torch==") for x in a)]
    assert len(torch_calls) == 1 and ae.TORCH_CPU_INDEX in torch_calls[0] and ae.PYPI_INDEX not in torch_calls[0]
    others = [a for a in installs if a not in torch_calls]
    assert others and all(ae.TORCH_CPU_INDEX not in a for a in others)
    assert not any(x.startswith("torch") for a in others for x in a if x.startswith(("torch==", "torchaudio==")))


def test_the_library_is_installed_without_its_dependencies_from_the_checkout(env):
    ae.ensure(env, check_only=False)
    lib = [c[0] for c in env.run.calls if c[0][:3] == ["uv", "pip", "install"] and str(env.src_dir) in c[0]]
    assert len(lib) == 1 and "--no-deps" in lib[0] and "-e" in lib[0]


def test_uv_gets_no_token_and_no_extra_index_from_the_shell(env):
    ae.ensure(env, check_only=False)
    for argv, e in env.run.calls:
        if argv[0] == "uv" and e is not None:
            assert "HF_TOKEN" not in e and "UV_INDEX_URL" not in e and e.get("PATH") == "/bin"
    git_calls = [e for a, e in env.run.calls if a[0] == "git"]
    assert git_calls and all(e is not None and "HF_TOKEN" not in e for e in git_calls)


def test_the_shared_tt_venv_is_refused(env, tmp_path):
    env.venv_dir = (tmp_path / "home" / ".tenstorrent-venv").resolve()
    results = ae.ensure(env, check_only=False)
    assert results[0].status == ae.FAIL and "shared" in results[0].detail and env.run.calls == []


def test_a_venv_inside_the_shared_venv_or_the_checkout_is_refused(env, tmp_path):
    env.venv_dir = (tmp_path / "home" / ".tenstorrent-venv" / "sub").resolve()
    assert ae.ensure(env, check_only=False)[0].status == ae.FAIL
    env.venv_dir = (tmp_path / "repo" / "venv").resolve()
    assert ae.ensure(env, check_only=False)[0].status == ae.FAIL


def test_a_venv_dir_reached_through_a_link_is_judged_by_where_it_leads(env, tmp_path):
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "home" / ".tenstorrent-venv")
    env.venv_dir = (link / "x")
    env.__post_init__()
    assert ae.ensure(env, check_only=False)[0].status == ae.FAIL


def test_a_source_directory_inside_the_venv_or_the_venv_inside_it_is_refused(env):
    env.src_dir = env.venv_dir / "src"
    assert ae.ensure(env, check_only=False)[0].status == ae.FAIL
    env.src_dir = env.venv_dir.parent
    assert ae.ensure(env, check_only=False)[0].status == ae.FAIL


def test_a_checkout_at_another_commit_is_refused_and_never_reset(env):
    env.src_dir.mkdir(parents=True)
    (env.src_dir / ".git").mkdir()
    env.run.head = "0" * 40
    results = ae.ensure(env, check_only=False)
    assert statuses(results)["source"] == ae.FAIL and "0" * 12 in results[0].detail
    cmds = [" ".join(c[0]) for c in env.run.calls]
    assert not any(w in argv for argv, _ in env.run.calls for w in ("checkout", "reset", "fetch"))


def test_a_checkout_at_the_pin_is_left_alone(env):
    env.src_dir.mkdir(parents=True)
    (env.src_dir / ".git").mkdir()
    results = ae.ensure(env, check_only=False)
    assert statuses(results)["source"] == ae.OK
    assert not any(c[0][:2] == ["git", "init"] for c in env.run.calls)


def test_a_non_empty_directory_without_git_is_left_alone(env):
    env.src_dir.mkdir(parents=True)
    (env.src_dir / "file").write_text("x")
    results = ae.ensure(env, check_only=False)
    assert statuses(results)["source"] == ae.FAIL and not any(c[0][:2] == ["git", "init"] for c in env.run.calls)


def test_a_venv_directory_with_files_and_no_python_is_left_alone(env):
    env.venv_dir.mkdir(parents=True)
    (env.venv_dir / "x").write_text("")
    assert statuses(ae.ensure(env, check_only=False))["venv"] == ae.FAIL


def test_a_failed_step_stops_the_steps_after_it(env):
    real = env.run

    def failing(argv, timeout, e=None):
        if "fetch" in list(map(str, argv)):
            return 1, "", "network down"
        return real(argv, timeout, e)
    env.run = failing
    results = ae.ensure(env, check_only=False)
    assert statuses(results)["source"] == ae.FAIL and "venv" not in statuses(results)
    assert "network down" in results[-1].detail


def test_the_import_rule_fails_when_the_library_does_not_import(env):
    env.run.import_ok = False
    results = ae.ensure(env, check_only=False)
    assert statuses(results)["import"] == ae.FAIL and "stable_audio_3" in [r for r in results if r.name == "import"][0].detail


def test_a_torch_that_sees_cuda_or_is_not_the_cpu_build_fails_the_rule(env):
    env.run.cuda = "True"
    assert statuses(ae.ensure(env, check_only=False))["cpu-only"] == ae.FAIL


def test_check_mode_runs_no_install_and_reports_the_rules(env):
    ae.ensure(env, check_only=False)
    env.run.calls.clear()
    results = ae.ensure(env, check_only=True)
    assert all(s == ae.OK for s in statuses(results).values())
    assert not any(c[0][:2] in (["uv", "venv"], ["uv", "pip"], ["git", "fetch"], ["git", "init"]) for c in env.run.calls)


def test_the_record_names_the_commit_the_installed_versions_and_no_secret(env):
    ae.ensure(env, check_only=False)
    rec = json.loads((env.venv_dir / ae.RECORD).read_text())
    assert rec["library_commit"] == PIN and rec["installed"]["torch"] == "2.7.1+cpu" and rec["torch_pin"] == ae.TORCH
    assert "secret" not in json.dumps(rec)


def test_main_exit_codes(env):
    out = io.StringIO()
    args = type("A", (), {"check": False, "dry_run": True})()
    assert ae.main(args, env=env, out=out) == 0
    args = type("A", (), {"check": False, "dry_run": False})()
    assert ae.main(args, env=env, out=out) == 0
    env.run.import_ok = False
    assert ae.main(type("A", (), {"check": True, "dry_run": False})(), env=env, out=out) == 2


def test_the_pins_are_what_the_library_asks_for():
    text = " ".join(ae.TORCH_PINS + ae.REQUIREMENTS)
    for needle in ("torch==2.7.1", "torchaudio==2.7.1", "transformers", "huggingface_hub", "safetensors", "einops", "soundfile", "numpy"):
        assert needle in text


def test_the_pinned_commit_is_a_full_hash():
    assert len(ae.LIBRARY_COMMIT) == 40 and all(c in "0123456789abcdef" for c in ae.LIBRARY_COMMIT)
