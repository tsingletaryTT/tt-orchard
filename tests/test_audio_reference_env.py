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

    def __init__(self, tmp_path, head=PIN, import_ok=True, cuda="False", dirty=""):
        self.calls, self.tmp, self.head, self.import_ok, self.cuda = [], tmp_path, head, import_ok, cuda
        self.dirty = dirty

    def __call__(self, argv, timeout, env=None):
        argv = [str(a) for a in argv]
        self.calls.append((argv, env))
        if argv[:2] == ["git", "init"]:
            Path(argv[-1]).mkdir(parents=True, exist_ok=True)
            (Path(argv[-1]) / ".git").mkdir(exist_ok=True)
            return 0, "", ""
        if "rev-parse" in argv:
            return 0, self.head + "\n", ""
        if "status" in argv:
            return 0, self.dirty, ""
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


# ---- guards found by the first mutation table -----------------------------------------------------

def fail_on(env, word):
    real = env.run

    def run(argv, timeout, e=None):
        if word in list(map(str, argv)):
            return 1, "", f"{word} broke"
        return real(argv, timeout, e)
    env.run = run
    env.real = real
    return real


def test_git_never_prompts_for_credentials(env):
    ae.ensure(env, check_only=False)
    git_envs = [e for a, e in env.run.calls if a[0] == "git"]
    assert git_envs and all(e["GIT_TERMINAL_PROMPT"] == "0" for e in git_envs)


def test_a_source_directory_inside_the_shared_venv_or_the_checkout_is_refused(env, tmp_path):
    env.src_dir = (tmp_path / "home" / ".tenstorrent-venv" / "src").resolve()
    assert ae.ensure(env, check_only=False)[0].status == ae.FAIL
    env.src_dir = (tmp_path / "repo" / "src").resolve()
    assert ae.ensure(env, check_only=False)[0].status == ae.FAIL


def test_a_venv_directory_inside_the_checkout_alone_is_refused(env, tmp_path):
    env.venv_dir = (tmp_path / "repo" / "v").resolve()
    assert ae.ensure(env, check_only=False)[0].status == ae.FAIL


def test_a_checkout_whose_head_cannot_be_read_fails_the_source_step(env):
    env.src_dir.mkdir(parents=True)
    (env.src_dir / ".git").mkdir()
    real = env.run

    def run(argv, timeout, e=None):
        if "rev-parse" in list(map(str, argv)):
            return 128, "", "fatal: not a git repository"
        return real(argv, timeout, e)
    env.run = run
    results = ae.ensure(env, check_only=False)
    assert results[0].status == ae.FAIL and "not a git repository" in results[0].detail


def test_the_venv_is_made_with_python_3_12(env):
    ae.ensure(env, check_only=False)
    [argv] = [a for a, _ in env.run.calls if a[:2] == ["uv", "venv"]]
    assert argv[argv.index("--python") + 1] == "3.12"


def test_a_failed_torch_install_stops_the_later_installs_and_the_record(env):
    fail_on(env, ae.TORCH_PINS[0])
    results = ae.ensure(env, check_only=False)
    assert statuses(results)["torch"] == ae.FAIL and "packages" not in statuses(results) and "library" not in statuses(results)
    assert "torch==2.7.1 broke" in [r for r in results if r.name == "torch"][0].detail
    assert not (env.venv_dir / ae.RECORD).exists()


def test_a_failed_package_install_stops_the_library_install_and_the_record(env):
    fail_on(env, ae.REQUIREMENTS[0])
    results = ae.ensure(env, check_only=False)
    assert statuses(results)["packages"] == ae.FAIL and "library" not in statuses(results)
    assert not (env.venv_dir / ae.RECORD).exists()


def test_a_failed_library_install_writes_no_record(env):
    fail_on(env, "--no-deps")
    results = ae.ensure(env, check_only=False)
    assert statuses(results)["library"] == ae.FAIL and statuses(results)["record"] == ae.FAIL
    assert not (env.venv_dir / ae.RECORD).exists()


def test_a_failed_rule_writes_no_record(env):
    env.run.import_ok = False
    results = ae.ensure(env, check_only=False)
    assert statuses(results)["record"] == ae.FAIL and not (env.venv_dir / ae.RECORD).exists()


def test_unreadable_installed_versions_fail_the_record(env):
    real = env.run

    def run(argv, timeout, e=None):
        if "importlib.metadata" in str(argv[-1]):
            return 0, "not json", ""
        return real(argv, timeout, e)
    env.run = run
    results = ae.ensure(env, check_only=False)
    assert statuses(results)["record"] == ae.FAIL and not (env.venv_dir / ae.RECORD).exists()


@pytest.mark.parametrize("version,cuda", [("2.7.1", "False"), ("2.8.0+cpu", "False"), ("2.7.1+cpu", "True")])
def test_only_the_pinned_cpu_build_without_cuda_passes(env, version, cuda):
    real = env.run

    def run(argv, timeout, e=None):
        if "torch.cuda" in str(argv[-1]):
            return 0, f"{version} {cuda}\n", ""
        return real(argv, timeout, e)
    env.run = run
    assert statuses(ae.ensure(env, check_only=False))["cpu-only"] == ae.FAIL


def test_extra_words_in_the_torch_answer_fail_the_rule(env):
    real = env.run

    def run(argv, timeout, e=None):
        if "torch.cuda" in str(argv[-1]):
            return 0, "2.7.1+cpu False extra\n", ""
        return real(argv, timeout, e)
    env.run = run
    assert statuses(ae.ensure(env, check_only=False))["cpu-only"] == ae.FAIL


def test_a_record_for_other_pins_or_another_commit_is_stale_in_check_mode(env):
    ae.ensure(env, check_only=False)
    path = env.venv_dir / ae.RECORD
    good = json.loads(path.read_text())
    for change in ({"requirements_sha256": "0" * 64}, {"library_commit": "1" * 40}):
        path.write_text(json.dumps({**good, **change}))
        results = ae.ensure(env, check_only=True)
        assert statuses(results)["record"] == ae.TODO and statuses(results)["packages"] == ae.TODO


def test_check_mode_with_no_record_is_not_ok(env):
    ae.ensure(env, check_only=False)
    (env.venv_dir / ae.RECORD).unlink()
    assert statuses(ae.ensure(env, check_only=True))["record"] == ae.TODO


def test_the_pin_hash_depends_on_the_commit(monkeypatch):
    before = ae.requirements_sha256()
    monkeypatch.setattr(ae, "LIBRARY_COMMIT", "2" * 40)
    assert ae.requirements_sha256() != before


def test_a_dry_run_asks_nothing_of_the_venv_interpreter(env):
    ae.ensure(env, check_only=True)
    assert not [a for a, _ in env.run.calls if a[0].endswith("bin/python")]


def test_main_with_dry_run_builds_nothing_and_exits_0_and_with_check_exits_2_on_a_missing_build(env):
    out = io.StringIO()
    assert ae.main(type("A", (), {"check": False, "dry_run": True})(), env=env, out=out) == 0
    assert not [a for a, _ in env.run.calls if a[:2] in (["uv", "venv"], ["uv", "pip"])] and not env.venv_dir.exists()
    assert ae.main(type("A", (), {"check": True, "dry_run": False})(), env=env, out=out) == 2


# ---- a checkout with local changes is not the pinned source (review of PR 123) ----------------------

@pytest.mark.parametrize("status", [" M stable_audio_3/model.py\n", "?? stable_audio_3/extra.py\n", "A  new.py\n D old.py\n"])
def test_a_checkout_with_tracked_edits_or_untracked_files_is_refused(env, status):
    env.src_dir.mkdir(parents=True)
    (env.src_dir / ".git").mkdir()
    env.run.dirty = status
    results = ae.ensure(env, check_only=False)
    assert statuses(results)["source"] == ae.FAIL and "local changes" in results[0].detail
    assert "venv" not in statuses(results)
    assert not any(a[:3] == ["uv", "pip", "install"] for a, _ in env.run.calls)


def test_the_status_check_lists_untracked_files_and_does_not_prompt(env):
    env.src_dir.mkdir(parents=True)
    (env.src_dir / ".git").mkdir()
    ae.ensure(env, check_only=True)
    [(argv, e)] = [c for c in env.run.calls if "status" in c[0]]
    assert "--porcelain" in argv and "--untracked-files=all" in argv and e["GIT_TERMINAL_PROMPT"] == "0"


def test_a_failed_status_command_fails_the_source_step(env):
    env.src_dir.mkdir(parents=True)
    (env.src_dir / ".git").mkdir()
    real = env.run

    def run(argv, timeout, e=None):
        if "status" in list(map(str, argv)):
            return 128, "", "fatal: bad index"
        return real(argv, timeout, e)
    env.run = run
    results = ae.ensure(env, check_only=False)
    assert results[0].status == ae.FAIL and "bad index" in results[0].detail


def test_a_clean_checkout_at_the_pin_is_accepted_in_check_mode_too(env):
    env.src_dir.mkdir(parents=True)
    (env.src_dir / ".git").mkdir()
    results = ae.ensure(env, check_only=True)
    assert statuses(results)["source"] == ae.OK and results[0].detail.endswith(", clean")
    env.run.dirty = "?? x\n"
    assert statuses(ae.ensure(env, check_only=True))["source"] == ae.FAIL
