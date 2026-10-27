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
        self.module_file = None          # where the fake interpreter says stable_audio_3 was imported from; the env fixture sets it

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
                return (0, (self.module_file or "") + "\n", "") if self.import_ok else (1, "", "ModuleNotFoundError: No module named 'stable_audio_3'")
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
    run.module_file = str(e.src_dir / "stable_audio_3" / "__init__.py")
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


# ---- round three: the imported library is the checked source; git and uv environments; a second build ---

def test_the_library_must_be_imported_from_the_checked_source_tree(env, tmp_path):
    other = tmp_path / "other-checkout" / "stable_audio_3" / "__init__.py"
    other.parent.mkdir(parents=True)
    other.write_text("")
    env.run.module_file = str(other)
    results = ae.ensure(env, check_only=False)
    bad = [r for r in results if r.name == "import"][0]
    assert bad.status == ae.FAIL and str(other.parent.parent) in bad.detail and str(env.src_dir) in bad.detail
    assert not (env.venv_dir / ae.RECORD).exists()


def test_check_mode_against_another_clean_checkout_fails_when_the_install_still_points_at_the_first(env, tmp_path):
    ae.ensure(env, check_only=False)
    other = (tmp_path / "second" / "stable-audio-3").resolve()
    other.mkdir(parents=True)
    (other / ".git").mkdir()
    env.src_dir = other
    results = ae.ensure(env, check_only=True)
    assert statuses(results)["source"] == ae.OK and statuses(results)["import"] == ae.FAIL


@pytest.mark.parametrize("answer", ["", "not a path\n", "relative/stable_audio_3/__init__.py\n", "None\n"])
def test_an_import_answer_that_is_not_an_absolute_path_fails(env, answer):
    env.run.module_file = answer.strip()
    real = env.run

    def run(argv, timeout, e=None):
        if argv[-1].startswith("import stable_audio_3"):
            return 0, answer, ""
        return real(argv, timeout, e)
    env.run = run
    assert statuses(ae.ensure(env, check_only=False))["import"] == ae.FAIL


def test_a_module_reached_through_a_link_into_the_source_tree_counts_as_inside(env, tmp_path):
    ae.ensure(env, check_only=False)
    (env.src_dir / "stable_audio_3").mkdir(parents=True, exist_ok=True)
    (env.src_dir / "stable_audio_3" / "__init__.py").write_text("")
    link = tmp_path / "link"
    link.symlink_to(env.src_dir)
    env.run.module_file = str(link / "stable_audio_3" / "__init__.py")
    assert statuses(ae.ensure(env, check_only=True))["import"] == ae.OK


def test_a_module_in_a_sibling_directory_with_the_same_prefix_is_not_inside(env, tmp_path):
    sibling = Path(str(env.src_dir) + "-old") / "stable_audio_3" / "__init__.py"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("")
    env.run.module_file = str(sibling)
    assert statuses(ae.ensure(env, check_only=False))["import"] == ae.FAIL


def test_the_import_rule_reads_the_module_file_and_not_a_marker_word(env):
    ae.ensure(env, check_only=False)
    [(argv, _)] = [c for c in env.run.calls if c[0][-1].startswith("import stable_audio_3")][:1]
    assert "__file__" in argv[-1] and argv[1] == "-I"


@pytest.mark.parametrize("name", ["GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_CONFIG_GLOBAL", "GIT_SSH_COMMAND", "GIT_EXEC_PATH",
                                  "UV_PYTHON", "UV_PROJECT_ENVIRONMENT", "UV_INDEX_URL", "PIP_INDEX_URL", "HF_HOME", "HUGGING_FACE_HUB_TOKEN",
                                  "GITHUB_TOKEN", "GH_TOKEN", "PYTHONPATH", "PYTHONHOME"])
def test_the_environment_for_git_and_uv_drops_variables_that_redirect_or_authenticate_them(tmp_path, name):
    e = ae.AEnv(tmp_path / "v", tmp_path / "s", tmp_path / "shared", tmp_path / "repo", environ={name: "x", "PATH": "/bin", "HOME": "/h"})
    cleaned = ae._clean(e)
    assert name not in cleaned and cleaned["PATH"] == "/bin" and cleaned["HOME"] == "/h" and cleaned["GIT_TERMINAL_PROMPT"] == "0"


def test_a_git_entry_that_is_a_file_or_a_link_is_not_a_checkout_we_trust(env, tmp_path):
    env.src_dir.mkdir(parents=True)
    (env.src_dir / ".git").write_text("gitdir: /somewhere/else\n")
    results = ae.ensure(env, check_only=False)
    assert statuses(results)["source"] == ae.FAIL and "directory" in results[0].detail
    assert not any("rev-parse" in a for a, _ in env.run.calls)
    (env.src_dir / ".git").unlink()
    (tmp_path / "elsewhere").mkdir()
    (env.src_dir / ".git").symlink_to(tmp_path / "elsewhere")
    assert statuses(ae.ensure(env, check_only=False))["source"] == ae.FAIL


def test_a_second_build_while_one_is_running_is_refused_and_runs_nothing(env):
    import fcntl
    env.venv_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(ae.lock_path(env), "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        results = ae.ensure(env, check_only=False)
    assert results[0].status == ae.FAIL and "already running" in results[0].detail and env.run.calls == []


def test_the_lock_is_released_after_a_build_and_after_a_failed_one(env):
    ae.ensure(env, check_only=False)
    env.run.import_ok = False
    ae.ensure(env, check_only=False)
    import fcntl
    with open(ae.lock_path(env), "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_check_and_dry_run_do_not_take_the_lock(env):
    import fcntl
    env.venv_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(ae.lock_path(env), "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        assert ae.ensure(env, check_only=True)[0].name != "lock"


def import_answer(env, answer, rc=0, err=""):
    real = env.run

    def run(argv, timeout, e=None):
        if argv[-1].startswith("import stable_audio_3"):
            return rc, answer, err
        return real(argv, timeout, e)
    env.run = run
    return [r for r in ae.ensure(env, check_only=False) if r.name == "import"][0]


def test_each_way_the_import_rule_can_fail_says_why(env):
    assert "failed" in import_answer(env, "", rc=1, err="ModuleNotFoundError").detail
    assert "did not say where" in import_answer(env, "", rc=0).detail
    assert "did not say where" in import_answer(env, "relative/x.py\n").detail
    assert "did not say where" in import_answer(env, "\n\n").detail


def test_the_import_rule_reads_the_last_line_of_the_answer(env):
    module = env.run.module_file
    ok = import_answer(env, "a warning from the library\n" + module + "\n")
    assert ok.status == ae.OK
    bad = import_answer(env, module + "\nlater noise\n")
    assert bad.status == ae.FAIL


def test_each_venv_directory_has_its_own_lock_beside_it(env, tmp_path):
    assert ae.lock_path(env) == env.venv_dir.parent / (env.venv_dir.name + ".lock")
    other = ae.AEnv(tmp_path / "x" / "other-venv", env.src_dir, env.shared_venv, env.repo_dir)
    assert ae.lock_path(other) != ae.lock_path(env)


# ---- round four: the source directory is locked too, in a fixed order -------------------------------------

def test_the_source_directory_has_its_own_lock_beside_it(env, tmp_path):
    assert ae.src_lock_path(env) == env.src_dir.parent / (env.src_dir.name + ".lock")
    assert ae.src_lock_path(env) != ae.lock_path(env)


def test_a_build_with_another_venv_but_the_same_source_directory_is_refused_while_the_source_is_locked(env, tmp_path):
    import fcntl
    env.src_dir.parent.mkdir(parents=True, exist_ok=True)
    with open(ae.src_lock_path(env), "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        env.venv_dir = (tmp_path / "other-home" / "venv").resolve()
        results = ae.ensure(env, check_only=False)
    assert results[0].status == ae.FAIL and "already running" in results[0].detail and str(ae.src_lock_path(env)) in results[0].detail
    assert env.run.calls == []


def test_a_build_that_cannot_take_the_second_lock_gives_back_the_first(env):
    import fcntl
    first, second = ae.lock_paths(env)
    second.parent.mkdir(parents=True, exist_ok=True)
    with open(second, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        results = ae.ensure(env, check_only=False)
    assert results[0].status == ae.FAIL and str(second) in results[0].detail
    with open(first, "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_both_locks_are_held_during_the_build_and_released_after(env):
    import fcntl
    seen = {}
    real = env.run

    def run(argv, timeout, e=None):
        if "locked" not in seen:
            for name, path in (("venv", ae.lock_path(env)), ("src", ae.src_lock_path(env))):
                with open(path, "w") as f:
                    try:
                        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        seen[name] = "free"
                    except OSError:
                        seen[name] = "held"
            seen["locked"] = True
        return real(argv, timeout, e)
    env.run = run
    ae.ensure(env, check_only=False)
    assert seen["venv"] == "held" and seen["src"] == "held"
    for path in (ae.lock_path(env), ae.src_lock_path(env)):
        with open(path, "w") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_the_locks_are_taken_in_one_fixed_order_whatever_the_directories_are_called(env, tmp_path):
    order = []
    real_flock = ae.fcntl.flock
    import fcntl

    def spy(handle, op):
        order.append(handle.name)
        return real_flock(handle, op)
    for venv_name, src_name in (("a-venv", "z-src"), ("z-venv", "a-src")):
        env.venv_dir = (tmp_path / "x" / venv_name).resolve()
        env.src_dir = (tmp_path / "y" / src_name).resolve()
        order.clear()
        orig = ae.fcntl.flock
        ae.fcntl.flock = spy
        try:
            ae.ensure(env, check_only=False)
        finally:
            ae.fcntl.flock = orig
        assert order[:2] == sorted(order[:2])


def test_the_lock_list_is_sorted_and_has_each_file_once(env, monkeypatch):
    assert ae.lock_paths(env) == sorted([ae.lock_path(env), ae.src_lock_path(env)])
    monkeypatch.setattr(ae, "src_lock_path", lambda e: ae.lock_path(e))
    assert ae.lock_paths(env) == [ae.lock_path(env)]


def test_a_directory_that_cannot_hold_a_lock_file_is_a_failed_step_not_an_exception(env, tmp_path):
    blocker = tmp_path / "blocked"
    blocker.write_text("a file where a directory should be")
    env.src_dir = (blocker / "src").resolve()
    results = ae.ensure(env, check_only=False)
    assert results[0].status == ae.FAIL and results[0].name == "lock"


# ---- round five: descriptor-anchored reads and writes in the environment module --------------------------------

def test_the_record_is_read_through_the_venv_directory_descriptor_and_a_link_is_not_followed(env, tmp_path):
    ae.ensure(env, check_only=False)
    good = ae.read_record(env)
    assert good and good["library_commit"] == PIN
    rec = env.venv_dir / ae.RECORD
    real = tmp_path / "elsewhere.json"
    real.write_text(rec.read_text())
    rec.unlink()
    rec.symlink_to(real)
    assert ae.read_record(env) is None


def test_an_oversized_or_malformed_record_is_ignored(env):
    ae.ensure(env, check_only=False)
    rec = env.venv_dir / ae.RECORD
    rec.write_text("x" * (2 << 20))
    assert ae.read_record(env) is None
    rec.write_text("[]")
    assert ae.read_record(env) is None
    rec.write_text('{"a": 1, "a": 2}')
    assert ae.read_record(env) is None


def test_the_record_is_written_without_following_a_link_at_its_name(env, tmp_path):
    ae.ensure(env, check_only=False)
    rec = env.venv_dir / ae.RECORD
    target = tmp_path / "must-not-be-written"
    rec.unlink()
    rec.symlink_to(target)
    results = ae.ensure(env, check_only=False)
    assert statuses(results)["record"] == ae.OK and not target.exists()
    assert not rec.is_symlink() and json.loads(rec.read_text())["library_commit"] == PIN


def test_a_record_write_leaves_no_temporary_file(env):
    ae.ensure(env, check_only=False)
    assert not [p for p in env.venv_dir.iterdir() if p.name.endswith(".tmp")]


def test_a_lock_file_that_is_a_link_is_refused_and_not_followed(env, tmp_path):
    first, _ = ae.lock_paths(env)
    first.parent.mkdir(parents=True, exist_ok=True)
    target = tmp_path / "lock-target"
    first.symlink_to(target)
    results = ae.ensure(env, check_only=False)
    assert results[0].status == ae.FAIL and results[0].name == "lock" and "link" in results[0].detail
    assert not target.exists() and env.run.calls == []


def test_a_lock_file_that_is_not_a_regular_file_is_refused(env):
    first, _ = ae.lock_paths(env)
    first.mkdir(parents=True)
    results = ae.ensure(env, check_only=False)
    assert results[0].status == ae.FAIL and results[0].name == "lock"


def test_the_source_directory_replaced_during_the_build_fails_the_identity_rule(env, tmp_path):
    real = env.run

    def run(argv, timeout, e=None):
        if argv[:3] == ["uv", "pip", "install"] and "-e" in argv:
            env.src_dir.rename(tmp_path / "moved-away")
            (env.src_dir / ".git").mkdir(parents=True)
        return real(argv, timeout, e)
    env.run = run
    results = ae.ensure(env, check_only=False)
    assert statuses(results)["source-identity"] == ae.FAIL and "replaced" in [r for r in results if r.name == "source-identity"][0].detail
    assert statuses(results)["record"] == ae.FAIL and not (env.venv_dir / ae.RECORD).exists()


def test_the_git_directory_replaced_during_the_build_fails_the_identity_rule(env, tmp_path):
    real = env.run

    def run(argv, timeout, e=None):
        if argv[:3] == ["uv", "pip", "install"] and "-e" in argv:
            (env.src_dir / ".git").rename(tmp_path / "old-git")
            (env.src_dir / ".git").mkdir()
        return real(argv, timeout, e)
    env.run = run
    assert statuses(ae.ensure(env, check_only=False))["source-identity"] == ae.FAIL


def test_an_untouched_source_directory_passes_the_identity_rule_in_build_and_check_mode(env):
    assert statuses(ae.ensure(env, check_only=False))["source-identity"] == ae.OK
    assert statuses(ae.ensure(env, check_only=True))["source-identity"] == ae.OK


def test_no_identity_rule_runs_when_the_source_step_failed(env):
    env.src_dir.mkdir(parents=True)
    (env.src_dir / "file").write_text("x")
    assert "source-identity" not in statuses(ae.ensure(env, check_only=False))


# ---- descriptor anchoring: the record, the locks and the identity ---------------------------------------

def _fds():
    import os
    return {int(n) for n in os.listdir("/proc/self/fd")}


GOOD_RECORD = {"library_commit": "x", "k": 1}


def test_a_record_inside_a_venv_directory_that_is_a_link_is_not_read(env, tmp_path):
    real = tmp_path / "real-venv"
    real.mkdir()
    (real / ae.RECORD).write_text(json.dumps(GOOD_RECORD))
    env.venv_dir.parent.mkdir(parents=True, exist_ok=True)
    env.venv_dir.symlink_to(real)
    assert ae.read_record(env) is None


def test_reading_a_record_closes_what_it_opened(env):
    env.venv_dir.mkdir(parents=True)
    (env.venv_dir / ae.RECORD).write_text(json.dumps(GOOD_RECORD))
    before = _fds()
    assert ae.read_record(env) == GOOD_RECORD
    (env.venv_dir / ae.RECORD).write_text("not json")
    assert ae.read_record(env) is None
    assert _fds() <= before


def test_writing_a_record_replaces_a_stale_temporary_file(env):
    env.venv_dir.mkdir(parents=True)
    (env.venv_dir / f".{ae.RECORD}.tmp").write_text("stale")
    ae.write_record(env, GOOD_RECORD)
    assert json.loads((env.venv_dir / ae.RECORD).read_text()) == GOOD_RECORD
    assert not (env.venv_dir / f".{ae.RECORD}.tmp").exists()


def test_a_temporary_record_name_that_appears_after_the_cleanup_is_not_followed(env, tmp_path, monkeypatch):
    """The stale-file cleanup runs first. A link planted right after it must still not be written through."""
    env.venv_dir.mkdir(parents=True)
    victim = tmp_path / "victim"
    victim.write_text("keep me")
    real = ae.pf.unlink_quietly
    state = []

    def unlink_then_plant(fd, name):
        real(fd, name)
        if not state:
            state.append(1)
            (env.venv_dir / name).symlink_to(victim)
    monkeypatch.setattr(ae.pf, "unlink_quietly", unlink_then_plant)
    with pytest.raises(ae.pf.PinnedError):
        ae.write_record(env, GOOD_RECORD)
    assert victim.read_text() == "keep me"


def test_a_record_that_cannot_be_written_leaves_no_temporary_file_and_no_open_descriptor(env):
    env.venv_dir.mkdir(parents=True)
    before = _fds()
    with pytest.raises(TypeError):
        ae.write_record(env, {"x": object()})
    assert not (env.venv_dir / f".{ae.RECORD}.tmp").exists()
    assert _fds() <= before


def test_a_record_that_cannot_be_written_fails_the_record_step_with_the_reason(env):
    env.venv_dir.mkdir(parents=True)
    (env.venv_dir / ae.RECORD).mkdir()                 # a directory where the file goes: the rename fails
    r = ae.step_record(env, check_only=False, upstream_ok=True)
    assert r.status == ae.FAIL and "could not write" in r.detail


def test_a_lock_file_that_is_a_link_says_it_is_a_link(env, tmp_path):
    first, _ = ae.lock_paths(env)
    first.parent.mkdir(parents=True, exist_ok=True)
    first.symlink_to(tmp_path / "nowhere")
    results = ae.ensure(env, check_only=False)
    assert "is a link" in results[0].detail


def test_a_lock_file_that_is_a_fifo_is_refused_as_not_regular(env):
    import os
    first, _ = ae.lock_paths(env)
    first.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(first)
    results = ae.ensure(env, check_only=False)
    assert results[0].name == "lock" and "not a regular file" in results[0].detail


def test_the_source_identity_is_missing_when_either_directory_is_missing(env):
    assert ae.source_identity(env) is None
    env.src_dir.mkdir(parents=True)
    assert ae.source_identity(env) is None
    (env.src_dir / ".git").mkdir()
    ident = ae.source_identity(env)
    assert ident is not None and len(ident) == 4 and ident[1] != 0 and ident[3] != 0 and ident[1] != ident[3]
