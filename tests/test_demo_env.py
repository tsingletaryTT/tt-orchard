# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""tt-orchard demo-env: the interpreter that runs tt-metal demos and tt-train (issue 92).

A fake runner stands in for git, uv and the venv's python, and every file lives in tmp_path, so nothing here
touches the network, the shared venv, the host tt-metal tree or a device. The fake answers a python call by the
code string it was given, so a test sees which questions the check asked."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from orchard import demo_env as de

TAG = "v0.80.0"
SHA = "978f93b56e36b68318d11ecf7b101085a81d718e"


class FakeRun:
    """Records every call. A rule is (predicate on argv, (rc, out, err), effect); the first match answers.
    A call with no rule answers 127, so a command the test did not expect is visible."""

    def __init__(self):
        self.calls, self.rules = [], []

    def on(self, match, result=(0, "", ""), effect=None):
        if isinstance(match, (list, tuple)):
            prefix = tuple(match)
            match = lambda argv, p=prefix: tuple(argv[:len(p)]) == p      # noqa: E731
        self.rules.append((match, result, effect))

    def __call__(self, argv, timeout, env=None):
        self.calls.append((list(argv), dict(env) if env is not None else None))
        for match, result, effect in self.rules:
            if match(argv):
                if effect:
                    effect(argv)
                return result
        return 127, "", f"no rule for {argv}"

    def argvs(self):
        return [a for a, _ in self.calls]

    def installs(self):
        return [a for a in self.argvs() if a[0] == "uv"]


class World:
    """A host tree, a venv directory and a runner that behaves like a healthy machine until a test breaks one thing."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.tree = tmp_path / "tt-metal"
        (self.tree / "ttnn" / "ttnn").mkdir(parents=True)
        (self.tree / "ttnn" / "ttnn" / "_ttnn.so").write_bytes(b"\x7fELF")
        (self.tree / "tools").mkdir()
        (self.tree / "tt-train" / "sources" / "ttml").mkdir(parents=True)
        (self.tree / "build" / "tt-train" / "sources" / "ttml").mkdir(parents=True)
        (self.tree / "pyproject.toml").write_text('[project]\nname = "ttnn"\ndependencies = ["numpy>=1.24.4,<2", "loguru>=0.6.0"]\n')
        self.venv = tmp_path / "venvs" / "tt-demo"
        self.site = self.venv / "lib" / "python3.12" / "site-packages"
        self.shared = tmp_path / "home" / ".tenstorrent-venv"
        self.repo = tmp_path / "checkout"
        self.repo.mkdir()
        self.run = FakeRun()
        self.numpy = "1.26.4"
        self.broken = set()           # import names that fail
        self.dists = {}               # distribution -> version the venv's metadata reports (set by the build effect)
        self.install_rules()

    # -- the fake machine -------------------------------------------------------------------------------------
    def install_rules(self):
        t = str(self.tree)
        self.run.on(["git", "-C", t, "describe", "--tags", "--exact-match"], (0, TAG + "\n", ""))
        self.run.on(["git", "-C", t, "rev-parse", "HEAD"], (0, SHA + "\n", ""))
        self.run.on(["uv", "venv"], effect=self._make_venv)
        self.run.on(["uv", "pip"])

    def _make_venv(self, argv):
        (self.venv / "bin").mkdir(parents=True, exist_ok=True)
        (self.venv / "bin" / "python").write_text("")
        self.site.mkdir(parents=True, exist_ok=True)

    def env(self, **over):
        args = dict(venv_dir=self.venv, tt_metal_dir=self.tree, shared_venv=self.shared, repo_dir=self.repo,
                    run=self.dynamic, environ={"PATH": "/usr/bin"})
        args.update(over)
        return de.DEnv(**args)

    def dynamic(self, argv, timeout, env=None):
        """Python questions get an answer from the world's current state; everything else goes to the rules."""
        py = str(self.venv / "bin" / "python")
        if argv[:1] == [py] and len(argv) >= 4 and argv[1:3] == ["-I", "-c"]:
            code = argv[3]
            self.run.calls.append((list(argv), dict(env) if env is not None else None))
            if code == de.PURELIB_CODE:
                return 0, f"{self.site}\n", ""
            for name, probe in (("cv2", de.CV2_CODE), ("datasets", de.DATASETS_CODE), ("torch", de.TORCH_CODE)):
                if code == probe:
                    if name in self.broken:
                        return 1, "", f"Traceback (most recent call last):\nImportError: {name} is broken here"
                    return 0, "", ""
            if code == de.VERSIONS_CODE:
                return 0, json.dumps({"numpy": self.numpy, "ttnn": "0.80.0"}), ""
            if code == de.dist_code("numpy"):
                return 0, f"{self.numpy}\n", ""
            for dist in ("ttnn", "ttml"):
                if code == de.dist_code(dist):
                    v = self.installed_dist(dist)
                    return (0, v + "\n", "") if v else (1, "", "importlib.metadata.PackageNotFoundError: " + dist)
            return 127, "", f"no answer for {code}"
        return self.run(argv, timeout, env)

    def installed_dist(self, dist):
        for d in self.site.glob(f"{dist}-*.dist-info"):
            for line in (d / "METADATA").read_text().splitlines():
                if line.startswith("Version: "):
                    return line.split(": ", 1)[1]
        return None

    def python_calls(self):
        py = str(self.venv / "bin" / "python")
        return [a for a in self.run.argvs() if a[:1] == [py]]

    def tree_snapshot(self):
        return sorted((str(p.relative_to(self.tree)), p.stat().st_mtime_ns) for p in self.tree.rglob("*"))


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


def names(results):
    return {r.name: r for r in results}


def build(world, **kw):
    env = world.env()
    return env, de.ensure(env, check_only=False, say=lambda s: None, **kw)


# ---- what gets installed -------------------------------------------------------------------------------------

def test_every_requirement_is_pinned_to_one_version():
    assert all("==" in r for r in de.REQUIREMENTS) and all("==" in de.TORCH for _ in [0])


def test_the_requirement_names_are_the_reviewed_set():
    """A name added or changed here shows up in a test, so a typosquat cannot slip in unseen (global CLAUDE.md)."""
    got = {r.split("==")[0].lower() for r in de.REQUIREMENTS}
    assert got == {"numpy", "huggingface_hub", "datasets", "pyarrow", "transformers", "tokenizers", "safetensors", "pytest", "requests", "loguru",
                   "networkx", "graphviz", "pyyaml", "click", "pandas", "seaborn", "ml_dtypes", "pillow", "termcolor", "tqdm",
                   "pydantic", "opencv-python-headless"}


def test_numpy_is_below_two_and_the_headless_opencv_is_the_one_asked_for():
    pins = dict(r.split("==") for r in de.REQUIREMENTS)
    assert pins["numpy"].startswith("1.")
    assert "opencv-python-headless" in pins and "opencv-python" not in pins


def test_pyarrow_is_pinned_to_a_version_that_imports_with_numpy_1():
    """The first real --check found `datasets` failing: the newest pyarrow (26) needs numpy 2. 21.0.0 is the oldest datasets 4.8.5 accepts."""
    pins = dict(r.split("==") for r in de.REQUIREMENTS)
    assert pins["pyarrow"] == "21.0.0" and pins["numpy"] == "1.26.4" and pins["datasets"] == "4.8.5"


def test_the_requirements_hash_changes_with_a_pin(monkeypatch):
    a = de.requirements_sha256()
    monkeypatch.setattr(de, "REQUIREMENTS", de.REQUIREMENTS[:-1] + ("numpy==1.26.3",))
    assert de.requirements_sha256() != a


# ---- the build -------------------------------------------------------------------------------------------------

def test_a_build_makes_the_venv_installs_torch_alone_from_the_cpu_index_and_then_the_pins(world):
    env, results = build(world)
    assert all(r.status == de.OK for r in results), [(r.name, r.status, r.detail) for r in results]
    uv = world.run.installs()
    assert uv[0][:2] == ["uv", "venv"] and "3.12" in uv[0] and str(world.venv) in uv[0]
    torch = [a for a in uv if de.TORCH in a]
    assert len(torch) == 1 and "--index-url" in torch[0] and de.TORCH_CPU_INDEX in torch[0]
    assert not any("--extra-index-url" in a for a in uv)
    pins = [a for a in uv if "numpy==1.26.4" in a]
    assert len(pins) == 1 and de.TORCH_CPU_INDEX not in pins[0]
    assert pins[0][pins[0].index("--index-url") + 1] == de.PYPI_INDEX
    assert all(r in pins[0] for r in de.REQUIREMENTS)
    assert uv.index(torch[0]) < uv.index(pins[0])


def test_every_install_names_the_demo_venv_interpreter_and_never_the_shared_one(world):
    build(world)
    for argv in world.run.installs():
        if argv[:2] == ["uv", "pip"]:
            assert argv[argv.index("--python") + 1] == str(world.venv / "bin" / "python")
    assert not any(str(world.shared) in " ".join(a) for a in world.run.argvs())


def test_the_host_tree_is_installed_as_path_entries_and_two_distributions_at_the_trees_tag(world):
    build(world)
    pth = (world.site / de.PTH_NAME).read_text().splitlines()
    t = world.tree
    assert pth == [str(t), str(t / "ttnn"), str(t / "tools"), str(t / "tt-train/sources/ttml"),
                   str(t / "build/tt-train/sources/ttml")]
    for dist in ("ttnn", "ttml"):
        assert world.installed_dist(dist) == "0.80.0"
        info = next(world.site.glob(f"{dist}-0.80.0.dist-info"))
        assert json.loads((info / "direct_url.json").read_text())["url"] == f"file://{t}"
        assert (info / "INSTALLER").read_text().strip() == "orchard-demo-env"


def test_an_index_from_the_operators_shell_never_reaches_uv(world):
    env = world.env(environ={"PATH": "/usr/bin", "UV_EXTRA_INDEX_URL": "https://evil.example/simple", "PIP_INDEX_URL": "https://evil.example/x",
                             "UV_INDEX": "x=https://evil.example"})
    de.ensure(env, check_only=False, say=lambda s: None)
    envs = [e for a, e in world.run.calls if a[:3] == ["uv", "pip", "install"]]
    assert envs and all(e is not None and e.get("PATH") == "/usr/bin" and not any(k.startswith(("UV_", "PIP_")) for k in e) for e in envs)


def test_a_ttnn_distribution_somebody_else_installed_is_left_alone_and_fails_the_step(world):
    world._make_venv(None)
    foreign = world.site / "ttnn-0.79.0.dist-info"
    foreign.mkdir()
    (foreign / "METADATA").write_text("Name: ttnn\nVersion: 0.79.0\n")
    (foreign / "INSTALLER").write_text("pip\n")
    _, results = build(world)
    r = names(results)["host-tree"]
    assert r.status == de.FAIL and "left alone" in r.detail
    assert (foreign / "METADATA").exists() and not (world.site / de.PTH_NAME).exists()


def test_an_old_distribution_this_command_wrote_is_replaced_when_the_tag_moves(world, monkeypatch):
    healthy(world)
    world.run.rules.insert(0, (lambda a: "--exact-match" in a, (0, "v0.81.1\n", ""), None))
    assert names(check(world))["record"].status == de.TODO                       # the record is for the old tag and sha
    build(world)
    assert world.installed_dist("ttnn") == "0.81.1" and not list(world.site.glob("ttnn-0.80.0.dist-info"))
    assert json.loads((world.venv / de.RECORD).read_text())["host_tree"]["tag"] == "v0.81.1"


def test_each_distribution_has_the_files_pip_and_uv_need_to_read_it(world):
    build(world)
    for dist in ("ttnn", "ttml"):
        info = world.site / f"{dist}-0.80.0.dist-info"
        assert "Wheel-Version: 1.0" in (info / "WHEEL").read_text()          # `uv pip check` calls a dist-info without one broken
        assert f"Name: {dist}" in (info / "METADATA").read_text() and (info / "RECORD").read_text().count(",,") == 6


def test_a_distribution_missing_a_file_is_written_again_by_the_next_build(world):
    build(world)
    (world.site / "ttnn-0.80.0.dist-info" / "WHEEL").unlink()
    assert names(check(world))["host-tree"].status == de.TODO
    build(world)
    assert (world.site / "ttnn-0.80.0.dist-info" / "WHEEL").is_file()


def test_a_path_the_tree_does_not_have_is_left_out_of_the_pth(world):
    import shutil
    shutil.rmtree(world.tree / "build")
    build(world)
    assert str(world.tree / "build/tt-train/sources/ttml") not in (world.site / de.PTH_NAME).read_text()


def test_the_build_never_writes_into_the_host_tree(world):
    before = world.tree_snapshot()
    build(world)
    assert world.tree_snapshot() == before


def test_the_record_holds_the_tag_the_sha_the_requirements_hash_and_the_resolved_versions(world):
    build(world)
    rec = json.loads((world.venv / de.RECORD).read_text())
    assert rec["host_tree"] == {"path": str(world.tree), "tag": TAG, "sha": SHA}
    assert rec["requirements"] == list(de.REQUIREMENTS) and rec["torch"] == de.TORCH
    assert rec["requirements_sha256"] == de.requirements_sha256()
    assert rec["installed"] == {"numpy": "1.26.4", "ttnn": "0.80.0"}


def test_a_second_build_changes_nothing_and_installs_nothing(world):
    build(world)
    n_installs, rec = len(world.run.installs()), (world.venv / de.RECORD).read_text()
    pth_stat = (world.site / de.PTH_NAME).stat().st_mtime_ns
    env, results = build(world)
    assert all(r.status == de.OK for r in results)
    assert len(world.run.installs()) == n_installs
    assert (world.venv / de.RECORD).read_text() == rec and (world.site / de.PTH_NAME).stat().st_mtime_ns == pth_stat


def test_a_host_tree_that_is_not_built_fails_and_nothing_is_written_into_the_venv(world):
    (world.tree / "ttnn" / "ttnn" / "_ttnn.so").unlink()
    env, results = build(world)
    r = names(results)["host-tree"]
    assert r.status == de.FAIL and "_ttnn.so" in r.detail
    assert not (world.site / de.PTH_NAME).exists() and not (world.venv / de.RECORD).exists()


def test_a_failed_install_leaves_no_record_and_the_next_run_installs_again(world):
    world.run.rules.insert(0, (lambda a: a[:3] == ["uv", "pip", "install"] and "--index-url" in a and de.PYPI_INDEX in a,
                               (1, "", "network is down"), None))
    _, results = build(world)
    assert names(results)["packages"].status == de.FAIL and "network is down" in names(results)["packages"].detail
    assert names(results)["record"].status == de.TODO and not (world.venv / de.RECORD).exists()
    world.run.rules.pop(0)
    _, again = build(world)
    assert all(r.status == de.OK for r in again) and (world.venv / de.RECORD).exists()


def test_a_tool_that_fails_does_not_pass_because_it_printed_a_tag(world):
    world.run.rules.insert(0, (lambda a: "--exact-match" in a, (128, "v0.80.0\n", "fatal: not a repository"), None))
    _, results = build(world)
    assert names(results)["host-tree"].status == de.FAIL


def test_a_host_tree_not_at_a_release_tag_fails(world):
    world.run.rules.insert(0, (lambda a: a[:2] == ["git", "-C"] and "--exact-match" in a, (128, "", "fatal: no tag exactly matches"), None))
    env, results = build(world)
    r = names(results)["host-tree"]
    assert r.status == de.FAIL and "release tag" in r.detail
    assert not (world.venv / de.RECORD).exists()


def test_a_tag_that_is_not_a_version_number_fails(world):
    world.run.rules.insert(0, (lambda a: "--exact-match" in a, (0, "weekly-build\n", ""), None))
    _, results = build(world)
    assert names(results)["host-tree"].status == de.FAIL


# ---- refusals ----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("where", ["shared", "inside_shared", "inside_repo", "repo"])
def test_the_command_refuses_the_shared_venv_and_the_repository_and_runs_nothing(world, where):
    venv = {"shared": world.shared, "inside_shared": world.shared / "sub", "inside_repo": world.repo / "venv", "repo": world.repo}[where]
    env = world.env(venv_dir=venv)
    results = de.ensure(env, check_only=False, say=lambda s: None)
    assert results[0].name == "venv-dir" and results[0].status == de.FAIL
    assert str(venv) in results[0].detail and "inside it" in results[0].detail
    assert world.run.calls == [] and (where == "repo" or not venv.exists())


def test_a_symlink_to_the_shared_venv_is_refused_too(world):
    world.shared.mkdir(parents=True)
    link = world.tmp / "link"
    link.symlink_to(world.shared)
    results = de.ensure(world.env(venv_dir=link), check_only=False, say=lambda s: None)
    assert results[0].status == de.FAIL and world.run.calls == []


def test_a_host_tree_inside_the_venv_is_refused(world):
    env = world.env(tt_metal_dir=world.venv / "tt-metal")
    results = de.ensure(env, check_only=False, say=lambda s: None)
    assert results[0].name == "venv-dir" and results[0].status == de.FAIL and world.run.calls == []


def test_a_venv_directory_that_holds_other_files_and_no_python_is_not_built_into(world):
    world.venv.mkdir(parents=True)
    (world.venv / "notes.txt").write_text("mine")
    results = de.ensure(world.env(), check_only=False, say=lambda s: None)
    assert names(results)["venv"].status == de.FAIL and world.run.installs() == []
    assert (world.venv / "notes.txt").read_text() == "mine"


# ---- --check ------------------------------------------------------------------------------------------------------

def healthy(world):
    build(world)
    world.run.calls.clear()


def check(world, **kw):
    return de.ensure(world.env(), check_only=True, say=lambda s: None, **kw)


def test_a_healthy_venv_passes_every_rule(world):
    healthy(world)
    results = check(world)
    assert all(r.status == de.OK for r in results), [(r.name, r.status, r.detail) for r in results]
    assert {"numpy", "cv2", "datasets", "torch", "ttnn", "ttml", "ttnn-version", "host-tree", "record"} <= set(names(results))


def test_check_changes_nothing_and_runs_no_installer(world):
    healthy(world)
    snap = sorted(str(p) for p in world.tmp.rglob("*"))
    check(world)
    assert world.run.installs() == [] and sorted(str(p) for p in world.tmp.rglob("*")) == snap


def test_check_never_imports_ttnn_or_ttml_and_runs_python_isolated(world):
    healthy(world)
    check(world)
    calls = world.python_calls()
    assert calls
    for argv in calls:
        assert argv[1:3] == ["-I", "-c"]
        code = argv[3]
        assert "import ttnn" not in code and "import ttml" not in code and "from ttnn" not in code
        assert "ttnn" not in code.replace("m.version('ttnn')", "")        # only the metadata question names it


def test_numpy_two_fails_the_numpy_rule(world):
    healthy(world)
    world.numpy = "2.5.3"
    r = names(check(world))["numpy"]
    assert r.status == de.FAIL and "2.5.3" in r.detail and "below 2" in r.detail


def test_an_unreadable_numpy_version_fails(world):
    healthy(world)
    world.numpy = "not-a-version"
    assert names(check(world))["numpy"].status == de.FAIL


@pytest.mark.parametrize("module", ["cv2", "datasets", "torch"])
def test_an_import_that_fails_is_reported_with_its_last_line(world, module):
    healthy(world)
    world.broken.add(module)
    r = names(check(world))[module]
    assert r.status == de.FAIL and f"ImportError: {module} is broken here" in r.detail
    others = [x for x in check(world) if x.name in {"cv2", "datasets", "torch"} - {module}]
    assert all(x.status == de.OK for x in others)


def test_a_missing_ttnn_distribution_fails(world):
    healthy(world)
    import shutil
    for d in world.site.glob("ttnn-*.dist-info"):
        shutil.rmtree(d)
    r = names(check(world))["ttnn"]
    assert r.status == de.FAIL and "not installed" in r.detail


def test_a_ttnn_version_that_is_not_the_trees_tag_fails_the_version_rule(world):
    healthy(world)
    world.run.rules.insert(0, (lambda a: "--exact-match" in a, (0, "v0.81.1\n", ""), None))
    r = names(check(world))["ttnn-version"]
    assert r.status == de.FAIL and "0.80.0" in r.detail and "0.81.1" in r.detail


def test_a_pth_entry_that_points_somewhere_else_fails_the_host_tree_rule(world):
    healthy(world)
    (world.site / de.PTH_NAME).write_text("/somewhere/else/tt-metal\n")
    assert names(check(world))["host-tree"].status == de.FAIL


def test_a_pth_entry_that_no_longer_exists_fails_the_host_tree_rule(world):
    healthy(world)
    import shutil
    shutil.rmtree(world.tree / "tools")
    assert names(check(world))["host-tree"].status == de.FAIL


def test_a_record_for_another_tag_is_reported_and_names_both_tags(world):
    healthy(world)
    rec = json.loads((world.venv / de.RECORD).read_text())
    rec["host_tree"]["tag"] = "v0.79.0"
    (world.venv / de.RECORD).write_text(json.dumps(rec))
    r = names(check(world))["record-tag"]
    assert r.status == de.FAIL and "v0.79.0" in r.detail and TAG in r.detail


def test_a_record_with_another_requirements_hash_is_to_do(world, monkeypatch):
    healthy(world)
    monkeypatch.setattr(de, "REQUIREMENTS", de.REQUIREMENTS[:-1] + ("numpy==1.26.3",))
    r = names(check(world))["record"]
    assert r.status == de.TODO and "requirements" in r.detail


def test_no_record_is_to_do(world):
    healthy(world)
    (world.venv / de.RECORD).unlink()
    assert names(check(world))["record"].status == de.TODO


def test_no_venv_reports_to_do_and_asks_nothing_of_a_python_that_is_not_there(world):
    results = check(world)
    assert names(results)["venv"].status == de.TODO
    assert world.python_calls() == []


def test_a_rebuild_after_a_new_requirements_pin_installs_the_pins_again(world, monkeypatch):
    healthy(world)
    monkeypatch.setattr(de, "REQUIREMENTS", de.REQUIREMENTS[:-1] + ("numpy==1.26.3",))
    build(world)
    assert any("numpy==1.26.3" in a for a in world.run.installs())
    assert json.loads((world.venv / de.RECORD).read_text())["requirements_sha256"] == de.requirements_sha256()


# ---- main ---------------------------------------------------------------------------------------------------------

class Args:
    def __init__(self, **kw):
        self.check, self.dry_run, self.venv_dir, self.tt_metal_dir = False, False, None, None
        self.__dict__.update(kw)


def lines(world, **kw):
    out = []
    code = de.main(Args(**kw), env=world.env(), out=_Sink(out))
    return code, "\n".join(out)


class _Sink:
    def __init__(self, lines):
        self.lines = lines

    def write(self, s):
        self.lines.extend(x for x in s.split("\n") if x)

    def flush(self):
        pass


def test_main_exits_0_after_a_build_and_prints_one_line_per_step(world):
    code, text = lines(world)
    assert code == 0 and "ok" in text and "host-tree" in text and "record" in text


def test_main_check_exits_2_for_an_unbuilt_venv_and_0_for_a_healthy_one(world):
    assert lines(world, check=True)[0] == 2
    healthy(world)
    assert lines(world, check=True)[0] == 0


def test_main_exits_2_when_a_step_failed(world):
    world.run.rules.insert(0, (lambda a: a[:2] == ["uv", "venv"], (1, "", "no python 3.12"), None))
    assert lines(world)[0] == 2


def test_dry_run_says_what_it_would_do_and_does_nothing(world):
    code, text = lines(world, dry_run=True)
    assert code == 0 and "would" in text and "uv venv" in text and de.TORCH_CPU_INDEX in text
    assert world.run.installs() == [] and not world.venv.exists()


def test_dry_run_of_a_healthy_venv_says_nothing_is_to_do(world):
    healthy(world)
    code, text = lines(world, dry_run=True)
    assert code == 0 and "would" not in text and world.run.installs() == []


def test_the_default_venv_is_under_orchard_venvs_in_the_home_and_an_environment_variable_overrides_it(tmp_path):
    a = Args()
    e = de.env_from_args(a, environ={"HOME": str(tmp_path)}, home=tmp_path, repo_dir=tmp_path / "repo")
    assert e.venv_dir == tmp_path / "orchard-venvs" / "tt-demo" and e.tt_metal_dir == tmp_path / "tt-metal"
    assert e.shared_venv == tmp_path / ".tenstorrent-venv"
    e = de.env_from_args(a, environ={"ORCHARD_DEMO_VENV": "/x/venv", "ORCHARD_TT_METAL_DIR": "/x/tree"}, home=tmp_path,
                         repo_dir=tmp_path / "repo")
    assert e.venv_dir == Path("/x/venv") and e.tt_metal_dir == Path("/x/tree")
    e = de.env_from_args(Args(venv_dir="/y/v", tt_metal_dir="/y/t"), environ={"ORCHARD_DEMO_VENV": "/x/venv"}, home=tmp_path,
                         repo_dir=tmp_path / "repo")
    assert e.venv_dir == Path("/y/v") and e.tt_metal_dir == Path("/y/t")


def test_the_fix_command_names_the_venv_and_the_tree(world):
    cmd = de.fix_command(world.env())
    assert cmd == f"tt-orchard demo-env --venv-dir {world.venv} --tt-metal-dir {world.tree}"
