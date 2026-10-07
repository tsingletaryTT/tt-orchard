"""scripts/setup.sh and orchard/setup_machine.py: prepare a machine for tt-orchard.

Everything runs against a fake machine: a dictionary of installed tools and canned command output, so no test
installs, downloads or touches a chip. The rules under test: every prerequisite is checked; tt-gozer is
installed with its own installer and must be new enough to reset and to lease under an owner pid; nothing
that resets, publishes, pushes, reconciles or leases is ever run; a file that exists is never overwritten; a
QuietBox 2 gets Coder-Next on one board by default and `--coder 27b` selects the earlier arrangement."""
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from orchard import setup_machine as sm
from orchard import tiers as tiers_module
from orchard.bringup_config import load as load_bringup

REPO = Path(__file__).resolve().parent.parent

QB2_STATUS = """grain: board   (2 boards, 4 chips)
board 0000046131924062  (p300c)
  chip 0  0000:01:00.0  FREE
  chip 1  0000:02:00.0  FREE
board 0000046131924055  (p300c)
  chip 2  0000:03:00.0  FREE
  chip 3  0000:04:00.0  FREE
"""
ONE_BOARD = "grain: board   (1 boards, 2 chips)\nboard 1  (p150)\n  chip 0  0000:01:00.0  FREE\n  chip 1  0000:02:00.0  FREE\n"
LIST_NEXT = "  ✓ raahemnabeel/qwen3-coder-next-blackhole           container  blackhole  image bd0ca927a46b  8.6 GB  profile p300x2\n"
LIST_27B = "  ✓ mando2222/qwen3.8-27b-dflash2-p300x2-q4kv  container  blackhole  image 5020  8.6 GB  profile batch8-dflash2\n"
ACQUIRE_HELP = "usage: gozer acquire [--owner-pid OWNER_PID] [--exact EXACT]"


class Machine:
    """A fake machine. `tools` are on PATH; `answers` map an argv prefix to (rc, out, err); `hooks` run
    when an argv with that prefix is executed; `ran` records every command."""

    def __init__(self, tmp_path, *, tools=("python3", "gozer", "tt-model", "docker", "hf", "uv", "ollama"),
                 status=QB2_STATUS, listing=LIST_NEXT + LIST_27B, ollama_list="qwen3-coder:30b  18 GB\n",
                 free=1000.0, venv_ok=True):
        self.tmp = tmp_path
        self.home = tmp_path / "home"
        self.checkout = tmp_path / "checkout"
        (self.checkout / "bin").mkdir(parents=True)
        (self.checkout / "bin" / "tt-orchard").write_text("#!/bin/sh\n")
        shutil_copytree(REPO / "config", self.checkout / "config")
        for name in ("bringup.toml", "tiers.toml"):                    # the checkout starts without live config
            p = self.checkout / "config" / name
            if p.exists():
                p.unlink()
        self.home.mkdir()
        self.tools, self.ran, self.spawned, self.hooks = set(tools), [], [], {}
        self.free, self.venv_ok = free, venv_ok
        self.answers = {
            ("gozer", "--version"): (0, "gozer 0.3.3\n", ""),
            ("gozer", "acquire", "--help"): (0, ACQUIRE_HELP, ""),
            ("gozer", "reset", "--help"): (0, "usage: gozer reset [-h] lease", ""),
            ("gozer", "status"): (0, status, ""),
            ("tt-model", "--version"): (0, "0.1.0\n", ""),
            ("tt-model", "list"): (0, listing, ""),
            ("docker", "info"): (0, "ok", ""),
            ("ollama", "list"): (0, ollama_list, ""),
        }
        self.existing = set()
        if venv_ok:
            self.mark_venv()

    def mark_venv(self):
        self.existing.add(str(self.home / ".local/share/tt-orchard/venvs/reference/bin/python"))

    def env(self, python=(3, 12, 1)):
        m = self
        return sm.Env(home=self.home, checkout=self.checkout, which=lambda n: f"/usr/bin/{n}" if n in m.tools else None,
                      run=self._run, spawn=self._spawn,
                      exists=lambda p: str(p) in m.existing or (str(p).startswith(str(m.tmp)) and os.path.exists(p)),
                      free_gb=lambda p: m.free,
                      python_version=python, environ={})

    def _run(self, argv, timeout=60):
        self.ran.append(list(argv))
        for n in range(len(argv), 0, -1):
            key = tuple(argv[:n])
            if key in self.hooks:
                self.hooks[key]()
            if key in self.answers:
                return self.answers[key]
        if argv[0].endswith("/bin/python") and argv[1:2] == ["-c"]:
            return (0, "", "") if self.venv_ok else (1, "", "ModuleNotFoundError: No module named 'torch'")
        return 0, "", ""

    def _spawn(self, argv, log):
        self.spawned.append((list(argv), log))
        return 4242


def shutil_copytree(src, dst):
    import shutil
    shutil.copytree(src, dst)


def steps_by_name(machine, **kw):
    opts = sm.Options(**kw)
    return {s.name: s for s in sm.plan(machine.env(), opts)}


def run_main(machine, *argv, answers=True, python=(3, 12, 1)):
    out = []
    code = sm.main(list(argv), env=machine.env(python), ask=lambda q: answers, say=out.append)
    return code, "\n".join(out)


# ---- a machine that is ready -------------------------------------------------------------------------

def test_a_ready_quietbox_has_nothing_to_do(tmp_path):
    m = Machine(tmp_path)
    for name in ("bringup.toml", "tiers.toml"):
        (m.checkout / "config" / name).write_text("x")
    link = m.home / ".local" / "bin" / "tt-orchard"
    link.parent.mkdir(parents=True)
    link.symlink_to(m.checkout / "bin" / "tt-orchard")
    m.existing.add("/dev/hugepages-1G")
    code, out = run_main(m, "--yes")
    assert code == 0 and "ready." in out
    assert not any(a[:2] in (["git", "clone"], ["ollama", "pull"], ["tt-model", "pull"]) for a in m.ran)


# ---- python --------------------------------------------------------------------------------------------

def test_an_old_python_is_a_failure(tmp_path):
    s = sm.check_python(Machine(tmp_path).env(python=(3, 10, 4)))
    assert s.status == sm.FAIL and "3.12" in s.detail and "3.10.4" in s.detail


def test_python_3_12_passes(tmp_path):
    assert sm.check_python(Machine(tmp_path / "a").env(python=(3, 12, 0))).status == sm.OK
    assert sm.check_python(Machine(tmp_path / "b").env(python=(3, 13, 1))).status == sm.OK


# ---- tt-gozer ------------------------------------------------------------------------------------------

def test_a_missing_gozer_is_cloned_and_installed_with_its_own_installer(tmp_path):
    m = Machine(tmp_path, tools=("python3", "tt-model", "docker", "hf", "uv", "ollama"))
    s = steps_by_name(m)["tt-gozer"]
    target = m.home / "code" / "tt-gozer"
    assert s.status == sm.TODO
    assert [a.args[0] for a in s.actions] == [["git", "clone", sm.GOZER_URL, str(target)], [str(target / "install.sh")]]


def test_an_existing_gozer_checkout_is_not_cloned_again(tmp_path):
    m = Machine(tmp_path, tools=("python3", "tt-model", "docker", "hf", "uv", "ollama"))
    m.existing.add(str(m.home / "code" / "tt-gozer" / "install.sh"))
    assert [a.args[0] for a in steps_by_name(m)["tt-gozer"].actions] == [[str(m.home / "code/tt-gozer/install.sh")]]


def test_the_gozer_directory_can_be_chosen(tmp_path):
    m = Machine(tmp_path, tools=("python3", "tt-model", "docker", "hf", "uv", "ollama"))
    s = steps_by_name(m, gozer_dir=tmp_path / "elsewhere")["tt-gozer"]
    assert s.actions[0].args[0][-1] == str(tmp_path / "elsewhere")


def test_yes_installs_gozer_then_checks_again_and_goes_green(tmp_path):
    m = Machine(tmp_path, tools=("python3", "tt-model", "docker", "hf", "uv", "ollama"))
    target = str(m.home / "code" / "tt-gozer")
    m.hooks[("git", "clone")] = lambda: m.existing.add(target + "/install.sh")
    m.hooks[(target + "/install.sh",)] = lambda: m.tools.add("gozer")
    code, out = run_main(m, "--yes")
    assert ["git", "clone", sm.GOZER_URL, target] in m.ran and [target + "/install.sh"] in m.ran
    assert m.ran.index(["git", "clone", sm.GOZER_URL, target]) < m.ran.index([target + "/install.sh"])
    assert "checking again" in out and "gozer 0.3.3" in out


def test_a_declined_step_runs_nothing(tmp_path):
    m = Machine(tmp_path, tools=("python3", "tt-model", "docker", "hf", "uv", "ollama"))
    code, out = run_main(m, answers=False)
    assert not any(a[:2] == ["git", "clone"] for a in m.ran) and "skipped" in out and code == 1


@pytest.mark.parametrize("version,ok", [("gozer 0.3.1", False), ("gozer 0.2.9", False), ("gozer 0.3.2", True),
                                        ("gozer 0.4.0", True), ("gozer 1.0.0", True)])
def test_gozer_must_be_0_3_2_or_newer(tmp_path, version, ok):
    m = Machine(tmp_path)
    m.answers[("gozer", "--version")] = (0, version + "\n", "")
    s = steps_by_name(m)["tt-gozer"]
    assert (s.status in (sm.OK, sm.WARN)) is ok
    if not ok:
        assert s.status == sm.FAIL and "older than 0.3.2" in s.detail and "install.sh" in s.detail


def test_an_unreadable_gozer_version_is_a_failure(tmp_path):
    m = Machine(tmp_path)
    m.answers[("gozer", "--version")] = (0, "no number here", "")
    assert steps_by_name(m)["tt-gozer"].status == sm.FAIL


def test_a_gozer_without_owner_pid_is_a_failure(tmp_path):
    m = Machine(tmp_path)
    m.answers[("gozer", "acquire", "--help")] = (0, "usage: gozer acquire [--exact EXACT]", "")
    s = steps_by_name(m)["tt-gozer"]
    assert s.status == sm.FAIL and "--owner-pid" in s.detail


def test_a_gozer_without_reset_is_a_failure(tmp_path):
    m = Machine(tmp_path)
    m.answers[("gozer", "reset", "--help")] = (2, "", "invalid choice: reset")
    s = steps_by_name(m)["tt-gozer"]
    assert s.status == sm.FAIL and "`reset`" in s.detail


def test_a_gozer_missing_both_names_both(tmp_path):
    m = Machine(tmp_path)
    m.answers[("gozer", "acquire", "--help")] = (0, "usage", "")
    m.answers[("gozer", "reset", "--help")] = (2, "", "bad")
    s = steps_by_name(m)["tt-gozer"]
    assert "--owner-pid" in s.detail and "`reset`" in s.detail and "and" in s.detail


def test_a_failing_gozer_status_is_a_failure_with_its_message(tmp_path):
    m = Machine(tmp_path)
    m.answers[("gozer", "status")] = (1, "", "cannot read sysfs")
    s = steps_by_name(m)["tt-gozer"]
    assert s.status == sm.FAIL and "cannot read sysfs" in s.detail


def test_a_gozer_that_lists_no_chips_is_a_failure(tmp_path):
    m = Machine(tmp_path, status="grain: board   (0 boards, 0 chips)\n")
    s = steps_by_name(m)["tt-gozer"]
    assert s.status == sm.FAIL and "no chips" in s.detail


def test_a_stale_lease_is_a_warning_that_names_reconcile_and_is_never_cleared(tmp_path):
    m = Machine(tmp_path, status=QB2_STATUS.replace("chip 0  0000:01:00.0  FREE", "chip 0  0000:01:00.0  STALE  x pid 1"))
    s = steps_by_name(m)["tt-gozer"]
    assert s.status == sm.WARN and "gozer reconcile" in s.detail and s.actions == []
    run_main(m, "--yes")
    assert not any("reconcile" in " ".join(a) for a in m.ran)


def test_chips_in_use_are_a_warning_not_a_failure(tmp_path):
    m = Machine(tmp_path, status=QB2_STATUS.replace("chip 2  0000:03:00.0  FREE", "chip 2  0000:03:00.0  HELD  x pid 1"))
    s = steps_by_name(m)["tt-gozer"]
    assert s.status == sm.WARN and "in use" in s.detail


def test_the_chip_count_is_reported(tmp_path):
    assert "4 chips" in steps_by_name(Machine(tmp_path))["tt-gozer"].detail


# ---- tt-model, docker, hf, hugepages -----------------------------------------------------------------

def test_a_missing_tt_model_is_a_failure_that_points_at_its_project(tmp_path):
    s = steps_by_name(Machine(tmp_path, tools=("python3", "gozer", "docker", "hf", "ollama")))["tt-model"]
    assert s.status == sm.FAIL and sm.TT_MODEL_URL in s.detail and s.actions == []


def test_a_broken_tt_model_is_a_failure(tmp_path):
    m = Machine(tmp_path)
    m.answers[("tt-model", "--version")] = (1, "", "boom")
    assert steps_by_name(m)["tt-model"].status == sm.FAIL


def test_a_missing_docker_and_an_unreachable_docker_are_failures(tmp_path):
    assert steps_by_name(Machine(tmp_path, tools=("python3", "gozer", "tt-model", "hf", "ollama")))["docker"].status == sm.FAIL
    m = Machine(tmp_path / "b")
    m.answers[("docker", "info")] = (1, "", "permission denied while trying to connect")
    s = steps_by_name(m)["docker"]
    assert s.status == sm.FAIL and "docker group" in s.detail


def test_a_missing_hf_is_installed_with_uv_or_pip(tmp_path):
    s = steps_by_name(Machine(tmp_path, tools=("python3", "gozer", "tt-model", "docker", "uv", "ollama")))["hf"]
    assert s.status == sm.TODO and s.actions[0].args[0] == ["uv", "tool", "install", "huggingface_hub"]
    s = steps_by_name(Machine(tmp_path / "b", tools=("python3", "gozer", "tt-model", "docker", "ollama")))["hf"]
    assert s.actions[0].args[0][-4:] == ["pip", "install", "--user", "huggingface_hub"]


def test_missing_hugepages_are_a_warning(tmp_path):
    m = Machine(tmp_path)
    assert steps_by_name(m)["hugepages"].status == sm.WARN
    m.existing.add("/dev/hugepages-1G")
    assert steps_by_name(m)["hugepages"].status == sm.OK


# ---- ollama --------------------------------------------------------------------------------------------

def test_a_missing_ollama_is_a_failure_with_its_download_page(tmp_path):
    s = steps_by_name(Machine(tmp_path, tools=("python3", "gozer", "tt-model", "docker", "hf")))["ollama"]
    assert s.status == sm.FAIL and "ollama.com" in s.detail


def test_an_ollama_server_that_is_down_is_a_warning_unless_asked_to_start_it(tmp_path):
    m = Machine(tmp_path)
    m.answers[("ollama", "list")] = (1, "", "could not connect")
    s = steps_by_name(m)["ollama"]
    assert s.status == sm.WARN and s.actions == [] and "--start-ollama" in s.detail
    s = steps_by_name(m, start_ollama=True)["ollama"]
    assert s.status == sm.TODO and s.actions[0].kind == "spawn" and s.actions[0].args[0] == ["ollama", "serve"]
    assert s.actions[0].args[1].endswith(".cache/tt-orchard/ollama.log")


def test_the_server_is_started_in_the_background_with_yes(tmp_path):
    m = Machine(tmp_path)
    m.answers[("ollama", "list")] = (1, "", "could not connect")
    run_main(m, "--yes", "--start-ollama")
    assert m.spawned and m.spawned[0][0] == ["ollama", "serve"]


def test_the_cpu_tier_model_is_pulled_when_it_is_missing(tmp_path):
    m = Machine(tmp_path, ollama_list="llama3  4 GB\n")
    s = steps_by_name(m)["ollama"]
    assert s.status == sm.TODO and s.actions[0].args[0] == ["ollama", "pull", "qwen3-coder:30b"]


# ---- the reference venv ------------------------------------------------------------------------------

def test_a_working_reference_venv_needs_nothing(tmp_path):
    assert steps_by_name(Machine(tmp_path))["reference venv"].status == sm.OK


def test_a_missing_reference_venv_is_built_with_uv_cpu_torch_and_the_tested_transformers(tmp_path):
    m = Machine(tmp_path, venv_ok=False)
    s = steps_by_name(m)["reference venv"]
    venv = str(m.home / ".local/share/tt-orchard/venvs/reference")
    cmds = [a.args[0] for a in s.actions]
    assert s.status == sm.TODO and cmds[0] == ["uv", "venv", venv, "--python", "3.12"]
    assert "--index-url" in cmds[1] and sm.TORCH_CPU_INDEX in cmds[1] and cmds[1][-1] == "torch"
    assert "transformers==5.19.0" in cmds[2]


def test_without_uv_the_venv_is_built_with_python_and_pip(tmp_path):
    m = Machine(tmp_path, venv_ok=False, tools=("python3", "gozer", "tt-model", "docker", "hf", "ollama"))
    cmds = [a.args[0] for a in steps_by_name(m)["reference venv"].actions]
    assert cmds[0][1:3] == ["-m", "venv"] and cmds[1][1:3] == ["-m", "pip"] and "transformers==5.19.0" in cmds[2]


def test_a_venv_that_exists_but_cannot_import_is_repaired_without_being_recreated(tmp_path):
    m = Machine(tmp_path, venv_ok=False)
    m.mark_venv()
    cmds = [a.args[0] for a in steps_by_name(m)["reference venv"].actions]
    assert all(c[:2] != ["uv", "venv"] for c in cmds) and len(cmds) == 2


def test_the_venv_location_can_be_chosen(tmp_path):
    m = Machine(tmp_path, venv_ok=False)
    s = steps_by_name(m, venv_dir=tmp_path / "myvenv")["reference venv"]
    assert str(tmp_path / "myvenv") in s.actions[0].args[0]


# ---- the coder package -------------------------------------------------------------------------------

def test_the_recommended_coder_package_is_pulled_when_it_is_missing(tmp_path):
    m = Machine(tmp_path, listing=LIST_27B)
    s = steps_by_name(m)["coder package"]
    assert s.status == sm.TODO and s.actions[0].args[0] == ["tt-model", "pull", "raahemnabeel/qwen3-coder-next-blackhole"]
    assert "160 GB" in s.detail


def test_an_installed_servable_package_is_ok(tmp_path):
    assert steps_by_name(Machine(tmp_path))["coder package"].status == sm.OK


def test_a_package_tt_model_says_cannot_be_served_is_a_warning(tmp_path):
    m = Machine(tmp_path, listing=LIST_NEXT.replace("✓", "✗"))
    s = steps_by_name(m)["coder package"]
    assert s.status == sm.WARN and "cannot be served" in s.detail


def test_little_free_disk_is_called_out_before_the_weights_are_downloaded(tmp_path):
    s = steps_by_name(Machine(tmp_path, listing=LIST_27B, free=90.0))["coder package"]
    assert "90 GB is free" in s.detail


def test_coder_27b_selects_the_other_package(tmp_path):
    m = Machine(tmp_path, listing=LIST_NEXT)
    s = steps_by_name(m, coder="27b")["coder package"]
    assert s.actions[0].args[0][-1] == "mando2222/qwen3.8-27b-dflash2-p300x2-q4kv"


# ---- configuration -----------------------------------------------------------------------------------

def test_a_quietbox_gets_coder_next_configuration_by_default(tmp_path):
    m = Machine(tmp_path)
    s = steps_by_name(m)["config"]
    assert s.status == sm.TODO and [Path(a.args[0]).name for a in s.actions] == ["bringup.toml", "tiers.toml"]
    bring = s.actions[0].args[1]
    assert 'target = "raahemnabeel/qwen3-coder-next-blackhole"' in bring and "port = 8001" in bring
    assert f'runs_root = "{m.home}/orchard-runs"' in bring
    assert f'reference_python = "{m.home}/.local/share/tt-orchard/venvs/reference/bin/python"' in bring
    assert "@" not in bring.replace("@gmail", "")                    # no placeholder left


def test_the_written_configuration_loads_with_the_real_loaders_and_agrees_with_itself(tmp_path):
    m = Machine(tmp_path)
    for a in steps_by_name(m)["config"].actions:
        Path(a.args[0]).write_text(a.args[1])
    cfg = load_bringup(m.checkout / "config" / "bringup.toml")
    tiers = tiers_module.load(m.checkout / "config" / "tiers.toml")
    from orchard import preflight
    assert preflight.check_tiers(lambda p: tiers, Path("t"), cfg.coder.port).status == "ok"
    assert cfg.coder.chips == 2 and cfg.required_chips == "2,4" and tiers.tiers["small"]["model"] == "Qwen/Qwen3-Coder-Next"


def test_coder_27b_writes_the_four_chip_arrangement_and_it_loads(tmp_path):
    m = Machine(tmp_path)
    for a in steps_by_name(m, coder="27b")["config"].actions:
        Path(a.args[0]).write_text(a.args[1])
    cfg = load_bringup(m.checkout / "config" / "bringup.toml")
    tiers = tiers_module.load(m.checkout / "config" / "tiers.toml")
    from orchard import preflight
    assert (cfg.coder.chips, cfg.coder.port, cfg.coder.profile) == (4, 8000, "batch8-dflash2")
    assert preflight.check_tiers(lambda p: tiers, Path("t"), cfg.coder.port).status == "ok"
    assert tiers.tiers["large"]["model"] == "Qwen/Qwen3.8-27B"


def test_existing_configuration_is_never_overwritten(tmp_path):
    m = Machine(tmp_path)
    (m.checkout / "config" / "bringup.toml").write_text("MINE")
    s = steps_by_name(m)["config"]
    assert [Path(a.args[0]).name for a in s.actions] == ["tiers.toml"]
    run_main(m, "--yes")
    assert (m.checkout / "config" / "bringup.toml").read_text() == "MINE"


def test_a_file_that_appears_after_the_plan_is_not_overwritten(tmp_path):
    m = Machine(tmp_path)
    target = m.checkout / "config" / "tiers.toml"
    act = sm.Action("write", (str(target), "NEW"))
    target.write_text("RACE")
    ok, note = sm.perform(act, m.env())
    assert not ok and target.read_text() == "RACE" and "not overwriting" in note


def test_another_machine_gets_a_warning_and_no_files(tmp_path):
    m = Machine(tmp_path, status=ONE_BOARD)
    s = steps_by_name(m)["config"]
    assert s.status == sm.WARN and s.actions == [] and "role-fit" in s.detail
    run_main(m, "--yes")
    assert not (m.checkout / "config" / "bringup.toml").exists()


def test_skills_dirs_are_added_only_when_the_plugin_directory_exists(tmp_path):
    m = Machine(tmp_path)
    assert "skills_dirs = [" not in steps_by_name(m)["config"].actions[0].args[1]
    skills = m.home / "code" / "skills" / "plugins" / "tt-model-bringup" / "skills"
    skills.mkdir(parents=True)
    assert f'skills_dirs = ["{skills}"]' in steps_by_name(m)["config"].actions[0].args[1]


def test_the_runs_root_can_be_chosen(tmp_path):
    s = steps_by_name(Machine(tmp_path), runs_root=Path("/data/runs"))["config"]
    assert 'runs_root = "/data/runs"' in s.actions[0].args[1]


# ---- the link ----------------------------------------------------------------------------------------

def test_tt_orchard_is_linked_into_local_bin(tmp_path):
    m = Machine(tmp_path)
    s = steps_by_name(m)["tt-orchard"]
    assert s.status == sm.TODO and s.actions[0].args == (str(m.home / ".local/bin/tt-orchard"), str(m.checkout / "bin/tt-orchard"))
    ok, _ = sm.perform(s.actions[0], m.env())
    assert ok and os.path.realpath(m.home / ".local/bin/tt-orchard") == os.path.realpath(m.checkout / "bin/tt-orchard")


def test_a_link_to_another_checkout_and_a_real_file_are_left_alone(tmp_path):
    m = Machine(tmp_path)
    link = m.home / ".local" / "bin" / "tt-orchard"
    link.parent.mkdir(parents=True)
    link.symlink_to(tmp_path / "other")
    s = steps_by_name(m)["tt-orchard"]
    assert s.status == sm.WARN and s.actions == []
    link.unlink()
    link.write_text("real file")
    assert steps_by_name(m)["tt-orchard"].status == sm.WARN and link.read_text() == "real file"


# ---- how it behaves ----------------------------------------------------------------------------------

def test_check_mode_runs_nothing_even_with_yes(tmp_path):
    m = Machine(tmp_path, tools=("python3", "tt-model", "docker"), venv_ok=False)
    before = len(m.ran)
    code, out = run_main(m, "--check", "--yes")
    installs = [a for a in m.ran if a[:1] in (["git"], ["uv"], ["ollama"]) or a[:2] == ["tt-model", "pull"]]
    assert installs == [] and m.spawned == [] and not (m.checkout / "config" / "bringup.toml").exists()
    assert code == 1


def test_every_command_it_prints_it_runs_after_asking(tmp_path):
    m = Machine(tmp_path, tools=("python3", "tt-model", "docker", "hf", "uv", "ollama"), venv_ok=False)
    asked = []
    out = []
    sm.main(["--yes"], env=m.env(), ask=lambda q: asked.append(q) or True, say=out.append)
    assert asked == []                                    # --yes never asks
    code, text = run_main(m, answers=False)
    assert "skipped" in text


def test_no_scenario_ever_runs_a_reset_publish_push_reconcile_or_lease_command(tmp_path):
    forbidden = sm.COMMANDS_NEVER_RUN
    for n, kw in enumerate([dict(), dict(tools=("python3",)), dict(venv_ok=False), dict(status=ONE_BOARD),
                            dict(status=QB2_STATUS.replace("FREE", "STALE"))]):
        m = Machine(tmp_path / str(n), **kw)
        run_main(m, "--yes", "--start-ollama")
        for argv in m.ran:
            words = [w for w in argv if not w.startswith("/")]
            if argv[:2] == ["gozer", "acquire"] or argv[:2] == ["gozer", "reset"]:
                assert argv[-1] == "--help", argv                 # the capability probes only
                continue
            assert not any(w in forbidden for w in words), argv


def test_a_failed_prerequisite_gives_exit_1_and_names_it(tmp_path):
    m = Machine(tmp_path, tools=("python3", "gozer", "docker", "hf", "ollama"))
    code, out = run_main(m, "--yes")
    assert code == 1 and "tt-model" in out.splitlines()[-1]


def test_a_failed_command_is_reported_and_stops_that_step(tmp_path):
    m = Machine(tmp_path, venv_ok=False)
    m.answers[("uv", "venv")] = (1, "", "no python 3.12")
    code, out = run_main(m, "--yes")
    assert "failed: uv venv" in out and "no python 3.12" in out
    assert not any(a[:3] == ["uv", "pip", "install"] for a in m.ran)


def test_the_report_has_one_line_per_step_and_plain_text_when_piped(tmp_path):
    code, out = run_main(Machine(tmp_path), "--check")
    for name in ("python", "tt-gozer", "tt-model", "docker", "hf", "ollama", "hugepages", "reference venv",
                 "coder package", "config", "tt-orchard"):
        assert any(line.lstrip().startswith(name) or f" {name}" in line[:20] for line in out.splitlines()), name
    assert "\x1b" not in out


def test_the_default_coder_is_the_recommendation():
    assert sm.CODERS["coder-next"]["package"].endswith("qwen3-coder-next-blackhole")
    p = subprocess.run([sys.executable, "-m", "orchard.setup_machine", "--help"], cwd=REPO, capture_output=True, text=True)
    assert "default: coder-next" in " ".join(p.stdout.split())


# ---- the shell wrapper -------------------------------------------------------------------------------

def test_the_wrapper_refuses_an_old_python_with_a_clear_message(tmp_path):
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "python3").write_text('#!/bin/sh\nif [ "$1" = "-c" ]; then exit 1; fi\necho "should not run"\n')
    (fake / "python3").chmod(0o755)
    p = subprocess.run(["bash", str(REPO / "scripts" / "setup.sh"), "--check"],
                       env={"PATH": f"{fake}:/usr/bin:/bin", "HOME": str(tmp_path)}, capture_output=True, text=True)
    assert p.returncode == 2 and "3.12" in p.stderr and "should not run" not in p.stdout


def test_the_wrapper_runs_the_module_from_any_directory(tmp_path):
    p = subprocess.run(["bash", str(REPO / "scripts" / "setup.sh"), "--help"], cwd=tmp_path, capture_output=True, text=True,
                       env={**os.environ, "HOME": str(tmp_path)})
    assert p.returncode == 0 and "--coder" in p.stdout


def test_the_wrapper_is_executable_and_uses_strict_mode():
    path = REPO / "scripts" / "setup.sh"
    assert path.stat().st_mode & stat.S_IXUSR
    assert "set -euo pipefail" in path.read_text()


def test_four_chips_on_two_boards_that_are_not_p300_is_not_a_quietbox_2(tmp_path):
    m = Machine(tmp_path, status=QB2_STATUS.replace("p300c", "p150"))
    s = steps_by_name(m)["config"]
    assert s.status == sm.WARN and s.actions == []


def test_cpu_torch_comes_from_the_pytorch_cpu_index(tmp_path):
    cmds = [a.args[0] for a in steps_by_name(Machine(tmp_path, venv_ok=False))["reference venv"].actions]
    torch_install = next(c for c in cmds if c[-1] == "torch")
    assert "https://download.pytorch.org/whl/cpu" in torch_install


def test_without_the_coder_option_the_recommendation_is_set_up(tmp_path):
    code, out = run_main(Machine(tmp_path, listing=LIST_27B), "--check")
    assert out.splitlines()[0].startswith("tt-orchard setup (coder-next:") or "coder-next:" in out.splitlines()[0]
    assert "raahemnabeel/qwen3-coder-next-blackhole" in out
    code, out = run_main(Machine(tmp_path / "b", listing=LIST_NEXT), "--check", "--coder", "27b")
    assert "27b:" in out.splitlines()[0] and "mando2222" in out
