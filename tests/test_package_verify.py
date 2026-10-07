# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 7, part 2: install a copy of the staged package and boot-check it (plan 5).

The copy's install.sh (from tests/fake_package_thin.py) makes venv/bin/python a wrapper that serves
tests/fake_swap_server.py. Everything else is the staged package as stage 7 wrote it: its run.sh,
prepare_model_dir.py and base_config/. Each test that starts a server kills its process group in
a finalizer, so a failing test leaves nothing behind.
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from orchard.package import (PackageError, Profile, aux_repos, bundle_name, prepare_verify,
                             read_run, stage_profile)
from package_fakes import (BASE, DRAFTER, GENERATED, HAVE_TOKENIZERS, NEW, NEW_REV, PROMPT_IDS,
                           VOCAB, fake_bin, make_run, make_source)

if not HAVE_TOKENIZERS:
    pytest.skip("SKIPPED: the `tokenizers` package is not importable, so the stage 7 boot check "
                "did not run. verify_bundle.py needs it.", allow_module_level=True)

FAKE_SERVER = Path(__file__).resolve().with_name("fake_swap_server.py")
pytestmark = pytest.mark.usefixtures("stub_tools")


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.fixture
def boot(tmp_path, monkeypatch):
    """A staged 2-chip package, an installed copy under stages/7/verify, and a runner for the check."""
    models = tmp_path / "models"
    r = make_run(tmp_path, make_source(models))
    fb, _ = fake_bin(tmp_path)
    monkeypatch.setenv("PATH", f"{fb}:{os.environ['PATH']}")
    facts = read_run(r["run"])
    out = r["run"] / "stages/7/package" / bundle_name(NEW, facts.source.name)
    out.parent.mkdir(parents=True)
    stage_profile(Profile(2, facts.source, True), facts, out)
    pid_file, server_cfg = tmp_path / "server-pid.json", tmp_path / "server.json"
    env = dict(os.environ, FAKE_TT_MODEL_PYTHON=sys.executable, FAKE_SWAP_SERVER=str(FAKE_SERVER),
               FAKE_SWAP_CONFIG=str(server_cfg))
    verify = r["run"] / "stages/7/verify"
    test = prepare_verify(out, verify, facts, env=env)
    cfg_path = verify / "verify_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["health_timeout_s"] = 30
    cfg_path.write_text(json.dumps(cfg))

    def check(mode="perfect", env_extra=None, **server):
        server_cfg.write_text(json.dumps({"mode": mode, "vocab": VOCAB, "prompt_ids": PROMPT_IDS,
                                          "generated_ids": GENERATED, "pid_file": str(pid_file),
                                          **server}))
        clean = {k: v for k, v in os.environ.items() if k not in ("MODEL_WEIGHTS_DIR", "HF_MODEL")}
        clean.update(env_extra or {})
        return subprocess.run([sys.executable, str(verify / "verify_bundle.py")], capture_output=True,
                              text=True, timeout=180, env=clean, cwd=r["run"])

    state = {**r, "facts": facts, "out": out, "verify": verify, "test": test, "check": check,
             "pid_file": pid_file, "cfg": cfg}
    try:
        yield state
    finally:
        if pid_file.exists():
            pgid = json.loads(pid_file.read_text())["pgid"]
            if group_alive(pgid):
                os.killpg(pgid, signal.SIGKILL)
                print(f"fixture killed a leaked fake server group {pgid}", file=sys.stderr)


def verify_json(state) -> dict:
    return json.loads((state["verify"] / "evidence" / "verify.json").read_text())


def wait_gone(pgid: int, within: float = 5.0) -> bool:
    end = time.monotonic() + within
    while time.monotonic() < end and group_alive(pgid):
        time.sleep(0.1)
    return not group_alive(pgid)


def test_the_installed_copy_serves_the_new_weights_and_agrees(boot):
    r = boot["check"]("perfect")
    assert r.returncode == 0, r.stdout + r.stderr
    v = verify_json(boot)
    md = str(boot["verify"] / "bundle" / "model-dir")
    assert v["top1_agreement"] == 1.0 and v["coherent"] is True and v["n_tokens"] == 32
    assert v["served_model"] == md
    assert v["server_weights_env"] == {"MODEL_WEIGHTS_DIR": md, "HF_MODEL": md}
    assert v["model_dir_weights"] == f"{NEW}@{NEW_REV}"
    assert v["evidence"] == ["stages/7/verify/evidence/verify.json", "stages/7/verify/evidence/server.log"]
    seen = json.loads(boot["pid_file"].read_text())
    assert seen["env"]["HF_HUB_OFFLINE"] == "1" and seen["model_arg"] == md
    assert seen["env"]["TT_CACHE_PATH"] == str(boot["verify"] / "bundle" / ".tt_cache")
    assert wait_gone(seen["pgid"])
    # The staged package itself was never installed into or served.
    assert not (boot["out"] / "venv").exists() and not (boot["out"] / "model-dir").exists()


def test_the_boot_check_hf_home_holds_the_new_model_and_the_drafter_only(boot):
    hub = boot["verify"] / "hf" / "hub"
    assert sorted(p.name for p in hub.iterdir()) == ["models--Altworld--Hemmingway-1",
                                                     "models--incoai--Qwen3.8-27B-DFlash2"]
    assert boot["test"]["hf_linked"] == [NEW, DRAFTER] and boot["test"]["hf_missing"] == []
    assert (boot["hf_op"] / "hub" / "models--Qwen--Qwen3.8-27B").is_dir()     # present, not linked


def test_the_hardware_test_runs_the_copied_script_with_the_supervisors_python(boot):
    assert boot["test"]["command"] == f"{sys.executable} stages/7/verify/verify_bundle.py"
    assert boot["test"]["deadline_s"] > 0
    assert (boot["verify"] / "serve_and_compare.py").is_file()


def test_low_agreement_is_measured_and_reported(boot):
    r = boot["check"]("every4")
    assert r.returncode == 0, r.stdout + r.stderr
    assert verify_json(boot)["top1_agreement"] == 0.75


def test_a_copy_whose_run_sh_lost_the_weights_directory_is_caught(boot):
    run_sh = boot["verify"] / "bundle" / "run.sh"
    run_sh.write_text(run_sh.read_text().replace('export MODEL_WEIGHTS_DIR="$HERE/model-dir"\n', ""))
    r = boot["check"]("perfect")
    assert r.returncode == 8, r.stdout + r.stderr
    assert "MODEL_WEIGHTS_DIR" in r.stdout
    assert wait_gone(json.loads(boot["pid_file"].read_text())["pgid"])


def test_the_operators_own_weights_variables_cannot_stand_in_for_the_bundles(boot):
    # The check removes MODEL_WEIGHTS_DIR and HF_MODEL from its environment, so only the bundle's
    # run.sh can set them. Here they point at the right place and run.sh does not set one of them.
    md = str(boot["verify"] / "bundle" / "model-dir")
    run_sh = boot["verify"] / "bundle" / "run.sh"
    run_sh.write_text(run_sh.read_text().replace('export MODEL_WEIGHTS_DIR="$HERE/model-dir"\n', ""))
    r = boot["check"]("perfect", env_extra={"MODEL_WEIGHTS_DIR": md, "HF_MODEL": md})
    assert r.returncode == 8, r.stdout + r.stderr


def test_a_port_that_already_answers_is_refused_before_any_server_starts(boot):
    with socket.socket() as s:
        s.bind(("127.0.0.1", boot["cfg"]["port"]))
        s.listen()
        r = boot["check"]("perfect")
    assert r.returncode == 6, r.stdout + r.stderr
    assert not boot["pid_file"].exists()


def test_a_server_that_serves_another_model_is_refused(boot):
    r = boot["check"]("perfect", model="/somewhere/else/model-dir")
    assert r.returncode == 7, r.stdout + r.stderr


def test_a_model_dir_built_from_other_weights_is_refused(boot):
    prep = boot["verify"] / "bundle" / "prepare_model_dir.py"
    prep.write_text(prep.read_text().replace('f"{repo}@{rev}", encoding', '"Other/Model@x", encoding'))
    r = boot["check"]("perfect")
    assert r.returncode == 9, r.stdout + r.stderr


def test_a_failed_install_is_reported_with_its_log(tmp_path, monkeypatch):
    r = make_run(tmp_path, make_source(tmp_path / "models"))
    fb, _ = fake_bin(tmp_path)
    monkeypatch.setenv("PATH", f"{fb}:{os.environ['PATH']}")
    facts = read_run(r["run"])
    out = r["run"] / "stages/7/package/x"
    out.parent.mkdir(parents=True)
    stage_profile(Profile(2, facts.source, True), facts, out)
    env = {k: v for k, v in os.environ.items() if not k.startswith("FAKE_")}
    with pytest.raises(PackageError, match="install.sh(.|\n)*FAKE_TT_MODEL_PYTHON"):
        prepare_verify(out, r["run"] / "stages/7/verify", facts, env=env)


def test_auxiliary_repos_never_include_the_nearest_model():
    env = {"DFLASH_WEIGHTS": f"{DRAFTER}@{'d' * 40}", "OTHER": BASE, "ARCH_NAME": "blackhole",
           "X": "a/b/c", "Y": "1"}
    assert aux_repos(env, nearest_model=BASE) == [DRAFTER]
