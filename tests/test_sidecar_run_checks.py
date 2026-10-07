"""run_sidecar_checks.py (orchard/skills/sidecar-parity-templates/).

It is the one command of stage 2's hardware test on a `weights+sidecar` model. It runs serve_and_compare.py
and then the parity launcher as subprocesses, always both, and writes evidence/sidecar-check.json with the
`result_draft` that stage 2's result.json is copied from. Fakes stand in for both children: no device, no
server, no model."""
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "orchard" / "skills" / "sidecar-parity-templates" / "run_sidecar_checks.py"

SWAP_DRAFT = {"serves": True, "server_ready_s": 280.5, "coherent": True, "free_run_text": "hello world",
              "top1_agreement": 0.94, "n_tokens": 32, "cache_dir": "/cache",
              "evidence": ["stages/2/evidence/swap-check.json", "stages/2/evidence/server.log"]}
PARITY = {"measured": True, "n_records": 6, "n_questions": 17, "hidden_pcc_min": 0.99,
          "wiring_check": {"cosine": 0.9999, "passed": True}}


def fake_swap(code=0, draft=SWAP_DRAFT, write=True, extra=""):
    body = f"""import json, os, sys, time
from pathlib import Path
here = Path(__file__).resolve().parent
(here / "order.txt").open("a").write("swap start %f\\n" % time.time())
print("swap says hello", flush=True)
{extra}
if {write!r}:
    (here / "evidence").mkdir(exist_ok=True)
    (here / "evidence" / "swap-check.json").write_text(json.dumps({{"result_draft": {draft!r}}}))
(here / "order.txt").open("a").write("swap end %f\\n" % time.time())
sys.exit({code})
"""
    return body


def fake_parity(code=0, evidence=PARITY, extra=""):
    ev = json.dumps(evidence) if evidence is not None else ""
    return f"""#!/usr/bin/env bash
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
echo "parity start $(date +%s.%N) tt=${{TT_VISIBLE_DEVICES:-unset}}" >> "$HERE/order.txt"
echo "parity says hello"
{extra}
mkdir -p "$HERE/evidence"
if [ -n '{ev}' ]; then echo '{ev}' > "$HERE/evidence/sidecar-parity.json"; fi
exit {code}
"""


@pytest.fixture
def stage(tmp_path):
    run_dir = tmp_path / "run"
    stage = run_dir / "stages" / "2"
    stage.mkdir(parents=True)
    shutil.copy(SCRIPT, stage)
    (stage / "swap_config.json").write_text(json.dumps({"run_dir": str(run_dir)}))
    (stage / "serve_and_compare.py").write_text(fake_swap())
    parity = stage / "parity-run.sh"
    parity.write_text(fake_parity())
    parity.chmod(0o755)
    return stage


def install(stage, swap=None, parity=None):
    if swap is not None:
        (stage / "serve_and_compare.py").write_text(swap)
    if parity is not None:
        p = stage / "parity-run.sh"
        p.write_text(parity)
        p.chmod(0o755)


def run(stage, env=None, **kw):
    e = {**os.environ, **(env or {})}
    return subprocess.run([sys.executable, str(stage / "run_sidecar_checks.py")], cwd=stage, capture_output=True,
                          text=True, env=e, **kw)


def check(stage):
    return json.loads((stage / "evidence" / "sidecar-check.json").read_text())


# ---- both halves work ----------------------------------------------------------------------------

def test_both_ok_exits_0_and_merges_the_swap_draft_with_the_parity_result(stage):
    done = run(stage)
    assert done.returncode == 0, done.stderr
    c = check(stage)
    d = c["result_draft"]
    for key in ("serves", "server_ready_s", "coherent", "free_run_text", "top1_agreement", "n_tokens", "cache_dir"):
        assert d[key] == SWAP_DRAFT[key]
    assert d["sidecar_parity"] == PARITY
    assert "failure" not in c and c["swap_exit"] == 0 and c["parity_exit"] == 0


def test_the_evidence_list_holds_the_swap_paths_and_the_parity_file_relative_to_the_run_dir(stage):
    run(stage)
    ev = check(stage)["result_draft"]["evidence"]
    assert ev == SWAP_DRAFT["evidence"] + ["stages/2/evidence/sidecar-parity.json"]


def test_the_swap_runs_first_and_the_parity_check_after_it_has_ended(stage):
    run(stage)
    lines = (stage / "order.txt").read_text().splitlines()
    assert [l.split()[0] + " " + l.split()[1] for l in lines] == ["swap start", "swap end", "parity start"]
    assert float(lines[1].split()[2]) <= float(lines[2].split()[2])


def test_the_child_output_is_streamed_to_the_wrapper_and_kept_in_logs(stage):
    done = run(stage)
    assert "swap says hello" in done.stdout and "parity says hello" in done.stdout
    assert "swap says hello" in (stage / "evidence" / "swap-run.log").read_text()
    assert "parity says hello" in (stage / "evidence" / "parity-run.log").read_text()


def test_the_environment_reaches_the_children(stage):
    run(stage, env={"TT_VISIBLE_DEVICES": "0000:01:00.0,0000:02:00.0"})
    assert "tt=0000:01:00.0,0000:02:00.0" in (stage / "order.txt").read_text()


def test_seconds_are_recorded_for_each_half(stage):
    s = check(stage) if run(stage).returncode == 0 else None
    assert set(s["seconds"]) == {"swap", "parity"} and all(v >= 0 for v in s["seconds"].values())


def test_the_parity_result_is_copied_verbatim_whatever_it_holds(stage):
    odd = {**PARITY, "extra": {"nested": [1, 2, {"x": None}]}}
    install(stage, parity=fake_parity(evidence=odd))
    run(stage)
    assert check(stage)["result_draft"]["sidecar_parity"] == odd


# ---- one half fails --------------------------------------------------------------------------------

def test_a_failed_swap_still_runs_the_parity_check_and_exits_with_the_swap_code(stage):
    install(stage, swap=fake_swap(code=4, write=False))
    done = run(stage)
    assert done.returncode == 4
    c = check(stage)
    assert c["swap_exit"] == 4 and c["parity_exit"] == 0
    assert "swap" in c["failure"] and "4" in c["failure"]
    assert c["result_draft"]["sidecar_parity"] == PARITY and "serves" not in c["result_draft"]


def test_a_failed_parity_check_exits_with_its_code_and_keeps_what_it_wrote(stage):
    install(stage, parity=fake_parity(code=6))
    done = run(stage)
    assert done.returncode == 6
    c = check(stage)
    assert "parity" in c["failure"] and "6" in c["failure"]
    assert c["result_draft"]["serves"] is True and c["result_draft"]["sidecar_parity"] == PARITY


def test_a_parity_failure_without_evidence_leaves_no_sidecar_field(stage):
    install(stage, parity=fake_parity(code=5, evidence=None))
    done = run(stage)
    assert done.returncode == 5 and "sidecar_parity" not in check(stage)["result_draft"]


def test_both_failing_exits_with_the_swap_code_and_names_both(stage):
    install(stage, swap=fake_swap(code=3, write=False), parity=fake_parity(code=6, evidence=None))
    done = run(stage)
    assert done.returncode == 3
    f = check(stage)["failure"]
    assert "swap" in f and "parity" in f


def test_the_failure_text_quotes_the_end_of_the_childs_output(stage):
    install(stage, parity=fake_parity(code=6, extra='for i in $(seq 1 40); do echo "line $i"; done'))
    run(stage)
    f = check(stage)["failure"]
    assert "line 40" in f and "line 1\n" not in f


def test_a_swap_that_exits_0_but_wrote_no_evidence_is_a_failure(stage):
    install(stage, swap=fake_swap(code=0, write=False))
    done = run(stage)
    assert done.returncode != 0
    assert "swap-check.json" in check(stage)["failure"]


def test_a_missing_parity_launcher_is_a_failure_that_names_the_file(stage):
    (stage / "parity-run.sh").unlink()
    done = run(stage)
    assert done.returncode != 0 and "parity-run.sh" in check(stage)["failure"]


def test_a_missing_swap_script_is_a_failure_that_names_the_file(stage):
    (stage / "serve_and_compare.py").unlink()
    done = run(stage)
    assert done.returncode != 0 and "serve_and_compare.py" in check(stage)["failure"]


def test_a_malformed_swap_evidence_file_is_a_failure_not_a_crash(stage):
    install(stage, swap=fake_swap(code=0, write=False,
                                  extra="(here / 'evidence').mkdir(exist_ok=True)\n"
                                        "(here / 'evidence' / 'swap-check.json').write_text('{not json')"))
    done = run(stage)
    assert done.returncode != 0 and "swap-check.json" in check(stage)["failure"]


def test_a_missing_run_dir_in_the_swap_config_falls_back_to_the_stage_layout(stage):
    (stage / "swap_config.json").write_text("{}")
    run(stage)
    assert check(stage)["result_draft"]["evidence"][-1] == "stages/2/evidence/sidecar-parity.json"


# ---- cleaning up ---------------------------------------------------------------------------------

def test_a_term_signal_stops_the_childs_whole_process_group_and_still_writes_the_file(stage):
    pidfile = stage / "grandchild.pid"
    slow = fake_parity(extra=f'sleep 300 &\necho $! > "{pidfile}"\nsleep 300')
    install(stage, parity=slow)
    e = {**os.environ}
    p = subprocess.Popen([sys.executable, str(stage / "run_sidecar_checks.py")], cwd=stage, env=e,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 30
    while not pidfile.exists() and time.time() < deadline:
        time.sleep(0.1)
    assert pidfile.exists()
    grandchild = int(pidfile.read_text())
    p.send_signal(signal.SIGTERM)
    p.wait(timeout=60)
    assert p.returncode != 0
    time.sleep(0.5)
    with pytest.raises(ProcessLookupError):
        os.kill(grandchild, 0)
    assert "interrupted" in check(stage)["failure"]


def test_nothing_is_left_running_after_a_normal_run(stage):
    pidfile = stage / "bg.pid"
    install(stage, parity=fake_parity(extra=f'sleep 300 &\necho $! > "{pidfile}"'))
    run(stage)
    time.sleep(0.5)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


# ---- gaps the mutation run found -----------------------------------------------------------------------

def test_the_recorded_seconds_are_real_durations(stage):
    install(stage, parity=fake_parity(extra="sleep 0.4"))
    run(stage)
    assert check(stage)["seconds"]["parity"] >= 0.3


def test_a_missing_swap_script_is_named_as_missing_and_not_run(stage):
    (stage / "serve_and_compare.py").unlink()
    run(stage)
    assert "does not exist" in check(stage)["failure"]


def test_a_parity_check_that_exits_0_without_evidence_is_a_failure(stage):
    install(stage, parity=fake_parity(code=0, evidence=None))
    done = run(stage)
    assert done.returncode != 0 and "sidecar-parity.json" in check(stage)["failure"]


def test_a_swap_file_without_a_result_draft_is_a_failure(stage):
    body = fake_swap(code=0, write=False,
                     extra="(here / 'evidence').mkdir(exist_ok=True)\n"
                           "(here / 'evidence' / 'swap-check.json').write_text('{\"other\": 1}')")
    install(stage, swap=body)
    done = run(stage)
    assert done.returncode != 0 and "result_draft" in check(stage)["failure"]
