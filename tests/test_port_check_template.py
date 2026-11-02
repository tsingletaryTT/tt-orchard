# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The supervisor-owned port check (orchard/skills/full-port-templates/port_check.py, issue 325): it runs a fake port module on the CPU with
a fake chip, writes the record itself, and the gate refuses each way a port module (or anything else the coder wrote) tries to forge it."""
import json
import os
import re
import time

import pytest

from orchard import full_port
from orchard.stages import gate_full_port
from port_check_fakes import CHIPS, HONEST, REF_IDS, WRONG, make_run, run_template, write_result

MARK = re.compile(r"^PORT_CHECK_SHA256 ([0-9a-f]{64})$", re.M)


def gate(run, stage):
    return gate_full_port(stage, run)


def record(stage):
    return json.loads((stage / "evidence" / "port-check.json").read_text())


def one(g, word):
    assert not g.ok, "the gate passed a result it must refuse"
    return [r for r in g.reasons if word in r]


# ---- the honest run -----------------------------------------------------------------------------------

def test_an_honest_port_gets_a_record_the_template_wrote_and_the_gate_passes_it(tmp_path):
    run, stage, _ = make_run(tmp_path)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0, done.stdout + done.stderr
    rec = record(stage)
    assert rec["chip_ids"] == REF_IDS and rec["reference_ids"] == REF_IDS and rec["top1_agreement"] == 1.0
    assert rec["chips"] == CHIPS and rec["load_s"] > 0 and rec["template_sha256"] == full_port.template_sha256()
    assert rec["free_run_text"].startswith("w100 w101") and len(rec["free_run_ids"]) == 48
    assert [s["path"] for s in rec["port_sources"]] == ["stages/2/port/port_model.py"]
    assert MARK.fullmatch(done.stdout.strip().splitlines()[-1])
    write_result(stage)
    g = gate(run, stage)
    assert g.ok, g.reasons


def test_the_worker_gets_the_chips_the_supervisor_leased(tmp_path):
    # Two chips leased: the template opens a mesh over both. One chip: a single device.
    port = HONEST.replace('return {"dir": model_dir, "device": mesh_device}', "print('device', mesh_device)\n    return {}")
    run, stage, _ = make_run(tmp_path, port)
    assert "port| device {'mesh': (1, 2)}" in run_template(tmp_path, run, stage).stdout
    run, stage, _ = make_run(tmp_path / "one", port)
    assert "port| device {'device': 0}" in run_template(tmp_path / "one", run, stage, chips=CHIPS[:1]).stdout


def test_a_port_that_computes_the_wrong_ids_is_refused_on_agreement(tmp_path):
    run, stage, _ = make_run(tmp_path, WRONG)
    assert run_template(tmp_path, run, stage).returncode == 0
    write_result(stage)
    assert one(gate(run, stage), "below the minimum")


def test_a_port_that_raises_fails_the_run_and_leaves_no_record(tmp_path):
    run, stage, _ = make_run(tmp_path, HONEST.replace("return _peak(100)", "raise RuntimeError('no kernel for this layer')"))
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "no kernel for this layer" in done.stdout
    assert not (stage / "evidence" / "port-check.json").exists()


# ---- the forgeries ----------------------------------------------------------------------------------------

def test_a_port_that_prints_its_own_check_line_fails_the_run(tmp_path):
    forged = HONEST.replace("def load(model_dir, mesh_device):", "def load(model_dir, mesh_device):\n    print('PORT_CHECK_SHA256 ' + '0' * 64)")
    run, stage, _ = make_run(tmp_path, forged)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "mentions PORT_CHECK" in done.stdout
    assert not (stage / "evidence" / "port-check.json").exists()
    assert not MARK.search(done.stdout)                       # the forged line carries the "port| " prefix, and the template printed none


def test_a_check_line_that_reaches_the_output_unprefixed_is_refused_by_the_gate(tmp_path):
    # The template prefixes the port's lines. A line added to the output file afterwards (or written past the prefix) still fails the gate.
    run, stage, _ = make_run(tmp_path)
    run_template(tmp_path, run, stage)
    write_result(stage)
    out = stage / "evidence" / "hw-test-output.txt"
    out.write_text(out.read_text() + "PORT_CHECK_SHA256 " + "0" * 64 + "\n")
    marker = json.loads((stage / "test-result.json").read_text())
    marker["output"] = full_port.file_record(run, out)
    (stage / "test-result.json").write_text(json.dumps(marker))
    assert one(gate(run, stage), "begin with PORT_CHECK_SHA256")


def test_a_port_that_writes_the_record_itself_has_it_replaced_by_the_templates(tmp_path):
    forged = WRONG.replace("def load(model_dir, mesh_device):", '''def load(model_dir, mesh_device):
    import json, pathlib
    p = pathlib.Path("stages/2/evidence/port-check.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"schema": "orchard-port-check/1", "chip_ids": list(range(100, 132)), "top1_agreement": 1.0}))''')
    run, stage, _ = make_run(tmp_path, forged)
    assert run_template(tmp_path, run, stage).returncode == 0
    assert record(stage)["chip_ids"][1:] == [0] * 31 and record(stage)["top1_agreement"] < 0.1     # the real ids, not the forged ones
    write_result(stage)
    assert one(gate(run, stage), "below the minimum")


def test_a_record_rewritten_after_the_run_does_not_match_the_output_and_is_refused(tmp_path):
    run, stage, _ = make_run(tmp_path, WRONG)
    run_template(tmp_path, run, stage)
    rec = record(stage)
    rec["chip_ids"] = list(rec["reference_ids"])
    rec["top1_agreement"] = 1.0
    (stage / "evidence" / "port-check.json").write_text(json.dumps(rec))
    write_result(stage)
    assert one(gate(run, stage), "PORT_CHECK_SHA256")


def test_a_port_that_replaces_the_template_copy_is_refused(tmp_path):
    # The copy on disk after the run is not the shipped file, whatever ran.
    swap = HONEST.replace("def load(model_dir, mesh_device):", '''def load(model_dir, mesh_device):
    import pathlib
    pathlib.Path("stages/2/port_check.py").write_text("print('forged')\\n")''')
    run, stage, _ = make_run(tmp_path, swap)
    run_template(tmp_path, run, stage)
    write_result(stage)
    assert one(gate(run, stage), "differs from the shipped template")


def test_a_record_written_by_an_edited_template_is_refused(tmp_path):
    edited = full_port.TEMPLATE.read_text().replace("MAX_POSITIONS = 64", "MAX_POSITIONS = 8")
    assert edited != full_port.TEMPLATE.read_text()
    run, stage, _ = make_run(tmp_path, template_text=edited)
    run_template(tmp_path, run, stage)
    write_result(stage)
    g = gate(run, stage)
    assert one(g, "differs from the shipped template") and one(g, "template_sha256")


def test_a_missing_template_copy_is_refused(tmp_path):
    run, stage, _ = make_run(tmp_path)
    run_template(tmp_path, run, stage)
    write_result(stage)
    (stage / full_port.TEMPLATE_NAME).unlink()
    assert one(gate(run, stage), "is not in this stage directory")


def test_a_different_command_than_the_templates_is_refused(tmp_path):
    run, stage, _ = make_run(tmp_path)
    run_template(tmp_path, run, stage)
    write_result(stage)
    marker = json.loads((stage / "test-result.json").read_text())
    marker["command"] = "python3 stages/2/port/check.py"
    (stage / "test-result.json").write_text(json.dumps(marker))
    assert one(gate(run, stage), "command for this stage")


def test_a_config_changed_after_the_test_is_refused(tmp_path):
    run, stage, _ = make_run(tmp_path)
    run_template(tmp_path, run, stage)
    write_result(stage)
    (stage / full_port.CONFIG_NAME).write_text(json.dumps({"model_snapshot": "/elsewhere"}))
    assert one(gate(run, stage), full_port.CONFIG_NAME)


def test_a_port_file_added_after_the_record_is_refused(tmp_path):
    run, stage, _ = make_run(tmp_path)
    run_template(tmp_path, run, stage)
    write_result(stage)
    (stage / "port" / "late.py").write_text("import ttnn\n")
    assert one(gate(run, stage), "late.py")


SITECUSTOMIZE = "import pathlib\npathlib.Path('SITECUSTOMIZE-RAN').write_text('yes')\n"


@pytest.mark.parametrize("where", ["sitecustomize.py", "port/sitecustomize.py", "json.py", "hashlib.py"])
def test_a_file_the_coder_leaves_next_to_the_template_or_in_the_port_directory_is_not_imported_at_startup(tmp_path, where):
    # `python3 -P -s` keeps stages/2 off sys.path, and the worker puts the port directory on it only after it has started,
    # so a sitecustomize.py or a module that shadows a standard one is never loaded as a startup hook.
    run, stage, _ = make_run(tmp_path, extra_files={where: SITECUSTOMIZE})
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0, done.stdout + done.stderr
    assert not (run / "SITECUSTOMIZE-RAN").exists()


def test_a_port_that_patches_the_templates_functions_and_hooks_imports_cannot_change_the_record(tmp_path):
    # Inside the worker the port can reach this file as __main__ and install an import hook. The record is made in the parent process,
    # which none of it reaches: the stamps and the sha256 are the parent's, and the counting is the parent's over the ids it was sent.
    patch = '''
import sys, json, hashlib
main = sys.modules["__main__"]
main.sha_bytes = lambda data: "0" * 64
main.sha_file = lambda path: "0" * 64
main.PREFIX = ""
class Hook:
    def find_spec(self, name, path=None, target=None):
        return None
sys.meta_path.insert(0, Hook())
main.main = lambda: 0
main.free_run_text = lambda d, ids: "forged forged forged forged forged"
'''
    run, stage, _ = make_run(tmp_path, patch + WRONG)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0, done.stdout + done.stderr
    rec = record(stage)
    assert rec["template_sha256"] == full_port.template_sha256() and rec["top1_agreement"] < 0.1
    assert MARK.fullmatch(done.stdout.strip().splitlines()[-1])
    write_result(stage)
    assert one(gate(run, stage), "below the minimum") and not one(gate(run, stage), "template")


LATE_WRITER = '''
def close(model):
    import threading, time, json, pathlib
    def later():
        rec = pathlib.Path("stages/2/evidence/port-check.json")
        for _ in range(300):                 # wait up to 3 s for the template's record, then overwrite it
            if rec.exists():
                rec.write_text(json.dumps({"chip_ids": [0] * 32, "top1_agreement": 0.0}))
                return
            time.sleep(0.01)
    threading.Thread(target=later).start()
'''


def test_a_port_thread_that_writes_the_record_after_close_is_waited_for_and_overwritten(tmp_path):
    # A thread the port leaves running after `close` rewrites the record a moment later. The template waits for the worker to exit
    # before it writes the record, so its own record is the last write and the gate passes the real ids (not the forged ones).
    run, stage, _ = make_run(tmp_path, HONEST + LATE_WRITER)
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 0, done.stdout + done.stderr
    time.sleep(0.5)                          # a writer that was not waited for would have struck by now
    assert record(stage)["chip_ids"] == REF_IDS
    write_result(stage)
    g = gate(run, stage)
    assert g.ok, g.reasons


# ---- review findings (PR 347) ---------------------------------------------------------------------------

def test_a_link_under_the_port_directory_fails_the_run_and_the_gate(tmp_path):
    run, stage, _ = make_run(tmp_path)
    (tmp_path / "elsewhere.py").write_text("import ttnn\n")
    os.symlink(tmp_path / "elsewhere.py", stage / "port" / "layers.py")
    done = run_template(tmp_path, run, stage)
    assert done.returncode == 2 and "is a link" in done.stdout
    # The gate refuses a record made before the link existed, too.
    run2, stage2, _ = make_run(tmp_path / "b")
    run_template(tmp_path / "b", run2, stage2)
    write_result(stage2)
    os.symlink(tmp_path / "elsewhere.py", stage2 / "port" / "layers.py")
    assert one(gate(run2, stage2), "is a link")


def test_a_directory_link_under_the_port_directory_is_refused_by_the_gate(tmp_path):
    run, stage, _ = make_run(tmp_path)
    run_template(tmp_path, run, stage)
    write_result(stage)
    (tmp_path / "pkg").mkdir()
    os.symlink(tmp_path / "pkg", stage / "port" / "pkg")
    assert one(gate(run, stage), "is a link")


def test_a_process_the_port_starts_is_killed_before_the_run_ends(tmp_path):
    spawn = HONEST.replace("def load(model_dir, mesh_device):", '''def load(model_dir, mesh_device):
    import subprocess
    p = subprocess.Popen(["sleep", "60"])
    open("CHILD-PID", "w").write(str(p.pid))''')
    run, stage, _ = make_run(tmp_path, spawn)
    assert run_template(tmp_path, run, stage).returncode == 0
    pid = int((run / "CHILD-PID").read_text())
    for _ in range(100):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    else:
        os.kill(pid, 9)
        pytest.fail("a process the port started was still running after the check ended")


def test_several_chips_get_the_fabric_turned_on_and_off_around_the_mesh_and_one_chip_gets_none(tmp_path):
    run, stage, _ = make_run(tmp_path)
    out = run_template(tmp_path, run, stage).stdout
    assert out.index("port| fabric fabric-1d") < out.index("port| fabric disabled")
    run, stage, _ = make_run(tmp_path / "one")
    assert "fabric" not in run_template(tmp_path / "one", run, stage, chips=CHIPS[:1]).stdout


def test_load_s_times_the_ports_load_and_not_the_device_open_or_the_module_import(tmp_path):
    slow_import = "import time\ntime.sleep(1.5)\n" + HONEST
    run, stage, _ = make_run(tmp_path, slow_import)
    assert run_template(tmp_path, run, stage).returncode == 0
    assert record(stage)["load_s"] < 1.0
