# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The supervisor runs stage 7 on the weights-only path when asked for a package (plan 5).

A scripted bring-up walks stages 0 to 8 against the fake two-board machine. Its stage 1 writes the
reference token ids, and its stage 2 prepare step writes the swap_config.json and model-dir config
that the weights-swap-check skill would leave. Stage 7 then runs as supervisor code: the fake
tt-model stages the package, the copy's install.sh builds a venv whose python serves
tests/fake_swap_server.py, and verify_bundle.py runs as the stage's hardware test under a lease.
"""
import json
import os
import socket
import sys
from pathlib import Path

import pytest

from fake_model import FakeModel
from fakes import Crash
from orchard.ledger import Ledger
from orchard.supervisor import EXIT_READY, EXIT_REFUSED, build, main, parse
from package_fakes import (BASE, BASE_REV, DRAFTER, DRAFTER_REV, GENERATED, HAVE_TOKENIZERS, NEW,
                           NEW_REV, PROMPT_IDS, SOURCE_ENV, VOCAB, calls, fake_bin, hf_snapshot,
                           make_source, tokenizer_json)
from run_fakes import (BOARDS, FILES, CrashingLedger, FakeContainers, Machine, MachineAdapter,
                       MachineCoder, argv, bringup, clock, hw, plenty, write_tiers)

if not HAVE_TOKENIZERS:
    pytest.skip("SKIPPED: the `tokenizers` package is not importable, so the stage 7 boot check "
                "did not run. verify_bundle.py needs it.", allow_module_level=True)

FAKE_SERVER = Path(__file__).resolve().with_name("fake_swap_server.py")
pytestmark = pytest.mark.usefixtures("stub_tools")


class Paused(Exception):
    """The run paused for the operator; the test ends here."""


class PackageRig:
    """The machine, the model servers, the installed bundles, the HF caches and the run options."""

    def __init__(self, tmp, monkeypatch, *, chips=2, source_env=None):
        self.tmp, self.m = tmp, Machine()
        models = tmp / "models"
        self.source = make_source(models, env=source_env)
        hf_run, hf_op = tmp / "hf-run", tmp / "hf-operator"
        snap = hf_snapshot(hf_run, NEW, NEW_REV, {
            "README.md": f"---\nlicense: cc-by-nc-4.0\nbase_model:\n- {BASE}\n---\n",
            "tokenizer.json": tokenizer_json(), "model-00001-of-00001.safetensors": "new"})
        hf_snapshot(hf_op, DRAFTER, DRAFTER_REV, {"model.safetensors": "drafter"})
        hf_snapshot(hf_op, BASE, BASE_REV, {"model-00001-of-00001.safetensors": "base"})
        self.run_dir = tmp / "run"
        swap = {"run_dir": str(self.run_dir), "bundle_dir": str(self.source),
                "nearest_model_id": BASE, "new_snapshot": str(snap), "new_model_id": NEW,
                "hf_home": str(hf_op), "port": 8100}
        # Stage 2's test writes swap-check.json as serve_and_compare.py does: the short label the
        # agent typed and the model-dir it served, whose weight file links to the new model's blob
        # (prepare_swap.py links each file to the realpath of the snapshot file).
        weights = "model-00001-of-00001.safetensors"
        md = self.run_dir / "stages/2/model-dir"
        report = json.dumps({"new_model_id": NEW, "model_dir": str(md)})
        test2 = (f"ln -s {os.path.realpath(snap / weights)} {md / weights} && "
                 f"printf '%s' '{report}' > stages/2/evidence/swap-check.json && "
                 "printenv > stages/2/evidence/devices.txt")
        self.files = {
            (0, "run"): {**FILES[(0, "run")], "delta.json": {
                **FILES[(0, "run")]["delta.json"], "model": f"{NEW}@{NEW_REV}",
                "nearest_model": f"{BASE}@{BASE_REV}"}},
            (1, "run"): {**FILES[(1, "run")],
                         "evidence/reference/prompt-ids.json": {"prompt_ids": PROMPT_IDS},
                         "evidence/reference/generated-ids.json": {"generated_ids": GENERATED}},
            (2, "prepare"): {**hw(2), "swap_config.json": swap,
                             "hw_test.json": {**hw(2)["hw_test.json"], "command": test2},
                             "model-dir/config.json": '{"architectures": ["Qwen3_5ForConditionalGeneration"]}',
                             "model-dir/preprocessor_config.json": "{}"}}
        self.pid_file, server_cfg = tmp / "server-pid.json", tmp / "server.json"
        server_cfg.write_text(json.dumps({"mode": "perfect", "vocab": VOCAB, "prompt_ids": PROMPT_IDS,
                                          "generated_ids": GENERATED, "pid_file": str(self.pid_file)}))
        fb, self.tt_log = fake_bin(tmp)
        monkeypatch.setenv("PATH", f"{fb}:{os.environ['PATH']}")
        self.chip_server = FakeModel(lambda r: bringup(r, overrides=self.files), models=["qwen-27b"])
        self.cpu_server = FakeModel(bringup, models=["cpu-model"])
        self.chip_server.__enter__()
        self.cpu_server.__enter__()
        tiers = write_tiers(tmp / "tiers.toml", self.chip_server.endpoint, self.cpu_server.endpoint)
        self.argv = argv(tmp, tiers, self.chip_server.endpoint, chips=chips) + [
            "--package-format", "v6", "--package-namespace", "episod",
            "--package-models-root", str(models),
            "--env", f"FAKE_TT_MODEL_PYTHON={sys.executable}",
            "--env", f"FAKE_SWAP_SERVER={FAKE_SERVER}", "--env", f"FAKE_SWAP_CONFIG={server_cfg}"]
        self.clock = clock()

    def sleep(self, s):
        self.clock.sleep(s)
        if any(e["event"] == "decision" and e["data"].get("decision") == "pause"
               for e in self.led.read()):
            raise Paused()

    def run(self, extra=(), pid=100, crash_if=None):
        path = self.run_dir / "ledger.jsonl"
        self.m.owner_pid = pid
        with (CrashingLedger(path, crash_if=crash_if) if crash_if else Ledger(path)) as led:
            self.led = led
            return build(parse(self.argv + list(extra)), led,
                         adapter=MachineAdapter(self.m, owner_pid=pid),
                         coder=MachineCoder(self.m), versions={"tt_model": "test"},
                         clock=self.clock, sleep=self.sleep, disk_usage=plenty,
                         home=self.tmp / "operator-home", containers=FakeContainers()).run()

    def entries(self):
        if not (self.run_dir / "ledger.jsonl").exists():
            return []
        with Ledger(self.run_dir / "ledger.jsonl") as led:
            return led.read()

    def close(self):
        self.chip_server.__exit__()
        self.cpu_server.__exit__()
        if self.pid_file.exists():
            try:
                os.killpg(json.loads(self.pid_file.read_text())["pgid"], 9)
            except ProcessLookupError:
                pass


@pytest.fixture
def rig(tmp_path, monkeypatch):
    made = []

    def make(**kw):
        r = PackageRig(tmp_path, monkeypatch, **kw)
        made.append(r)
        return r
    yield make
    for r in made:
        r.close()


def seq(entries, n):
    return [(e["event"], e["data"].get("step") or e["data"].get("decision") or e["data"].get("what"))
            for e in entries if e["stage"] == n]


def test_a_weights_only_run_with_a_package_format_stages_and_boot_checks_in_stage_7(rig):
    r = rig(chips=2)
    assert r.run() == EXIT_READY
    es = r.entries()
    start = next(e for e in es if e["event"] == "run_start")
    assert start["data"]["package"] == {"format": "v6", "namespace": "episod",
                                        "models_root": str(r.tmp / "models")}
    end7 = [e["data"] for e in es if e["event"] == "stage_end" and e["stage"] == 7]
    assert [d["result"] for d in end7] == ["pass"]
    s7 = seq(es, 7)
    assert ("evidence", "package staged") in s7
    assert s7.index(("evidence", "package staged")) < s7.index(("decision", "hardware test started"))
    lease = next(e["data"]["test_lease"] for e in es
                 if e["stage"] == 7 and e["data"].get("decision") == "test lease taken")
    assert lease["chips"] == list(BOARDS["B1"])           # the free board; the coder stays loaded
    assert not [e for e in es if e["event"] == "escalate" and e["stage"] == 7]
    verify = json.loads((r.run_dir / "stages/7/verify/evidence/verify.json").read_text())
    assert verify["top1_agreement"] == 1.0
    assert verify["served_model"] == str(r.run_dir / "stages/7/verify/bundle/model-dir")
    bundle = r.run_dir / "stages/8/bundle/package"
    assert sorted(p.name for p in bundle.iterdir()) == ["PUBLISH_COMMANDS.txt", "hemmingway-1-p300",
                                                       "hemmingway-1-p300-README.md", "package.json"]
    # The only tt-model calls were package-thin with --out; the stub tools (hf, git, gh, docker,
    # curl and the rest) never ran; the fake server was stopped.
    assert calls(r.tt_log) and all(c[0] == "package-thin" and "--out" in c for c in calls(r.tt_log))
    assert not (r.tmp / "stub-bin" / "called.log").exists()
    assert not r.m.leases and not r.m.coder_running


def test_with_the_coder_on_four_chips_stage_7_parks_it_for_the_boot_check(rig):
    r = rig(chips=4)
    assert r.run() == EXIT_READY
    s7 = seq(r.entries(), 7)
    test = s7.index(("decision", "hardware test started"))
    assert s7.index(("park", "reset")) < test < s7.index(("restore", "reset"))
    assert ("restore", "resumed") in s7[test:]


def test_a_scrub_hit_pauses_the_run_at_stage_7_without_escalating_or_leasing(rig):
    r = rig(chips=2, source_env={**SOURCE_ENV, "BUILD_HOST": socket.gethostname()})
    with pytest.raises(Paused):
        r.run()
    es = r.entries()
    end7 = [e["data"] for e in es if e["event"] == "stage_end" and e["stage"] == 7]
    assert [d["result"] for d in end7] == ["fail"]
    assert "scrub" in end7[0]["reasons"][0]
    pause = [e["data"]["reason"] for e in es if e["data"].get("decision") == "pause"]
    assert pause and "not escalated" in pause[-1]
    assert not [e for e in es if e["stage"] == 7 and e["event"] == "escalate"]
    assert not [e for e in es if e["stage"] == 7 and e["data"].get("decision") == "test lease taken"]


def test_without_a_package_format_stage_7_is_skipped(rig):
    r = rig(chips=2)
    r.argv = r.argv[:r.argv.index("--package-format")] + r.argv[r.argv.index("--package-models-root") + 2:]
    assert r.run() == EXIT_READY
    assert [e["data"]["result"] for e in r.entries()
            if e["event"] == "stage_end" and e["stage"] == 7] == ["skipped"]
    assert calls(r.tt_log) == []


def test_v5_1_is_refused_at_start_and_says_why(tmp_path, capsys):
    tiers = write_tiers(tmp_path / "t.toml", "http://127.0.0.1:8000/v1", "http://127.0.0.1:11434/v1")
    a = argv(tmp_path, tiers, "http://127.0.0.1:8000/v1") + [
        "--package-format", "v5.1", "--package-namespace", "episod", "--gozer", "/nonexistent"]
    assert main(a, home=tmp_path / "operator-home") == EXIT_REFUSED
    err = capsys.readouterr().err
    assert "v5.1 is deferred" in err and "image build" in err
    assert not (tmp_path / "run" / "ledger.jsonl").exists()


def test_a_package_format_needs_a_namespace(tmp_path, capsys):
    tiers = write_tiers(tmp_path / "t.toml", "http://127.0.0.1:8000/v1", "http://127.0.0.1:11434/v1")
    a = argv(tmp_path, tiers, "http://127.0.0.1:8000/v1") + ["--package-format", "v6",
                                                             "--gozer", "/nonexistent"]
    assert main(a, home=tmp_path / "operator-home") == EXIT_REFUSED
    assert "--package-namespace" in capsys.readouterr().err


def test_a_resumed_run_keeps_its_package_options(rig):
    r = rig(chips=2)
    assert r.run() == EXIT_READY
    with pytest.raises(ValueError, match="has package options"):
        r.run(extra=["--package-namespace", "someone-else"])


def test_a_run_started_without_package_options_can_get_them_before_stage_7(rig):
    r = rig(chips=2)
    plain = r.argv[:r.argv.index("--package-format")] + r.argv[r.argv.index("--package-models-root") + 2:]
    full, r.argv = r.argv, plain
    with pytest.raises(Crash):
        r.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 2)
    r.argv = full
    assert r.run(pid=200) == EXIT_READY
    es = r.entries()
    assert next(e for e in es if e["event"] == "run_start")["data"]["package"] is None
    added = [e for e in es if e["data"].get("decision") == "package options set"]
    assert len(added) == 1 and added[0]["data"]["package"]["namespace"] == "episod"
    assert [e["data"]["result"] for e in es if e["event"] == "stage_end" and e["stage"] == 7] == ["pass"]


def test_package_options_cannot_be_added_once_stage_7_has_started(tmp_path):
    tiers = write_tiers(tmp_path / "t.toml", "http://127.0.0.1:8000/v1", "http://127.0.0.1:11434/v1")
    base = argv(tmp_path, tiers, "http://127.0.0.1:8000/v1")
    with Ledger(tmp_path / "run" / "ledger.jsonl") as led:
        led.append("run_start", None, model=NEW, versions={}, inputs={}, required_chips=None,
                   package=None)
        led.append("stage_start", 7, skip=True)
        with pytest.raises(ValueError, match="only before stage 7 starts"):
            build(parse(base + ["--package-format", "v6", "--package-namespace", "episod"]), led,
                  adapter=MachineAdapter(Machine()), coder=MachineCoder(Machine()),
                  versions={"tt_model": "test"}, home=tmp_path / "operator-home")


def test_without_package_models_root_stage_7_looks_in_the_runs_tt_model_root(rig):
    # The models root follows the run's recorded paths (orchard/paths.py), so --operator-home moves
    # it with the other tt-model paths.
    r = rig(chips=2)
    i = r.argv.index("--package-models-root")
    r.argv = r.argv[:i] + r.argv[i + 2:] + ["--operator-home", str(r.tmp / "op")]
    # No bundle is installed there, so stage 7 stages the required profile (stage 2's bundle) only.
    assert r.run() == EXIT_READY
    es = r.entries()
    start = next(e["data"] for e in es if e["event"] == "run_start")
    want = str(r.tmp / "op" / ".cache" / "tt-model" / "models")
    assert start["paths"]["tt_model_root"] == want and start["package"]["models_root"] == want
    names = [p["name"] for p in json.loads((r.run_dir / "stages/7/package.json").read_text())["profiles"]]
    assert names == ["hemmingway-1-p300"]


def test_a_run_paused_in_an_old_stage_5_attempt_resumes_into_stage_7(rig):
    """Run 3's state on 2026-10-03: stages 0 to 4 passed, older code started stage 5 with an agent
    step, and the operator paused the run. Restarted on this code with the package options, the
    open stage 5 is recorded as skipped, stage 6 too, and stage 7 packages the model."""
    from orchard.supervisor import Control
    r = rig(chips=2)
    plain = r.argv[:r.argv.index("--package-format")] + r.argv[r.argv.index("--package-models-root") + 2:]
    full, r.argv = r.argv, plain
    with pytest.raises(Crash):
        r.run(crash_if=lambda e: e["event"] == "stage_end" and e["stage"] == 4)
    with Ledger(r.run_dir / "ledger.jsonl") as led:      # what the older code and the operator left
        led.append("stage_start", 5, escalated=False, resumed=False)
        led.append("decision", 5, decision="pause", reason="operator")
    (r.run_dir / "stages" / "5" / "evidence").mkdir(parents=True)
    Control(r.run_dir).write("resume")
    r.sleep = r.clock.sleep                  # this test expects the recorded pause; do not stop on it
    r.argv = full
    assert r.run(pid=200) == EXIT_READY
    es = r.entries()
    ends = {n: [e["data"]["result"] for e in es if e["event"] == "stage_end" and e["stage"] == n]
            for n in (5, 6, 7)}
    assert ends == {5: ["skipped"], 6: ["skipped"], 7: ["pass"]}
    assert len([e for e in es if e["data"].get("decision") == "package options set"]) == 1
    assert not [e for e in es if e["stage"] in (5, 6) and e["data"].get("decision") == "agent step"]
