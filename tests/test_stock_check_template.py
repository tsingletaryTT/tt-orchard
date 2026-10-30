# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/skills/stock-runtime-templates/stock_check.py on a fake host tt-metal tree whose demo is a small pytest file that prints what
tt-transformers' demo prints. No chip and no network. The real demo's log lines were read from simple_text_demo.py (tt-metal v0.80.0);
the first hardware run (docs/run-logs) is what shows they match."""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from orchard import stages
from orchard.stages import gate_stock_runtime
from test_stock_runtime_gates import DELTA

TEMPLATE = stages.SHIPPED_TEMPLATES["stock_check.py"]

FAKE_DEMO = '''
import os, pathlib
import pytest

@pytest.mark.parametrize("opt", ["performance", "accuracy"])
@pytest.mark.parametrize("case", ["batch-1", "ci-token-matching"])
def test_demo_text(opt, case):
    pathlib.Path("seen-" + opt + "-" + case + ".env").write_text("\\n".join(
        f"{k}={os.environ.get(k)}" for k in ("HF_MODEL", "HF_HUB_OFFLINE", "MESH_DEVICE", "PYTHONPATH", "TT_CACHE_PATH")) + "\\n" + os.getcwd())
    if pathlib.Path("fake-fail").exists() or (case == "batch-1" and pathlib.Path("fake-fail-free").exists()):
        raise SystemExit("device open failed")
    if case == "ci-token-matching":
        for i in range(500):
            print(f"Iteration {i}: 25ms @ 40.0 tok/s/user (40.0 tok/s throughput)")
        print(" Top1 Accuracy: 91.20%, Top5 Accuracy: 98.40%")
    else:
        for i in range(200):
            print(f"2026-10-26 10:00:00.000 | INFO     | demo:run:1 - Iteration {i}: 25ms @ 40.0 tok/s/user (40.0 tok/s throughput)")
        print("2026-10-26 10:00:01.000 | INFO     | demo:run:2 - Finished decoding, printing the final outputs...\\n")
        print("2026-10-26 10:00:01.001 | INFO     | demo:run:3 - \\n==REPEAT BATCH 0\\n==USER 0 - PROMPT\\nWhat is the capital of France? \\n"
              "==USER 0 - OUTPUT\\nParis is the capital of France.\\n\\nIt sits on the Seine and has been a centre of art for centuries.\\n")
        print("2026-10-26 10:00:02.000 | INFO     | demo:run:4 - Average Time to First Token (TTFT): 20.5ms")
        print("2026-10-26 10:00:02.001 | INFO     | demo:run:5 - Average speed: 25.0ms @ 41.5 tok/s/user (41.5 tok/s throughput)")
    if pathlib.Path("fake-fetch").exists():
        print("Fetching 28 files: 100%|")
'''


class Rig:
    def __init__(self, tmp_path, *, refpt=True, snapshot_config=None):
        self.tmp = tmp_path
        self.host = tmp_path / "tt-metal"
        demo = self.host / "models/tt_transformers/demo"
        demo.mkdir(parents=True)
        (demo / "simple_text_demo.py").write_text(FAKE_DEMO)
        ref = self.host / "models/tt_transformers/tests/reference_outputs"
        ref.mkdir(parents=True)
        if refpt:
            (ref / "Llama-3.1-8B-Instruct.refpt").write_bytes(b"refpt")
        gen = self.host / "models/tt_transformers/tests/generate_reference_hf.py"
        gen.write_text("import sys, pathlib\nout = sys.argv[sys.argv.index('--output_file') + 1]\npathlib.Path(out).write_bytes(b'made')\n")
        (self.host / "tests/scripts").mkdir(parents=True)
        (self.host / "tests/__init__.py").write_text("")
        (self.host / "tests/scripts/common.py").write_text("X = 1\n")
        (self.host / "conftest.py").write_text("")
        (self.host / "pytest.ini").write_text("[pytest]\n")
        (self.host / "models/__pycache__").mkdir()
        (self.host / "models/__pycache__/x.pyc").write_bytes(b"x")
        for args in (["init", "-q"], ["add", "-A"], ["-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-q", "-m", "x"],
                     ["tag", "v0.80.0-test"]):
            subprocess.run(["git", "-C", str(self.host), *args], check=True, capture_output=True)
        self.snap = tmp_path / "snap"
        self.snap.mkdir()
        (self.snap / "config.json").write_text(json.dumps(snapshot_config or {"_name_or_path": "x/y", "hidden_size": 8}))
        (self.snap / "model.safetensors").write_bytes(b"weights")
        (self.snap / "tokenizer.json").write_text("{}")
        self.run_dir = tmp_path / "run"
        self.stage = self.run_dir / "stages" / "2"
        self.stage.mkdir(parents=True)
        self.cfg = {"run_dir": str(self.run_dir), "model_id": "unsloth/Llama-3.1-8B-Instruct", "model_name": "Llama-3.1-8B-Instruct",
                    "model_snapshot": str(self.snap), "tt_metal_dir": str(self.host), "scratch_dir": str(self.run_dir / "tt-metal-scratch"),
                    "mesh_device": "P300", "chips": 2, "optimizations": "accuracy", "tt_cache": str(tmp_path / "cache" / "P300"),
                    "deadline_s": 600, "import_check": []}
        self.write_cfg()
        (self.stage / "stock_check.py").write_text(TEMPLATE.read_text())

    def write_cfg(self):
        (self.stage / "stock_config.json").write_text(json.dumps(self.cfg))

    def go(self, command):
        return subprocess.run([sys.executable, str(self.stage / "stock_check.py"), command], cwd=self.run_dir, capture_output=True, text=True,
                              env={"PATH": os.environ["PATH"], "HOME": str(self.tmp)})

    def host_digest(self):
        h = hashlib.sha256()
        for p in sorted(self.host.rglob("*")):
            if p.is_file():
                h.update(str(p.relative_to(self.host)).encode() + p.read_bytes())
        return h.hexdigest()

    def evidence(self):
        return json.loads((self.stage / "evidence" / "stock-check.json").read_text())


@pytest.fixture
def rig(tmp_path):
    return Rig(tmp_path)


# ---- prepare ------------------------------------------------------------------------------------------------------

def test_prepare_copies_the_host_models_into_the_scratch_directory_and_never_writes_the_host(rig):
    before = rig.host_digest()
    done = rig.go("prepare")
    assert done.returncode == 0, done.stderr
    sc = rig.run_dir / "tt-metal-scratch"
    assert (sc / "models/tt_transformers/demo/simple_text_demo.py").read_text() == FAKE_DEMO
    assert (sc / "conftest.py").is_file() and (sc / "pytest.ini").is_file()
    assert not (sc / "models/__pycache__").exists()
    assert rig.host_digest() == before


def test_prepare_also_copies_the_package_the_root_conftest_imports(rig):
    rig.go("prepare")
    sc = rig.run_dir / "tt-metal-scratch"
    assert (sc / "tests/__init__.py").is_file() and (sc / "tests/scripts/common.py").read_text() == "X = 1\n"


def test_prepare_refuses_an_interpreter_that_cannot_import_a_named_module_before_anything_else(rig):
    rig.cfg["import_check"] = ["json", "no_such_module_for_this_test"]
    rig.write_cfg()
    done = rig.go("prepare")
    assert done.returncode == 2 and "cannot import no_such_module_for_this_test" in done.stderr and "model was not tried" in done.stderr
    assert not (rig.run_dir / "tt-metal-scratch").exists()


def test_prepare_passes_when_every_named_module_imports(rig):
    rig.cfg["import_check"] = ["json", "os"]
    rig.write_cfg()
    assert rig.go("prepare").returncode == 0


def test_a_directory_in_python_path_extra_is_used_by_the_import_check_and_the_run(rig):
    extra = rig.tmp / "extra"
    extra.mkdir()
    (extra / "only_here_for_the_test.py").write_text("V = 1\n")
    rig.cfg["import_check"] = ["only_here_for_the_test"]
    rig.write_cfg()
    assert rig.go("prepare").returncode == 2                         # not importable without the directory
    rig.cfg["python_path_extra"] = [str(extra)]
    rig.write_cfg()
    assert rig.go("prepare").returncode == 0
    rig.go("run")
    sc = rig.run_dir / "tt-metal-scratch"
    seen = (sc / "seen-accuracy-ci-token-matching.env").read_text().splitlines()
    assert seen[3] == f"PYTHONPATH={sc}{os.pathsep}{extra}"


@pytest.mark.parametrize("key,value,message", [("python_path_extra", "x", "list of strings"), ("python_path_extra", ["rel"], "absolute"),
                                               ("import_check", [3], "list of strings"), ("import_check", ["a b"], "module names")])
def test_a_bad_optional_key_is_exit_2(rig, key, value, message):
    rig.cfg[key] = value
    rig.write_cfg()
    done = rig.go("prepare")
    assert done.returncode == 2 and message in done.stderr


def test_prepare_builds_the_model_directory_named_after_the_tt_transformers_model(rig):
    rig.go("prepare")
    d = rig.run_dir / "tt-metal-scratch/hf-models/Llama-3.1-8B-Instruct"
    assert (d / "model.safetensors").is_symlink() and (d / "model.safetensors").read_bytes() == b"weights"
    assert (d / "tokenizer.json").is_symlink()
    cfg = json.loads((d / "config.json").read_text())
    assert not (d / "config.json").is_symlink() and "_name_or_path" not in cfg and cfg["hidden_size"] == 8
    assert json.loads((rig.snap / "config.json").read_text())["_name_or_path"] == "x/y"          # the snapshot's own file is untouched


def test_prepare_writes_the_files_the_supervisor_reads(rig):
    rig.go("prepare")
    test = json.loads((rig.stage / "hw_test.json").read_text())
    assert test == {"command": "python3 stages/2/stock_check.py run", "deadline_s": 600}
    note = json.loads((rig.stage / "handoff.json").read_text())
    assert note["stage"] == 2 and note["goal"] and note["next_action"] and note["check_on_return"] and note["evidence"]


def test_a_second_prepare_keeps_the_scratch_copy_and_a_stages_changes(rig):
    rig.go("prepare")
    edited = rig.run_dir / "tt-metal-scratch/models/tt_transformers/demo/simple_text_demo.py"
    edited.write_text("# changed by a stage\n")
    done = rig.go("prepare")
    assert done.returncode == 0 and edited.read_text() == "# changed by a stage\n"


def test_a_directory_that_is_not_a_scratch_copy_is_refused(rig):
    sc = rig.run_dir / "tt-metal-scratch"
    sc.mkdir()
    (sc / "mine.txt").write_text("keep")
    done = rig.go("prepare")
    assert done.returncode == 2 and "not a scratch copy" in done.stderr and (sc / "mine.txt").read_text() == "keep"


def ref_path(r):
    return r.run_dir / "tt-metal-scratch/models/tt_transformers/tests/reference_outputs/Llama-3.1-8B-Instruct.refpt"


def test_the_reference_is_always_made_from_the_runs_snapshot_even_when_the_tree_has_a_file_of_that_name(tmp_path):
    # the tree's file belongs to the model the tree lists (any org's copy shares its name), not to this snapshot (review of PR 78)
    r = Rig(tmp_path)
    done = r.go("prepare")
    assert done.returncode == 0 and "generated" in done.stdout
    assert ref_path(r).read_bytes() == b"made"
    side = json.loads(ref_path(r).with_name(ref_path(r).name + ".orchard.json").read_text())
    assert side == {"model_id": "unsloth/Llama-3.1-8B-Instruct", "snapshot": "snap"}
    assert (r.host / "models/tt_transformers/tests/reference_outputs/Llama-3.1-8B-Instruct.refpt").read_bytes() == b"refpt"   # host untouched


def test_a_second_prepare_reuses_a_reference_made_for_the_same_snapshot_and_remakes_one_for_another(rig):
    rig.go("prepare")
    ref_path(rig).write_bytes(b"kept")
    assert "already made" in rig.go("prepare").stdout and ref_path(rig).read_bytes() == b"kept"
    side = ref_path(rig).with_name(ref_path(rig).name + ".orchard.json")
    side.write_text(json.dumps({"model_id": "unsloth/Llama-3.1-8B-Instruct", "snapshot": "another"}))
    assert "generated" in rig.go("prepare").stdout and ref_path(rig).read_bytes() == b"made"


def test_a_generator_that_writes_nothing_does_not_leave_the_old_reference_in_place(rig):
    rig.go("prepare")
    side = ref_path(rig).with_name(ref_path(rig).name + ".orchard.json")
    side.write_text(json.dumps({"model_id": "unsloth/Llama-3.1-8B-Instruct", "snapshot": "another"}))
    (rig.run_dir / "tt-metal-scratch/models/tt_transformers/tests/generate_reference_hf.py").write_text("pass\n")
    done = rig.go("prepare")
    assert done.returncode == 2 and not ref_path(rig).exists()


def test_a_reference_with_no_record_is_remade(rig):
    rig.go("prepare")
    ref_path(rig).with_name(ref_path(rig).name + ".orchard.json").unlink()
    ref_path(rig).write_bytes(b"unknown origin")
    assert "generated" in rig.go("prepare").stdout and ref_path(rig).read_bytes() == b"made"


def test_run_records_the_reference_it_used_with_its_hash(rig):
    rig.go("prepare")
    rig.go("run")
    ev = rig.evidence()
    assert ev["reference"] == {"source": "generated", "model_id": "unsloth/Llama-3.1-8B-Instruct", "snapshot": "snap",
                               "sha256": hashlib.sha256(b"made").hexdigest()}


def test_run_refuses_a_reference_that_changed_after_prepare(rig):
    rig.go("prepare")
    side = ref_path(rig).with_name(ref_path(rig).name + ".orchard.json")
    side.write_text(json.dumps({"model_id": "x/other", "snapshot": "snap"}))
    done = rig.go("run")
    assert done.returncode == 2 and "reference" in done.stderr


def test_a_reference_that_cannot_be_generated_is_exit_2(tmp_path):
    r = Rig(tmp_path, refpt=False)
    (r.host / "models/tt_transformers/tests/generate_reference_hf.py").write_text("raise SystemExit(1)\n")
    done = r.go("prepare")
    assert done.returncode == 2 and "reference file could not be generated" in done.stderr


@pytest.mark.parametrize("edit,message", [
    (lambda c: c.pop("mesh_device"), "needs mesh_device"),
    (lambda c: c.update(scratch_dir="relative/dir"), "absolute"),
    (lambda c: c.update(model_name="../x"), "one path part"),
    (lambda c: c.update(tt_metal_dir="/nonexistent/tt-metal"), "not a tt-metal tree"),
])
def test_a_bad_config_is_exit_2_and_names_the_problem(rig, edit, message):
    edit(rig.cfg)
    rig.write_cfg()
    done = rig.go("prepare")
    assert done.returncode == 2 and message in done.stderr


def test_a_snapshot_with_no_config_is_exit_2(rig):
    (rig.snap / "config.json").unlink()
    done = rig.go("prepare")
    assert done.returncode == 2 and "has no config.json" in done.stderr


def test_an_unknown_command_is_exit_2(rig):
    assert rig.go("serve").returncode == 2


# ---- run ----------------------------------------------------------------------------------------------------------

def test_run_without_prepare_is_exit_2(rig):
    done = rig.go("run")
    assert done.returncode == 2 and "prepare" in done.stderr


def test_run_parses_the_accuracy_the_free_run_and_the_speed_and_passes_the_gate(rig):
    assert rig.go("prepare").returncode == 0
    done = rig.go("run")
    assert done.returncode == 0, done.stderr + done.stdout
    ev = rig.evidence()
    assert ev["teacher_forced"] == {"tokens": 500, "top1": 0.912, "top5": 0.984}
    assert ev["free_run"]["text"] == "Paris is the capital of France.\n\nIt sits on the Seine and has been a centre of art for centuries."
    assert ev["free_run"]["prompt"] == "What is the capital of France?"
    assert ev["tt_metal"] == {"tag": "v0.80.0-test", "dirty": False}
    assert ev["free_run"]["new_tokens"] == 200
    assert ev["speed"] == {"decode_tokens_per_s": 41.5, "ttft_s": 0.0205}
    assert ev["device_name"] == "P300" and ev["chips"] == 2 and ev["optimizations"] == "accuracy" and ev["hub_fetch"] == []
    assert ev["template_sha256"] == hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()
    assert ev["scratch_patch"] == "evidence/scratch.patch" and (rig.stage / "evidence/scratch.patch").read_text() == ""
    run = rig.run_dir
    (run / "stages/0/evidence").mkdir(parents=True)
    (run / "stages/0/evidence/diff.txt").write_text("e")
    (run / "stages/0/delta.json").write_text(json.dumps({**DELTA, "model": "unsloth/Llama-3.1-8B-Instruct"}))
    g = gate_stock_runtime(rig.stage, run)
    assert g.ok, g.reasons


def test_run_sets_the_environment_the_runtime_reads_and_runs_from_the_scratch_copy(rig):
    rig.go("prepare")
    rig.go("run")
    sc = rig.run_dir / "tt-metal-scratch"
    seen = (sc / "seen-accuracy-ci-token-matching.env").read_text().splitlines()
    assert seen[0] == f"HF_MODEL={sc / 'hf-models/Llama-3.1-8B-Instruct'}"
    assert seen[1] == "HF_HUB_OFFLINE=1" and seen[2] == "MESH_DEVICE=P300" and seen[3] == f"PYTHONPATH={sc}"
    assert seen[4] == f"TT_CACHE_PATH={rig.cfg['tt_cache']}" and seen[5] == str(sc)
    assert (sc / "seen-accuracy-batch-1.env").is_file()
    assert not (sc / "seen-performance-batch-1.env").exists()        # only the accuracy mode runs


def test_a_failed_demo_is_exit_3_and_the_log_is_kept(rig):
    rig.go("prepare")
    (rig.run_dir / "tt-metal-scratch" / "fake-fail").write_text("x")
    done = rig.go("run")
    assert done.returncode == 3 and "see log/token-accuracy.log" in done.stderr
    assert "device open failed" in (rig.stage / "log/token-accuracy.log").read_text()


def test_a_failed_free_run_is_exit_3_even_when_the_accuracy_run_passed(rig):
    rig.go("prepare")
    (rig.run_dir / "tt-metal-scratch" / "fake-fail-free").write_text("x")
    done = rig.go("run")
    assert done.returncode == 3 and "free run 1" in done.stderr
    assert rig.evidence()["exit_codes"]["token_accuracy"] == 0


def test_a_dirty_host_tree_is_recorded_as_dirty(rig):
    (rig.host / "models/tt_transformers/demo/simple_text_demo.py").write_text(FAKE_DEMO + "\n# edited\n")
    rig.go("prepare")
    rig.go("run")
    assert rig.evidence()["tt_metal"] == {"tag": "v0.80.0-test", "dirty": True}


def test_the_patch_ends_with_a_newline_and_an_unchanged_copy_gives_an_empty_one(rig):
    rig.go("prepare")
    (rig.run_dir / "tt-metal-scratch" / "models" / "added.txt").write_text("one")
    rig.go("run")
    assert (rig.stage / "evidence/scratch.patch").read_text().endswith("\n")


def test_a_fetch_from_the_hub_is_exit_6_and_is_recorded(rig):
    rig.go("prepare")
    (rig.run_dir / "tt-metal-scratch" / "fake-fetch").write_text("x")
    done = rig.go("run")
    assert done.returncode == 6 and "fetched from the Hub" in done.stderr
    assert rig.evidence()["hub_fetch"]


def test_the_patch_shows_what_a_stage_changed_and_leaves_out_the_reference_files(rig):
    rig.go("prepare")
    sc = rig.run_dir / "tt-metal-scratch"
    (sc / "models/tt_transformers/new_param.json").write_text("{}")
    (sc / "models/tt_transformers/tests/reference_outputs/Llama-3.1-8B-Instruct.refpt").write_bytes(b"other")
    rig.go("run")
    patch = (rig.stage / "evidence/scratch.patch").read_text()
    assert "new_param.json" in patch and "refpt" not in patch and str(rig.tmp) not in patch


# ---- the parsers --------------------------------------------------------------------------------------------------

def load_module():
    spec = importlib.util.spec_from_file_location("stock_check_under_test", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_accuracy_parser_reads_the_last_pair_as_fractions():
    m = load_module()
    assert m.parse_accuracy("Top1 Accuracy: 10.00%, Top5 Accuracy: 20.00%\n Top1 Accuracy: 90.50%, Top5 Accuracy: 99.00%") == (0.905, 0.99)
    assert m.parse_accuracy("nothing") is None


def test_the_iteration_parser_counts_from_zero():
    m = load_module()
    assert m.parse_iterations("Iteration 0: 1ms @ 1.0 tok/s/user\nIteration 9: 1ms @ 1.0 tok/s/user") == 10
    assert m.parse_iterations("none") is None


def test_the_free_run_parser_handles_a_log_with_no_output_block():
    m = load_module()
    got = m.parse_free_run("nothing here")
    assert got["text"] == "" and got["prompt"] is None and got["decode_tokens_per_s"] is None and got["ttft_s"] is None


def test_the_default_import_check_names_accelerate():
    """The first real prepare (2026-10-29) loaded an 8B model on the CPU before it failed on a missing `accelerate`, which transformers needs for
    device_map="auto" in tt-transformers' reference script. The default check names it, so the failure comes in seconds, before any model loads."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "orchard" / "skills" / "stock-runtime-templates"))
    import importlib
    mod = importlib.import_module("stock_check")
    assert "accelerate" in mod.OPTIONAL_KEYS["import_check"]
