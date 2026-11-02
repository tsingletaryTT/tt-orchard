# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fakes for the full-port check template (orchard/skills/full-port-templates/port_check.py): a stand-in `ttnn`, a word-level tokenizer
and port modules (one honest, several that try to forge the record). Nothing here touches a chip. The template runs as the real
subprocess it is under the supervisor, on the CPU; its worker imports the fake `ttnn` through PYTHONPATH."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from orchard import full_port

REF_IDS = list(range(100, 132))                    # 32 reference ids; the honest port's next id is always the last id plus 1
PROMPT_IDS = [1, 2, 3]
CHIPS = ["0000:01:00.0", "0000:02:00.0"]

# The stand-in for the runtime: the same names the template calls, and nothing else.
FAKE_TTNN = '''
class MeshShape:
    def __init__(self, r, c): self.shape = (r, c)
def open_device(device_id=0): return {"device": device_id}
def close_device(d): pass
def open_mesh_device(shape): return {"mesh": shape.shape}
def close_mesh_device(d): pass
'''

# An honest port: logits peak at the id after the last one it was given. prefill answers REF[0]; teacher forcing (reference id i - 1 in)
# gives REF[i] because the reference counts up by one.
HONEST = '''
import ttnn
VOCAB = 200
def _peak(i):
    v = [0.0] * VOCAB
    v[i] = 1.0
    return v
def load(model_dir, mesh_device):
    return {"dir": model_dir, "device": mesh_device}
def prefill(model, ids):
    return _peak(100)
def decode_step(model, token_id, position):
    return _peak(token_id + 1)
def close(model):
    pass
'''

# A port that computes nothing useful: its argmax is always 0, so its agreement with the reference is 0.
WRONG = HONEST.replace("return _peak(100)", "return _peak(0)").replace("return _peak(token_id + 1)", "return _peak(0)")


def tokenizer_json(path: Path) -> None:
    """A word-level tokenizer.json: id i decodes to the word `w<i>`."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {f"w{i}": i for i in range(200)}
    tok = Tokenizer(models.WordLevel(vocab, unk_token="w0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(path))


def site_with_tokenizers() -> str:
    """The directory `tokenizers` imports from. The supervisor tests run the template with the system python3, which may not have it."""
    import tokenizers
    return str(Path(tokenizers.__file__).resolve().parent.parent)


def make_run(tmp_path: Path, port_src: str = HONEST, *, extra_files: dict | None = None, template_text: str | None = None):
    """A run directory with the stage 1 reference, a snapshot with a tokenizer, stages/2/port/port_model.py, and the template and its config
    copied the way the supervisor copies them. Returns (run_dir, stage_dir, snapshot_dir)."""
    run = tmp_path / "run"
    ev = run / "stages" / "1" / "evidence" / "reference"
    ev.mkdir(parents=True)
    (ev / "generated-ids.json").write_text(json.dumps({"generated_ids": REF_IDS, "generated_text": "x"}))
    (ev / "prompt-ids.json").write_text(json.dumps({"prompt_ids": PROMPT_IDS, "prompt_text": "p"}))
    snap = tmp_path / "snapshot"
    snap.mkdir()
    tokenizer_json(snap / "tokenizer.json")
    stage = run / "stages" / "2"
    (stage / "port").mkdir(parents=True)
    (stage / "port" / "port_model.py").write_text(port_src)
    for rel, text in (extra_files or {}).items():
        p = stage / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    if template_text is None:
        shutil.copy2(full_port.TEMPLATE, stage / full_port.TEMPLATE_NAME)
    else:
        (stage / full_port.TEMPLATE_NAME).write_text(template_text)
    (stage / full_port.CONFIG_NAME).write_text(json.dumps({"model_snapshot": str(snap)}))
    install_fake_ttnn(tmp_path)
    return run, stage, snap


def install_fake_ttnn(tmp_path: Path) -> None:
    fake = tmp_path / "fake-ttnn"
    fake.mkdir(exist_ok=True)
    (fake / "ttnn.py").write_text(FAKE_TTNN)


def python_path(tmp_path: Path) -> str:
    """PYTHONPATH for a test process: the fake runtime, and the directory `tokenizers` comes from."""
    return os.pathsep.join([str(tmp_path / "fake-ttnn"), site_with_tokenizers()])


def run_template(tmp_path: Path, run: Path, stage: Path, *, chips=CHIPS, command: str = full_port.COMMAND):
    """Run the supervisor's command in the run directory the way the supervisor does, and leave what the supervisor leaves: the output
    file and test-result.json. Returns the completed process."""
    cmd = command.split()
    cmd[0] = sys.executable
    env = {"PATH": os.environ.get("PATH", ""), "TT_VISIBLE_DEVICES": ",".join(chips), "PYTHONPATH": python_path(tmp_path)}
    done = subprocess.run(cmd, cwd=run, env=env, capture_output=True, text=True, timeout=120)
    out = stage / "evidence" / "hw-test-output.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(done.stdout + done.stderr)
    marker = {"command": command, "returncode": done.returncode, "timed_out": False, "seconds": 1.0, "chips": list(chips),
              "output": full_port.file_record(run, out)}
    (stage / "test-result.json").write_text(json.dumps(marker))
    return done


def write_result(stage: Path, **over) -> None:
    """The finish step's result.json, copied from the record the way the skill says."""
    rec = json.loads((stage / "evidence" / "port-check.json").read_text())
    result = {"runs": True, "coherent": True, "load_s": rec["load_s"], "n_tokens": len(rec["chip_ids"]),
              "top1_agreement": rec["top1_agreement"], "free_run_text": rec["free_run_text"],
              "evidence": ["stages/2/evidence/port-check.json", "stages/2/evidence/hw-test-output.txt"]}
    (stage / "result.json").write_text(json.dumps({**result, **over}))
