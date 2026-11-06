# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fakes for stage 6 of a full port (orchard/skills/full-port-templates/port_quality.py): a run directory with the stage 1 files (ids, prompt, top five
logits), a snapshot with a tokenizer and a config, the port copied into stages/6/port/, the template and its config, and a source record from stage 2
(or 3) that names the same port files. Port modules for the cases the tests need. Nothing here touches a chip; the template runs as the real subprocess it
is under the supervisor, on the CPU, against the fake `ttnn` of tests/port_check_fakes.py."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from orchard import full_port, full_port_quality
from port_check_fakes import CHIPS, HONEST, PROMPT_IDS, REF_IDS, install_fake_ttnn, python_path, tokenizer_json

# Reference top five at position k: the reference id first, then four ids that are never a reference id. The honest port peaks at the reference id.
OTHERS = [0, 1, 2, 3]
TINY_CONFIG = {"model_type": "qwen3", "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2, "num_attention_heads": 4,
               "num_key_value_heads": 2, "vocab_size": 200, "head_dim": 16}
# A model so large that its decode floor is far below what a fake port measures: the share of the floor comes out above the bound.
HUGE_CONFIG = {**TINY_CONFIG, "hidden_size": 16384, "intermediate_size": 65536, "num_hidden_layers": 200, "num_attention_heads": 128,
               "num_key_value_heads": 16, "head_dim": 128, "vocab_size": 200000}

# The honest port, plus the optional DTYPE_POLICY a port may declare.
DECLARED = HONEST + "\nDTYPE_POLICY = {'default': 'bfp8'}\n"
# Teacher forcing (the first prefill) is honest. From the second prefill on, decode cycles through five ids: the free run loops on a few words.
LOOPS = HONEST.replace("def prefill(model, ids):\n    return _peak(100)",
                       "CALLS = [0]\ndef prefill(model, ids):\n    CALLS[0] += 1\n    return _peak(100)").replace(
    "def decode_step(model, token_id, position):\n    return _peak(token_id + 1)",
    "def decode_step(model, token_id, position):\n    return _peak(token_id + 1) if CALLS[0] < 2 else _peak(7 + token_id % 5)")
# The right ids, but the logits are shaped so the gaps to the other ids are five logits wide, not one.
SKEWED = HONEST.replace("v[i] = 1.0", "v[i] = 6.0")
# A port that rewrites its own source while it loads.
REWRITES_ITSELF = HONEST.replace("def load(model_dir, mesh_device):\n",
                                 "def load(model_dir, mesh_device):\n"
                                 "    import pathlib\n"
                                 "    p = pathlib.Path(__file__)\n"
                                 "    p.write_text(p.read_text() + '\\n# changed while the test ran\\n')\n", 1)
PRINTS_THE_MARK = HONEST.replace("def load(model_dir, mesh_device):\n",
                                 "def load(model_dir, mesh_device):\n    print('PORT_QUALITY_SHA256 ' + '0' * 64, flush=True)\n", 1)
# A port that reads the stage 1 reference files.
READS_REFERENCE = HONEST + "\nREF = 'stages/1/evidence/reference/generated-ids.json'\n"


def top5_file(n: int = 32) -> dict:
    """What stage 1's template writes: for each step the reference's top five (id, logit, text)."""
    steps = [{"step": k, "position": 3 + k, "chosen": REF_IDS[k],
              "top5": [{"id": REF_IDS[k], "logit": 1.0, "text": "x"}] + [{"id": o, "logit": 0.0, "text": "y"} for o in OTHERS]} for k in range(n)]
    return {"prompt_last_position": steps[0], "steps": steps}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def make_run(tmp_path: Path, port_src: str = HONEST, *, source: int = 2, config: dict = TINY_CONFIG, top5: bool = True,
             template_text: str | None = None, source_port_src: str | None = None):
    """A run directory ready for the stage 6 command. Returns (run_dir, stage_dir, snapshot_dir, seeded) where `seeded` is the list the ledger would
    hold for the copied files."""
    run = tmp_path / "run"
    ref = run / "stages" / "1" / "evidence" / "reference"
    ref.mkdir(parents=True)
    (ref / "generated-ids.json").write_text(json.dumps({"generated_ids": REF_IDS, "generated_text": "x"}))
    (ref / "prompt-ids.json").write_text(json.dumps({"prompt_ids": PROMPT_IDS, "prompt_text": "p"}))
    if top5:
        (ref / "top5-logits.json").write_text(json.dumps(top5_file()))
    snap = tmp_path / "snapshot"
    snap.mkdir()
    tokenizer_json(snap / "tokenizer.json")
    (snap / "config.json").write_text(json.dumps(config))
    # the earlier stage's record: only the part the stage 6 gate reads (port_sources)
    src_port = (source_port_src or port_src).encode()
    src_dir = run / "stages" / str(source)
    (src_dir / "evidence").mkdir(parents=True)
    (src_dir / full_port.RECORD_FILE).write_text(json.dumps({"port_sources": [
        {"path": f"stages/{source}/port/port_model.py", "sha256": sha(src_port), "role": "model"}]}))
    stage = run / "stages" / "6"
    (stage / "port").mkdir(parents=True)
    (stage / "port" / "port_model.py").write_text(port_src)
    seeded = [{"path": "stages/6/port/port_model.py", "sha256": sha(src_port)}]
    if template_text is None:
        shutil.copy2(full_port_quality.TEMPLATE, stage / full_port_quality.TEMPLATE_NAME)
    else:
        (stage / full_port_quality.TEMPLATE_NAME).write_text(template_text)
    (stage / full_port_quality.CONFIG_NAME).write_text(json.dumps({"model_snapshot": str(snap)}))
    install_fake_ttnn(tmp_path)
    return run, stage, snap, seeded


def run_template(tmp_path: Path, run: Path, stage: Path, *, chips=CHIPS, command: str = full_port_quality.COMMAND):
    """Run the supervisor's command in the run directory the way the supervisor does, and leave what the supervisor leaves: the output file and
    test-result.json. Returns the completed process."""
    cmd = command.split()
    cmd[0] = sys.executable
    env = {"PATH": os.environ.get("PATH", ""), "TT_VISIBLE_DEVICES": ",".join(chips), "PYTHONPATH": python_path(tmp_path)}
    done = subprocess.run(cmd, cwd=run, env=env, capture_output=True, text=True, timeout=300)
    out = stage / "evidence" / "hw-test-output.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(done.stdout + done.stderr)
    marker = {"command": command, "returncode": done.returncode, "timed_out": False, "seconds": 1.0, "chips": list(chips),
              "output": full_port.file_record(run, out)}
    (stage / "test-result.json").write_text(json.dumps(marker))
    return done


def derive_and_write(run: Path, stage: Path, *, source: int = 2, rehearsal: bool = False):
    """The supervisor's derived result.json, written to the stage directory. Returns (result, why)."""
    result, why = full_port_quality.derive(stage, run, source_stage=source, rehearsal=rehearsal)
    if result is not None:
        (stage / "result.json").write_text(json.dumps(result, indent=2))
    return result, why


def check(run: Path, stage: Path, seeded, **kw):
    kw.setdefault("source_stage", 2)
    kw.setdefault("expected_chips", len(CHIPS))
    # What the ledger would hold: the hash of the source stage's record as its gate accepted it (the supervisor passes it; a test may override).
    kw.setdefault("accepted_record_sha", full_port._sha(run / "stages" / str(kw["source_stage"]) / full_port.RECORD_FILE))
    kw.setdefault("top5_accepted_sha", full_port._sha(run / full_port_quality.TOP5_REL))
    return full_port_quality.check(stage, run, seeded=seeded, **kw)
