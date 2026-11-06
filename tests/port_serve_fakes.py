# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fakes for the stage 5 serving template (orchard/skills/full-port-templates/port_serve.py): ports that answer the passkey, ports that do not, ports
that carry state from one request to the next, a tokenizer with the passkey words, and a runner that leaves what the supervisor leaves. Nothing here
touches a chip. The template runs as the real subprocess it is under the supervisor, on the CPU, against the fake `ttnn` of `port_check_fakes`."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from orchard import full_port, full_port_serve, full_port_serve_stage
from port_check_fakes import CHIPS, REF_IDS, install_fake_ttnn, make_run, python_path, run_template

REPO = Path(__file__).resolve().parent.parent
WORDS = ("The grass is green sky blue sun warm Here we go again pass key Remember it What user assistant : . ?").replace(":", " : ").split()

# A port with a vocabulary of 1000. Like the stage 2 fakes it counts up by one. When the prompt is long (a passkey prompt) it returns the id that
# follows the first "pass key is" it finds, which is what a model that can read its context would do.
ANSWERING = '''
import ttnn
VOCAB = 1000
PASS, KEY, IS = {pass_id}, {key_id}, {is_id}
LAYERS = 64                           # a stand-in decoder stack (deeper than any fake config): one ttnn.linear per layer per pass
def _peak(i):
    for _ in range(LAYERS):
        ttnn.linear(None, None)
    v = [0.0] * VOCAB
    v[i % VOCAB] = 1.0
    return v
def load(model_dir, mesh_device):
    return {{"dir": model_dir, "device": mesh_device}}
def prefill(model, ids):
    if len(ids) > 60:
        for j in range(len(ids) - 3):
            if ids[j:j + 3] == [PASS, KEY, IS]:
                return _peak(ids[j + 3])
    return _peak(100)
def decode_step(model, token_id, position):
    return _peak(token_id + 1)
def close(model):
    pass
'''

# The same port with no reading of the context: what a model with random weights does. It cannot answer a passkey.
RANDOM_WEIGHTS = ANSWERING.replace("    if len(ids) > 60:", "    if False:")

# A port that remembers the length of the previous prompt once the replay is over (three prefills in): the same prompt asked twice with another
# request between gives two different answers. Stage 2's check and the corpus replay do not see it.
CARRIES_STATE = ANSWERING.replace("def load(model_dir, mesh_device):", "STATE = {{\"n\": 0, \"prev\": 0}}\ndef load(model_dir, mesh_device):").replace(
    "def prefill(model, ids):\n", "def prefill(model, ids):\n    STATE[\"n\"] += 1\n    before, STATE[\"prev\"] = STATE[\"prev\"], len(ids)\n"
    "    if STATE[\"n\"] > 2 and len(ids) <= 60:\n        return _peak(100 + before % 7)\n")


# A port that changes after its first N prefills: it gives id 7 where it gave 100. Stage 2's check (two prefills) cannot see it; the corpus replay
# (two prefills) can when N is 1, the canary (a third and fourth prefill) when N is 2.
def leaky(after: int) -> str:
    return ANSWERING.replace("def load(model_dir, mesh_device):", "COUNT = [0]\ndef load(model_dir, mesh_device):").replace(
        "def prefill(model, ids):\n", f"def prefill(model, ids):\n    COUNT[0] += 1\n    if COUNT[0] > {after}:\n        return _peak(7)\n")


def tokenizer_with(path: Path, extra: list) -> None:
    """A word-level tokenizer.json: ids 0 to 199 decode to `w<i>`, then the passkey words and `extra` (the keys) from id 200."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab = {f"w{i}": i for i in range(200)}
    for word in [*WORDS, *extra]:
        vocab.setdefault(word, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="w0"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(path))


def port_source(template: str, snapshot: Path) -> str:
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    ids = {w: tok.token_to_id(w) for w in ("pass", "key", "is")}
    return template.format(pass_id=ids["pass"], key_id=ids["key"], is_id=ids["is"])


def stage_end(stage: int, run: Path, result: str = "pass") -> dict:
    """The ledger entry a passing stage leaves: the files its gate accepted, with their sha256."""
    rec = full_port.file_record(run, run / "stages" / str(stage) / "evidence" / "port-check.json")
    return {"event": "stage_end", "stage": stage, "data": {"result": result, "evidence": [rec]}}


def build(tmp_path: Path, port_template: str = ANSWERING, *, chips=CHIPS, stage3: bool = False):
    """A run directory where stage 2 (and with `stage3`, stage 3) has passed with the supervisor's own template, and stage 5 is prepared by
    `full_port_serve_stage.prepare`. Returns a namespace-like dict with run, stage, snap, prep, entries. The stage 5 test has not run yet."""
    run, stage2, snap = make_run(tmp_path, "import ttnn\n")
    tokenizer_with(snap / "tokenizer.json", [])
    source = port_source(port_template, snap)
    (stage2 / "port" / "port_model.py").write_text(source)
    run_template(tmp_path, run, stage2, chips=chips)                       # stage 2's real check: the accepted record
    entries = [stage_end(2, run)]
    if stage3:
        s3 = run / "stages" / "3"
        shutil.copytree(stage2, s3, ignore=shutil.ignore_patterns("evidence", "result.json", "test-result.json"))
        run_template(tmp_path, run, s3, chips=chips, command=full_port.command(3))
        entries.append(stage_end(3, run))
    stage = run / "stages" / "5"
    prep = full_port_serve_stage.prepare(run, stage, entries, snapshot=snap, orchard_dir=REPO)
    assert not prep.problems, prep.problems
    tokenizer_with(snap / "tokenizer.json", [p["key"] for p in json.loads((stage / full_port_serve.CONFIG_NAME).read_text())["passkeys"]])
    return {"run": run, "stage": stage, "snap": snap, "prep": prep, "entries": entries, "tmp": tmp_path, "chips": list(chips)}


def run_serve(b: dict, *, chips=None, cache: Path | None = None, command: str = full_port_serve.COMMAND):
    """Run the stage 5 command in the run directory as the supervisor does, and leave the output file and test-result.json."""
    chips = b["chips"] if chips is None else chips
    cmd = command.split()
    cmd[0] = sys.executable
    env = {"PATH": os.environ.get("PATH", ""), "TT_VISIBLE_DEVICES": ",".join(chips), "PYTHONPATH": python_path(b["tmp"]), "HOME": str(b["tmp"])}
    if cache is not None:
        env["TT_METAL_CACHE"] = str(cache)
    done = subprocess.run(cmd, cwd=b["run"], env=env, capture_output=True, text=True, timeout=180)
    out = b["stage"] / "evidence" / "hw-test-output.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(done.stdout + done.stderr)
    marker = {"command": command, "returncode": done.returncode, "timed_out": False, "seconds": 1.0, "chips": list(chips),
              "output": full_port.file_record(b["run"], out)}
    (b["stage"] / "test-result.json").write_text(json.dumps(marker))
    return done


def derive(b: dict, *, rehearsal: bool = False):
    """Write result.json the way the supervisor does. Returns the (result, why) of `derive`."""
    result, why = full_port_serve.derive(b["stage"], b["run"], rehearsal=rehearsal)
    if result is not None:
        (b["stage"] / "result.json").write_text(json.dumps(result))
    return result, why


def gate(b: dict, *, rehearsal: bool = False, source: int | None = None, expected_chips=None, corpus_sha=...):
    waived: list = []
    reasons, seen = full_port_serve.check(
        b["stage"], b["run"], rehearsal=rehearsal, waived=waived, source=source or (3 if any(e["stage"] == 3 for e in b["entries"]) else 2),
        expected_chips=len(b["chips"]) if expected_chips is None else expected_chips,
        corpus_sha256=b["prep"].corpus_sha256 if corpus_sha is ... else corpus_sha)
    return reasons, seen, waived


def clone(b: dict, dest: Path) -> dict:
    """A copy of a built and run directory, for a test that edits one file and asks the gate. The config still names the first run's snapshot."""
    run = dest / "run"
    shutil.copytree(b["run"], run)
    return {**b, "run": run, "stage": run / "stages" / "5"}
