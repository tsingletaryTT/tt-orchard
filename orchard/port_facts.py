# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The facts file and the skeleton the coder starts stage 2 of a full port from.

In three unattended rehearsal runs (2026-11-03) the coder's first attempt at stage 2 read for 20 turns and wrote nothing. Its context
grew from 6,000 to 44,000 tokens: it paged through the stage 1 reference files 12,000 characters at a time (the check reads those; the
port may not) and probed for a `model.safetensors.index.json` that a single-file checkpoint does not have. The watchdog blocked the
attempt and the retry, after a twelve-minute coder reboot, wrote a port. The retry then re-read tensor shapes with near-identical
snippets until the watchdog stopped it again, and two of its replies ran to the 16,384-token cap and ended in an empty command.

`build` reads what the coder would otherwise collect by hand, from the checkpoint itself:

  * the config's small scalar and list fields, and the layer schedule grouped by type;
  * every tensor's dtype and shape from the safetensors headers, one line per group of tensors that differ only by a number in the
    name (`model.layers.{i}.mlp.up_proj.weight  BF16 [256, 64]  x4  layers 0-3`), whether the checkpoint is one file or sharded;
  * the config fields that hold layer ids, with the index convention the reference class states for them. The first run's port read
    `ple_layer_ids` as zero-indexed; the reference's configuration docstring says one-indexed and the checkpoint agrees.

Nothing in it is specific to one model. The reference's convention comes from the docstring of the `transformers` configuration class
named by the config's `model_type`, found without importing `transformers`. When it is not found the file says so.

The checkpoint is read through `residency.read_headers` (links followed once, regular files only, header and tensor-count caps) and
`config.json` through `safeio.read_json`. The file is bounded to one `read_file` page (`MAX_CHARS` against the tool's 12,000), and
says how many tensor groups it left out. `build` raises PortFactsError for anything it cannot read; the supervisor then writes no
facts file and the coder works from its skill as before.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from orchard import full_port, residency, safeio
from orchard.commands import run_command

SCHEMA = "orchard-port-facts/1"
FACTS_NAME = "port_facts.json"
SKELETON = Path(__file__).resolve().parent / "skills" / "full-port-templates" / "port_model_skeleton.py"
SKELETON_NAME = "port_model.py"
FIRST_VERSION_TURNS = 12          # choice: the watchdog nudges at 20 turns with no file written; a first version by 12 leaves room to refine
MAX_CHARS = 10_500                # choice: under the agent's 12,000-character tool page, so one read_file returns the whole file
CONFIG_MAX = 1 << 20              # config.json is a few kilobytes; this bound only stops a hostile file
SOURCE_MAX = 1 << 20              # a transformers configuration module is tens of kilobytes
SKELETON_MAX = 1 << 20
MAX_CONFIG_FIELDS = 40            # per config section
MAX_LIST = 32                     # elements of a list kept in the summary
MAX_STR = 60
MAX_NOTES = 12
MAX_NOTE_CHARS = 300
MAX_SHARDS_LISTED = 8
MAX_SCHEDULE = 4096
# Config fields that hold layer, block, expert or position numbers. `eos_token_id` lists hold token ids and do not match.
INDEX_FIELD = re.compile(r"(layer|block|expert|position).*(_ids?|_idx|_indices|_numbers?)$|_layers?$")
CONVENTION = re.compile(r"\b(?:one|zero|1|0)[- ]?(?:based|indexed)\b", re.I)
_FIELD_HEAD = re.compile(r"^\s{4,8}(\w+) \(`")                       # `    name (`type`, *optional*):` in a configuration docstring
_MODEL_TYPE = re.compile(r"[a-z0-9_]{1,64}")
SUB_CONFIG_SUFFIXES = ("text", "vision", "audio")                    # `qwen4_exp_text` lives in models/qwen4_exp/configuration_qwen4_exp.py
_NUMBER_IN_NAME = re.compile(r"(?<=\.)\d+(?=\.|$)")
UNKNOWN_CONVENTION = ("not stated in the reference configuration source (or that source was not found): compare these ids with the "
                      "layer numbers of the tensors below before using them as layer indexes")
_FIND_TRANSFORMERS = "import importlib.util as u; s = u.find_spec('transformers'); print(list(s.submodule_search_locations)[0] if s else '')"


def default_run(argv, timeout):
    """`run(argv, timeout)` for `reference_root`: (returncode, stdout, stderr), with a nearly empty environment."""
    result = run_command(argv, timeout, env={"PATH": os.environ.get("PATH", os.defpath)})
    return (result.returncode if result.returncode is not None else 1), result.stdout, result.stderr


class PortFactsError(Exception):
    """The checkpoint or its config cannot be read for the facts file. The message names the file and the cause, never its content."""


# ---- the reference's conventions ---------------------------------------------------------------------------------------------

def convention_notes(source) -> list[dict]:
    """The fields of a configuration docstring whose description says how they are numbered (one-indexed, zero-based and the like):
    [{"field", "note"}] in file order, at most MAX_NOTES, each note at most MAX_NOTE_CHARS."""
    if not isinstance(source, str):
        return []
    notes, field, body = [], None, []

    def close():
        text = " ".join(body).strip()
        if field and text and CONVENTION.search(text) and len(notes) < MAX_NOTES and all(n["field"] != field for n in notes):
            notes.append({"field": field, "note": text[:MAX_NOTE_CHARS]})

    for line in source.splitlines():
        head = _FIELD_HEAD.match(line)
        if head:
            close()
            field, body = head.group(1), []
        elif field is not None:
            if line.strip().startswith('"""') or line.strip().endswith('"""'):
                close()
                field = None
            else:
                body.append(line.strip())
    close()
    return notes


def reference_root(run, python) -> Path | None:
    """The `transformers` package directory of the reference interpreter, found with `find_spec` so the package is not imported (its
    import is slow and may open nothing it should not). `run(argv, timeout)` returns (returncode, stdout, stderr). None when there is no
    interpreter, the call fails, or the answer is not an absolute path."""
    if not isinstance(python, str) or not python:
        return None
    try:
        code, out, _ = run([python, "-I", "-c", _FIND_TRANSFORMERS], 30)
    except Exception:       # a missing interpreter or a timeout: the facts file goes without conventions
        return None
    path = Path(out.strip().splitlines()[0]) if code == 0 and out.strip() else None
    return path if path is not None and path.is_absolute() else None


def reference_source(root, model_type) -> str | None:
    """The text of `models/<model_type>/configuration_<model_type>.py` under the transformers directory `root`, or None. A model type that
    ends in _text, _vision or _audio is also tried without that suffix. The model type must be lowercase letters, digits and underscores,
    so it cannot name a path outside `root`."""
    if root is None or not isinstance(model_type, str) or not _MODEL_TYPE.fullmatch(model_type):
        return None
    names = [model_type]
    head, _, tail = model_type.rpartition("_")
    if head and tail in SUB_CONFIG_SUFFIXES:          # the text part of a family is declared in the family's own module
        names.append(head)
    for name in names:
        try:
            data = safeio.read_bytes(Path(root) / "models" / name / f"configuration_{name}.py", max_bytes=SOURCE_MAX)
        except safeio.SafeReadError:
            continue
        return data.decode("utf-8", "replace")
    return None


# ---- reading the checkpoint --------------------------------------------------------------------------------------------------

def _config(snapshot: Path) -> dict:
    try:
        return safeio.read_json(safeio.resolved(snapshot / "config.json"), max_bytes=CONFIG_MAX, expect=dict)
    except safeio.SafeReadError as exc:
        raise PortFactsError(f"config.json is not usable ({exc.reason})") from None


def _scalars(section: dict) -> tuple[dict, list]:
    """The small scalar and list-of-scalar fields of a config section (sorted, at most MAX_CONFIG_FIELDS), and the names left out."""
    kept, left = {}, []
    for key in sorted(section):
        v = section[key]
        simple = isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) <= MAX_STR)
        if isinstance(v, list) and len(v) <= MAX_LIST and all(isinstance(x, (bool, int, float)) or (isinstance(x, str) and len(x) <= MAX_STR) for x in v):
            simple = True
        if simple and len(kept) < MAX_CONFIG_FIELDS:
            kept[key] = v
        else:
            left.append(key)
    return kept, left


def _ranges(numbers) -> str:
    """Sorted whole numbers as `0-2,5,7-9`."""
    nums, parts = sorted(set(numbers)), []
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(str(nums[i]) if i == j else f"{nums[i]}-{nums[j]}")
        i = j + 1
    return ",".join(parts)


def _schedule(text_config: dict):
    types = text_config.get("layer_types")
    if not isinstance(types, list) or not types or len(types) > MAX_SCHEDULE or not all(isinstance(t, str) and len(t) <= MAX_STR for t in types):
        return None
    by_type: dict = {}
    for i, t in enumerate(types):
        by_type.setdefault(t, []).append(i)
    return {t: _ranges(ix) for t, ix in by_type.items()}


def _groups(tensors) -> list[str]:
    """One line per group of tensors with one name pattern (numbers in the name replaced by {i}), one dtype and one shape."""
    groups: dict = {}
    for t in sorted(tensors, key=lambda t: t.name):
        first = _NUMBER_IN_NAME.search(t.name)
        key = (_NUMBER_IN_NAME.sub("{i}", t.name), t.dtype, tuple(t.shape))
        g = groups.setdefault(key, [0, []])
        g[0] += 1
        if first:
            g[1].append(int(first.group()))
    out = []
    for (pattern, dtype, shape), (count, firsts) in groups.items():
        out.append(f"{pattern}  {dtype} {list(shape)}  x{count}" + (f"  layers {_ranges(firsts)}" if firsts else ""))
    return out


def _weights(snapshot: Path, headers) -> dict:
    files = sorted({t.file for t in headers.tensors})
    index = os.path.lexists(snapshot / residency.INDEX_FILE)
    dtypes: dict = {}
    for t in headers.tensors:
        dtypes[t.dtype] = dtypes.get(t.dtype, 0) + 1
    if index:
        note = (f"{residency.INDEX_FILE} maps each tensor to its shard in its weight_map. Every shard's tensors are listed below; "
                "you do not need to open the index.")
    elif len(files) == 1:
        note = (f"There is no {residency.INDEX_FILE}; every tensor is in {files[0]}. Read names from the list below or from the file's "
                "header (safetensors.safe_open(path, 'pt').keys()).")
    else:
        note = f"There is no {residency.INDEX_FILE}; the tensors are spread over {len(files)} files, all listed below."
    return {"layout": "sharded" if index or len(files) > 1 else "single file", "index_file": index, "file_count": len(files),
            "files": files[:MAX_SHARDS_LISTED], "tensor_count": len(headers.tensors), "dtypes": dtypes,
            "bytes": headers.checkpoint_bytes, "note": note}


def build(snapshot, *, reference_source=None, reference_lookup=None) -> dict:
    """The facts of a checkpoint directory as a JSON-ready dict. `reference_source` is the text of the reference's configuration module;
    `reference_lookup(model_type)` finds it when it is not given. Raises PortFactsError when the config or a header cannot be read."""
    snap = Path(snapshot)
    config = _config(snap)
    try:
        headers = residency.read_headers(snap)
    except residency.ResidencyError as exc:
        raise PortFactsError(str(exc)) from None
    nested = config.get("text_config") if isinstance(config.get("text_config"), dict) else None
    top, top_left = _scalars({k: v for k, v in config.items() if k != "text_config"})
    text, text_left = _scalars(nested) if nested is not None else ({}, [])
    model_types = [m for m in (config.get("model_type"), (nested or {}).get("model_type")) if isinstance(m, str)]
    if reference_source is None and reference_lookup is not None:
        reference_source = next((s for s in (reference_lookup(m) for m in model_types) if s), None)
    notes = convention_notes(reference_source)
    by_field = {n["field"]: n["note"] for n in notes}
    index_fields = []
    for where, section in (("config", top), ("text_config", text)):
        for key, v in section.items():
            if INDEX_FIELD.search(key) and isinstance(v, list) and v and all(isinstance(x, int) and not isinstance(x, bool) for x in v):
                index_fields.append({"field": key, "in": where, "values": v, "convention": by_field.get(key, UNKNOWN_CONVENTION)})
    lines = _groups(headers.tensors)
    facts = {"schema": SCHEMA, "model_type": model_types[0] if model_types else None, "config": top,
             "config_left_out": sorted(top_left)}
    if nested is not None:
        facts["text_config"], facts["text_config_left_out"] = text, sorted(text_left)
    facts.update({"layer_schedule": _schedule(nested if nested is not None else config), "index_fields": index_fields,
                  "conventions": notes, "weights": _weights(snap, headers), "tensor_groups": len(lines), "tensors": lines,
                  "tensors_left_out": 0,
                  "reading": ("Tensor lines are one per name pattern ({i} stands for a number in the name) with dtype, shape, how many "
                              "tensors and, when the name has a number, which numbers. Layer ids in the config may be counted from 0 or "
                              "from 1: see index_fields.")})
    return _fit(facts)


def _fit(facts: dict) -> dict:
    """Drop tensor lines from the end, then config fields, until the file fits MAX_CHARS."""
    def size():
        return len(json.dumps(facts, indent=2)) + 1
    while size() > MAX_CHARS and facts["tensors"]:
        facts["tensors"].pop()
        facts["tensors_left_out"] += 1
    for section in ("text_config", "config"):
        while size() > MAX_CHARS and facts.get(section):
            facts[section].popitem()
    if size() > MAX_CHARS:
        raise PortFactsError("the facts do not fit one read_file page even without tensors")
    return facts


def write(stage_dir, facts: dict) -> Path:
    path = Path(stage_dir) / FACTS_NAME
    safeio.atomic_write(path, json.dumps(facts, indent=2) + "\n", mode=0o644)
    return path


def copy_skeleton(stage_dir) -> bool:
    """Put the skeleton at stages/2/port/port_model.py unless a file is already there (a resumed stage keeps the coder's port). True when
    it was copied."""
    port = Path(stage_dir) / "port"
    dest = port / SKELETON_NAME
    if os.path.lexists(dest):
        return False
    port.mkdir(parents=True, exist_ok=True)
    try:
        safeio.atomic_write(dest, safeio.read_bytes(SKELETON, max_bytes=SKELETON_MAX), mode=0o644, replace=False)
    except FileExistsError:
        return False
    return True


def prepare_note(*, facts: bool, skeleton: bool) -> str:
    """What the supervisor adds to the coder's prepare task for a full port. The generic prepare task names only hw_test.json and
    handoff.json; the port itself is described only in the skill, and the coder explored and wrote nothing."""
    parts = [f"This stage's check is the supervisor's: when you finish, it puts its own `stages/2/port_check.py` back and rewrites "
             f"`hw_test.json` to its fixed command (`{full_port.COMMAND}`). Your work in this step is the port module, "
             f"`stages/2/port/port_model.py`."]
    if facts:
        parts.append(f"Read `stages/2/{FACTS_NAME}` first. It holds the config fields, the layer schedule, every tensor's dtype and shape "
                     "from the checkpoint's own headers (the checkpoint may be one file with no index, or sharded), and the index "
                     "conventions the reference states for the config's id lists. You do not need to read the config, the headers or "
                     "the weights again.")
    if skeleton:
        parts.append("`stages/2/port/port_model.py` is a skeleton: `load`, `prefill`, `decode_step` and `close` each raise "
                     "NotImplementedError until you replace them, and `read_tensors` reads the checkpoint whether it is one file or "
                     "shards.")
    parts.append(f"Write your first version of `port_model.py` within {FIRST_VERSION_TURNS} turns, even if it is incomplete, then refine "
                 "it. Do not page through the stage 1 reference files: the check reads them, and a port may not. Do not search the "
                 "machine for other models' code; the worked example the skill names is enough.")
    parts.append(f"When the port is written, write `hw_test.json` as {{\"command\": \"{full_port.COMMAND}\", \"deadline_s\": "
                 f"{full_port.DEADLINE_S}}} and `handoff.json`, then reply with a short summary and no tool call.")
    return "\n\n".join(parts)
