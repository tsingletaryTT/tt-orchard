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
from orchard.defaults import FIRST_WRITE_TURNS

SCHEMA = "orchard-port-facts/1"
FACTS_NAME = "port_facts.json"
SKELETON = Path(__file__).resolve().parent / "skills" / "full-port-templates" / "port_model_skeleton.py"
SKELETON_NAME = "port_model.py"
FIRST_VERSION_TURNS = FIRST_WRITE_TURNS   # choice (defaults.py): the watchdog nudges at 20 turns with no file written; a first version by 12 leaves room to refine, and at 12 the write-deadline hint arrives
MAX_CHARS = 10_500                # choice: under the agent's 12,000-character tool page, so one read_file returns the whole file
CONFIG_MAX = 1 << 20              # config.json is a few kilobytes; this bound only stops a hostile file
SOURCE_MAX = 1 << 20              # a transformers configuration module is tens of kilobytes
SKELETON_MAX = 1 << 20
MAX_CONFIG_FIELDS = 40            # per config section
MAX_LIST = 32                     # elements of a list kept in the summary
MAX_STR = 60
MAX_NOTES = 12
MAX_NOTE_CHARS = 300
MAX_KEY_CHARS = 80                # a config key, a class name or a field name longer than this is counted and not copied into the file
MAX_LEFT_NAMES = 100              # names listed in a `*_left_out` list; the rest are counted
MAX_INDEX_FIELDS = 24             # id-list fields described in `index_fields`; the rest are counted
MAX_SHARDS_LISTED = 8
MAX_SCHEDULE = 4096
MAX_NAME_CHARS = 200              # a tensor name longer than this is counted and not listed: the header reader bounds names only by the header's size
MAX_NUMBER_DIGITS = 6             # a numeric name component of more digits is no layer or expert number and stays in the pattern as it is
# Config fields that hold layer, block, expert or position numbers. `eos_token_id` lists hold token ids and do not match.
INDEX_FIELD = re.compile(r"(layer|block|expert|position).*(_ids?|_idx|_indices|_numbers?)$|_layers?$")
CONVENTION = re.compile(r"\b(?:one|zero|1|0)[- ]?(?:based|indexed)\b", re.I)
_FIELD_HEAD = re.compile(r"^\s{4,8}(\w+) \(`")                       # `    name (`type`, *optional*):` in a configuration docstring
_CLASS_HEAD = re.compile(r"^class (\w+)")                            # a configuration module holds a text, a vision and a wrapper class
_ONE_BASED = re.compile(r"\b(?:one|1)[- ]?(?:based|indexed)\b", re.I)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
_MODEL_TYPE = re.compile(r"[a-z0-9_]{1,64}")
_SHOWN_MODEL_TYPE = re.compile(r"[A-Za-z0-9_.-]{1,64}")              # a model type the file shows; anything else is counted and not copied
_KEY = re.compile(rf"[A-Za-z_][A-Za-z0-9_]{{0,{MAX_KEY_CHARS - 1}}}")  # a config key the file copies: a plain identifier of bounded length
ELISION = " [...] "
SUB_CONFIG_SUFFIXES = ("text", "vision", "audio")                    # `qwen4_exp_text` lives in models/qwen4_exp/configuration_qwen4_exp.py
_NUMBER_IN_NAME = re.compile(rf"(?<=\.)\d{{1,{MAX_NUMBER_DIGITS}}}(?=\.|$)")
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

def _note_text(text: str) -> tuple[str, bool, bool]:
    """(the note, whether the description was shortened, whether the sentence that states the convention was itself cut). The note is at
    most MAX_NOTE_CHARS and always holds the whole phrase that matched CONVENTION. A description that fits is kept as it is. A longer one
    keeps its start when the convention is stated early; otherwise it keeps the start that fits, an elision mark and the sentence that
    states the convention (cut around the phrase only when that sentence alone is longer than the cap). Cutting the description at a fixed
    length lost a convention stated late, and the field then looked as if it had none: `ple_layer_ids` is one-indexed."""
    if len(text) <= MAX_NOTE_CHARS:
        return text, False, False
    found = CONVENTION.search(text)
    if found is None or found.end() <= MAX_NOTE_CHARS:
        return text[:MAX_NOTE_CHARS], True, False
    sentence, offset = text, 0
    for part in _SENTENCE_END.split(text):
        start = text.index(part, offset)
        offset = start + len(part)
        if start <= found.start() < offset:
            sentence = part
            break
    if len(sentence) <= MAX_NOTE_CHARS:
        room = MAX_NOTE_CHARS - len(sentence) - len(ELISION)
        lead = text[:room].rsplit(" ", 1)[0].rstrip() if room >= 20 else ""
        return (lead + ELISION + sentence if lead else sentence), True, False
    match = CONVENTION.search(sentence)
    width = MAX_NOTE_CHARS - 6                                          # room for an elision mark at each end
    middle = (match.start() + match.end()) // 2
    lo = max(0, min(middle - width // 2, len(sentence) - width))
    hi = lo + width
    return ("..." if lo else "") + sentence[lo:hi] + ("..." if hi < len(sentence) else ""), True, True


def convention_notes(source, left_out: list | None = None) -> list[dict]:
    """The fields of a configuration docstring whose description says how they are numbered (one-indexed, zero-based and the like):
    [{"class", "field", "note"}] in file order, at most MAX_NOTES, each note at most MAX_NOTE_CHARS (see `_note_text`; a note whose
    description was shortened has `cut`, one whose convention sentence was cut has `convention_cut`). A configuration module holds several
    classes (the text, vision and wrapper configurations), so a field is noted once per class and each note names its class. A field the
    cap leaves out is appended to `left_out` when that list is given, so the file can say how many there were. A class or field name
    longer than MAX_KEY_CHARS is not copied: it is counted in `left_out` under its first MAX_KEY_CHARS characters."""
    if not isinstance(source, str):
        return []
    notes, seen, cls, field, body = [], set(), None, None, []

    def close():
        text = " ".join(body).strip()
        if not (field and cls and text and CONVENTION.search(text)) or (cls, field) in seen:
            return
        seen.add((cls, field))
        if len(notes) >= MAX_NOTES or len(cls) > MAX_KEY_CHARS or len(field) > MAX_KEY_CHARS:
            if left_out is not None:
                left_out.append(field[:MAX_KEY_CHARS])
            return
        note, cut, convention_cut = _note_text(text)
        notes.append({"class": cls, "field": field, "note": note, **({"cut": True} if cut else {}),
                      **({"convention_cut": True} if convention_cut else {})})

    for line in source.splitlines():
        head_class = _CLASS_HEAD.match(line)
        head = _FIELD_HEAD.match(line)
        if head_class:
            close()
            cls, field, body = head_class.group(1), None, []
        elif head:
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


def _class_kind(name: str) -> str:
    """Which config section a configuration class describes, from its name: `text`, `vision`, `audio`, or `top` (the wrapper class)."""
    lower = name.lower()
    return next((k for k in ("text", "vision", "audio") if k in lower), "top")


def _section_kind(where: str, model_type) -> str:
    """The class kind that documents a config section. A nested `text_config` is text. A top-level config whose own model type ends in
    `_text`, `_vision` or `_audio` is that section itself (a flat checkpoint config such as `qwen4_exp_text`), so a wrapper class that
    documents the same field does not win. Any other top-level config is the wrapper (`top`)."""
    if where == "text_config":
        return "text"
    if isinstance(model_type, str):
        tail = model_type.rpartition("_")[2]
        if tail in SUB_CONFIG_SUFFIXES and model_type.rpartition("_")[0]:
            return tail
    return "top"


def _pick_convention(notes: list, field: str, where: str, model_type=None) -> str:
    """The convention text for `field` of the config section `where` ("config" for the top level, "text_config"): the note of the class that
    matches the section, with its class name first. A class of another section that states a different numbering is named too, so
    the porter sees the disagreement. When no class of the section documents the field, the classes that do are named and the note says
    they belong to another section."""
    candidates = [n for n in notes if n["field"] == field]
    if not candidates:
        return UNKNOWN_CONVENTION
    wanted = _section_kind(where, model_type)
    mine = [n for n in candidates if _class_kind(n["class"]) == wanted]
    if not mine:
        listed = "; ".join(f"{n['class']}: {n['note']}" for n in candidates[:2])
        return (f"the reference states this only in classes for other sections than {where} ({listed}): compare these ids with the "
                "layer numbers of the tensors below before using them as layer indexes")
    first = mine[0]
    text = f"{first['class']}: {first['note']}"
    first_one = bool(_ONE_BASED.search(first["note"]))
    for other in (n for n in candidates if n is not first and bool(_ONE_BASED.search(n["note"])) != first_one):
        text += f" | A different class, {other['class']}, states: {other['note'][:120]}"
    return text


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


def reference_source(root, model_type, kind: str = "configuration") -> str | None:
    """The text of `models/<model_type>/<kind>_<model_type>.py` (kind `configuration` or `modeling`) under the transformers directory `root`, or None. A model type that
    ends in _text, _vision or _audio is also tried without that suffix. The model type must be lowercase letters, digits and underscores,
    so it cannot name a path outside `root`."""
    if root is None or not isinstance(model_type, str) or not _MODEL_TYPE.fullmatch(model_type) or kind not in ("configuration", "modeling"):
        return None
    names = [model_type]
    head, _, tail = model_type.rpartition("_")
    if head and tail in SUB_CONFIG_SUFFIXES:          # the text part of a family is declared in the family's own module
        names.append(head)
    for name in names:
        try:
            data = safeio.read_bytes(Path(root) / "models" / name / f"{kind}_{name}.py", max_bytes=SOURCE_MAX)
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


def _scalars(section: dict) -> tuple[dict, list, int, int]:
    """The small scalar and list-of-scalar fields of a config section (sorted, at most MAX_CONFIG_FIELDS) and, for the rest:

      * the names left out (at most MAX_LEFT_NAMES; the number beyond that is counted),
      * the number of keys that are not copied at all because they are not a plain identifier of at most MAX_KEY_CHARS characters.
    A key comes from the model's own config, and the agent reads this file first, so a long or odd key is never copied: neither as a field
    nor in a left-out list. Values are capped by MAX_STR and MAX_LIST."""
    kept, left, more, unlisted = {}, [], 0, 0
    for key in sorted(section):
        if not _KEY.fullmatch(key):
            unlisted += 1
            continue
        v = section[key]
        simple = isinstance(v, (bool, int, float)) or (isinstance(v, str) and len(v) <= MAX_STR)
        if isinstance(v, list) and len(v) <= MAX_LIST and all(isinstance(x, (bool, int, float)) or (isinstance(x, str) and len(x) <= MAX_STR) for x in v):
            simple = True
        if simple and len(kept) < MAX_CONFIG_FIELDS:
            kept[key] = v
        elif len(left) < MAX_LEFT_NAMES:
            left.append(key)
        else:
            more += 1
    return kept, left, more, unlisted


def _is_flag_list(values: list) -> bool:
    """A list of three or more 0s and 1s is one flag per layer (`no_rope_layers: [1, 1, 1, 0]`), not a set of layer ids: a set of ids has no
    repeats, and a list of only 0 and 1 with three entries or more must repeat. Two entries (`[0, 1]`) stay ids."""
    return len(values) >= 3 and all(isinstance(x, int) and not isinstance(x, bool) and x in (0, 1) for x in values)


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


def _schedule(text_config: dict) -> tuple:
    """(the layer schedule grouped by type, or None) and a note that says why a `layer_types` the config holds was not summarised (else None)."""
    types = text_config.get("layer_types")
    if types is None:
        return None, None
    if not isinstance(types, list) or not types:
        return None, "layer_types is in the config but is not a non-empty list, so no schedule is given"
    if len(types) > MAX_SCHEDULE:
        return None, f"layer_types is in the config but has more than {MAX_SCHEDULE} entries, so no schedule is given"
    if not all(isinstance(t, str) and len(t) <= MAX_STR for t in types):
        return None, "layer_types is in the config but holds something other than short strings, so no schedule is given"
    by_type: dict = {}
    for i, t in enumerate(types):
        by_type.setdefault(t, []).append(i)
    return {t: _ranges(ix) for t, ix in by_type.items()}, None


def _groups(tensors) -> tuple:
    """(one line per group of tensors with one name pattern (numbers in the name replaced by {i}), one dtype and one shape, and the
    count of tensors whose names were too long to list). A name over MAX_NAME_CHARS is never read for numbers or copied into a line."""
    groups: dict = {}
    too_long = 0
    for t in sorted(tensors, key=lambda t: t.name):
        if len(t.name) > MAX_NAME_CHARS:
            too_long += 1
            continue
        first = _NUMBER_IN_NAME.search(t.name)
        key = (_NUMBER_IN_NAME.sub("{i}", t.name), t.dtype, tuple(t.shape))
        g = groups.setdefault(key, [0, []])
        g[0] += 1
        if first:
            g[1].append(int(first.group()))
    out = []
    for (pattern, dtype, shape), (count, firsts) in groups.items():
        out.append(f"{pattern}  {dtype} {list(shape)}  x{count}" + (f"  layers {_ranges(firsts)}" if firsts else ""))
    return out, too_long


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


def read_config(snapshot) -> dict:
    """The checkpoint's config.json as a dict (bounded, no links followed); PortFactsError when it is not usable."""
    return _config(Path(snapshot))


EXCERPT_FILE = "reference_excerpts.md"
EXCERPT_CLASSES_LISTED = 12
EXCERPT_LEFT_OUT_LISTED = 6
EXCERPT_TEXT_CHARS = 100


def _excerpts_field(summary) -> dict | None:
    """The facts file's pointer to the excerpts file, rebuilt from the summary `port_excerpts.Excerpts.summary()` returns so that nothing in it
    is longer than the caps here: the file name is fixed, the class names are plain identifiers, the left-out sentences are cut."""
    if not isinstance(summary, dict):
        return None
    size, total = summary.get("bytes"), summary.get("classes_total")
    classes = [c for c in summary.get("classes", []) if isinstance(c, str) and _KEY.fullmatch(c)][:EXCERPT_CLASSES_LISTED]
    left = [x[:EXCERPT_TEXT_CHARS] for x in summary.get("left_out", []) if isinstance(x, str)][:EXCERPT_LEFT_OUT_LISTED]
    field = {"file": EXCERPT_FILE, "bytes": size if isinstance(size, int) and not isinstance(size, bool) else None,
             "classes": classes, "classes_total": total if isinstance(total, int) and not isinstance(total, bool) else len(classes),
             "left_out": left,
             "note": (f"{EXCERPT_FILE} holds the constructor signature, forward method and small helpers of these reference classes. "
                      "Read it first and write a first port_model.py from it before you read any other file.")}
    more = summary.get("left_out_total")
    if isinstance(more, int) and not isinstance(more, bool):
        field["left_out_total"] = more
    return field


def build(snapshot, *, reference_source=None, reference_lookup=None, excerpts=None) -> dict:
    """The facts of a checkpoint directory as a JSON-ready dict. `reference_source` is the text of the reference's configuration module;
    `reference_lookup(model_type)` finds it when it is not given. `excerpts` is the summary of the excerpts file (port_excerpts), which the
    facts then point at. Raises PortFactsError when the config or a header cannot be read."""
    snap = Path(snapshot)
    config = _config(snap)
    try:
        headers = residency.read_headers(snap)
    except residency.ResidencyError as exc:
        raise PortFactsError(str(exc)) from None
    nested = config.get("text_config") if isinstance(config.get("text_config"), dict) else None
    top, top_left, top_more, top_unlisted = _scalars({k: v for k, v in config.items() if k != "text_config"})
    text, text_left, text_more, text_unlisted = _scalars(nested) if nested is not None else ({}, [], 0, 0)
    model_types = [m for m in (config.get("model_type"), (nested or {}).get("model_type")) if isinstance(m, str)]
    shown_types = [m for m in model_types if _SHOWN_MODEL_TYPE.fullmatch(m)]       # the model's own text is shown only as a short identifier
    if reference_source is None and reference_lookup is not None:
        reference_source = next((s for s in (reference_lookup(m) for m in shown_types) if s), None)
    unlisted: list = []
    notes = convention_notes(reference_source, unlisted)
    index_fields, flag_fields, index_more = [], [], 0
    section_types = {"config": config.get("model_type"), "text_config": (nested or {}).get("model_type")}
    for where, section in (("config", top), ("text_config", text)):
        for key, v in section.items():
            if INDEX_FIELD.search(key) and isinstance(v, list) and v and all(isinstance(x, int) and not isinstance(x, bool) for x in v):
                if _is_flag_list(v):
                    flag_fields.append(key)
                elif len(index_fields) < MAX_INDEX_FIELDS:
                    index_fields.append({"field": key, "in": where, "values": v, "convention": _pick_convention(notes, key, where, section_types[where])})
                else:
                    index_more += 1
    lines, long_names = _groups(headers.tensors)
    schedule, schedule_note = _schedule(nested if nested is not None else config)
    facts = {"schema": SCHEMA, "model_type": shown_types[0] if shown_types else None,
             "model_type_not_listed": bool(model_types) and not shown_types, "config": top,
             "config_left_out": sorted(top_left), "config_left_out_more": top_more, "config_keys_not_listed": top_unlisted}
    if nested is not None:
        facts.update({"text_config": text, "text_config_left_out": sorted(text_left), "text_config_left_out_more": text_more,
                      "text_config_keys_not_listed": text_unlisted})
    pointer = _excerpts_field(excerpts)
    if pointer is not None:
        facts["reference_excerpts"] = pointer
    facts.update({"layer_schedule": schedule, "layer_schedule_note": schedule_note, "index_fields": index_fields,
                  "index_fields_left_out": index_more, "layer_flag_fields": sorted(flag_fields),
                  "conventions": notes, "conventions_left_out": len(unlisted), "weights": _weights(snap, headers),
                  "tensor_groups": len(lines), "tensors": lines, "tensors_left_out": 0, "tensors_with_long_names": long_names,
                  "reading": ("Tensor lines are one per name pattern ({i} stands for a number in the name) with dtype, shape, how many "
                              "tensors and, when the name has a number, which numbers. Layer ids in the config may be counted from 0 or "
                              "from 1: see index_fields. layer_flag_fields hold one 0 or 1 per layer, not layer ids. A config key that is "
                              "not a plain identifier of at most 80 characters is counted (`*_keys_not_listed`) and never copied.")})
    return _fit(facts)


def _fit(facts: dict) -> dict:
    """Drop tensor lines from the end, then config fields, until the file fits MAX_CHARS. Every field dropped is named in the matching
    `*_left_out` list (up to MAX_LEFT_NAMES names; the rest are counted in `*_left_out_more`), so the coder can tell a setting the summary
    dropped from one the checkpoint does not have.

    The header reader allows a million tensors, so the tensor lines are cut in one pass over their own encoded lengths (each costs its
    text, a four-space indent, a comma and a newline) to the longest prefix that can fit; the exact check then removes at most the few lines
    that the rest of the file needs. Serialising the whole file once per removed line was quadratic in the number of groups."""
    def size():
        return len(json.dumps(facts, indent=2)) + 1
    lines, total, already = facts["tensors"], len(facts["tensors"]), facts["tensors_left_out"]
    facts["tensors"], facts["tensors_left_out"] = [], already + total          # measured with every line counted as left out: the digits are the longest
    room = MAX_CHARS - size() - 2                                              # the rest of the file, once; a non-empty list is two characters longer than `[]`
    facts["tensors"] = lines
    used = keep = 0
    for line in lines:
        cost = len(json.dumps(line)) + 6
        if used + cost > room:
            break
        used += cost
        keep += 1
    del lines[keep:]
    facts["tensors_left_out"] = already + total - keep
    while size() > MAX_CHARS and lines:
        lines.pop()
        facts["tensors_left_out"] += 1
    for section, left_out in (("text_config", "text_config_left_out"), ("config", "config_left_out")):
        while size() > MAX_CHARS and facts.get(section):
            key, _ = facts[section].popitem()
            if len(facts[left_out]) < MAX_LEFT_NAMES:                     # the list of names stays capped; the rest are counted
                facts[left_out] = sorted([*facts[left_out], key])
            else:
                facts[left_out + "_more"] += 1
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


def stage_3_note(*, chips: int, copied: bool) -> str:
    """What the supervisor adds to the coder's prepare task for stage 3 of a full port (issue 326): the port passed stage 2 on one board and
    this stage runs the same check on `chips` chips. The generic prepare task names only hw_test.json and handoff.json."""
    parts = [f"This stage runs the port that passed stage 2 on one board again, on {chips} chips (a mesh of 1 x {chips}, which the check opens "
             f"with the 1-D fabric on, as it does for any stage with more than one chip). The check is the supervisor's: when you finish, it "
             f"puts its own `stages/3/port_check.py` back and rewrites `hw_test.json` to its fixed command (`{full_port.command(3)}`). Your "
             f"work in this step is the port under `stages/3/port/`; write nothing under `stages/2/`."]
    if copied:
        parts.append("`stages/3/port/` already holds the files of the stage 2 port, copied by the supervisor. Change what a mesh needs: how "
                     "each weight is placed (replicated, or sharded across the devices), where a collective is required, and how "
                     "`load`, `prefill` and `decode_step` use the mesh device they are given. A port that runs on one chip and ignores "
                     "the others can still pass, but it is not what this stage is for. Leave what already works.")
    else:
        parts.append("The supervisor could not copy the stage 2 port (the ledger says why). Copy the files of `stages/2/port/` into "
                     "`stages/3/port/` yourself, then adapt them to the mesh.")
    parts.append(f"When the port is ready, write `hw_test.json` as {{\"command\": \"{full_port.command(3)}\", \"deadline_s\": "
                 f"{full_port.DEADLINE_S}}} and `handoff.json`, then reply with a short summary. Do not run the check yourself.")
    return " ".join(parts)


def tools_text(stage: int = 2, *, facts: bool = True) -> str:
    """The stage tools (orchard/skills/full-port-templates/stage_tools.py, issue 505) as the coder's task names them: the exact command and one line each,
    and the optional `layer_outputs` hook that `layer-diff` needs. The commands are the ones the watchdog counts by tool name and arguments."""
    base = f"python3 stages/{stage}/stage_tools.py"
    lines = [
        "Fixed tools. Run them from the run directory in place of `cat`, `head`, `grep` or `python3 -c` on those files. Each prints a short report.",
        f"- `{base} last-check`: which check failed, where, the numbers and the evidence file.",
        f"- `{base} layer-diff`: after a numerics failure, the first layer whose output differs from the CPU reference (needs the optional hook below).",
        f"- `{base} facts KEY`: one value of {FACTS_NAME} by dotted path, for example `facts config.hc_count`; `facts --find TEXT` searches keys and text.",
        f"- `{base} show FILE --lines A:B`: a range of a file in the run directory, with line numbers.",
        f"- `{base} edit stages/{stage}/port/port_model.py --old TEXT --new TEXT`: after your first write, change a port file with this. Do not write the whole file again.",
        "`edit` needs TEXT to occur once (add the lines around it if it does not). For several lines, write the old and the new text with write_file, a new "
        "file name for each edit, and pass `--old-file` and `--new-file`.",
        "Optional hook for `layer-diff`: define `layer_outputs(model, ids)` in `port_model.py`. It runs the prompt `ids` once and returns the hidden state "
        "after each decoder layer, in order, as a list of torch tensors or numpy arrays. The check calls it only when the agreement is below "
        f"{full_port.TOP1_MIN}, after the comparison, in the same run; its result cannot change the verdict, and an error in it is printed and ignored."]
    if not facts:                    # no facts file was written for this run, so no line may name it
        lines = [l for l in lines if "`facts" not in l]
    return "\n".join(lines)


def prepare_note(*, facts: bool, skeleton: bool, excerpts: bool = False) -> str:
    """What the supervisor adds to the coder's prepare task for a full port. The generic prepare task names only hw_test.json and
    handoff.json; the port itself is described only in the skill, and the coder explored and wrote nothing."""
    parts = [f"This stage's check is the supervisor's: when you finish, it puts its own `stages/2/port_check.py` back and rewrites "
             f"`hw_test.json` to its fixed command (`{full_port.COMMAND}`). Your work in this step is the port module, "
             f"`stages/2/port/port_model.py`."]
    if excerpts:
        parts.append(f"Read `stages/2/{EXCERPT_FILE}` first, before you read any other file. It holds the source of the reference classes "
                     "your model uses (constructor signature, `forward` and small helpers), taken from the installed transformers "
                     "package. Write a first `port_model.py` from it within your first turns, before you open any other file or "
                     "page through the reference module yourself.")
    if facts:
        parts.append(f"Read `stages/2/{FACTS_NAME}` {'next' if excerpts else 'first'}. It holds the config fields, the layer schedule, every tensor's dtype and shape "
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
    parts.append(tools_text(2, facts=facts))
    return "\n\n".join(parts)


def write_deadline_hint(turns: int, *, port_is_skeleton: bool, excerpts: bool) -> str:
    """The text of the one hint a stage 2 step gets when it has read for `turns` turns and written nothing (watchdog FirstWriteDeadline).
    While the port is still the supervisor's skeleton it names the file and the four functions to fill in, and where the source to copy
    from is. Once the port has been edited, the step is a repair or a finish step, and the hint says only to write what the step asks for."""
    if not port_is_skeleton:
        return (f"{turns} model turns have passed and you have written no file. Stop reading. Make the edit this step asks for now, "
                "or say what blocks you.")
    source = (f"`stages/2/{EXCERPT_FILE}` holds the source of the reference classes, and `stages/2/{FACTS_NAME}` holds the shapes"
              if excerpts else f"`stages/2/{FACTS_NAME}` holds the shapes")
    return (f"{turns} model turns have passed and `stages/2/port/port_model.py` is still the skeleton. Write it now with the write_file tool: "
            "replace the NotImplementedError bodies of `load`, `prefill`, `decode_step` and `close`, keep `read_tensors`, and put the layer "
            f"code in helper functions above them. {source}; you have everything you need in your context. An incomplete first version is "
            "what this step needs. Do not open another reference file before you have written it.")
