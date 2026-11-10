# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The thin repair of stage 2 of a full port (issue 677, arm E2): a failed board test is repaired by one chat request per failing function.

Why. In arm E (the thin prepare step, PR 689) both board checks failed at prefill with `AttributeError: 'dict' object has no attribute 'config'`:
the stub bodies treated the port's `model` dict as an object. The old agent loop then ran out of 60 turns on a one-line fault. The check's own
failure line names the file, the line, the code on that line and the error. That is enough to find the function that holds it, to ask for that
function alone, and to add the hint for the error shapes the arms have shown.

What. `run_round` (called by the supervisor's repair rounds when the run has `--thin-port`):
1. reads the check's failed line (`parse_failed`: one strict, bounded pattern; a failure that is not a Python error in port_model.py is not
   handled here and the round goes to the agent loop as before);
2. finds the top-level function of port_model.py that holds the failing line (`function_at`; a line outside any function is not handled here);
3. sends one request (orchard/stub_prompts.py `repair_request`: the model/constants section, the check's report, the function, the rules) with a
   hint for known error shapes (`HINTS`, one small table), splices the reply (`stub_prompts.splice`, with its guard on what a reply may
   contain) and writes the port;
4. a refused reply gets one more request with the reason added and a cooler sampling profile; a second refusal hands the round to the agent loop.

Bounds: `MAX_ROUNDS` thin rounds per attempt, `REQUESTS_PER_ROUND` requests per round, a prompt-character cap per round, the stub token cap and the
request timeout of orchard/thin_port.py. The result holds no reply text.
"""
from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass

from orchard import op_cheats, port_repair, port_template, safeio, stub_prompts, thin_port

MAX_ROUNDS = 6                    # thin repair rounds per attempt: round 7 and later go to the agent loop (a round is a request and a board check)
REQUESTS_PER_ROUND = 2            # the first request and one retry after a refusal
ROUND_PROMPT_CHARS_CAP = 200_000  # all requests of one round together (one request is about 20K characters)
FAILED_LINE_MAX = 2_000
PORT_NAME = thin_port.PORT_NAME

# The check's failure text: `the port failed at '<op>': "[port_model.py:<line>: <code>] <ErrorType>: <message>"` (the quote is " or ').
_FAILED = re.compile(r"^the port failed at '(?P<op>[A-Za-z_]\w{0,40})': (?P<q>[\"'])\[(?P<file>[\w./-]{1,80}\.py):(?P<line>\d{1,6}): "
                     r"(?P<code>.{0,400}?)\] (?P<etype>[A-Za-z_][\w.]{0,60}): (?P<msg>.{0,600})", re.S)       # the end is open: a long message is cut by failed_line, and the quote may be gone

# Hints for the error shapes the arms showed: (error type pattern, message pattern, text). `{0}` is the first group of the message pattern and
# `{names}` the port's top-level function names. One place, each entry tested.
HINTS = (
    (re.compile(r"^AttributeError$"), re.compile(r"'dict' object has no attribute '(\w+)'"),
     "`model` is a plain dict: write model['dev'], model['host'], model['mesh_device'], never model.{0}. The config values are in the module "
     "constant CONFIG (CONFIG['head_dim']), and the sizes in N_LAYERS, HIDDEN and VOCAB."),
    (re.compile(r"^NameError$"), re.compile(r"name '(\w+)' is not defined"),
     "`{0}` does not exist in this port. Import what you need inside the function, or call a helper the port already has. The port's "
     "top-level functions are: {names}."),
    (re.compile(r"^TypeError$"), re.compile(r"is not a Module subclass|ModuleList"),
     "This port is not a torch module: use plain functions and the `model` dict, and do not subclass torch.nn.Module or build ModuleList."),
    # The shapes below are failures the arm E2 run showed on the chip (issue 677).
    (re.compile(r"^ImportError$"), re.compile(r"attempted relative import"),
     "The port is one module that the check imports by its file name: it has no package, so there is no relative import. The helpers are "
     "functions in this same file: {names}. Call them by name."),
    (re.compile(r"^TypeError$"), re.compile(r"device=nanobind|nb_bound_method|\.device\b"),
     "An activation `x` is a ttnn tensor with no torch device: do not read x.device or pass it to torch. Build host tensors with plain torch "
     "(no device argument) and upload them with _upload, or read a tensor back with _to_host first."),
    (re.compile(r"^RuntimeError$"), re.compile(r"padded widths must match|gamma.*padded_shape"),
     "ttnn.rms_norm needs a weight whose last dimension equals the input's. For a norm with a group_size smaller than the input width "
     "(the hyper-connection norm), reshape x to [..., width / group_size, group_size], norm over the last dimension with the group's "
     "weight, and reshape back."),
    (re.compile(r"^KeyError$"), re.compile(r"^'(?P<key>\w+)'$"),
     "`{0}` is not a key of CONFIG. CONFIG has exactly these keys: {config_keys}. Read the value from the key that holds it."),
    (re.compile(r"^KeyError$"), re.compile(r"\{i\}"),
     "A key that still holds `{{i}}` is a pattern, not a tensor name. Look a weight up in model['dev'] by its real name with the layer number "
     "filled in (for example 'model.layers.0.…'). _pattern(name) turns a real name into the pattern form that PLACEMENT is keyed by; call "
     "_rms_norm(model, name, x) for an RMS norm and do not index NORM_EPS yourself."),
    # The shapes below are failures the arm E7 run showed on the chip (issue 677).
    (re.compile(r"^TypeError$"), re.compile(r"bad operand type for unary -"),
     "A ttnn tensor has no unary minus. Use ttnn.neg(x) or ttnn.multiply(x, -1.0). Rotate-half is ttnn.concat([ttnn.neg(x2), x1], dim=-1) "
     "(the graph-fusing skill of tenstorrent/skills shows it, with ttnn.experimental.rotary_embedding as the fused op)."),
    (re.compile(r"^IndexError$"), re.compile(r"Too many indices for tensor of dimension"),
     "A ttnn activation is 4-D, [1, 1, S, F]: index it with at most four indices. For a loop over tokens, read the tensor back with _to_host "
     "first and work on the torch tensor, or slice with ttnn.slice on the sequence dimension."),
    (re.compile(r"^RuntimeError$"), re.compile(r"slice_start|padded_shape\(\)\[i\]"),
     "A slice must lie inside the padded shape. For partial rotary application, take the rotary part x[..., :rot] and the rest x[..., rot:] "
     "with ttnn.slice, where rot is the cos/sin table's width, and check each end against the last dimension. Rotate-half on the rotary part is "
     "ttnn.concat([ttnn.neg(x2), x1], dim=-1) (graph-fusing skill, tenstorrent/skills)."),
    (re.compile(r"^RuntimeError$"), re.compile(r"same shape except in the concat dim|concat"),
     "ttnn.concat needs pieces whose shapes match in every dimension except the concat dimension. For partial rotary application, rotate only "
     "the first rot columns ([1, 1, S, rot]), keep the remaining columns ([1, 1, S, F - rot]) unchanged, and concatenate the two on dim=-1."),
    # The shapes below are failures the arm E8 run showed on the chip (issue 677).
    (re.compile(r"^TypeError$"), re.compile(r"ttnn\.repeat\(\)"),
     "ttnn.repeat takes the tensor and one repeat shape, not one argument per dimension: ttnn.repeat(x, ttnn.Shape([1, 1, 1, n])) repeats the "
     "last dimension n times."),
    (re.compile(r"^KeyError$"), re.compile(r"^'model\.[\w.]+\.weight'$"),
     "model['dev'] is keyed by the checkpoint tensor's own name. A tensor that is stored without a '.weight' suffix (A_log, dt_bias and other "
     "bare parameters) has no suffix in its key: look the name up exactly as the tensor list in the port facts spells it."),
)


@dataclass
class Failure:
    op: str
    file: str
    line: int
    code: str
    etype: str
    message: str


@dataclass
class Round:
    status: str                      # "spliced", "refused" (two refusals), "fallback" (the agent loop takes the round) or "error"
    function: str | None = None
    etype: str | None = None
    hint: bool = False
    requests: int = 0
    prompt_chars: int = 0
    completion_tokens: int = 0
    reason: str = ""


def parse_failed(text) -> Failure | None:
    """The check's last failed line as a Failure, or None when it is not a Python error at a line of a port file."""
    if not isinstance(text, str):
        return None
    line = port_repair.failed_line(text, FAILED_LINE_MAX)
    m = _FAILED.match(line)
    if not m:
        return None
    msg = m["msg"]
    if msg.endswith(m["q"]):
        msg = msg[:-1]                      # the closing quote of the check's line, when the line was not cut
    return Failure(m["op"], m["file"], int(m["line"]), m["code"], m["etype"], msg)


def function_at(port_text: str, line: int) -> str | None:
    """The name of the top-level function whose source holds `line`, or None (module top level, a class, a line past the file, no parse)."""
    try:
        tree = ast.parse(port_text)
    except (SyntaxError, ValueError, RecursionError):
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.lineno <= line <= (node.end_lineno or node.lineno):
            return node.name
    return None


def hint_for(failure: Failure, port_text: str) -> str | None:
    """The hint for the first table entry that matches the error, with the group of its message pattern and the port's function names filled in."""
    try:
        tree = ast.parse(port_text)
        names = ", ".join(n.name for n in tree.body if isinstance(n, ast.FunctionDef))[:1200] or "none"
    except (SyntaxError, ValueError, RecursionError):
        tree, names = None, "none"
    keys = stub_prompts.config_keys(tree) if tree is not None else []
    for etype, message, text in HINTS:
        if etype.match(failure.etype.split(".")[-1]):
            m = message.search(failure.message)
            if m:
                if "{config_keys}" in text and (not keys or m.group(1) in keys):
                    continue                    # the key is in CONFIG, or the port has no CONFIG literal: this entry does not apply
                groups = m.groups() or ("",)
                return text.format(*groups, names=names, config_keys=", ".join(keys)[:1500])
    return None


def run_round(*, stage_dir, k: int, raw_output: str, post, endpoint: str, model: str, sampling: dict, thinking: bool = False,
              sheet=None, clock=time.time) -> Round:
    """One thin repair round. Never raises for a coder or input problem; the status says what happened."""
    from pathlib import Path
    port_dir = Path(stage_dir) / "port"
    if k > MAX_ROUNDS:
        return Round("fallback", reason=f"round {k} is past the {MAX_ROUNDS} thin rounds of an attempt")
    failure = parse_failed(raw_output)
    if failure is None:
        return Round("fallback", reason="the check's failure is not a Python error at a line of a port file")
    if failure.file != PORT_NAME:
        return Round("fallback", etype=failure.etype, reason="the failing line is in another file than port_model.py")
    try:
        text = safeio.read_bytes(port_dir / PORT_NAME, max_bytes=thin_port.PORT_FILE_MAX).decode("utf-8", "replace")
    except (safeio.SafeReadError, OSError, ValueError):
        return Round("fallback", etype=failure.etype, reason="port_model.py cannot be read")
    function = function_at(text, failure.line)
    if function is None:
        return Round("fallback", etype=failure.etype, reason="the failing line is not inside a top-level function")
    if function in port_template.FIXED_HELPERS:
        # The helper is generated text that is the same in every port. The stub that called it passed it something it cannot take, and the check's
        # failure line holds one frame only, so the agent loop (which reads the whole traceback) takes the round.
        return Round("fallback", function=function, etype=failure.etype,
                     reason=f"the failing line is in a generated helper ({function}); a stub that calls it passed something it cannot take")
    try:
        if sheet is None:
            sheet = op_cheats.load_sheet(op_cheats.TTNN_SHEET)
        report = "FAILED: " + port_repair.failed_line(raw_output, 1500)
        request = stub_prompts.repair_request(text, function, last_check=report, sheet=sheet)
    except (op_cheats.SheetError, KeyError, ValueError, TypeError, SyntaxError, RecursionError) as exc:
        return Round("fallback", function=function, etype=failure.etype, reason=f"the request cannot be built ({exc.__class__.__name__})")
    hint = hint_for(failure, text)
    base_user = request["user"] + (f"\n\n## Hint for this error ({failure.etype})\n{hint}" if hint else "")
    base = {**sampling, "max_tokens": min(sampling.get("max_tokens", thin_port.STUB_MAX_TOKENS), thin_port.STUB_MAX_TOKENS)}
    url = endpoint.rstrip("/") + "/chat/completions"
    out = thin_port.Outcome("done")
    refusal, was_capped = None, False
    for attempt in range(1, REQUESTS_PER_ROUND + 1):
        user = base_user if refusal is None else base_user + f"\n\n## The previous reply was refused\n{refusal}\nWrite the function again so that it passes."
        chars = len(request["system"]) + len(user)
        if out.prompt_chars + chars > ROUND_PROMPT_CHARS_CAP:
            return _round("fallback", out, function, failure, hint, "the round's prompt size cap")
        changed = thin_port.retry_sampling(base, capped=was_capped) if attempt > 1 else None
        body = thin_port._body(model, request, user, {**base, **(changed or {})}, thinking)
        sent = thin_port._send(post, url, body, out, REQUESTS_PER_ROUND)
        if sent is None:
            return _round("fallback", out, function, failure, hint, "the round's request cap")
        data, error = sent
        out.prompt_chars += chars
        if error is not None:
            return _round("error", out, function, failure, hint, error)
        content, tokens, reason = thin_port._reply_of(data)
        out.completion_tokens += tokens or 0
        new = None
        if reason is None:
            try:
                new = stub_prompts.splice(text, function, content)
                if port_template.check_sources({PORT_NAME: new})["parse_errors"]:
                    new, reason = None, "the spliced port does not parse"
            except stub_prompts.SpliceError as exc:
                reason = str(exc)[:200]
        if new is not None:
            thin_port._write_port(port_dir, new)
            return _round("spliced", out, function, failure, hint, "")
        refusal, was_capped = reason, reason == thin_port.CAPPED_REASON
    return _round("refused", out, function, failure, hint, refusal or "")


def _round(status: str, out, function, failure, hint, reason: str) -> Round:
    return Round(status, function=function, etype=failure.etype, hint=bool(hint), requests=out.requests, prompt_chars=out.prompt_chars,
                 completion_tokens=out.completion_tokens, reason=reason[:300])
