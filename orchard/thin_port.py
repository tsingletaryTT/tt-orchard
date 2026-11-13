# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The thin prompt path of stage 2 of a full port (issue 677, arm E): the prepare step done with one chat request per stub.

Why. In the cp4 audition the coder's prepare step read 25K to 36K tokens of torch reference code before its first write, 85 to 96 percent of
the prompt tokens were repeated context, and the Coder-Next arms wrote host-torch ports that failed the device-ops gate (the design note
docs/superpowers/specs/2026-11-10-thin-prompt-path.md has the data). The port generator (orchard/port_template.py) writes a first port that
already makes `ttnn.linear` calls and leaves the blocks it cannot write as named stubs. This module asks the coder for one stub at a time.

What. `run` (or `run_with_inputs` when the inputs are already in hand) does, behind the supervisor's `--thin-port` option and for stage 2 of a
full port only:
1. generates `stages/2/port/port_model.py` from the checkpoint's config and tensor headers, the effective construction, stage 0's ops
   inventory and the verified cheat sheet (a port left by an earlier attempt that is not the shipped skeleton is kept, and only its remaining
   stubs are asked for);
2. for each stub, in file order, sends one request (orchard/stub_prompts.py `stub_request`: the shared helpers, the verified calls, the
   construction lines and the reference excerpt of that one class, the stub last) to the coder tier's chat endpoint with the tier's sampling,
   no tools and no tool loop, and splices the reply in (`splice`, which refuses a reply that is not one code block of imports and function
   definitions for that stub or that uses anything a port function has no use for);
3. retries a refused reply once with the refusal reason added to the request; a second refusal leaves the stub as it was and goes on;
4. ends with the port as it stands. The rest of stage 2 (the CPU device-ops pre-check, the board test, repair rounds by the agent loop,
   result derivation) is the existing flow and reads the port from the same place.

Bounds. A request cap (two per stub plus two), a cap on the total characters sent, a timeout per request, a reply size cap (stub_prompts), and
a check that the port directory holds only the file this step writes. A coder endpoint that fails ends the step with status "error" after one
retry of the identical request. The ledger gets one decision per request with sizes, seconds and an outcome. It never holds reply text.
Anything that stops the generator from planning (no reference interpreter, an unreadable snapshot, a port directory with other files) returns
status "skipped" with the reason, and the supervisor runs the ordinary agent prepare step.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from orchard import op_cheats, port_construction, port_facts, port_template, residency, safeio, stub_prompts

REQUEST_TIMEOUT_S = 900.0                # one request; a 27B model at 20 tok/s writes a stub of 2,000 tokens in under two minutes
PROMPT_CHARS_CAP = 1_500_000             # all requests of one step together: 15 stubs of 20K characters, with two attempts each, fit
RETRIES_PER_STUB = 1                     # a refused reply gets one more request with the reason added
STUB_MAX_TOKENS = 4096                   # one stub's reply: the largest stub arm E spliced took 2,509 tokens, and its five runaway replies each ran to the
                                         # tier's 16,384 and cost minutes; a request is capped at the smaller of the tier's cap and this
RETRY_TEMPERATURE_MAX = 0.3              # a retry samples at most this warm (never warmer than the tier asked)
RETRY_REPETITION_PENALTY = 1.1           # added to the retry of a reply that ran to the token cap (a repetition loop) when the tier has no larger one
CAPPED_REASON = "the reply was cut off at the token cap"
PORT_NAME = port_facts.SKELETON_NAME
PORT_FILE_MAX = 4 << 20
INPUT_MAX = port_template.INPUT_MAX
PORT_DIR_ENTRIES_MAX = 200
ALLOWED_IN_PORT_DIR = (PORT_NAME, "__pycache__")


class Skip(Exception):
    """The thin path cannot run for this stage; the message says why and holds no file content."""


@dataclass
class Inputs:
    config: dict
    construction: dict
    tensors: list
    inventory: dict | None = None
    excerpts: str | None = None
    construction_text: str | None = None


@dataclass
class Outcome:
    status: str                                   # "done" (stubs may be left), "error" or "skipped"
    detail: str = ""
    stubs_left: list = field(default_factory=list)
    filled: list = field(default_factory=list)
    requests: int = 0
    prompt_chars: int = 0
    completion_tokens: int = 0


def request_cap(n_stubs: int) -> int:
    """Requests one step may send: two per stub (the first and the one retry) plus two for a request repeated after a transport failure."""
    return n_stubs * (1 + RETRIES_PER_STUB) + 2


# ---- the inputs --------------------------------------------------------------------------------------------------------------------------

def _optional_json(path):
    try:
        return safeio.read_json(path, max_bytes=INPUT_MAX, expect=dict)
    except (safeio.SafeReadError, OSError, ValueError):
        return None


def _optional_text(path):
    try:
        return safeio.read_bytes(path, max_bytes=INPUT_MAX).decode("utf-8", "replace")
    except (safeio.SafeReadError, OSError):
        return None


def gather_inputs(*, run_dir, stage_dir, snapshot, python, construct=None) -> Inputs:
    """The generator's inputs from the run: the snapshot's config and tensor headers, the effective construction (run again under the
    reference interpreter, the same script the facts file's text comes from), and the optional inventory, excerpts and construction text.
    Raises Skip with a reason that holds no file content."""
    if not snapshot:
        raise Skip("the run has no model snapshot")
    snap = Path(snapshot)
    try:
        config = port_template.read_snapshot_config(snap)
    except (safeio.SafeReadError, OSError, ValueError) as exc:
        raise Skip(f"the snapshot's config.json cannot be read ({exc.__class__.__name__})") from None
    try:
        tensors = list(residency.read_headers(snap).tensors)
    except (residency.ResidencyError, safeio.SafeReadError, OSError, ValueError) as exc:
        raise Skip(f"the snapshot's tensor headers cannot be read ({exc.__class__.__name__})") from None
    data, why = (construct or port_construction.run)(python, str(snap))
    if data is None:
        raise Skip(f"the effective construction could not be built: {why[:160]}")
    return Inputs(config=config, construction=data, tensors=tensors,
                  inventory=_optional_json(Path(run_dir) / "stages" / "0" / "ops-inventory.json"),
                  excerpts=_optional_text(Path(stage_dir) / "reference_excerpts.md"),
                  construction_text=_optional_text(Path(stage_dir) / "effective_construction.txt"))


# ---- the port directory ------------------------------------------------------------------------------------------------------------------

def _port_dir_others(port_dir: Path) -> list:
    """Names under the port directory other than the file this step writes (and `__pycache__`). Raises Skip when it cannot be listed."""
    try:
        with safeio.pinned_root(port_dir) as pin:
            found = safeio.scan_beneath(pin, max_depth=3, max_entries=PORT_DIR_ENTRIES_MAX)
    except safeio.SafeReadError as exc:
        raise Skip(f"the port directory cannot be listed ({exc.__class__.__name__})") from None
    if found.incomplete:
        raise Skip("the port directory holds too many entries to list")
    return sorted({e.parts[0] for e in found.entries if e.parts and e.parts[0] not in ALLOWED_IN_PORT_DIR})


def _write_port(port_dir: Path, text: str) -> None:
    safeio.atomic_write(port_dir / PORT_NAME, text.encode("utf-8"), mode=0o644)


# ---- one request -------------------------------------------------------------------------------------------------------------------------

def _body(model: str, request: dict, user: str, sampling: dict, thinking: bool) -> dict:
    body = {"model": model, "messages": [{"role": "system", "content": request["system"]}, {"role": "user", "content": user}],
            "temperature": sampling.get("temperature", 0), "max_tokens": sampling.get("max_tokens", 4096), "stream": False}
    for key in ("top_p", "top_k", "frequency_penalty", "repetition_penalty"):
        if key in sampling:
            body[key] = sampling[key]
    if not thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return body


def _reply_of(data) -> tuple:
    """(content or None, completion tokens or None, a reason the reply cannot be used or None)."""
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return None, None, "the response has no choices[0].message"
    usage = data.get("usage") if isinstance(data, dict) and isinstance(data.get("usage"), dict) else {}
    tokens = usage.get("completion_tokens") if isinstance(usage.get("completion_tokens"), int) else None
    if choice.get("finish_reason") == "length":
        return None, tokens, CAPPED_REASON
    if not isinstance(content, str) or not content.strip():
        return None, tokens, "the reply has no content"
    return content, tokens, None


def retry_sampling(sampling: dict, *, capped: bool) -> dict:
    """The sampling of a retry: the temperature lowered to at most RETRY_TEMPERATURE_MAX (never raised), and after a reply that ran to the
    token cap a repetition penalty of at least RETRY_REPETITION_PENALTY. Returns the changed keys only (what the ledger records)."""
    changed = {"temperature": min(sampling.get("temperature", 0), RETRY_TEMPERATURE_MAX)}
    if capped:
        changed["repetition_penalty"] = max(sampling.get("repetition_penalty", 0), RETRY_REPETITION_PENALTY)
    return changed


def run(*, run_dir, stage_dir, snapshot, python, ledger, stage: int, post, endpoint: str | None, model: str | None, sampling: dict | None,
        thinking: bool = False, sheet=None, construct=None, connect=None, **bounds) -> Outcome:
    """Gather the inputs from the run and do the step. Anything that stops the generator from planning returns status "skipped".

    `connect(n_stubs)`, when given, is called once, after the plan is made and before the first request, and only when the port has stubs to
    send; it returns (endpoint, model, sampling) and replaces the three arguments. The supervisor uses it to start the coder only for a plan
    that needs it (issue 713). A plan with no stub to send never calls it."""
    try:
        given = gather_inputs(run_dir=run_dir, stage_dir=stage_dir, snapshot=snapshot, python=python, construct=construct)
    except Skip as exc:
        return Outcome("skipped", str(exc))
    return run_with_inputs(stage_dir=stage_dir, ledger=ledger, stage=stage, post=post, endpoint=endpoint, model=model, sampling=sampling,
                           inputs=given, thinking=thinking, sheet=sheet, connect=connect, **bounds)


STUBS_RECORD = "thin_stubs.json"          # in the stage directory (the port directory must hold only port_model.py)


def _record_stub_names(stage_dir, todo) -> None:
    """The names of the functions that are stubs, kept for the repair step: it sends a failure in a generated helper to the one *stub* that calls
    the helper, and the template's own functions (`_mlp`, `_forward`) call the helpers too. Names from earlier attempts stay in the record."""
    path = Path(stage_dir) / STUBS_RECORD
    names = set(todo)
    try:
        old = safeio.read_json(path, max_bytes=65536, expect=list)
        names |= {n for n in old if isinstance(n, str)}
    except (safeio.SafeReadError, OSError, ValueError):
        pass
    try:
        safeio.atomic_write(path, json.dumps(sorted(names)) + "\n")
    except (OSError, ValueError):
        pass                                   # a record that cannot be written leaves the repair step with its earlier rule


def run_with_inputs(*, stage_dir, ledger, stage: int, post, endpoint: str | None, model: str | None, sampling: dict | None, inputs: Inputs,
                    thinking: bool = False, sheet=None, max_requests: int | None = None, max_prompt_chars: int = PROMPT_CHARS_CAP,
                    clock=time.time, connect=None) -> Outcome:
    """The step itself. `post(url, body, timeout)` is the agent loop's HTTP client. Returns an Outcome; never raises for a coder or input
    problem."""
    stage_dir = Path(stage_dir)
    port_dir = stage_dir / "port"
    port_dir.mkdir(parents=True, exist_ok=True)
    try:
        others = _port_dir_others(port_dir)
        if others:
            return Outcome("skipped", f"the port directory holds other files than {PORT_NAME} ({len(others)} found)")
        if sheet is None:
            sheet = op_cheats.load_sheet(op_cheats.TTNN_SHEET)
        text, plan = port_template.generate(config=inputs.config, construction=inputs.construction, tensors=inputs.tensors,
                                            inventory=inputs.inventory, excerpts=inputs.excerpts)
        existing = _read_existing(port_dir)
        skeleton = safeio.read_bytes(port_facts.SKELETON, max_bytes=port_facts.SKELETON_MAX)
    except Skip as exc:
        return Outcome("skipped", str(exc))
    except (port_template.PlanError, op_cheats.SheetError, safeio.SafeReadError, OSError, ValueError, KeyError, TypeError) as exc:
        return Outcome("skipped", f"the port cannot be planned ({exc.__class__.__name__})")
    if existing is not None and existing != skeleton:
        text = existing.decode("utf-8", "replace")             # an earlier attempt's port: keep it, ask only for what is still a stub
        if port_template.check_sources({PORT_NAME: text})["parse_errors"]:
            return Outcome("skipped", "the port left by an earlier attempt does not parse")
    else:
        _write_port(port_dir, text)
    by_name = {s.function: s for s in plan.stubs}
    todo = stub_prompts.stub_names(text)
    _record_stub_names(stage_dir, todo)
    asked = [name for name in todo if name in by_name]          # the stubs the loop below sends a request for
    if connect is not None and asked:
        endpoint, model, sampling = connect(len(asked))           # the coder is started here, for a plan that needs it (issue 713)
    endpoint, sampling = endpoint or "", sampling or {}          # with no stub to send, neither is used
    cap = max_requests if max_requests is not None else request_cap(len(todo))
    url = endpoint.rstrip("/") + "/chat/completions"
    base = {**sampling, "max_tokens": min(sampling.get("max_tokens", STUB_MAX_TOKENS), STUB_MAX_TOKENS)}
    out = Outcome("done")
    stopped = None
    for name in todo:
        stub = by_name.get(name)
        if stub is None:
            continue                                            # a marked function the plan does not know: left for the repair loop
        refusal, was_capped = None, False
        for attempt in range(1, 2 + RETRIES_PER_STUB):
            request = stub_prompts.stub_request(text, stub, sheet=sheet, excerpts=inputs.excerpts, construction_text=inputs.construction_text,
                                                inventory=inputs.inventory)
            user = request["user"] if refusal is None else (
                request["user"] + f"\n\n## The previous reply was refused\n{refusal}\nWrite the function again so that it passes.")
            chars = len(request["system"]) + len(user)
            if out.prompt_chars + chars > max_prompt_chars:
                stopped = "prompt size cap"
                break
            changed = retry_sampling(base, capped=was_capped) if attempt > 1 else None
            body = _body(model, request, user, {**base, **(changed or {})}, thinking)
            started = clock()
            sent = _send(post, url, body, out, cap)
            seconds = round(clock() - started, 3)
            if sent is None:
                stopped = "request cap"
                break
            data, error = sent
            out.prompt_chars += chars
            if error is not None:
                ledger.append("decision", stage, decision="thin port request", function=name, attempt=attempt, prompt_chars=chars,
                              seconds=seconds, outcome="error", reason=error)
                out.status, out.detail = "error", error
                return _finish(out, ledger, stage, text, port_dir, plan)
            content, tokens, reason = _reply_of(data)
            out.completion_tokens += tokens or 0
            new = None
            if reason is None:
                try:
                    new = stub_prompts.splice(text, name, content)
                    if port_template.check_sources({PORT_NAME: new})["parse_errors"]:
                        new, reason = None, "the spliced port does not parse"
                except stub_prompts.SpliceError as exc:
                    reason = str(exc)[:200]
            fields = {"function": name, "attempt": attempt, "prompt_chars": chars, "seconds": seconds}
            if changed:
                fields["retry_sampling"] = changed
            if tokens is not None:
                fields["completion_tokens"] = tokens
            if new is not None:
                text = new
                _write_port(port_dir, text)
                ledger.append("decision", stage, decision="thin port request", outcome="spliced", **fields)
                out.filled.append(name)
                break
            last = attempt == 1 + RETRIES_PER_STUB
            ledger.append("decision", stage, decision="thin port request", outcome="left as a stub" if last else "refused", reason=reason, **fields)
            refusal, was_capped = reason, reason == CAPPED_REASON
        if stopped:
            break
    if stopped:
        ledger.append("decision", stage, decision="thin port stopped", why=stopped, requests=out.requests, prompt_chars=out.prompt_chars,
                      note="the port stands as it is; the repair loop takes the stubs that are left")
    return _finish(out, ledger, stage, text, port_dir, plan)


def _read_existing(port_dir: Path):
    """The bytes of a port file an earlier attempt left, or None when there is none. Raises Skip when it cannot be read."""
    if not os.path.lexists(port_dir / PORT_NAME):
        return None
    try:
        return safeio.read_bytes(port_dir / PORT_NAME, max_bytes=PORT_FILE_MAX)
    except safeio.SafeReadError as exc:
        raise Skip(f"the port file cannot be read ({exc.__class__.__name__})") from None


def _send(post, url: str, body: dict, out: Outcome, cap: int):
    """(response, None) on success, (None, error text) after the identical request failed twice, or None when the request cap is reached."""
    last = None
    for _ in range(2):
        if out.requests >= cap:
            return None
        out.requests += 1
        try:
            return post(url, body, REQUEST_TIMEOUT_S), None
        except (OSError, ValueError) as exc:
            last = exc
    return None, f"the model request failed and was retried once ({last.__class__.__name__})"


def _finish(out: Outcome, ledger, stage: int, text: str, port_dir: Path, plan) -> Outcome:
    """The summary decision, the check of the port directory, and the stubs that are left."""
    out.stubs_left = stub_prompts.stub_names(text)
    check = port_template.check_sources({PORT_NAME: text})
    ledger.append("decision", stage, decision="thin port finished", status=out.status, filled=list(out.filled), left=list(out.stubs_left),
                  requests=out.requests, prompt_chars=out.prompt_chars, completion_tokens=out.completion_tokens,
                  matmul_calls=check["matmul_calls"], parse_errors=len(check["parse_errors"]))
    try:
        extra = _port_dir_others(port_dir)
    except Skip:
        extra = ["unlisted"]
    if extra and out.status != "error":
        out.status, out.detail = "error", "the port directory holds a file this step did not write"
        ledger.append("decision", stage, decision="thin port refused", why=out.detail)
    return out
