# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The trace-safe prefill flow for GraniteChip. No ttnn here, so the flow is tested on the CPU with a fake chip (tests/test_granite_trace.py).

ttnn warns that device buffers allocated while a trace is alive may be corrupted. The chunked prefill allocates and frees buffers for every
block of tokens, and one trace-on run of it left a chip unreadable (issues 252 and 290). The rule kept here: no buffer is allocated between
a trace capture and its release. The flow is

    1. check the request (context limit) while the traces are still alive and nothing has changed;
    2. release the decode traces;
    3. run the chunked blocks (they allocate freely, no trace exists);
    4. capture the decode traces again (a capture records the program and runs nothing, so the state the blocks left is untouched);
    5. run the left-over tokens and the last token through the traced decode step, which only writes persistent buffers and replays.

Step 4 runs in a `finally`, so a failure in step 3 still leaves the chip with its traces for the next request. The chip object supplies
`release_traces`, `capture_traces`, `traces_alive`, `run_chunk`, `run_tail_token`, `step`, `run_step_prefill`, optionally `mark_unusable` and `trace_names`, and the fields `ssd`, `pos`, `L`,
`CHUNKS`.
"""
from __future__ import annotations


CHUNKS = (128, 64, 32)


def chunk_sizes(chip, chunk):
    """Block sizes allowed for a `chunk` limit, largest first. Always includes 32 (the tile size)."""
    return [c for c in chip.CHUNKS if c <= chunk] or [32]


def trace_names(chunked: bool, fast_prefill: bool) -> tuple:
    """The decode traces a chip captures. `full` is the decode step with the final norm and lm_head. `nohead` stops after the last layer and
    is a small gain for a step-loop prefill. Capturing it allocates device tensors while `full` exists, so a chip that uses the chunked
    prefill (whose step-loop part is at most 31 tokens) captures `full` only: no capture ever starts while another trace is alive."""
    return ("full", "nohead") if fast_prefill and not chunked else ("full",)


def check_usable(chip):
    """Refuse an unusable chip. `mark_unusable` sets `chip.unusable` after a failed trace capture: a half-captured trace may exist, so nothing
    more runs on the chip until the server is restarted."""
    if getattr(chip, "unusable", None):
        raise RuntimeError(f"the chip is unusable until the server is restarted: {chip.unusable}")


def prefill(chip, ids, chunked=True, chunk=128, report=None):
    """The prefill the server uses. Chunked in composed mode when `chunked` is on and the prompt has a full block of 32 before its last
    token. Everything else is the step loop (one decode step per token). A chip whose traces could not be captured again is refused
    (`unusable` is set by `mark_unusable`): it may hold a half-captured trace, so nothing more runs until the server is restarted."""
    check_usable(chip)
    if chunked and chip.ssd == "composed" and len(ids) - 1 >= chunk_sizes(chip, chunk)[-1]:
        return prefill_chunked(chip, ids, chunk, report=report)
    return chip.run_step_prefill(ids)


def prefill_chunked(chip, ids, chunk=128, report=None):
    """Chunked prefill with no decode trace alive while blocks run. Returns the logits after the last prompt token.

    `report(exc)` is called with a failure of a block before the traces are captured again, so the failure is on record even if the capture
    then hangs. A failure of the capture while a block failure is pending raises the block failure, with the capture failure as its cause."""
    check_usable(chip)
    if chip.ssd != "composed":
        raise ValueError("prefill_chunked needs ssd='composed'")
    if chip.pos + len(ids) > chip.L:
        raise ValueError(f"position {chip.pos + len(ids)} is past the context limit {chip.L}")
    sizes = chunk_sizes(chip, chunk)
    body = ids[:-1]
    if len(body) < sizes[-1]:
        return chip.run_step_prefill(ids)                       # no full block: nothing for the chunked path to do
    had_traces = chip.traces_alive()
    if had_traces and "nohead" in (chip.trace_names() if hasattr(chip, "trace_names") else ()):
        raise ValueError("prefill_chunked needs a chip built with chunked_prefill=True: this chip captures the no-head trace, and recapturing it "
                         "after the blocks would allocate while the full trace is alive")
    if had_traces:
        try:
            chip.release_traces()
        except BaseException as exc:                # the traces may be half released: refuse every later request
            if hasattr(chip, "mark_unusable"):
                try:
                    chip.mark_unusable(exc)
                except Exception:
                    pass
            raise
    failed = None
    try:
        i = 0
        while len(body) - i >= sizes[-1]:
            T = next(c for c in sizes if c <= len(body) - i)
            chip.run_chunk(body[i:i + T])
            i += T
    except BaseException as exc:                    # recapture first, then raise this failure
        failed = exc
        if report is not None:
            try:
                report(exc)
            except Exception:                       # reporting is best effort: the recapture and the original failure matter more
                pass
    if had_traces:
        try:
            chip.capture_traces()
        except BaseException as cap:
            if hasattr(chip, "mark_unusable"):
                try:
                    chip.mark_unusable(cap)         # a half-captured trace may exist: refuse every later request
                except Exception:                   # best effort: the capture error must still reach the caller
                    pass
            if failed is not None:
                raise failed from cap
            raise
    if failed is not None:
        raise failed
    for t in body[i:]:
        chip.run_tail_token(t)
    return chip.step(ids[-1])
