"""The trace-safe prefill flow (pkgsrc/runtime/granite_trace.py). CPU only: a fake chip records what happens to its traces and buffers.

ttnn warns that device buffers allocated while a trace is alive may be corrupted, and one trace-on chunked prefill left a chip unreadable
(issues 252 and 290). The rule these tests hold: no buffer is allocated between a trace capture and its release. The fake chip's `alloc`
records a violation when a trace is alive; the chunked path in the fake allocates the way the real one does (a block of buffers per chunk)."""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("granite_trace", ROOT / "pkgsrc" / "runtime" / "granite_trace.py")
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)


class FakeChip:
    """Stands in for GraniteChip. Records events. A violation is an allocation while a trace is alive."""

    CHUNKS = (128, 64, 32)

    def __init__(self, traced=True, ssd="composed", fail_in_chunk=False):
        self.ssd, self.use_trace, self.events, self.violations = ssd, traced, [], []
        self.tid = self.tid_nohead = None
        self.pos, self.fail_in_chunk, self.L = 0, fail_in_chunk, 2048
        if traced:
            self.capture_traces()

    # -- what the real chip does to its traces
    def traces_alive(self):
        return self.tid is not None or self.tid_nohead is not None

    def alloc(self, what):
        self.events.append(("alloc", what))
        if self.traces_alive():
            self.violations.append(what)

    def release_traces(self):
        self.events.append(("release",))
        self.tid = self.tid_nohead = None

    def capture_traces(self):
        self.events.append(("capture",))
        self.tid, self.tid_nohead = "head", "nohead"

    # -- the work
    def run_chunk(self, toks):
        self.alloc(f"chunk{len(toks)}")
        self.pos += len(toks)

    def run_tail_token(self, t):
        """One prompt token through the decode step, no read-back. Traced when traces are alive (a replay allocates nothing)."""
        self.events.append(("tail", "traced" if self.traces_alive() else "eager"))
        if not self.traces_alive():
            self.alloc("eager-tail")           # an eager step allocates its temporaries
        self.pos += 1

    def step(self, t):
        self.events.append(("step", "traced" if self.traces_alive() else "eager"))
        if not self.traces_alive():
            self.alloc("eager-step")
        self.pos += 1
        return "logits"

    def reset(self):
        self.pos = 0


def test_chunked_prefill_allocates_nothing_while_a_trace_is_alive():
    chip = FakeChip(traced=True)
    gt.prefill_chunked(chip, list(range(300)), chunk=128)
    assert chip.violations == []
    assert any(e[0] == "alloc" for e in chip.events), "the fake must really allocate, or this test proves nothing"


def test_traces_are_back_after_the_prefill_so_decode_is_traced():
    chip = FakeChip(traced=True)
    gt.prefill_chunked(chip, list(range(300)), chunk=128)
    assert chip.traces_alive()
    chip.step(5)
    assert chip.events[-1] == ("step", "traced")


def test_events_are_release_then_allocations_then_capture():
    chip = FakeChip(traced=True)
    chip.events.clear()
    gt.prefill_chunked(chip, list(range(70)), chunk=128)
    kinds = [e[0] for e in chip.events]
    assert kinds[0] == "release"
    assert kinds.index("capture") > max(i for i, k in enumerate(kinds) if k == "alloc")   # every allocation is before the capture
    assert kinds.count("release") == 1 and kinds.count("capture") == 1


def test_a_second_request_releases_and_recaptures_again():
    chip = FakeChip(traced=True)
    for _ in range(3):
        chip.reset()
        gt.prefill_chunked(chip, list(range(200)), chunk=128)
        chip.step(1)
        chip.step(2)
    assert chip.violations == []
    assert [e[0] for e in chip.events].count("release") == 3
    assert chip.traces_alive()


def test_traces_come_back_even_when_the_prefill_raises():
    chip = FakeChip(traced=True, fail_in_chunk=True)
    chip.run_chunk = lambda toks: (_ for _ in ()).throw(RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        gt.prefill_chunked(chip, list(range(200)), chunk=128)
    assert chip.traces_alive()


def test_an_untraced_chip_never_releases_or_captures():
    chip = FakeChip(traced=False)
    gt.prefill_chunked(chip, list(range(200)), chunk=128)
    assert [e for e in chip.events if e[0] in ("release", "capture")] == []


def test_the_default_prefill_is_chunked_for_a_long_prompt_in_composed_mode():
    chip = FakeChip(traced=True)
    gt.prefill(chip, list(range(300)), chunked=True)
    assert any(e == ("alloc", "chunk128") for e in chip.events)
    assert chip.violations == []


def test_a_short_prompt_stays_on_the_traced_step_path():
    """Fewer than 33 tokens leaves no block of 32 for the chunked path, so nothing is released and no trace is rebuilt."""
    chip = FakeChip(traced=True)
    chip.events.clear()
    chip.run_step_prefill = lambda ids: chip.step(ids[-1])
    gt.prefill(chip, list(range(20)), chunked=True)
    assert [e for e in chip.events if e[0] in ("release", "capture")] == []


def test_the_disable_flag_keeps_the_step_loop():
    chip = FakeChip(traced=True)
    chip.events.clear()
    chip.run_step_prefill = lambda ids: chip.step(ids[-1])
    gt.prefill(chip, list(range(300)), chunked=False)
    assert not any(e[0] in ("release", "capture") or e == ("alloc", "chunk128") for e in chip.events)


def test_a_kernel_mode_chip_uses_the_step_loop_even_when_chunked_is_on():
    chip = FakeChip(traced=True, ssd="state_full")
    chip.events.clear()
    chip.run_step_prefill = lambda ids: chip.step(ids[-1])
    gt.prefill(chip, list(range(300)), chunked=True)
    assert not any(e[0] == "release" for e in chip.events)


def test_chunk_sizes_and_tail_match_the_chip_rules():
    """300 tokens: body 299 = 128 + 128 + 32 + 11 left over for the decode-step tail, then the last token."""
    chip = FakeChip(traced=False)
    gt.prefill_chunked(chip, list(range(300)), chunk=128)
    assert [e[1] for e in chip.events if e[0] == "alloc" and e[1].startswith("chunk")] == ["chunk128", "chunk128", "chunk32"]
    assert len([e for e in chip.events if e[0] == "tail"]) == 11
    assert chip.pos == 300


def test_context_limit_is_checked_before_any_trace_is_released():
    chip = FakeChip(traced=True)
    chip.L = 100
    chip.events.clear()
    with pytest.raises(ValueError, match="context limit"):
        gt.prefill_chunked(chip, list(range(300)), chunk=128)
    assert [e for e in chip.events if e[0] == "release"] == []


def test_the_tail_runs_on_the_recaptured_trace():
    """The 11 left-over tokens and the last one replay the new trace: capture comes before them."""
    chip = FakeChip(traced=True)
    chip.events.clear()
    gt.prefill_chunked(chip, list(range(300)), chunk=128)
    kinds = [e[0] for e in chip.events]
    assert kinds.index("capture") < kinds.index("tail")
    assert [e for e in chip.events if e[0] == "tail"] == [("tail", "traced")] * 11
    assert chip.events[-1] == ("step", "traced")
