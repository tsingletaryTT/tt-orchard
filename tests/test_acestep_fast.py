# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Targeted tests for scripts/acestep/acestep_fast.py: the request-level savings that leave every output unchanged (a repeated condition-encoder call, the constant
timbre encoding of the silence reference, the garbage collector). No chip, no ACE-Step code, no weights."""
import gc
import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "acestep"
spec = importlib.util.spec_from_file_location("acestep_fast", SCRIPTS / "acestep_fast.py")
fast = importlib.util.module_from_spec(spec)
sys.modules["acestep_fast"] = fast
spec.loader.exec_module(fast)


class Model:
    """Stands in for AceStepConditionGenerationModel: prepare_condition takes keyword tensors and returns three tensors."""

    def __init__(self):
        self.calls = 0

    def prepare_condition(self, *, text_hidden_states, lyric_hidden_states, attention_mask=None, is_covers=None, audio_codes=None):
        self.calls += 1
        return text_hidden_states * 2, lyric_hidden_states + 1, text_hidden_states.sum(-1)


def _args(v=1.0):
    return dict(text_hidden_states=torch.full((1, 3, 4), v), lyric_hidden_states=torch.zeros(1, 2, 4), attention_mask=torch.ones(1, 5),
                is_covers=torch.zeros(1), audio_codes=None)


def test_second_identical_call_is_answered_from_the_first():
    m = Model()
    fast.memoize_prepare_condition(m)
    a = m.prepare_condition(**_args())
    b = m.prepare_condition(**_args())
    assert m.calls == 1
    assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_a_changed_tensor_or_flag_is_recomputed():
    m = Model()
    fast.memoize_prepare_condition(m)
    m.prepare_condition(**_args(1.0))
    m.prepare_condition(**_args(2.0))
    assert m.calls == 2
    kw = _args(2.0)
    kw["audio_codes"] = "codes"
    m.prepare_condition(**kw)
    assert m.calls == 3


def test_only_the_last_call_is_kept():
    m = Model()
    fast.memoize_prepare_condition(m)
    m.prepare_condition(**_args(1.0))
    m.prepare_condition(**_args(2.0))
    m.prepare_condition(**_args(1.0))
    assert m.calls == 3


def test_cached_results_are_not_poisoned_by_the_caller():
    m = Model()
    fast.memoize_prepare_condition(m)
    a = m.prepare_condition(**_args())
    a[0].zero_()
    b = m.prepare_condition(**_args())
    assert float(b[0].sum()) != 0.0
    b[0].zero_()                                   # a result served from the cache must not be the cache's own tensor
    c = m.prepare_condition(**_args())
    assert float(c[0].sum()) != 0.0


class Timbre:
    def __init__(self):
        self.calls = 0

    def forward(self, refer_audio_acoustic_hidden_states_packed=None, refer_audio_order_mask=None, **kw):
        self.calls += 1
        return refer_audio_acoustic_hidden_states_packed.mean(1, keepdim=True), torch.ones(1, 1, dtype=torch.long)


def test_timbre_encoding_of_the_same_reference_is_computed_once():
    t = Timbre()
    fast.cache_timbre(t)
    ref = torch.zeros(1, 750, 8)
    order = torch.tensor([0])
    t.forward(ref, order)
    t.forward(ref.clone(), order.clone())
    assert t.calls == 1
    t.forward(ref + 1, order)
    assert t.calls == 2


def test_timbre_cache_is_bounded():
    t = Timbre()
    fast.cache_timbre(t, entries=2)
    order = torch.tensor([0])
    for i in range(4):
        t.forward(torch.full((1, 3, 8), float(i)), order)
    t.forward(torch.full((1, 3, 8), 0.0), order)       # the oldest entry was dropped
    assert t.calls == 5


def test_freeze_gc_calls_gc_freeze(monkeypatch):
    seen = []
    monkeypatch.setattr(gc, "freeze", lambda: seen.append(1))
    fast.freeze_gc()
    assert seen == [1]


def test_apply_rejects_an_unknown_step_name():
    with pytest.raises(ValueError, match="unknown"):
        fast.parse_steps("cond_cache,warp_drive")
    assert fast.parse_steps("none") == ()
    assert set(fast.parse_steps("all")) == set(fast.STEPS)


def test_an_argument_passed_as_none_equals_an_argument_left_out():
    # the handler calls prepare_condition without audio_codes; generate_audio passes audio_codes=None
    m = Model()
    fast.memoize_prepare_condition(m)
    kw = _args()
    del kw["audio_codes"]
    m.prepare_condition(**kw)
    m.prepare_condition(**_args())
    assert m.calls == 1


def test_chip_steps_need_a_worker():
    # the pieces that run on the chip cannot be installed on a CPU session: that is an error, not a silent no-op
    class H:
        model = None

    assert "kv_on_chip" in fast.STEPS
    with pytest.raises(ValueError, match="chip"):
        fast.apply(H(), ("kv_on_chip",), worker=None)


def test_chip_steps_are_installed_through_acestep_chip(monkeypatch):
    seen = []
    fake = type(sys)("acestep_chip")
    fake.install_kv_on_chip = lambda worker: seen.append(("kv", worker))
    fake.install_text_on_chip = lambda worker, handler: seen.append(("text", worker, handler))
    fake.install_lyric_on_chip = lambda worker, handler: seen.append(("lyric", worker, handler))
    monkeypatch.setitem(sys.modules, "acestep_chip", fake)

    class H:
        model = None

    w, h = object(), H()
    assert fast.apply(h, ("kv_on_chip", "text_on_chip", "lyric_on_chip"), worker=w) == ("kv_on_chip", "text_on_chip", "lyric_on_chip")
    assert seen == [("kv", w), ("text", w, h), ("lyric", w, h)]
    seen.clear()
    fast.apply(h, ("text_on_chip",), worker=w)
    assert seen == [("text", w, h)], "a step that was not asked for is not installed"


def test_reset_forgets_the_last_call_so_the_cache_never_spans_two_requests():
    # the saving is the repeat inside one request; an identical next request must be computed again (a benchmark that repeats one prompt would otherwise
    # measure the cache instead of the request)
    m = Model()
    fast.memoize_prepare_condition(m)
    m.prepare_condition(**_args())
    m.prepare_condition(**_args())
    assert m.calls == 1
    m.prepare_condition.reset()
    m.prepare_condition(**_args())
    assert m.calls == 2


def test_session_one_resets_the_cache_before_each_request():
    # Session.one is the request boundary; it must call fast.new_request(handler)
    import re
    src = (SCRIPTS / "acestep_run.py").read_text()
    body = src[src.index("    def one(self"):]
    assert re.search(r"acestep_fast\.new_request\(self\.handler\)", body.split("    def close")[0])


def test_new_request_calls_reset_when_the_cache_is_installed_and_is_silent_otherwise():
    m = Model()
    fast.memoize_prepare_condition(m)
    m.prepare_condition(**_args())

    class H:
        model = m

    fast.new_request(H())
    m.prepare_condition(**_args())
    assert m.calls == 2

    class Plain:
        class model:
            @staticmethod
            def prepare_condition(**kw):
                return ()

    fast.new_request(Plain())          # nothing installed: no error
