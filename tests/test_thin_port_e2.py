"""Arm E2 of the stage 2 experiment (issue 677): the stub prompt names the `model` dict and the module constants of the generated port,
capped and refused replies are retried with a stronger sampling profile under a smaller token cap, and a failed board test is repaired by
one request per failing function (the thin repair). Arm E (PR 689) failed both board checks with `'dict' object has no attribute 'config'`:
the stubs treated `model` as an object."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from orchard import op_cheats, port_template as pt, stub_prompts as sp

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "port-generator"
SHEET = op_cheats.load_sheet(op_cheats.TTNN_SHEET)

def _generate_without_families(**kw):
    """The Qwen4 plan with no registered block family: the hyper-connection blocks stay stubs, as before issue 696. These tests exercise the stub
    requests for that whole list; the plan with the registry is tested in tests/test_port_template_registry.py."""
    from unittest import mock
    with mock.patch.dict(pt.BLOCK_FAMILIES, clear=True):
        return pt.generate(**kw)


@pytest.fixture(autouse=True)
def _no_block_families(monkeypatch):
    """Code under test that generates a port itself (the thin port step) also sees the plan without block families (see _generate_without_families)."""
    monkeypatch.setattr(pt, "BLOCK_FAMILIES", {})



def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def tiny():
    text, plan = _generate_without_families(config=_load("tiny-qwen4-config.json"), construction=_load("tiny-qwen4-construction.json"),
                             tensors=_load("tiny-qwen4-tensors.json"), inventory=_load("tiny-qwen4-ops-inventory.json"))
    return text, plan


# ---- (1) the prompt says what the arguments are -------------------------------------------------------------------------------------------

def _model_keys_set_by_load(text: str) -> set:
    """The keys the generated load() puts in `model`: the dict literal it starts from and every `model["key"] = ...` after it."""
    load = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "load")
    keys = set()
    for node in ast.walk(load):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict) and any(getattr(t, "id", "") == "model" for t in node.targets):
            keys |= {k.value for k in node.value.keys if isinstance(k, ast.Constant)}
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Subscript) and getattr(t.value, "id", "") == "model" and isinstance(t.slice, ast.Constant)
                        and isinstance(t.slice.value, str)):
                    keys.add(t.slice.value)
    return keys


def _module_constants(text: str) -> set:
    return {t.id for n in ast.parse(text).body if isinstance(n, ast.Assign) for t in n.targets
            if isinstance(t, ast.Name) and not t.id.startswith("_")}


def test_the_model_keys_in_the_prompt_are_exactly_the_keys_the_generated_load_sets(tiny):
    text, _ = tiny
    assert _model_keys_set_by_load(text) == set(pt.MODEL_KEYS)


def test_the_module_constants_in_the_prompt_are_the_ones_the_generated_port_defines(tiny):
    text, _ = tiny
    assert _module_constants(text) == set(pt.MODULE_CONSTANTS)


def test_a_stub_request_names_every_model_key_and_constant_and_says_there_is_no_attribute_access(tiny):
    text, plan = tiny
    stub = next(s for s in plan.stubs if s.function == "_model_prelude")
    user = sp.stub_request(text, stub, sheet=SHEET, excerpts=None)["user"]
    for key in pt.MODEL_KEYS:
        assert f"`{key}`" in user, key
    for name in pt.MODULE_CONSTANTS:
        assert f"`{name}`" in user, name
    assert "plain dict" in user
    for missing in ("model.config", "rotary_emb", "past_key_values"):
        assert missing in user                              # named as things that do not exist
    assert "build position tables from CONFIG" in user
    assert "`_rope_tables`" in user or "_rope_tables" in user or "`_linear`" in user       # the port's own helper names are listed


def test_a_repair_request_carries_the_same_section(tiny):
    text, _ = tiny
    req = sp.repair_request(text, "_A_linear_attn", last_check="FAILED: x", sheet=SHEET)
    assert "plain dict" in req["user"] and "`CONFIG`" in req["user"] and "`dev`" in req["user"]


def test_the_model_text_lists_only_the_constants_the_port_has():
    port = "CONFIG = {}\nN_LAYERS = 2\n\ndef _linear(model, prefix, x):\n    return x\n"
    text = sp.model_text(port)
    assert "`CONFIG`" in text and "`N_LAYERS`" in text and "`PAD_ID`" not in text and "`_linear`" in text


# ---- (2) the token cap of a stub request, and the sampling of a retry --------------------------------------------------------------------------

from orchard import thin_port as tp                                                              # noqa: E402
from test_thin_port import Coder, run as run_step, SENTINEL                                      # noqa: E402


class Capped(Coder):
    """A coder whose n-th reply (n in `capped`) ends at the token cap; the others are good."""

    def __init__(self, capped=(), refuse=()):
        super().__init__()
        self.capped, self.refuse = set(capped), set(refuse)

    def __call__(self, url, body, timeout):
        self.bodies.append(body)
        n = len(self.bodies)
        from test_thin_port import target
        function, args = target(body["messages"][-1]["content"])
        if n in self.capped:
            return {"choices": [{"message": {"content": "x " * 50}, "finish_reason": "length"}], "usage": {"completion_tokens": 16384}}
        content = "no code here" if n in self.refuse else f"```python\ndef {function}({args}):\n    return x  # {SENTINEL}\n```"
        return {"choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {"completion_tokens": 40}}


def test_the_request_token_cap_is_the_smaller_of_the_tier_cap_and_the_stub_bound(tmp_path):
    coder = Capped()
    run_step(tmp_path, coder, sampling={"temperature": 0.6, "max_tokens": 16384})
    assert coder.bodies and {b["max_tokens"] for b in coder.bodies} == {tp.STUB_MAX_TOKENS}
    small = Capped()
    run_step(tmp_path / "b", small, sampling={"temperature": 0.6, "max_tokens": 1000})
    assert {b["max_tokens"] for b in small.bodies} == {1000}


def test_the_first_request_uses_the_tiers_sampling_unchanged(tmp_path):
    coder = Capped()
    run_step(tmp_path, coder, sampling={"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 4096})
    first = coder.bodies[0]
    assert first["temperature"] == 0.6 and first["top_p"] == 0.95 and first["top_k"] == 20 and "repetition_penalty" not in first


def test_a_retry_after_a_refusal_lowers_the_temperature_and_adds_no_penalty(tmp_path):
    coder = Capped(refuse={1})
    out, ledger, _ = run_step(tmp_path, coder, sampling={"temperature": 0.6, "max_tokens": 4096})
    assert coder.bodies[1]["temperature"] <= tp.RETRY_TEMPERATURE_MAX and "repetition_penalty" not in coder.bodies[1]
    retried = [d for d in ledger.decisions("thin port request") if d.get("attempt") == 2]
    assert retried and retried[0]["retry_sampling"] == {"temperature": tp.RETRY_TEMPERATURE_MAX}


def test_a_retry_never_raises_the_temperature(tmp_path):
    coder = Capped(refuse={1})
    run_step(tmp_path, coder, sampling={"temperature": 0.1, "max_tokens": 4096})
    assert coder.bodies[1]["temperature"] == 0.1


def test_a_retry_after_a_capped_reply_adds_a_repetition_penalty_when_the_tier_has_none(tmp_path):
    coder = Capped(capped={1})
    out, ledger, _ = run_step(tmp_path, coder, sampling={"temperature": 0.6, "max_tokens": 4096})
    assert coder.bodies[1]["repetition_penalty"] == tp.RETRY_REPETITION_PENALTY and coder.bodies[1]["temperature"] <= tp.RETRY_TEMPERATURE_MAX
    retried = [d for d in ledger.decisions("thin port request") if d.get("attempt") == 2]
    assert retried[0]["retry_sampling"] == {"temperature": tp.RETRY_TEMPERATURE_MAX, "repetition_penalty": tp.RETRY_REPETITION_PENALTY}


def test_a_tier_penalty_above_the_retry_penalty_is_kept(tmp_path):
    coder = Capped(capped={1})
    run_step(tmp_path, coder, sampling={"temperature": 0.6, "repetition_penalty": 1.3, "max_tokens": 4096})
    assert coder.bodies[1]["repetition_penalty"] == 1.3


def test_one_retry_only_and_the_second_failure_leaves_the_stub(tmp_path):
    coder = Capped(refuse={1, 2})
    out, ledger, _ = run_step(tmp_path, coder, sampling={"temperature": 0.6, "max_tokens": 4096})
    first = [d for d in ledger.decisions("thin port request") if d.get("function") == out.stubs_left[0]] if out.stubs_left else []
    assert out.stubs_left and len(first) == 2 and first[-1]["outcome"] == "left as a stub"


# ---- (3) the thin repair: one request per failing function -----------------------------------------------------------------------------------

from orchard import thin_repair as tr                                                            # noqa: E402

FAILED = ("FAILED: the port failed at 'prefill': \"[port_model.py:{line}: config = model.config] AttributeError: "
          "'dict' object has no attribute 'config'\"\nPORT_CHECK_FAILURE {{\"kind\": \"port\", \"op\": \"prefill\"}}\n")

PORT = '''import ttnn

CONFIG = {"head_dim": 4}
N_LAYERS = 2


def _linear(model, prefix, x):
    return x


def _model_prelude(model, x, ids):
    config = model.config
    return x


def helper_two(a):
    return a
'''


def _line_of(text, needle):
    return next(n for n, l in enumerate(text.split("\n"), 1) if needle in l)


def test_the_failed_line_is_parsed_strictly():
    f = tr.parse_failed("port| log line\n" + FAILED.format(line=12))
    assert (f.op, f.file, f.line, f.etype) == ("prefill", "port_model.py", 12, "AttributeError")
    assert f.code == "config = model.config" and f.message == "'dict' object has no attribute 'config'"
    single = ("FAILED: the port failed at 'load': '[port_model.py:69: x = G(config)] NameError: name \\'G\\' is not defined'\n")
    assert tr.parse_failed(single).etype == "NameError"


@pytest.mark.parametrize("text", [
    "", None, 7, "no failure here", "FAILED: device-ops: no source file under stages/2/port/ makes a reachable ttnn matmul-family call",
    "FAILED: the port worker did not answer 'load' (it exited or stopped answering)",
    "FAILED: the port failed at 'prefill': \"[port_model.py:x: code] AttributeError: m\"",          # a line number that is not digits
    "FAILED: the port failed at 'prefill': \"[port_model.py:12: code] 9Error: m\"",                    # an error type that is not a name
    "FAILED: the port failed at 'prefill': \"[port_model.py:12: " + "c" * 500 + "] AttributeError: m\"",   # code over its bound
])
def test_a_failure_that_is_not_a_python_error_at_a_port_line_is_not_parsed(text):
    assert tr.parse_failed(text) is None


def test_only_the_last_failed_line_counts_and_a_port_cannot_write_one():
    text = FAILED.format(line=5) + "port| FAILED: the port failed at 'load': \"[port_model.py:1: x] KeyError: 'k'\"\n"
    assert tr.parse_failed(text).line == 5                       # the worker's `port| ` lines never start with FAILED


def test_function_at_finds_the_top_level_function_that_holds_the_line():
    assert tr.function_at(PORT, _line_of(PORT, "config = model.config")) == "_model_prelude"
    assert tr.function_at(PORT, _line_of(PORT, "return a")) == "helper_two"
    assert tr.function_at(PORT, _line_of(PORT, "CONFIG = ")) is None                            # module level
    assert tr.function_at(PORT, 9999) is None and tr.function_at("def f(:\n", 1) is None


def test_the_hint_table_has_one_entry_per_shape_and_fills_in_the_groups():
    attr = tr.Failure("prefill", "port_model.py", 1, "x", "AttributeError", "'dict' object has no attribute 'config'")
    h = tr.hint_for(attr, PORT)
    assert "plain dict" in h and "model.config" in h and "CONFIG" in h
    name = tr.Failure("load", "port_model.py", 1, "x", "NameError", "name 'GatedResidual' is not defined")
    h = tr.hint_for(name, PORT)
    assert "`GatedResidual`" in h and "_linear" in h and "helper_two" in h
    mod = tr.Failure("load", "port_model.py", 1, "x", "TypeError", "port_model.Qwen4ExpTextDecoderLayer is not a Module subclass")
    assert "plain functions" in tr.hint_for(mod, PORT)
    other = tr.Failure("load", "port_model.py", 1, "x", "KeyError", "'weight'")
    assert "not a key of CONFIG" in tr.hint_for(other, PORT) and "head_dim" in tr.hint_for(other, PORT)      # PORT's CONFIG has no `weight`
    inside = tr.Failure("load", "port_model.py", 1, "x", "KeyError", "'head_dim'")
    assert tr.hint_for(inside, PORT) is None                         # a key that CONFIG has: the hint does not apply
    assert tr.hint_for(tr.Failure("p", "port_model.py", 1, "x", "ValueError", "'dict' object has no attribute 'a'"), PORT) is None      # the type must match too


def _stage(tmp_path, text=PORT):
    d = tmp_path / "stages" / "2"
    (d / "port").mkdir(parents=True)
    (d / "port" / "port_model.py").write_text(text)
    return d


FIXED = "```python\ndef _model_prelude(model, x, ids):\n    return x  # " + SENTINEL + "\n```"


class OneShot:
    def __init__(self, *contents, finish="stop"):
        self.contents, self.bodies, self.finish = list(contents), [], finish

    def __call__(self, url, body, timeout):
        self.bodies.append(body)
        content = self.contents[min(len(self.bodies) - 1, len(self.contents) - 1)]
        return {"choices": [{"message": {"content": content}, "finish_reason": self.finish}], "usage": {"completion_tokens": 30}}


def _round(tmp_path, post, k=1, raw=None, **kw):
    d = _stage(tmp_path, kw.pop("text", PORT))
    raw = raw if raw is not None else FAILED.format(line=_line_of(PORT, "config = model.config"))
    out = tr.run_round(stage_dir=d, k=k, raw_output=raw, post=post, endpoint="http://x/v1", model="m",
                       sampling=kw.pop("sampling", {"temperature": 0.6, "max_tokens": 16384}), sheet=SHEET, **kw)
    return out, d


def test_a_round_asks_for_the_failing_function_with_the_hint_and_splices_the_reply(tmp_path):
    coder = OneShot(FIXED)
    out, d = _round(tmp_path, coder)
    assert out.status == "spliced" and out.function == "_model_prelude" and out.etype == "AttributeError" and out.hint is True
    body = coder.bodies[0]
    user = body["messages"][-1]["content"]
    assert "config = model.config" in user and "## Hint for this error (AttributeError)" in user and "plain dict" in user
    assert body["max_tokens"] == tp.STUB_MAX_TOKENS and "tools" not in body
    new = (d / "port" / "port_model.py").read_text()
    assert SENTINEL in new and "model.config" not in new and "def helper_two" in new           # the other functions are untouched


def test_a_refused_reply_is_retried_once_with_the_reason_and_a_cooler_sampling(tmp_path):
    coder = OneShot("no code here", FIXED)
    out, _ = _round(tmp_path, coder)
    assert out.status == "spliced" and out.requests == 2
    assert "## The previous reply was refused" in coder.bodies[1]["messages"][-1]["content"]
    assert coder.bodies[1]["temperature"] <= tp.RETRY_TEMPERATURE_MAX


def test_two_refusals_hand_the_round_to_the_agent_loop_and_leave_the_port_alone(tmp_path):
    coder = OneShot("no code here")
    out, d = _round(tmp_path, coder)
    assert out.status == "refused" and out.requests == tr.REQUESTS_PER_ROUND == len(coder.bodies)
    assert (d / "port" / "port_model.py").read_text() == PORT


def test_a_round_past_the_cap_asks_nobody(tmp_path):
    coder = OneShot(FIXED)
    out, _ = _round(tmp_path, coder, k=tr.MAX_ROUNDS + 1)
    assert out.status == "fallback" and coder.bodies == []
    ok, _ = _round(tmp_path / "b", OneShot(FIXED), k=tr.MAX_ROUNDS)
    assert ok.status == "spliced"


@pytest.mark.parametrize("raw,why", [
    ("", "not a Python error"),
    ("FAILED: device-ops: no source file under stages/2/port/", "not a Python error"),
    ("FAILED: the port failed at 'load': \"[other.py:3: x] KeyError: 'k'\"\n", "another file"),
    ("FAILED: the port failed at 'load': \"[port_model.py:4: CONFIG = {}] KeyError: 'k'\"\n", "not inside a top-level function"),
])
def test_a_failure_the_thin_repair_cannot_place_goes_to_the_agent_loop_without_a_request(tmp_path, raw, why):
    coder = OneShot(FIXED)
    out, _ = _round(tmp_path, coder, raw=raw)
    assert out.status == "fallback" and why in out.reason and coder.bodies == []


def test_an_endpoint_that_fails_ends_the_round_in_error(tmp_path):
    def boom(url, body, timeout):
        raise OSError("down")
    out, d = _round(tmp_path, boom)
    assert out.status == "error" and out.requests <= tr.REQUESTS_PER_ROUND and (d / "port" / "port_model.py").read_text() == PORT


def test_the_result_holds_no_reply_text(tmp_path):
    out, _ = _round(tmp_path, OneShot(FIXED))
    assert SENTINEL not in json.dumps(out.__dict__)


def test_the_round_prompt_cap_stops_a_round_whose_prompt_is_too_large(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "ROUND_PROMPT_CHARS_CAP", 100)
    coder = OneShot(FIXED)
    out, _ = _round(tmp_path, coder)
    assert out.status == "fallback" and "prompt size cap" in out.reason and coder.bodies == []


def test_function_at_includes_the_first_and_last_line_of_the_function():
    first = _line_of(PORT, "def _model_prelude")
    last = _line_of(PORT, "return x") if False else first + 2
    assert tr.function_at(PORT, first) == "_model_prelude" and tr.function_at(PORT, last) == "_model_prelude"
    assert tr.function_at(PORT, first - 1) is None


def test_the_bounds_of_a_round_are_two_requests_and_six_rounds():
    assert tr.REQUESTS_PER_ROUND == 2 and tr.MAX_ROUNDS == 6
    coder = OneShot("no code here")
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as t:
        out, _ = _round(Path(t), coder)
    assert out.status == "refused" and len(coder.bodies) == 2
