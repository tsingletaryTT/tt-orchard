# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The chat versions of the HumanEval and MBPP tasks (orchard/lm_eval_tasks/).

On 2026-10-19 the pilot scored bytkim/Qwen3.8-27B-pi at 0.0 pass@1 on 20 HumanEval items, and 12 of the 20 stored replies were empty strings.
lm-evaluation-harness's `humaneval_instruct` (and `mbpp_instruct`) expect the assistant's reply to continue a prefilled turn (`gen_prefix`),
which the chat completions API does not do, and they stop generation at strings such as `\\ndef`, which a chat reply that writes the whole
function hits at once. The instrument was wrong, not the model. These tasks ask for a complete function in a code block, stop at nothing but the
end of the reply, and build the program to run from the block.

The tests run the built programs with the problems' own checks, so "the extractor works" means "a correct reply passes and a wrong one fails".
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

DIR = Path(__file__).resolve().parent.parent / "orchard" / "lm_eval_tasks"


def load():
    spec = importlib.util.spec_from_file_location("code_chat_utils_under_test", DIR / "code_chat_utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


u = load()

HE_DOC = {
    "task_id": "HumanEval/0", "entry_point": "has_close_elements",
    "prompt": 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n'
              '    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    """\n',
    "test": "def check(candidate):\n    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n    assert candidate([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True\n",
}
GOOD_BODY = "    return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i + 1:])\n"
GOOD_FUNC = "def has_close_elements(numbers, threshold):\n" + GOOD_BODY
BAD_FUNC = "def has_close_elements(numbers, threshold):\n    return False\n"

# HumanEval/38 shape: the prompt defines a helper, and the function to write comes after it.
HELPER_DOC = {
    "task_id": "HumanEval/38", "entry_point": "decode_cyclic",
    "prompt": 'def encode_cyclic(s: str):\n    return "".join(s[(3 * i):min(3 * i + 3, len(s))][1:] + s[(3 * i):min(3 * i + 3, len(s))][:1] if len(s[(3 * i):min(3 * i + 3, len(s))]) == 3 else s[(3 * i):min(3 * i + 3, len(s))] for i in range((len(s) + 2) // 3))\n\n\n'
              'def decode_cyclic(s: str):\n    """ takes as input string encoded with encode_cyclic function. Returns decoded string.\n    """\n',
    "test": "def check(candidate):\n    for s in ['abcdef', 'abcdefg', 'xyz', 'ab']:\n        assert candidate(encode_cyclic(s)) == s\n",
}
HELPER_GOOD = "def decode_cyclic(s: str):\n    return encode_cyclic(encode_cyclic(s))\n"


def passes(program: str, doc: dict) -> bool:
    code = program + "\n\n" + doc["test"] + f"\ncheck({doc['entry_point']})\n"
    return subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, timeout=30).returncode == 0


def fenced(code: str, lang="python") -> str:
    return f"```{lang}\n{code}```"


# ---- finding the code in a reply ----------------------------------------------------------------------------------------------------

def test_a_fenced_block_is_found_with_its_language():
    assert u.extract_blocks("Here you go:\n```python\nx = 1\n```\nDone.") == [("python", "x = 1\n")]


def test_every_block_is_returned_in_order():
    got = u.extract_blocks("```bash\nls\n```\ntext\n```python\nprint(1)\n```\n```\nplain\n```")
    assert got == [("bash", "ls\n"), ("python", "print(1)\n"), ("", "plain\n")]


def test_a_reply_cut_off_inside_a_block_keeps_what_was_written():
    assert u.extract_blocks("```python\ndef f():\n    return 1\n") == [("python", "def f():\n    return 1\n")]


def test_a_reply_with_no_fence_has_no_blocks():
    assert u.extract_blocks("def f():\n    return 1") == []


def test_a_python_block_is_preferred_over_an_earlier_shell_block():
    blocks = [("bash", "pip install x\n"), ("python", "def f():\n    return 1\n")]
    assert u.choose_block(blocks, "f") == "def f():\n    return 1\n"


def test_the_block_that_defines_the_entry_point_is_preferred_over_an_earlier_python_block():
    blocks = [("python", "import math\n"), ("python", "def target(x):\n    return x\n"), ("python", "assert target(1) == 1\n")]
    assert u.choose_block(blocks, "target") == "def target(x):\n    return x\n"


def test_with_no_entry_point_the_first_block_that_defines_a_function_is_used():
    blocks = [("python", "x = 1\n"), ("python", "def g():\n    return 2\n")]
    assert u.choose_block(blocks, None) == "def g():\n    return 2\n"


def test_with_no_function_anywhere_the_first_python_block_is_used():
    assert u.choose_block([("bash", "ls\n"), ("python", "x = 1\n")], "f") == "x = 1\n"


def test_no_blocks_gives_none():
    assert u.choose_block([], "f") is None


def test_a_name_that_only_contains_the_entry_point_does_not_count_as_defining_it():
    blocks = [("python", "def has_close_elements_slow(a):\n    return 1\n"), ("python", "def has_close_elements(a):\n    return 2\n")]
    assert u.choose_block(blocks, "has_close_elements") == "def has_close_elements(a):\n    return 2\n"


# ---- HumanEval programs -------------------------------------------------------------------------------------------------------------

def test_a_correct_whole_function_in_a_block_passes_and_a_wrong_one_fails():
    assert passes(u.humaneval_program("Sure.\n" + fenced(GOOD_FUNC) + "\nThat works.", HE_DOC), HE_DOC)
    assert not passes(u.humaneval_program(fenced(BAD_FUNC), HE_DOC), HE_DOC)


def test_the_programs_for_good_and_bad_replies_differ_only_where_the_reply_differs():
    good, bad = u.humaneval_program(fenced(GOOD_FUNC), HE_DOC), u.humaneval_program(fenced(BAD_FUNC), HE_DOC)
    assert good.startswith(HE_DOC["prompt"]) and bad.startswith(HE_DOC["prompt"]) and good != bad


def test_a_reply_that_repeats_the_whole_prompt_then_the_body_passes():
    reply = fenced(HE_DOC["prompt"] + GOOD_BODY)
    assert passes(u.humaneval_program(reply, HE_DOC), HE_DOC)


def test_a_body_only_reply_is_indented_under_the_prompt_signature():
    body = "return any(abs(a - b) < threshold for i, a in enumerate(numbers) for b in numbers[i + 1:])\n"
    assert passes(u.humaneval_program(fenced(body), HE_DOC), HE_DOC)


def test_a_body_only_reply_that_is_already_indented_is_not_indented_again():
    assert passes(u.humaneval_program(fenced(GOOD_BODY), HE_DOC), HE_DOC)


def test_a_helper_defined_in_the_prompt_is_still_available_to_the_reply():
    assert passes(u.humaneval_program(fenced(HELPER_GOOD), HELPER_DOC), HELPER_DOC)
    assert not passes(u.humaneval_program(fenced("def decode_cyclic(s):\n    return s\n"), HELPER_DOC), HELPER_DOC)


def test_the_imports_in_the_prompt_are_kept_when_the_reply_leaves_them_out():
    reply = fenced("def has_close_elements(numbers: List[float], threshold: float) -> bool:\n" + GOOD_BODY)
    assert passes(u.humaneval_program(reply, HE_DOC), HE_DOC)                       # `List` is only defined by the prompt's import


def test_a_main_guard_with_example_calls_is_cut_so_it_cannot_fail_the_run():
    reply = fenced(GOOD_FUNC + "\nif __name__ == '__main__':\n    assert has_close_elements([1.0], 5.0) == 'wrong'\n")
    assert passes(u.humaneval_program(reply, HE_DOC), HE_DOC)


def test_a_reply_with_no_fence_is_used_as_it_is():
    assert passes(u.humaneval_program(GOOD_FUNC, HE_DOC), HE_DOC)


def test_an_empty_reply_gives_a_program_that_fails_the_check_and_does_not_crash():
    program = u.humaneval_program("", HE_DOC)
    assert program.startswith(HE_DOC["prompt"]) and not passes(program, HE_DOC)


def test_a_reply_with_a_syntax_error_fails_the_check_and_does_not_crash():
    program = u.humaneval_program(fenced("def has_close_elements(:\n    pass\n"), HE_DOC)
    assert not passes(program, HE_DOC)


def test_the_filter_returns_one_program_per_reply_per_document():
    docs = [HE_DOC, HELPER_DOC]
    got = u.build_humaneval([[fenced(GOOD_FUNC)], [fenced(HELPER_GOOD)]], docs)
    assert len(got) == 2 and len(got[0]) == 1 and passes(got[0][0], HE_DOC) and passes(got[1][0], HELPER_DOC)


def test_a_filter_call_with_several_replies_per_document_keeps_them_separate():
    got = u.build_humaneval([[fenced(GOOD_FUNC), fenced(BAD_FUNC)]], [HE_DOC])
    assert [passes(p, HE_DOC) for p in got[0]] == [True, False]


# ---- MBPP programs -----------------------------------------------------------------------------------------------------------------

MBPP_DOC = {"task_id": 11, "text": "Write a function to add two numbers.", "test_setup_code": "",
            "test_list": ["assert add(1, 2) == 3", "assert add(0, 0) == 0", "assert add(-1, 1) == 0"]}


def mbpp_passes(program: str, doc=MBPP_DOC) -> bool:
    code = program + "\n" + "\n".join(doc["test_list"]) + "\n"
    return subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, timeout=30).returncode == 0


def test_an_mbpp_reply_with_a_correct_function_passes_and_a_wrong_one_fails():
    assert mbpp_passes(u.mbpp_program("Here:\n" + fenced("def add(a, b):\n    return a + b\n"), MBPP_DOC))
    assert not mbpp_passes(u.mbpp_program(fenced("def add(a, b):\n    return a - b\n"), MBPP_DOC))


def test_an_mbpp_reply_with_a_usage_example_after_the_function_uses_the_function_block():
    reply = fenced("def add(a, b):\n    return a + b\n") + "\nExample:\n" + fenced("print(add(1, 2))\n")
    assert mbpp_passes(u.mbpp_program(reply, MBPP_DOC))


def test_the_test_setup_code_of_the_problem_runs_before_the_reply():
    doc = {**MBPP_DOC, "test_setup_code": "import math\n"}
    program = u.mbpp_program(fenced("def add(a, b):\n    return math.fsum([a, b])\n"), doc)
    assert program.startswith("import math") and mbpp_passes(program, doc)


def test_an_mbpp_reply_with_no_block_is_used_as_it_is():
    assert mbpp_passes(u.mbpp_program("def add(a, b):\n    return a + b", MBPP_DOC))


def test_an_empty_mbpp_reply_gives_a_program_that_fails_and_does_not_crash():
    assert not mbpp_passes(u.mbpp_program("", MBPP_DOC))


def test_an_mbpp_main_guard_is_cut():
    reply = fenced("def add(a, b):\n    return a + b\n\nif __name__ == '__main__':\n    raise SystemExit(3)\n")
    assert mbpp_passes(u.mbpp_program(reply, MBPP_DOC))


def test_the_mbpp_filter_returns_one_program_per_reply():
    got = u.build_mbpp([[fenced("def add(a, b):\n    return a + b\n")]], [MBPP_DOC])
    assert len(got) == 1 and mbpp_passes(got[0][0])


# ---- the task definitions ------------------------------------------------------------------------------------------------------------

class Loader(yaml.SafeLoader):
    pass


Loader.add_constructor("!function", lambda loader, node: "function:" + loader.construct_scalar(node))


def task(name):
    return yaml.load((DIR / f"{name}.yaml").read_text(), Loader=Loader)


@pytest.mark.parametrize("name,dataset", [("humaneval_chat", "openai/openai_humaneval"), ("mbpp_chat", "google-research-datasets/mbpp")])
def test_the_task_asks_for_a_whole_reply_and_stops_at_no_string(name, dataset):
    t = task(name)
    assert t["task"] == name and t["dataset_path"] == dataset and t["unsafe_code"] is True and t["output_type"] == "generate_until"
    assert t["generation_kwargs"]["until"] == [] and "gen_prefix" not in t and t["num_fewshot"] == 0
    assert "```python" in t["doc_to_text"]


@pytest.mark.parametrize("name,fn", [("humaneval_chat", "build_humaneval"), ("mbpp_chat", "build_mbpp")])
def test_the_task_builds_its_programs_with_the_function_in_this_directory(name, fn):
    t = task(name)
    assert t["filter_list"][0]["filter"][0]["filter_fn"] == f"function:code_chat_utils.{fn}"
    assert t["metric_list"][0]["metric"].startswith("function:code_chat_utils.")


def test_humaneval_chat_runs_the_problems_own_test_and_check():
    t = task("humaneval_chat")
    assert t["doc_to_target"] == "{{test}}\ncheck({{entry_point}})"


def test_mbpp_chat_checks_the_three_listed_asserts():
    t = task("mbpp_chat")
    for i in range(3):
        assert "{{test_list[%d]}}" % i in t["doc_to_target"]


def test_the_module_imports_only_the_standard_library():
    src = (DIR / "code_chat_utils.py").read_text()
    top = [l for l in src.splitlines() if l.startswith(("import ", "from "))]
    allowed = {"re", "os", "importlib", "pathlib", "concurrent", "typing", "__future__"}
    assert all(l.split()[1].split(".")[0] in allowed for l in top), top
    assert "evaluate" not in "".join(top), "the harness's code_eval runs model code in the harness's own process"


# ---- the metric wrappers -------------------------------------------------------------------------------------------------------------

class FakeRun:
    """Stands in for the sandbox: a program passes when it contains the word PASS."""
    def __init__(self):
        self.programs = []

    def __call__(self, program):
        self.programs.append(program)
        return "PASS" in program


@pytest.fixture
def fake_run(monkeypatch):
    fake = FakeRun()
    monkeypatch.setattr(u, "run_one", fake)
    return fake


def test_pass_at_k_runs_each_prediction_with_its_reference_and_returns_the_score_dictionary(fake_run):
    assert u.pass_at_k(["T"], [["PASS a", "no"]], k=1) == {"pass@1": 0.5}
    assert fake_run.programs == ["PASS a\nT", "no\nT"]
    assert u.pass_at_k(["T"], [["PASS", "no"]], k=[1, 5]) == {"pass@1": 0.5}      # k=5 needs 5 predictions, so it is left out
    assert u.pass_at_k(["T"], [["PASS", "no"]], k=[1, 2]) == {"pass@1": 0.5, "pass@2": 1.0}


def test_pass_at_k_averages_over_problems(fake_run):
    assert u.pass_at_k(["A", "B"], [["PASS"], ["fail"]], k=1) == {"pass@1": 0.5}


def test_pass_at_k_needs_a_k():
    with pytest.raises(AssertionError):
        u.pass_at_k(["ref"], [["pred"]])


def test_pass_at_1_wraps_single_strings_and_returns_the_number(fake_run):
    assert u.pass_at_1("asserts", ["PASS code"]) == 1.0
    assert u.pass_at_1(["asserts"], [["no"]]) == 0.0
    assert fake_run.programs == ["PASS code\nasserts", "no\nasserts"]


def test_the_estimator_is_the_unbiased_pass_at_k():
    assert u.estimate_pass_at_k(1, 1, 1) == 1.0 and u.estimate_pass_at_k(1, 0, 1) == 0.0
    assert u.estimate_pass_at_k(4, 1, 1) == pytest.approx(0.25)
    assert u.estimate_pass_at_k(4, 1, 2) == pytest.approx(0.5)            # 1 - C(3,2)/C(4,2)
    assert u.estimate_pass_at_k(4, 3, 2) == 1.0                           # fewer than k wrong ones


# ---- the sandbox: no program of the model's runs in the harness's process, with the operator's files and network --------------------------

def test_a_sandbox_that_cannot_start_stops_scoring_instead_of_scoring_zero(monkeypatch):
    box = u._load_sandbox()
    monkeypatch.setattr(box, "run_program", lambda program, **kw: box.Outcome("sandbox_unavailable", "this kernel has no Landlock"))
    monkeypatch.delenv(u.UNSANDBOXED_ENV, raising=False)
    with pytest.raises(RuntimeError, match="sandbox could not start"):
        u.pass_at_k(["T"], [["x"]], k=1)


def test_the_unsandboxed_variable_is_the_only_way_to_run_without_the_sandbox(monkeypatch):
    box = u._load_sandbox()
    seen = []
    monkeypatch.setattr(box, "run_program", lambda program, **kw: seen.append(kw["unsandboxed"]) or box.Outcome("passed"))
    monkeypatch.delenv(u.UNSANDBOXED_ENV, raising=False)
    u.run_one("x")
    monkeypatch.setenv(u.UNSANDBOXED_ENV, "0")
    u.run_one("x")
    monkeypatch.setenv(u.UNSANDBOXED_ENV, "1")
    u.run_one("x")
    assert seen == [False, False, True]


needs_landlock = pytest.mark.skipif(not u._load_sandbox().landlock_abi(), reason="SKIPPED: this kernel has no Landlock")


@needs_landlock
def test_a_reply_that_reads_a_planted_token_file_scores_zero_and_a_plain_reply_passes(tmp_path, monkeypatch):
    monkeypatch.delenv(u.UNSANDBOXED_ENV, raising=False)
    token = tmp_path / "token"
    token.write_text("hf_planted\n")
    thief = f"assert open({str(token)!r}).read().startswith('hf_')\n"      # passes only if it can read the file
    assert u.pass_at_k(["pass"], [[thief]], k=1) == {"pass@1": 0.0}
    monkeypatch.setenv(u.UNSANDBOXED_ENV, "1")                              # control: with no sandbox the same program passes
    assert u.pass_at_k(["pass"], [[thief]], k=1) == {"pass@1": 1.0}
    monkeypatch.delenv(u.UNSANDBOXED_ENV)
    assert u.pass_at_k(["assert f(2) == 4"], [["def f(x):\n    return x * 2\n"]], k=1) == {"pass@1": 1.0}


def test_an_unlabeled_block_counts_as_python():
    blocks = [("bash", "ls\n"), ("", "def f():\n    return 1\n")]
    assert u.choose_block(blocks, "f") == "def f():\n    return 1\n"
