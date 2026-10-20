# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Build the program to run from a chat model's reply, for the `humaneval_chat` and `mbpp_chat` tasks.

lm-evaluation-harness loads this file by path from the task directory (`!function code_chat_utils.build_humaneval`), so it imports nothing from
this repository and nothing outside the standard library. The tests load it the same way.

The programs are run by `code_sandbox.py` (the file beside this one), which gives each program no network and no files outside a scratch
directory. The scorer never runs a model's code in the harness's own process: Hugging Face's `code_eval` would, with the operator's HOME, token
file and network, and its `reliability_guard` is not a sandbox. When the sandbox cannot start, scoring raises an error. Setting
`ORCHARD_CODE_UNSANDBOXED=1` (what `bench_session --allow-unsandboxed-code` does) runs the programs without it.

A chat reply is a few sentences and a fenced code block. The block may hold the whole function, the whole prompt and the body, or only the body;
there may be a second block with example calls. `choose_block` picks the block that defines the function, and `humaneval_program` puts the
problem's own prompt in front of it, so helper functions and imports in the prompt stay available and the reply's definition replaces the
prompt's stub. Module-level `if __name__ == ...` code is cut, because example calls there can fail a run for a reason that has nothing to do with
the function.
"""
from __future__ import annotations

import importlib.util
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PYTHON_LANGS = ("", "python", "py", "python3")
FENCE = re.compile(r"```([A-Za-z0-9_+-]*)[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)
MAIN_GUARD = re.compile(r"^if\s+__name__\s*==", re.MULTILINE)


def extract_blocks(text: str) -> list[tuple[str, str]]:
    """Each fenced block as (language, code). A reply cut off inside a block keeps what it wrote."""
    return [(lang, code) for lang, code in FENCE.findall(text or "")]


def defines(code: str, name: str) -> bool:
    return re.search(rf"^[ \t]*(?:async\s+)?def\s+{re.escape(name)}\s*\(", code, re.MULTILINE) is not None


def choose_block(blocks: list[tuple[str, str]], entry_point):
    """The code to use: a Python (or unlabeled) block that defines the entry point, else one that defines any function, else the first Python
    block, else the first block of any language. None when there are no blocks."""
    if not blocks:
        return None
    py = [code for lang, code in blocks if lang.lower() in PYTHON_LANGS]
    if entry_point:
        for code in py:
            if defines(code, entry_point):
                return code
    for code in py:
        if re.search(r"^[ \t]*def\s", code, re.MULTILINE):
            return code
    return py[0] if py else blocks[0][1]


def cut_main_guard(code: str) -> str:
    return MAIN_GUARD.split(code, maxsplit=1)[0]


def _indent_body(code: str) -> str:
    lines = code.splitlines()
    first = next((l for l in lines if l.strip()), "")
    if first[:1] in (" ", "\t"):
        return code if code.endswith("\n") else code + "\n"
    return "".join(("    " + l if l.strip() else l) + "\n" for l in lines)


def humaneval_program(reply: str, doc: dict) -> str:
    entry, prompt = doc["entry_point"], doc["prompt"]
    code = choose_block(extract_blocks(reply), entry)
    if code is None:
        code = reply or ""
    code = cut_main_guard(code)
    if not code.strip():
        return prompt                                  # the stub the prompt ends with: a docstring body that returns None
    if defines(code, entry):
        return prompt + "\n\n" + code.rstrip() + "\n"
    return prompt + _indent_body(code.rstrip("\n"))


def mbpp_program(reply: str, doc: dict) -> str:
    code = choose_block(extract_blocks(reply), None)
    if code is None:
        code = reply or ""
    code = cut_main_guard(code).rstrip() + "\n" if code.strip() else ""
    setup = (doc.get("test_setup_code") or "").strip()
    return (setup + "\n" if setup else "") + code


def build_humaneval(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[humaneval_program(r, doc) for r in resp] for resp, doc in zip(resps, docs)]


def build_mbpp(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[mbpp_program(r, doc) for r in resp] for resp, doc in zip(resps, docs)]


# ---- scoring: each program runs in the sandbox, and the score is HumanEval's unbiased pass@k estimate ------------------------------------------

UNSANDBOXED_ENV = "ORCHARD_CODE_UNSANDBOXED"
TIMEOUT_S = 10.0           # per program; the harness's own `code_eval` allows 3 s, and this includes starting the sandboxed interpreter
_sandbox = None


def _load_sandbox():
    global _sandbox
    if _sandbox is None:
        path = Path(__file__).resolve().with_name("code_sandbox.py")
        spec = importlib.util.spec_from_file_location("code_sandbox_for_chat_tasks", path)
        _sandbox = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_sandbox)
    return _sandbox


def run_one(program: str) -> bool:
    """Whether `program` ran to the end without an error, in the sandbox. A sandbox that could not start raises RuntimeError (the score would
    otherwise be a column of failures that looks like a result)."""
    box = _load_sandbox()
    outcome = box.run_program(program, timeout_s=TIMEOUT_S, unsandboxed=os.environ.get(UNSANDBOXED_ENV) == "1")
    if outcome.status == "sandbox_unavailable":
        raise RuntimeError("the code sandbox could not start, so no program was run: " + outcome.detail
                           + f". Fix the kernel's Landlock support, or set {UNSANDBOXED_ENV}=1 to run the programs without isolation.")
    return outcome.status == "passed"


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """The unbiased estimator of Chen et al. 2021: the chance that at least one of k samples drawn from n, c of which are correct, is correct."""
    if n - c < k:
        return 1.0
    miss = 1.0
    for i in range(n - c + 1, n + 1):
        miss *= 1.0 - k / i
    return 1.0 - miss


def _compute(references, predictions, k):
    """Like `evaluate.load("code_eval").compute`: each prediction plus its reference test, run as one program. Returns {"pass@k": mean}
    for every k that no problem has fewer predictions than."""
    _load_sandbox()                      # once, before the worker threads start
    programs = [(i, pred + "\n" + ref) for i, (ref, preds) in enumerate(zip(references, predictions)) for pred in preds]
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
        ok = list(pool.map(lambda item: run_one(item[1]), programs))
    passed = {}
    counts = {}
    for (i, _), result in zip(programs, ok):
        counts[i] = counts.get(i, 0) + 1
        passed[i] = passed.get(i, 0) + (1 if result else 0)
    out = {}
    for kk in k:
        if counts and all(n >= kk for n in counts.values()):
            out[f"pass@{kk}"] = sum(estimate_pass_at_k(counts[i], passed[i], kk) for i in counts) / len(counts)
    return out


def pass_at_k(references, predictions, k=None):
    assert k is not None
    if isinstance(k, int):
        k = [k]
    return _compute(references, predictions, k)


def pass_at_1(references, predictions):
    if isinstance(references, str):
        references = [references]
    if isinstance(predictions[0], str):
        predictions = [[p] for p in predictions]
    return _compute(references, predictions, [1])["pass@1"]
