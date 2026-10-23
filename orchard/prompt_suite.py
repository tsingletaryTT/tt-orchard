# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The prompt-suite check of the stage 2 and stage 4 swap tests (issue 15).

The 32-token check compares one prompt with the stage 1 CPU reference. It cannot tell a model that
serves badly on other kinds of prompt from one that serves well, and it cannot tell a wrong model
from a quantised right one when the two are close. This module holds the part that runs outside the
hardware test:

- the committed suite (`prompt_suite.json` next to the templates) and its checks,
- the checks that read a reply for the failure shapes the tenstorrent `qualitative-check` skill lists
  (added further down), and
- the comparison with a control, which decides what the stage gates fail.

The template (serve_and_compare.py) only sends the prompts and records the raw replies in
evidence/swap-check.json. The gates read those replies and decide here, so an agent cannot change a
verdict by editing its copy of the template. A copy that sends other prompts is caught by the
suite's sha256.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from orchard.defaults import SUITE_MIN_PROMPTS

SUITE_FILE = Path(__file__).resolve().parent / "skills" / "weights-swap-templates" / "prompt_suite.json"
ROLES = ("system", "user", "assistant")
SCRIPTS = ("Latin", "Cyrillic", "Greek", "CJK", "Hangul", "Arabic", "Hebrew", "Thai", "Devanagari")


class SuiteError(Exception):
    """The suite file cannot be read or does not describe a usable suite."""


def check_suite(data) -> list[str]:
    """Every problem with a suite, one sentence each; an empty list means the suite is usable."""
    if not isinstance(data, dict):
        return ["the suite is not a JSON object"]
    out = []
    if not isinstance(data.get("version"), int) or isinstance(data.get("version"), bool):
        out.append("the suite has no whole-number version")
    prompts = data.get("prompts")
    if not isinstance(prompts, list) or len(prompts) < SUITE_MIN_PROMPTS:
        return out + [f"the suite needs at least {SUITE_MIN_PROMPTS} prompts"]
    seen, markers = set(), {}
    for i, p in enumerate(prompts):
        where = f"prompt {i}"
        if not isinstance(p, dict):
            out.append(f"{where} is not an object")
            continue
        pid = p.get("id")
        where = f"prompt {pid!r}" if isinstance(pid, str) and pid else where
        if not isinstance(pid, str) or not pid:
            out.append(f"{where} has no id")
        elif pid in seen:
            out.append(f"{where}: duplicate id")
        seen.add(pid)
        text = p.get("text")
        if not isinstance(text, str) or not text.strip():
            out.append(f"{where} has no text (the plain form used when the model has no chat template)")
        msgs = p.get("messages")
        if not isinstance(msgs, list) or not msgs:
            out.append(f"{where} has no messages")
        else:
            for m in msgs:
                if not (isinstance(m, dict) and m.get("role") in ROLES and isinstance(m.get("content"), str)
                        and m["content"].strip()):
                    out.append(f"{where}: a message needs a role from {ROLES} and non-empty content")
                    break
            if isinstance(msgs[-1], dict) and msgs[-1].get("role") != "user":
                out.append(f"{where}: the last message must be the user's")
        if p.get("expect_script") not in SCRIPTS:
            out.append(f"{where}: expect_script must be one of {SCRIPTS}")
        has_tools = "tools" in p
        if p.get("needs_tools") is True and not (isinstance(p.get("tools"), list) and p["tools"]):
            out.append(f"{where}: needs_tools is set but the prompt has no tools")
        if has_tools and p.get("needs_tools") is not True:
            out.append(f"{where}: it has tools but needs_tools is not true, so a template without tools would still send them")
        if p.get("marker") is not None:
            if not isinstance(p["marker"], str) or not p["marker"]:
                out.append(f"{where}: a marker must be a non-empty string")
            else:
                markers[pid] = p["marker"]
    for pid, marker in markers.items():
        for p in prompts:
            if not isinstance(p, dict) or not isinstance(p.get("messages"), list):
                continue
            text = json.dumps(p["messages"], ensure_ascii=False) + str(p.get("text"))
            if (marker in text) != (p.get("id") == pid):
                where = "is missing from its own prompt" if p.get("id") == pid else f"also appears in prompt {p.get('id')!r}"
                out.append(f"marker {marker!r} of prompt {pid!r} {where}")
    return out


def load_suite(path=None) -> dict:
    """The suite from `path` (the committed file by default). Raises SuiteError with every problem."""
    path = Path(path) if path else SUITE_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SuiteError(f"cannot read the prompt suite {path}: {exc}") from exc
    problems = check_suite(data)
    if problems:
        raise SuiteError(f"the prompt suite {path} has problems: " + "; ".join(problems))
    return data


def suite_sha256(path=None) -> str:
    """The sha256 of the suite file's bytes. The template records its copy's; the gate compares it with the shipped file's."""
    return hashlib.sha256(Path(path or SUITE_FILE).read_bytes()).hexdigest()


# ---- reading one reply -----------------------------------------------------------------------------------------------
#
# Each check is deterministic, standard library only, and returns at most one finding per kind. The numbers are
# provisional choices in orchard/defaults.py (SUITE_*); no chip has run the suite yet.

import re  # noqa: E402
import unicodedata  # noqa: E402
from collections import Counter  # noqa: E402

from orchard import defaults  # noqa: E402

KINDS = ("request_failed", "empty_output", "mechanical_repetition", "doubled_subwords", "prompt_echo",
         "control_token_leak", "cross_request_leak", "repeated_first_token", "corrupt_first_token", "gibberish",
         "wrong_language")
THINK_TAGS = ("<think>", "</think>")
PIPE_TOKEN = re.compile(r"<\|[A-Za-z0-9_]{1,40}\|>")
TOOL_MARKUP = re.compile(r"</?tool_call>|</?function(?:=|>)|<parameter=|</parameter>")
# Words that really are one unit twice. A reply with three or more other such words is the finding.
DOUBLED_OK = frozenset("bonbon booboo byebye cancan chichi coco couscous dodo gaga gogo lulu mama murmur nono papa tartar tutu yoyo "
                       "blahblah dumdum".split())
DOUBLED = re.compile(r"\b([A-Za-z]{2,12})\1\b", re.IGNORECASE)
UNIT_LOOP = re.compile(r"(.{2,24}?)\1{%d,}" % (defaults.SUITE_UNIT_REPEATS - 1), re.DOTALL)
LEADING_PUNCT = re.compile(r"[,;:!?)\]}]{3}")
SCRIPT_OF = {"LATIN": "Latin", "CYRILLIC": "Cyrillic", "GREEK": "Greek", "ARABIC": "Arabic", "HEBREW": "Hebrew",
             "HANGUL": "Hangul", "THAI": "Thai", "DEVANAGARI": "Devanagari", "CJK": "CJK", "HIRAGANA": "CJK",
             "KATAKANA": "CJK"}
PUNCTUATION = set(".,;:!?'\"-()[]{}/\\&%$#@*+=<>_`~|^‘’“”–—…")      # the swap check's list of ordinary punctuation
VOWELS = set("aeiouyAEIOUY")


def judged_text(entry: dict) -> str:
    """What the checks read: the answer when there is one, else the reasoning the server returned."""
    content = entry.get("content") or ""
    return content if content.strip() else (entry.get("reasoning") or "")


def _all_text(entry: dict) -> str:
    return (entry.get("reasoning") or "") + "\n" + (entry.get("content") or "")


def _snip(text: str, at: int = 0, width: int = 60) -> str:
    return text[max(0, at):at + width].replace("\n", " ")


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def _repetition(text: str) -> str | None:
    words = text.split()
    n = defaults.SUITE_PHRASE_WORDS
    phrases = Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    # A phrase of punctuation alone ("= = = = = =") is a rule drawn across the page, not a loop.
    phrases = Counter({ph: c for ph, c in phrases.items() if any(ch.isalnum() for w in ph for ch in w)})
    if phrases:
        phrase, count = phrases.most_common(1)[0]
        if count > defaults.SUITE_PHRASE_REPEATS_MAX:
            return f"a {n}-word phrase appears {count} times: {' '.join(phrase)[:60]!r}"
    run, prev = 0, None
    for line in (ln.strip() for ln in text.splitlines()):
        if not line:
            continue
        run = run + 1 if line == prev else 1
        prev = line
        if run >= defaults.SUITE_LINE_REPEATS and len(line) >= defaults.SUITE_LINE_MIN_CHARS:
            return f"the line {line[:60]!r} repeats {run} times in a row"
    for m in UNIT_LOOP.finditer(text):
        if len(m.group(0)) >= defaults.SUITE_UNIT_MIN_SPAN and any(c.isalnum() for c in m.group(1)):
            return f"{m.group(1)[:24]!r} repeats over {len(m.group(0))} characters"
    return None


def _script_share(text: str, expected: str) -> tuple[int, float]:
    counts = Counter()
    for ch in text:
        if ch.isalpha():
            name = unicodedata.name(ch, "").split(" ")[0]
            counts[SCRIPT_OF.get(name, "Other")] += 1
    total = sum(counts.values())
    return total, (counts[expected] / total if total else 0.0)


def findings(entry: dict, prompt: dict | None = None, *, specials=(), markers=None) -> list[dict]:
    """The failure shapes in one reply, as [{"kind", "detail"}], one per kind.

    `entry` is a recorded reply: id, content, reasoning, tool_calls and error. `prompt` is the suite prompt it answered
    (None for the stage 1 free run). `specials` are the tokenizer's special token strings. `markers` maps every
    prompt id to its marker, for the leak between requests."""
    out: dict[str, str] = {}
    if entry.get("error"):
        return [{"kind": "request_failed", "detail": str(entry["error"])[:200]}]
    content, reasoning = entry.get("content") or "", entry.get("reasoning") or ""
    if not content.strip() and not reasoning.strip() and not entry.get("tool_calls"):
        return [{"kind": "empty_output", "detail": "the reply has no text and no tool call"}]
    text = judged_text(entry)
    both = _all_text(entry)

    loop = _repetition(text)
    if loop:
        out["mechanical_repetition"] = loop
    doubled = [m.group(0) for m in DOUBLED.finditer(text) if m.group(0).lower() not in DOUBLED_OK]
    if len(doubled) >= defaults.SUITE_DOUBLED_MIN:
        out["doubled_subwords"] = f"{len(doubled)} words are one unit twice, for example {doubled[0]!r}"
    if prompt and content.strip():
        asked = _norm(prompt["messages"][-1]["content"])
        if len(asked) >= defaults.SUITE_ECHO_MIN_CHARS and _norm(content).startswith(asked[:60]):
            out["prompt_echo"] = f"the reply begins with the question: {_snip(content)!r}"

    leak = PIPE_TOKEN.search(both)
    if not leak:
        for tok in specials:
            if isinstance(tok, str) and len(tok) >= 3 and tok not in THINK_TAGS and not tok.isalnum() and tok in both:
                leak = re.search(re.escape(tok), both)
                break
    if leak:
        out["control_token_leak"] = f"the reply contains the token {leak.group(0)!r}"
    elif TOOL_MARKUP.search(content):
        m = TOOL_MARKUP.search(content)
        out["control_token_leak"] = f"tool-call markup came back as text ({m.group(0)!r}); the server's parser did not take it"
    for pid, marker in (markers or {}).items():
        if pid != entry.get("id") and re.search(r"(?<![A-Za-z0-9])" + re.escape(marker) + r"(?![A-Za-z0-9])", both):
            out["cross_request_leak"] = f"the reply contains {marker!r}, a word from the prompt {pid!r}"
            break

    lead = text.lstrip()
    if lead:
        words = re.findall(r"\w+", lead.casefold())[:defaults.SUITE_FIRST_TOKEN_REPEATS]
        if len(words) == defaults.SUITE_FIRST_TOKEN_REPEATS and len(set(words)) == 1:
            out["repeated_first_token"] = f"the reply begins with {words[0]!r} {len(words)} times"
        first = lead[0]
        if (first == "�" or unicodedata.category(first) in ("Mn", "Mc", "Me", "Cf", "Cc", "Co", "Cs")
                or LEADING_PUNCT.match(lead)):
            out["corrupt_first_token"] = f"the reply begins with {lead[:6]!r}"

    # Letters, combining marks (Thai and Devanagari vowel signs are marks), digits, punctuation and separators are readable;
    # block elements, replacement characters and the like are not.
    ok = sum(1 for c in text if unicodedata.category(c)[0] in "LMNPZ" or c.isspace() or c in PUNCTUATION)
    share = ok / len(text) if text else 1.0
    long_words = [w for w in re.findall(r"[A-Za-z]+", text) if len(w) >= defaults.SUITE_NO_VOWEL_WORD_CHARS]
    bare = [w for w in long_words if not (VOWELS & set(w))]
    if len(text) >= defaults.SUITE_READABLE_MIN_CHARS and share < defaults.SUITE_READABLE_MIN:
        out["gibberish"] = f"only {share:.0%} of the characters are letters, digits, spaces or common punctuation"
    elif len(bare) >= defaults.SUITE_NO_VOWEL_MIN_WORDS and len(bare) / len(long_words) > defaults.SUITE_NO_VOWEL_SHARE_MAX:
        out["gibberish"] = f"{len(bare)} of {len(long_words)} long words have no vowel, for example {bare[0]!r}"

    if prompt and prompt.get("expect_script") and content.strip():
        total, frac = _script_share(content, prompt["expect_script"])
        if total >= defaults.SUITE_MIN_LETTERS and frac < defaults.SUITE_SCRIPT_MIN_SHARE:
            out["wrong_language"] = (f"{frac:.0%} of the answer's letters are {prompt['expect_script']}; "
                                     f"the prompt expects that script")
    return [{"kind": k, "detail": out[k][:200]} for k in KINDS if k in out]


# ---- the verdict -----------------------------------------------------------------------------------------------------
#
# What evidence/swap-check.json holds under "prompt_suite" (written by serve_and_compare.py):
#   ran, suite_sha256 (of the template's copy of prompt_suite.json), mode ("chat" or "completion"), max_tokens,
#   chat_template {present, source, sha256, mentions_tools}, special_tokens [str],
#   entries [{id, mode, content, reasoning, tool_calls, finish_reason, error}], skipped [{id, reason}],
#   control null or {source, sha256, record} where record has the same shape (the swap-check.json of a control run).
# The gate reads that record and the stage 1 free run beside it (free_run_text, reference_generated_text) here.

from dataclasses import dataclass, field  # noqa: E402

MAX_REASONS = defaults.SUITE_MAX_REASONS
MODES = ("chat", "completion")


@dataclass
class Verdict:
    ran: bool
    reasons: list = field(default_factory=list)       # a gate fails on each of these
    advisory: list = field(default_factory=list)      # recorded, not gated
    summary: dict = field(default_factory=dict)


def _specials(record: dict) -> list:
    tokens = record.get("special_tokens")
    return [t for t in tokens if isinstance(t, str)] if isinstance(tokens, list) else []


def _entries(record: dict) -> dict:
    return {e["id"]: e for e in record.get("entries", []) if isinstance(e, dict) and isinstance(e.get("id"), str)}


def _control_problems(record: dict, ctl) -> tuple[dict | None, list, list]:
    """The control's record when it can explain this one, with the reasons it cannot and the notes about it."""
    if not isinstance(ctl, dict) or not isinstance(ctl.get("record"), dict):
        return None, ["the control is not a recorded prompt suite (it has no record)"], []
    other = ctl["record"]
    bad = []
    if other.get("suite_sha256") != record.get("suite_sha256"):
        bad.append(f"the control ran other prompts (suite sha256 {str(other.get('suite_sha256'))[:12]} against "
                   f"{str(record.get('suite_sha256'))[:12]}), so it explains nothing")
    if other.get("mode") != record.get("mode"):
        bad.append(f"the control was rendered in another prompt format ({other.get('mode')!r} against "
                   f"{record.get('mode')!r}), so it explains nothing")
    if bad:
        return None, bad, []
    notes = []
    mine = (record.get("chat_template") or {}).get("sha256")
    theirs = (other.get("chat_template") or {}).get("sha256")
    if mine != theirs:
        notes.append(f"the control's chat template differs from this model's (sha256 {str(theirs)[:12]} against "
                     f"{str(mine)[:12]}), so the two were rendered into different text")
    return other, [], notes


def evaluate(report: dict, *, suite_path=None) -> Verdict:
    """Read the prompt-suite record in a swap-check report. `reasons` fail a gate; `advisory` lines are recorded only.

    A finding fails a gate when its kind is in SUITE_BLOCKING_KINDS and the control does not show that kind on the same
    prompt. A report without a record has no verdict (older runs, and runs that did not opt in)."""
    record = report.get("prompt_suite") if isinstance(report, dict) else None
    if record is None:
        return Verdict(False, summary={"ran": False})
    if not isinstance(record, dict) or record.get("ran") is not True or not isinstance(record.get("entries"), list):
        return Verdict(True, ["the prompt suite record is malformed, or says the suite did not run"], summary={"ran": True})
    try:
        suite = load_suite(suite_path)
    except SuiteError as exc:
        return Verdict(True, [f"the committed prompt suite cannot be read: {exc}"], summary={"ran": True})
    reasons, advisory, rows = [], [], []

    if record.get("suite_sha256") != suite_sha256(suite_path):
        reasons.append(f"the recorded prompts are not the committed prompt suite (sha256 "
                       f"{str(record.get('suite_sha256'))[:12]} against {suite_sha256(suite_path)[:12]})")
    mode = record.get("mode")
    template = record.get("chat_template") if isinstance(record.get("chat_template"), dict) else {}
    if mode not in MODES:
        reasons.append(f"the recorded prompt mode {mode!r} is neither 'chat' nor 'completion'")
    elif (mode == "chat") != bool(template.get("present")):
        reasons.append(f"the recorded prompt format {mode!r} contradicts the chat template record "
                       f"(present: {template.get('present')!r}); the format must follow the tokenizer's template")

    prompts = {p["id"]: p for p in suite["prompts"]}
    markers = {pid: p["marker"] for pid, p in prompts.items() if p.get("marker")}
    entries = _entries(record)
    if len(entries) != sum(1 for e in record["entries"] if isinstance(e, dict)):
        reasons.append("an entry of the record has no id, or two entries share one")
    if any(not isinstance(e, dict) for e in record["entries"]):
        reasons.append("an entry of the record is not an object")
    skipped = {s.get("id"): s.get("reason") for s in record.get("skipped") or [] if isinstance(s, dict)}
    for pid, p in prompts.items():
        if pid in skipped:
            if not (p.get("needs_tools") and not template.get("mentions_tools")):
                reasons.append(f"prompt {pid} was skipped ({skipped[pid]}); only the tool prompt may be skipped, and only "
                               "when the chat template does not mention tools")
        elif pid not in entries:
            reasons.append(f"prompt {pid} has no reply in the record")
    for pid in entries:
        if pid not in prompts:
            reasons.append(f"the record has a reply for {pid!r}, which is not in the committed suite")

    control, bad, notes = (None, [], [])
    if record.get("control") is not None:
        control, bad, notes = _control_problems(record, record["control"])
    reasons += [f"control: {b}" for b in bad]
    advisory += notes
    control_entries = _entries(control) if control else {}
    control_specials = _specials(control) if control else []

    checked = []
    for pid, p in prompts.items():
        if pid in entries:
            fl = findings(entries[pid], p, specials=_specials(record), markers=markers)
            cl = findings(control_entries[pid], p, specials=control_specials, markers=markers) if pid in control_entries else None
            checked.append((f"prompt {pid}", pid, fl, cl, "The control does not show it."))
    free, ref = report.get("free_run_text"), report.get("reference_generated_text")
    if isinstance(free, str):
        fl = findings({"id": "stage1", "content": free, "reasoning": "", "tool_calls": [], "error": None}, None,
                      specials=_specials(record), markers=markers)
        cl = (findings({"id": "stage1", "content": ref, "reasoning": "", "tool_calls": [], "error": None}, None)
              if isinstance(ref, str) else None)
        checked.append(("the stage 1 free run", "stage1", fl, cl, "The CPU reference does not show it."))

    for label, pid, fl, cl, absent in checked:
        have = {f["kind"] for f in cl} if cl is not None else None
        for f in fl:
            shared = have is not None and f["kind"] in have
            blocking = f["kind"] in defaults.SUITE_BLOCKING_KINDS
            rows.append({"prompt": pid, "kind": f["kind"], "detail": f["detail"], "shared": shared, "blocking": blocking})
            if shared:
                continue
            line = f"prompt suite, {label}: {f['kind']}: {f['detail']}"
            if blocking:
                reasons.append(line + ". " + (absent if have is not None else
                                              "There is no control for this prompt, so nothing explains it."))
            else:
                advisory.append(line)
    if len(reasons) > MAX_REASONS:
        extra = len(reasons) - (MAX_REASONS - 1)
        reasons = reasons[:MAX_REASONS - 1] + [f"{extra} more prompt suite reasons (see the findings in the record)"]
    return Verdict(True, reasons, advisory,
                   {"ran": True, "prompts": len(entries), "mode": mode, "control": "file" if control else "none",
                    "findings": rows})
