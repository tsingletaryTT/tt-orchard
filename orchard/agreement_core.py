# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The agreement-set arithmetic shared by the stage 2 and stage 4 serve-and-compare templates and by the gate (issues 15 and 18).

This file is the shipped copy. `orchard/skills/weights-swap-templates/agreement_core.py` is a byte-identical copy that the stage copies next
to `serve_and_compare.py`, because a template cannot import orchard. `tests/test_agreement_core.py` fails when the two differ. Edit this
file and copy it over the template. It imports only the standard library.

What it does. The stage 2 check compares 32 greedy tokens of one prompt with the CPU reference (`top1_agreement`, bar SWAP_TOP1_MIN). One
prompt with a thin margin cannot tell a fine-tune from the base weights by mistake (0.78 against 0.94 in the first prototype). The agreement
set adds several fixed prompts of different kinds and lengths, one of them long, each with its own CPU reference, and scores every one
teacher-forced the same way. A record holds, per prompt, the reference ids and the ids the chip produced, so a gate recomputes every number
from the raw ids and does not trust a number the template printed.

Three guards sit on top of the arithmetic:

  * the old bar stays. The record carries the 32-token baseline (`baseline`) next to the set's verdict, so the new check does not replace
    the old measurement and a card can still cite it;
  * a damage control. `damaged_copy` makes a deliberately wrong run from the reference (every fourth token replaced). `calibrate` scores that
    run with the same bars: bars that pass it are reported unusable, and a record whose bars cannot separate good from damaged is refused;
  * a recorded control. A run of the same set on the nearest model (the base), served from the same bundle, is embedded in the record. The
    new model has to agree with the reference better than the control does by CONTROL_MARGIN. When the control agrees as well as the new
    model, the check cannot tell the two apart and says so.

Limits. The damage control is synthetic: it shows the bars are not vacuous, not that a real wrong-weights run would be caught. Every numeric
bar here is a choice, because no fine-tune has been run through the set against a measured control yet. The long prompt's reference is a CPU
run, so it stays short (see LONG_PROMPT_TOKENS); a 100K-token agreement needs a different design (see `sweep_plan`).
"""
from __future__ import annotations

import hashlib
import json
import math

# ---- bars (choices; orchard/defaults.py holds the same numbers and a test keeps the two equal) -------------------------------------------
BASELINE_TOP1_MIN = 0.85       # the stage 2 bar for the 32-token check (SWAP_TOP1_MIN)
POOLED_TOP1_MIN = 0.85         # choice: the same bar over all tokens of all prompts of the set
PROMPT_TOP1_MIN = 0.70         # choice: no single prompt below this (one drifting prompt must not hide in the pool)
MIN_PROMPTS = 5                # choice: issue 15 asks for at least five prompts
MIN_TOKENS = 32                # choice: tokens compared per prompt
CONTROL_MARGIN = 0.05          # choice: the new model's pooled agreement must exceed the control's by this much
DAMAGE_EVERY = 4               # the damage control replaces every fourth token, so its agreement is 0.75 on a long enough run
LONG_PROMPT_TOKENS = 2000      # choice: the long prompt of the set, about this many tokens: a CPU reference at this length takes minutes
CPU_REFERENCE_MAX_TOKENS = 4096  # choice: above this input length a CPU reference is too slow; compare against the base on the chip

MAX_SKIP_FRACTION = 0.25       # choice: a prompt may leave out at most this share of its positions as not recoverable from the server's text
MAX_ENTRIES = 64               # a record with more entries than this is refused
MAX_IDS = 100_000              # and so is one with more ids than this in one list


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def _is_unit(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1


# ---- the prompt set --------------------------------------------------------------------------------------------------------------------

def long_prompt_text(approx_tokens: int = LONG_PROMPT_TOKENS) -> str:
    """A deterministic long prompt: numbered filler lines with one fact near the start, and a question about that fact at the end. About
    1.3 tokens a word for an English tokenizer; the reference records the true count. The same arguments give the same text on every machine."""
    if not _is_int(approx_tokens) or approx_tokens < 100:
        raise ValueError("approx_tokens must be a whole number of at least 100")
    words_wanted = int(approx_tokens / 1.3)
    lines = ["Read the log below. It has one entry that names the access code for the archive room."]
    lines.append("Entry 0: the access code for the archive room is violet-4417.")
    words, i = sum(len(s.split()) for s in lines), 1
    while words < words_wanted:
        line = f"Entry {i}: the visitor desk logged {i % 7 + 1} deliveries and {i % 5 + 1} visitors before noon on day {i}."
        lines.append(line)
        words += len(line.split())
        i += 1
    lines.append("What is the access code for the archive room? Answer with the code only.")
    return "\n".join(lines)


def prompt_set_problems(data) -> list[str]:
    """Why `data` is not a usable agreement prompt set (empty when it is). Each prompt needs a unique string id, a string kind and a
    non-empty list of chat messages; `text` (a plain continuation for a tokenizer without a chat template) is optional."""
    if not isinstance(data, dict) or not isinstance(data.get("prompts"), list):
        return ["the prompt set has no prompts list"]
    out, seen = [], set()
    if len(data["prompts"]) < MIN_PROMPTS:
        out.append(f"the prompt set has {len(data['prompts'])} prompts and at least {MIN_PROMPTS} are needed")
    for i, p in enumerate(data["prompts"]):
        ok = (isinstance(p, dict) and isinstance(p.get("id"), str) and p["id"] and isinstance(p.get("kind"), str)
              and isinstance(p.get("messages"), list) and p["messages"]
              and all(isinstance(m, dict) and isinstance(m.get("role"), str) and isinstance(m.get("content"), str) for m in p["messages"]))
        if not ok:
            out.append(f"prompt {i} needs a string id, a string kind and a non-empty messages list of role and content")
            continue
        if p["id"] in seen:
            out.append(f"duplicate prompt id {p['id']!r}")
        seen.add(p["id"])
    return out


# ---- scoring ---------------------------------------------------------------------------------------------------------------------------

def skipped_positions(skipped, n: int) -> list[int]:
    """The positions a record leaves out of the score (issue 406), checked: a list of distinct whole numbers inside the prompt, at most
    MAX_SKIP_FRACTION of its positions. The template skips a position only when the reference token cannot be read back from the text the
    server returns (half of a character, a token that does not re-encode to itself, a stop token). A record that skips more than a quarter
    of a prompt is refused, so a list of skipped positions cannot hide mismatches."""
    if skipped is None:
        return []
    if not isinstance(skipped, list) or not all(_is_int(k) for k in skipped):
        raise ValueError("skipped must be a list of whole-number positions")
    if len(set(skipped)) != len(skipped) or any(k < 0 or k >= n for k in skipped):
        raise ValueError(f"skipped must hold distinct positions from 0 to {n - 1}")
    if len(skipped) > n * MAX_SKIP_FRACTION:
        raise ValueError(f"{len(skipped)} of {n} positions are not recoverable from the server's text, more than the {MAX_SKIP_FRACTION:.0%} "
                         "a prompt may leave out; the check is not comparable for this prompt")
    return sorted(skipped)


def stop_ids_from(*docs) -> set[int]:
    """The whole-number `eos_token_id` values (a number or a list) of the top level of each JSON document given (generation_config.json and
    config.json of the served directory). Anything else in the field is ignored; a bool is not an id."""
    out: set[int] = set()
    for doc in docs:
        value = doc.get("eos_token_id") if isinstance(doc, dict) else None
        for v in value if isinstance(value, list) else [value]:
            if _is_int(v):
                out.add(v)
    return out


def special_ids(tokenizer) -> set[int]:
    """The ids of the tokenizer's special added tokens; empty when the tokenizer cannot list them."""
    try:
        return {int(i) for i, tok in tokenizer.get_added_tokens_decoder().items() if getattr(tok, "special", False) is True}
    except (AttributeError, TypeError, ValueError):
        return set()


def unrecoverable_ids(tokenizer, stops, ids) -> set[int]:
    """Which of `ids` cannot be read back from the text a server returns for one generated token (issue 406): a stop id (`stops` plus the
    tokenizer's special tokens), a token whose text is empty or holds a replacement character, and a token whose text re-encodes to other
    ids. It reads the REFERENCE ids, so what the chip answered cannot move a position in or out of the score. The serve-and-compare template
    has the same rule in its own file (it cannot import this one), and a test keeps the two equal."""
    stopped = set(stops) | special_ids(tokenizer)
    out: set[int] = set()
    for i in set(ids):
        if i in stopped:
            out.add(i)
            continue
        text = tokenizer.decode([i])
        back = tokenizer.encode(text, add_special_tokens=False).ids if text else []
        if not text or "\ufffd" in text or back[:1] != [i]:
            out.add(i)
    return out


def expected_skipped(tokenizer, stops, reference_ids) -> list[int]:
    """The positions of `reference_ids` a correct run leaves out of the score: what the template's mask must be, and what the gate compares
    a record's `skipped` list with."""
    bad = unrecoverable_ids(tokenizer, stops, reference_ids)
    return [k for k, r in enumerate(reference_ids) if r in bad]


def score_prompt(reference_ids, chip_ids, skipped=None) -> dict:
    """Teacher-forced agreement of one prompt: `chip_ids[k]` is the first id the chip returned at position k when it was given the
    reference's first k tokens, and `reference_ids[k]` is the reference's token. A missing or None chip id is a mismatch. The two lists
    must have the same length; a shorter chip list is a reason, not a silent shorter score. `skipped` lists positions left out of both the
    matches and the count (see skipped_positions); `n` is the number of positions compared and `skipped` the number left out."""
    if (not isinstance(reference_ids, list) or not isinstance(chip_ids, list) or not reference_ids
            or len(reference_ids) > MAX_IDS or len(chip_ids) > MAX_IDS):
        raise ValueError("reference_ids and chip_ids must be non-empty lists of at most %d ids" % MAX_IDS)
    if len(chip_ids) != len(reference_ids):
        raise ValueError(f"the chip list has {len(chip_ids)} ids and the reference has {len(reference_ids)}")
    if not all(_is_int(x) for x in reference_ids) or not all(x is None or _is_int(x) for x in chip_ids):
        raise ValueError("ids must be whole numbers (a chip id may be None)")
    left_out = set(skipped_positions(skipped, len(reference_ids)))
    pairs = [(k, r, c) for k, (r, c) in enumerate(zip(reference_ids, chip_ids)) if k not in left_out]
    matches = sum(1 for _, r, c in pairs if r == c)
    first = next((k for k, r, c in pairs if r != c), None)
    return {"n": len(pairs), "matches": matches, "top1": matches / len(pairs), "first_mismatch": first, "skipped": len(left_out)}


def damaged_copy(reference_ids, vocab_size: int | None = None, every: int = DAMAGE_EVERY) -> list[int]:
    """The reference with every `every`-th token (positions every-1, 2*every-1 ...) replaced by a different id: a deliberately wrong run."""
    mod = vocab_size if _is_int(vocab_size) and vocab_size > 1 else None
    out = list(reference_ids)
    for k in range(every - 1, len(out), every):
        out[k] = (out[k] + 1) % mod if mod else out[k] + 1
    return out


def summarise(entries, *, pooled_min: float = POOLED_TOP1_MIN, prompt_min: float = PROMPT_TOP1_MIN,
              min_prompts: int = MIN_PROMPTS, min_tokens: int = MIN_TOKENS) -> dict:
    """Score every entry ({id, kind, reference_ids, chip_ids}) and judge the set. `passes` is true when there are enough prompts, every
    prompt compared enough tokens, the pooled agreement meets `pooled_min` and no prompt falls below `prompt_min`. `reasons` names each failure.
    A malformed entry is a reason; the other entries are still scored."""
    per, reasons = [], []
    if not isinstance(entries, list) or len(entries) > MAX_ENTRIES:
        return {"passes": False, "per_prompt": [], "pooled_top1": None, "worst_top1": None, "worst_id": None,
                "reasons": ["the agreement set has no entries list (or more than %d)" % MAX_ENTRIES]}
    for e in entries:
        if not isinstance(e, dict):
            reasons.append("an entry of the agreement set is not an object")
            continue
        pid = e.get("id")
        try:
            s = score_prompt(e.get("reference_ids"), e.get("chip_ids"), e.get("skipped"))
        except ValueError as exc:
            reasons.append(f"prompt {pid!r}: {exc}")
            continue
        per.append({"id": pid, "kind": e.get("kind"), **s})
        if s["n"] + s["skipped"] < min_tokens:          # the run must cover min_tokens positions; skipped ones are capped by MAX_SKIP_FRACTION
            reasons.append(f"prompt {pid!r} compared {s['n']} tokens and at least {min_tokens} are needed")
    if len(per) < min_prompts:
        reasons.append(f"{len(per)} prompts were scored and at least {min_prompts} are needed")
    total = sum(p["n"] for p in per)
    pooled = sum(p["matches"] for p in per) / total if total else None
    worst = min(per, key=lambda p: p["top1"]) if per else None
    if pooled is not None and pooled < pooled_min:
        reasons.append(f"pooled top-1 agreement {pooled:.3f} over {total} tokens is below the bar {pooled_min}")
    for p in per:
        if p["top1"] < prompt_min:
            reasons.append(f"prompt {p['id']!r} agrees on {p['top1']:.3f} of its tokens, below the per-prompt bar {prompt_min}")
    return {"passes": not reasons, "per_prompt": per, "pooled_top1": pooled,
            "worst_top1": None if worst is None else worst["top1"], "worst_id": None if worst is None else worst["id"],
            "reasons": reasons}


def calibrate(entries, *, vocab_size: int | None = None, **bars) -> dict:
    """The damage control. Replace every DAMAGE_EVERY-th chip id of every entry with a wrong id, score that with the same bars as the real
    record, and say whether the bars can tell the two apart. `usable` is true when the damaged run fails the bars. `real_passes` repeats
    whether the real record passes, so a reader sees both sides in one place. Entries that cannot be scored are left out of the damaged run."""
    real = summarise(entries, **bars)
    damaged_entries = []
    for e in entries if isinstance(entries, list) else []:
        try:
            ref = e["reference_ids"]
            score_prompt(ref, ref, e.get("skipped"))  # the same shape rules as a real entry
            left_out = set(skipped_positions(e.get("skipped"), len(ref)))
            measured = [k for k in range(len(ref)) if k not in left_out]
            chips = list(ref)                         # the damage goes on the compared positions, so a skipped position cannot hide it
            for k, v in zip(measured, damaged_copy([ref[k] for k in measured], vocab_size)):
                chips[k] = v
            damaged_entries.append({"id": e.get("id"), "kind": e.get("kind"), "reference_ids": ref, "chip_ids": chips,
                                    "skipped": e.get("skipped")})
        except (ValueError, KeyError, TypeError, AttributeError):
            continue
    damaged = summarise(damaged_entries, **bars)
    usable = damaged["passes"] is False and damaged["pooled_top1"] is not None
    return {"usable": usable, "real_passes": real["passes"], "damaged_pooled_top1": damaged["pooled_top1"],
            "damaged_passes": damaged["passes"], "damage": f"every {DAMAGE_EVERY}th token replaced",
            "reason": None if usable else "the bars pass a run with every %dth token wrong, so they cannot tell a wrong run from a good one" % DAMAGE_EVERY}


def control_verdict(new_summary: dict, control_summary: dict | None, *, margin: float = CONTROL_MARGIN) -> dict:
    """The new model's pooled agreement read against the control's (the nearest model served from the same bundle on the same prompts).
    `separates` is true when the new model exceeds the control by at least `margin`; None when there is no control."""
    if control_summary is None:
        return {"present": False, "separates": None, "margin": margin}
    a, b = new_summary.get("pooled_top1"), control_summary.get("pooled_top1")
    if not _is_unit(a) or not _is_unit(b):
        return {"present": True, "separates": False, "margin": margin, "new_pooled_top1": a, "control_pooled_top1": b,
                "reason": "the new model or the control has no pooled agreement"}
    sep = a - b >= margin
    out = {"present": True, "separates": sep, "margin": margin, "new_pooled_top1": a, "control_pooled_top1": b}
    if not sep:
        out["reason"] = (f"the new model's pooled agreement {a:.3f} is not above the control's {b:.3f} by {margin}: the check cannot tell "
                         "this model from the control, so it cannot show the chip serves the new weights")
    return out


def id_problems(entries, expected_ids) -> list[str]:
    """Why the entries' prompt ids are not exactly the expected ids, each once (empty when they are). Five copies of one prompt, a missing
    long-context prompt or an extra prompt are all problems. An id that is not a string is a problem (it could not be matched)."""
    if not isinstance(entries, list):
        return ["the agreement set has no entries list"]
    ids = [e.get("id") if isinstance(e, dict) else None for e in entries]
    if not all(isinstance(i, str) for i in ids):
        return ["every entry of the agreement set needs a string id"]
    out = []
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        out.append(f"prompt ids appear more than once: {dup}")
    missing, extra = sorted(set(expected_ids) - set(ids)), sorted(set(ids) - set(expected_ids))
    if missing:
        out.append(f"prompts missing from the agreement set: {missing}")
    if extra:
        out.append(f"prompts that are not in the shipped set: {extra}")
    return out


def control_problems(control_entries, entries) -> list[str]:
    """Why a control cannot be used for the margin (empty when it can). A control is a complete run of the same prompts against the same
    reference ids. A control with a failed request, a missing id (`None`), another list of reference ids or another set of prompts would
    lower its pooled agreement for reasons that say nothing about the nearest model, and the margin would pass. A complete control that
    agrees poorly is allowed: that is what a different model looks like."""
    if not isinstance(control_entries, list) or not isinstance(entries, list):
        return ["the control has no entries list"]
    mine = {e.get("id"): e.get("reference_ids") for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}
    mine_skipped = {e.get("id"): e.get("skipped") or [] for e in entries if isinstance(e, dict) and isinstance(e.get("id"), str)}
    out, seen = [], set()
    for c in control_entries:
        if not isinstance(c, dict) or not isinstance(c.get("id"), str):
            out.append("a control entry has no string id")
            continue
        cid = c["id"]
        if cid in seen:
            out.append(f"control prompt {cid!r} appears more than once")
        seen.add(cid)
        if cid not in mine:
            out.append(f"control prompt {cid!r} is not in the run's set")
            continue
        if c.get("reference_ids") != mine[cid]:
            out.append(f"control prompt {cid!r} has other reference ids than the run")
        if (c.get("skipped") or []) != mine_skipped[cid]:
            out.append(f"control prompt {cid!r} skipped other positions than the run, so the two scores count other tokens")
        chip = c.get("chip_ids")
        if c.get("error") not in (None, ""):
            out.append(f"control prompt {cid!r} had a failed request: {str(c['error'])[:120]}")
        try:
            left_out = set(skipped_positions(c.get("skipped"), len(c["reference_ids"])))
        except (ValueError, KeyError, TypeError):
            out.append(f"control prompt {cid!r} has a malformed skipped list")
            left_out = set()
        if not isinstance(chip, list) or any(x is None for k, x in enumerate(chip) if k not in left_out):
            out.append(f"control prompt {cid!r} has missing chip ids")
    for cid in sorted(set(mine) - seen):
        out.append(f"control prompt {cid!r} is missing")
    return out


def judge(record, *, vocab_size: int | None = None, expected_ids=None, **bars) -> dict:
    """Everything the gate needs from one `agreement_set` record, recomputed from its raw ids: {passes, reasons, summary, calibration,
    control}. The reasons are the gate's. A record that did not run has no reasons here (the set is opt-in; the gate decides whether the
    run asked for it). `expected_ids` is the shipped prompt set's ids: when given, the record must hold exactly those, each once. A
    control that is incomplete is a reason and is never read as a weak model."""
    if not isinstance(record, dict) or record.get("ran") is not True:
        return {"passes": True, "reasons": [], "summary": None, "calibration": None, "control": None, "ran": False}
    entries = record.get("entries")
    summary = summarise(entries, **bars)
    reasons = list(summary["reasons"])
    if expected_ids is not None:
        reasons += id_problems(entries, expected_ids)
    for e in entries if isinstance(entries, list) else []:
        if isinstance(e, dict) and e.get("error") not in (None, ""):
            reasons.append(f"prompt {e.get('id')!r} had a failed request: {str(e['error'])[:120]}")
    cal = calibrate(entries, vocab_size=vocab_size, **bars)
    if not cal["usable"]:
        reasons.append("the agreement bars are unusable: " + (cal["reason"] or "the damage control could not be scored"))
    control = record.get("control")
    if control is None:
        verdict = control_verdict(summary, None)
    else:
        problems = control_problems(control.get("entries") if isinstance(control, dict) else None, entries)
        if problems:
            verdict = {"present": True, "separates": False, "margin": CONTROL_MARGIN, "reason": "the control cannot be used: " + "; ".join(problems[:4])}
        else:
            verdict = control_verdict(summary, summarise(control["entries"], **bars))
    if verdict["present"] and verdict["separates"] is not True:
        reasons.append(verdict.get("reason", "the control does not separate"))
    elif record.get("control_expected") is True and not verdict["present"]:
        reasons.append("the run asked for a control and the record holds none")
    return {"passes": not reasons, "reasons": reasons, "summary": summary, "calibration": cal, "control": verdict, "ran": True}


def baseline_record(top1_agreement, n_tokens) -> dict:
    """The 32-token check's number, kept next to the set's verdict (issue 15: keep the old bar as a recorded baseline)."""
    return {"check": "32-token teacher-forced, one prompt", "top1_agreement": top1_agreement, "n_tokens": n_tokens,
            "bar": BASELINE_TOP1_MIN}


# ---- long-context sweep design (issue 18) ----------------------------------------------------------------------------------------------

DEFAULT_LENGTHS = (1024, 8192, 32768, 100000, 200000)     # input tokens; the bundle advertises 262144
DEFAULT_CONCURRENCY = (1, 4)                              # the benchmark so far uses 4 sequences
DECODE_DROP_WARN = 0.5         # choice: a decode speed below this share of the shortest input's is called out


def sweep_plan(lengths=DEFAULT_LENGTHS, concurrency=DEFAULT_CONCURRENCY, *, cpu_max: int = CPU_REFERENCE_MAX_TOKENS) -> list[dict]:
    """The cells of a speed sweep, shortest first. Each cell says how its output would be checked: against the CPU reference when the
    input is at most `cpu_max` tokens, otherwise against the base model served on the chip (a CPU reference at 100K tokens is too slow)."""
    if not lengths or not concurrency or not all(_is_int(n) and n > 0 for n in (*lengths, *concurrency)):
        raise ValueError("lengths and concurrency must be non-empty lists of positive whole numbers")
    return [{"input_tokens": n, "concurrency": c, "reference": "cpu" if n <= cpu_max else "base-on-chip"}
            for n in sorted(set(lengths)) for c in sorted(set(concurrency))]


def sweep_summary(cells) -> dict:
    """Per cell: prefill tokens/s and decode tokens/s as measured; per concurrency: decode speed as a share of the shortest input's, and
    the cells where it falls below DECODE_DROP_WARN. A cell without a number keeps None and is counted, so a missing cell is not a pass."""
    out, by_c, missing = [], {}, 0
    for c in cells if isinstance(cells, list) else []:
        if not isinstance(c, dict):
            continue
        dec, pre = c.get("decode_tok_s"), c.get("prefill_tok_s")
        ok = isinstance(dec, (int, float)) and not isinstance(dec, bool) and math.isfinite(dec) and dec > 0
        if not ok:
            missing += 1
        out.append({"input_tokens": c.get("input_tokens"), "concurrency": c.get("concurrency"),
                    "prefill_tok_s": pre if isinstance(pre, (int, float)) and not isinstance(pre, bool) else None,
                    "decode_tok_s": dec if ok else None, "error": c.get("error")})
        if ok:
            by_c.setdefault(c.get("concurrency"), []).append((c.get("input_tokens"), dec))
    drops = []
    for conc, rows in by_c.items():
        rows.sort(key=lambda r: r[0])
        base = rows[0][1]
        for n, dec in rows[1:]:
            if dec < DECODE_DROP_WARN * base:
                drops.append({"concurrency": conc, "input_tokens": n, "decode_tok_s": dec, "share_of_shortest": round(dec / base, 3)})
    return {"cells": out, "missing_cells": missing, "decode_drops": drops, "drop_warn_share": DECODE_DROP_WARN}


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)
