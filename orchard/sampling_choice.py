# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 2 chooses the served configuration by what the model is for (issue 107).

A fast decoding path can be greedy only. The Qwen3.8 DFlash2 bundle with its drafter on accepts a request with temperature above 0 and samples
only the first token (docs/run-logs/2026-10-27-greedy-only-sampling.md). That is the right trade for a model that is used greedily and the wrong one
for a model that is used with sampling. This module holds the rule that picks between the configuration stage 2 served and a drafter-off variant:

  * `need` reads whether the model's purpose needs sampling, from files (the operator's flag, the card's warning against greedy decoding, the
    model's own generation_config.json, the card's recommended settings);
  * `side_from_report` reads a swap check's sampling probe (orchard/sampling_probe.py) into the numbers the rule uses;
  * `decide` is the rule: the table in docs/superpowers/specs/2026-10-27-sampling-choice-design.md;
  * `decision_problems` recomputes a decision from the files on disk, for the stage 2 gate.

Every number is recomputed from the probe's raw requests. A summary written into a record is never read.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

from orchard import defaults, safeio, sampling_probe

DECISION_FILE = "sampling-decision.json"      # in stages/2/
DECISION_PATH = "stages/2/" + DECISION_FILE   # run-relative; the package ships a scrubbed copy at evidence/<this>
TRIAL_DIR = "sampling-trial"                  # in stages/2/
TRIAL_RESULT = "trial-result.json"            # in TRIAL_DIR, written by the template script
GRAFT_TRIAL_DIR = "graft-trial"               # the grafted variant's own swap check, read for its sampling verdict
OVERRIDES = ("auto", "needed", "not-needed")
CHOICES = ("keep", "drafter-off")
SLOW = "sampling-applied-below-floor"      # the chosen configuration applies sampling but decodes below the floor (the operator preferred it to a fast one that does not)
FLAGS = (SLOW, "no-variant-serves-purpose", "unverified", "variant-not-tried")
APPLIED = ("sampled-full-speed", "sampled-slower")          # the verdicts that show sampling is applied
BAD = ("sampled-ignored", "sampled-refused", "sampled-incoherent", "greedy-refused")   # the verdicts that show it is not
# The probe's suspicion: sampled replies equal the greedy text or each other, with no high-temperature request that confirms it. That fits ignored sampling
# and also correct sampling where one token is eligible at every step. For a model that needs sampling it is treated like `sampled-ignored` for the choice
# (try the drafter-off variant; trusting it wrongly ships a greedy-only server), and every record says it is suspected and not confirmed.
SUSPECT = ("sampled-matches-greedy",)
CARD_MARKERS = {"keep": "kept the configuration it served", "drafter-off": "chose the drafter-off configuration"}


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


MAX_JSON_BYTES = 4 << 20       # the largest file of this module's kind (a swap check or a trial result) is far smaller


def _read_json(path):
    """A JSON file of the run, read through orchard/safeio.py (no link at the last component, a size cap, a regular file only, strict JSON).
    None when it is missing, unreadable, too large or not JSON; the callers treat all of those as "no evidence"."""
    try:
        return safeio.read_json(path, max_bytes=MAX_JSON_BYTES)
    except safeio.SafeReadError:
        return None


# ---- is sampling needed? ------------------------------------------------------------------------------------------------------

def _sampling_value(name, value) -> bool:
    """Is this generation-config or card value a setting that makes decoding sample?"""
    if not _number(value):
        return False
    if name == "temperature":
        return value > 0
    if name == "top_p":
        return 0 < value < 1
    if name == "top_k":
        return value > 0                  # 0 and -1 both turn top-k filtering off
    if name == "min_p":
        return value > 0
    return False


def need(run_dir, override: str = "auto") -> dict:
    """{needed, source, override, evidence}: does this model's purpose need sampling? In order: the operator's flag, a card warning against greedy
    decoding, the model's own generation_config.json (not the nearest model's copy that staging adds), the card's recommended settings, and
    otherwise no. `evidence` quotes the file and the value or line that decided it; it says so when there was none."""
    if override not in OVERRIDES:
        raise ValueError(f"sampling override must be one of {', '.join(OVERRIDES)}, got {override!r}")
    if override != "auto":
        return {"needed": override == "needed", "source": "operator", "override": override,
                "evidence": [f"the operator passed --sampling {override}"]}
    g = _read_json(Path(run_dir) / "stages" / "0" / "evidence" / "genconfig-license.json")
    if not isinstance(g, dict):
        return {"needed": False, "source": "none", "override": override,
                "evidence": ["no evidence: stage 0's evidence/genconfig-license.json is missing or unreadable"]}
    warns = [w for w in g.get("card_warns_greedy") or [] if isinstance(w, dict) and isinstance(w.get("quote"), str)]
    if warns:
        w = warns[0]
        return {"needed": True, "source": "card-warning", "override": override,
                "evidence": [f"the model card warns against greedy decoding (line {w.get('line')}): {w['quote'][:200]}"]}
    own = g.get("model_generation_config") if g.get("model_generation_config_present") is True else None
    own = own if isinstance(own, dict) else None
    if own is not None:
        found = [f"{k} {own[k]}" for k in ("temperature", "top_p", "top_k", "min_p") if _sampling_value(k, own.get(k))]
        if own.get("do_sample") is True:
            found.insert(0, "do_sample true")
        if found:
            return {"needed": True, "source": "generation_config", "override": override,
                    "evidence": ["the model's own generation_config.json sets " + ", ".join(found)]}
    hints = g.get("card_sampling_hints") if isinstance(g.get("card_sampling_hints"), dict) else {}
    found = [f"{k} {v}" for k in ("temperature", "top_p", "top_k", "min_p") for v in (hints.get(k) or []) if _sampling_value(k, v)]
    if found:
        return {"needed": True, "source": "card-hints", "override": override,
                "evidence": ["the model card recommends " + ", ".join(found)]}
    if own is not None and own.get("do_sample") is False:
        return {"needed": False, "source": "generation_config", "override": override,
                "evidence": ["the model's own generation_config.json sets do_sample false"]}
    return {"needed": False, "source": "none", "override": override,
            "evidence": ["no evidence that the model needs sampling: its own generation_config.json sets no sampling default and its card "
                         "recommends no sampling setting and does not warn against greedy decoding"]}


# ---- what a configuration does with a sampled request ------------------------------------------------------------------------------

def drafter_of(serving) -> str | None:
    """"on" or "off" from a swap check's `serving.drafter` text, None when the run script does not say."""
    text = serving.get("drafter") if isinstance(serving, dict) else None
    if not isinstance(text, str):
        return None
    if text.strip() == "off":
        return "off"
    return "on" if text.strip().startswith("on") else None


def side_from_report(report) -> dict | None:
    """{drafter, verdict, greedy_tok_s, sampled_tok_s, served} from one swap-check.json, or None when it is not an object. The speeds are
    recomputed from the probe's tokens and seconds."""
    if not isinstance(report, dict):
        return None
    s = sampling_probe.summarize(report.get("sampling_probe"))
    draft = report.get("result_draft")
    return {"drafter": drafter_of(report.get("serving")), "verdict": s["verdict"],
            "greedy_tok_s": (s["greedy"] or {}).get("tokens_per_s"), "sampled_tok_s": (s["sampled"] or {}).get("tokens_per_s"),
            "served": draft.get("serves") is True if isinstance(draft, dict) else None}


def speed(side) -> float | None:
    """The speed a request is served at: the sampled requests' mean, else the greedy request's."""
    for key in ("sampled_tok_s", "greedy_tok_s"):
        v = side.get(key) if isinstance(side, dict) else None
        if _number(v) and v > 0:
            return float(v)
    return None


def speed_basis(side) -> str | None:
    """Which request `speed` read: "sampled" (the sampled requests' mean), "greedy" (no sampled speed exists, for example a refused sampled request),
    or None (no speed). The decision records it so a reader knows a figure is a fallback."""
    for key, name in (("sampled_tok_s", "sampled"), ("greedy_tok_s", "greedy")):
        v = side.get(key) if isinstance(side, dict) else None
        if _number(v) and v > 0:
            return name
    return None


def needs_variant(need_: dict, current: dict) -> bool:
    """Can the rule not be settled without serving the drafter-off variant? Sampling is needed, the served configuration has the drafter on,
    and its probe does not show that sampling is applied."""
    return bool(need_.get("needed")) and current.get("drafter") == "on" and current.get("verdict") not in APPLIED


# ---- the rule -------------------------------------------------------------------------------------------------------------------------

def _fmt(v) -> str:
    return "an unmeasured speed" if v is None else f"{v:.1f} tokens/s"


def comparable(numbers: dict) -> bool:
    """Were both speeds measured, on the same basis (both sampled or both greedy)? Only then may a decision, a card or a bundle say one is faster. A
    measurement is a finite number above zero (a zero or negative speed is not one), and a basis is one of the two the probe records and the same on
    both sides."""
    served, var = numbers.get("served_tok_s"), numbers.get("variant_tok_s")
    basis = numbers.get("served_speed_basis")
    return (_number(served) and served > 0 and _number(var) and var > 0
            and basis in ("sampled", "greedy") and basis == numbers.get("variant_speed_basis"))


def decide(need_: dict, current: dict, variant: dict | None = None, *, variant_note: str | None = None, floor: float | None = None) -> dict:
    """The choice and why. `current` is the configuration stage 2 serves, `variant` the drafter-off variant when it was served (None when it
    was not tried; `variant_note` says why). See the design note's table for every row."""
    floor = defaults.SAMPLING_MIN_TOKS_PER_S if floor is None else floor
    cur_speed = speed(current)
    var_speed = speed(variant) if variant else None
    numbers = {"served_drafter": current.get("drafter"), "served_verdict": current.get("verdict"),
               "served_tok_s": cur_speed, "served_greedy_tok_s": current.get("greedy_tok_s"),
               "variant_served": variant.get("served") if variant else None,
               "variant_verdict": variant.get("verdict") if variant else None,
               "variant_tok_s": var_speed, "variant_greedy_tok_s": variant.get("greedy_tok_s") if variant else None,
               "served_speed_basis": speed_basis(current), "variant_speed_basis": speed_basis(variant) if variant else None}

    def out(choice, flag, reason):
        return {"choice": choice, "flag": flag, "reason": reason, "numbers": numbers, "thresholds": {"min_tok_s": floor},
                "needed": bool(need_.get("needed")), "need_source": need_.get("source"), "need_evidence": list(need_.get("evidence") or []),
                "override": need_.get("override"), "variant_note": variant_note}

    verdict = current.get("verdict")
    if not need_.get("needed"):
        return out("keep", None, f"sampling is not needed for this model ({need_.get('source')}), so the configuration stage 2 served stays "
                                 f"({_fmt(cur_speed)})")
    if verdict in APPLIED:
        if cur_speed is None:
            return out("keep", "unverified", "the served configuration applies sampling, but its speed was not measured, so the floor "
                                             f"of {floor:g} tokens/s could not be checked")
        if cur_speed < floor:
            return out("keep", SLOW, f"the served configuration applies sampling but is slow: it decodes at {_fmt(cur_speed)}, below the floor "
                                     f"of {floor:g} tokens/s, and no other configuration is available; it is kept and needs the "
                                     "operator's attention")
        return out("keep", None, f"the served configuration applies sampling and decodes at {_fmt(cur_speed)}, at or above the floor of "
                                 f"{floor:g} tokens/s, and the model needs sampling ({need_.get('source')})")
    # Sampling is needed and the served configuration does not show that it applies it.
    measured_bad = verdict in BAD
    why_current = ("ignores sampling (the probe's verdict is sampled-ignored)" if verdict == "sampled-ignored" else
                   "may not apply sampling: the probe's verdict is sampled-matches-greedy (suspected, not confirmed)" if verdict in SUSPECT else
                   f"does not apply sampling (the probe's verdict is {verdict})" if measured_bad else
                   f"has no usable sampling verdict (the probe's verdict is {verdict})")
    if current.get("drafter") != "on":
        flag = "no-variant-serves-purpose" if measured_bad else "unverified"
        return out("keep", flag, f"the model needs sampling, and the served configuration {why_current}. It has no drafter to switch off, so "
                                 "no drafter-off variant exists; it is kept and needs the operator's attention")
    if variant is None:
        note = f": {variant_note}" if variant_note else ""
        return out("keep", "variant-not-tried", f"the model needs sampling and the served configuration {why_current}, but the drafter-off "
                                                f"variant was not tried{note}; the served configuration is kept and needs the operator's attention")
    v_verdict = variant.get("verdict")
    if variant.get("served") is not True:
        flag = "no-variant-serves-purpose" if measured_bad else "unverified"
        failed = f" or failed its checks ({variant['failure']})" if variant.get("failure") else ""
        return out("keep", flag, f"the model needs sampling and the served configuration {why_current}; the drafter-off variant did not serve{failed}, "
                                 "so the served configuration is kept and needs the operator's attention")
    both = comparable(numbers)

    def neither(why: str) -> dict:
        """Neither configuration applies sampling and clears the floor. When both speeds were measured on the same basis (both from sampled requests,
        or both from greedy ones) the faster one is used and the choice is flagged. Equal speeds, a speed that was not measured, or speeds from
        different requests leave the configuration stage 2 served: nothing shows the variant is faster."""
        if both and var_speed > cur_speed:
            return out("drafter-off", "no-variant-serves-purpose",
                       f"the model needs sampling; neither configuration does what it is for: {why}. The drafter-off variant is the faster of the two "
                       f"({_fmt(var_speed)} against {_fmt(cur_speed)}), so it is chosen and needs the operator's attention")
        faster = ("The served configuration decodes at least as fast as the variant" if both else
                  "The two speeds could not be compared (one was not measured, or they come from different requests), so the served configuration stays")
        return out("keep", "no-variant-serves-purpose",
                   f"the model needs sampling; neither configuration does what it is for: {why}. {faster}, and it needs the operator's attention")

    if v_verdict in BAD:
        return neither(f"the served configuration {why_current}, and the drafter-off variant's probe verdict is {v_verdict}")
    if v_verdict in APPLIED and var_speed is not None and var_speed < floor:
        # The operator's rule: a configuration that cannot do what the model is for has no value however fast it is, so the one that applies sampling
        # is chosen even though it is slow, and the decision is flagged so the operator sees it.
        return out("drafter-off", SLOW, f"the model needs sampling and the served configuration {why_current}; the drafter-off variant applies sampling "
                                        f"but is slow: it decodes at {_fmt(var_speed)}, below the floor of {floor:g} tokens/s. It is chosen because a "
                                        "configuration that does not apply sampling cannot do what the model is for, and it needs the operator's attention")
    if var_speed is not None and var_speed < floor:
        return neither(f"the served configuration {why_current} ({_fmt(cur_speed)}), and the drafter-off variant decodes at {_fmt(var_speed)}, "
                       f"below the floor of {floor:g} tokens/s")
    verified = (measured_bad or verdict in SUSPECT) and v_verdict in APPLIED and var_speed is not None
    flag = None if verified else "unverified"
    served = f"the served configuration {why_current}" + (f" at {_fmt(cur_speed)}" if cur_speed is not None else "")
    if v_verdict in APPLIED and var_speed is not None:
        said = f"the drafter-off variant applies sampling and decodes at {_fmt(var_speed)}, at or above the floor of {floor:g} tokens/s"
    elif v_verdict in APPLIED:
        said = "the drafter-off variant applies sampling, but its speed was not measured, so the floor was not checked"
    elif var_speed is not None:
        said = (f"the drafter-off variant has no sampling verdict ({v_verdict}) and decodes at {_fmt(var_speed)}, at or above the floor of "
                f"{floor:g} tokens/s")
    else:
        said = f"the drafter-off variant has no sampling verdict ({v_verdict}) and no measured speed"
    tail = "" if verified else "; this choice rests on the documented host-sampling path of the drafter-off configuration"
    return out("drafter-off", flag, f"the model needs sampling ({need_.get('source')}) and {served}; {said}, so it is chosen{tail}")

NEED_WORDS = {"operator": "the operator said so", "card-warning": "the model card warns against greedy decoding",
              "generation_config": "the model's own generation_config.json sets sampling defaults",
              "card-hints": "the model card recommends sampling settings", "none": "nothing in the model's files says it does"}


def served_clause(verdict) -> str:
    """What the probe established about the configuration stage 2 first served, as a clause for the card (no number): it ignored sampled requests, did
    not apply sampling, or has no usable sampling verdict. Only `sampled-ignored` says the drafter ignored sampling."""
    return ("ignored sampled requests" if verdict == "sampled-ignored" else
            "gave sampled replies equal to the greedy text or to each other, which suggests sampling was not applied (suspected, not confirmed)"
            if verdict in SUSPECT else
            "did not apply sampling" if verdict in BAD else "has no usable sampling verdict")


def card_sentence(decision: dict) -> str:
    """The package card's plain sentences for the choice. They carry the marker phrase the gate looks for and no number: the card's Numbers table is
    the only place for a measured speed (the card gate refuses one in prose), and this sentence points to the sampling rows in that table."""
    n = decision.get("numbers") or {}
    why = NEED_WORDS.get(decision.get("need_source"), "the sampling decision says so")
    verdict = n.get("served_verdict")
    first = served_clause(verdict)
    flag = decision.get("flag")
    if decision.get("choice") == "drafter-off":
        s = (f"Stage 2 chose the drafter-off configuration. The model needs sampling ({why}), and the configuration stage 2 first served, with the "
             f"speculative drafter on, {first}.")
        if flag is None:
            s += (" The drafter-off configuration applies sampling and decodes above the usability floor in `orchard/defaults.py`. Its speeds are "
                  "in the Numbers table, in the rows marked sampling probe.")
        elif flag == SLOW:
            s += (" The drafter-off configuration applies sampling but is slow: it decodes below the usability floor in `orchard/defaults.py`. It is "
                  "chosen because a configuration that does not apply sampling cannot do what the model is for. Its speeds are in the Numbers table, "
                  "in the rows marked sampling probe.")
        elif flag == "no-variant-serves-purpose":
            s += (" Neither configuration that stage 2 measured both applies sampling and decodes above the usability floor, so this package serves "
                  "the drafter-off configuration, which does not do what the model needs.")
            if comparable(n) and n["variant_tok_s"] > n["served_tok_s"]:
                s += " It decodes faster than the configuration stage 2 first served."
        else:
            s += (" This choice relies on the documented host-sampling path of the drafter-off configuration. A sampling verdict or a speed was not "
                  "measured for it, so this card does not claim that sampling was shown to apply or that the usability floor was cleared.")
    elif not decision.get("needed"):
        s = f"Stage 2 kept the configuration it served. Sampling is not needed for this model ({why})."
    elif verdict in APPLIED:
        s = (f"Stage 2 kept the configuration it served. The model needs sampling ({why}) and this configuration applies it. "
             "The speeds are in the Numbers table, in the rows marked sampling probe.")
    else:
        s = f"Stage 2 kept the configuration it served. The model needs sampling ({why})."
    if decision.get("choice") == "keep" and flag == "no-variant-serves-purpose":
        s += (" This package does not do what the model needs: no configuration that stage 2 measured both applies sampling and decodes above the "
              "usability floor, so the configuration stage 2 served is kept.")
        if comparable(n) and n["served_tok_s"] >= n["variant_tok_s"]:
            s += " It decodes at least as fast as the drafter-off variant."
        s += " Read the decision file before relying on it."
    elif flag == SLOW and decision.get("choice") == "keep":
        s += " This package applies sampling but is slow: it decodes below the usability floor. Read the decision file before relying on it."
    elif flag == SLOW:
        s += " Read the decision file before relying on it."
    elif flag in ("unverified", "variant-not-tried"):
        s += " This choice is not fully verified: a verdict or a speed was not measured. The decision file says which."
    return s + f" The decision and the numbers behind it are in `evidence/{DECISION_PATH}`."


# ---- the files -----------------------------------------------------------------------------------------------------------------------

def write_decision(stage_dir, decision: dict, *, trial: bool) -> Path:
    """Write stages/2/sampling-decision.json whole or not at all. `trial` says whether the drafter-off variant was served."""
    sd = Path(stage_dir)
    record = {**decision, "trial": f"{TRIAL_DIR}/{TRIAL_RESULT}" if trial else None}
    path = sd / DECISION_FILE
    tmp = sd / (DECISION_FILE + ".tmp")
    tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def read_decision(stage_dir) -> dict | None:
    d = _read_json(Path(stage_dir) / DECISION_FILE)
    return d if isinstance(d, dict) and d.get("choice") in CHOICES else None


def drafter_off_chosen(run_dir) -> bool:
    """Did stage 2 choose the drafter-off variant? False for a missing or malformed decision, so nothing downstream changes without a record."""
    d = read_decision(Path(run_dir) / "stages" / "2")
    return d is not None and d["choice"] == "drafter-off"


def read_override(run_dir) -> str:
    """The run's `--sampling` value from its ledger's run_start, "auto" when there is none."""
    from orchard import ledger
    try:
        entries = ledger.read_entries(Path(run_dir) / "ledger.jsonl")
    except (ledger.LedgerCorrupt, OSError):
        return "auto"
    for e in entries:
        if e["event"] == "run_start":
            v = e["data"].get("sampling")
            return v if v in OVERRIDES else "auto"
    return "auto"


def served_source(stage_dir) -> str:
    """The swap check, relative to stage 2's directory, that describes the configuration stage 2 serves now: the grafted variant's when the graft
    decision is graft, else stage 2's own."""
    g = _read_json(Path(stage_dir) / "graft-decision.json")
    if isinstance(g, dict) and g.get("choice") == "graft":
        variant = f"{GRAFT_TRIAL_DIR}/variant/evidence/swap-check.json"
        report = _read_json(Path(stage_dir) / variant)
        probe = report.get("sampling_probe") if isinstance(report, dict) else None
        if isinstance(probe, dict) and probe.get("ran"):          # a variant with no probe leaves stage 2's own swap check as the only one
            return variant
    return "evidence/swap-check.json"


def served_side(stage_dir) -> dict | None:
    """The configuration stage 2 serves now (see `served_source`)."""
    return side_from_report(_read_json(Path(stage_dir) / served_source(stage_dir)))


def probe_source(stage_dir) -> str:
    """The swap check, relative to stage 2's directory, that describes the configuration the package serves: the drafter-off variant's when stage 2
    chose it, else `served_source`. The package card's sampling rows and the stage 7 gate read the probe from there."""
    d = read_decision(stage_dir)
    if d is not None and d["choice"] == "drafter-off":
        return f"{TRIAL_DIR}/variant/evidence/swap-check.json"
    return served_source(stage_dir)


def _correctness_failures(report, run_dir) -> list[str]:
    """Why a swap check does not pass the checks every swap check must pass (stage 2's swap fields, and the prompt suite when `run_dir` is given)."""
    from orchard import stages          # imported here: orchard.stages imports this module
    draft = report.get("result_draft") if isinstance(report, dict) else None
    out = stages._swap_reasons(draft if isinstance(draft, dict) else {})
    if run_dir is not None and isinstance(report, dict):
        out += stages._suite_reasons(report, run_dir)
    return out


def variant_side(stage_dir, run_dir=None) -> dict | None:
    """The drafter-off variant's side, None when the trial left nothing. A trial result that says the variant did not serve gives a side that did not
    serve (a server that never started writes no swap check). Otherwise the side comes from the variant's own swap check, and a result that says it
    served with no swap check gives a side with no verdict and no speed.

    The variant is admitted only when it passes the checks any swap check must pass: stage 2's swap fields (coherent text, top-1 agreement at the bar,
    enough tokens, a ready time) and, when `run_dir` is given, the prompt suite (orchard/prompt_suite.py, read the way the stage gates read it). A
    variant that fails them is a side that did not serve, with the reasons in `failure` and no speeds, so a healthy sampling probe cannot select it."""
    sd = Path(stage_dir)
    # The supervisor's record of the trial's test comes first. A test that timed out or exited non-zero can still have left a swap check, and the
    # supervisor discards that variant, so this reader must too.
    marker_path = sd / TRIAL_DIR / "test-result.json"
    if marker_path.exists():
        marker = _read_json(marker_path)
        if not (isinstance(marker, dict) and marker.get("returncode") == 0 and not marker.get("timed_out")):
            return None
    trial = _read_json(sd / TRIAL_DIR / TRIAL_RESULT)
    v = trial.get("variant") if isinstance(trial, dict) else None
    from_report = side_from_report(_read_json(sd / TRIAL_DIR / "variant" / "evidence" / "swap-check.json"))
    if isinstance(v, dict) and v.get("served") is False:
        return {"drafter": "off", "verdict": "not-run", "greedy_tok_s": None, "sampled_tok_s": None, "served": False}
    if from_report is not None:
        failures = _correctness_failures(_read_json(sd / TRIAL_DIR / "variant" / "evidence" / "swap-check.json"), run_dir)
        if failures:
            return {"drafter": "off", "verdict": "not-run", "greedy_tok_s": None, "sampled_tok_s": None, "served": False,
                    "failure": "; ".join(failures)[:400]}
        return from_report
    if isinstance(v, dict) and v.get("served") is True:
        return {"drafter": "off", "verdict": "not-run", "greedy_tok_s": None, "sampled_tok_s": None, "served": True}
    return None


def decision_problems(stage_dir, run_dir) -> list[str]:
    """What is wrong with stage 2's sampling decision, for the stage 2 gate. No file is no problem (older runs, and runs where stage 2 is not a
    swap check). The decision is recomputed from the files on disk: the need from stage 0's evidence and the override the ledger recorded, the
    sides from the swap checks. A file whose choice or flag differs from the recomputation is refused."""
    sd = Path(stage_dir)
    path = sd / DECISION_FILE
    if not path.exists():
        return []
    d = _read_json(path)
    if not isinstance(d, dict):
        return [f"{DECISION_FILE} is not a JSON object"]
    if d.get("choice") not in CHOICES:
        return [f"{DECISION_FILE} choice must be one of {', '.join(CHOICES)}, got {d.get('choice')!r}"]
    if not isinstance(d.get("reason"), str) or not d["reason"].strip():
        return [f"{DECISION_FILE} needs a reason for the choice"]
    override = read_override(run_dir)
    if d.get("override") != override:
        return [f"{DECISION_FILE} records the sampling override {d.get('override')!r}; the run's ledger records {override!r}"]
    current = served_side(sd)
    if current is None:
        return [f"{DECISION_FILE} cannot be checked: the swap check it was made from is missing or unreadable"]
    variant = variant_side(sd, run_dir)
    note = d.get("variant_note") if isinstance(d.get("variant_note"), str) else None
    again = decide(need(run_dir, override), current, variant, variant_note=note)      # the note is free text; it only words the reason
    out = []
    # Everything the card and the bundle read is derived, so every derived field is compared, not only the choice and the flag.
    for key in ("needed", "need_source", "need_evidence", "thresholds", "reason", "variant_note"):
        if d.get(key) != again.get(key):
            out.append(f"{DECISION_FILE} has `{key}` {d.get(key)!r}, but the files on disk give {again.get(key)!r}")
    saved_numbers = d.get("numbers") if isinstance(d.get("numbers"), dict) else {}
    for key in sorted(set(saved_numbers) | set(again["numbers"])):
        if saved_numbers.get(key) != again["numbers"].get(key):
            out.append(f"{DECISION_FILE} has `numbers.{key}` {saved_numbers.get(key)!r}, but the files on disk give {again['numbers'].get(key)!r}")
    if again["choice"] != d["choice"]:
        out.append(f"{DECISION_FILE} chose {d['choice']!r}, but the files on disk give {again['choice']!r}: {again['reason']}")
    elif again["flag"] != d.get("flag"):
        out.append(f"{DECISION_FILE} has the flag {d.get('flag')!r}, but the files on disk give {again['flag']!r}: {again['reason']}")
    if d["choice"] == "drafter-off" and (variant is None or variant.get("served") is not True):
        out.append(f"{DECISION_FILE} chose the drafter-off variant, but {TRIAL_DIR}/variant/evidence/swap-check.json does not show it serving")
    return out
