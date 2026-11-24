# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Will this model fit on the chips I have, and at what precision? Answered from a checkpoint's safetensors headers alone.

    tt-orchard plan SNAPSHOT_DIR [--chips 4] [--json]

It reads only the headers of the `*.safetensors` files (tensor names and shapes, never a weight), so it can run before any chip is leased. It splits
the parameters into the routed experts of a mixture-of-experts model (tensors with `.experts.` in the name) and everything else, prices each part in
bf16, bfp8 and bfp4, and says for the chip count asked what each combination needs per chip, in two placements:

  replicated  every chip holds every weight (what the generated ports do today);
  sharded     the routed experts are split over the chips and everything else is replicated (not written yet: issue 723).

Bytes per element: bf16 2, bfp8 1.0625 and bfp4 0.5625 (one shared exponent byte per 16 values), which are the ratios of the probe 14 figures for
Flash-Next. A chip's usable DRAM is 93.4% of its 32 GiB (29.9 of 32, probe 14). Weights may take WEIGHT_SHARE of that; the rest is left for the KV
cache and activations. That share is an assumption and a flag changes it. The plan does not measure speed, and it does not measure quality: the notes
quote the one measured fidelity pair (scripts/fidelity_score.py) and say to score a candidate before choosing bfp4.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from orchard import defaults, safeio

BYTES_PER_ELEMENT = {"bf16": 2.0, "bfp8": 1.0625, "bfp4": 0.5625}
USABLE_FRACTION = 0.934            # probe 14: 29.9 GB usable of a 32 GB chip
WEIGHT_SHARE = 0.85                # choice: the share of usable DRAM the weights may take; the KV cache and activations need the rest
MAX_HEADER_BYTES = 64 << 20
MAX_FILES = 4096
# (name, format of everything else, format of the routed experts), best quality first
COMBINATIONS = (("all bf16", "bf16", "bf16"), ("all bfp8", "bfp8", "bfp8"), ("bfp8 + bfp4 experts", "bfp8", "bfp4"), ("all bfp4", "bfp4", "bfp4"))
QUALITY_NOTES = {
    "bfp8": "bfp8 weights scored 9.991 of 10 on 4000 sampled tokens of one 7B port (Olmo-Hybrid-7B): close to the reference.",
    "bfp4": "bfp4 weights scored 9.821 on the same port: a visible loss. Score a candidate with scripts/fidelity_score.py before choosing it.",
}
GIB = float(1 << 30)
# The rule for a format below bf16 (issue 731). bfp8 is the default: its measured score on the one pair we have is 9.991. bfp4 is recommended only
# when a score measured on the model's own port (scripts/fidelity_score.py, or the stage 2 fidelity block) is at or above FIDELITY_BAR on at least
# MIN_SCORED_TOKENS tokens. The bar is provisional: it sits between the two measured points (bfp8 9.991, bfp4 9.821 on one 7B port) and a loss of
# 1 percent of perplexity or less is what it allows. More calibration points should move it. A score below the bar removes the row.
FIDELITY_BAR = 9.9
MIN_SCORED_TOKENS = 4000


class FidelityInputError(ValueError):
    """A fidelity record, rule name or rule limit that is not usable. The message names the key and holds no input text."""


@dataclass(frozen=True)
class FidelityRule:
    """What a format below bf16 must show besides the perplexity ratio. The ratio bar (FIDELITY_BAR) always applies; a rule adds limits.
    `kl_max` is the largest mean KL(reference || candidate) allowed, `top1_min` the smallest argmax agreement, `abs_dppl_percent_max` the largest
    perplexity change in either direction, in percent. A limit that is None is not checked."""
    name: str
    kl_max: float | None = None
    top1_min: float | None = None
    abs_dppl_percent_max: float | None = None
    source: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "kl_max": self.kl_max, "top1_min": self.top1_min, "abs_dppl_percent_max": self.abs_dppl_percent_max,
                "source": self.source}


# The ratio bar alone reads one number, and a build can pass it with a large KL (the Lottolabs LFM2.5-24B-A2B card: perplexity +0.29 percent, KL 0.0458,
# top-1 0.915). `lfm3-card` is the rule the Lottolabs LFM3-1.7B card states for its own choice of precision. A caller can ask for it by name. It is never the
# default: the limits are one author's, measured on natural text for one family.
RULES = {
    "perplexity": FidelityRule("perplexity", source=f"the ratio bar FIDELITY_BAR {FIDELITY_BAR} and nothing else (issue 731); every caller before the rules existed"),
    "lfm3-card": FidelityRule("lfm3-card", kl_max=0.03, abs_dppl_percent_max=1.0,
                              source="Lottolabs LFM3-1.7B card: the cheapest allocation with KL at most 0.03 and |perplexity change| at most 1 percent"),
}
DEFAULT_RULE = RULES["perplexity"]
MAX_DPPL_PERCENT = 1e4                 # a perplexity change this large is a typing mistake


def rule_from(name: str | None, *, kl_max=None, top1_min=None) -> FidelityRule:
    """The named rule (None means the default) with the caller's own limits laid over it. Raises FidelityInputError for an unknown name or a limit that is
    not a finite number in range (`kl_max` above 0, `top1_min` above 0 and at most 1)."""
    base = RULES.get(name) if name is not None else DEFAULT_RULE
    if base is None:
        raise FidelityInputError(f"unknown fidelity rule {str(name)[:40]!r}; the known ones are {', '.join(RULES)}")
    try:
        if kl_max is not None:
            safeio.finite_number(kl_max, lo=0, what="the KL limit")
            if kl_max == 0:
                raise safeio.BadValue("the KL limit must be above 0")
        if top1_min is not None:
            safeio.finite_number(top1_min, lo=0, hi=1, what="the top-1 floor")
            if top1_min == 0:
                raise safeio.BadValue("the top-1 floor must be above 0")
    except safeio.BadValue as exc:
        raise FidelityInputError(exc.reason if hasattr(exc, "reason") else str(exc)) from exc
    if kl_max is None and top1_min is None:
        return base
    return FidelityRule(f"{base.name}+limits" if name is not None else "custom", kl_max if kl_max is not None else base.kl_max,
                        top1_min if top1_min is not None else base.top1_min, base.abs_dppl_percent_max,
                        f"{base.source}; with the caller's own limits" if name is not None else "the caller's own limits")


_RECORD_KEYS = ("score", "tokens", "kl", "top1", "dppl_percent")


def measures(record) -> dict:
    """One measured record as a dict of score, tokens and the optional kl, top1 and dppl_percent. The record is the pair `(score, tokens)` every
    caller used first, or a dict with those keys. Raises FidelityInputError, naming the key, for anything else."""
    if isinstance(record, (tuple, list)):
        if len(record) != 2:
            raise FidelityInputError("a fidelity record is (score, tokens) or an object with score and tokens")
        record = {"score": record[0], "tokens": record[1]}
    if not isinstance(record, dict):
        raise FidelityInputError("a fidelity record is (score, tokens) or an object with score and tokens")
    for k in record:
        if k not in _RECORD_KEYS:
            raise FidelityInputError(f"a fidelity record has an unknown key {str(k)[:40]!r} (known: {', '.join(_RECORD_KEYS)})")
    for k in ("score", "tokens"):
        if k not in record:
            raise FidelityInputError(f"a fidelity record needs {k}")
    try:
        out = {"score": safeio.finite_number(record["score"], lo=0, hi=10.5, what="fidelity score"),
               "tokens": safeio.whole_number(record["tokens"], lo=0, what="fidelity tokens")}
        for k, lo, hi in (("kl", 0, None), ("top1", 0, 1), ("dppl_percent", -MAX_DPPL_PERCENT, MAX_DPPL_PERCENT)):
            out[k] = None if record.get(k) is None and k not in record else safeio.finite_number(record[k], lo=lo, hi=hi, what=f"fidelity {k}")
    except safeio.BadValue as exc:
        raise FidelityInputError(exc.reason if hasattr(exc, "reason") else str(exc)) from exc
    return out


def _dppl_percent(m: dict) -> float:
    """The perplexity change in percent. The record's own figure when it has one, else from the ratio: score = 10 x ppl_reference / ppl_candidate,
    so the candidate's perplexity is 10 / score of the reference's."""
    if m["dppl_percent"] is not None:
        return float(m["dppl_percent"])   # finite: ok `measures` already passed it through safeio.finite_number with a bound of 1e4
    return 100.0 * (10.0 / m["score"] - 1.0) if m["score"] > 0 else float("inf")   # finite: ok a literal


def gate_detail(fmt: str, scores: dict | None, rule: FidelityRule | None = None) -> tuple[str, str | None]:
    """(gate, reason) for one weight format: "ok", "unscored", "too-few-tokens" or "below-bar", and a sentence for any gate but "ok". bf16 and bfp8
    are not gated. For a lower format the order is: no record, too few tokens, the ratio bar, a measure the rule needs and the record lacks, then the
    rule's own limits."""
    rule = rule or DEFAULT_RULE
    if fmt in ("bf16", "bfp8"):
        return "ok", None
    got = (scores or {}).get(fmt)
    if got is None:
        return "unscored", f"no fidelity score for {fmt}"
    m = measures(got)
    if m["tokens"] < MIN_SCORED_TOKENS:
        return "too-few-tokens", f"{m['tokens']} scored tokens, fewer than {MIN_SCORED_TOKENS}"
    if m["score"] < FIDELITY_BAR:
        return "below-bar", f"score {m['score']} is below the bar {FIDELITY_BAR}"
    needs = [(rule.kl_max, "kl", "a KL (--fidelity-kl)"), (rule.top1_min, "top1", "a top-1 agreement (--fidelity-top1)")]
    missing = [what for limit, key, what in needs if limit is not None and m[key] is None]
    if missing:
        return "unscored", f"the {rule.name} rule needs {' and '.join(missing)} and the record has none"
    if rule.kl_max is not None and m["kl"] > rule.kl_max:
        return "below-bar", f"KL {m['kl']} is above the {rule.kl_max} the {rule.name} rule allows"
    if rule.top1_min is not None and m["top1"] < rule.top1_min:
        return "below-bar", f"top-1 agreement {m['top1']} is below the {rule.top1_min} the {rule.name} rule requires"
    if rule.abs_dppl_percent_max is not None:
        d = _dppl_percent(m)
        if abs(d) > rule.abs_dppl_percent_max:
            return "below-bar", f"perplexity change {d:+.2f} percent is outside the {rule.abs_dppl_percent_max} percent the {rule.name} rule allows"
    return "ok", None


def is_routed_expert(name: str) -> bool:
    return ".experts." in name          # a shared expert is named `shared_expert` or `shared_experts`, so it does not match


def inventory(snapshot: Path) -> dict:
    """Element counts split into routed experts and the rest, from the headers of the snapshot's safetensors files."""
    files = sorted(p for p in snapshot.iterdir() if p.name.endswith(".safetensors"))[:MAX_FILES]
    experts = other = tensors = 0
    unreadable = []
    for f in files:
        target = safeio.resolved_inside(f, snapshot)
        if target is None:
            unreadable.append(f"{f.name}: a link that leaves the snapshot")
            continue
        try:
            header = safeio.safetensors_header(target, max_header_bytes=MAX_HEADER_BYTES)
        except safeio.SafeReadError as exc:
            unreadable.append(f"{f.name}: {exc.reason}")
            continue
        for name, e in header.entries.items():
            n = 1
            for d in e.shape:
                n *= d
            tensors += 1
            if is_routed_expert(name):
                experts += n
            else:
                other += n
    return {"files": len(files), "tensors": tensors, "routed_expert_params": experts, "other_params": other, "unreadable": unreadable}


def budget_gib(chip_gib: float, share: float) -> float:
    return chip_gib * USABLE_FRACTION * share


def format_gate(fmt: str, scores: dict | None, rule: FidelityRule | None = None) -> str:
    """"ok", "unscored", "too-few-tokens" or "below-bar" for one weight format, from the measured records {format: (score, tokens) or a dict}
    (see `gate_detail` for the order of the checks and the reason)."""
    return gate_detail(fmt, scores, rule)[0]


def row_gate(other_fmt: str, expert_fmt: str, scores: dict | None, rule: FidelityRule | None = None) -> tuple[str, str | None]:
    """The worse of the two formats' gates and that gate's reason: for a row with one format for everything else and one for the experts."""
    a, b = gate_detail(other_fmt, scores, rule), gate_detail(expert_fmt, scores, rule)
    gate = _worst(a[0], b[0])
    return gate, next((why for g, why in (a, b) if g == gate and why), None)


def _worst(*gates: str) -> str:
    for g in ("below-bar", "too-few-tokens", "unscored"):
        if g in gates:
            return g
    return "ok"


def rows(inv: dict, chips: int, chip_gib: float, share: float, scores: dict | None = None, rule: FidelityRule | None = None) -> list[dict]:
    budget = budget_gib(chip_gib, share)
    out = []
    for name, other_fmt, expert_fmt in COMBINATIONS:
        if not inv["routed_expert_params"] and other_fmt != expert_fmt:
            continue                                  # a dense model has no experts to price differently
        o = inv["other_params"] * BYTES_PER_ELEMENT[other_fmt] / GIB
        e = inv["routed_expert_params"] * BYTES_PER_ELEMENT[expert_fmt] / GIB
        sharded = lambda n: o + e / n
        gate, why = row_gate(other_fmt, expert_fmt, scores, rule)
        out.append({"name": name, "other_format": other_fmt, "expert_format": expert_fmt, "total_gib": round(o + e, 2),
                    "format_gate": gate, "format_gate_reason": why,
                    "replicated_per_chip_gib": round(o + e, 2), "fits_replicated": o + e <= budget,
                    "sharded_per_chip_gib": round(sharded(chips), 2), "fits_sharded": sharded(chips) <= budget,
                    "min_chips_sharded": next((n for n in range(1, 65) if sharded(n) <= budget), None)})
    return out


def recommend(table: list[dict]) -> dict | None:
    """The best-quality row that fits and whose formats pass the rule (`format_gate`), replicated first; a row that needs sharded experts says so.
    When no row passes, the best fitting row that is only unscored is returned with its gate, so the plan can say what to measure; a row below the
    bar is never recommended."""
    sharded_note = "the routed experts split over the chips, which the generator does not write yet (issue 723)"
    for wanted in (("ok",), ("unscored", "too-few-tokens")):
        for placement, key in (("replicated", "fits_replicated"), ("sharded", "fits_sharded")):
            for r in table:
                if r[key] and r["format_gate"] in wanted:
                    return {"row": r["name"], "placement": placement, "gate": r["format_gate"],
                            "needs": sharded_note if placement == "sharded" else None}
    return None


def plan(snapshot: Path, chips: int, chip_gib: float = defaults.CHIP_DRAM_GIB, share: float = WEIGHT_SHARE, scores: dict | None = None,
         rule: FidelityRule | None = None) -> dict:
    rule = rule or DEFAULT_RULE
    inv = inventory(snapshot)
    table = rows(inv, chips, chip_gib, share, scores, rule)
    rec = recommend(table)
    notes = [QUALITY_NOTES[f] for f in ("bfp8", "bfp4") if rec and f in (next(r for r in table if r["name"] == rec["row"])["other_format"],
                                                                         next(r for r in table if r["name"] == rec["row"])["expert_format"])]
    if rec is None and table:
        below = [r["name"] for r in table if r["format_gate"] == "below-bar" and (r["fits_replicated"] or r["fits_sharded"])]
        notes.append("nothing that passes the fidelity rule fits at this chip count" + (f" (rows that fit but score below {FIDELITY_BAR}: {', '.join(below)})" if below
                     else " even with the experts split") + "; add chips, choose a smaller model or a better-scoring format")
        notes += [f"{r['name']}: {r['format_gate_reason']}" for r in table if r["format_gate"] == "below-bar" and r["format_gate_reason"]
                  and (r["fits_replicated"] or r["fits_sharded"]) and rule is not DEFAULT_RULE]
    if rec and rec["gate"] != "ok":
        notes.append(f"this row uses bfp4 with no usable score ({rec['gate']}): score the port with scripts/fidelity_score.py on at least {MIN_SCORED_TOKENS} tokens "
                     f"and pass the result with --fidelity-bfp4 SCORE --fidelity-tokens N; the bar is {FIDELITY_BAR}")
    return {"snapshot": snapshot.name, **inv, "chips": chips, "chip_gib": chip_gib, "weight_budget_gib_per_chip": round(budget_gib(chip_gib, share), 2),
            "weight_share": share, "fidelity_rule": rule.as_dict(), "rows": table, "recommendation": rec, "notes": notes}


def render(res: dict) -> str:
    lines = [f"{res['snapshot']}: {res['tensors']} tensors in {res['files']} files, {res['routed_expert_params'] / 1e9:.2f} B routed-expert and "
             f"{res['other_params'] / 1e9:.2f} B other parameters",
             f"{res['chips']} chips, {res['weight_budget_gib_per_chip']} GiB per chip for weights ({res['weight_share']:.0%} of usable DRAM)", ""]
    rule = res.get("fidelity_rule") or {}
    if rule.get("name") not in (None, DEFAULT_RULE.name):
        lines.append(f"fidelity rule: {rule['name']} (ratio bar {FIDELITY_BAR}" + "".join(
            f"; {label} {rule[key]}" for key, label in (("kl_max", "KL at most"), ("top1_min", "top-1 at least"), ("abs_dppl_percent_max", "|perplexity change| percent at most"))
            if rule.get(key) is not None) + ")")
        lines.append("")
    lines.append(f"{'combination':22} {'total GiB':>10} {'replicated':>11} {'experts split':>14}   {'min chips if split':<19}fidelity rule")
    for r in res["rows"]:
        lines.append(f"{r['name']:22} {r['total_gib']:>10} {'fits' if r['fits_replicated'] else 'no':>11} "
                     f"{'fits' if r['fits_sharded'] else 'no':>14}   {str(r['min_chips_sharded'] if r['min_chips_sharded'] else 'more than 64'):<19}{r['format_gate']}")
    rec = res["recommendation"]
    lines.append("")
    lines.append(f"best fit: {rec['row']}, {rec['placement']}" + (f"; needs {rec['needs']}" if rec["needs"] else "") if rec else "best fit: none at this chip count")
    lines += [f"note: {n}" for n in res["notes"]] + [f"unreadable: {u}" for u in res["unreadable"]]
    return "\n".join(lines)


def add_fidelity_arguments(p: argparse.ArgumentParser, bar: float | None = None) -> None:
    """The fidelity options, shared by `plan` and `tier-plan`. The measured numbers all describe one run of the bfp4 port, and
    scripts/fidelity_score.py prints them as one line to paste."""
    bar = FIDELITY_BAR if bar is None else bar
    p.add_argument("--fidelity-bfp4", type=float, help=f"a fidelity score measured for this model's port with bfp4 weights (the bar is {bar})")
    p.add_argument("--fidelity-tokens", type=int, default=0, help=f"scored tokens behind --fidelity-bfp4 (at least {MIN_SCORED_TOKENS} to count)")
    p.add_argument("--fidelity-kl", type=float, help="mean KL(reference || port) from the same run as --fidelity-bfp4 (scripts/fidelity_score.py prints it as kl_full)")
    p.add_argument("--fidelity-top1", type=float, help="argmax agreement with the reference from the same run, 0 to 1 (top1_agree in scripts/fidelity_score.py)")
    p.add_argument("--fidelity-dppl-percent", type=float,
                   help="perplexity change in percent, when it was measured on other text; without it the change is worked out from --fidelity-bfp4")
    p.add_argument("--fidelity-rule", choices=sorted(RULES), help="the rule a bfp4 score must pass (default perplexity: the bar alone); lfm3-card also needs "
                   "KL at most 0.03 and a perplexity change within 1 percent")
    p.add_argument("--fidelity-kl-max", type=float, help="the largest KL a bfp4 port may show; it adds to the chosen rule")
    p.add_argument("--fidelity-top1-min", type=float, help="the smallest top-1 agreement a bfp4 port may show, above 0 and at most 1; it adds to the chosen rule")


_FIDELITY_FLAGS = (("fidelity_bfp4", "--fidelity-bfp4"), ("fidelity_tokens", "--fidelity-tokens"), ("fidelity_kl", "--fidelity-kl"),
                   ("fidelity_top1", "--fidelity-top1"), ("fidelity_dppl_percent", "--fidelity-dppl-percent"), ("fidelity_rule", "--fidelity-rule"),
                   ("fidelity_kl_max", "--fidelity-kl-max"), ("fidelity_top1_min", "--fidelity-top1-min"))


def fidelity_argv(args) -> list[str]:
    """The fidelity options of parsed arguments as an argument list, for a command that hands them on to `main`. A zero token count is the default and is left out."""
    out: list[str] = []
    for attr, flag in _FIDELITY_FLAGS:
        v = getattr(args, attr, None)
        if v is not None and not (attr == "fidelity_tokens" and v == 0):
            out += [flag, str(v)]
    return out


def fidelity_inputs(args) -> tuple[dict | None, FidelityRule]:
    """(measured records, rule) from parsed arguments. Raises FidelityInputError. A KL, top-1 or perplexity change with no score is refused: each
    describes the run that the score came from."""
    extras = {k: getattr(args, a, None) for k, a in (("kl", "fidelity_kl"), ("top1", "fidelity_top1"), ("dppl_percent", "fidelity_dppl_percent"))}
    score = getattr(args, "fidelity_bfp4", None)
    scores = None
    if score is not None:
        record = {"score": score, "tokens": getattr(args, "fidelity_tokens", 0), **{k: v for k, v in extras.items() if v is not None}}
        measures(record)                  # refuse a bad number here, with the option's own message, before any plan is built
        scores = {"bfp4": record}
    elif any(v is not None for v in extras.values()):
        raise FidelityInputError("--fidelity-kl, --fidelity-top1 and --fidelity-dppl-percent describe the run behind --fidelity-bfp4, which is missing")
    return scores, rule_from(getattr(args, "fidelity_rule", None), kl_max=getattr(args, "fidelity_kl_max", None),
                             top1_min=getattr(args, "fidelity_top1_min", None))


def parser(add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tt-orchard plan", add_help=add_help,
                                description="say what precision and placement a checkpoint needs on N chips, from its safetensors headers alone")
    p.add_argument("snapshot", help="a directory of *.safetensors files (a Hugging Face snapshot)")
    p.add_argument("--chips", type=int, default=4, help="chips to plan for (default 4)")
    p.add_argument("--chip-gib", type=float, default=defaults.CHIP_DRAM_GIB, help="DRAM per chip in GiB (default %(default)s)")
    p.add_argument("--weight-share", type=float, default=WEIGHT_SHARE, help="share of usable DRAM the weights may take (default %(default)s)")
    add_fidelity_arguments(p)
    p.add_argument("--json", action="store_true", help="print JSON")
    return p


def main(argv=None, out=None) -> int:
    args = parser().parse_args(argv)
    out = out or sys.stdout
    snap = Path(args.snapshot)
    if not snap.is_dir():
        print(f"{args.snapshot} is not a directory", file=sys.stderr)
        return 2
    if not 1 <= args.chips <= 64 or not 0 < args.weight_share <= 1 or not 0 < args.chip_gib <= 4096:
        print("--chips must be 1 to 64, --weight-share in (0, 1] and --chip-gib positive", file=sys.stderr)
        return 2
    try:
        scores, rule = fidelity_inputs(args)
    except FidelityInputError as exc:
        print(f"fidelity input: {exc}", file=sys.stderr)
        return 2
    res = plan(snap, args.chips, args.chip_gib, args.weight_share, scores, rule)
    if not res["tensors"]:
        print("no readable safetensors tensors in the snapshot" + "".join(f"; {u}" for u in res["unreadable"]), file=sys.stderr)
        return 2
    print(json.dumps(res, indent=2) if args.json else render(res), file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
