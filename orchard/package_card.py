"""The package card and the license it carries (plan 5, stage 7).

This module owns the README.md of a staged package: what it says, and the check that it says only
what the run can back. `render_card` writes it from `CardFacts`; `card_problems` reads a card back
and lists what is wrong. The gate (orchard/stages.py, gate_package) runs the check on the file on
disk, so a card edited after it was rendered is checked as it is.

The license comes from the new model's own Hugging Face card (`read_license`, the `license:` key
in the snapshot's README.md front matter). A license this module cannot name, and `other`, count
as non-commercial: the card then says "Non-commercial use only." and the check refuses wording
that implies commercial use. Altworld/Hemmingway-1 is CC BY-NC 4.0, and the base model's Apache
2.0 license does not carry over to it.

Numbers appear only in the card's Numbers table. Each row is labelled `measured` or `TODO`. A
measured row names its evidence files, relative to the run directory; the first one is the result
file the number was read from, and the check requires the value to appear in it. A number with a
performance unit anywhere else in the card is refused, because nothing ties it to evidence.

What the check does not do: judge whether the prose is true, or catch a commercial claim worded in
a way the patterns below do not list.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from orchard.stages import inside

LICENSES = {"cc-by-nc-4.0": ("CC BY-NC 4.0", "https://creativecommons.org/licenses/by-nc/4.0/"),
            "cc-by-nc-sa-4.0": ("CC BY-NC-SA 4.0", "https://creativecommons.org/licenses/by-nc-sa/4.0/"),
            "cc-by-nc-nd-4.0": ("CC BY-NC-ND 4.0", "https://creativecommons.org/licenses/by-nc-nd/4.0/"),
            "cc-by-4.0": ("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/"),
            "apache-2.0": ("Apache 2.0", "https://www.apache.org/licenses/LICENSE-2.0"),
            "mit": ("MIT", "https://opensource.org/licenses/MIT")}
PERMISSIVE = frozenset({"apache-2.0", "mit", "cc-by-4.0"})
NC_LINE = "Non-commercial use only."
# Wording that implies commercial use. Checked case-insensitively on cards for non-commercial models.
COMMERCIAL_CLAIMS = (r"(?<!non-)(?<!non )commercial (?:use|deployments?|purposes?|products?|"
                     r"applications?)(?! is not permitted)",
                     r"\bcommercially\b", r"production[- ]ready", r"\bfor (?:your )?customers\b",
                     r"\benterprise\b", r"\bresell", r"\bmonetiz")
PERF_IN_PROSE = re.compile(r"\b\d[\d.,]*\s*(?:tok/s|tokens/s|t/s|tokens per second|ms\b|%)")
NUMBERS_HEAD = "## Expected performance"
CARD_TAGS = ("tt-model-cache", "vllm", "thin")       # the tags tt-model's own push adds to a v6 bundle


def read_license(snapshot) -> str | None:
    """The `license:` value of the front matter in `<snapshot>/README.md`, lowercased, or None."""
    try:
        text = (Path(snapshot) / "README.md").read_text(encoding="utf-8")
    except OSError:
        return None
    if not text.startswith("---\n"):
        return None
    front = text[4:].split("\n---", 1)[0]
    m = re.search(r"^license:\s*['\"]?([A-Za-z0-9.+-]+)['\"]?\s*$", front, re.M)
    return m.group(1).lower() if m else None


def non_commercial(license_id: str) -> bool:
    """True for any CC NC license and for anything this module does not know to be permissive."""
    return license_id not in PERMISSIVE


@dataclass(frozen=True)
class Number:
    name: str
    value: float | int | None
    unit: str
    label: str                       # "measured" or "TODO"
    evidence: tuple[str, ...]        # run-relative; the first is the result file holding the value


@dataclass(frozen=True)
class CardFacts:
    name: str                        # the bundle name, also the repo name the publish line uses
    namespace: str                   # the operator's Hugging Face namespace
    model_id: str
    revision: str
    nearest_model: str
    source_name: str                 # the nearest model's bundle this package was built from
    license_id: str
    chips: int
    mesh: str
    arch: str
    max_model_len: int
    max_num_seqs: int
    drafter: str | None
    verified: bool                   # the stage 7 boot check passed for this profile
    numbers: tuple[Number, ...]
    not_measured: tuple[str, ...]
    drafter_off: bool = False        # served without the speculative drafter (no mtp.* tensors)
    host_sampling: bool = False      # tokens are sampled on the host
    sidecars: tuple[str, ...] = ()   # sidecar files the model's repository ships and the bundle does not serve


def _value(n: Number) -> str:
    return "TODO" if n.label != "measured" else f"{json.dumps(n.value)} {n.unit}"


def render_card(f: CardFacts) -> str:
    lic_name, lic_link = LICENSES.get(f.license_id, (f.license_id, None))
    lic = f"{lic_name} ({lic_link})" if lic_link else lic_name
    tags = [*CARD_TAGS[:1], f.arch, *CARD_TAGS[1:], f.mesh.lower()]
    lines = ["---", f"license: {f.license_id}", f"base_model: {f.model_id}",
             "pipeline_tag: text-generation", "tags:", *[f"- {t}" for t in tags], "---", "",
             f"# {f.name}", "",
             f"A tt-model v6 thin bundle that serves {f.model_id} (revision `{f.revision}`) on "
             f"{f.chips} Tenstorrent {f.arch} chips (mesh {f.mesh}). The tt-orchard harness staged it "
             "from a run of that model.", "",
             "## License", ""]
    if non_commercial(f.license_id):
        lines.append(f"The weights this bundle points to are licensed {lic}. {NC_LINE} "
                     "Commercial use is not permitted by that license.")
    else:
        lines.append(f"The weights this bundle points to are licensed {lic}.")
    lines += ["The bundle ships no weights. `tt-model` downloads them from "
              f"{f.model_id} with your own Hugging Face account.", "",
              "## What it runs", "",
              f"- Weights: {f.model_id} at revision `{f.revision}`.",
              f"- Model code: the Tenstorrent implementation of {f.nearest_model}, taken from the "
              f"bundle {f.source_name}. {f.model_id} has the same text architecture, so only the "
              "weights differ.",
              f"- At each start the launcher builds `model-dir/` from the configuration files of "
              f"{f.nearest_model} (shipped in `base_config/`) and the tokenizer and weights of "
              f"{f.model_id}. It sets MODEL_WEIGHTS_DIR and HF_MODEL to that directory, so the chips "
              f"load the weights of {f.model_id}.",
              f"- Context length {f.max_model_len} tokens; up to {f.max_num_seqs} sequences at a time."]
    if f.drafter:
        lines.append(f"- Speculative decoding uses the drafter {f.drafter}, which was trained on "
                     f"{f.nearest_model}. Its acceptance rate on this model is listed under Not "
                     "measured.")
    if f.drafter_off:
        lines.append(f"- Speculative decoding is off. The drafter of {f.nearest_model} needs the MTP "
                     f"tensors of the model, and {f.model_id} has none.")
    if f.host_sampling:
        lines.append("- Sampling runs on the host, because on-device sampling is not used with the "
                     "drafter off on this mesh.")
    if f.sidecars:
        lines += ["", "## Not served", "",
                  f"{f.model_id} ships files this bundle does not load: "
                  + ", ".join(f"`{s}`" for s in f.sidecars) + ". The bundle serves the language-model "
                  "backbone only, so the sidecar head is not served. The run checked the head on the "
                  "host against the CPU reference; that check does not make it part of this bundle."]
    lines += ["", "## Intended use", "",
              f"Text generation with {f.model_id} through the OpenAI-compatible server that "
              "`tt-model serve` starts."]
    if f.sidecars:
        lines.append("Out of scope: the sidecar head, which this bundle does not serve.")
    lines += ["", NUMBERS_HEAD, "",
              "Every number is labelled. A `measured` number names its evidence files; the first file "
              "holds the value. The files are in this repository under `evidence/`, at the paths "
              "shown, with the run's directory, the home directory and the host name replaced by "
              "`<RUN_DIR>`, `<HOME>` and `<HOST>`. `TODO` means not measured.", "",
              "| Number | Value | Label | Evidence |", "|---|---|---|---|"]
    for n in f.numbers:
        ev = ", ".join(f"`{e}`" for e in n.evidence) if n.label == "measured" else "-"
        lines.append(f"| {n.name} | {_value(n)} | {n.label} | {ev} |")
    limits = [f"Tested on {f.chips} chips (mesh {f.mesh}) only; other configurations are not claimed "
              "by this bundle.",
              "Accuracy was compared with the CPU reference on one short fixed prompt only."]
    if f.drafter_off:
        limits.append("Speculative decoding is off, so decode speed is that of plain decoding.")
    if f.host_sampling:
        limits.append("Sampling runs on the host.")
    limits += [f"`{s}` is not served." for s in f.sidecars]
    lines += ["", "## Limitations", "", *[f"- {item}" for item in limits], "",
              "## Risks and safety considerations", "",
              f"The check above does not cover long outputs, tool calling or safety behaviour. "
              f"Output can differ from {f.model_id} run on a CPU or GPU in ways it does not show.", "",
              "## Not measured", "", *[f"- {item}" for item in f.not_measured], "",
              "## Boot check", "",
              ("Stage 7 of the run installed this bundle, served it on a leased board and compared "
               "its tokens with the CPU reference. The Expected performance table has the result."
               if f.verified else
               "This profile was not booted on hardware by the run. Boot it before publishing."),
              "", "## How to serve", "", f"    tt-model serve {f.namespace}/{f.name}", ""]
    return "\n".join(lines)


def card_evidence_paths(card: str) -> list[str]:
    """Every file the Numbers table cites for a `measured` number, in order, without repeats."""
    section, _ = _numbers_section(card)
    seen: list[str] = []
    for row in (r for r in section.splitlines() if r.startswith("| ") and not r.startswith("| Number ")):
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[2] == "measured":
            for p in cells[3].split(","):
                p = p.strip().strip("`")
                if p and p != "-" and p not in seen:
                    seen.append(p)
    return seen


def _front(card: str) -> str:
    return card[4:].split("\n---", 1)[0] if card.startswith("---\n") else ""


def _numbers_section(card: str) -> tuple[str, str]:
    """(the Numbers section, the rest of the card)."""
    if NUMBERS_HEAD not in card:
        return "", card
    before, after = card.split(NUMBERS_HEAD, 1)
    section, sep, rest = after.partition("\n## ")
    return section, before + (sep + rest if sep else "")


def card_problems(card: str, *, license_id: str, run_dir, sidecars=()) -> list[str]:
    problems = []
    for s in sidecars:
        if s not in card or "not served" not in card.lower():
            problems.append(f"the card must name the sidecar {s} and say it is not served")
    m = re.search(r"^license:\s*(\S+)\s*$", _front(card), re.M)
    if m is None:
        problems.append("the card's front matter has no license")
    elif m.group(1) != license_id:
        problems.append(f"the card's license is {m.group(1)}; the model's license is {license_id}")
    if non_commercial(license_id):
        if NC_LINE not in card:
            problems.append(f"the card for a {license_id} model must say {NC_LINE!r}")
        name = LICENSES.get(license_id, (license_id, None))[0]
        if name not in card:
            problems.append(f"the card must name the license {name!r}")
        for pattern in COMMERCIAL_CLAIMS:
            hit = re.search(pattern, card, re.I)
            if hit:
                problems.append(f"the card implies commercial use of a {license_id} model: "
                                f"{hit.group(0)!r}")
    for head in ("## Intended use", "## Limitations", "## Risks and safety considerations"):
        if head not in card:
            problems.append(f"the card has no {head!r} section")
    section, rest = _numbers_section(card)
    if not section:
        problems.append("the card has no Numbers section")
    for row in (r for r in section.splitlines() if r.startswith("| ") and not r.startswith("| Number ")):
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) != 4:
            problems.append(f"Numbers row {row!r} needs four cells")
            continue
        name, value, label, evidence = cells
        if label == "TODO":
            if value != "TODO":
                problems.append(f"number {name!r} is TODO and shows a value")
            continue
        if label != "measured":
            problems.append(f"number {name!r} must be labelled measured or TODO")
            continue
        shown = value.split(" ", 1)[0]
        try:
            float(shown)
        except ValueError:
            problems.append(f"number {name!r} is labelled measured and has no numeric value")
            continue
        paths = [p.strip().strip("`") for p in evidence.split(",") if p.strip() not in ("", "-")]
        if not paths:
            problems.append(f"number {name!r} is labelled measured and names no evidence")
            continue
        found = [inside(run_dir, p) for p in paths]
        for p, f in zip(paths, found):
            if f is None:
                problems.append(f"number {name!r}: evidence {p} is not a file inside the run directory")
        if found[0] is not None and shown not in found[0].read_text(encoding="utf-8", errors="replace"):
            problems.append(f"number {name!r}: the value {shown} is not in {paths[0]}")
    for hit in PERF_IN_PROSE.finditer(rest):
        problems.append(f"a number outside the Numbers table: {hit.group(0)!r}")
    return problems
