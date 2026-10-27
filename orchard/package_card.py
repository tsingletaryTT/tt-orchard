# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
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

from orchard.card_use import PERF_IN_PROSE, UNTESTED_SENTENCE, UseFacts, has_facts
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
NUMBERS_HEAD = "## Expected performance"
REPORTED_HEAD = "## Reported by the authors"
NOT_MEASURED_SENTENCE = "were not measured here"
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
class Reported:
    """A score the model's authors report. The run did not measure it."""
    name: str                        # the benchmark, as the authors name it
    value: str                       # as the authors wrote it, for example "81.8" or "93.8%"
    note: str = ""                   # the protocol or metric, if the authors state one


def _cell(text: str) -> str:
    """One table cell: a pipe or a line break in the text cannot break the row."""
    return " ".join(str(text).replace("|", "/").split())


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
    base_files: tuple[str, ...] = () # files the package ships in base_config/, taken from the nearest model
    sampling_from_base: bool = False # the model ships no generation_config.json, so the nearest model's is used
    reported: tuple[Reported, ...] = ()   # scores the authors report, shown apart from the measured numbers
    reported_source: str = ""        # where the authors report them (a card or paper), for the sentence above them
    graft: dict | None = None        # graft.json: the nearest model's mtp.* tensors are added at each start
    stage_2_without_graft: bool = False   # stage 2 and 4 served without the graft (the drafter off, host sampling)
    graft_decision: str | None = None     # "graft" or "safe" when stage 2's graft trial ran and decided (orchard/graft_trial.py)
    use: UseFacts | None = None           # what the model is for and how to call it, read from its own files (orchard/card_use.py)
    acceptance_measured: bool = False     # the Numbers table has the drafter's acceptance rate (stage 2's graft trial measured it)
    vocab: tuple[int, int] | None = None  # (this model's vocabulary size, the nearest model's) when the model is a vocabulary trim
    sampling: dict | None = None     # what stage 2's sampling probe found (package.sampling_summary): card_sentence, serving; only the required profile


def _count(n: int) -> str:
    return f"{n:,}"


def _value(n: Number) -> str:
    return "TODO" if n.label != "measured" else f"{json.dumps(n.value)} {n.unit}"


def _base_files(f: CardFacts) -> str:
    """The names of the files taken from the nearest model, each followed by ", ", or nothing."""
    return "".join(f"`{n}`, " for n in f.base_files)


CUTOFF_NAME = "replies cut off at the token limit"
CUTOFF_SENTENCE = ("A reply that reaches `max_tokens` while the model is still thinking has no answer in `content`. "
                   "Set `max_tokens` high. The Numbers table counts the replies the benchmark runs cut off at the limit.")


def _list(values) -> str:
    return ", ".join(f"`{v}`" for v in values)


def cutoffs_measured(numbers) -> bool:
    """The run measured a count of replies cut off at the token limit and it is above zero."""
    return any(CUTOFF_NAME in n.name and n.label == "measured" and isinstance(n.value, (int, float)) and n.value > 0
               for n in numbers)


def intended_use_lines(f: CardFacts) -> list[str]:
    """The Intended use section after its heading: the authors' words, then what the package does with the model's
    chat template and run.sh flags. A fact taken from those files and not tried on the served model says so."""
    u = f.use
    lines: list[str] = []
    if u.authors_use:
        text = " ".join(u.authors_use.split())
        lines += [f"The authors of {f.model_id} describe the intended use this way. These are their words, from the model "
                  "card. They were not measured here:", "", f"> {text}", ""]
    lines.append(f"This package serves {f.model_id} on {f.chips} chips through the OpenAI-compatible chat endpoint. "
                 "How the model behaves there decides how to call it:")
    lines.append("")
    untested = f"({UNTESTED_SENTENCE})"
    if u.thinking_default is True:
        think = ("Thinking is on by default: the chat template makes the model think before it answers unless a request "
                 "turns thinking off.")
        if u.reasoning_parser:
            think += " The server returns the thinking in `reasoning_content` and the answer in `content`."
        lines.append(f"- {think} {untested}")
    elif u.thinking_default is False:
        lines.append("- Thinking is off by default: the chat template leaves it off unless a request turns it on. "
                     f"{untested}")
    if u.effort_values:
        default = f" (default `{u.effort_default}`)" if u.effort_default else ""
        lines.append(f"- Reasoning effort can be set per request with `chat_template_kwargs` and `reasoning_effort`. "
                     f"The template accepts {_list(u.effort_values)}{default}. {untested}")
    if u.thinking_default is True and u.can_disable_thinking:
        lines.append('- Thinking can be turned off per request with `chat_template_kwargs` set to '
                     f'`{{"enable_thinking": false}}`. {untested}')
    if u.tools_ok:
        if u.tool_evidence:
            path, names = u.tool_evidence
            lines.append(f"- Tool calls come back in `tool_calls`. In the runs recorded in `{path}` the tool-use prompts "
                         f"returned tool calls ({_list(names)}).")
        else:
            lines.append(f"- The server parses tool calls into `tool_calls` (parser `{u.tool_parser}`). {untested}")
    if u.text_only:
        lines.append("- Text only: run.sh skips the vision tower, so the package does not take images or video, whatever "
                     f"the model accepts upstream. {untested}")
    if u.thinking_default is True and u.reasoning_parser and cutoffs_measured(f.numbers):
        lines.append(f"- {CUTOFF_SENTENCE}")
    return lines


EXAMPLE_TOOL = ('{"type": "function", "function": {"name": "run_tests", "description": "Run pytest on a path", '
                '"parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}')


def how_to_call_lines(f: CardFacts) -> list[str]:
    """The How to call it section after its heading: where the server listens, how to read the model name, and one
    request. The request carries a generic tool only when the server can return tool calls."""
    u = f.use
    tools = u.tools_ok
    prompt = "Run the tests under tests/utils and tell me which fail." if tools else \
        "Explain the difference between a mutex and a semaphore."
    body = ['      "model": "\'"$MODEL"\'",',
            f'      "messages": [{{"role": "user", "content": "{prompt}"}}],']
    if tools:
        body.append(f'      "tools": [{EXAMPLE_TOOL}],')
    body.append(f'      "max_tokens": {32768 if u.thinking_default else 1024}')
    return ["`tt-model serve` starts an OpenAI-compatible server. Its port is vLLM's default, 8000, unless you pass "
            "`--port` after the bundle id. The server's model name is the path of the package's model directory, so "
            "read it from `/v1/models` first:", "",
            "    MODEL=$(curl -s http://localhost:8000/v1/models | python3 -c "
            "'import sys, json; print(json.load(sys.stdin)[\"data\"][0][\"id\"])')",
            "    curl http://localhost:8000/v1/chat/completions -H 'content-type: application/json' -d '{",
            *body, "    }'", ""]


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
              f"{f.nearest_model} ({_base_files(f)}shipped in `base_config/`) and the tokenizer and weights of "
              f"{f.model_id}. It sets MODEL_WEIGHTS_DIR and HF_MODEL to that directory, so the chips "
              f"load the weights of {f.model_id}.",
              f"- Context length {f.max_model_len} tokens; up to {f.max_num_seqs} sequences at a time."]
    if f.sampling_from_base:
        lines.append(f"- {sampling_disclosures(f.model_id, f.nearest_model)[0]}")
    if f.drafter:
        where = ("is in the table below (graft trial)" if f.acceptance_measured else "is listed under Not measured")
        lines.append(f"- Speculative decoding uses the drafter {f.drafter}, which was trained on "
                     f"{f.nearest_model}. Its acceptance rate on this model {where}.")
    if f.graft:
        g = f.graft
        lines.append(f"- {f.model_id} ships no mtp.* tensors, so the drafter has no MTP head of its own. The launcher "
                     f"reads the {len(g['tensors'])} mtp.* tensors of {f.nearest_model} from that model's pinned "
                     f"snapshot (revision `{g['revision']}`, file `{g['file']}`, downloaded alone on the first start). "
                     f"They are added at each start to `model-dir/` as one more weight file, `{g['shard']}` "
                     "(`graft.json` names them). The head was trained on "
                     f"{f.nearest_model}, so how well it suits {f.model_id} is shown only by the rows below.")
    if f.graft_decision == "graft":
        lines.append("- Stage 2 chose the graft by a measurement. Its graft trial served this model with and without the nearest model's "
                     "head in one lease and kept the graft because it served, agreed with the CPU reference and decoded faster. The rows "
                     "marked graft trial in the table below are that trial's.")
    elif f.graft_decision == "safe":
        lines.append("- Stage 2 tried the nearest model's mtp.* head grafted onto this model (its graft trial) and kept the safe "
                     "configuration. The rows marked graft trial in the table below show what the trial measured, and "
                     "`graft-decision.json` in the evidence gives the reason.")
    if f.drafter_off and f.vocab:
        lines.append(f"- Speculative decoding is off. The drafter of {f.nearest_model} works on the token ids of {f.nearest_model}, "
                     f"and the vocabulary of {f.model_id} has {_count(f.vocab[0])} entries against {_count(f.vocab[1])}.")
    elif f.drafter_off:
        lines.append(f"- Speculative decoding is off. The drafter of {f.nearest_model} needs the MTP "
                     f"tensors of the model, and {f.model_id} has none.")
    if f.vocab:
        lines.append(f"- The vocabulary is a trim of {f.nearest_model}'s: every row of its embedding and lm_head is a row of "
                     f"{f.nearest_model}'s, and the token ids are renumbered. The launcher writes {_count(f.vocab[0])} as `vocab_size` "
                     "in the served `config.json`.")
    if f.host_sampling:
        lines.append("- Sampling runs on the host, because on-device sampling is not used with the "
                     "drafter off on this mesh.")
    if f.sidecars:
        lines += ["", "## Not served", "",
                  f"{f.model_id} ships files this bundle does not load: "
                  + ", ".join(f"`{s}`" for s in f.sidecars) + ". The bundle serves the language-model "
                  "backbone only, so the sidecar head is not served. The run checked the head on the "
                  "host against the CPU reference; that check does not make it part of this bundle."]
    if has_facts(f.use):
        lines += ["", "## Intended use", "", *intended_use_lines(f)]
    else:
        lines += ["", "## Intended use", "",
                  f"Text generation with {f.model_id} through the OpenAI-compatible server that "
                  "`tt-model serve` starts."]
    if f.sidecars:
        lines.append("Out of scope: the sidecar head, which this bundle does not serve.")
    if has_facts(f.use):
        lines += ["", "## How to call it", "", *how_to_call_lines(f)]
    lines += ["", NUMBERS_HEAD, "",
              "Every number is labelled. A `measured` number names its evidence files; the first file "
              "holds the value. The files are in this repository under `evidence/`, at the paths "
              "shown, with the run's directory, the home directory and the host name replaced by "
              "`<RUN_DIR>`, `<HOME>` and `<HOST>`. `TODO` means not measured.", "",
              "| Number | Value | Label | Evidence |", "|---|---|---|---|"]
    for n in f.numbers:
        ev = ", ".join(f"`{e}`" for e in n.evidence) if n.label == "measured" else "-"
        lines.append(f"| {n.name} | {_value(n)} | {n.label} | {ev} |")
    if f.reported:
        lines += ["", REPORTED_HEAD, "", *reported_lines(f.reported, f.reported_source)]
    if f.sampling:
        lines += ["", SAMPLING_HEAD, "", *sampling_lines(f.sampling)]
    limits = [f"Tested on {f.chips} chips (mesh {f.mesh}) only; other configurations are not claimed "
              "by this bundle.",
              "Accuracy was compared with the CPU reference on one short fixed prompt only."]
    if f.graft:
        rate = ("The share of the drafter's guesses that are accepted on this model is in the table above (graft trial)."
                if f.acceptance_measured else
                "How often the drafter's guesses are accepted on this model was not measured.")
        limits.append(f"The drafter's MTP head is the nearest model's ({f.nearest_model}), added to {f.model_id} at "
                      f"start. {rate}")
    if f.stage_2_without_graft:
        limits.append("The stage 2 and stage 4 numbers were measured without the graft (the drafter off and sampling "
                      "on the host). The stage 7 rows were measured with this package's settings.")
    elif f.graft and f.graft_decision == "graft":
        limits.append("The stage 2 rows were measured without the graft (the drafter off and sampling on the host). The graft trial "
                      "and the stage 7 rows were measured with it.")
    if f.drafter_off:
        limits.append("Speculative decoding is off, so decode speed is that of plain decoding.")
    if f.vocab:
        limits.append(f"Token ids differ from {f.nearest_model}'s: a client that sends or reads token ids, or uses a tokenizer other than "
                      f"this model's, gets different ids. {_count(f.vocab[1] - f.vocab[0])} entries of {f.nearest_model}'s vocabulary are absent.")
    if f.host_sampling:
        limits.append("Sampling runs on the host.")
    if f.sampling_from_base:
        limits.append(f"{sampling_disclosures(f.model_id, f.nearest_model)[1]} Send your own settings or use "
                      "those on the model's card.")
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
    """Every file the Numbers table cites for a `measured` number, in order, without repeats, then the file the tool-call
    sentence under Intended use cites, if it does."""
    section, _ = _numbers_section(card)
    seen: list[str] = []
    for row in (r for r in section.splitlines() if r.startswith("| ") and not r.startswith("| Number ")):
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[2] == "measured":
            for p in cells[3].split(","):
                p = p.strip().strip("`")
                if p and p != "-" and p not in seen:
                    seen.append(p)
    intended, _ = _section(card, "## Intended use")
    cited = re.search(r"In the runs recorded in `([^`]+)` the tool-use prompts", " ".join(intended.split()))
    if cited and cited.group(1) not in seen:
        seen.append(cited.group(1))
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


def _section(card: str, head: str) -> tuple[str, str]:
    """(the text of the section that starts with `head`, the card without it); ("", card) when it is absent."""
    if head not in card:
        return "", card
    before, after = card.split(head, 1)
    section, sep, rest = after.partition("\n## ")
    return section, before + (sep + rest if sep else "")


def sampling_disclosures(model_id: str, nearest_model: str) -> tuple[str, str]:
    """The two sentences a card must keep when the package serves the nearest model's generation_config.json: the
    one in the facts list and the one under Limitations. `render_card` writes them and `card_problems` looks for them."""
    return (f"{model_id} ships no generation_config.json, so the stop tokens and the sampling defaults the "
            f"server applies when a request sets none are those of {nearest_model}.",
            f"Default sampling settings are those of {nearest_model}, because {model_id} ships "
            "no generation_config.json.")


def _units(text: str) -> list[str]:
    """Prose and table rows to compare: a table row is one unit, and a run of prose lines (a paragraph the author
    of a card may have re-wrapped) is one unit with its whitespace collapsed."""
    units, prose = [], []
    for ln in text.splitlines() + [""]:
        if ln.strip() and not ln.lstrip().startswith("|"):
            prose.append(ln)
            continue
        if prose:
            units.append(" ".join(" ".join(prose).split()))
            prose = []
        if ln.strip():
            units.append(" ".join(ln.split()))
    return units


def reported_lines(reported, source: str) -> list[str]:
    """The lines of the "Reported by the authors" section after its heading, one list item per line: the sentence
    that cites the source and says the scores were not measured, a blank line, and the table. `render_card` writes
    them and `card_problems` compares the card's section with them."""
    return [f"These scores come from {_cell(source) or 'the model authors'}. They {NOT_MEASURED_SENTENCE}: "
            "the authors' protocol, tools and judge differ from the checks in the table above, and nothing in "
            "this repository reproduces them.", "",
            "| Benchmark | Reported | Note |", "|---|---|---|",
            *[f"| {_cell(r.name)} | {_cell(r.value)} | {_cell(r.note)} |" for r in reported]]


def graft_markers(nearest_model: str) -> tuple[str, ...]:
    """What a card must say when the package adds the nearest model's mtp.* tensors at start."""
    return (f"mtp.* tensors of {nearest_model}", "added at each start", "`graft.json`")


GRAFT_DECISION_MARKERS = {"graft": "chose the graft", "safe": "kept the safe configuration"}

SAMPLING_HEAD = "## Sampling"
SAMPLING_TESTED = "top-k, penalties and logprobs were not tested"


def sampling_lines(sampling: dict) -> list[str]:
    """The lines of the Sampling section (issue 107): the verdict's sentence, then the configuration stage 2 served when it measured, then what was
    tested. `sampling` is package.sampling_summary's dict. No number is written here; the Numbers table has them."""
    serving = sampling.get("serving") if isinstance(sampling.get("serving"), dict) else {}
    return [sampling["card_sentence"],
            f"Stage 2's swap check measured this on one short prompt, served with its own configuration (drafter: {serving.get('drafter', 'not recorded')}, "
            f"sampling: {serving.get('sampling', 'not recorded')}). It sent greedy requests and sampled requests with temperature above 0, top-p below 1 "
            f"and a seed. The speeds are in the Numbers table, in the rows marked sampling probe. Other sampling settings such as {SAMPLING_TESTED}."]


CUTOFF_MARKER = "reaches `max_tokens` while the model is still thinking"
# "(default `x`)", "the default effort is `x`", "default reasoning effort `x`"
_EFFORT_DEFAULT_CLAIM = re.compile(r"\bdefault(?:\s+(?:reasoning\s+)?effort)?(?:\s+(?:is|of))?\s+`([^`]+)`", re.I)
_VISION_CLAIM = re.compile(r"(?<!not )(?<!n't )\b(?:accepts?|supports?|takes?|handles?|processes|serves?)\s+"
                           r"(?:images?|videos?|vision)\b|\b(?:image|video)_url\b", re.I)


def _flat(text: str) -> str:
    return " ".join(text.split())


def _cutoffs_in_table(card: str) -> bool:
    """The card's Numbers table shows a measured count of replies cut off at the token limit above zero."""
    section, _ = _numbers_section(card)
    for row in (r for r in section.splitlines() if r.startswith("| ") and not r.startswith("| Number ")):
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) == 4 and CUTOFF_NAME in cells[0] and cells[2] == "measured":
            try:
                if float(cells[1].split(" ", 1)[0]) > 0:
                    return True
            except ValueError:
                pass
    return False


def use_problems(card: str, use: UseFacts | None) -> list[str]:
    """What is wrong with the card's account of how to use the model, given the facts read from the model's files.
    A fact the sources give must be in the Intended use or How to call it section; a claim the sources contradict is
    refused wherever the card makes it. Nothing is checked when the run found no facts."""
    if not has_facts(use):
        return []
    u, problems = use, []
    flat = _flat(card)
    intended, _ = _section(card, "## Intended use")
    if "## How to call it" not in card:
        problems.append("the card has no '## How to call it' section: the model's files give facts about how to call it")
    how, _ = _section(card, "## How to call it")
    uflat = _flat(intended + "\n" + how)
    own_prose = card                       # what the card says about this package, without the authors' words
    if u.authors_use:
        quote_lines = [ln for ln in intended.splitlines() if ln.startswith(">")]
        quoted = _flat(" ".join(ln.lstrip("> ").strip() for ln in quote_lines))
        if quoted != _flat(u.authors_use):
            problems.append("the card must quote the authors' intended use unchanged, as a block quote under Intended use")
        else:                              # a verified quotation says what the authors say, not what this package does
            own_prose = "\n".join(ln for ln in card.splitlines() if ln not in quote_lines)
    on, off = "Thinking is on by default", "Thinking is off by default"
    if u.thinking_default is True and on not in uflat:
        problems.append(f"the chat template turns thinking on by default, so the card must say '{on}'")
    if u.thinking_default is False and off not in uflat:
        problems.append(f"the chat template turns thinking off by default, so the card must say '{off}'")
    if u.thinking_default is not True and re.search(on, flat, re.I):
        problems.append("the card says thinking is on by default, and the chat template does not show that")
    if u.thinking_default is not False and re.search(off, flat, re.I):
        problems.append("the card says thinking is off by default, and the chat template does not show that")
    if u.thinking_default is True and u.reasoning_parser and "`reasoning_content`" not in uflat:
        problems.append("run.sh sets a reasoning parser, so the card must say the thinking comes back in `reasoning_content`")
    if not u.reasoning_parser and "reasoning_content" in card:
        problems.append("the card names reasoning_content, and run.sh sets no reasoning parser")
    if u.effort_values:
        missing = [v for v in u.effort_values if f"`{v}`" not in uflat]
        if missing or "`reasoning_effort`" not in uflat:
            problems.append("the card must list the reasoning effort values the chat template accepts "
                            f"({', '.join(u.effort_values)}) and name `reasoning_effort`")
        stated = {v for v in _EFFORT_DEFAULT_CLAIM.findall(uflat) if v in u.effort_values}
        if u.effort_default and stated != {u.effort_default}:
            problems.append(f"the chat template's default reasoning effort is `{u.effort_default}`, so the card must say "
                            f"(default `{u.effort_default}`) and no other default (it says: "
                            f"{', '.join(sorted(stated)) or 'none'})")
        if not u.effort_default and stated:
            problems.append("the card names a default reasoning effort, and the chat template does not give one: "
                            f"{', '.join(sorted(stated))}")
    elif "reasoning_effort" in card:
        problems.append("the card names reasoning_effort, and the chat template lists no reasoning effort values")
    if u.thinking_default is True and u.can_disable_thinking and '"enable_thinking": false' not in flat:
        problems.append('the chat template has an enable_thinking switch, so the card must show {"enable_thinking": false}')
    if not u.can_disable_thinking and "enable_thinking" in card:
        problems.append("the card names enable_thinking, and the chat template has no such switch")
    if u.tools_ok:
        if "`tool_calls`" not in uflat:
            problems.append("the server parses tool calls, so the card must say they come back in `tool_calls`")
    elif "`tool_calls`" in card or '"tools"' in card:
        problems.append("the card claims tool calls, and the chat template or run.sh gives the server no way to return them")
    evidence = u.tool_evidence if u.tools_ok else None       # the renderer says it only when the server can return tool calls
    if evidence:
        path, names = evidence
        if "returned tool calls" not in flat or f"`{path}`" not in flat or any(f"`{n}`" not in flat for n in names):
            problems.append(f"the tool-use prompts returned tool calls (`{path}`), so the card must say they returned tool "
                            f"calls and name the file and the tools: {', '.join(names)}")
    elif "returned tool calls" in flat:
        problems.append("the card says the tool-use prompts returned tool calls, and no evidence file in the run shows it "
                        "for this package's server")
    if evidence:
        claimed = re.search(r"returned tool calls \(([^)]*)\)", flat)
        extra = [n for n in re.findall(r"`([^`]+)`", claimed.group(1)) if n not in evidence[1]] if claimed else []
        if extra:
            problems.append(f"the card names tools the evidence does not hold: {', '.join(extra)}")
    if u.text_only and "Text only" not in uflat:
        problems.append("run.sh skips the vision tower, so the card must say the package is text only")
    if not u.text_only and "Text only" in card:
        problems.append("the card says the package is text only, and run.sh does not skip the vision tower")
    if u.text_only and _VISION_CLAIM.search(own_prose):
        problems.append("the card claims image or video input, and run.sh skips the vision tower")
    untested = (u.thinking_default is not None or bool(u.effort_values) or u.text_only
                or (u.tools_ok and not u.tool_evidence))
    if untested and UNTESTED_SENTENCE not in uflat:
        problems.append(f"facts taken from the chat template and run.sh must say '{UNTESTED_SENTENCE}'")
    warned = u.thinking_default is True and bool(u.reasoning_parser)
    if warned and _cutoffs_in_table(card) and CUTOFF_MARKER not in uflat:
        problems.append("the benchmark runs cut off replies at the token limit, so the card must warn that a reply that "
                        f"{CUTOFF_MARKER}")
    if CUTOFF_MARKER in flat and not warned:
        problems.append("the card warns about a reply that reaches max_tokens while thinking, and the sources do not show "
                        "that the model thinks by default with a reasoning field")
    return problems


def card_problems(card: str, *, license_id: str, run_dir, sidecars=(), base_files=(), reported=(), reported_source="", model_id="",
                  nearest_model="", graft: bool = False, graft_decision: str | None = None, trial_rows=(),
                  use: UseFacts | None = None, vocab: tuple[int, int] | None = None, sampling: dict | None = None,
                  sampling_rows=()) -> list[str]:
    problems = use_problems(card, use)
    if sampling:
        body, _ = _section(card, SAMPLING_HEAD)
        if not body:
            problems.append(f"the card must have the section {SAMPLING_HEAD!r}: stage 2's sampling probe recorded a result")
        elif " ".join(sampling["card_sentence"].split()) not in " ".join(body.split()):
            problems.append("the sampling section must give the sampling probe's verdict sentence unchanged: "
                            f"{sampling['card_sentence']}")
    elif SAMPLING_HEAD in card:
        problems.append(f"the card has the section {SAMPLING_HEAD!r}, and the run has no sampling probe record to support it")
    if vocab:
        flat_vocab = " ".join(card.split())
        for size in vocab:
            if _count(size) not in flat_vocab:
                problems.append(f"the model is a vocabulary trim, so the card must give the vocabulary sizes; {_count(size)} is missing")
        if "token ids are renumbered" not in flat_vocab:
            problems.append("the model is a vocabulary trim, so the card must say the token ids are renumbered")
        if "needs the MTP tensors of the model" in flat_vocab:
            problems.append(f"the card says the model has no mtp.* tensors; the drafter is off because the vocabulary differs "
                            f"({_count(vocab[0])} entries against {_count(vocab[1])})")
        limits = card.split("## Limitations", 1)[1].split("\n## ", 1)[0] if "## Limitations" in card else ""
        if "Token ids differ" not in limits:
            problems.append("the model is a vocabulary trim, so the card must say under Limitations that token ids differ from the "
                            "nearest model's")
    if graft_decision in GRAFT_DECISION_MARKERS:
        if GRAFT_DECISION_MARKERS[graft_decision] not in " ".join(card.split()):
            problems.append(f"stage 2's graft trial decided {graft_decision!r}, so the card must say that stage 2 "
                            f"{GRAFT_DECISION_MARKERS[graft_decision]}")
    if graft:
        flat_card = " ".join(card.split())
        missing = [m for m in graft_markers(nearest_model or "the nearest model") if m not in flat_card]
        if missing:
            problems.append("the package grafts the nearest model's mtp.* tensors at each start, so the card must say "
                            f"so; it is missing: {', '.join(missing)}")
    reported_section, card_without_reported = _section(card, REPORTED_HEAD)
    if not reported and reported_section:
        problems.append(f"the card has the section {REPORTED_HEAD!r}, and the run has no reported-results.json to "
                        "support it")
        card_without_reported = card                  # its text is not exempt from the number check
    if reported:
        if not reported_section:
            problems.append(f"the card must have the section {REPORTED_HEAD!r}: the run was given scores the authors report")
        else:
            have, want = _units(reported_section), _units("\n".join(reported_lines(reported, reported_source)))
            intro = want[0]
            if not any(NOT_MEASURED_SENTENCE in ln for ln in have):
                problems.append(f"the reported-scores section must say the scores {NOT_MEASURED_SENTENCE}")
            elif intro not in have:
                problems.append("the reported-scores section must cite the source of the scores, "
                                f"{reported_source or 'the model authors'}, and give the protocol sentence unchanged")
            for r, row in zip(reported, want[3:]):
                if row not in have:
                    shown = f"{r.name}: {r.value}" + (f" ({r.note})" if r.note else "")
                    problems.append(f"the reported score {shown} is not in the reported-scores section as given "
                                    "in reported-results.json (name, value and note)")
            for ln in have:
                if ln not in want:
                    problems.append(f"the reported-scores section has a line that reported-results.json does not "
                                    f"give: {ln[:80]!r}")
    for n in base_files:
        if f"`{n}`" not in card:
            problems.append(f"the card must name `{n}`, which the package takes from the nearest model")
    if "generation_config.json" in base_files:
        flat = " ".join(card.split())
        if not all(s in flat for s in sampling_disclosures(model_id, nearest_model)):
            problems.append("the package serves the nearest model's generation_config.json, so the card must say "
                            "the model ships none, that the stop tokens and sampling defaults are "
                            f"{nearest_model or 'the nearest model'}'s, and give the same default-sampling "
                            "limitation under Limitations")
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
    section, rest = _numbers_section(card_without_reported)
    if not section:
        problems.append("the card has no Numbers section")
    names = {_cell(r.name) for r in reported}
    shown = {r.strip().strip("|").split("|")[0].strip() for r in section.splitlines() if r.startswith("| ")}
    for name in trial_rows:
        if name not in shown:
            problems.append(f"the Numbers table must show {name!r}: stage 2's graft trial measured it")
    for name in sampling_rows:
        if name not in shown:
            problems.append(f"the Numbers table must show {name!r}: stage 2's sampling probe measured it")
    for row in (r for r in section.splitlines() if r.startswith("| ") and not r.startswith("| Number ")):
        first = row.strip().strip("|").split("|")[0].strip()
        if first in names:
            problems.append(f"the score {first} is reported by the authors, and the Numbers table lists it as a "
                            "number the run measured")
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
