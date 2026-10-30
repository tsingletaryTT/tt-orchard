# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 7: stage a v6 thin package of a weights-only model, as supervisor code (plan 5).

This module owns the package a run hands to the operator. It never publishes. It runs one external
program on its own, `tt-model package-thin`, always with `--out` and never with a repo id
(`assert_no_publish`), and it runs the staged package's `install.sh` in a separate copy. It never
calls `tt-model push` or `publish`, `hf upload`, `git push` or the Hugging Face hub API. The
publish commands it writes are text for the operator (`publish_commands`).

The package is built from the nearest model's installed v6 bundle (the "source bundle") with the
same model code, wheels, environment and fixed vLLM arguments, and `--weights` naming the new
model at its pinned revision. Three edits follow, each found by failing first on this machine:

1. Fixed vLLM arguments. package-thin has no flag for them, so they are copied from the source
   bundle's run.sh and spliced in before the passed-through `"$@"` (`extra_args_from`,
   `splice_extra_args`).
2. Weights. The TT runtime takes its weights directory from MODEL_WEIGHTS_DIR, then HF_MODEL, then
   the config path. The bundle's model class is registered for the nearest model's architecture
   (a vision-language config), and the new model's config.json names a text-only architecture,
   which vLLM refuses. So run.sh runs `prepare_model_dir.py` before vLLM: it builds `model-dir/`
   from the nearest model's config files (shipped in `base_config/`) and the new model's tokenizer
   and weights. `--model`, HF_MODEL and MODEL_WEIGHTS_DIR all name that directory
   (`wire_weights`). `weights_wiring_problems` checks the result, and the gate runs it again.
3. Provenance. The manifest's `producer.hostname` is replaced, run.sh and install.sh are made
   executable (package-thin leaves them without +x), and every wheel must be byte-identical to
   the source bundle's, because wheels are binary and the scrub does not read them.

Stage 7 ships no tensor cache and no weights (orchard/scrub.py, scrub_package). The boot check
(`prepare_verify`, package_templates/verify_bundle.py) installs and serves a copy, so the staged
directory never gains a venv, a model-dir or a cache. The supervisor passes the agent shells'
environment to every command here (no tokens, HOME inside the run directory), so a call that
tried to upload would also find no credentials.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

from orchard.defaults import (PACKAGE_HEALTH_TIMEOUT_S, PACKAGE_INSTALL_TIMEOUT_S,
                              PACKAGE_THIN_TIMEOUT_S, PACKAGE_VERIFY_DEADLINE_S)
from orchard.graft_trial import DECISION_FILE, TRIAL_DIR, TRIAL_RESULT, accept_rate, read_decision
from orchard import safeio, sampling_choice, sampling_probe
from orchard.card_use import read_use_facts
from orchard.package_card import (CardFacts, Number, Reported, card_evidence_paths, inside, non_commercial,
                                  read_license, render_card)
from orchard.scrub import HOME_PATH, scrub_package
from orchard.stages import delta_model_has_mtp, gate_weights_swap

MAX_RECORD_BYTES = 4 * 1024 * 1024    # choice: a stage record, package.json or the reported scores is a few kilobytes (issue 137)

TEMPLATES = Path(__file__).with_name("package_templates")
SWAP_TEMPLATES = Path(__file__).with_name("skills") / "weights-swap-templates"
# generation_config.json is here for a model that ships none: stage 2's staging copied the nearest model's as a
# real file. A model's own is a link and is left out, so the package takes it from the model.
BASE_CONFIG_FILES = ("config.json", "generation_config.json", "preprocessor_config.json",
                     "video_preprocessor_config.json")

GRAFT_FILE = "graft.json"
GRAFT_SHARD = "model-mtp-from-base.safetensors"      # the name prepare_swap.py and the package's prepare_model_dir.py use
MODEL_DIR = '"$HERE/model-dir"'
WIRING_LINES = (f"export HF_MODEL={MODEL_DIR}", f"export MODEL_WEIGHTS_DIR={MODEL_DIR}")
PREPARE_LINE = '"$PYBIN" "$HERE/prepare_model_dir.py"'
EXEC_LINE = 'exec "${CMD[@]}"'
PASS_THROUGH = ' "$@")'


class PackageError(Exception):
    """Stage 7 cannot stage or check the package. The message says why."""


# ---- run.sh edits -------------------------------------------------------------------------------

def _cmd_line(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.startswith("CMD=(")]
    if len(lines) != 1:
        raise PackageError(f"expected exactly one CMD=( line in run.sh, found {len(lines)}")
    return lines[0]


def extra_args_from(source_run_sh: str) -> str:
    """The fixed vLLM arguments a source bundle's run.sh adds after the generated ones.

    package-thin writes `--max_model_len N` last and the passed-through `"$@"` at the end, so
    whatever stands between the two was spliced in by the bundle's author."""
    m = re.search(r'--max_model_len \d+(?P<extra>.*) "\$@"\)\s*$', _cmd_line(source_run_sh))
    if m is None:
        raise PackageError('the source run.sh command has no "--max_model_len N ... "$@")" to read '
                           "the fixed vLLM arguments from")
    return m.group("extra").strip()


def splice_extra_args(text: str, extra: str) -> str:
    """Put `extra` just before the passed-through arguments, so an operator's own arguments still
    come last and win."""
    if not extra:
        return text
    line = _cmd_line(text)
    if not line.endswith(PASS_THROUGH):
        raise PackageError('the run.sh command does not end with "$@")')
    return text.replace(line, line[:-len(PASS_THROUGH)] + " " + extra + PASS_THROUGH)


ADDITIONAL = re.compile(r"--additional-config\s+'([^']*)'")
DRAFTER_ENV = re.compile(r"^(DFLASH_WEIGHTS|QWEN36_DFLASH_\w+)$")


def drop_device_sampling(extra: str) -> str:
    """Remove "sample_on_device_mode" from the one --additional-config in `extra`.

    Stage 2 and 4 served a model without mtp.* tensors with the drafter off, and plain decoding on a
    1x2 mesh refuses on-device sampling, so the host samples. A package that kept the key would not be
    the thing that was tested. An option that is absent, or a config without the key, is left alone."""
    found = ADDITIONAL.findall(extra)
    if len(found) > 1:
        raise PackageError(f"expected at most one --additional-config in the fixed vLLM arguments, "
                           f"found {len(found)}")
    if not found:
        return extra
    try:
        cfg = json.loads(found[0])
        tt = cfg.get("tt")
    except (ValueError, AttributeError):
        raise PackageError("the --additional-config in the fixed vLLM arguments is not valid JSON")
    if not isinstance(tt, dict) or "sample_on_device_mode" not in tt:
        return extra
    del tt["sample_on_device_mode"]
    return ADDITIONAL.sub(lambda m: f"--additional-config '{json.dumps(cfg)}'", extra, count=1)


def serving_env(env: dict, *, drafter_off: bool) -> dict:
    """The manifest environment package-thin is given. With the drafter off, QWEN36_DRAFTER is empty
    and the drafter's own settings (its repo, tensor-parallel size, block sizes) are not passed, so the
    package names no drafter repo and downloads none."""
    if not drafter_off:
        return dict(env)
    out = {k: v for k, v in env.items() if not DRAFTER_ENV.match(k)}
    out["QWEN36_DRAFTER"] = ""
    return out


def _drafter_on(run_sh: str, env: dict) -> bool:
    """Does the package turn the speculative drafter on? The last `export QWEN36_DRAFTER=` line of run.sh decides,
    else the manifest's value."""
    values = [m.group(1).strip().strip("\"'") for ln in run_sh.splitlines()
              if (m := re.match(r"\s*export\s+QWEN36_DRAFTER=(.*)$", ln))]
    return bool(values[-1]) if values else bool(env.get("QWEN36_DRAFTER"))


def serving_problems(run_sh: str, env: dict, *, drafter_off: bool, host_sampling: bool,
                     model_has_mtp: bool | None = None, graft: bool = False) -> list[str]:
    """What is wrong with a staged package's serving settings next to what stage 2 served with.

    `model_has_mtp` is stage 0's finding (None when it did not record one). A package that turns the drafter on
    for a model with no mtp.* tensors needs a graft (`graft`: the package has a graft.json); without one the
    server stops at start with "model has no MTP head"."""
    problems = []
    if drafter_off:
        drafter = [m.group(1).strip().strip("\"'") for ln in run_sh.splitlines()
                   if (m := re.match(r"\s*export\s+QWEN36_DRAFTER=(.*)$", ln))]
        if any(drafter):
            problems.append(f"run.sh turns the drafter on ({drafter}); stage 2 served with it off")
        if env.get("QWEN36_DRAFTER"):
            problems.append(f"the manifest sets QWEN36_DRAFTER={env['QWEN36_DRAFTER']!r}; stage 2 "
                            "served with the drafter off")
        left = sorted(k for k in env if DRAFTER_ENV.match(k)) + \
            (["DFLASH_WEIGHTS in run.sh"] if "DFLASH_WEIGHTS" in run_sh else [])
        if left:
            problems.append(f"the drafter's settings are still in the package: {left}")
    if host_sampling and "sample_on_device_mode" in run_sh:
        problems.append("run.sh sets sample_on_device_mode; stage 2 served with host sampling")
    on = _drafter_on(run_sh, env)
    if graft and not on:
        problems.append(f"the package has a {GRAFT_FILE}, but run.sh and the manifest turn the drafter off; the "
                        "grafted MTP head is used only with the drafter on")
    if on and model_has_mtp is False and not graft:
        problems.append(f"run.sh turns the drafter on for a model with no mtp.* tensors, and the package has no "
                        f"{GRAFT_FILE}: the server stops at start with 'model has no MTP head'")
    return problems


def stage_2_serving(run_dir) -> tuple[bool, bool]:
    """(drafter_off, host_sampling) as the run script stage 2 served with says. A run without that
    file (older runs) is taken to have served as the source bundle does."""
    path = Path(run_dir) / "stages/2/run.sh"
    if not path.is_file():
        return False, False
    text = path.read_text(encoding="utf-8")
    drafter = [m.group(1).strip().strip("\"'") for ln in text.splitlines()
               if (m := re.match(r"\s*export\s+QWEN36_DRAFTER=(.*)$", ln))]
    cfg = ADDITIONAL.search(text)
    return bool(drafter) and drafter[-1] == "", bool(cfg) and "sample_on_device_mode" not in cfg.group(1)


def served_drafter_state(run_dir) -> tuple[bool, bool]:
    """(drafter_off, host_sampling) of the package: stage 2's run script says, and stage 2's sampling decision overrides it when it chose the
    drafter-off configuration (orchard/sampling_choice.py): that variant was served with the drafter off and the host sampling."""
    if sampling_choice.drafter_off_chosen(run_dir):
        return True, True
    return stage_2_serving(run_dir)


def stage_2_vocab(run_dir) -> tuple[int, int] | None:
    """(the model's vocabulary size, the nearest model's) when stage 2's swap recorded a vocabulary trim (model-dir-sources.json's
    `vocab` record, written by prepare_swap.py; issue 56), else None. Anything that is not whole numbers with the model's smaller and
    `trimmed` true is no record."""
    try:
        rec = safeio.read_json(safeio.resolved(Path(run_dir) / "stages/2/model-dir-sources.json"), max_bytes=MAX_RECORD_BYTES)
        v = rec["vocab"]
        m, b = v["model"], v["base"]
    except (safeio.SafeReadError, ValueError, KeyError, TypeError):
        return None
    if v.get("trimmed") is not True or any(isinstance(x, bool) or not isinstance(x, int) for x in (m, b)) or not 0 < m < b:
        return None
    return m, b


def _stage_2_grafted(run: Path) -> bool:
    """Did stage 2's swap graft the nearest model's mtp.* tensors? (model-dir-sources.json's mtp_graft record)"""
    try:
        rec = safeio.read_json(safeio.resolved(run / "stages/2/model-dir-sources.json"), max_bytes=MAX_RECORD_BYTES)
    except safeio.SafeReadError:
        return False
    return isinstance(rec, dict) and isinstance(rec.get("mtp_graft"), dict)


def _package_grafted(run: Path) -> bool:
    """Did stage 7 stage a grafted package? (package.json's mtp_graft record; finish and the gate read it)"""
    try:
        rec = safeio.read_json(safeio.resolved(run / "stages/7/package.json"), max_bytes=MAX_RECORD_BYTES)
    except safeio.SafeReadError:
        return False
    return isinstance(rec, dict) and isinstance(rec.get("mtp_graft"), dict)


def _stage_2_graft_record(run: Path) -> dict | None:
    try:
        rec = safeio.read_json(safeio.resolved(run / "stages/2/model-dir-sources.json"), max_bytes=MAX_RECORD_BYTES)
    except safeio.SafeReadError:
        return None
    rec = rec.get("mtp_graft") if isinstance(rec, dict) else None
    return rec if isinstance(rec, dict) else None


def stage_2_graft_problems(model_dir: Path, record: dict, nearest_model: str, base_snapshot: Path,
                           new_snapshot: Path) -> list[str]:
    """Why the graft shard in stage 2's model directory is not the nearest model's mtp.* tensors the record says
    it is; empty when it is. The shard is a regular file prepare_swap.py wrote, so the Hugging Face links that
    vouch for the model's own weights cannot vouch for it. Its bytes are hashed and compared with the record's
    sha256, and that digest is rebuilt from the nearest model's pinned snapshot with the code that built it."""
    shard = model_dir / GRAFT_SHARD
    if record.get("file") != GRAFT_SHARD:
        return [f"model-dir-sources.json's mtp_graft file is {record.get('file')!r}; it must be {GRAFT_SHARD}"]
    want = record.get("sha256")
    if not isinstance(want, str) or not re.fullmatch(r"[0-9a-f]{64}", want):
        return ["model-dir-sources.json's mtp_graft sha256 is not 64 hex characters"]
    if shard.is_symlink() or not shard.is_file():
        return [f"{shard} is not the regular file prepare_swap.py writes for a graft"]
    if sha256(shard) != want:
        return [f"{shard} does not match the sha256 in model-dir-sources.json's mtp_graft record"]
    rev = record.get("base_revision")
    if rev != base_snapshot.name:
        return [f"model-dir-sources.json's mtp_graft base_revision is {rev!r}; stage 2's nearest-model snapshot is "
                f"{base_snapshot.name!r}"]
    source = plan_graft(nearest_model, base_snapshot, new_snapshot)          # PackageError when it cannot be made
    if source["sha256"] != want:
        return [f"the graft shard's sha256 is not what the nearest model's snapshot {base_snapshot.name} "
                "gives for its mtp.* tensors"]
    return []


def _swap_template():
    """prepare_swap.py, the stage 2 template, loaded as a module whose failures raise PackageError. Stage 7 builds
    the graft record with the code stage 2 built the graft with, so the two cannot disagree."""
    spec = importlib.util.spec_from_file_location("orchard_prepare_swap", SWAP_TEMPLATES / "prepare_swap.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def fail(message: str) -> None:
        raise PackageError(message)
    mod.fail = fail
    return mod


def plan_graft(nearest_model: str, base_snapshot: Path, new_snapshot: Path) -> dict:
    """The record that becomes graft.json: which file of the nearest model's pinned snapshot holds the mtp.* tensors,
    their names, and the sha256 of the shard they make. Raises PackageError when they cannot be grafted."""
    if not base_snapshot.is_dir() or not REVISION.fullmatch(base_snapshot.name):
        raise PackageError(f"the graft needs the nearest model's pinned snapshot: stage 2's swap_config.json base_snapshot "
                           f"{str(base_snapshot)!r} is not a snapshot directory named by a commit")
    swap = _swap_template()
    plan = swap.plan_mtp_graft(base_snapshot, new_snapshot)
    with tempfile.TemporaryDirectory() as tmp:
        digest, total = swap.write_graft_shard(plan, Path(tmp) / "shard.safetensors")
    return {"schema": 1, "repo": nearest_model, "revision": base_snapshot.name, "file": plan["file"],
            "shard": GRAFT_SHARD, "tensors": sorted(plan["entries"]), "sha256": digest, "bytes": total}


def graft_problems(pkg: Path, *, nearest_model: str, base_revision: str) -> list[str]:
    """What is wrong with a staged package's graft.json next to the run's nearest model and its pinned snapshot."""
    path = pkg / GRAFT_FILE
    try:
        g = json.loads(path.read_text(encoding="utf-8"))
    except OSError:
        return [f"{GRAFT_FILE} is missing"]
    except ValueError:
        return [f"{GRAFT_FILE} is not valid JSON"]
    if not isinstance(g, dict):
        return [f"{GRAFT_FILE} is not a JSON object"]
    out = []
    if g.get("repo") != nearest_model:
        out.append(f"{GRAFT_FILE}'s repo is {g.get('repo')!r}; the nearest model is {nearest_model!r}")
    rev = g.get("revision")
    if not isinstance(rev, str) or not REVISION.fullmatch(rev):
        out.append(f"{GRAFT_FILE}'s revision {rev!r} is not a 40-character commit")
    elif base_revision and rev != base_revision:
        out.append(f"{GRAFT_FILE}'s revision is {rev}; stage 2's nearest-model snapshot is {base_revision}")
    if not isinstance(g.get("file"), str) or not g["file"].endswith(".safetensors"):
        out.append(f"{GRAFT_FILE}'s file {g.get('file')!r} is not a .safetensors file name")
    if g.get("shard") != GRAFT_SHARD:
        out.append(f"{GRAFT_FILE}'s shard is {g.get('shard')!r}; it must be {GRAFT_SHARD}")
    tensors = g.get("tensors")
    if not isinstance(tensors, list) or not tensors or not all(isinstance(t, str) and t for t in tensors):
        out.append(f"{GRAFT_FILE}'s tensors must be a non-empty list of tensor names")
    if not isinstance(g.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", g["sha256"]):
        out.append(f"{GRAFT_FILE}'s sha256 is not 64 hex characters")
    try:
        script = (pkg / "prepare_model_dir.py").read_text(encoding="utf-8")
    except OSError:
        script = ""
    if GRAFT_FILE not in script:
        out.append(f"prepare_model_dir.py does not read {GRAFT_FILE}, so the package would start without the grafted head")
    return out


GRAFT_IDENTITY = ("repo", "revision", "file", "shard", "tensors", "sha256")


def graft_evidence_problems(pkg: Path, verify) -> list[str]:
    """How a staged profile's graft.json differs from the graft the boot check served; empty when they agree.

    The boot check (verify_bundle.py) records, in verify.json's `graft`, the graft.json its installed copy read and the
    sha256 of the shard that copy's launcher built. graft.json's digest is only a format until it is compared with
    that: a package whose digest was edited after the boot check would pass every other check and then refuse to start.
    Every profile is compared, so an optional profile that was never booted cannot carry another graft."""
    has = (pkg / GRAFT_FILE).is_file()
    served = verify.get("graft") if isinstance(verify, dict) else None
    if not has:
        return ["the boot check served a graft, but this profile has no graft.json"] if isinstance(served, dict) and served else []
    if not isinstance(verify, dict):
        return []            # no boot check evidence at all: the gate already says so
    if not isinstance(served, dict) or not served:
        return [f"{GRAFT_FILE} cannot be checked against the boot check: its verify.json records no graft"]
    try:
        g = json.loads((pkg / GRAFT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []            # graft_problems reports an unreadable graft.json
    if not isinstance(g, dict):
        return []
    out = []
    for key in GRAFT_IDENTITY:
        mine, theirs = g.get(key), served.get(key)
        if key == "tensors" and isinstance(mine, list) and isinstance(theirs, list):
            mine, theirs = sorted(map(str, mine)), sorted(map(str, theirs))
        if mine != theirs:
            out.append(f"{GRAFT_FILE}'s {key} is {mine!r}; the boot check served {theirs!r}")
    if served.get("served_shard_sha256") != g.get("sha256"):
        out.append(f"the boot check's served_shard_sha256 is {served.get('served_shard_sha256')!r}; "
                   f"{GRAFT_FILE}'s sha256 is {g.get('sha256')!r}")
    return out


def _once(pattern: str, repl: str, text: str, what: str, flags=0) -> str:
    out, n = re.subn(pattern, lambda m: repl, text, flags=flags)
    if n != 1:
        raise PackageError(f"expected exactly one {what} in run.sh, found {n}")
    return out


# The package-thin of tt-model 0.1.0 (seen 2026-10-19) resolves the pinned weights itself: it sets a default HF_MODEL near the top,
# resolves the snapshot with snapshot_download into `resolved_hf_model`, and just before the exec sets HF_MODEL and MODEL_WEIGHTS_DIR to it.
RESOLVED_LINES = ('export HF_MODEL="$resolved_hf_model"', 'export MODEL_WEIGHTS_DIR="$resolved_hf_model"')
DEFAULT_HF_MODEL = re.compile(r'^\s*export HF_MODEL="\$\{HF_MODEL:-[A-Za-z0-9_.@/-]+\}"$')


def wire_weights(text: str, model_id: str) -> str:
    """Point every weights setting in a generated run.sh at the bundle's model-dir.

    Two generated formats: one `export HF_MODEL=` default (older package-thin), or that default plus a resolver block whose result is
    exported as HF_MODEL and MODEL_WEIGHTS_DIR before the exec (current package-thin). In the second the default and the resolver stay
    (the resolver only makes sure the pinned snapshot is in the Hugging Face cache, which prepare_model_dir.py needs) and the two exports
    after it become the wiring lines."""
    q = re.escape(model_id)
    text = _once(rf'--model (?:"{q}"|{q})(?=\s)', f"--model {MODEL_DIR}", text, f'--model "{model_id}"')
    text = _once(r" --revision [0-9a-f]{40}(?=\s)", "", text, "--revision <40 hex>")
    text = _once(r" --tokenizer-revision [0-9a-f]{40}(?=\s)", "", text, "--tokenizer-revision <40 hex>")
    if "resolved_hf_model" in text:
        for resolved, wired in zip(RESOLVED_LINES, WIRING_LINES):
            text = _once(rf"^{re.escape(resolved)}$", wired, text, f"{resolved} line", flags=re.M)
    else:
        text = _once(r"^export HF_MODEL=.*$", "\n".join(WIRING_LINES), text, "export HF_MODEL= line",
                     flags=re.M)
    return _once(r"^exec \"\$\{CMD\[@\]\}\"$", f"{PREPARE_LINE}\n{EXEC_LINE}", text,
                 'exec "${CMD[@]}" line', flags=re.M)


def _names_model(text: str, model: str) -> bool:
    """True when `text` holds the whole model id: not as the start of a longer id (`org/model-v2`, `org/model.5`) and not as the end of one
    with a longer owner (a `-` before it counts as part of an owner, except in the shell default `:-`). A sentence's closing full stop still counts as the end of the id."""
    return re.search(rf"(?<![\w./])(?<![^:]-){re.escape(model)}(?![\w\-])(?!\.\w)", text) is not None


def weights_wiring_problems(text: str, *, nearest_model: str) -> list[str]:
    """What is wrong with a staged run.sh's weights settings; empty when the chips will load the
    weights in model-dir and nothing names the nearest model."""
    problems = []
    lines = text.splitlines()
    for var, want in (("HF_MODEL", WIRING_LINES[0]), ("MODEL_WEIGHTS_DIR", WIRING_LINES[1])):
        sets = [ln for ln in lines if re.match(rf"\s*(?:export\s+)?{var}=", ln)]
        if var == "HF_MODEL" and len(sets) == 2 and DEFAULT_HF_MODEL.match(sets[0]) and not _names_model(sets[0], nearest_model):
            sets = sets[1:]          # the resolving format's default (the id its resolver reads), then the wiring line
        if sets != [want]:
            problems.append(f"run.sh must set {var} exactly once, as {want!r}; found {sets}")
    cmd = [ln for ln in lines if ln.startswith("CMD=(")]
    if len(cmd) != 1 or cmd[0].count("--model ") != 1 or f"--model {MODEL_DIR} " not in cmd[0]:
        problems.append(f"the run.sh command must pass --model {MODEL_DIR} once")
    if re.search(r"--(?:tokenizer-)?revision\b", text):
        problems.append("run.sh pins a --revision, which a local model-dir does not have")
    if PREPARE_LINE not in lines or EXEC_LINE not in lines or lines.index(PREPARE_LINE) > lines.index(EXEC_LINE):
        problems.append("run.sh must run prepare_model_dir.py before it starts vLLM")
    if _names_model(text, nearest_model):
        problems.append(f"run.sh names the nearest model {nearest_model}")
    return problems


# ---- the run's results and the source bundles ---------------------------------------------------

def _json(path: Path, what: str) -> dict:
    try:
        data = safeio.read_json(safeio.resolved(path), max_bytes=MAX_RECORD_BYTES)
    except safeio.SafeReadError as exc:
        raise PackageError(f"{what} ({path}) is missing or not JSON: {exc.reason}") from None
    if not isinstance(data, dict):
        raise PackageError(f"{what} ({path}) is not a JSON object")
    return data


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class Source:
    """An installed v6 thin bundle of the nearest model: the model code, wheels and settings the
    package reuses."""
    path: Path
    manifest: dict

    @property
    def name(self) -> str:
        return self.manifest["name"]

    @property
    def chips(self) -> int:
        return int(self.manifest["device_count"])

    @property
    def weights_repo(self) -> str:
        return self.manifest["weights"]["repo_id"]

    @property
    def entry_cls(self) -> str:
        return self.manifest["entrypoint"]["cls"]

    def wheels(self) -> list[str]:
        deps = self.manifest["deps"]
        return list(deps.get("wheels") or []) + list(deps.get("models_wheels") or [])


def load_source(path) -> Source:
    path = Path(path)
    m = _json(path / "tt_kernel_manifest.json", "the source bundle's manifest")
    deps = m.get("deps") or {}
    if m.get("schema_version") != "6" or deps.get("kind") != "vllm":
        raise PackageError(f"{path} is not a v6 thin vLLM bundle")
    need = ["run.sh", "model.py", deps.get("requirements") or "requirements.txt",
            f"vllm_models/{m.get('name')}/vllm_metadata.json", *(deps.get("wheels") or []),
            *(deps.get("models_wheels") or [])]
    missing = [rel for rel in need if not (path / rel).is_file()]
    if missing:
        raise PackageError(f"the source bundle {path} lacks {missing}")
    return Source(path, m)


def find_sources(models_root, *, entry_cls: str, nearest_model: str) -> list[Source]:
    """Every installed v6 bundle under `models_root` that serves `nearest_model` with the model
    class `entry_cls`, fewest chips first. A directory that is not such a bundle is skipped."""
    found = []
    for mf in sorted(Path(models_root).expanduser().glob("*/*/tt_kernel_manifest.json")):
        try:
            s = load_source(mf.parent)
            if s.entry_cls == entry_cls and s.weights_repo == nearest_model:
                found.append(s)
        except (PackageError, KeyError, TypeError, ValueError):
            continue
    return sorted(found, key=lambda s: (s.chips, s.name))


# ---- which weights stage 2 served ----------------------------------------------------------------

REVISION = re.compile(r"[0-9a-f]{40}")
# A path inside a Hugging Face cache repo: .../models--<org>--<name>/(snapshots/<rev>|blobs)/...
HF_SNAPSHOT_PATH = re.compile(r"(?:^|/)models--(?P<org>[^/]+?)--(?P<name>[^/]+)/snapshots/"
                              r"(?P<rev>[0-9a-f]{40})/")
HF_BLOB_PATH = re.compile(r"^(?P<root>.*/models--(?P<org>[^/]+?)--(?P<name>[^/]+))/blobs/[^/]+$")
# The shared store of a newer huggingface_hub: a repo's blobs/<hash> is a link to <hub>/blobs/<2 hex>/<hash>.
HF_SHARED_BLOB_PATH = re.compile(r"^(?P<hub>.*)/blobs/[0-9a-f]{2}/[0-9a-f]{64}$")
MAX_LINK_HOPS = 40


def split_model_id(model_id) -> tuple[str, str | None]:
    """'org/name@<40 hex>' -> ('org/name', '<40 hex>'); 'org/name' -> ('org/name', None).

    Stage 0's delta.json writes `<repo>@<revision>` (the delta-triage skill). Anything after '@'
    that is not a 40-hex revision is kept as part of the repo, so a malformed id never matches."""
    text = model_id if isinstance(model_id, str) else ""
    repo, sep, rev = text.rpartition("@")
    if sep and REVISION.fullmatch(rev):
        return repo, rev
    return text, None


def _weight_origin(path: Path, prefer: str | None = None) -> tuple[str, set[str]] | None:
    """The repo and the candidate revisions one weight file in a model dir comes from, or None.

    Each hop of the file's symlink chain is read in turn. A hop inside `snapshots/<rev>/` names the
    revision directly (the layout the HF cache uses, and any link made into it). prepare_swap.py
    instead links each file to `os.path.realpath` of the snapshot file, which is a blob and names
    no revision. Then the candidates are the snapshots of that repo whose same-named file resolves
    to the same blob. Two revisions that share a blob hold identical bytes for that file.

    In the shared blob store the real path names no repo either. The candidates are then the snapshots
    of every `models--*` directory in the same hub whose same-named file resolves to that blob. A blob
    that two repos hold (a fine-tune keeps some of its base model's shards byte for byte) is read as
    `prefer`'s when `prefer` holds it."""
    hop = str(path)
    for _ in range(MAX_LINK_HOPS):
        if not os.path.islink(hop):
            break
        hop = os.path.normpath(os.path.join(os.path.dirname(hop), os.readlink(hop)))
        m = HF_SNAPSHOT_PATH.search(hop)
        if m:
            return f"{m['org']}/{m['name']}", {m["rev"]}
    real = os.path.realpath(path)
    m = HF_SNAPSHOT_PATH.search(real)
    if m:
        return f"{m['org']}/{m['name']}", {m["rev"]}
    m = HF_BLOB_PATH.match(real)
    if m is None:
        shared = HF_SHARED_BLOB_PATH.match(real)
        return _shared_blob_origin(Path(shared["hub"]), path.name, real, prefer) if shared else None
    snaps = Path(m["root"]) / "snapshots"
    revs = {s.name for s in (sorted(snaps.iterdir()) if snaps.is_dir() else [])
            if REVISION.fullmatch(s.name) and os.path.realpath(s / path.name) == real}
    return (f"{m['org']}/{m['name']}", revs) if revs else None


def _shared_blob_origin(hub: Path, name: str, real: str, prefer: str | None) -> tuple[str, set[str]] | None:
    """The repo and revisions in `hub` that hold the file `name` as the shared-store blob `real`."""
    found: dict[str, set[str]] = {}
    for repo_dir in sorted(hub.glob("models--*--*")):
        m = re.fullmatch(r"models--(?P<org>[^/]+?)--(?P<name>[^/]+)", repo_dir.name)
        snaps = repo_dir / "snapshots"
        if m is None or not snaps.is_dir():
            continue
        revs = {s.name for s in sorted(snaps.iterdir())
                if REVISION.fullmatch(s.name) and os.path.realpath(s / name) == real}
        if revs:
            found[f"{m['org']}/{m['name']}"] = revs
    if not found:
        return None
    repo = prefer if prefer in found else sorted(found)[0]
    return repo, found[repo]


def served_weights_problems(stage0_model, label, model_dir, graft_shard: str | None = None) -> list[str]:
    """Why the weights stage 2 served are not the ones stage 0 names; empty when they are.

    `stage0_model` is delta.json's `model` (`<repo>@<revision>`). `label` is swap-check.json's
    `new_model_id`, which the stage agent typed, and `model_dir` is the directory the test served.
    The label must name stage 0's repo, and if it carries an `@revision`, stage 0's revision. The
    weights decide the rest: every `*.safetensors` file in model_dir must link into stage 0's repo
    in the HF cache, all at one revision, and that revision must be stage 0's. `graft_shard` names the one
    file that is exempt: stage 2's graft shard, a regular file written by prepare_swap.py that links into no
    snapshot. The caller has already checked it against the graft's recorded source (`stage_2_graft_problems`);
    any other file that links into no snapshot still fails."""
    want_repo, want_rev = split_model_id(stage0_model)
    if not want_repo or want_rev is None:
        return [f"stage 0's model {stage0_model!r} names no revision; delta.json must give "
                "<repo>@<revision>"]
    label_repo, label_rev = split_model_id(label)
    if label_repo != want_repo:
        return [f"stage 2 served {label!r}; stage 0 names {stage0_model!r}. The repos differ."]
    if label_rev is not None and label_rev != want_rev:
        return [f"stage 2 served {label!r}; stage 0 names {stage0_model!r}. The revisions differ."]
    md = Path(model_dir) if isinstance(model_dir, str) and model_dir else None
    if md is None or not md.is_dir():
        return [f"stage 2's model_dir {model_dir!r} is not a directory, so the weights it served "
                "cannot be checked"]
    weights = sorted(md.glob("*.safetensors"))
    if not weights:
        return [f"stage 2's model_dir {md} has no *.safetensors file, so the weights it served "
                "cannot be checked"]
    weights = [w for w in weights if w.name != graft_shard]
    if not weights:
        return [f"stage 2's model_dir {md} has no *.safetensors file besides the graft shard, so the weights "
                "it served cannot be checked"]
    origins = {}
    for w in weights:
        origin = _weight_origin(w, want_repo)
        if origin is None:
            return [f"{w} links into no Hugging Face snapshot, so its revision is unknown; stage 0 "
                    f"names {stage0_model}"]
        origins[w.name] = origin
    repos = sorted({repo for repo, _ in origins.values()})
    if repos != [want_repo]:
        return [f"the weights in {md} come from {', '.join(repos)}; stage 0 names {stage0_model}"]
    common = set.intersection(*(revs for _, revs in origins.values()))
    if not common:
        found = "; ".join(f"{n}: {', '.join(sorted(revs))}" for n, (_, revs) in origins.items())
        return [f"the weights in {md} resolve to different revisions ({found}); stage 0 names "
                f"{stage0_model}"]
    if want_rev not in common:
        return [f"the weights in {md} are revision {', '.join(sorted(common))}; stage 0 names "
                f"{stage0_model}"]
    return []


REPORTED_FILE = "reported-results.json"
MAX_REPORTED = 60                    # rows
MAX_REPORTED_NAME, MAX_REPORTED_VALUE, MAX_REPORTED_NOTE, MAX_REPORTED_SOURCE = 100, 40, 300, 200


def read_reported(run_dir) -> tuple[tuple[Reported, ...], str]:
    """The scores the model's authors report, from `<run>/reported-results.json` (written by a person from the
    model's card; the harness does not parse cards): `{"source": "...", "rows": [{"name", "value", "note"?}]}`.
    A value may be a string or a number. Returns ((), "") when the file is absent; raises PackageError naming
    what is wrong. The package card shows these apart from the numbers the run measured."""
    path = Path(run_dir) / REPORTED_FILE
    if not path.exists():
        return (), ""
    return read_reported_file(path)


def read_reported_file(path) -> tuple[tuple[Reported, ...], str]:
    """`read_reported` for one named file: the same checks, so a file can be judged before it is installed
    in a run directory."""
    data = _json(Path(path), "the reported scores")
    source = data.get("source")
    if not isinstance(source, str) or not source.strip():
        raise PackageError(f"{REPORTED_FILE} needs a non-empty source (where the authors report the scores)")
    if len(source) > MAX_REPORTED_SOURCE:
        raise PackageError(f"{REPORTED_FILE}: the source is too long (over {MAX_REPORTED_SOURCE} characters)")
    rows = data.get("rows")
    if not isinstance(rows, list) or not rows:
        raise PackageError(f"{REPORTED_FILE} needs rows: a non-empty list")
    if len(rows) > MAX_REPORTED:
        raise PackageError(f"{REPORTED_FILE} has {len(rows)} rows; at most {MAX_REPORTED} are allowed")
    out, seen = [], set()
    for i, row in enumerate(rows, 1):
        value = row.get("value") if isinstance(row, dict) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = str(value)
        name, note = (row.get("name"), row.get("note", "")) if isinstance(row, dict) else (None, "")
        if not (isinstance(name, str) and name.strip() and isinstance(value, str) and value.strip()
                and isinstance(note, str)):
            raise PackageError(f"{REPORTED_FILE}: row {i} needs a name, a value (text or a number) and an "
                               "optional text note")
        for what, text, limit in (("name", name, MAX_REPORTED_NAME), ("value", value, MAX_REPORTED_VALUE),
                                  ("note", note, MAX_REPORTED_NOTE)):
            if len(text) > limit:
                raise PackageError(f"{REPORTED_FILE}: the {what} in row {i} is too long (over {limit} characters)")
        if name in seen:
            raise PackageError(f"{REPORTED_FILE}: {name!r} appears twice")
        seen.add(name)
        out.append(Reported(name, value, note))
    return tuple(out), source


@dataclass(frozen=True)
class RunFacts:
    run_dir: Path
    model_id: str
    revision: str
    nearest_model: str
    new_snapshot: Path
    hf_home: Path                    # the HF home stage 2 served with (it holds the drafter)
    source: Source                   # the bundle stage 2 served the new weights with
    base_config: tuple[Path, ...]    # the nearest model's config files stage 2 used
    passing_chips: frozenset[int]    # chip counts with a passing stage 4 configuration
    license_id: str
    drafter_off: bool = False        # stage 2 served with the speculative drafter off
    host_sampling: bool = False      # ... and with the host sampling
    sidecars: tuple = ()             # stage 0's sidecar files (the package does not serve them)
    reported: tuple = ()             # scores the authors report (reported-results.json), not measured here
    reported_source: str = ""
    base_snapshot: Path = Path()     # the nearest model's pinned snapshot stage 2 swapped from (the graft reads it)
    model_has_mtp: bool | None = None    # stage 0's finding; None when it did not record one
    graft: dict | None = None        # the graft.json this package carries (mtp.* tensors added at each start)
    stage_2_grafted: bool = False    # stage 2's own swap grafted them too
    graft_decision: str | None = None   # "graft" or "safe" when stage 2's graft trial ran and decided (orchard/graft_trial.py)
    vocab: tuple[int, int] | None = None   # (the model's vocabulary size, the nearest model's) for a vocabulary trim
    sampling_decision: dict | None = None  # stage 2's sampling decision (orchard/sampling_choice.py), None when it made none


def read_run(run_dir, graft_mtp: bool = False, no_graft_mtp: bool = False) -> RunFacts:
    """What stage 7 needs from stages 0, 2 and 4. Raises PackageError naming what is missing.

    `graft_mtp` asks for the nearest model's mtp.* tensors to be added at start (a graft.json), with the drafter and
    on-device sampling on. The package is also grafted when stage 2's graft trial chose the graft (stages/2/graft-decision.json),
    when stage 2's swap grafted, and when stage 7 already staged a grafted package (the boot check's finish step and the gate read
    that record). `no_graft_mtp` refuses all of those but the last: a package without the graft would not be the configuration
    stage 2 chose or stage 2's numbers describe."""
    if graft_mtp and no_graft_mtp:
        raise PackageError("--graft-mtp and --no-graft-mtp both given; they ask for opposite packages")
    run = Path(run_dir).resolve()
    delta = _json(run / "stages/0/delta.json", "stage 0's delta")
    if delta.get("path") != "weights-only":
        raise PackageError(f"stage 7 packages weights-only runs; stage 0 chose {delta.get('path')!r}")
    gate = gate_weights_swap(run / "stages/2", run)
    if not gate.ok:
        raise PackageError("stage 2's result does not pass its gate: " + "; ".join(gate.reasons))
    cfg = _json(run / "stages/2/swap_config.json", "stage 2's swap_config.json")
    # What was served is read from the weights the test loaded (swap-check.json's model_dir), and
    # the label the stage agent typed only has to agree with it (served_weights_problems).
    served = _json(run / "stages/2/evidence/swap-check.json", "stage 2's swap-check.json")
    nearest_model = split_model_id(delta.get("nearest_model"))[0]
    snap = Path(cfg.get("new_snapshot") or "")
    if not REVISION.fullmatch(snap.name) or not snap.is_dir():
        raise PackageError(f"stage 2's new_snapshot {str(snap)!r} is not a pinned snapshot directory")
    # A run that grafted in stage 2 has one regular file among the links: the graft shard. It is exempt from the
    # provenance check only when its bytes are the nearest model's mtp.* tensors, as the record says.
    graft_record, exempt = _stage_2_graft_record(run), None
    if graft_record is not None:
        md_served = Path(served.get("model_dir") or "")
        bad = stage_2_graft_problems(md_served, graft_record, nearest_model, Path(cfg.get("base_snapshot") or ""), snap)
        if bad:
            raise PackageError("stage 2's graft: " + "; ".join(bad))
        exempt = GRAFT_SHARD
    problems = served_weights_problems(delta.get("model"), served.get("new_model_id"),
                                       served.get("model_dir"), graft_shard=exempt)
    if problems:
        raise PackageError("; ".join(problems))
    # From here on the ids are bare repo ids: package-thin, the manifest and the card take the
    # revision separately.
    model_id, revision = split_model_id(delta.get("model"))
    if snap.name != revision:
        raise PackageError(f"stage 2's new_snapshot is revision {snap.name}; stage 0 names "
                           f"{delta.get('model')}")
    source = load_source(cfg.get("bundle_dir") or "")
    if source.weights_repo != nearest_model:
        raise PackageError(f"stage 2's bundle serves {source.weights_repo}; the nearest model is "
                           f"{delta.get('nearest_model')}")
    md = run / "stages/2/model-dir"
    base = tuple(md / n for n in BASE_CONFIG_FILES if (md / n).is_file() and not (md / n).is_symlink())
    if md / "config.json" not in base:
        raise PackageError(f"{md}/config.json (the nearest model's config) is missing")
    configs = _json(run / "stages/4/result.json", "stage 4's result").get("configs") or []
    passing = frozenset(c["chips"] for c in configs if isinstance(c, dict) and c.get("pass") is True
                        and isinstance(c.get("chips"), int) and not isinstance(c.get("chips"), bool))
    license_id = read_license(snap)
    if not license_id:
        raise PackageError(f"the new model's license is not in {snap}/README.md; stage 7 stages no "
                           "package without it")
    drafter_off, host_sampling = served_drafter_state(run)
    reported, reported_source = read_reported(run)
    model_has_mtp = delta_model_has_mtp(delta)
    facts = RunFacts(run, model_id, revision, nearest_model, snap,
                     Path(cfg.get("hf_home") or ""), source, base, passing, license_id,
                     drafter_off, host_sampling, tuple(delta.get("sidecars") or ()), reported, reported_source,
                     Path(cfg.get("base_snapshot") or ""), model_has_mtp, None, _stage_2_grafted(run), vocab=stage_2_vocab(run))
    decision = read_decision(run / "stages/2")
    decided = decision is not None and decision["choice"] == "graft"
    # A drafter-off sampling decision supersedes what stage 2's own build suggests: the sampling trial builds its variant without the graft, so
    # a stage 2 that grafted (or an earlier staged package that did) no longer describes the package. Only an explicit request conflicts with it.
    off = sampling_choice.drafter_off_chosen(run)
    if no_graft_mtp and (decided or (facts.stage_2_grafted and not off)):
        why = ("stage 2's graft trial chose the graft and stage 4's bundle configurations were built with it, so a package without it "
               "would not be the configuration those results describe; run again with --no-graft-trial to skip the trial and decide "
               "on the safe configuration from the start" if decided else
               "stage 2 itself served with the graft, so its numbers describe a grafted package")
        raise PackageError(f"--no-graft-mtp: {why}")
    trial_ran = decision is not None and decision.get("trial") is not None
    facts = replace(facts, graft_decision=decision["choice"] if trial_ran else None,
                    sampling_decision=sampling_choice.read_decision(run / "stages/2"))
    if off and (graft_mtp or decided):
        raise PackageError("stage 2's sampling decision chose the drafter-off configuration (the model needs sampling), and a graft keeps the "
                           "drafter on; the two ask for opposite packages. Run again with --no-graft-trial, or with --sampling not-needed")
    if not no_graft_mtp and not off and (graft_mtp or decided or facts.stage_2_grafted or _package_grafted(run)):
        # The drafter and on-device sampling are on in a grafted package, whatever stage 2's run.sh says.
        facts = replace(facts, drafter_off=False, host_sampling=False,
                        graft=plan_graft(nearest_model, facts.base_snapshot, snap))
    return facts


@dataclass(frozen=True)
class Profile:
    chips: int
    source: Source
    required: bool                   # the profile stage 2 checked; stage 7 boots it


def plan_profiles(facts: RunFacts, others: list[Source]) -> tuple[list[Profile], list[dict]]:
    """The profiles to stage, and the chip counts stage 4 passed that get no package, with why.

    The required profile uses stage 2's bundle. Another installed bundle of the nearest model is
    an optional profile when stage 4 passed its chip count."""
    req = facts.source
    if req.chips not in facts.passing_chips:
        raise PackageError(f"stage 4 has no passing {req.chips}-chip configuration")
    profiles, skipped = [Profile(req.chips, req, True)], []
    for s in others:
        if any(p.chips == s.chips for p in profiles):
            continue
        if s.chips in facts.passing_chips:
            profiles.append(Profile(s.chips, s, False))
        else:
            skipped.append({"chips": s.chips, "reason": f"stage 4 has no passing {s.chips}-chip "
                                                         f"configuration (bundle {s.name} exists)"})
    for chips in sorted(facts.passing_chips - {p.chips for p in profiles}):
        skipped.append({"chips": chips, "reason": f"no v6 bundle of {facts.nearest_model} for "
                                                  f"{chips} chips is installed"})
    return sorted(profiles, key=lambda p: p.chips), sorted(skipped, key=lambda s: s["chips"])


def bundle_name(model_id: str, source_name: str) -> str:
    """The new model's name plus the source bundle's board suffix: hemmingway-1-p300."""
    return f"{model_id.split('/')[-1].lower()}-{source_name.rsplit('-', 1)[-1]}"


# ---- tt-model package-thin, with --out only ------------------------------------------------------

THIN_VALUE_OPTIONS = frozenset({
    "--model-py", "--kind", "--app", "--requirements", "--plugin-wheel", "--ops-wheel",
    "--models-wheel", "--vllm-wheel", "--vllm-version", "--arch", "--arch-name", "--main-class",
    "--metadata", "--weights", "--weights-revision", "--mesh", "--device-count", "--python",
    "--tt-metal-version", "--max-num-seqs", "--block-size", "--max-model-len", "--env", "--name",
    "--out",
    # the canning step (orchard/can.py) also passes these
    "--extra-code", "--verify", "--readme", "--provenance", "--mesh-fabric", "--constraints"})
THIN_FLAGS = frozenset({"--no-vllm"})     # options with no value; the canning step passes this one for kind tt-dit-server


def assert_no_publish(argv) -> None:
    """Refuse any tt-model call that could upload or list something.

    `package-thin` pushes when it is given a repo id as a positional argument, and `--public` and
    `--publish` change visibility or list the repo. Stage 7 passes only the options it knows, each
    with a value, and `--out`."""
    argv = [str(a) for a in argv]
    if argv[:2] != ["tt-model", "package-thin"]:
        raise PackageError(f"stage 7 runs only `tt-model package-thin`, not {argv[:2]}")
    if "--out" not in argv:
        raise PackageError("tt-model package-thin needs --out, so it stages and does not push")
    i = 2
    while i < len(argv):
        a = argv[i]
        if a in THIN_VALUE_OPTIONS and i + 1 < len(argv):
            i += 2
        elif a in THIN_FLAGS:
            i += 1
        elif a.startswith("-"):
            raise PackageError(f"option {a} is not one stage 7 passes (--public and --publish "
                               "change visibility or list the repo)")
        else:
            raise PackageError(f"{a!r} is a positional argument; package-thin pushes to a repo id "
                               "given that way")


def thin_argv(source: Source, *, model_id: str, revision: str, name: str, out: Path,
              drafter_off: bool = False) -> list[str]:
    m, s = source.manifest, source.path
    deps, res = m["deps"], m["resources"]
    plugin = [w for w in deps.get("wheels") or [] if Path(w).name.startswith("vllm_tt_plugin")]
    ops = [w for w in deps.get("wheels") or [] if w not in plugin]
    argv = ["tt-model", "package-thin", "--model-py", s / "model.py", "--kind", "vllm",
            "--requirements", s / (deps.get("requirements") or "requirements.txt"),
            *[x for w in plugin for x in ("--plugin-wheel", s / w)],
            *[x for w in ops for x in ("--ops-wheel", s / w)],
            *[x for w in deps.get("models_wheels") or [] for x in ("--models-wheel", s / w)],
            "--metadata", s / "vllm_models" / m["name"] / "vllm_metadata.json",
            "--weights", model_id, "--weights-revision", revision,
            "--arch", m["arch"], "--mesh", m["mesh"]["topology"], "--device-count", m["device_count"],
            "--python", deps.get("python") or "3.12", "--vllm-version", deps["vllm"]["version"],
            "--max-num-seqs", res["max_num_seqs"], "--block-size", res["block_size"],
            "--max-model-len", res["max_model_len"], "--name", name,
            *[x for k, v in serving_env(m.get("env") or {}, drafter_off=drafter_off).items()
               for x in ("--env", f"{k}={v}")],
            "--out", out]
    return [str(a) for a in argv]


def run_logged(argv, *, log: Path, timeout: float, env: dict | None = None) -> int | None:
    """Run one supervisor command with its output appended to `log`. Returns the exit code, or None
    on a timeout. It runs in its own session, and a timeout, a signal or an error kills the whole
    session, so nothing it started outlives the stage."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as out:
        try:
            proc = subprocess.Popen([str(a) for a in argv], stdout=out, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, env=env, start_new_session=True)
        except FileNotFoundError as exc:
            out.write(f"not found: {exc}\n".encode())
            return 127
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_session(proc)
            return None
        except BaseException:
            _kill_session(proc)
            raise


def _kill_session(proc) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def _tail(log: Path, n: int = 20) -> str:
    try:
        return "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def stage_profile(profile: Profile, facts: RunFacts, out: Path, *, env: dict | None = None) -> dict:
    """Run package-thin for one profile into `out` (which must not exist) and apply the edits."""
    src = profile.source
    name = bundle_name(facts.model_id, src.name)
    out = Path(out)
    if out.exists():
        raise PackageError(f"{out} already exists; stage 7 stages into a fresh directory")
    argv = thin_argv(src, model_id=facts.model_id, revision=facts.revision, name=name, out=out,
                     drafter_off=facts.drafter_off)
    assert_no_publish(argv)
    log = out.parent / f"{name}.package-thin.log"
    rc = run_logged(argv, log=log, timeout=PACKAGE_THIN_TIMEOUT_S, env=env)
    if rc != 0:
        raise PackageError(f"tt-model package-thin for {name} exited {rc}; the end of {log.name}:\n"
                           + _tail(log))
    run_sh = out / "run.sh"
    extra = extra_args_from((src.path / "run.sh").read_text(encoding="utf-8"))
    if facts.host_sampling:
        extra = drop_device_sampling(extra)
    text = splice_extra_args(run_sh.read_text(encoding="utf-8"), extra)
    text = wire_weights(text, facts.model_id)
    problems = weights_wiring_problems(text, nearest_model=facts.nearest_model)
    if problems:
        raise PackageError(f"{name}/run.sh: " + "; ".join(problems))
    manifest_env = _json(out / "tt_kernel_manifest.json", "the staged manifest").get("env") or {}
    problems = serving_problems(text, manifest_env, drafter_off=facts.drafter_off,
                                host_sampling=facts.host_sampling, model_has_mtp=facts.model_has_mtp,
                                graft=facts.graft is not None)
    if problems:
        raise PackageError(f"{name}: " + "; ".join(problems))
    run_sh.write_text(text, encoding="utf-8")
    mpath = out / "tt_kernel_manifest.json"
    m = _json(mpath, "the staged manifest")
    if m.get("weights", {}).get("repo_id") != facts.model_id or m["weights"].get("revision") != facts.revision:
        raise PackageError(f"the staged manifest's weights are {m.get('weights')}; expected "
                           f"{facts.model_id} at {facts.revision}")
    m.setdefault("producer", {})["hostname"] = "redacted"
    mpath.write_text(json.dumps(m, indent=2) + "\n", encoding="utf-8")
    (out / "base_config").mkdir()
    for f in facts.base_config:
        shutil.copyfile(f, out / "base_config" / f.name)
    shutil.copyfile(TEMPLATES / "prepare_model_dir.py", out / "prepare_model_dir.py")
    if facts.graft:
        (out / GRAFT_FILE).write_text(json.dumps(facts.graft, indent=2) + "\n", encoding="utf-8")
    for f in ("run.sh", "install.sh", "prepare_model_dir.py"):
        (out / f).chmod(0o755)
    for w in sorted((out / "wheels").glob("*.whl")):
        theirs = src.path / "wheels" / w.name
        if not theirs.is_file() or sha256(theirs) != sha256(w):
            raise PackageError(f"{name}/wheels/{w.name} is not byte-identical to the source bundle's")
    return {"chips": profile.chips, "name": name, "required": profile.required, "source": src.name,
            "source_manifest_sha256": sha256(src.path / "tt_kernel_manifest.json"),
            "mesh": m["mesh"]["topology"]}


# ---- the boot check: an installed copy, served on a leased board ---------------------------------

AUX_REPO = re.compile(r"^([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)(?:@[0-9a-f]{40})?$")


def aux_repos(env: dict, *, nearest_model: str) -> list[str]:
    """Hugging Face repos the bundle's environment names (such as the drafter in DFLASH_WEIGHTS),
    without the nearest model, whose weights the package must never load."""
    found = {m.group(1) for v in env.values() if (m := AUX_REPO.match(str(v)))}
    return sorted(found - {nearest_model})


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _repo_dir(hub: Path, repo: str) -> Path:
    org, name = repo.split("/", 1)
    return hub / f"models--{org}--{name}"


def prepare_verify(staged: Path, verify_dir: Path, facts: RunFacts, *, env: dict | None = None) -> dict:
    """Install a copy of a staged package and lay out its boot check. Returns hw_test.json's content
    plus what was linked. The staged directory itself is never installed into or served."""
    verify_dir = Path(verify_dir)
    verify_dir.mkdir(parents=True)
    bundle = verify_dir / "bundle"
    shutil.copytree(staged, bundle, symlinks=True)
    log = verify_dir / "install.log"
    rc = run_logged(["bash", bundle / "install.sh"], log=log, timeout=PACKAGE_INSTALL_TIMEOUT_S,
                    env=env)
    if rc != 0:
        raise PackageError(f"install.sh of the package copy exited {rc}; the end of install.log:\n"
                           + _tail(log))
    hub = verify_dir / "hf" / "hub"
    hub.mkdir(parents=True)
    _repo_dir(hub, facts.model_id).symlink_to(facts.new_snapshot.parents[1])
    linked, missing = [facts.model_id], []
    manifest = _json(bundle / "tt_kernel_manifest.json", "the staged manifest")
    for repo in aux_repos(manifest.get("env") or {}, nearest_model=facts.nearest_model):
        theirs = _repo_dir(facts.hf_home / "hub", repo)
        if theirs.is_dir():
            _repo_dir(hub, repo).symlink_to(theirs)
            linked.append(repo)
        else:
            missing.append(repo)
    if facts.graft:       # the package reads the nearest model's mtp.* shard from the Hugging Face cache
        theirs = _repo_dir(facts.hf_home / "hub", facts.nearest_model)
        if theirs.is_dir():
            _repo_dir(hub, facts.nearest_model).symlink_to(theirs)
            linked.append(facts.nearest_model)
        else:
            missing.append(facts.nearest_model)
    (verify_dir / "verify_config.json").write_text(json.dumps({
        "run_dir": str(facts.run_dir), "bundle": str(bundle), "port": free_port(),
        "model_id": facts.model_id, "revision": facts.revision, "hf_home": str(hub.parent),
        "health_timeout_s": PACKAGE_HEALTH_TIMEOUT_S}, indent=2))
    shutil.copyfile(TEMPLATES / "verify_bundle.py", verify_dir / "verify_bundle.py")
    shutil.copyfile(SWAP_TEMPLATES / "serve_and_compare.py", verify_dir / "serve_and_compare.py")
    script = os.path.relpath(verify_dir / "verify_bundle.py", facts.run_dir)
    return {"command": f"{shlex.quote(sys.executable)} {script}",
            "deadline_s": PACKAGE_VERIFY_DEADLINE_S, "hf_linked": linked, "hf_missing": missing}


# ---- the evidence a card cites ships with the package -------------------------------------------

EVIDENCE_MAX_BYTES = 5_000_000


def redact_evidence(text: str, *, run_dir: Path, hostname: str | None, home: str | None) -> str:
    """The run directory, the home directory (any user's) and the host name replaced by labels. Tokens
    are not touched: a token in evidence stops the package at the scrub."""
    text = text.replace(str(run_dir), "<RUN_DIR>")
    if home and home != "/":
        text = text.replace(home, "<HOME>")
    text = HOME_PATH.sub("<HOME>", text)
    # tt-model installs a bundle under <org>/<name>; the org there is the operator's namespace.
    text = re.sub(r"<HOME>/\.cache/tt-model/models/[^/\s]+/", "<TT_MODEL_MODELS>/", text)
    if hostname and hostname != "localhost":
        text = re.sub(rf"\b{re.escape(hostname)}\b", "<HOST>", text)
    return text


def copy_evidence(facts: RunFacts, out: Path, card: str, *, hostname: str | None = None) -> list[str]:
    """Copy every file the card's measured numbers cite into out/evidence/<run-relative path>, redacted.
    Returns the paths. A cited file that is missing or over EVIDENCE_MAX_BYTES is an error."""
    dest = out / "evidence"
    if dest.exists():
        shutil.rmtree(dest)
    paths = card_evidence_paths(card)
    for rel in paths:
        src = inside(facts.run_dir, rel)
        if src is None:
            raise PackageError(f"the card cites {rel}, which is not a file inside the run directory")
        if src.stat().st_size > EVIDENCE_MAX_BYTES:
            raise PackageError(f"{rel} is too large to ship ({src.stat().st_size} bytes; the limit is "
                               f"{EVIDENCE_MAX_BYTES})")
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        text = src.read_text(encoding="utf-8", errors="replace")
        target.write_text(redact_evidence(text, run_dir=facts.run_dir, hostname=(
            socket.gethostname() if hostname is None else hostname), home=os.path.expanduser("~")),
            encoding="utf-8")
    return paths


# ---- cards, publish commands and the stage's record ----------------------------------------------

PUBLISH_FILE = "PUBLISH_COMMANDS.txt"
PUBLISH_LINE = re.compile(r"^hf upload --repo-type model --private [A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+ "
                          r"stages/7/package/[A-Za-z0-9_.-]+ \.$")


def publish_commands(profiles: list[dict], *, namespace: str, license_id: str) -> str:
    """The operator's publish commands, as text. Only a boot-checked profile gets a live line."""
    nc = " (non-commercial)" if non_commercial(license_id) else ""
    lines = ["# Publish commands for the packages stage 7 staged. The run never ran them.",
             "# Read RESULTS.md, RISKS.md and each package's README.md before running any of them.",
             "# Run them from the run directory. `hf upload --private` creates each repo private.",
             "# The card's front matter sets the repo's license and the tags `tt-model search` uses.",
             f"# License: {license_id}{nc}. Keep the card's license section as it is.",
             "# Making a repo public, or listing it in the tt-model catalog, is a separate decision.",
             "# `tt-model package-thin <repo>` rebuilds the bundle without stage 7's run.sh edits,",
             "# so it is not used to publish these packages.", ""]
    for p in sorted(profiles, key=lambda p: p["chips"]):
        line = (f"hf upload --repo-type model --private {namespace}/{p['name']} "
                f"stages/7/package/{p['name']} .")
        if p.get("verified"):
            lines += [f"# {p['chips']} chips ({p['mesh']}), boot-checked in stage 7:", line, ""]
        else:
            lines += [f"# {p['chips']} chips ({p['mesh']}), NOT boot-checked. Boot it before "
                      "publishing:", f"# {line}", ""]
    return "\n".join(lines)


def publish_problems(text: str) -> list[str]:
    problems = []
    live = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("#")]
    if not live:
        problems.append(f"{PUBLISH_FILE} has no command for a boot-checked package")
    for ln in live:
        if not PUBLISH_LINE.match(ln):
            problems.append(f"{PUBLISH_FILE} line {ln!r} is not a private hf upload of a staged package")
    for flag in ("--public", "--publish"):
        if re.search(rf"(?<![\w-]){flag}\b", text):
            problems.append(f"{PUBLISH_FILE} uses {flag}")
    return problems


def _rel(run_dir: Path, path: Path) -> str:
    return os.path.relpath(path, run_dir)


BENCHMARK_NUMBERS_FILE = "stages/7/benchmark-numbers.json"


def read_benchmark_numbers(run_dir) -> tuple[Number, ...]:
    """The measured rows `tt-orchard package-card` left for the card: `{"rows": [{"name", "value", "unit",
    "evidence"}]}` under stages/7. Absent means none. Each cited file must exist in the run."""
    run = Path(run_dir)
    path = run / BENCHMARK_NUMBERS_FILE
    if not path.is_file():
        return ()
    rows = _json(path, "the benchmark numbers").get("rows")
    if not isinstance(rows, list):
        raise PackageError(f"{BENCHMARK_NUMBERS_FILE} needs a list of rows")
    out = []
    for i, r in enumerate(rows, 1):
        ok = (isinstance(r, dict) and isinstance(r.get("name"), str) and r["name"].strip()
              and isinstance(r.get("unit"), str) and isinstance(r.get("evidence"), list) and r["evidence"]
              and all(isinstance(e, str) for e in r["evidence"])
              and isinstance(r.get("value"), (int, float)) and not isinstance(r.get("value"), bool))
        if not ok:
            raise PackageError(f"{BENCHMARK_NUMBERS_FILE}: row {i} needs a name, a number, a unit and evidence")
        for e in r["evidence"]:
            if not (run / e).is_file():
                raise PackageError(f"{BENCHMARK_NUMBERS_FILE}: row {i} cites {e}, which is not in the run")
        out.append(Number(r["name"], r["value"], r["unit"], "measured", tuple(r["evidence"])))
    return tuple(out)


GRAFT_TRIAL_ROWS = (("decode speed, safe configuration (graft trial, median of four prompts)", "safe", "decode_tok_s", "tokens/s"),
                    ("decode speed, grafted drafter (graft trial, median of four prompts)", "variant", "decode_tok_s", "tokens/s"),
                    ("time to first token, safe configuration (graft trial, median of four prompts)", "safe", "ttft_s", "s"),
                    ("time to first token, grafted drafter (graft trial, median of four prompts)", "variant", "ttft_s", "s"),
                    ("top1 agreement with the CPU reference, grafted drafter (graft trial)", "variant", "top1_agreement", "fraction"))


ACCEPTANCE_ROW = "drafter acceptance, grafted drafter (graft trial: accepted over drafted tokens)"


def graft_trial_numbers(run_dir) -> tuple[Number, ...]:
    """The rows stage 2's graft trial measured, for the card: the speed of the safe configuration and of the grafted variant, and the
    variant's agreement with the CPU reference. A row appears when the trial's result holds a positive number for it. The values are
    the ones in stages/2/graft-trial/trial-result.json (the script rounds them), and the decision file is the second evidence file."""
    run = Path(run_dir)
    decision = read_decision(run / "stages/2")
    if decision is None or decision.get("trial") is None:
        return ()
    try:
        trial = json.loads((run / "stages/2" / TRIAL_DIR / TRIAL_RESULT).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    evidence = (f"stages/2/{TRIAL_DIR}/{TRIAL_RESULT}", f"stages/2/{DECISION_FILE}")
    out = []
    for name, side, key, unit in GRAFT_TRIAL_ROWS:
        v = (trial.get(side) or {}).get(key) if isinstance(trial, dict) and isinstance(trial.get(side), dict) else None
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0:
            out.append(Number(name, v, unit, "measured", evidence))
    rate = accept_rate((trial.get("variant") or {}).get("drafter")) if isinstance(trial, dict) and isinstance(trial.get("variant"), dict) else None
    if rate is not None:              # 0 is a measurement here, so this row is not held to the "positive" rule above
        out.append(Number(ACCEPTANCE_ROW, rate, "fraction", "measured", evidence))
    return tuple(out)


SAMPLING_SWAP_CHECK = "stages/2/evidence/swap-check.json"       # the default source; `sampling_source` follows stage 2's decisions
SAMPLING_SUMMARY = "stages/2/sampling-summary.json"
SAMPLING_ROWS = (("decode speed, greedy request (stage 2 sampling probe)", "greedy_tokens_per_s", "tokens/s"),
                 ("decode speed, sampled request (stage 2 sampling probe)", "sampled_tokens_per_s", "tokens/s"),
                 ("drafter acceptance, greedy request (stage 2 sampling probe)", "greedy_drafter_accept_rate", "fraction"),
                 ("drafter acceptance, sampled request (stage 2 sampling probe)", "sampled_drafter_accept_rate", "fraction"))


def sampling_source(run_dir) -> str:
    """The run-relative path of the swap check the card's sampling rows are derived from: the configuration the package serves
    (sampling_choice.probe_source)."""
    return "stages/2/" + sampling_choice.probe_source(Path(run_dir) / "stages" / "2")


def _rounded(value, places):
    return round(value, places) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def sampling_summary(run_dir) -> dict | None:
    """What stage 2's sampling probe found, derived from the raw requests in `stages/2/evidence/swap-check.json` (issue 107), or None when the file has
    no probe record, the record was not run, or it is malformed. Speeds and the drafter's accepted share are recomputed here
    (orchard/sampling_probe.py) and rounded to what the card shows; a summary field written into the record is never read."""
    try:
        report = json.loads((Path(run_dir) / sampling_source(run_dir)).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(report, dict):
        return None
    s = sampling_probe.summarize(report.get("sampling_probe"))
    if s["card_sentence"] is None or s["verdict"] in ("not-run", "malformed"):
        return None
    g, sm = s["greedy"] or {}, s["sampled"] or {}
    serving = report.get("serving") if isinstance(report.get("serving"), dict) else {}
    return {"label": "measured", "derived_from": sampling_source(run_dir), "verdict": s["verdict"], "card_sentence": s["card_sentence"],
            "greedy_tokens_per_s": _rounded(g.get("tokens_per_s"), 1), "sampled_tokens_per_s": _rounded(sm.get("tokens_per_s"), 1),
            "greedy_drafter_accept_rate": _rounded(g.get("drafter_accept_rate"), 3),
            "sampled_drafter_accept_rate": _rounded(sm.get("drafter_accept_rate"), 3),
            "serving": {"drafter": str(serving.get("drafter") or "not recorded"), "sampling": str(serving.get("sampling") or "not recorded")}}


def write_sampling_summary(run_dir) -> dict | None:
    """Write `stages/2/sampling-summary.json`, the file the card's sampling rows cite first, and return its content (None: nothing is written)."""
    summary = sampling_summary(run_dir)
    if summary is not None:
        _write_json(Path(run_dir) / SAMPLING_SUMMARY, summary)
    return summary


def sampling_summary_problems(run_dir) -> list[str]:
    """Why the summary file is not what the raw record gives, for the gate: the file is missing or differs from `sampling_summary`. Nothing when the
    run has no usable probe record."""
    expected = sampling_summary(run_dir)
    if expected is None:
        return []
    try:
        have = json.loads((Path(run_dir) / SAMPLING_SUMMARY).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return [f"{SAMPLING_SUMMARY} is missing or unreadable; the card's sampling rows cite it"]
    if have != expected:
        return [f"{SAMPLING_SUMMARY} is not the summary derived from the sampling probe's raw requests in {sampling_source(run_dir)}"]
    return []


def sampling_numbers(run_dir) -> tuple[Number, ...]:
    """The card's sampling rows: decode speed and the drafter's accepted share of a greedy and of a sampled request, from the derived summary. A row
    appears when the summary holds a number for it; a share of 0 is a measurement, a speed of 0 is not."""
    summary = sampling_summary(run_dir)
    if summary is None:
        return ()
    evidence = (SAMPLING_SUMMARY, sampling_source(run_dir))
    out = []
    for name, key, unit in SAMPLING_ROWS:
        v = summary.get(key)
        if v is not None and (v > 0 or unit == "fraction"):
            out.append(Number(name, v, unit, "measured", evidence))
    return tuple(out)


PROBE_RATE_ROWS = tuple(name for name, _, unit in SAMPLING_ROWS if "drafter acceptance" in name)


def acceptance_sources(numbers) -> tuple[str, ...]:
    """Which steps measured the drafter's acceptance, from the measured rows in `numbers`: the graft trial's grafted-drafter row, and the sampling
    probe's greedy and sampled request rows. The card's Not measured list and its limitation lines follow this."""
    measured = {n.name for n in numbers if n.label == "measured"}
    out = []
    if ACCEPTANCE_ROW in measured:
        out.append("graft trial")
    if measured & set(PROBE_RATE_ROWS):
        out.append("sampling probe")
    return tuple(out)


def card_numbers(facts: RunFacts, profile: dict, verify: dict | None) -> list[Number]:
    """The numbers one package's card may show. Only the boot-checked profile shows the run's
    measurements, because they were taken on that profile's chip count."""
    run = facts.run_dir
    if not profile["required"]:
        return [Number("top1 agreement with the CPU reference, this package", None, "fraction",
                       "TODO", ())]
    s2 = _json(run / "stages/2/result.json", "stage 2's result")
    ev2 = ("stages/2/result.json", *s2.get("evidence", []))
    nums = [Number(f"top1 agreement with the CPU reference, stage 2 ({profile['chips']} chips, "
                   f"bundle {profile['source']})", s2["top1_agreement"], "fraction", "measured", ev2),
            Number("server ready after start, stage 2 (empty tensor cache)", s2["server_ready_s"],
                   "s", "measured", ev2)]
    s6_path = run / "stages/6/result.json"
    if s6_path.is_file():
        for n in _json(s6_path, "stage 6's result").get("numbers") or []:
            measured = n.get("label") == "measured"
            nums.append(Number(f"{n['name']} (stage 6)", n.get("value") if measured else None,
                               n["unit"], "measured" if measured else "TODO",
                               ("stages/6/result.json", *n.get("evidence", [])) if measured else ()))
    if verify is None:
        nums.append(Number("top1 agreement with the CPU reference, this package", None, "fraction",
                           "TODO", ()))
    else:
        ev7 = tuple(verify["evidence"])
        nums += [Number("top1 agreement with the CPU reference, this package (stage 7)",
                        verify["top1_agreement"], "fraction", "measured", ev7),
                 Number("server ready after start, this package (stage 7, fresh install, empty "
                        "tensor cache)", verify["server_ready_s"], "s", "measured", ev7)]
    nums += graft_trial_numbers(run)
    nums += sampling_numbers(run)
    nums += read_benchmark_numbers(run)
    return nums


def write_card(out: Path, facts: RunFacts, profile: dict, *, namespace: str,
               verify: dict | None) -> None:
    m = _json(out / "tt_kernel_manifest.json", "the staged manifest")
    drafters = aux_repos(m.get("env") or {}, nearest_model=facts.nearest_model)
    not_measured = ["a download of the weights through `tt-model pull`, and a boot of the package "
                    "from the Hub"]
    sampling = write_sampling_summary(facts.run_dir) if profile["required"] else None      # before the numbers: the rows cite this file
    numbers = tuple(card_numbers(facts, profile, verify))
    acceptance_from = " and ".join(acceptance_sources(numbers))
    acceptance_measured = bool(acceptance_from)
    if drafters and not acceptance_measured:
        not_measured.insert(0, "the drafter's acceptance rate on this model")
    run_sh_path = out / "run.sh"
    use = read_use_facts(facts.new_snapshot, run_sh_path.read_text(encoding="utf-8") if run_sh_path.is_file() else "",
                         facts.run_dir, identity=(facts.model_id, facts.revision), measured=bool(profile["required"]))
    facts_card = CardFacts(
        name=profile["name"], namespace=namespace, model_id=facts.model_id, revision=facts.revision,
        nearest_model=facts.nearest_model, source_name=profile["source"],
        license_id=facts.license_id, chips=profile["chips"], mesh=m["mesh"]["topology"],
        arch=m["arch"], max_model_len=m["resources"]["max_model_len"],
        max_num_seqs=m["resources"]["max_num_seqs"], drafter=drafters[0] if drafters else None,
        verified=verify is not None and profile["required"],
        numbers=numbers, not_measured=tuple(not_measured), acceptance_measured=acceptance_measured, acceptance_from=acceptance_from,
        drafter_off=facts.drafter_off, host_sampling=facts.host_sampling,
        sidecars=tuple(s["file"] for s in facts.sidecars),
        base_files=tuple(sorted(f.name for f in facts.base_config)),
        sampling_from_base=any(f.name == "generation_config.json" for f in facts.base_config),
        reported=facts.reported, reported_source=facts.reported_source,
        graft=facts.graft, graft_decision=facts.graft_decision, use=use, vocab=facts.vocab, sampling=sampling,
        sampling_decision=facts.sampling_decision if profile["required"] else None,
        drafter_off_by_sampling=bool(facts.sampling_decision and facts.sampling_decision.get("choice") == "drafter-off"),
        sampling_trial_chips=facts.source.chips,
        sampling_first_verdict=((facts.sampling_decision or {}).get("numbers") or {}).get("served_verdict"),
        stage_2_without_graft=(facts.graft is not None and not facts.stage_2_grafted and facts.graft_decision != "graft"))
    (out / "README.md").write_text(render_card(facts_card), encoding="utf-8")


def _write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def stage_all(run_dir, stage_dir, *, namespace: str, models_root, env: dict | None = None,
              hostname: str | None = None, graft_mtp: bool = False, no_graft_mtp: bool = False) -> dict:
    """Stage every profile, scrub each one (a hit stops the stage before anything is installed),
    install a copy of the required profile and write hw_test.json, handoff.json and package.json.

    With `graft_mtp`, every profile carries a graft.json and keeps the drafter and on-device sampling on; the
    boot check then serves the package that way, whatever stage 2 served."""
    facts = read_run(run_dir, graft_mtp=graft_mtp, no_graft_mtp=no_graft_mtp)
    stage_dir = Path(stage_dir)
    others = find_sources(models_root, entry_cls=facts.source.entry_cls,
                          nearest_model=facts.nearest_model)
    profiles, skipped = plan_profiles(facts, others)
    root = stage_dir / "package"
    root.mkdir(parents=True)
    records = []
    for p in profiles:
        out = root / bundle_name(facts.model_id, p.source.name)
        rec = stage_profile(p, facts, out, env=env)
        write_card(out, facts, rec, namespace=namespace, verify=None)
        hits = scrub_package(out, hostname=hostname, namespace=namespace)
        if hits:
            raise PackageError(f"scrub of {rec['name']}: " + "; ".join(hits))
        records.append({**rec, "dir": _rel(facts.run_dir, out), "verified": False})
    req = next(r for r in records if r["required"])
    test = prepare_verify(facts.run_dir / req["dir"], stage_dir / "verify", facts, env=env)
    _write_json(stage_dir / "hw_test.json",
                {"command": test["command"], "deadline_s": test["deadline_s"]})
    _write_json(stage_dir / "handoff.json", {
        "goal": f"package {facts.model_id} as a v6 thin bundle", "stage": 7,
        "evidence": [_rel(facts.run_dir, stage_dir / "package.json")],
        "next_action": f"boot the installed copy of {req['name']} and compare it with the reference",
        "check_on_return": "stages/7/test-result.json and stages/7/verify/evidence/verify.json"})
    package = {"format": "v6", "namespace": namespace, "model": facts.model_id,
               "revision": facts.revision, "nearest_model": facts.nearest_model,
               "license": facts.license_id, "non_commercial": non_commercial(facts.license_id),
               "profiles": records, "skipped_profiles": skipped, "hf_linked": test["hf_linked"],
               "hf_missing": test["hf_missing"], "publish_commands": f"stages/7/{PUBLISH_FILE}"}
    if facts.graft:
        package["mtp_graft"] = facts.graft
    _write_json(stage_dir / "package.json", package)
    return package


def finish(run_dir, stage_dir, *, hostname: str | None = None) -> dict:
    """After the boot check: record its result, rewrite the cards with it, scrub again and write
    the publish commands."""
    facts = read_run(run_dir)
    stage_dir = Path(stage_dir)
    package = _json(stage_dir / "package.json", "stage 7's package.json")
    test = _json(stage_dir / "test-result.json", "stage 7's test-result.json")
    vpath = stage_dir / "verify" / "evidence" / "verify.json"
    verify = None
    if test.get("returncode") == 0 and vpath.is_file():
        verify = _json(vpath, "the boot check's verify.json")
    for rec in package["profiles"]:
        out = facts.run_dir / rec["dir"]
        rec["verified"] = bool(rec["required"] and verify is not None)
        if rec["required"]:
            if verify is None:
                rec["verify"] = {"failed": f"the boot check exited {test.get('returncode')} "
                                           f"(timed out: {test.get('timed_out')}); see "
                                           f"{test.get('output', {}).get('path')}"}
            else:
                rec["verify"] = {k: verify[k] for k in ("top1_agreement", "coherent", "n_tokens",
                                                        "server_ready_s", "evidence")}
        write_card(out, facts, rec, namespace=package["namespace"],
                   verify=verify if rec["required"] else None)
        rec["evidence_files"] = copy_evidence(facts, out, (out / "README.md").read_text(encoding="utf-8"),
                                              hostname=hostname)
        rec["scrub"] = scrub_package(out, hostname=hostname, namespace=package["namespace"])
    (stage_dir / PUBLISH_FILE).write_text(
        publish_commands(package["profiles"], namespace=package["namespace"],
                         license_id=facts.license_id), encoding="utf-8")
    _write_json(stage_dir / "package.json", package)
    return package
