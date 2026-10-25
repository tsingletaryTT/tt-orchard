# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What `tt-orchard bringup` checks before it starts anything.

Each check is a pure function of values handed to it, so a test needs no network, no gozer, no socket and
no disk. `Signals` holds the six things that do touch the outside world, and `default_signals` builds the
real ones. A check returns a `Check`: `ok`, `warn` (the operator should know) or `block` (the run cannot
start), and a block names its reason from the set in the bringup spec, section 5.

Two rules run through the file:

- The harness never uses the operator's Hugging Face token. The hub lookup sends no credentials, so a gated
  or private model is a block, not a login. (The supervisor's own credentials check is separate: it
  covers files an agent could read.)
- Nothing is cleared. A stale gozer lease is a warning that names `gozer reconcile`; a chip in use is a
  warning. The preflight only reads.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from orchard.defaults import TEST_DISK_GB

OK, WARN, BLOCK = "ok", "warn", "block"
HUB_TIMEOUT_S = 20
GOZER_TIMEOUT_S = 10
CPU_PROBE_TIMEOUT_S = 5
REFERENCE_IMPORT_TIMEOUT_S = 180     # importing torch and transformers cold can take a minute
SKILLS_REUSE_S = 600.0               # choice: the UI asks for a preflight on every form edit; one check in 10 min will do
_LICENSE_FILE = re.compile(r"(licen[cs]e|copying)", re.IGNORECASE)
_CHIP_LINE = re.compile(r"^\s*chip\s+(\d+)\s+\S+\s+(\S+)", re.MULTILINE)


@dataclass
class Check:
    name: str
    status: str                     # ok, warn or block
    detail: str
    reason: str | None = None       # set for a block: one of the spec's enumerated reasons
    data: object = None             # what the check learned that the command reuses (the hub check: HubInfo)


@dataclass
class HubInfo:
    id: str
    sha: str
    private: bool
    gated: bool
    license: str | None
    total_bytes: int
    files: list[str]
    code_files: list[str]
    has_license_file: bool
    base_models: list[str] = field(default_factory=list)       # the card's base_model, as a list
    architectures: list[str] = field(default_factory=list)


def parse_hub(payload: dict) -> HubInfo:
    """The parts of `GET /api/models/<id>?blobs=true` the preflight reads."""
    siblings = payload.get("siblings") or []
    names = [s.get("rfilename", "") for s in siblings]
    lic = (payload.get("cardData") or {}).get("license")
    if not lic:
        for tag in payload.get("tags") or []:
            if isinstance(tag, str) and tag.startswith("license:"):
                lic = tag.split(":", 1)[1]
                break
    raw_base = (payload.get("cardData") or {}).get("base_model")
    bases = [raw_base] if isinstance(raw_base, str) else [b for b in raw_base if isinstance(b, str)] \
        if isinstance(raw_base, list) else []
    archs = (payload.get("config") or {}).get("architectures")
    return HubInfo(
        base_models=[b for b in bases if b != payload.get("id")],
        architectures=[a for a in archs if isinstance(a, str)] if isinstance(archs, list) else [],
        id=payload.get("id", ""), sha=payload.get("sha") or "", private=bool(payload.get("private")),
        gated=bool(payload.get("gated")), license=lic or None,
        total_bytes=sum(int(s.get("size") or 0) for s in siblings), files=names,
        code_files=[n for n in names if n.endswith(".py")],
        has_license_file=any(_LICENSE_FILE.match(Path(n).name) for n in names),
    )


def check_hub(hub: HubInfo | None, error: str | None, local: bool) -> Check:
    if hub is None:
        if local:
            return Check("hub", WARN, f"the hub could not be reached ({error}); using the local snapshot")
        return Check("hub", BLOCK, f"cannot look the model up on the hub: {error}", "model-unavailable")
    if hub.private or hub.gated:
        return Check("hub", BLOCK, "the model is gated or private. The harness never uses your Hugging Face "
                     "token, so it cannot fetch it. Download it yourself, then run bringup again",
                     "credentials-needed")
    if not hub.license and not hub.has_license_file:
        return Check("hub", BLOCK, "the model has no license in its card and no license file; a person "
                     "must read what it allows", "license-needs-review")
    detail = (f"{hub.id} at {hub.sha[:8]}, license {hub.license or 'in the license file'}, "
              f"{hub.total_bytes / 1e9:.1f} GB")
    if hub.code_files:
        return Check("hub", WARN, detail + f". The repo ships code ({', '.join(hub.code_files)}); it is "
                     "fetched and recorded, and the harness does not run it outside the agent sandbox",
                     data=hub)
    return Check("hub", OK, detail, data=hub)


HINT_COUNT, HINT_MIN_GB = 6, 1.0


def _dir_gb(d: Path) -> float:
    """Bytes under `d` in GB. A link counts as the file it reaches (a hub model's blobs are links into a shared store) and each
    file counts once, by device and inode."""
    total, seen = 0, set()
    for top, _, files in os.walk(d):
        for f in files:
            try:
                st = os.stat(os.path.join(top, f))
            except OSError:
                continue
            if (st.st_dev, st.st_ino) not in seen:
                seen.add((st.st_dev, st.st_ino))
                total += st.st_size
    return total / 1e9


def _repo_of(model: str) -> str:
    return model.split("@", 1)[0].strip()


def disk_hints(hf_home, cache_root, exclude) -> list[tuple[str, float, str]]:
    """Every Hugging Face hub model and tensor cache of at least HINT_MIN_GB, biggest first: (name, GB, disk), where disk is
    `hf_home` or `cache_root`. Read only. `exclude` holds the model ids this run needs (the model and its base); neither their hub
    directories nor their tensor caches (whose marker may carry an `@revision`) are offered. check_disk keeps the hints of the short disks."""
    skip = {_repo_of(m) for m in exclude}
    found: list[tuple[str, float, str]] = []
    hub = Path(hf_home) / "hub"
    if hub.is_dir():
        for d in hub.glob("models--*"):
            repo = d.name[len("models--"):].replace("--", "/", 1)
            if repo in skip:
                continue
            found.append((repo, _dir_gb(d / "blobs") if (d / "blobs").is_dir() else _dir_gb(d), "hf_home"))
    from orchard.caches import KERNEL_MODEL, local_caches
    for c in local_caches(cache_root):
        if c.model != KERNEL_MODEL and _repo_of(c.model or "") not in skip:
            found.append((f"tensor cache {c.model}", c.gb, "cache_root"))
    return sorted(((n, round(g, 1), label) for n, g, label in found if g >= HINT_MIN_GB), key=lambda t: -t[1])


def safe_disk_hints(hf_home, cache_root, exclude) -> list[tuple[str, float, str]]:
    """disk_hints for the preflight: a hint is a convenience, so a failure gives none and never blocks the check."""
    try:
        return disk_hints(hf_home, cache_root, [m for m in exclude if m])
    except Exception:
        return []


def check_disk(*, free_gb: dict[str, float], model_gb: float, local: bool, min_free_gb: float,
               same_device: bool, hints: list[tuple[str, float, str]] | None = None) -> Check:
    """`free_gb` has the keys `hf_home` and `cache_root`. The download needs room on the hf_home disk
    unless the snapshot is already local; each cache disk needs the margin. Paths on one disk share it."""
    need = {"hf_home": (0.0 if local else model_gb) + min_free_gb, "cache_root": min_free_gb}
    if same_device:
        need = {"hf_home": need["hf_home"] + need["cache_root"]}
    short = [f"{label}: needs {need[label]:.0f} GB, {free_gb[label]:.0f} GB free" for label in need
             if free_gb[label] < need[label]]
    if short:
        # How far short each disk is, so `--prune-caches` knows how much to free and whether it can.
        data = {"short_gb": {label: round(need[label] - free_gb[label], 3) for label in need
                             if free_gb[label] < need[label]}, "same_device": same_device}
        detail = "; ".join(short)
        # Only what lives on a short disk can clear the block; one shared disk takes both kinds.
        useful = [(n, g) for n, g, label in (hints or []) if same_device or label in data["short_gb"]][:HINT_COUNT]
        if useful:
            detail += (". Largest on these disks: " + ", ".join(f"{n} {g:.0f} GB" for n, g in useful)
                       + ". The harness removes none of the hub models; --prune-caches removes only tensor caches no unfinished run uses")
        return Check("disk", BLOCK, detail, "disk-full", data)
    return Check("disk", OK, ", ".join(f"{label} {free_gb[label]:.0f} GB free (needs {need[label]:.0f})"
                                       for label in need))


def prunable_shortfall(check: Check) -> float | None:
    """GB that removing tensor caches would have to free to clear this disk block, or None when it cannot help.

    The caches live under cache_root. Removing them frees the cache_root disk, and the hf_home disk only when the two
    share one (check_disk then reports the sum under hf_home). A shortfall on a separate hub disk is not theirs to fix."""
    data = check.data if check.status == BLOCK and isinstance(check.data, dict) else None
    if not data or "short_gb" not in data:
        return None
    short = data["short_gb"]
    if "hf_home" in short and not data.get("same_device"):
        return None
    return max(short.values()) if short else None


def check_credentials(found: list[Path], accepted: bool) -> Check:
    if not found:
        return Check("credentials", OK, "no credential files visible to agent shells")
    names = ", ".join(str(p) for p in found)
    if accepted:
        return Check("credentials", WARN, f"visible to agent shells and accepted: {names}")
    return Check("credentials", BLOCK, f"credential files agents could read: {names}. Move them, or run "
                 "bringup without --refuse-credentials-visible to accept the risk", "credentials-needed")


def check_tiers(load: Callable, path: Path, coder_port: int) -> Check:
    try:
        cfg = load(path)
    except (ValueError, OSError) as exc:
        return Check("tiers", BLOCK, f"the tier config is not usable: {exc}", "config-invalid")
    on_port = [name for name, t in cfg.tiers.items()
               if t.get("placement") == "chips" and urlparse(str(t.get("endpoint", ""))).port == coder_port]
    if len(on_port) != 1:
        return Check("tiers", BLOCK, f"exactly one chips tier must use the coder port {coder_port}; "
                     f"found {len(on_port)}", "config-invalid")
    return Check("tiers", OK, f"tier {on_port[0]!r} serves on port {coder_port}")


def check_port(port: int, in_use: bool, resuming: bool = False, kept: bool = False) -> Check:
    if in_use and kept:
        return Check("port", OK, f"the coder an earlier lab run kept up listens on port {port}; the run checks "
                     "it and uses it (`tt-orchard coder stop` stops it)")
    if in_use and resuming:
        return Check("port", WARN, f"something listens on the coder port {port}. This run is resuming, so it "
                     "may be this run's own coder; the supervisor stops it on recovery")
    if in_use:
        return Check("port", BLOCK, f"something already listens on the coder port {port}; stop it or "
                     "change coder.port", "coder-unusable")
    return Check("port", OK, f"coder port {port} is free")


def _kept_on_port(cfg) -> bool:
    """A lab run with keep_up, and a kept coder recorded for this port (orchard/kept_coder.py)."""
    if getattr(cfg, "mode", "local") != "lab" or not getattr(cfg.coder, "keep_up", False):
        return False
    from orchard import kept_coder
    from orchard.caches import cache_root_of
    rec = kept_coder.read(cache_root_of(cfg)) or {}
    return (rec.get("identity") or {}).get("port") == cfg.coder.port


def check_cpu_standin(problem: str | None, endpoint: str, model: str) -> Check:
    """The CPU tier takes the coder's place while stage 4's 4-chip test parks it. A park whose stand-in is down
    fails its canary hours into a run (RAG-Gate-27B, 2026-10-13), so the run does not start without it."""
    if problem is None:
        return Check("cpu", OK, f"the CPU stand-in answers: {model} at {endpoint}")
    return Check("cpu", BLOCK, f"the CPU stand-in {model} at {endpoint} is not usable: {problem}. Start it with "
                 "`OLLAMA_MODELS=<your models directory> ollama serve` (and `ollama pull` the model if it is "
                 "missing); the coder is parked behind it during stage 4's 4-chip test", "coder-unusable")


def cpu_standin_problem(endpoint: str, model: str, timeout: float = CPU_PROBE_TIMEOUT_S) -> str | None:
    """Ask the CPU tier's server for its model list. None means it answers and lists `model`."""
    try:
        with urllib.request.urlopen(endpoint.rstrip("/") + "/models", timeout=timeout) as resp:
            ids = [str(m.get("id")) for m in json.load(resp).get("data", [])]
    except (OSError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
    if model not in ids:
        return f"the server does not list {model} (it lists {', '.join(ids) or 'nothing'}); run `ollama pull {model}`"
    return None


def check_reference(python: Path | None, problem: str | None) -> Check:
    """The interpreter stage 1 runs the CPU reference with. Without one the agent looks for an interpreter
    itself, which is how an earlier run ended up installing packages into the machine's shared venv."""
    if python is None:
        return Check("reference", WARN, "no reference_python is configured, so the stage 1 agent will look for "
                     "an interpreter with torch and transformers itself; set reference_python in bringup.toml")
    if problem:
        return Check("reference", BLOCK, f"reference_python {python} cannot import torch, transformers, "
                     f"tokenizers and safetensors: {problem}", "config-invalid")
    return Check("reference", OK, f"{python} imports torch, transformers, tokenizers and safetensors")


def check_skills(results: list) -> Check:
    """The tenstorrent/skills clone(s) that skills_dirs point into, compared with upstream main (orchard/skillsfresh).

    The operator's rule is that runs use the latest tenstorrent/skills. Bringup runs unattended, so anything short of
    current is a warning that says what was found and how to update; it never blocks. Orchard updates nothing."""
    if not results:
        return Check("skills", WARN, "bringup.toml names no skills_dirs, so the stage skills from tenstorrent/skills "
                     "(model-bringup and the others) are not read and nothing was compared with upstream")
    notes, worst = [], OK
    for r in results:
        text = f"{r.path}: {r.state}. {r.detail}"
        if r.state != "current":
            worst = WARN
            text += " " + r.update_text()
        if r.warnings:
            worst = WARN
            text += " " + " ".join(r.warnings)
        notes.append(text)
    return Check("skills", worst, " | ".join(notes))


def _class_note(report, repo: str) -> str:
    for c in (report or {}).get("candidates", []):
        est = c.get("class_estimate")
        if c.get("repo") == repo and est:
            return f"; research expects class {est['class']}"
    return ""


def _closest(report) -> str:
    choices = (report or {}).get("bundle_choices") or []
    if not choices:
        return ""
    return " Closest bundles found: " + ", ".join(
        f"{c['name']} ({c.get('class') or 'no estimate'}, {c['confidence']})" for c in choices[:3]) + "."


def _port_note(report) -> str:
    found = (report or {}).get("ports") or []
    if not found:
        return ""
    return " Existing work: " + "; ".join(f"{p['claim']} ({p['url']})" for p in found[:2]) + "."


LISTED_CODE_ROWS = 3
LISTED_BUNDLES = 8
_BLACKHOLE_NAMED = re.compile(r"\b(?:p100a?|p150[abc]?|p300[abc]?|blackhole)\b", re.I)


def _code_note(report, base: str) -> str:
    """Code for the base in tt-metal or tt-transformers, when no bundle serves it. Code is not a bundle: stages 2 and
    4 swap weights into an installed v6 thin bundle, so the operator has to get one first. The hardware sentence
    comes from the facts research put in the claims (orchard/research/ports.py); with none, it says nothing."""
    from orchard.research.types import LACKS_BLACKHOLE
    code = [p for p in (report or {}).get("ports") or []
            if p.get("source") in ("tt-metal", "tt-transformers") and p.get("confidence") == "inferred"]
    if not code:
        return ""
    claims = "; ".join(f"{p['claim']} ({p['url']})" for p in code[:LISTED_CODE_ROWS])
    text = (f" Code for it exists, but code is not a bundle: {claims}. A bundle that serves {base} has to be built "
            "or installed outside tt-orchard first.")
    # "no Blackhole board named" holds the word too, so a row that says it must not count as naming one
    names = any(_BLACKHOLE_NAMED.search(p["claim"].replace(LACKS_BLACKHOLE, "")) for p in code)
    lacks = any(LACKS_BLACKHOLE in p["claim"] for p in code)
    if names:
        return text + " The code names a Blackhole board."
    if lacks:
        return text + (" It names no Blackhole board, and this box has Blackhole boards, so check that it runs here "
                       "before building a bundle.")
    return text


def _selection_basis(report) -> str:
    """How the research chose the base when the card names none: its confidence and the candidate row's reason."""
    sel = report["selected"]
    conf = sel.get("confidence") or "inferred"
    why = next((c.get("why") for c in report.get("candidates") or [] if c.get("repo") == sel.get("repo")), None)
    if why:
        return f"{conf} ({why})"
    return "inferred from its architecture" if conf == "inferred" else conf


def _confirm_inferred(base: str, serving, report) -> Check:
    """An inferred base is a guess from the architecture. Using it means downloading its snapshot and basing the run
    on it, so it waits for the operator: a block that offers its bundle first, like a missing bundle does."""
    first = [{"name": b.name, "installed": True} for b in serving]
    names = {c["name"] for c in first}
    rest = [dict(c) for c in (report or {}).get("bundle_choices") or [] if c.get("name") not in names]
    return Check("base", BLOCK, f"the card names no base model; research infers {base} ({_selection_basis(report)}), "
                 f"served by the installed bundle {serving[0].name}. An inferred base is not downloaded or used "
                 f"without a confirmation: pass --base {serving[0].name} to use it, --accept-inferred-base to take "
                 "what research inferred, or --base <bundle or model id> to choose another" + _port_note(report),
                 "inferred-base-unconfirmed",
                 data={"base": base, "snapshot": None, "bundle": None, "needs_fetch": False, "inferred": True,
                       "candidates": first + rest, "report": report})


def check_base(*, model_id: str, hub: HubInfo | None, base_override: str | None, local_snapshot: Callable,
               installed: list, search: Callable | None, report: dict | None = None,
               accept_inferred: bool = False) -> Check:
    """The model this run is compared with and served through. It needs an installed bundle (stages 2 and
    4 serve the new weights with it) and a snapshot on disk (stage 0 compares tensors and configs). A missing
    snapshot is downloaded by the command. A missing bundle is a block that lists what `tt model search`
    found, so the operator can pick one with --base. `report` is the research report
    (orchard/research/report.py): it supplies an inferred base when the card names none, the expected
    class, ranked bundle choices and existing ports. Without a report the behaviour is the earlier one. A base
    research inferred (not stated by the card or its tags) blocks unless `accept_inferred` is set."""
    from orchard import nearest
    base = base_override or (hub.base_models[0] if hub and hub.base_models else None)
    inferred = False
    if not base and report and report.get("selected"):
        base, inferred = report["selected"]["repo"], True
    if not base:
        return Check("base", WARN, "the model card names no base_model, so stage 0 has nothing to compare it "
                     "with. Pass --base <bundle or model id> to name the model to start from"
                     + _closest(report) + _port_note(report),
                     data={"base": None, "snapshot": None, "needs_fetch": False, "candidates": [], "report": report})
    serving = [b for b in installed if b.weights_repo == base]
    if serving and inferred and report["selected"].get("confidence") != "stated" and not accept_inferred:
        return _confirm_inferred(base, serving, report)
    if serving:
        snap = local_snapshot(base)
        data = {"base": base, "snapshot": snap, "bundle": serving[0].name, "needs_fetch": snap is None,
                "candidates": [], "report": report}
        where = "its snapshot is local" if snap else "its snapshot will be downloaded"
        note = _class_note(report, base)
        if inferred:
            return Check("base", WARN, f"base {base}: {_selection_basis(report)}; the card names none. Bundle "
                         f"{serving[0].name} is installed; {where}{note}. Pass --base to choose another", data=data)
        if not any(b.schema == "6" for b in serving):
            return Check("base", WARN, f"base {base}: {serving[0].name} is installed, but it is a "
                         f"{serving[0].schema} container package. Stages 2 and 4 serve the new weights through a container package's "
                         "default profile, and only when exactly one installed package serves the base on the chip count of the "
                         f"stage; {where}{note}", data=data)
        return Check("base", OK, f"base {base}: bundle {serving[0].name} is installed; {where}{note}", data=data)
    family = nearest.family_query(base)
    found = None
    if search is not None:
        try:
            found = search(family)
        except Exception:
            found = None
    choices = (report or {}).get("bundle_choices") or []
    if choices:
        ordered = [dict(c) for c in choices]
        listing = "ranked bundles: " + ", ".join(
            f"{c['name']} ({c.get('class') or 'no estimate'}, {c['confidence']})" for c in ordered[:5])
    else:
        cands: dict[str, dict] = {}
        for b in installed:
            if nearest.family_query(b.weights_repo) == family and b.name != model_id:
                cands[b.name] = {"name": b.name, "installed": True}
        for c in found or []:
            if c["name"] != model_id and c["name"] not in cands:
                cands[c["name"]] = {"name": c["name"], "installed": c.get("installed")}
        ordered = sorted(cands.values(), key=lambda c: (not c["installed"], c["name"]))
        if found is None:
            listing = f"the search for '{family}' bundles could not be made"
        elif ordered:
            shown = ", ".join(c["name"] for c in ordered[:LISTED_BUNDLES])
            more = f" (and {len(ordered) - LISTED_BUNDLES} more)" if len(ordered) > LISTED_BUNDLES else ""
            listing = f"`tt model search {family}` found: {shown}{more}"
        else:
            listing = f"`tt model search {family}` found no bundle"
    code = _code_note(report, base)
    return Check("base", BLOCK, f"no installed bundle serves {base}, the model this one is based on, so stages "
                 f"0, 2 and 4 have nothing to compare it with or serve it through.{code} {listing}."
                 f"{'' if code else _port_note(report)} "
                 "Pick one with --base <bundle>, or install a bundle yourself and run bringup again",
                 "nearest-model-missing", data={"base": base, "snapshot": None, "bundle": None,
                                                "needs_fetch": False, "candidates": ordered, "report": report})


def check_gozer(text: str, kept: bool = False) -> Check:
    """`kept`: a lab run will use the coder an earlier run kept up, so the chips its lease holds are
    expected to be HELD (orchard/kept_coder.py), and the run does not wait for them."""
    if not text.strip():
        return Check("gozer", BLOCK, "gozer gave no status; the chips cannot be leased", "hardware-unhealthy")
    chips = _CHIP_LINE.findall(text)
    if not chips:
        return Check("gozer", BLOCK, "could not read gozer's status", "hardware-unhealthy")
    from orchard.kept_coder import KEPT_WHO
    kept_chips = ({n for line in text.splitlines() if KEPT_WHO in line for n, _ in _CHIP_LINE.findall(line)}
                  if kept else set())
    stale = [n for n, state in chips if state == "STALE"]
    busy = [(n, state) for n, state in chips if state not in ("FREE", "STALE") and n not in kept_chips]
    notes = []
    if kept_chips:
        notes.append(f"chip {', '.join(sorted(kept_chips))} serve the kept coder")
    if stale:
        notes.append(f"chip {', '.join(stale)} hold a stale lease; `gozer reconcile` clears it (not done here)")
    if busy:
        notes.append("chips in use: " + ", ".join(f"chip {n} {s}" for n, s in busy)
                     + "; the run waits for them and takes none it does not hold")
    if stale or busy:
        return Check("gozer", WARN, "; ".join(notes))
    if notes:
        return Check("gozer", OK, "; ".join(notes) + "; the others are free")
    return Check("gozer", OK, f"all {len(chips)} chips are free")


@dataclass(frozen=True)
class Signals:
    hub_info: Callable            # (model_id) -> (HubInfo | None, error text | None)
    local_snapshot: Callable      # (model_id) -> Path of a local snapshot, or None
    free_gb: Callable             # (path) -> free decimal GB on that path's disk
    same_device: Callable         # (path, path) -> bool
    credentials: Callable         # () -> list of credential files an agent could read
    port_in_use: Callable         # (port) -> bool
    gozer_status: Callable        # () -> text of `gozer status`, "" when unavailable
    load_tiers: Callable          # (path) -> TierConfig
    reference_problem: Callable | None = None   # (python path) -> why it cannot import the packages, or None
    installed_bundles: Callable | None = None   # () -> list[nearest.Installed]
    search_bundles: Callable | None = None      # (query) -> [{"name", "installed"}] or None
    lab_status: Callable | None = None          # () -> (answered, text of the lab's `gozer status`, error)
    research: Callable | None = None            # (model_id, hub, base_override, installed) -> report dict
    cpu_standin: Callable | None = None         # (endpoint, model) -> why the CPU tier does not answer, or None
    sfpi: Callable | None = None                # (bundle name) -> (installed SFPI version, version the bundle pins)
    lab_sfpi: Callable | None = None            # () -> the lab's installed SFPI version, or None (lab mode)
    skills_freshness: Callable | None = None    # () -> list[skillsfresh.Freshness] for the skills_dirs clones
    bus: Callable | None = None                 # () -> orchard.bus.Bus: the chips the kernel shows (sysfs and /dev; no lease)


def check_lab(lab, status) -> Check:
    """A run with a lab box needs the lab to answer over ssh; its chips should be free (the run waits
    for busy ones, as it does here)."""
    answered, text, err = status
    if not answered:
        return Check("lab", BLOCK, f"the lab {lab.host} did not answer over ssh: {(err or '').strip()[-200:]}. "
                                   "Check it with `tt-orchard lab setup --check`", "hardware-unhealthy")
    states = [line.split()[3] for line in text.splitlines() if line.split()[:1] == ["chip"] and len(line.split()) >= 4]
    if not states:
        return Check("lab", WARN, f"the lab {lab.host} answers, but `gozer status` there listed no chips")
    busy = [s for s in states if s != "FREE"]
    if busy:
        return Check("lab", WARN, f"the lab {lab.host} answers; {len(busy)} of {len(states)} chips are in use "
                                  f"({', '.join(sorted(set(busy)))}); its tests wait for them")
    return Check("lab", OK, f"the lab {lab.host} answers; {len(states)} chips, all free")


class _SkipRow(Exception):
    """Raised inside a check's try block to leave its row out."""


def has_weights(snapshot: Path) -> bool:
    """True when the snapshot directory holds a readable weight file (see `local_snapshot`)."""
    return (any(w.exists() for w in snapshot.rglob("*.safetensors"))
            or any(w.exists() for w in snapshot.glob("pytorch_model*.bin")))


def hf_home_for(cfg) -> Path:
    from orchard.supervisor import operator_home
    if cfg.hf_home:
        return Path(cfg.hf_home)
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"])
    return Path(cfg.operator_home or operator_home()) / ".cache" / "huggingface"


def cache_root_for(cfg) -> Path:
    return Path(cfg.cache_root) if cfg.cache_root else Path(cfg.runs_root) / "cache"


def run_preflight(cfg, model_id: str, *, accept_credentials: bool, signals: Signals | None = None,
                  resuming: bool = False, base_override: str | None = None,
                  accept_inferred_base: bool = False, progress: Callable[[str], None] | None = None) -> list[Check]:
    """`progress(text)` is told what is about to run (the hub lookup, the research, the machine checks), so a
    terminal is not silent while the research reads weight headers for a minute or two."""
    from orchard.research.types import reporter
    say = reporter(progress)
    s = signals or default_signals(cfg)
    out: list[Check] = []

    def guarded(name: str, reason: str, fn: Callable[[], Check]) -> None:
        try:
            out.append(fn())
        except Exception as exc:                      # a failing signal is a result, not a crash
            out.append(Check(name, BLOCK, f"could not check: {type(exc).__name__}: {exc}", reason))

    say(f"asking the hub about {model_id}")
    try:
        hub, err = s.hub_info(model_id)
    except Exception as exc:
        hub, err = None, f"{type(exc).__name__}: {exc}"
    try:
        local = s.local_snapshot(model_id) is not None
    except Exception:
        local = False
    out.append(check_hub(hub, err, local))

    def base() -> Check:
        inst = list(s.installed_bundles()) if s.installed_bundles else []
        report, failed = None, None
        if s.research is not None:
            say("researching the base model: lineage, bundles, weight headers and existing ports")
            try:
                report = s.research(model_id, hub, base_override, inst)
            except Exception as exc:                  # a failed research call leaves the earlier checks in force
                report, failed = None, type(exc).__name__
        found = check_base(model_id=model_id, hub=hub, base_override=base_override,
                           local_snapshot=s.local_snapshot, installed=inst, search=s.search_bundles, report=report,
                           accept_inferred=accept_inferred_base)
        if failed:
            from orchard.research.types import clean
            found.detail += f" (research failed: {clean(failed)})"
        return found

    base_check = None
    try:
        base_check = base()
    except Exception as exc:
        base_check = Check("base", BLOCK, f"could not check: {type(exc).__name__}: {exc}", "nearest-model-missing")
    base_gb = 0.0
    if base_check.data and base_check.data.get("needs_fetch"):
        try:
            base_hub, _ = s.hub_info(base_check.data["base"])
            base_gb = base_hub.total_bytes / 1e9 if base_hub else 0.0
        except Exception:
            base_gb = 0.0

    say("checking this machine: disk, tiers, ports and chips")

    def disk() -> Check:
        paths = {"hf_home": hf_home_for(cfg), "cache_root": cache_root_for(cfg)}
        return check_disk(free_gb={k: s.free_gb(p) for k, p in paths.items()},
                          model_gb=(hub.total_bytes / 1e9 if hub else 0.0) + base_gb, local=local,
                          min_free_gb=cfg.min_free_gb or TEST_DISK_GB,
                          same_device=bool(s.same_device(paths["hf_home"], paths["cache_root"])),
                          hints=safe_disk_hints(paths["hf_home"], paths["cache_root"], [model_id, (base_check.data or {}).get("base")]))

    guarded("disk", "disk-full", disk)
    guarded("credentials", "credentials-needed",
            lambda: check_credentials(list(s.credentials()), accept_credentials))
    guarded("tiers", "config-invalid", lambda: check_tiers(s.load_tiers, Path(cfg.tiers), cfg.coder.port))
    guarded("port", "coder-unusable", lambda: check_port(cfg.coder.port, bool(s.port_in_use(cfg.coder.port)), resuming,
                                                          kept=_kept_on_port(cfg)))
    in_lab = getattr(cfg, "mode", "local") == "lab"
    # The stand-in takes the coder's place while stage 4's 4-chip test parks it. A lab run never parks (its tests run on
    # the lab), so the row would block a run for a server it never uses.
    if s.cpu_standin is not None and not in_lab:
        def cpu() -> Check:
            tier = next(t for t in s.load_tiers(Path(cfg.tiers)).tiers.values() if t.get("placement") == "cpu")
            return check_cpu_standin(s.cpu_standin(tier["endpoint"], tier["model"]), tier["endpoint"], tier["model"])
        # The tiers row already reports a config that cannot be read; this row is only about the stand-in.
        try:
            s.load_tiers(Path(cfg.tiers))
            guarded("cpu", "coder-unusable", cpu)
        except Exception:
            pass
    gozer_seen: list[str] = []

    def gozer_text() -> str:
        """`gozer status` once per preflight: the gozer row and the bus row both read it."""
        if not gozer_seen:
            gozer_seen.append(s.gozer_status())
        return gozer_seen[0]

    guarded("gozer", "hardware-unhealthy", lambda: check_gozer(gozer_text(), kept=_kept_on_port(cfg)))
    if s.bus is not None:
        def bus_row() -> Check:
            from orchard import bus as bus_check
            try:
                text = gozer_text()
            except Exception:                         # the gozer row already says so; the bus findings stand alone
                text = ""
            return bus_check.check_bus(s.bus(), text, bus_check.needed_chips(cfg))
        guarded("bus", "hardware-unhealthy", bus_row)
    if in_lab and getattr(cfg, "lab", None) is not None and s.lab_status is not None:
        guarded("lab", "hardware-unhealthy", lambda: check_lab(cfg.lab, s.lab_status()))
    out.append(base_check)
    bundle = (base_check.data or {}).get("bundle") if isinstance(base_check.data, dict) else None
    if bundle and s.sfpi is not None:
        from orchard import sfpi as sfpi_check
        try:
            installed, pinned = s.sfpi(bundle)
            where = None
            if in_lab:
                # The kernels compile on the lab, so its SFPI is the one that counts. The pin is read from this box's copy
                # of the bundle, which `lab setup` mirrors to the lab. With no way to read the lab's version, this box's
                # would be the wrong box, so the row is left out.
                lab = getattr(cfg, "lab", None)
                if lab is None or s.lab_sfpi is None:
                    raise _SkipRow
                installed, where = s.lab_sfpi(), f"the lab {lab.host}"
            row = sfpi_check.check_sfpi(installed, pinned, bundle, where=where)
            if row is not None:
                out.append(row)
        except _SkipRow:
            pass
        except Exception as exc:                      # a warning: the row guards against a drift it cannot always see
            out.append(Check("sfpi", WARN, f"could not check the SFPI compiler: {type(exc).__name__}: {exc}"))
    ref = cfg.reference_python
    guarded("reference", "config-invalid", lambda: check_reference(
        ref, s.reference_problem(ref) if ref is not None and s.reference_problem else None))
    if s.skills_freshness is not None:
        try:
            out.append(check_skills(list(s.skills_freshness())))
        except Exception as exc:                      # a warning, like the row itself: bringup is not blocked
            out.append(Check("skills", WARN, f"could not check the skills clone: {type(exc).__name__}: {exc}"))
    return out


def blocked(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.status == BLOCK]


# ---- the real signals ---------------------------------------------------------------------------

def _nearest_existing(path: Path) -> Path:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def default_signals(cfg, research_options=None) -> Signals:
    from orchard import supervisor, tiers

    def hub_info(model_id: str):
        # No Authorization header: the harness never sends the operator's token.
        req = urllib.request.Request(f"https://huggingface.co/api/models/{model_id}?blobs=true",
                                     headers={"User-Agent": "tt-orchard"})
        try:
            with urllib.request.urlopen(req, timeout=HUB_TIMEOUT_S) as resp:
                return parse_hub(json.load(resp)), None
        except urllib.error.HTTPError as exc:
            return None, f"HTTP {exc.code}"
        except (OSError, ValueError) as exc:
            return None, f"{type(exc).__name__}: {exc}"

    def local_snapshot(model_id: str):
        org, name = model_id.split("/", 1)
        snaps = hf_home_for(cfg) / "hub" / f"models--{org}--{name}" / "snapshots"
        found = sorted(snaps.glob("*/config.json")) if snaps.is_dir() else []
        # A config alone is not a model. An old manifest-only fetch leaves config.json and README.md, and counting that as
        # local skipped the download. A weight file counts when its link resolves (`exists()` follows it). The kinds are the
        # ones the stage templates read: *.safetensors at any depth (some repos keep a component in a subfolder) and the
        # top-level pytorch_model*.bin that stage 0's backbone rule names. Other files, such as training_args.bin, do not.
        found = [c for c in found if has_weights(c.parent)]
        return found[-1].parent if found else None

    def gozer_status() -> str:
        try:
            done = subprocess.run([cfg.gozer, "status"], capture_output=True, text=True,
                                  timeout=GOZER_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError):
            return ""
        return done.stdout if done.returncode == 0 else ""

    def port_in_use(port: int) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                return True
        except OSError:
            return False

    def reference_problem(python) -> str | None:
        try:
            done = subprocess.run([str(python), "-c", "import torch, transformers, tokenizers, safetensors"],
                                  capture_output=True, text=True, timeout=REFERENCE_IMPORT_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError) as exc:
            return f"{type(exc).__name__}: {exc}"
        if done.returncode == 0:
            return None
        lines = (done.stderr or "").strip().splitlines()
        return lines[-1] if lines else f"exit {done.returncode}"

    def models_root() -> Path:
        if cfg.package_models_root:
            return Path(cfg.package_models_root)
        return Path(cfg.operator_home or supervisor.operator_home()) / ".cache" / "tt-model" / "models"

    from orchard import nearest
    research = None
    if research_options is not None:
        from orchard.research import report as research_report
        from orchard.research.http import Fetcher
        fetcher = Fetcher(research_options.cache_dir, offline=research_options.offline,
                          refresh=research_options.refresh)
        research = lambda model_id, hub, override, inst: research_report.build(     # noqa: E731
            model_id=model_id, hub=hub, base_override=override, installed=inst, search=nearest.search,
            hub_info=hub_info, fetch=fetcher, no_github=research_options.no_github,
            progress=getattr(research_options, "progress", None))
    def sfpi_versions(bundle: str):
        from orchard import sfpi as sfpi_check
        return sfpi_check.installed_version(), sfpi_check.pinned_version(models_root(), bundle)

    def lab_sfpi():
        from orchard.lab_setup import remote
        lab = getattr(cfg, "lab", None)
        if lab is None:
            return None
        try:
            done = subprocess.run(remote(lab, "dpkg-query -W -f='${Version}' sfpi"), capture_output=True, text=True,
                                  timeout=GOZER_TIMEOUT_S)
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() or None if done.returncode == 0 else None

    def skills_freshness():
        # The research rule for a dry run (and the UI's preflight, which runs like one) is "no cache and no file": both
        # pass research options with no cache directory. The skills result cache then stays in memory. A caller with
        # no options is treated the same way.
        from orchard import skillsfresh
        offline = bool(getattr(research_options, "offline", False)) if research_options is not None else False
        write = research_options is not None and getattr(research_options, "cache_dir", None) is not None
        return skillsfresh.check_skills_dirs(cfg.skills_dirs, reuse_s=SKILLS_REUSE_S, offline=offline,
                                             cache=skillsfresh.Cache(write=write))

    def read_bus():
        from orchard import bus as bus_check
        return bus_check.read_bus()

    return Signals(
        cpu_standin=cpu_standin_problem, sfpi=sfpi_versions, lab_sfpi=lab_sfpi, skills_freshness=skills_freshness, bus=read_bus,
        installed_bundles=lambda: nearest.read_all_installed(models_root(), cfg.operator_home or supervisor.operator_home()),
        search_bundles=nearest.search,
        hub_info=hub_info, local_snapshot=local_snapshot,
        free_gb=lambda p: shutil.disk_usage(_nearest_existing(p)).free / 1e9,
        same_device=lambda a, b: os.stat(_nearest_existing(a)).st_dev == os.stat(_nearest_existing(b)).st_dev,
        credentials=lambda: supervisor.visible_credentials(cfg.operator_home or supervisor.operator_home()),
        port_in_use=port_in_use, gozer_status=gozer_status, load_tiers=tiers.load,
        reference_problem=reference_problem, lab_status=(lambda: _lab_status(cfg.lab)) if getattr(cfg, "lab", None) else None,
        research=research)


def _lab_status(lab) -> tuple[bool, str, str]:
    from orchard.lab_setup import remote
    try:
        done = subprocess.run(remote(lab, f"{lab.gozer} status"), capture_output=True, text=True, timeout=GOZER_TIMEOUT_S)
    except (OSError, subprocess.SubprocessError) as exc:
        return False, "", f"{type(exc).__name__}: {exc}"
    if done.returncode == 255:                      # ssh's own failure
        return False, "", done.stderr
    return True, done.stdout if done.returncode == 0 else "", done.stderr
