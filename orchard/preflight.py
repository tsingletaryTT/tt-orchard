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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

from orchard.defaults import TEST_DISK_GB

OK, WARN, BLOCK = "ok", "warn", "block"
HUB_TIMEOUT_S = 20
GOZER_TIMEOUT_S = 10
REFERENCE_IMPORT_TIMEOUT_S = 180     # importing torch and transformers cold can take a minute
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
    return HubInfo(
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


def check_disk(*, free_gb: dict[str, float], model_gb: float, local: bool, min_free_gb: float,
               same_device: bool) -> Check:
    """`free_gb` has the keys `hf_home` and `cache_root`. The download needs room on the hf_home disk
    unless the snapshot is already local; each cache disk needs the margin. Paths on one disk share it."""
    need = {"hf_home": (0.0 if local else model_gb) + min_free_gb, "cache_root": min_free_gb}
    if same_device:
        need = {"hf_home": need["hf_home"] + need["cache_root"]}
    short = [f"{label}: needs {need[label]:.0f} GB, {free_gb[label]:.0f} GB free" for label in need
             if free_gb[label] < need[label]]
    if short:
        return Check("disk", BLOCK, "; ".join(short), "disk-full")
    return Check("disk", OK, ", ".join(f"{label} {free_gb[label]:.0f} GB free (needs {need[label]:.0f})"
                                       for label in need))


def check_credentials(found: list[Path], accepted: bool) -> Check:
    if not found:
        return Check("credentials", OK, "no credential files visible to agent shells")
    names = ", ".join(str(p) for p in found)
    if accepted:
        return Check("credentials", WARN, f"visible to agent shells and accepted: {names}")
    return Check("credentials", BLOCK, f"credential files agents could read: {names}. Move them, or pass "
                 "--accept-credentials-visible to accept the risk", "credentials-needed")


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


def check_port(port: int, in_use: bool, resuming: bool = False) -> Check:
    if in_use and resuming:
        return Check("port", WARN, f"something listens on the coder port {port}. This run is resuming, so it "
                     "may be this run's own coder; the supervisor stops it on recovery")
    if in_use:
        return Check("port", BLOCK, f"something already listens on the coder port {port}; stop it or "
                     "change coder.port", "coder-unusable")
    return Check("port", OK, f"coder port {port} is free")


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


def check_gozer(text: str) -> Check:
    if not text.strip():
        return Check("gozer", BLOCK, "gozer gave no status; the chips cannot be leased", "hardware-unhealthy")
    chips = _CHIP_LINE.findall(text)
    if not chips:
        return Check("gozer", BLOCK, "could not read gozer's status", "hardware-unhealthy")
    stale = [n for n, state in chips if state == "STALE"]
    busy = [(n, state) for n, state in chips if state not in ("FREE", "STALE")]
    notes = []
    if stale:
        notes.append(f"chip {', '.join(stale)} hold a stale lease; `gozer reconcile` clears it (not done here)")
    if busy:
        notes.append("chips in use: " + ", ".join(f"chip {n} {s}" for n, s in busy)
                     + "; the run waits for them and takes none it does not hold")
    if notes:
        return Check("gozer", WARN, "; ".join(notes))
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
                  resuming: bool = False) -> list[Check]:
    s = signals or default_signals(cfg)
    out: list[Check] = []

    def guarded(name: str, reason: str, fn: Callable[[], Check]) -> None:
        try:
            out.append(fn())
        except Exception as exc:                      # a failing signal is a result, not a crash
            out.append(Check(name, BLOCK, f"could not check: {type(exc).__name__}: {exc}", reason))

    try:
        hub, err = s.hub_info(model_id)
    except Exception as exc:
        hub, err = None, f"{type(exc).__name__}: {exc}"
    try:
        local = s.local_snapshot(model_id) is not None
    except Exception:
        local = False
    out.append(check_hub(hub, err, local))

    def disk() -> Check:
        paths = {"hf_home": hf_home_for(cfg), "cache_root": cache_root_for(cfg)}
        return check_disk(free_gb={k: s.free_gb(p) for k, p in paths.items()},
                          model_gb=(hub.total_bytes / 1e9 if hub else 0.0), local=local,
                          min_free_gb=cfg.min_free_gb or TEST_DISK_GB,
                          same_device=bool(s.same_device(paths["hf_home"], paths["cache_root"])))

    guarded("disk", "disk-full", disk)
    guarded("credentials", "credentials-needed",
            lambda: check_credentials(list(s.credentials()), accept_credentials))
    guarded("tiers", "config-invalid", lambda: check_tiers(s.load_tiers, Path(cfg.tiers), cfg.coder.port))
    guarded("port", "coder-unusable", lambda: check_port(cfg.coder.port, bool(s.port_in_use(cfg.coder.port)), resuming))
    guarded("gozer", "hardware-unhealthy", lambda: check_gozer(s.gozer_status()))
    ref = cfg.reference_python
    guarded("reference", "config-invalid", lambda: check_reference(
        ref, s.reference_problem(ref) if ref is not None and s.reference_problem else None))
    return out


def blocked(checks: list[Check]) -> list[Check]:
    return [c for c in checks if c.status == BLOCK]


# ---- the real signals ---------------------------------------------------------------------------

def _nearest_existing(path: Path) -> Path:
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def default_signals(cfg) -> Signals:
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

    return Signals(
        hub_info=hub_info, local_snapshot=local_snapshot,
        free_gb=lambda p: shutil.disk_usage(_nearest_existing(p)).free / 1e9,
        same_device=lambda a, b: os.stat(_nearest_existing(a)).st_dev == os.stat(_nearest_existing(b)).st_dev,
        credentials=lambda: supervisor.visible_credentials(cfg.operator_home or supervisor.operator_home()),
        port_in_use=port_in_use, gozer_status=gozer_status, load_tiers=tiers.load,
        reference_problem=reference_problem)
