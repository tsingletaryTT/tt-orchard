# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0's gate check for a bundle that ships `orchard-capabilities.json` (docs/bundle-capabilities.md).

The stage 0 template (orchard/skills/delta-triage-templates/delta_triage.py) applies the nearest bundle's capabilities only when the
agent's triage_config.json names `nearest_bundle_dir`, and records what it did in delta.json's `capabilities`. Stage 2 then serves the
new model's own config.json for such a bundle (prepare_swap.py). A forgotten key would judge the release by the rule without the file
and still serve its config, so the supervisor checks the record against the file it reads itself from the run's `bundle` input: the
record must name that directory and the file's current sha256, and a file that was not applied must have left the path full-port.
A bundle without the file needs no record. Nothing here reads the file's content beyond its hash."""
from __future__ import annotations

import functools
import importlib.util
import os
from pathlib import Path

from orchard import safeio

FILE = "orchard-capabilities.json"
MAX_BYTES = 256 * 1024           # the triage template's CAPABILITIES_MAX_BYTES; a test keeps the two equal
INPUT = "bundle"                 # the run input that names the bundle's directory (the bring-up command records it)
DELTA_MAX_BYTES = 16 * 1024 * 1024
TEMPLATE = Path(__file__).resolve().parent / "skills" / "delta-triage-templates" / "delta_triage.py"


@functools.lru_cache(maxsize=1)
def _template():
    """The shipped stage 0 template, loaded by path (it is a script that stage 0 copies, not a package module). Loading it runs no check."""
    spec = importlib.util.spec_from_file_location("orchard_delta_triage_template", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def config_reasons(capabilities, release_config, dev_config=None) -> list[str]:
    """Why a bundle's capabilities do not cover a release's config, [] when they do. `capabilities` is the parsed
    orchard-capabilities.json, `release_config` the release's config.json, `dev_config` the config.json of the bundle's own weights or
    None (then every differing key is not checked, only the rules, model_types and architectures). This is the stage 0 template's own
    code (capability_config_reasons), the code decide_path uses, so the release-day check (orchard/drop_check.py) and stage 0 judge a
    config the same way. Needs no run directory and reads no file but the template."""
    return _template().capability_config_reasons(capabilities, release_config, dev_config)


def file_state(bundle_dir) -> tuple[str | None, str | None]:
    """(sha256, problem) of the bundle's file: (None, None) when the bundle ships none."""
    path = Path(bundle_dir) / FILE
    try:
        os.lstat(path)
    except FileNotFoundError:
        return None, None
    except (OSError, ValueError) as exc:
        return None, f"the bundle's {FILE} cannot be checked ({type(exc).__name__})"
    try:
        return safeio.sha256_file(path, max_bytes=MAX_BYTES), None
    except safeio.TooLarge:
        return None, f"the bundle's {FILE} is larger than {MAX_BYTES} bytes"
    except safeio.SafeReadError as exc:
        return None, f"the bundle's {FILE} was refused: {exc.reason.replace('the file', 'it')}"


def _same_dir(a: str, b: str) -> bool:
    try:
        return os.path.realpath(a) == os.path.realpath(b)
    except (OSError, ValueError):
        return False


def stage0_problems(delta, bundle_dir) -> list[str]:
    """Why delta.json does not show that stage 0 applied the run's bundle capabilities, or [] when it does or none apply.

    With no bundle input the supervisor has no file to vouch for, and the stage 0 agent writes triage_config.json, so it could name a folder it
    wrote as `nearest_bundle_dir` and get a record that sets differences aside. A record that says the file is present or applied is then a
    reason (PR 659 review); a record that says none was found is fine."""
    if not isinstance(bundle_dir, str) or not bundle_dir:
        rec = delta.get("capabilities") if isinstance(delta, dict) else None
        if isinstance(rec, dict) and (rec.get("present") is True or rec.get("applied") is True):
            return [f"delta.json records that the nearest bundle's {FILE} was found or applied, but this run has no bundle input, so nothing "
                    "vouches for that file; start the run with the bundle as an input (--base) or leave the record out"]
        return []
    try:
        is_dir = "\0" not in bundle_dir and os.path.isdir(bundle_dir)
    except (OSError, ValueError):
        is_dir = False
    if not is_dir:
        return [f"the run's bundle input {bundle_dir!r} is not a directory, so its {FILE} cannot be checked"]
    sha, problem = file_state(bundle_dir)
    if problem:
        return [f"{problem}; stage 0 cannot pass until the bundle ships a regular file or none"]
    rec = delta.get("capabilities") if isinstance(delta, dict) else None
    if sha is None:
        if isinstance(rec, dict) and (rec.get("present") is not False or rec.get("applied") is not False):
            return [f"delta.json records capabilities, but the run's bundle has no {FILE}; run the stage 0 template again"]
        return []
    if not isinstance(rec, dict):
        return [f"the run's bundle ships {FILE} and delta.json has no capabilities record: write nearest_bundle_dir (the run's input "
                "bundle) into triage_config.json and run the stage 0 template again"]
    where = rec.get("bundle_dir")
    if not isinstance(where, str) or not where or "\0" in where or not _same_dir(where, bundle_dir):
        return [f"delta.json's capabilities record names the bundle directory {str(where)[:200]!r}, and the run's bundle is {bundle_dir}; "
                "write the run's input bundle into triage_config.json and run the stage 0 template again"]
    if rec.get("present") is not True or rec.get("sha256") != sha:
        got = rec.get("sha256")
        return [f"delta.json's capabilities record has sha256 {str(got)[:12]!r} and the bundle's {FILE} has {sha[:12]}: stage 0 did not "
                "read the file the bundle ships now. Run the stage 0 template again"]
    applied, problems = rec.get("applied"), rec.get("problems")
    if not isinstance(applied, bool) or not isinstance(problems, list):
        return ["delta.json's capabilities record needs applied (true or false) and problems (a list)"]
    if not applied and (not isinstance(delta, dict) or delta.get("path") != "full-port"):
        return [f"the bundle's {FILE} was not applied ({'; '.join(str(p)[:200] for p in problems[:3]) or 'no reason recorded'}), so the path "
                "must be full-port"]
    return []


def gate_reasons(stage_dir, bundle_dir) -> list[str]:
    """stage0_problems for the delta.json in `stage_dir`, read with the safe reader. A delta.json that cannot be read is one reason."""
    named = isinstance(bundle_dir, str) and bool(bundle_dir)
    try:
        delta = safeio.read_json(Path(stage_dir) / "delta.json", max_bytes=DELTA_MAX_BYTES, expect=dict)
    except safeio.SafeReadError as exc:
        # With no bundle input, stage 0's own gate judges an unreadable delta.json; adding a reason here would only repeat it.
        return [f"delta.json cannot be read to check the bundle's {FILE}: {exc.reason}"] if named else []
    return stage0_problems(delta, bundle_dir)
