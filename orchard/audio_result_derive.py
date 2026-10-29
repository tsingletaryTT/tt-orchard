# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Derive an audio result file for a newer chip run (issue 161), without editing any run file.

`orchard/audio_package.py` reads `stages/2/audio_result.json`. When a later chip run exists (here: the same model with the autoencoder decoder's transformer
on the chip, made after stage 2 had finished), its agreement with the CPU reference is a new measurement and the run's own result file does not hold it.
`derive` takes the run's result as the base and returns a new dictionary:

    replaced:        runs.chip (the later run's facts, with `chip_decoder` true), agreement.chip_vs_reference (the later run's comparison)
    dropped:         agreement.chip_vs_host_same_env and repeats.chip (both belong to the earlier chip run, so they no longer describe the result)
    kept from base:  the CPU reference runs, the environment floor, the control runs and both control checks, and the earlier repeats
    added:           `derived_from` (the base file's sha256, what was replaced, dropped and kept, and the later run's label), a `runtime` name and `limits`

The later run has to be the same model revision, library commit and settings as the base's chip run, to have recorded the chip decoder's call time, and its
agreement has to have passed. Anything else is a `DeriveError`. The control (which tells the fine-tune from the base model) is not measured again for the
later run, and `derived_from` says so.

Command line (reads run directories, writes one JSON file, opens no device):

    python -m orchard.audio_result_derive RUN_DIR CHIP_RUN_DIR OUT_JSON [--agreement-report-out FILE]

`CHIP_RUN_DIR` is a directory recorded by `chip_generate.py` (it has `manifest.json`). The CPU reference it is compared with is the base result's own
`reference_runs.reference.dir` under RUN_DIR. The comparison is `audio_stages.compare` (the same one stage 2 uses), with the dtype setting left out of the match.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

MATCH_KEYS = ("prompt", "steps", "cfg_scale", "sampler", "seed", "sample_size", "duration")
DECODER_KEY = "chip_decoder_call_s"
RUNTIME_NAME = "stable-audio-3-dit-and-decoder-on-chip"
KEPT = ("reference_runs", "environment_floor", "control_discrimination", "control_difference", "runs.control_chip", "repeats.control_chip")


class DeriveError(Exception):
    """The later run cannot stand in for the base result's chip run; the message says why."""


def derive(base: dict, *, chip_run: dict, agreement: dict, base_sha256: str, chip_label: str) -> dict:
    """A new result dictionary (the base is not changed). `chip_run` is `audio_stages.run_facts` of the later run (dir, model, revision, library_commit,
    settings, timings); `agreement` is `audio_stages.compare(reference, later run)`."""
    old = ((base.get("runs") or {}).get("chip")) if isinstance(base, dict) else None
    if not isinstance(old, dict):
        raise DeriveError("the base result has no runs.chip to replace")
    for key in ("model", "revision", "library_commit"):
        if chip_run.get(key) != old.get(key):
            raise DeriveError(f"the later run's {key} is {chip_run.get(key)!r} and the base result's chip run has {old.get(key)!r}; it is not the same model, "
                              "revision and library")
    got, want = chip_run.get("settings") or {}, old.get("settings") or {}
    for key in MATCH_KEYS:
        if got.get(key) != want.get(key):
            raise DeriveError(f"the later run's {key} is {got.get(key)!r} and the base result's chip run used {want.get(key)!r}; the two are not comparable")
    timings = chip_run.get("timings") or {}
    if not isinstance(timings.get(DECODER_KEY), (int, float)):
        raise DeriveError(f"the later run recorded no {DECODER_KEY}, so it did not run the autoencoder decoder's transformer on the chip")
    if not (isinstance(agreement, dict) and agreement.get("ok") is True and agreement.get("comparable") is True):
        raise DeriveError("the later run did not agree with the CPU reference within the bars (or the two were not comparable)")
    out = copy.deepcopy(base)
    chip = copy.deepcopy(chip_run)
    chip["role"] = "chip"
    chip["chip_decoder"] = True
    chip["derived_from_run"] = chip_label
    out["runs"]["chip"] = chip
    out.setdefault("agreement", {})["chip_vs_reference"] = copy.deepcopy(agreement)
    dropped = []
    if out["agreement"].pop("chip_vs_host_same_env", None) is not None:
        dropped.append("agreement.chip_vs_host_same_env")
    if isinstance(out.get("repeats"), dict) and out["repeats"].pop("chip", None) is not None:      # the earlier chip run's repeats and their timings
        dropped.append("repeats.chip")
    out["runtime"] = RUNTIME_NAME
    out["limits"] = ["Nobody has listened to the clips; no check here says the music is good.",
                     "The agreement bars are provisional choices, not yet set from a measured noise floor.",
                     "The diffusion transformer and the autoencoder decoder's transformer ran on the chip. The T5Gemma encoder, the sampler, the decoder's first linear "
                     "layer, its final convolution and the waveform unpatching ran on the host.",
                     "Agreement is measured on one prompt, one duration and one seed, with the later chip run.",
                     "The control that tells the fine-tune from the base model was measured on the earlier chip run, with the decoder on the host."]
    out["derived_from"] = {"base_sha256": base_sha256, "chip_run": chip_label, "replaced": ["runs.chip", "agreement.chip_vs_reference", "runtime", "limits"],
                           "dropped": dropped, "kept_from_base": [k for k in KEPT if _has(base, k)]}
    return out


def _has(d: dict, dotted: str) -> bool:
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False
        cur = cur[part]
    return True


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="python -m orchard.audio_result_derive", description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir")
    p.add_argument("chip_run_dir")
    p.add_argument("out_json")
    p.add_argument("--agreement-report-out", help="also write the full per-array comparison report here")
    a = p.parse_args(argv)
    from orchard import audio_reference as ar
    from orchard import audio_stages as st
    run, chip_dir, out = Path(a.run_dir), Path(a.chip_run_dir).resolve(), Path(a.out_json)
    if out.exists():
        print(f"{out} exists; a derived file is never overwritten", file=sys.stderr)
        return 2
    base_path = run / "stages" / "2" / "audio_result.json"
    from orchard import audio_package as pkg
    from orchard import safeio
    try:                                                   # one read, through safeio (a regular file, capped, never blocks); the hash is of those bytes
        raw = safeio.read_bytes(base_path, max_bytes=pkg.MAX_JSON_BYTES)
        base = pkg.read_json_file(base_path, "stages/2/audio_result.json")
    except (safeio.SafeReadError, pkg.AudioPackageError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    ref_dir = run / base["reference_runs"]["reference"]["dir"]
    facts = st.run_facts(chip_dir.parent, chip_dir.name)
    with ar.load_run(ref_dir) as ref, ar.load_run(chip_dir) as cand:
        report = ar.compare_runs(ref, cand, cross_precision=True)
    agreement = st.compare(ref_dir, chip_dir)
    try:
        result = derive(base, chip_run=facts, agreement=agreement, base_sha256=hashlib.sha256(raw).hexdigest(), chip_label=chip_dir.name)
    except DeriveError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    out.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    if a.agreement_report_out:
        Path(a.agreement_report_out).write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
