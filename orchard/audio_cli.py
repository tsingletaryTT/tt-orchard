# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The numeric verbs of the audio stages, run in the interpreter that has numpy and soundfile.

The supervisor needs only the standard library. It starts this script with the configured reference interpreter (orchard/audio_remote.py), writes one
JSON object on its standard input and reads one JSON object from its standard output:

    {"verb": "references" | "finish" | "gate_reference" | "gate_chip", "args": {...}}  ->  {"result": ...}  or  {"error": "..."}

`dispatch` is the same code the tests call in process. Nothing here opens a device.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def dispatch(verb: str, args: dict) -> dict:
    from orchard import audio_stages as st
    from orchard.audio_runtime import AudioPaths
    try:
        if verb == "references":
            notes: list[str] = []
            rec = st.run_references(args["stage_dir"], args["run_dir"], AudioPaths.from_record(args["paths"]), args["plan"],
                                    st.subprocess_runner(args["env"]), expected=args["expected"], library_commit=args["library_commit"],
                                    cache_dir=args.get("cache_dir"), timeout_s=float(args.get("timeout_s", 3600.0)), say=notes.append)
            return {"result": {"record": rec, "notes": notes}}
        if verb == "finish":
            return {"result": st.finish_chip(args["stage_dir"], args["run_dir"], args["ref_stage_dir"], args["plan"], args["roles"])}
        if verb == "gate_reference":
            return {"result": st.reference_problems(args["stage_dir"], args["run_dir"])}
        if verb == "gate_chip":
            return {"result": st.chip_problems(args["stage_dir"], args["run_dir"], args["ref_stage_dir"])}
        return {"error": f"unknown verb {verb!r}"}
    except st.AudioStageError as exc:
        return {"error": str(exc)}
    except (KeyError, TypeError, ValueError) as exc:          # a malformed request
        return {"error": f"bad request for {verb}: {type(exc).__name__}: {exc}"}


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    try:
        request = json.loads(sys.stdin.read())
        out = dispatch(request["verb"], request["args"])
    except (ValueError, KeyError, TypeError) as exc:
        out = {"error": f"unreadable request: {exc}"}
    sys.stdout.write(json.dumps(out) + "\n")
    return 0 if "error" not in out else 1


if __name__ == "__main__":
    raise SystemExit(main())
