# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The numeric verbs of the text stages, run in the interpreter that has numpy (the supervisor needs only the standard library).

The supervisor starts this script with the configured interpreter (orchard/text_remote.py), writes one JSON object on its standard input and reads one from its
standard output:

    {"verb": "references" | "finish_chip" | "finish_bench" | "gate_reference" | "gate_chip" | "gate_bench", "args": {...}}  ->  {"result": ...}  or  {"error": "..."}

`dispatch` is the same code the tests call in process. Nothing here opens a device.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def dispatch(verb: str, args: dict) -> dict:
    from orchard import text_runtime as rt
    from orchard import text_stages as st
    try:
        if verb == "references":
            rtm = rt.get(args["runtime"])
            if rtm is None:
                return {"error": f"no text runtime named {args['runtime']!r}"}
            notes: list[str] = []
            ctx = rt.TextContext(**args["context"])
            rec = st.run_references(args["stage_dir"], args["run_dir"], rtm, ctx, st.subprocess_runner(args["env"]), timeout_s=float(args.get("timeout_s", rt.REFERENCE_TIMEOUT_S)),
                                    say=notes.append)
            return {"result": {"record": rec, "notes": notes}}
        if verb == "finish_chip":
            return {"result": st.finish_chip(args["stage_dir"], args["run_dir"], args["ref_stage_dir"], args["runtime"])}
        if verb == "finish_bench":
            return {"result": st.finish_bench(args["stage_dir"], args["run_dir"], args["ref_stage_dir"], args["runtime"])}
        if verb == "gate_reference":
            return {"result": st.reference_problems(args["stage_dir"], args["run_dir"])}
        if verb == "gate_chip":
            return {"result": st.chip_problems(args["stage_dir"], args["run_dir"], args["ref_stage_dir"])}
        if verb == "gate_bench":
            return {"result": st.bench_problems(args["stage_dir"], args["run_dir"], args["ref_stage_dir"])}
        return {"error": f"unknown verb {verb!r}"}
    except rt.TextStageError as exc:
        return {"error": str(exc)}
    except (KeyError, TypeError, ValueError) as exc:          # a malformed request
        return {"error": f"bad request for {verb}: {type(exc).__name__}: {exc}"}


class Cancelled(BaseException):
    """The supervisor cancelled the verb (SIGTERM). A BaseException, so the cleanup in `except BaseException` blocks runs and nothing is caught by accident."""


def _on_term(signum, frame):
    raise Cancelled()


def main() -> int:
    import signal
    signal.signal(signal.SIGTERM, _on_term)
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
