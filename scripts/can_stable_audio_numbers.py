#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Merge the raw files of the Stable Audio 3 can's benchmark sessions into evidence files, and write the card's numbers from them.

    python3 scripts/can_stable_audio_numbers.py --raw RAW_DIR --evidence EVIDENCE_DIR --table numbers.md [--date 2026-10-30]

`RAW_DIR` holds what the sessions wrote: `<label>-summary.json` and `<label>-<seconds>s.json` for each served session (written by the session driver
around `pkgsrc/tools/audio_client.py`), `recipe-<seconds>s.json` for the recipe package, `cpu-floor.json` (`pkgsrc/tools/cpu_floor.py`), the clip
comparisons, the can's `boot-check.json` and the tt-model session's `ttmodel-summary.json`. The merged files go to `EVIDENCE_DIR` (they are copied into the
bundle's `evidence/`), and `--table` gets the card's numbers section: a paragraph of conditions and a `Number | Value | Label | Evidence` table. Every value
in the table is computed from those files by this script. A value that a file does not hold is not written, and a missing file is an error.

Labels follow docs/benchmark-rules.md: `benchmark` for 3 or more measured runs after a discarded run in a process, `single sample` for fewer.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

SESSIONS = (("s1", "session 1, empty kernel cache"), ("s3", "session 3, empty kernel cache"), ("s2", "session 2, kernel cache on disk"))
LENGTHS = ("10", "30", "60", "120")
MIN_REPEATS = 3                       # docs/benchmark-rules.md


class EvidenceError(Exception):
    pass


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise EvidenceError(f"cannot read {path}: {exc}") from exc


def merge_raw(raw: Path, out: Path) -> list[str]:
    """Write the evidence files from the raw ones. Returns their names."""
    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    made = []

    def put(name, value):
        (out / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
        made.append(name)

    for label, _ in SESSIONS:
        summary = _read(raw / f"{label}-summary.json")
        by_len, health = {}, None
        for run in summary["runs"]:
            data = _read(raw / run["file"])
            by_len[str(run["duration"])] = data["jobs"]
            health = health or data.get("health")
        put(f"session-{label}.json", {"label": label, "ready_s": summary["ready_s"], "host_load_at_start": round(summary["host_load_at_start"], 2),
                                      "host_load_at_end": round(summary["host_load_at_end"], 2), "settings": data["settings"], "jobs_by_seconds": by_len,
                                      "health": health})
    tm = _read(raw / "ttmodel-summary.json")
    by_len = {}
    for name in tm["jobs"]:
        data = _read(raw / name)
        by_len[name.split("-")[1].rstrip("s.json")] = data["jobs"]
    put("session-ttmodel.json", {"label": "tt-model serve of a copy installed with tt-model's installer", "install_s": tm["install_s"], "ready_s": tm["ready_s"],
                                 "host_load_at_ready": round(tm["host_load_at_ready"], 2), "jobs_by_seconds": by_len, "openapi_status": tm["openapi_status"]})
    recipe = {}
    for sec in ("10", "120"):
        recipe[sec] = _read(raw / f"recipe-{sec}s.json")["jobs"]
    put("session-recipe-package.json", {"label": "the recipe package's job server (a new process for each clip)", "jobs_by_seconds": recipe})
    boot = _read(raw / "boot-check.json")
    boot.pop("server_pid", None)
    put("boot-check.json", boot)
    put("cpu-floor.json", _read(raw / "cpu-floor.json"))
    comparison = {**_read(raw / "clip-comparison.json"), **_read(raw / "clip-comparison-ttmodel.json"), **_read(raw / "clip-vs-cpu.json")}
    put("clip-comparison.json", comparison)
    return made


# ---- the table ------------------------------------------------------------------------------------------------------

def _f(x: float) -> str:
    return f"{x:.1f}" if x >= 10 else f"{x:.2f}"


def spread(values: list[float]) -> str:
    """`median (min to max)` of a list; one value is written once."""
    if len(values) == 1:
        return _f(values[0])
    return f"{_f(statistics.median(values))} ({_f(min(values))} to {_f(max(values))})"


def _label(n_measured: int, processes: int = 1) -> str:
    if n_measured >= MIN_REPEATS:
        return f"benchmark ({n_measured} measured runs after 1 discarded run" + (f" in each of {processes} processes)" if processes > 1 else ")")
    return f"single sample ({n_measured} measured run{'s' if n_measured != 1 else ''})"


def _jobs(session: dict, seconds: str) -> list[dict]:
    jobs = session["jobs_by_seconds"].get(seconds, [])
    bad = [j for j in jobs if not j.get("ok")]
    if bad:
        raise EvidenceError(f"a failed job is in the evidence for {seconds} s: {bad[0].get('error')}")
    return jobs


def _rows(ev: Path) -> list[tuple[str, str, str, str]]:
    """The table's rows. The card viewer reads at most 30 rows of a section, so the order puts what matters most first."""
    s = {label: _read(ev / f"session-{label}.json") for label, _ in SESSIONS}
    tm = _read(ev / "session-ttmodel.json")
    recipe = _read(ev / "session-recipe-package.json")
    boot = _read(ev / "boot-check.json")
    cpu = _read(ev / "cpu-floor.json")
    cmp = _read(ev / "clip-comparison.json")
    groups = {k: [] for k in ("start", "ident", "cpu", "recipe", "tm", "firstlen", "warm", "parts")}

    def add(group, name, value, label, files):
        groups[group].append((name, value, label, ", ".join(f"`evidence/{f}`" for f in files)))

    # starts and first requests
    cold = {"session 1": s["s1"], "session 3": s["s3"]}
    first10 = [s["s1"]["jobs_by_seconds"]["10"][0]["server_total_s"], s["s3"]["jobs_by_seconds"]["10"][0]["server_total_s"],
               boot["audio"]["first"]["server_total_s"], tm["jobs_by_seconds"]["10"][0]["server_total_s"]]
    ready = [s["s1"]["ready_s"], s["s3"]["ready_s"], boot["ready_s"], tm["ready_s"]]
    cold_files = ["session-s1.json", "session-s3.json", "boot-check.json", "session-ttmodel.json"]
    add("start", "Server ready after start, empty kernel cache (s)", spread(ready), f"single samples from {len(ready)} processes", cold_files)
    add("start", "First 10 s clip after start, empty kernel cache (s)", spread(first10), f"single samples from {len(first10)} processes", cold_files)
    s2 = s["s2"]
    add("start", "Server ready after start, kernel cache on disk (s)", _f(s2["ready_s"]), "single sample", ["session-s2.json"])
    add("start", "First 10 s clip after start, kernel cache on disk (s)", _f(s2["jobs_by_seconds"]["10"][0]["server_total_s"]), "single sample", ["session-s2.json"])
    per_len = []
    for sec in ("30", "60", "120"):
        firsts = [c["jobs_by_seconds"][sec][0]["server_total_s"] for c in cold.values()]
        per_len.append(f"{sec} s: {spread(firsts)}")
    add("firstlen", "First clip of a new length in a process that has made 10 s clips only, empty kernel cache (s)", "; ".join(per_len), f"single samples from {len(cold)} processes",
        ["session-s1.json", "session-s3.json"])

    # warm generation time per session and length (the first job of a length is the discarded one)
    for label, name in SESSIONS:
        sess = s[label]
        load = f"host load {_f(sess['host_load_at_start'])} at the start, {_f(sess['host_load_at_end'])} at the end"
        for sec in LENGTHS:
            measured = _jobs(sess, sec)[1:]
            if not measured:
                continue
            totals = [j["server_total_s"] for j in measured]
            rtf = statistics.median(j["rtf"] for j in measured)
            add("warm", f"Generation time for a {sec} s clip, {name} (s)", f"{spread(totals)}; {_f(rtf)} s per second of audio", f"{_label(len(measured))}; {load}", [f"session-{label}.json"])

    # the split of the fastest warm 10 s clip
    quiet = min((j for label, _ in SESSIONS for j in s[label]["jobs_by_seconds"]["10"][1:]), key=lambda j: j["server_total_s"])
    add("parts", "Parts of the fastest warm 10 s clip: conditioning, diffusion transformer calls, decoder calls (s)",
        f"{_f(quiet['stages']['conditioning_s'])}, {_f(quiet['stages']['chip_transformer_call_s'])}, {_f(quiet['stages']['chip_decoder_call_s'])} of {_f(quiet['server_total_s'])} in all",
        "single sample (the fastest of the measured 10 s clips of the three sessions)", ["session-s1.json", "session-s2.json", "session-s3.json"])

    # the recipe package, a new process for each clip
    for sec in ("10", "120"):
        jobs = _jobs(recipe, sec)
        later = [j["server_total_s"] for j in jobs[1:]]
        loads = [j["stages"].get("model_load_s") for j in jobs[1:] if j.get("stages", {}).get("model_load_s") is not None]
        add("recipe", f"Recipe package (a new process for each clip): a {sec} s clip, after the first (s)",
            spread(later) + (f"; the model load inside it takes {_f(statistics.median(loads))} s" if loads else ""),
            _label(len(later)) + f"; the first clip took {_f(jobs[0]['server_total_s'])} s", ["session-recipe-package.json"])

    # the tt-model installer path
    add("tm", "tt-model's installer: install time (s)", _f(tm["install_s"]), "single sample", ["session-ttmodel.json"])
    for sec in ("10", "120"):
        later = [j["server_total_s"] for j in tm["jobs_by_seconds"][sec][1:]]
        add("tm", f"tt-model serve: a {sec} s clip after the first (s)", spread(later), _label(len(later)) + f"; host load {_f(tm['host_load_at_ready'])} at ready", ["session-ttmodel.json"])

    # the CPU
    for sec in LENGTHS:
        run = cpu["runs"].get(f"{float(sec)}")
        if run:
            loads = [r["host_loadavg_1m"] for r in run["runs"]]
            add("cpu", f"CPU, float32, {cpu['threads']} threads: a {sec} s clip (s)", spread([r["generate_and_decode_s"] for r in run["runs"]]),
                f"{_label(len(run['runs']))}; host load {_f(min(loads))} to {_f(max(loads))}", ["cpu-floor.json"])

    # identity with the recipe package's clips
    for via, key in (("bundle", "against recipe package"), ("tt-model's installer path", "tt-model installer path against recipe package")):
        same = [(k.split(",")[0], c) for k, c in cmp.items() if (("tt-model" in k) == (via != "bundle")) and "against recipe package" in k]
        if not same:
            raise EvidenceError(f"clip-comparison.json has no comparison of the {via} with the recipe package")
        ok = all(c["identical_samples"] for _, c in same)
        worst = max(c["max_abs_diff"] for _, c in same)
        add("ident", f"Clip samples identical to the recipe package's ({via}; {' and '.join(n for n, _ in same)})", "yes" if ok else f"no (largest sample difference {worst:.4f})",
            "single samples, same prompt, seed and settings", ["clip-comparison.json"])
    order = ("start", "ident", "cpu", "recipe", "tm", "firstlen", "warm", "parts")
    return [row for g in order for row in groups[g]]


def eval_table(ev: Path) -> str:
    """The accuracy table in the layout tt-model's card guidance asks for: eval, this package, reference, prompts."""
    cmp = _read(ev / "clip-comparison.json")
    key = next((k for k in cmp if "against the CPU float32 clip" in k), None)
    if key is None:
        raise EvidenceError("clip-comparison.json has no comparison with the CPU clip")
    c = cmp[key]
    return "\n".join(["| eval | this package | reference | prompts |", "| --- | --- | --- | --- |",
                      f"| Waveform correlation with the float32 CPU clip (10 s, seed 42; Pearson) | {c['pcc']:.4f} | 1 (the CPU clip) | 1 |",
                      f"| Waveform relative L2 error against the float32 CPU clip (10 s, seed 42) | {c['rel_l2']:.4f} | 0 (the CPU clip) | 1 |"])


def numbers_section(evidence: Path, *, date: str, versions: dict) -> str:
    ev = Path(evidence)
    rows = _rows(ev)
    s = {label: _read(ev / f"session-{label}.json") for label, _ in SESSIONS}
    loads = [v for sess in s.values() for v in (sess["host_load_at_start"], sess["host_load_at_end"])]
    lines = [f"Measured on {date} on one chip of a Blackhole p300c board (a two-chip board; the other chip was idle), through the served bundle: the clips are made by `POST /v1/audio/jobs`,"
             " and the times are the server's own time for each job. Every run used the same prompt (`A warm acoustic guitar melody with soft percussion, 100 BPM`), seed 42, 8 steps,"
             " guidance scale 1.0 and the ping-pong sampler, with the autoencoder decoder's transformer layers on the chip. Software: ttnn " + versions["ttnn"] + " from PyPI, torch "
             + versions["torch"] + " (CPU), numpy " + versions["numpy"] + ", Python " + versions["python"] + ". The host (16 cores) was shared with other jobs. Its one-minute load average was "
             + f"{_f(min(loads))} to {_f(max(loads))} at the start and end of the three served sessions (the per-run values are in the evidence files). A label states how many runs are behind a "
             "number. A session is one server process; its first request at each length is the discarded run, and in a process with an empty kernel cache that run compiles kernels. The "
             "recipe package and the CPU rows come from other processes on the same host.", "",
             "Accuracy against the reference: the same prompt and seed on the float32 CPU path of the model library, one clip.", "", eval_table(ev), "",
             "Speed and identity:", "", "| Number | Value | Label | Evidence |", "| --- | --- | --- | --- |"]
    lines += [f"| {n} | {v} | {lab} | {ev_} |" for n, v, lab, ev_ in rows]
    lines += ["", "Values are median (minimum to maximum). A time per second of audio below 1 means a clip is made faster than it plays.",
              "", "Not measured: agreement per sampler step with the CPU reference (the per-step bars of `orchard/step_bars.py` need a teacher-forced bfloat16 CPU run of this model, "
              "which only the ACE-Step runner writes), clips other than the prompt and seed above, and requests that arrive together."]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--ttnn", default="0.80.0")
    ap.add_argument("--torch", default="2.7.1")
    ap.add_argument("--numpy", default="1.26.4")
    ap.add_argument("--python", default="3.12")
    args = ap.parse_args(argv)
    try:
        merge_raw(Path(args.raw), Path(args.evidence))
        text = numbers_section(Path(args.evidence), date=args.date, versions={"ttnn": args.ttnn, "torch": args.torch, "numpy": args.numpy, "python": args.python})
    except (EvidenceError, KeyError) as exc:
        print(f"can_stable_audio_numbers: {exc}", file=sys.stderr)
        return 2
    Path(args.table).write_text(text + "\n", encoding="utf-8")
    print(f"wrote {args.table}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
