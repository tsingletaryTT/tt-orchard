"""A hardware phase that runs several tests: stage 4 on the weights-only path (spec section 5).

This module owns the list of tests one stage runs, one per chip configuration, and the files that
record it. The stage's prepare step writes `hw_tests.json`:

    {"tests": [{"chips": 2, "script": "serve_and_compare.py", "deadline_s": 2400},
               {"chips": 4, "script": "serve_and_compare_container.py", "deadline_s": 3600}]}

Each test's files live in `configs/<chips>/` of the stage directory: the template script and its
`swap_config.json`. The supervisor builds the command itself (`python3 stages/<n>/configs/<chips>/
<script>`), so an agent cannot hand it an arbitrary shell string. `read_plan` refuses a list the
supervisor must not run: a bad entry, a repeated chip count, a required count with no test, deadlines
that add up to more than the stage budget, a tensor cache inside a shared cache directory, or two
configurations that share one tensor cache. The tensor cache is keyed only by layer name and mesh,
so a shared one serves another model's or another mesh's tensors without an error.

After the prepare step, the supervisor writes the validated list to `tests/plan.json` (the stage's
resume marker) and runs the tests in order of chip count, each under its own lease. Each test's
record goes to `tests/<chips>/test-result.json` (orchard/stages.py, `hw_record_path`) as soon as the
test ends, so a supervisor that crashes during the list resumes at the first configuration with no
record. When the list is done, `write_summary` writes the stage's `test-result.json` for the finish
step.

An agent's write_file can write anywhere in its stage directory, tests/ included. `unrecorded`
compares every tests/<chips>/test-result.json with the sha256 the ledger recorded when the
supervisor wrote it, so a record the agent wrote or changed fails the stage.

A test that did not exit 0 may have stopped while it was converting weights into its tensor cache,
and a part-written cache is read back without an error. `suspect_caches` finds those caches in the
ledger, and the supervisor moves such a cache aside (`move_aside`, never a delete) before the next
test that would use it.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from pathlib import Path

from orchard.defaults import CHIPS_PER_BOARD
from orchard.handoff import NOTE_KEYS
from orchard.stages import evidence_record, hw_record_path

SCRIPTS = {"serve_and_compare.py": "bundle", "serve_and_compare_container.py": "container"}
# Tensor caches that belong to installed packages or to hand builds, relative to the operator's
# home. A test's tt_cache must not be inside one of them.
SHARED_CACHE_DIRS = (".cache/tt-model", ".cache/qwen36-src-build")
PLAN = "tests/plan.json"


@dataclass(frozen=True)
class HwTest:
    chips: int
    script: str
    deadline_s: float
    cache: str                    # the tt_cache from configs/<chips>/swap_config.json, resolved

    @property
    def boards(self) -> int:
        return -(-self.chips // CHIPS_PER_BOARD)

    def command(self, stage: int) -> str:
        return f"python3 stages/{stage}/configs/{self.chips}/{self.script}"


def _inside(path: str, root: str) -> bool:
    root = os.path.realpath(root)
    return os.path.commonpath([path, root]) == root


def read_plan(stage_dir, *, required, max_chips: int, budget_s: float,
              home) -> tuple[list[HwTest], list[str]]:
    """The tests in hw_tests.json sorted by chip count, and every reason not to run them."""
    stage_dir = Path(stage_dir)
    try:
        data = json.loads((stage_dir / "hw_tests.json").read_text(encoding="utf-8"))
        note = json.loads((stage_dir / "handoff.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return [], [f"hw_tests.json and handoff.json must exist and be JSON: {exc}"]
    problems = []
    if not isinstance(note, dict) or any(note.get(k) in (None, "") for k in NOTE_KEYS):
        problems.append(f"handoff.json needs {', '.join(NOTE_KEYS)}")
    items = data.get("tests") if isinstance(data, dict) else None
    if not isinstance(items, list) or not items:
        return [], problems + ["hw_tests.json needs a non-empty list under \"tests\""]
    shared = [str(Path(home) / d) for d in SHARED_CACHE_DIRS]
    tests: list[HwTest] = []
    caches: dict[str, int] = {}
    for i, t in enumerate(items):
        where = f"tests[{i}]"
        n = t.get("chips") if isinstance(t, dict) else None
        if isinstance(n, bool) or not isinstance(n, int) or not 1 <= n <= max_chips:
            problems.append(f"{where} needs chips from 1 to {max_chips}, got {n!r}")
            continue
        if any(x.chips == n for x in tests):
            problems.append(f"{where}: the {n}-chip configuration is listed more than once")
            continue
        if t.get("script") not in SCRIPTS:
            problems.append(f"{where} script must be one of {sorted(SCRIPTS)}, got {t.get('script')!r}")
            continue
        d = t.get("deadline_s")
        if isinstance(d, bool) or not isinstance(d, (int, float)) or d <= 0:
            problems.append(f"{where} needs a positive deadline_s, got {d!r}")
            continue
        cdir = stage_dir / "configs" / str(n)
        if not (cdir / t["script"]).is_file():
            problems.append(f"{where}: configs/{n}/{t['script']} does not exist")
            continue
        try:
            cache = json.loads((cdir / "swap_config.json").read_text(encoding="utf-8")).get("tt_cache")
        except (OSError, ValueError, AttributeError) as exc:
            problems.append(f"{where}: configs/{n}/swap_config.json must exist and be a JSON object: {exc}")
            continue
        if not isinstance(cache, str) or not os.path.isabs(cache):
            problems.append(f"{where}: configs/{n}/swap_config.json needs an absolute tt_cache, got {cache!r}")
            continue
        real = os.path.realpath(cache)
        hit = next((s for s in shared if _inside(real, s)), None)
        if hit:
            problems.append(f"{where}: tt_cache {cache} is inside {hit}, a cache another model "
                            "uses; each model and configuration needs a new directory")
            continue
        if real in caches:
            problems.append(f"{where}: tt_cache {cache} is also the {caches[real]}-chip "
                            "configuration's; each configuration needs its own")
            continue
        caches[real] = n
        tests.append(HwTest(n, t["script"], float(d), real))
    listed = {t.chips for t in tests}
    for c in required or ():
        if c not in listed:
            problems.append(f"the {c}-chip configuration is required and hw_tests.json has no test for it")
    total = sum(t.deadline_s for t in tests)
    if total > budget_s:
        problems.append(f"the tests' deadlines add up to {total:g} s, more than the stage budget of "
                        f"{budget_s:g} s")
    return sorted(tests, key=lambda t: t.chips), problems


def _write_json(path: Path, data) -> None:
    """Whole or not at all: a crash never leaves half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def write_plan(stage_dir, tests: list[HwTest]) -> Path:
    path = Path(stage_dir) / PLAN
    _write_json(path, {"tests": [dataclasses.asdict(t) for t in tests]})
    return path


def load_plan(stage_dir) -> list[HwTest] | None:
    """The validated list from tests/plan.json, or None when the file is missing or unreadable."""
    try:
        data = json.loads((Path(stage_dir) / PLAN).read_text(encoding="utf-8"))
        return [HwTest(**t) for t in data["tests"]]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def pending(stage_dir, tests: list[HwTest]) -> list[HwTest]:
    """The tests that have no record yet, in order."""
    return [t for t in tests if not hw_record_path(stage_dir, t.chips).is_file()]


def write_record(stage_dir, chips: int, record: dict) -> Path:
    path = hw_record_path(stage_dir, chips)
    _write_json(path, record)
    return path


def write_summary(stage_dir, tests: list[HwTest]) -> Path:
    """The stage's test-result.json: every configuration's record, for the finish step."""
    out = []
    for t in tests:
        path = hw_record_path(stage_dir, t.chips)
        out.append(json.loads(path.read_text(encoding="utf-8")) if path.is_file()
                   else {"chips": t.chips, "missing": True})
    path = Path(stage_dir) / "test-result.json"
    _write_json(path, {"tests": out})
    return path


def suspect_caches(entries: list[dict], stage: int) -> set[str]:
    """Tensor caches whose latest test started and did not end with exit 0, from the ledger."""
    state: dict[str, bool] = {}
    for e in entries:
        d = e["data"]
        if e["stage"] != stage or not d.get("cache"):
            continue
        if e["event"] == "decision" and d.get("decision") == "hardware test started":
            state[d["cache"]] = False
        elif e["event"] == "evidence" and d.get("what") == "hardware test":
            state[d["cache"]] = d.get("returncode") == 0 and d.get("timed_out") is False
    return {c for c, ok in state.items() if not ok}


def move_aside(cache) -> Path | None:
    """Rename a non-empty cache directory to <cache>.interrupted-<k> and return the new path. An
    absent or empty directory is left as it is (None)."""
    cache = Path(cache)
    if not cache.is_dir() or not any(cache.iterdir()):
        return None
    k = 1
    while (aside := cache.with_name(f"{cache.name}.interrupted-{k}")).exists():
        k += 1
    os.rename(cache, aside)
    return aside


def unrecorded(entries: list[dict], stage_dir, run_dir, stage: int) -> list[str]:
    """A reason for every tests/<chips>/test-result.json whose sha256 is not the one the ledger
    recorded for that path when the supervisor last wrote it."""
    latest: dict[str, str] = {}
    for e in entries:
        d = e["data"]
        if e["event"] == "evidence" and e["stage"] == stage and d.get("what") == "hardware test":
            latest[d.get("path")] = d.get("sha256")
    out = []
    for path in sorted(Path(stage_dir).glob("tests/*/test-result.json")):
        rec = evidence_record(run_dir, path)
        if latest.get(rec["path"]) != rec["sha256"]:
            out.append(f"{rec['path']} was not written by the supervisor: its sha256 is not the one "
                       "the ledger recorded")
    return out
