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
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from orchard.defaults import (PACKAGE_HEALTH_TIMEOUT_S, PACKAGE_INSTALL_TIMEOUT_S,
                              PACKAGE_THIN_TIMEOUT_S, PACKAGE_VERIFY_DEADLINE_S)
from orchard.package_card import CardFacts, Number, non_commercial, read_license, render_card
from orchard.scrub import scrub_package
from orchard.stages import gate_weights_swap

TEMPLATES = Path(__file__).with_name("package_templates")
SWAP_TEMPLATES = Path(__file__).with_name("skills") / "weights-swap-templates"
BASE_CONFIG_FILES = ("config.json", "preprocessor_config.json", "video_preprocessor_config.json")

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


def _once(pattern: str, repl: str, text: str, what: str, flags=0) -> str:
    out, n = re.subn(pattern, lambda m: repl, text, flags=flags)
    if n != 1:
        raise PackageError(f"expected exactly one {what} in run.sh, found {n}")
    return out


def wire_weights(text: str, model_id: str) -> str:
    """Point every weights setting in a generated run.sh at the bundle's model-dir."""
    q = re.escape(model_id)
    text = _once(rf'--model (?:"{q}"|{q})(?=\s)', f"--model {MODEL_DIR}", text, f'--model "{model_id}"')
    text = _once(r" --revision [0-9a-f]{40}(?=\s)", "", text, "--revision <40 hex>")
    text = _once(r" --tokenizer-revision [0-9a-f]{40}(?=\s)", "", text, "--tokenizer-revision <40 hex>")
    text = _once(r"^export HF_MODEL=.*$", "\n".join(WIRING_LINES), text, "export HF_MODEL= line",
                 flags=re.M)
    return _once(r"^exec \"\$\{CMD\[@\]\}\"$", f"{PREPARE_LINE}\n{EXEC_LINE}", text,
                 'exec "${CMD[@]}" line', flags=re.M)


def weights_wiring_problems(text: str, *, nearest_model: str) -> list[str]:
    """What is wrong with a staged run.sh's weights settings; empty when the chips will load the
    weights in model-dir and nothing names the nearest model."""
    problems = []
    lines = text.splitlines()
    for var, want in (("HF_MODEL", WIRING_LINES[0]), ("MODEL_WEIGHTS_DIR", WIRING_LINES[1])):
        sets = [ln for ln in lines if re.match(rf"\s*(?:export\s+)?{var}=", ln)]
        if sets != [want]:
            problems.append(f"run.sh must set {var} exactly once, as {want!r}; found {sets}")
    cmd = [ln for ln in lines if ln.startswith("CMD=(")]
    if len(cmd) != 1 or cmd[0].count("--model ") != 1 or f"--model {MODEL_DIR} " not in cmd[0]:
        problems.append(f"the run.sh command must pass --model {MODEL_DIR} once")
    if re.search(r"--(?:tokenizer-)?revision\b", text):
        problems.append("run.sh pins a --revision, which a local model-dir does not have")
    if PREPARE_LINE not in lines or EXEC_LINE not in lines or lines.index(PREPARE_LINE) > lines.index(EXEC_LINE):
        problems.append("run.sh must run prepare_model_dir.py before it starts vLLM")
    if nearest_model in text:
        problems.append(f"run.sh names the nearest model {nearest_model}")
    return problems


# ---- the run's results and the source bundles ---------------------------------------------------

def _json(path: Path, what: str) -> dict:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackageError(f"{what} ({path}) is missing or not JSON: {exc}") from None
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


def read_run(run_dir) -> RunFacts:
    """What stage 7 needs from stages 0, 2 and 4. Raises PackageError naming what is missing."""
    run = Path(run_dir).resolve()
    delta = _json(run / "stages/0/delta.json", "stage 0's delta")
    if delta.get("path") != "weights-only":
        raise PackageError(f"stage 7 packages weights-only runs; stage 0 chose {delta.get('path')!r}")
    gate = gate_weights_swap(run / "stages/2", run)
    if not gate.ok:
        raise PackageError("stage 2's result does not pass its gate: " + "; ".join(gate.reasons))
    cfg = _json(run / "stages/2/swap_config.json", "stage 2's swap_config.json")
    model_id = delta.get("model")
    if cfg.get("new_model_id") != model_id:
        raise PackageError(f"stage 2 served {cfg.get('new_model_id')!r}; stage 0 names {model_id!r}")
    snap = Path(cfg.get("new_snapshot") or "")
    if not re.fullmatch(r"[0-9a-f]{40}", snap.name) or not snap.is_dir():
        raise PackageError(f"stage 2's new_snapshot {str(snap)!r} is not a pinned snapshot directory")
    source = load_source(cfg.get("bundle_dir") or "")
    if source.weights_repo != delta.get("nearest_model"):
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
    return RunFacts(run, model_id, snap.name, delta["nearest_model"], snap,
                    Path(cfg.get("hf_home") or ""), source, base, passing, license_id)


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
    "--out"})


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
        elif a.startswith("-"):
            raise PackageError(f"option {a} is not one stage 7 passes (--public and --publish "
                               "change visibility or list the repo)")
        else:
            raise PackageError(f"{a!r} is a positional argument; package-thin pushes to a repo id "
                               "given that way")


def thin_argv(source: Source, *, model_id: str, revision: str, name: str, out: Path) -> list[str]:
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
            *[x for k, v in (m.get("env") or {}).items() for x in ("--env", f"{k}={v}")],
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
    argv = thin_argv(src, model_id=facts.model_id, revision=facts.revision, name=name, out=out)
    assert_no_publish(argv)
    log = out.parent / f"{name}.package-thin.log"
    rc = run_logged(argv, log=log, timeout=PACKAGE_THIN_TIMEOUT_S, env=env)
    if rc != 0:
        raise PackageError(f"tt-model package-thin for {name} exited {rc}; the end of {log.name}:\n"
                           + _tail(log))
    run_sh = out / "run.sh"
    text = splice_extra_args(run_sh.read_text(encoding="utf-8"),
                             extra_args_from((src.path / "run.sh").read_text(encoding="utf-8")))
    text = wire_weights(text, facts.model_id)
    problems = weights_wiring_problems(text, nearest_model=facts.nearest_model)
    if problems:
        raise PackageError(f"{name}/run.sh: " + "; ".join(problems))
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
    (verify_dir / "verify_config.json").write_text(json.dumps({
        "run_dir": str(facts.run_dir), "bundle": str(bundle), "port": free_port(),
        "model_id": facts.model_id, "revision": facts.revision, "hf_home": str(hub.parent),
        "health_timeout_s": PACKAGE_HEALTH_TIMEOUT_S}, indent=2))
    shutil.copyfile(TEMPLATES / "verify_bundle.py", verify_dir / "verify_bundle.py")
    shutil.copyfile(SWAP_TEMPLATES / "serve_and_compare.py", verify_dir / "serve_and_compare.py")
    script = os.path.relpath(verify_dir / "verify_bundle.py", facts.run_dir)
    return {"command": f"{shlex.quote(sys.executable)} {script}",
            "deadline_s": PACKAGE_VERIFY_DEADLINE_S, "hf_linked": linked, "hf_missing": missing}
