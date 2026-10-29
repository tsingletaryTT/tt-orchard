# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The text-model runtime route: which models it serves, and the small adapter boundary a new model plugs into (issues 61, 90, 214, 206).

A text model that the harness's own scripts serve on one chip (a "text full port": the model's code is a set of scripts in this repository, not a bundle or
tt-transformers) takes this route. The run is started with the input `text_runtime=<name>`; stage 0 and stages 1 to 8 are then supervisor code and none
of them boots the local coder. `orchard/stages.py` has the stage table, `orchard/text_stages.py` the gates, and this module the adapter.

The adapter boundary
--------------------
A runtime is a `TextRuntime` subclass registered in `REGISTRY`. The harness never imports the runtime's scripts: it starts them as child processes, so the
supervisor stays standard-library only and a chip script cannot hold the supervisor's device. The adapter supplies four things.

1. Facts: `name`, `model_ids`, `package_name`, `license_id`, `scripts_dir`, `shipped_files`, `variants` (each a chip configuration; the first is the default
   and the one the gates judge), `context_limit`, `limits` (what the adapter's port does not cover), `config_problems(model_dir)` (why the runtime cannot serve
   this config; [] when it can) and `dtype_policy(variant)` (what the variant's weights are, for the decode floor).
2. Weights: `weights_path(model_dir, cache_dir)` is the single safetensors file the scripts read; `prepare_argv(ctx)` is a command that makes it, or None when
   it exists. A runtime whose repository already ships safetensors can return the file in place.
3. Commands, each an argv list started in `ctx.out_dir`'s run, each writing the files named in its contract below:
       reference_argv(ctx)               writes <ctx.out_dir>/float32, bfloat16 and a second bfloat16 run (`floor_runs`); recorded runs in the layout of
                                         orchard/text_reference.py. The float32 run chooses the greedy token path; the bfloat16 runs follow it.
       ppl_reference_argv(ctx, out)      writes the JSON `out` (keys tokens, perplexity, mean_nll, logprobs): float32 on the CPU, teacher-forced, on a fixed window
       agree_argv(ctx, variant, out)     records the chip run `out` (a directory) in the same layout, teacher-forced on the float32 run's tokens (`ctx.ref_dir`)
       ppl_chip_argv(ctx, variant, out)  writes the JSON `out`, same window and keys, on the chip
       bench_argv(ctx, variant, out, steps, repeats)   writes the JSON `out`: a record from `orchard.resident.summarise_runs` (keys label, summary, runs, config,
                                         share_of_floor_median) with one warm-up and `repeats` measured runs in one process
4. Serving, for the package: `serve_argv(variant, port)` and `shipped_files`.

Everything the gates trust is re-derived from the recorded arrays and JSON files. A script that writes a number the arrays do not support is caught.
`ctx` is a `TextContext`. The adapter never opens a device itself; `orchard/skills/text-chip-templates/text_chip_test.py` runs its chip commands under the
supervisor's lease.

This module opens no device and imports no ttnn, torch or numpy.
"""
from __future__ import annotations

import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path

from orchard import resident, safeio

SCHEMA = 1
RUNTIME_INPUT = "text_runtime"            # the run_start input that selects the route
REFERENCE_FILE = "text_reference.json"    # stage 1's record
RESULT_FILE = "text_result.json"          # stage 2's record
BENCH_FILE = "text_bench.json"            # stage 6's record
STAGE_CONFIG_FILE = "text_config.json"    # the planner's input to a stage
CHIP_CONFIG_FILE = "text_chip_config.json"
STAGE_DISK_GB = 25.0                      # the weights as safetensors (8 GB for a 4B model) plus the runs' arrays and the cache of the converted file
CHIP_DEADLINE_S = 3000.0
BENCH_DEADLINE_S = 2400.0
REFERENCE_TIMEOUT_S = 7200.0
DECODE_TOKENS = 16
PPL_TOKENS = 768
BENCH_STEPS = 64
TEMPLATES = Path(__file__).with_name("skills") / "text-chip-templates"
CHIP_TEST = TEMPLATES / "text_chip_test.py"
REPO = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = ("Write a short paragraph about how a lighthouse keeper spends a stormy night, mentioning the lamp, the logbook, "
                  "and a ship in trouble.")
ROLE_REFERENCE, ROLE_FLOOR, ROLE_FLOOR_NATIVE = "reference", "floor_bfloat16", "floor_native"
REFERENCE_DIRS = {ROLE_REFERENCE: "float32", ROLE_FLOOR: "bfloat16", ROLE_FLOOR_NATIVE: "native"}


class TextStageError(Exception):
    """A problem a text stage reports as its failure reason."""


@dataclass(frozen=True)
class Variant:
    """One chip configuration of a runtime. `name` is a file-name-safe word; `args` are the runtime's own flags for it; `description` is for the card."""
    name: str
    args: tuple[str, ...]
    description: str


@dataclass(frozen=True)
class TextPaths:
    """The interpreter the text stages use for the CPU reference and for the chip scripts: one environment with torch, transformers, safetensors and ttnn."""
    python: str

    def record(self) -> dict:
        return {"python": self.python}

    @classmethod
    def from_record(cls, rec: dict) -> "TextPaths":
        return cls(rec["python"])


def text_paths(operator_home, environ=None) -> TextPaths:
    env = os.environ if environ is None else environ
    return TextPaths(env.get("ORCHARD_TEXT_PYTHON") or str(Path(operator_home) / ".tenstorrent-venv" / "bin" / "python"))


@dataclass(frozen=True)
class TextContext:
    """What a command needs. All paths are absolute strings."""
    python: str
    model_dir: str
    weights: str
    out_dir: str                    # where the command writes (the stage directory for stage commands)
    ref_dir: str = ""               # stage 1's reference directory (holds float32/, bfloat16/, native/)
    prompt_file: str = ""
    decode_tokens: int = DECODE_TOKENS
    ppl_parquet: str = ""
    ppl_tokens: int = PPL_TOKENS
    max_len: int = 256


class TextRuntime:
    """The base class of a text runtime adapter. See the module text for the contract. A subclass sets the class attributes and the methods marked abstract."""
    name = ""
    model_ids: tuple[str, ...] = ()
    package_name = ""
    license_id = ""                 # the SPDX id the model's card states (the package gate compares it with the card's front matter)
    scripts_dir: Path = REPO
    shipped_files: tuple[str, ...] = ()     # names under scripts_dir that a package carries
    variants: tuple[Variant, ...] = ()
    context_limit = 0               # the context length the package states: the server's default cache length
    limits: tuple[str, ...] = ()
    min_layers = 1

    # -- facts
    def config_problems(self, model_dir) -> list[str]:
        raise NotImplementedError

    def dtype_policy(self, variant: Variant) -> dict:
        """The weights' dtype by decode-floor group (orchard/decode_floor.py): {"default": "bf16", "dense_mlp": "bf16"}."""
        raise NotImplementedError

    def n_layers(self, config: dict) -> int:
        return int(config["num_hidden_layers"])

    # -- weights
    def weights_path(self, model_dir, cache_dir) -> Path:
        raise NotImplementedError

    def prepare_argv(self, ctx: TextContext) -> list[str] | None:
        return None

    # -- commands (abstract)
    def reference_argv(self, ctx: TextContext) -> list[str]:
        raise NotImplementedError

    def ppl_reference_argv(self, ctx: TextContext, out: str) -> list[str]:
        raise NotImplementedError

    def agree_argv(self, ctx: TextContext, variant: Variant, out: str) -> list[str]:
        raise NotImplementedError

    def ppl_chip_argv(self, ctx: TextContext, variant: Variant, out: str) -> list[str]:
        raise NotImplementedError

    def bench_argv(self, ctx: TextContext, variant: Variant, out: str, steps: int, repeats: int) -> list[str]:
        raise NotImplementedError

    def serve_argv(self, variant: Variant, port: int) -> list[str]:
        raise NotImplementedError

    # -- helpers
    def default_variant(self) -> Variant:
        return self.variants[0]

    def variant(self, name: str) -> Variant:
        for v in self.variants:
            if v.name == name:
                return v
        raise TextStageError(f"{self.name} has no variant {name!r}")

    def script(self, name: str) -> str:
        return str(self.scripts_dir / name)


class MiniCPM3(TextRuntime):
    """openbmb/MiniCPM3-4B on one Blackhole chip (scripts/minicpm3, PR 210): absorbed multi-latent attention, 62 layers, a dense gated-SiLU MLP."""
    name = "minicpm3"
    model_ids = ("openbmb/MiniCPM3-4B",)
    package_name = "minicpm3-4b-blackhole"
    license_id = "apache-2.0"
    scripts_dir = REPO / "scripts" / "minicpm3"
    shipped_files = ("chip_model.py", "ref_model.py", "native_model.py", "make_reference.py", "run_chip.py", "ppl_check.py", "qual_check.py", "ctx_bench.py",
                     "serve.py", "summarize.py", "convert_weights.py")
    variants = (Variant("bf16", ("--opt", "3", "--upd", "paged"), "all weights bfloat16, HiFi2 for the MLP"),
                Variant("mlp8", ("--opt", "3", "--upd", "paged", "--mlp-dtype", "bfp8", "--mlp-fidelity", "lofi"),
                        "MLP weights 8-bit block floating point with LoFi, everything else bfloat16"))
    context_limit = 2048           # the server's default cache length (serve.py --max-len); the model's own limit is 32768
    limits = ("One chip only on this route. The two-chip tensor-parallel version (chip_model_tp.py) is in the scripts and was not run by the stages.",
              "Decoding is greedy only; the server ignores temperature and top_p.",
              "The server feeds the prompt one token per decode step, so a long prompt starts slowly. The batched prefill is used by the benchmark only: in the server it hung the chip on the second request after a trace (found by the package's boot check).")

    def config_problems(self, model_dir) -> list[str]:
        try:
            cfg = safeio.read_json((Path(model_dir) / "config.json").resolve(), max_bytes=1 << 20, expect=dict)
        except (OSError, ValueError, safeio.SafeReadError) as exc:
            return [f"config.json cannot be read: {exc}"]
        out = []
        if cfg.get("model_type") != "minicpm3":
            out.append(f"model_type is {cfg.get('model_type')!r}; this runtime serves minicpm3")
        for key in ("num_hidden_layers", "hidden_size", "num_attention_heads", "q_lora_rank", "kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim",
                    "intermediate_size", "vocab_size", "scale_emb", "scale_depth", "dim_model_base", "rms_norm_eps", "rope_scaling"):
            if key not in cfg:
                out.append(f"config.json has no {key}")
        if not out and cfg.get("num_key_value_heads") not in (None, cfg.get("num_attention_heads")):
            out.append("the chip port assumes as many key-value heads as attention heads")
        return out

    def dtype_policy(self, variant):
        return {"default": "bf16", "dense_mlp": "bfp8" if "bfp8" in variant.args else "bf16"}

    def weights_path(self, model_dir, cache_dir):
        d = Path(model_dir).resolve()
        if (d / "model.safetensors").is_file():
            return d / "model.safetensors"
        return Path(cache_dir) / f"{d.name}.safetensors"

    def prepare_argv(self, ctx):
        if Path(ctx.weights).is_file():
            return None
        return [ctx.python, self.script("convert_weights.py"), "--model-dir", ctx.model_dir, "--out", ctx.weights]

    def reference_argv(self, ctx):
        argv = [ctx.python, self.script("make_reference.py"), "--model-dir", ctx.model_dir, "--weights", ctx.weights, "--out-dir", ctx.out_dir,
                "--decode", str(ctx.decode_tokens)]
        return argv + (["--prompt-file", ctx.prompt_file] if ctx.prompt_file else [])

    def ppl_reference_argv(self, ctx, out):
        return [ctx.python, self.script("ppl_check.py"), "ref", "--parquet", ctx.ppl_parquet, "--tokens", str(ctx.ppl_tokens), "--model-dir", ctx.model_dir,
                "--weights", ctx.weights, "--out", out]

    def agree_argv(self, ctx, variant, out):
        argv = [ctx.python, self.script("run_chip.py"), "agree", *variant.args, "--model-dir", ctx.model_dir, "--weights", ctx.weights, "--ref",
                str(Path(ctx.ref_dir) / "float32"), "--out", out, "--max-len", str(ctx.max_len)]
        return argv + (["--prompt-file", ctx.prompt_file] if ctx.prompt_file else [])

    def ppl_chip_argv(self, ctx, variant, out):
        keep, args = [], list(variant.args)
        for i, a in enumerate(args):          # ppl_check takes --opt, --mlp-dtype and --mlp-fidelity only
            if a in ("--opt", "--mlp-dtype", "--mlp-fidelity"):
                keep += args[i:i + 2]
        return [ctx.python, self.script("ppl_check.py"), "chip", *keep, "--parquet", ctx.ppl_parquet, "--tokens", str(ctx.ppl_tokens), "--model-dir",
                ctx.model_dir, "--weights", ctx.weights, "--ref", str(Path(ctx.ref_dir).parent / "ppl_reference.json"), "--out", out]

    def bench_argv(self, ctx, variant, out, steps, repeats):
        argv = [ctx.python, self.script("run_chip.py"), "bench", *variant.args, "--trace", "--batched-prefill", "--model-dir", ctx.model_dir, "--weights",
                ctx.weights, "--steps", str(steps), "--repeats", str(repeats), "--out", out, "--max-len", str(ctx.max_len)]
        return argv + (["--prompt-file", ctx.prompt_file] if ctx.prompt_file else [])

    def serve_argv(self, variant, port):
        return ["python", "scripts/serve.py", "--port", str(port)] + (["--mlp-dtype", "bfp8", "--mlp-fidelity", "lofi"] if "bfp8" in variant.args else [])


REGISTRY: dict[str, TextRuntime] = {r.name: r for r in (MiniCPM3(),)}


def runtime_for_model(model_id) -> TextRuntime | None:
    """The registered runtime that serves a repository id (a revision after `@` is ignored), or None."""
    base = str(model_id or "").split("@", 1)[0]
    return next((r for r in REGISTRY.values() if base in r.model_ids), None)


def get(name) -> TextRuntime | None:
    return REGISTRY.get(name) if isinstance(name, str) else None


def selected(entries: list[dict]) -> str | None:
    """The runtime name in run_start's inputs, when the run was started on this route and the name is registered. The run's choice is read from the ledger so a
    resume keeps it."""
    for e in entries:
        if e.get("event") == "run_start":
            value = (e.get("data", {}).get("inputs") or {}).get(RUNTIME_INPUT)
            return value if isinstance(value, str) and value in REGISTRY else None
    return None


TEXT_RUNTIME_SELECTED = "text runtime selected"


def selection(entries: list[dict]) -> dict | None:
    """The run's last `TEXT_RUNTIME_SELECTED` decision (written by stage 0: the runtime name, the interpreter, the script hashes), or None."""
    found = None
    for e in entries:
        if e.get("event") == "decision" and e.get("data", {}).get("decision") == TEXT_RUNTIME_SELECTED:
            found = e["data"]
    return found


def selection_from_run_dir(run_dir) -> tuple[str, TextPaths] | None:
    """(runtime name, interpreter paths) of the run's selection, read from its ledger file (a gate gets only directories). None when there is none."""
    entries = []
    try:
        with open(Path(run_dir) / "ledger.jsonl", encoding="utf-8") as f:
            for line in f:
                if TEXT_RUNTIME_SELECTED in line:
                    try:
                        entries.append(json.loads(line))
                    except ValueError:
                        continue
    except OSError:
        return None
    rec = selection(entries)
    try:
        return rec["runtime"], TextPaths.from_record(rec["paths"])
    except (KeyError, TypeError):
        return None


def script_hashes(rtm: TextRuntime) -> dict:
    """sha256 of each script the runtime ships, recorded with the selection so a later change of the files is visible."""
    import hashlib
    return {n: hashlib.sha256((rtm.scripts_dir / n).read_bytes()).hexdigest() for n in rtm.shipped_files if (rtm.scripts_dir / n).is_file()}


def environment_problems(paths: TextPaths) -> list[str]:
    if not os.access(paths.python, os.X_OK):
        return [f"the interpreter {paths.python} is missing; set ORCHARD_TEXT_PYTHON"]
    return []


def card_license(model_dir) -> str | None:
    """The `license:` value in the front matter of the model directory's README.md (the Hub card), or None. Read with a size limit."""
    try:
        text = safeio.read_bytes((Path(model_dir) / "README.md").resolve(), max_bytes=2 << 20).decode("utf-8", "replace")
    except (OSError, ValueError, safeio.SafeReadError):
        return None
    if not text.startswith("---"):
        return None
    for line in text.split("---", 2)[1].splitlines():
        if line.startswith("license:"):
            return line.split(":", 1)[1].strip() or None
    return None


def model_facts(model_dir, rtm: TextRuntime) -> dict:
    """What stage 0 records about the model directory: the configuration's sha256 and size facts, the card's license, the weight files, and whether the runtime
    accepts the configuration."""
    import hashlib
    d = Path(model_dir).resolve()
    cfg_file = (d / "config.json").resolve()          # a Hugging Face snapshot holds the file as a link into blobs/
    cfg = safeio.read_json(cfg_file, max_bytes=1 << 20, expect=dict)
    weights = sorted(p.name for p in d.iterdir() if p.suffix in (".safetensors", ".bin") and p.is_file())
    return {"config_sha256": hashlib.sha256(safeio.read_bytes(cfg_file, max_bytes=1 << 20)).hexdigest(), "architectures": cfg.get("architectures"),
            "model_type": cfg.get("model_type"), "n_layers": rtm.n_layers(cfg), "hidden_size": cfg.get("hidden_size"), "card_license": card_license(d),
            "weight_files": weights, "config_problems": rtm.config_problems(d)}


def prompt_file_for(stage_dir) -> str:
    return str(Path(stage_dir) / "prompt.txt")


def find_ppl_parquet(hf_home, explicit: str | None = None) -> str | None:
    """The WikiText-2 (raw) test parquet the perplexity window is cut from: the run's explicit input, else the Hugging Face cache's copy."""
    if explicit:
        return explicit if Path(explicit).is_file() else None
    for base in (Path(hf_home) / "hub", Path(hf_home)):
        hits = sorted(base.glob("datasets--Salesforce--wikitext/snapshots/*/wikitext-2-raw-v1/test-*.parquet"))
        if hits:
            return str(hits[-1])
    return None


def write_json(path, value) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_json(path, what: str) -> dict:
    try:
        return safeio.read_json(path, max_bytes=8 << 20, expect=dict)
    except (OSError, ValueError, safeio.SafeReadError) as exc:
        raise TextStageError(f"{what} cannot be read: {exc}") from exc


def adapter_problems(rtm: TextRuntime, ctx: TextContext | None = None) -> list[str]:
    """Why an adapter breaks the boundary in the module text, or []. A new adapter's tests call this with a context: it checks the facts, that each shipped file
    exists, that every command is a list of strings starting with the context's interpreter, that a variant name is a file-name-safe word and unique, and that the
    default variant comes first. It runs no command."""
    out = []
    for attr in ("name", "package_name", "license_id"):
        if not isinstance(getattr(rtm, attr), str) or not getattr(rtm, attr):
            out.append(f"{attr} must be a non-empty string")
    if not rtm.model_ids:
        out.append("model_ids is empty")
    if not rtm.variants:
        return out + ["variants is empty"]
    names = [v.name for v in rtm.variants]
    if len(set(names)) != len(names) or not all(n.replace("_", "").isalnum() and n == n.lower() for n in names):
        out.append(f"variant names must be unique lowercase words, got {names}")
    out += [f"shipped file {f} is not in {rtm.scripts_dir}" for f in rtm.shipped_files if not (rtm.scripts_dir / f).is_file()]
    if rtm.context_limit <= 0:
        out.append("context_limit must be the context length the package states")
    if ctx is not None:
        v = rtm.default_variant()
        cmds = {"reference": rtm.reference_argv(ctx), "ppl_reference": rtm.ppl_reference_argv(ctx, "out.json"), "agree": rtm.agree_argv(ctx, v, "out"),
                "ppl_chip": rtm.ppl_chip_argv(ctx, v, "out.json"), "bench": rtm.bench_argv(ctx, v, "out.json", BENCH_STEPS, resident.MIN_REPEATS)}
        for name, argv in cmds.items():
            if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
                out.append(f"{name}_argv must be a non-empty list of strings")
            elif argv[0] != ctx.python:
                out.append(f"{name}_argv must start with the context's interpreter")
        if not isinstance(rtm.dtype_policy(v), dict):
            out.append("dtype_policy must return a dict")
    return out


def chip_config_commands(rt: TextRuntime, ctx: TextContext, stage: int, repeats: int = resident.MIN_REPEATS) -> list[dict]:
    """The commands the hardware phase runs for stage 2 (`stage` 2: agreement and perplexity for each variant) or stage 6 (benchmarks). Each is
    {name, argv, outputs}; the orchestrator runs them in order and records each one's exit status."""
    out = Path(ctx.out_dir)
    cmds = []
    for v in rt.variants:
        if stage == 2:
            cmds.append({"name": f"agree_{v.name}", "argv": rt.agree_argv(ctx, v, str(out / "chip" / v.name)), "outputs": [f"chip/{v.name}/manifest.json"]})
            cmds.append({"name": f"ppl_{v.name}", "argv": rt.ppl_chip_argv(ctx, v, str(out / f"ppl_chip_{v.name}.json")), "outputs": [f"ppl_chip_{v.name}.json"]})
        else:
            cmds.append({"name": f"bench_{v.name}", "argv": rt.bench_argv(ctx, v, str(out / f"bench_{v.name}.json"), BENCH_STEPS, repeats),
                         "outputs": [f"bench_{v.name}.json"]})
    return cmds


def write_chip_config(stage_dir, commands: list[dict]) -> Path:
    path = Path(stage_dir) / CHIP_CONFIG_FILE
    write_json(path, {"schema": SCHEMA, "commands": commands})
    return path


def chip_test_command(paths: TextPaths, stage_dir_abs: Path) -> str:
    """The one command the hardware phase runs under the lease."""
    return shlex.join([paths.python, str(CHIP_TEST), "--config", str(Path(stage_dir_abs) / CHIP_CONFIG_FILE)])
