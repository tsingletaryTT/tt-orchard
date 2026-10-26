# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 0 of a stock-runtime run: is this model one that tt-transformers already runs?

tt-transformers (tt-metal's `models/tt_transformers`) keeps a `model_params/<name>/config.json` for each model it supports. When
research finds such a directory for the model (or for its stated base) and no installed bundle serves the base, the run takes the
stock-runtime path: this module is stage 0, written as supervisor code because the comparison is with a tt-metal directory and
not with a weights snapshot.

The measurement decides, as it does for the other classes. The model's config must equal the `model_params` config on the keys
that decide the architecture, and its tensors must be in the layout tt-transformers loads (Hugging Face Llama or Qwen2 names, the
shapes the config implies, a float dtype, no extra tensors). Anything else is `full-port`, with the reasons, and the run then stops
as any full port does. Nothing here opens a device, loads the weights or downloads anything: it reads `config.json` and the
safetensors headers.

The record never holds a machine path: the parameters directory is named relative to tt-metal."""
from __future__ import annotations

import json
import re
import struct
import subprocess
from pathlib import Path

PARAMS_SUB = Path("models") / "tt_transformers" / "model_params"
MESH_DEVICE = {1: "P150", 2: "P300", 4: "P150x4"}      # chips -> the board name tt-transformers uses (determine_device_name)
STOCK_DEADLINE_S = 5400.0                              # choice: one hardware test: a cold weight load of an 8B model, the token-accuracy run
                                                       # and a short free run. Unmeasured; the first real run replaces it.
NAME_OK = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,100}")
MAX_HEADER = 64 * 1024 * 1024           # a larger claimed header is refused, as headers.MAX_HEADER_READ does for research
FLOATS = ("BF16", "F16", "F32")
SUPPORTED = {"LlamaForCausalLM": False, "Qwen2ForCausalLM": True}      # architecture -> has attention biases
# Config keys that decide the architecture. A context length, a dtype and a library version are left out on purpose: a fine-tune
# may change them without changing the model code.
DECISIVE = ("architectures", "model_type", "hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
            "intermediate_size", "vocab_size", "rope_theta", "rope_scaling", "head_dim", "tie_word_embeddings", "hidden_act",
            "rms_norm_eps", "attention_bias", "mlp_bias")
DEFAULTS = {"tie_word_embeddings": False, "attention_bias": False, "mlp_bias": False, "rope_scaling": None}
ROPE_FIELDS = ("rope_type", "factor", "low_freq_factor", "high_freq_factor", "original_max_position_embeddings")


def find_params(tt_metal_dir, name) -> Path | None:
    """`<tt-metal>/models/tt_transformers/model_params/<name>` when it holds a config.json. The name is one path part."""
    if not isinstance(name, str) or not NAME_OK.fullmatch(name):
        return None
    d = Path(tt_metal_dir) / PARAMS_SUB / name
    return d if (d / "config.json").is_file() else None


def support_row(run_dir, name: str) -> dict:
    """What research recorded about tt-transformers support for `name`: the model_params row (which id matched) with the boards and
    Blackhole fact of the README row when it has them. An absent or unreadable report gives an empty row."""
    empty = {"via": None, "matched": None, "boards": [], "blackhole": None}
    try:
        rows = json.loads((Path(run_dir) / "research" / "base-candidates.json").read_text(encoding="utf-8")).get("tt_transformers")
    except (OSError, ValueError, AttributeError):
        return empty
    rows = [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []
    params = next((r for r in rows if r.get("via") == "model_params" and r.get("name") == name), None)
    readme = next((r for r in rows if r.get("via") == "readme" and r.get("boards")), None)
    if params is None:
        return empty
    return {"via": "model_params", "matched": params.get("matched"),
            "boards": list(readme["boards"]) if readme else list(params.get("boards") or []),
            "blackhole": readme.get("blackhole") if readme else params.get("blackhole")}


def tt_metal_tree(tt_metal_dir) -> dict:
    """The host tree's tag and whether it has uncommitted changes (read only: describe and status). Unknown values are None."""
    def git(*args):
        try:
            done = subprocess.run(["git", "-C", str(tt_metal_dir), *args], capture_output=True, text=True, timeout=60,
                                  env={"PATH": "/usr/bin:/bin", "GIT_OPTIONAL_LOCKS": "0"})
        except (OSError, subprocess.SubprocessError):
            return None
        return done.stdout.strip() if done.returncode == 0 else None
    tag = git("describe", "--tags", "--always")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"tag": tag or None, "dirty": None if status is None else bool(status)}


# ---- reading ------------------------------------------------------------------------------------------------------

def _load_config(path: Path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return None, f"{path.name} is not readable JSON: {type(exc).__name__}"
    return (data, None) if isinstance(data, dict) else (None, f"{path.name} is not a JSON object")


def _header(path: Path):
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError("fewer than 8 bytes")
        (n,) = struct.unpack("<Q", raw)
        if n > MAX_HEADER:
            raise ValueError(f"the header claims {n} bytes")
        blob = f.read(n)
        if len(blob) != n:
            raise ValueError("the header is cut short")
    head = json.loads(blob)
    if not isinstance(head, dict):
        raise ValueError("the header is not an object")
    return {k: v for k, v in head.items() if k != "__metadata__"}


def read_tensors(snapshot: Path) -> tuple[dict, list[str]]:
    """{name: (dtype, shape)} from the backbone shards (the index's, else every `model*.safetensors`), and the problems met."""
    problems: list[str] = []
    files: list[str] = []
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        try:
            wm = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
            files = sorted({v for v in wm.values() if isinstance(v, str)}) if isinstance(wm, dict) else []
        except (OSError, ValueError, AttributeError):
            problems.append("model.safetensors.index.json is not readable")
    if not files:
        files = sorted(p.name for p in snapshot.glob("model*.safetensors"))
    if not files:
        return {}, problems + ["the snapshot has no safetensors weights"]
    out: dict = {}
    for name in files:
        path = snapshot / name
        try:
            head = _header(path)
        except (OSError, ValueError, struct.error) as exc:
            problems.append(f"the safetensors header of {name} is not readable: {exc}")
            continue
        for k, v in head.items():
            if isinstance(v, dict) and isinstance(v.get("dtype"), str) and isinstance(v.get("shape"), list):
                out[k] = (v["dtype"], [int(d) for d in v["shape"]])
    return out, problems


# ---- the config comparison -----------------------------------------------------------------------------------------

def effective(cfg: dict, key: str):
    """The value of a decisive key, with the defaults a missing key means (and the head size the heads imply)."""
    if key == "head_dim":
        if cfg.get("head_dim") is not None:
            return cfg["head_dim"]
        h, n = cfg.get("hidden_size"), cfg.get("num_attention_heads")
        return h // n if isinstance(h, int) and isinstance(n, int) and n else None
    if key == "architectures":
        a = cfg.get("architectures")
        return a[0] if isinstance(a, list) and a else a
    value = cfg.get(key, DEFAULTS.get(key))
    if key == "rope_scaling" and isinstance(value, dict):
        norm = {("rope_type" if k == "type" else k): v for k, v in value.items()}
        return {k: norm.get(k) for k in ROPE_FIELDS if norm.get(k) is not None}
    return value


def compare_configs(model: dict, params: dict) -> list[str]:
    return [k for k in DECISIVE if effective(model, k) != effective(params, k)]


# ---- the layout ----------------------------------------------------------------------------------------------------

def expected_layout(cfg: dict, biases: bool) -> dict:
    h, ff, v = cfg["hidden_size"], cfg["intermediate_size"], cfg["vocab_size"]
    nh, nkv = cfg["num_attention_heads"], cfg["num_key_value_heads"]
    hd = effective(cfg, "head_dim")
    want = {"model.embed_tokens.weight": [v, h], "model.norm.weight": [h]}
    if not effective(cfg, "tie_word_embeddings"):
        want["lm_head.weight"] = [v, h]
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}."
        want.update({p + "self_attn.q_proj.weight": [nh * hd, h], p + "self_attn.k_proj.weight": [nkv * hd, h],
                     p + "self_attn.v_proj.weight": [nkv * hd, h], p + "self_attn.o_proj.weight": [h, nh * hd],
                     p + "mlp.gate_proj.weight": [ff, h], p + "mlp.up_proj.weight": [ff, h], p + "mlp.down_proj.weight": [h, ff],
                     p + "input_layernorm.weight": [h], p + "post_attention_layernorm.weight": [h]})
        if biases:
            want.update({p + "self_attn.q_proj.bias": [nh * hd], p + "self_attn.k_proj.bias": [nkv * hd],
                         p + "self_attn.v_proj.bias": [nkv * hd]})
    return want


def layout_problems(cfg: dict, tensors: dict) -> list[str]:
    arch = effective(cfg, "architectures")
    if arch not in SUPPORTED:
        return [f"the architecture {arch!r} is not one whose tensor layout this check knows ({', '.join(SUPPORTED)})"]
    need = ("hidden_size", "intermediate_size", "vocab_size", "num_attention_heads", "num_key_value_heads", "num_hidden_layers")
    if any(isinstance(cfg.get(k), bool) or not isinstance(cfg.get(k), int) or cfg[k] <= 0 for k in need) or effective(cfg, "head_dim") is None:
        return ["config.json lacks the whole-number sizes the layout is computed from"]
    want = expected_layout(cfg, SUPPORTED[arch])
    out = []
    for name, shape in want.items():
        if name not in tensors:
            out.append(f"missing tensor {name}")
        elif tensors[name][1] != shape:
            out.append(f"{name} has shape {tensors[name][1]}, expected {shape}")
    for name in sorted(set(tensors) - set(want)):
        out.append(f"extra tensor {name} (tt-transformers loads no such tensor)")
    for name, (dtype, _s) in sorted(tensors.items()):
        if dtype not in FLOATS:
            out.append(f"{name} has dtype {dtype}; tt-transformers loads {'/'.join(FLOATS)} checkpoints")
            break
    return out


def _count(shape) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


# ---- the stage ----------------------------------------------------------------------------------------------------

def triage(*, stage_dir, run_dir, model_id: str, snapshot, params_dir, support: dict, tt_metal: dict) -> dict:
    """Compare the model with the tt-transformers parameters, write evidence and delta.json in `stage_dir`, return delta."""
    stage_dir, snapshot, params_dir = Path(stage_dir), Path(snapshot), Path(params_dir)
    name = params_dir.name
    evidence = stage_dir / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    reasons: list[str] = []
    mcfg, err = _load_config(snapshot / "config.json")
    if err:
        reasons.append(f"the model's {err}")
    pcfg, err = _load_config(params_dir / "config.json")
    if err:
        reasons.append(f"the tt-transformers {err}")
    differing = compare_configs(mcfg, pcfg) if mcfg is not None and pcfg is not None else []
    if mcfg is not None and pcfg is not None:
        for k in differing:
            reasons.append(f"config differs from tt-transformers' {name} on {k}: model {effective(mcfg, k)!r}, "
                           f"tt-transformers {effective(pcfg, k)!r}")
    tensors, read_problems = read_tensors(snapshot)
    problems = list(read_problems)
    reasons += read_problems
    if mcfg is not None and tensors:
        extra = layout_problems(mcfg, tensors)
        problems += extra
        reasons += [f"layout: {p}" for p in extra]
    cls = "stock-runtime" if not reasons else "full-port"
    params_b = sum(_count(s) for _d, s in tensors.values()) / 1e9 if tensors else 0.0
    layers = mcfg.get("num_hidden_layers") if isinstance(mcfg, dict) else None
    (evidence / "stock-config-compare.json").write_text(json.dumps({
        "name": name, "keys_compared": list(DECISIVE), "keys_differing": differing,
        "model": {k: effective(mcfg, k) for k in DECISIVE} if mcfg is not None else None,
        "tt_transformers": {k: effective(pcfg, k) for k in DECISIVE} if pcfg is not None else None}, indent=2, default=str) + "\n")
    (evidence / "stock-layout.json").write_text(json.dumps({
        "tensors": len(tensors), "params_b": params_b, "problems": problems,
        "dtypes": sorted({d for d, _s in tensors.values()})}, indent=2) + "\n")
    dirty = tt_metal.get("dirty")
    tree = ("a tree with uncommitted changes" if dirty else "a tree whose cleanliness is not known" if dirty is None
            else "an unpushed tree")
    hazards = [{"area": "other", "finding": (
        "The run uses the host tt-metal tree as it is, with a scratch overlay for anything that has to change "
        f"(tag {tt_metal.get('tag')}). It is not a published or pinned build, so a card built from it would come from {tree}.")}]
    if support.get("blackhole") is not True:
        hazards.append({"area": "other", "finding": (
            f"tt-transformers' README row for this model names {', '.join(support.get('boards') or []) or 'no board'}: no Blackhole "
            "board is named, and this box has Blackhole boards. Stage 2 measures whether it runs here.")})
    diffs = [{"area": "config", "finding": (f"The config equals tt-transformers' {name} on every decisive key." if not differing
                                              else f"The config differs from tt-transformers' {name} on {', '.join(differing)}."),
              "evidence": ["stages/0/evidence/stock-config-compare.json"]},
             {"area": "tensors", "finding": (f"{len(tensors)} tensors ({params_b} billion parameters) are in the layout tt-transformers loads."
                                              if not problems else f"{len(problems)} tensor layout problem(s): {problems[0]}"),
              "evidence": ["stages/0/evidence/stock-layout.json"]},
             {"area": "tensor_names", "finding": ("Tensor names match the Hugging Face layout tt-transformers converts from."
                                                   if not problems else "See stock-layout.json for the names that do not."),
              "evidence": ["stages/0/evidence/stock-layout.json"]}]
    delta = {"model": model_id, "nearest_model": f"tt-transformers:{name}", "path": cls if cls == "stock-runtime" else "full-port",
             "class": cls, "sidecars": [], "differences": diffs, "hazards": hazards,
             "stock_runtime": {"name": name, "matched": support.get("matched"), "via": support.get("via"),
                               "boards": list(support.get("boards") or []), "blackhole": support.get("blackhole"),
                               "params_b": params_b, "layers": layers, "config_equal": not differing,
                               "keys_differing": differing, "layout_problems": problems,
                               "params_dir": (PARAMS_SUB / name).as_posix(),
                               "tt_metal": {"tag": tt_metal.get("tag"), "dirty": tt_metal.get("dirty")}}}
    if reasons:
        delta["path_reasons"] = reasons
    (stage_dir / "delta.json").write_text(json.dumps(delta, indent=2) + "\n")
    return delta


def draft_config(*, run_dir, delta, inputs: dict, chips, cache_root) -> tuple[dict | None, list[str]]:
    """stock_config.json for stage 2, from facts the run holds: stage 0's delta.json (the class and the tt-transformers name), the
    run's inputs (the model's snapshot and the parameters directory) and the stage's chip count. Returns (config, []) or
    (None, the problems): nothing is written when a fact is missing."""
    problems: list[str] = []
    rec = delta.get("stock_runtime") if isinstance(delta, dict) and delta.get("class") == "stock-runtime" else None
    if not isinstance(rec, dict) or not isinstance(rec.get("name"), str):
        problems.append("stage 0 did not record the class stock-runtime with a stock record")
    if isinstance(chips, bool) or chips not in MESH_DEVICE:
        problems.append(f"{chips!r} chips has no tt-transformers board name (one of {sorted(MESH_DEVICE)})")
    snapshot, params = inputs.get("model"), inputs.get("stock_params")
    if not snapshot:
        problems.append("the run has no model input (the snapshot)")
    if not params:
        problems.append("the run has no stock_params input")
    elif Path(params).parent.name != "model_params" or Path(params).parent.parent.name != "tt_transformers":
        problems.append(f"stock_params {params!r} is not a models/tt_transformers/model_params/<name> directory")
    if problems:
        return None, problems
    name = rec["name"]
    return {"run_dir": str(run_dir), "model_id": delta.get("model"), "model_name": name, "model_snapshot": str(snapshot),
            "tt_metal_dir": str(Path(params).parents[3]), "scratch_dir": str(Path(run_dir) / "tt-metal-scratch"),
            "mesh_device": MESH_DEVICE[chips], "chips": chips, "optimizations": "accuracy",
            "tt_cache": str(Path(cache_root) / "stock" / name.lower() / MESH_DEVICE[chips]), "deadline_s": STOCK_DEADLINE_S}, []
