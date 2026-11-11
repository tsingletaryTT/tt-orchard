#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Layer-by-layer check of a GENERATED port against the transformers reference, with no coder (issue 696).

The port generator (orchard/port_template.py) writes most of a Qwen4-style port: the hyper-connection blocks, the gated delta net, the PLE
block, the layers composed from them and the rotary helpers. A stub the generator could not write (today the indexer attention) raises
NotImplementedError, so the port cannot run end to end. This script runs the layers that do not depend on a stub, one at a time, on one chip
(or on the fake ttnn of tests/fake_ttnn_compute.py for `--selftest`), and compares the residual stream after each layer with the stream the
transformers model produces for the same ids.

Two phases, which can run in one process (`--phase both`) or in two interpreters:

  reference   Loads the snapshot with the reference loader of the stage 1 template (it handles a dense Qwen4 checkpoint that upstream
              transformers cannot build), runs it in float32 on the CPU, and saves the ids, the output of every decoder layer and the
              logits to a .pt file. Needs torch and transformers. Never imports ttnn.
  device      Generates the port from the snapshot, the construction file, the optional ops inventory and the optional excerpts (the same
              inputs the supervisor's thin-port step uses), imports it, opens one chip, loads the weights and runs the embedding, the
              prelude and each layer in the order of the generated `_forward`. After each layer it reads the stream back and compares it
              with the saved reference. A layer whose functions reach a stub is skipped with the stub's name, and the run stops there (the
              later layers read that layer's output). A layer that raises is reported with the exception, the port function and the line
              of the generated port; the layers before it keep their numbers. When no layer is skipped, the hyper-connection mixer and
              `lm_head` are compared with the reference logits as well. Needs ttnn: the script exits 2 with a message when it is missing.

The stream compared is what the reference decoder layer returns: [1, S, HC * HIDDEN], the HC residual streams side by side (the generated
port keeps it as [1, 1, S_padded, HC * HIDDEN]; the first S rows are compared, because the port pads the ids to a multiple of 32 and every
layer is causal). The logits row is [S, VOCAB].

The result is one line, `LAYER_CHECK_RESULT {json}`, and the same JSON in `--out` when given. Per layer: pcc, max_abs, max_rel (max_abs over
the largest reference value), mean_abs, the count of non-finite values and `ok` (pcc at least PCC_OK and nothing non-finite).

Commands (the second runs on a board, inside a gozer lease; the paths are the caller's):
  python scripts/layer_check_generated.py --phase reference --snapshot SNAPSHOT --reference-out REF.pt
  gozer run --chips 1 --who "claude:layercheck" --reason "layer check of the generated port" -- \\
      python scripts/layer_check_generated.py --phase device --snapshot SNAPSHOT --construction CONSTRUCTION \\
      [--inventory INVENTORY.json] [--excerpts EXCERPTS.md] --reference REF.pt --out RESULT.json
  python scripts/layer_check_generated.py --selftest          (CPU only: tiny checkpoint, fake ttnn)
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import math
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

PCC_OK = 0.99                 # the stream after a layer is ok when it correlates with the reference at least this well (bf16 device, float32 reference)
UPDATE_PCC_OK = 0.95          # the layer's update (its output stream minus its input stream) is ok at this correlation. The residual input dominates
                              # the stream, so the stream's pcc stays high even when a layer's update is wrong; the update is the sensitive number.
UPDATE_REL_OK = 0.25          # and the update's error (L2 norm of the difference over the L2 norm of the reference update) is at most this; pcc ignores scale
PORT_MODULE = "generated_layer_check_port"
RESULT_TAG = "LAYER_CHECK_RESULT"
EXIT_NO_TTNN = 2
# What the generator reads from the reference excerpts about the norm class: without it the norm weights are used as stored, and a class that
# adds 1 to its weight (the Qwen4 norms) gives wrong numbers. The real run passes the stage's reference_excerpts.md.
SELFTEST_EXCERPTS = "class Qwen4ExpTextRMSNorm(nn.Module):\n    def forward(self, x):\n        output = output * (1.0 + self.weight.float())\n"


# ---- the comparison ------------------------------------------------------------------------------------------------------------------------

def compare(got, want, threshold: float = PCC_OK, rel_max: float | None = None) -> dict:
    """Error statistics of `got` against `want` (torch tensors of the same shape): pcc, max_abs, max_rel (against the largest reference
    value), mean_abs, the count of non-finite values in `got`, ref_abs_max, rel_l2 (the L2 norm of the difference over the L2 norm of `want`)
    and ok (pcc at least `threshold`, nothing non-finite, and rel_l2 at most `rel_max` when that is given)."""
    import torch
    got = got.detach().to(torch.float64).reshape(-1)
    want = want.detach().to(torch.float64).reshape(-1)
    if got.shape != want.shape:
        return {"ok": False, "error": f"shape mismatch: got {tuple(got.shape)} values, reference {tuple(want.shape)}"}
    nonfinite = int((~torch.isfinite(got)).sum())
    g = torch.where(torch.isfinite(got), got, torch.zeros_like(got))
    diff = (g - want).abs()
    ref_max = float(want.abs().max()) if want.numel() else 0.0
    gc, wc = g - g.mean(), want - want.mean()
    denom = float(gc.norm() * wc.norm())
    pcc = float((gc * wc).sum() / denom) if denom > 0 else float("nan")
    out = {"pcc": pcc, "max_abs": float(diff.max()), "max_rel": float(diff.max()) / max(ref_max, 1e-12), "mean_abs": float(diff.mean()),
           "nonfinite": nonfinite, "ref_abs_max": ref_max}
    wn = float(want.norm())
    out["rel_l2"] = float((g - want).norm()) / wn if wn > 0 else float("nan")
    out["ok"] = bool(nonfinite == 0 and not math.isnan(pcc) and pcc >= threshold
                     and (rel_max is None or (not math.isnan(out["rel_l2"]) and out["rel_l2"] <= rel_max)))
    return out


# ---- the stubs of a generated port ---------------------------------------------------------------------------------------------------------

def stub_functions(text: str) -> set:
    """Names of the top-level functions that are stubs (the generator's marker comment or its `is not written yet` raise)."""
    from orchard import port_template
    tree = ast.parse(text)
    spans = [(n.lineno, n.end_lineno or n.lineno, n.name) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    names = set()
    for line, _text in port_template.stub_lines(text):
        for first, last, name in spans:
            if first <= line <= last:
                names.add(name)
    return names


def call_graph(text: str) -> dict:
    """{top-level function name: the top-level function names it calls}."""
    tree = ast.parse(text)
    funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    return {name: {c.func.id for c in ast.walk(node) if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in funcs}
            for name, node in funcs.items()}


def reaches(graph: dict, start: str, targets: set) -> list:
    """The targets that `start` reaches through calls to port functions (itself included), sorted."""
    seen, stack, found = set(), [start], set()
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        if f in targets:
            found.add(f)
        stack.extend(graph.get(f, ()))
    return sorted(found)


# ---- phase reference -----------------------------------------------------------------------------------------------------------------------

def make_ids(vocab: int, seq: int, seed: int) -> list:
    """A fixed prompt: `seq` ids drawn from 1 .. vocab - 1 (0 is the pad id of the generated port)."""
    import torch
    g = torch.Generator().manual_seed(seed)
    return [int(t) for t in torch.randint(1, max(vocab, 2), (seq,), generator=g)]


def _load_reference_loader():
    path = REPO / "orchard" / "skills" / "reference-gate-templates" / "reference_gate.py"
    spec = importlib.util.spec_from_file_location("reference_gate_for_layer_check", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def reference_phase(snapshot, seq: int, seed: int, out_path) -> dict:
    """Run the transformers model on a fixed prompt and save the ids, each decoder layer's output and the logits."""
    import torch
    import transformers
    gate = _load_reference_loader()
    model, _report = gate.load_model(torch, transformers, Path(snapshot))
    model = model.float().eval()
    text_model = model.model
    layers = list(text_model.layers)
    outs: dict = {}
    hooks = []
    stream_in: dict = {}
    hooks.append(layers[0].register_forward_pre_hook(lambda _m, args: stream_in.__setitem__("x", args[0].detach().float())))
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(lambda _m, _a, out, i=i: outs.__setitem__(i, (out[0] if isinstance(out, tuple) else out).detach().float())))
    ids = make_ids(int(model.config.vocab_size), seq, seed)
    with torch.no_grad():
        logits = model(input_ids=torch.tensor([ids])).logits
    for h in hooks:
        h.remove()
    if sorted(outs) != list(range(len(layers))):
        raise RuntimeError(f"the reference ran {sorted(outs)} of {len(layers)} layers")
    cfg = model.config
    ref = {"ids": ids, "stream_in": stream_in["x"][0], "layers": [outs[i][0] for i in range(len(layers))], "logits": logits[0].float(),
           "hc_count": int(getattr(cfg, "hc_count", 1)), "hidden_size": int(cfg.hidden_size),
           "stream_layout": "decoder layer output [S, HC * HIDDEN] (batch dimension removed)"}
    torch.save(ref, str(out_path))
    return {"phase": "reference", "layers": len(layers), "seq": len(ids), "path": str(out_path)}


def run_reference(python, snapshot, seq: int, seed: int, out_path) -> dict:
    """The reference phase in `python` (an interpreter with torch and a transformers that runs Qwen4Exp end to end) when it is another
    interpreter than this one; otherwise in this process. The device interpreter's torch may be too old for the model's indexer
    (torch 2.7 refuses its `scatter` index), so the phase can run in another interpreter."""
    if python and os.path.abspath(python) != os.path.abspath(sys.executable):      # not resolved: two venvs can share one binary
        cmd = [python, str(Path(__file__).resolve()), "--phase", "reference", "--snapshot", str(snapshot), "--seq", str(seq),
               "--seed", str(seed), "--reference-out", str(out_path)]
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if done.returncode != 0:
            raise RuntimeError(f"the reference phase failed in {python} (exit {done.returncode}): {done.stderr.strip()[-600:]}")
        return {"phase": "reference", "via": str(python), "path": str(out_path)}
    return reference_phase(snapshot, seq, seed, out_path)


# ---- phase device --------------------------------------------------------------------------------------------------------------------------

def _import_port(text: str, workdir: Path):
    path = workdir / f"{PORT_MODULE}.py"
    path.write_text(text, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(PORT_MODULE, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[PORT_MODULE] = mod
    spec.loader.exec_module(mod)
    return mod, path


def _port_frame(exc: BaseException, port_path: Path) -> dict:
    """The innermost traceback frame in the generated port: its function, line number and code."""
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if Path(f.filename) == port_path]
    if not frames:
        return {}
    f = frames[-1]
    return {"port_function": f.name, "port_line": f.lineno, "code": (f.line or "").strip()[:200]}


def device_phase(*, snapshot, construction, inventory, excerpts, reference_path, device, patch=None) -> dict:
    """Generate the port, run it layer by layer on `device` and compare each layer's stream with the saved reference.
    `patch(module)` may replace port functions before the run (a test uses it to make one raise)."""
    import torch
    from orchard import port_template
    ref = torch.load(str(reference_path), weights_only=True)
    text, plan = port_template.generate_from_files(snapshot=snapshot, construction=construction, inventory=inventory, excerpts=excerpts)
    stubs = stub_functions(text)
    graph = call_graph(text)
    result: dict = {"phase": "device", "pcc_ok": PCC_OK, "stubs": sorted(stubs), "plan_stubs": sorted({s.function for s in plan.stubs}),
                    "stream": ref.get("stream_layout"), "plan_notes": list(plan.notes), "excerpts_given": excerpts is not None,
                    "layers": [], "covered": [], "skipped": [], "raised": [], "ok": False}
    with tempfile.TemporaryDirectory() as tmp:
        mod, port_path = _import_port(text, Path(tmp))
        if patch is not None:
            patch(mod)
        ids = list(ref["ids"])
        n = len(ids)
        n_layers = int(mod.N_LAYERS)
        if n_layers != len(ref["layers"]):
            result["error"] = f"the port has {n_layers} layers and the reference {len(ref['layers'])}"
            return result
        model = mod.load(str(snapshot), device)
        try:
            padded = ids + [mod.PAD_ID] * (-n % mod.PAD_TO)
            uses_rope = "_rope_tables(" in _forward_source(text)
            cos, sin = mod._rope_tables(model, len(padded)) if uses_rope else (None, None)
            x = mod._embedding(model, "model.embed_tokens", padded)
            x = mod._model_prelude(model, x, padded)
            prev_dev = _host_rows(mod, model, x, n)               # the stream the first layer reads, and the reference's
            prev_ref = ref["stream_in"][:n]
            stopped = False
            for i in range(n_layers):
                key = mod.LAYER_VARIANT[i]
                fn = mod.LAYERS[key]
                row = {"layer": i, "variant": key, "function": fn.__name__}
                blocked = reaches(graph, fn.__name__, stubs)
                if stopped or blocked:
                    row.update(status="skipped", why=(f"stub: {', '.join(blocked)}" if blocked else "an earlier layer was not run"))
                    result["skipped"].append(i)
                    result["layers"].append(row)
                    stopped = True
                    continue
                try:
                    x = fn(model, i, x, padded, cos, sin)
                    got = _host_rows(mod, model, x, n)
                    stream = compare(got, ref["layers"][i])
                    update = compare(got - prev_dev, ref["layers"][i] - prev_ref, UPDATE_PCC_OK, UPDATE_REL_OK)
                    row.update(status="compared", stream=stream, update=update, ok=bool(stream["ok"] and update["ok"]))
                    prev_dev, prev_ref = got, ref["layers"][i]
                    result["covered"].append(i)
                except Exception as exc:  # noqa: BLE001 - a layer that raises is a result, the layers before it keep their numbers
                    row.update(status="raised", error=f"{type(exc).__name__}: {exc}"[:600], **_port_frame(exc, port_path))
                    result["raised"].append(i)
                    stopped = True
                result["layers"].append(row)
            if not stopped and not result["skipped"] and not result["raised"]:
                result["layers"].append(_final_row(mod, model, x, n, ref, port_path))
        finally:
            try:
                mod.close(model)
            except Exception:  # noqa: BLE001 - closing must not hide the result
                pass
    compared = [r for r in result["layers"] if r.get("status") == "compared"]
    result["first_bad_layer"] = next((r["layer"] for r in compared if r.get("layer") is not None and not r["ok"]), None)
    result["warnings"] = ([] if excerpts is not None else
                          ["no --excerpts: the generator uses norm weights as stored; a norm class that adds 1 to its weight (the Qwen4 norms) "
                           "then gives wrong numbers. Pass the stage's reference_excerpts.md."])
    result["ok"] = bool(compared) and all(r["ok"] for r in compared) and not result["raised"]
    result["note"] = _note(result, n_layers)
    return result


def _host_rows(mod, model, x, n: int):
    """The first `n` rows of a device stream as a float32 torch tensor [n, features]."""
    got = mod._to_host(model, x).float()
    return got.reshape(-1, got.shape[-1])[:n]


def _forward_source(text: str) -> str:
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_forward":
            return ast.get_source_segment(text, node) or ""
    return ""


def _final_row(mod, model, x, n: int, ref: dict, port_path: Path) -> dict:
    """The hyper-connection mixer and lm_head on the last layer's stream, against the reference logits, when every layer ran."""
    row = {"layer": None, "variant": None, "function": "_model_hyper_connection_mixer + lm_head", "status": "compared", "what": "logits"}
    try:
        mixed = mod._model_hyper_connection_mixer(model, x)
        logits = mod._to_host(model, mod._linear(model, "lm_head", mixed)).float()
        logits = logits.reshape(-1, logits.shape[-1])[:n, :mod.VOCAB]
        row.update(compare(logits, ref["logits"][:n, :mod.VOCAB]))
        row["stream"] = None
    except Exception as exc:  # noqa: BLE001
        row.update(status="raised", error=f"{type(exc).__name__}: {exc}"[:600], **_port_frame(exc, port_path))
    return row


def _note(result: dict, n_layers: int) -> str:
    covered, raised, skipped = result["covered"], result["raised"], result["skipped"]
    if not raised and not skipped:
        return f"all {n_layers} layers and the logits were compared."
    parts = [f"layers {covered} were compared"]
    if raised:
        parts.append(f"layer {raised[0]} raised (see its row)")
    if skipped:
        names = sorted({r["why"] for r in result["layers"] if r.get("status") == "skipped" and r["why"].startswith("stub")})
        parts.append(f"layers {skipped} of {n_layers} were not run ({'; '.join(names) or 'an earlier layer did not run'})")
    return "; ".join(parts) + ". The logits were not compared."


def open_chip(ttnn):
    """One chip, as the supervisor's port check opens it for a single-chip run."""
    return ttnn.open_device(device_id=0), ttnn.close_device


def require_ttnn(importer=None):
    """The ttnn module, or None when it cannot be imported. `importer` is for tests."""
    try:
        return (importer or (lambda: __import__("ttnn")))()
    except ImportError:
        return None


def device_main(args, *, importer=None, device_factory=None) -> int:
    ttnn = require_ttnn(importer)
    if ttnn is None:
        print("layer_check_generated: ttnn cannot be imported; run the device phase with the ttnn Python, inside a gozer lease "
              "(for example: gozer run --chips 1 --who claude:layercheck --reason 'layer check' -- python scripts/layer_check_generated.py --phase device ...)",
              file=sys.stderr)
        return EXIT_NO_TTNN
    device, closer = (device_factory or open_chip)(ttnn)
    try:
        result = device_phase(snapshot=args.snapshot, construction=args.construction, inventory=args.inventory, excerpts=args.excerpts,
                              reference_path=args.reference or args.reference_out, device=device)
    finally:
        try:
            closer(device)
        except Exception:  # noqa: BLE001
            pass
    emit(result, args.out)
    return 0 if result["ok"] else 1


def emit(result: dict, out) -> None:
    line = json.dumps(result, sort_keys=True)
    print(f"{RESULT_TAG} {line}", flush=True)
    if out:
        Path(out).write_text(json.dumps(result, indent=1, sort_keys=True) + "\n", encoding="utf-8")


# ---- the selftest: tiny checkpoint, fake ttnn, no hardware -------------------------------------------------------------------------------

def _install_fake_ttnn():
    sys.path.insert(0, str(REPO / "tests"))
    import fake_ttnn_compute as fake
    m = fake.make()
    sys.modules["ttnn"] = m
    sys.modules["ttnn.experimental"] = m.experimental
    sys.modules["ttnn.transformer"] = m.transformer
    return fake


def control_references(ref: dict) -> dict:
    """Wrong references for the negative controls of the selftest. A harness that cannot tell a right reference from a wrong one proves
    nothing, so the device phase run against each of these must fail every layer it compares.
      wrong_layer   each layer's output replaced by the next layer's (the layer index is wrong);
      zero_update   every layer's output equal to the first layer's input (the layers did nothing): on the tiny model the stream still
                    correlates at 0.97 or better with this, and the update has no variance, which is the case the update guards;
      scaled_update each layer's update multiplied by 1.5 along the reference's own chain: the update still correlates perfectly, so only
                    its relative error catches it."""
    layers = list(ref["layers"])
    stream_in = ref["stream_in"]
    scaled, prev, prev_scaled = [], stream_in, stream_in
    for out in layers:
        prev_scaled = prev_scaled + 1.5 * (out - prev)
        scaled.append(prev_scaled)
        prev = out
    return {"wrong_layer": {**ref, "layers": layers[1:] + layers[:1]},
            "zero_update": {**ref, "layers": [stream_in.clone() for _ in layers]},
            "scaled_update": {**ref, "layers": scaled}}


def selftest(seq: int = 24, reference_python=None) -> dict:
    """Build the tiny dense Qwen4 checkpoint, save the reference, run the generated port through the fake ttnn layer by layer. The result is
    ok when every layer that does not reach a stub was compared and ok, the first layer that does is the first one skipped, and the same
    run against each wrong reference of `control_references` fails on every layer it compares (`control_detected`)."""
    sys.path.insert(0, str(REPO / "scripts" / "tiny_qwen4"))
    fixtures = REPO / "tests" / "fixtures" / "port-generator"
    saved_ttnn = {k: sys.modules.get(k) for k in ("ttnn", "ttnn.experimental", "ttnn.transformer")}
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        import make_checkpoint
        snap = tmpdir / "snapshot"
        rc = make_checkpoint.main(["--out", str(snap), "--variant", "dense", "--no-tokenizer", "--no-reference"])
        if rc != 0:
            return {"ok": False, "error": f"make_checkpoint exited {rc}"}
        ref_path = tmpdir / "reference.pt"
        run_reference(reference_python, snap, seq, 0, ref_path)
        excerpts = tmpdir / "excerpts.md"
        excerpts.write_text(SELFTEST_EXCERPTS, encoding="utf-8")
        fake = _install_fake_ttnn()
        inputs = dict(snapshot=snap, construction=fixtures / "tiny-qwen4-construction.json",
                      inventory=fixtures / "tiny-qwen4-ops-inventory.json", excerpts=excerpts)
        try:
            result = device_phase(reference_path=ref_path, device=fake.MeshDevice(1), **inputs)
            # The negative control: the same run against a reference whose layer outputs are rotated by one layer. A harness that cannot
            # tell the right reference from the wrong one proves nothing, so every compared layer must fail here.
            import torch
            true_ref = torch.load(str(ref_path), weights_only=True)
            detected = {}
            for name, variant in control_references(true_ref).items():
                path = tmpdir / f"reference_{name}.pt"
                torch.save(variant, str(path))
                control = device_phase(reference_path=path, device=fake.MeshDevice(1), **inputs)
                rows = [r for r in control["layers"] if r.get("status") == "compared"]
                detected[name] = bool(rows) and not any(r["ok"] for r in rows)
            result["controls"] = detected
            result["control_detected"] = bool(detected) and all(detected.values())
        finally:
            for k, v in saved_ttnn.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v
    expected_skipped = result["skipped"]
    first_stub_layer = expected_skipped[0] if expected_skipped else None
    covered_ok = result["covered"] == list(range(first_stub_layer if first_stub_layer is not None else len(result["covered"])))
    result["selftest_ok"] = bool(result["ok"] and covered_ok and result["covered"] and not result["raised"] and result.get("control_detected"))
    return result


# ---- the command line ----------------------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--phase", choices=("reference", "device", "both"), default="both")
    ap.add_argument("--selftest", action="store_true", help="CPU only: build the tiny checkpoint, use the fake ttnn, check the harness")
    ap.add_argument("--snapshot", help="the checkpoint directory (config.json and the safetensors file)")
    ap.add_argument("--construction", help="the construction file of the model (stage 2's effective construction)")
    ap.add_argument("--inventory", help="the ops inventory JSON (optional)")
    ap.add_argument("--excerpts", help="the reference excerpts Markdown (optional)")
    ap.add_argument("--seq", type=int, default=24, help="prompt length for the reference phase")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference-out", help="where the reference phase writes its .pt file")
    ap.add_argument("--reference-python", default=os.environ.get("ORCHARD_REFERENCE_PYTHON"),
                    help="interpreter for the reference phase when it is not this one (default: $ORCHARD_REFERENCE_PYTHON)")
    ap.add_argument("--reference", help="the .pt file the device phase compares with (default: --reference-out)")
    ap.add_argument("--out", help="also write the result JSON here")
    args = ap.parse_args(argv)
    if args.selftest:
        res = selftest(args.seq, args.reference_python)
        emit(res, args.out)
        return 0 if res.get("selftest_ok") else 1
    if not args.snapshot:
        ap.error("--snapshot is required")
    if args.phase in ("reference", "both"):
        if not args.reference_out:
            ap.error("--reference-out is required for the reference phase")
        print(json.dumps(run_reference(args.reference_python, args.snapshot, args.seq, args.seed, args.reference_out)), flush=True)
        if args.phase == "reference":
            return 0
    if not args.construction:
        ap.error("--construction is required for the device phase")
    return device_main(args)


if __name__ == "__main__":
    sys.exit(main())
