# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Per-step agreement bars for audio models with a sampler loop, set from a teacher-forced floor (issue 262).

Why this exists. An end-to-end comparison of a sampler run on the chip with a float32 CPU run feeds each step's error into the next step. For ACE-Step 1.5
turbo (8 steps) the same two stages in bfloat16 on the CPU already fail the end-to-end bars by more than the chip does, so `orchard.noise_floor` cannot
set usable bars from that floor. A per-step check holds the inputs equal: step k of the run under test receives the recorded float32 input of step k, and its
velocity is compared with the recorded float32 velocity of step k. Nothing compounds. The decoder gets the same treatment: it decodes the float32 run's final
latent.

Three recorded files go in (JSON written by `scripts/acestep/acestep_run.py`):
  floor    a teacher-forced CPU run in bfloat16 (`--teacher-forced RUN --cpu-dtype bfloat16`). It is the error that bfloat16 arithmetic alone makes on each step.
  chip     the chip's transformer and decoder on the same inputs (`--stepcheck RUN`).
  controls computed here from the float32 run's own velocities: the next step's velocity (a wrong answer of the right size), and a stated relative noise.
The bar of step k is `margin` times the floor of step k (never below `MIN_FLOOR`). The decoder has one bar by the same rule. The bars are usable when the
chip passes at every step and the decoder, and the wrong-step control fails at every step. The noise control is not a condition: it reports, per step, whether
noise of the stated size gets under the bar. The bar itself is the largest relative error that can hide at that step (the detection limit), so the result
lists it. A bar that the wrong-step control passes is refused, not widened.

The floor is measured with the encoders and the sampler in float32 and only the transformer and the decoder in bfloat16, the same split as the chip.
It does not contain the chip's own arithmetic (different kernels, different accumulation order); the margin leaves room for that, and the chip result
says how much was used.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

MARGIN = 2.0              # bar = MARGIN x floor; room for the chip's arithmetic differing from the CPU's (on ACE-Step the chip's decoder error is 1.88 x the floor)
MIN_FLOOR = 0.005         # a floor measured at zero is lifted to this relative L2 error
CONTROL_NOISE = 0.25      # relative size of the noise control (a damaged run must fail by a visible margin)
_FIELDS = ("pcc", "rel_l2", "slope")


def load(path) -> dict:
    """Reads a per-step JSON written by acestep_run.py and checks its shape; any gap is a ValueError that names the field."""
    data = json.loads(Path(path).read_text())
    steps = data.get("dit_steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"{path}: no dit_steps")
    for i, s in enumerate(steps):
        if s.get("step") != i:
            raise ValueError(f"{path}: dit_steps[{i}] is step {s.get('step')!r}; steps must be 0..n-1 in order")
        for f in _FIELDS:
            if not isinstance(s.get(f), (int, float)) or not math.isfinite(s[f]):
                raise ValueError(f"{path}: dit_steps[{i}].{f} is not a finite number")
    vae = data.get("vae_on_cpu_final_latent") or {}
    if not isinstance(vae.get("rel_l2"), (int, float)) or not math.isfinite(vae["rel_l2"]):
        raise ValueError(f"{path}: vae_on_cpu_final_latent.rel_l2 is not a finite number")
    return data


def rel_l2(ref: np.ndarray, cand: np.ndarray) -> float:
    ref = np.asarray(ref, dtype=np.float64)
    cand = np.asarray(cand, dtype=np.float64)
    return float(np.linalg.norm(cand - ref) / max(np.linalg.norm(ref), 1e-30))


def derive_bars(floor: dict, margin: float = MARGIN) -> dict:
    """Per-step relative L2 bars and one decoder bar: `margin` times the floor, lifted to `MIN_FLOOR` first."""
    if not margin > 1.0:
        raise ValueError("margin must exceed 1: a bar at the floor fails the floor's own run")
    return {"margin": margin,
            "dit": [max(s["rel_l2"], MIN_FLOOR) * margin for s in floor["dit_steps"]],
            "vae": max(floor["vae_on_cpu_final_latent"]["rel_l2"], MIN_FLOOR) * margin}


def controls_from_velocities(velocities: list, seed: int = 0, noise: float = CONTROL_NOISE) -> dict:
    """Relative L2 errors of damaged copies of the float32 run's own velocities, per step. `wrong_step` uses the neighbouring step's velocity (the next,
    or the previous for the last step). `noise` adds Gaussian noise of the stated relative size. Both are computed on the recorded float32 arrays."""
    rng = np.random.default_rng(seed)
    out = {"wrong_step": [], "noise": []}
    for k, v in enumerate(velocities):
        v = np.asarray(v, dtype=np.float64)
        other = velocities[k + 1] if k + 1 < len(velocities) else velocities[k - 1]
        out["wrong_step"].append(rel_l2(v, other))
        n = rng.standard_normal(v.shape)
        out["noise"].append(rel_l2(v, v + noise * np.linalg.norm(v) / np.linalg.norm(n) * n))
    return out


REQUIRED_CONTROLS = ("wrong_step",)


def analyse(floor: dict, chip: dict, controls: dict, margin: float = MARGIN, required=REQUIRED_CONTROLS) -> dict:
    """Bars from the floor, the chip's result against them, and the controls against them. `usable` is true when the chip passes at every step and
    the decoder, and every control named in `required` fails at every step. Other controls are reported as detection limits. `reasons` lists each refusal."""
    n = len(floor["dit_steps"])
    if len(chip["dit_steps"]) != n:
        raise ValueError(f"floor has {n} steps and the chip result {len(chip['dit_steps'])}")
    for name, vals in controls.items():
        if len(vals) != n:
            raise ValueError(f"control {name} has {len(vals)} steps, expected {n}")
    bars = derive_bars(floor, margin)
    reasons = [f"required control {name} was not supplied" for name in required if name not in controls]
    chip_rows = []
    for k in range(n):
        got = chip["dit_steps"][k]["rel_l2"]
        ok = got <= bars["dit"][k]
        chip_rows.append({"step": k, "floor": floor["dit_steps"][k]["rel_l2"], "bar": bars["dit"][k], "chip": got,
                          "chip_over_floor": got / max(floor["dit_steps"][k]["rel_l2"], MIN_FLOOR), "pass": ok})
        if not ok:
            reasons.append(f"chip step {k}: relative L2 {got:.4f} is above the bar {bars['dit'][k]:.4f}")
    vae_got = chip["vae_on_cpu_final_latent"]["rel_l2"]
    vae_ok = vae_got <= bars["vae"]
    if not vae_ok:
        reasons.append(f"chip decoder: relative L2 {vae_got:.4f} is above the bar {bars['vae']:.4f}")
    ctl = {}
    for name, vals in controls.items():
        passed = [k for k in range(n) if vals[k] <= bars["dit"][k]]
        ctl[name] = {"rel_l2": list(vals), "passed_the_bar_at_steps": passed}
        ctl[name]["required"] = name in required
        if passed and name in required:
            reasons.append(f"control {name} passes the bar at step(s) {passed}: the bars cannot tell it from a good step")
    return {"margin": margin, "bars": bars, "chip": chip_rows, "vae": {"floor": floor["vae_on_cpu_final_latent"]["rel_l2"], "bar": bars["vae"],
                                                                    "chip": vae_got, "pass": vae_ok},
            "detection_limit_rel_l2": bars["dit"], "controls": ctl, "usable": not reasons, "reasons": reasons}


def main(argv=None) -> int:
    import argparse
    from orchard import audio_reference as ar
    p = argparse.ArgumentParser(description="Per-step bars from a teacher-forced floor; refuses when a control passes them.")
    p.add_argument("--floor", required=True, help="teacher-forced bfloat16 CPU JSON (acestep_run.py --teacher-forced)")
    p.add_argument("--chip", required=True, help="chip per-step JSON (acestep_run.py --stepcheck)")
    p.add_argument("--run", required=True, help="the recorded float32 run the two were forced with (the controls use its velocities)")
    p.add_argument("--margin", type=float, default=MARGIN)
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    floor_doc, chip_doc = load(a.floor), load(a.chip)
    import hashlib
    digest = hashlib.sha256((Path(a.run) / ar.MANIFEST).read_bytes()).hexdigest()
    for name, doc in (("floor", floor_doc), ("chip", chip_doc)):
        if doc.get("ref_manifest_sha256") != digest:
            print(f"refused: the {name} file was not made from the run given with --run (manifest sha256 {doc.get('ref_manifest_sha256')!r}, run {digest})")
            return 2
    with ar.load_run(a.run) as run:
        vel = [np.asarray(run.array("step_denoised", f"v{k}")) for k in range(len(load(a.floor)["dit_steps"]))]
    result = analyse(floor_doc, chip_doc, controls_from_velocities(vel), a.margin)
    text = json.dumps(result, indent=1) + "\n"
    if a.out:
        Path(a.out).write_text(text)
    print(f"usable: {result['usable']}")
    for r in result["reasons"]:
        print("  " + r)
    return 0 if result["usable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
