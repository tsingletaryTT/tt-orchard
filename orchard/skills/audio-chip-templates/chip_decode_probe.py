# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Decode one saved latent with the library's decoder on the CPU and with `ChipDecoder` on a chip, and compare layer by layer and on the waveform.

    PYTHONPATH=<stable-audio-3 checkout> python chip_decode_probe.py --model small-music --latent LATENT.npy --out OUT_DIR [--repeats 3] [--seed 42]
        [--no-chip] [--weights-dtype bfloat16]

Run under a gozer lease with the shared Tenstorrent environment when the chip is used (it imports ttnn then). `--no-chip` runs the CPU parts only.

What it records in OUT_DIR/decode_probe.json (and the waveforms in OUT_DIR/*.npy):
  * CPU float32 decode of the latent (the reference), timed, with the stream after each of the 6 layers recorded by hooks;
  * the CPU decode again with the decoder weights cast to bfloat16 (the floor: the error a bfloat16 decoder shows with no chip involved);
  * the chip decode with the stream after each layer read back, compared to the reference layer by layer (PCC, relative L2 error, slope of the best-fit line);
  * the waveform comparison (PCC, relative L2 error, and an L1 distance between log-mel spectrograms, which does not depend on phase as closely as PCC does);
  * repeated chip decodes (no probing) with wall times.
The new tokens carry noise (`mask_noise` 0.01) drawn from the global torch generator, so every decode is preceded by `torch.manual_seed(SEED)`; the CPU and
the chip decodes then use the same noise.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def load_sibling(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def stats(np, got, want):
    """PCC, relative L2 error, slope of the best-fit line of got against want, and the largest absolute difference."""
    g, w = np.asarray(got, dtype=np.float64).ravel(), np.asarray(want, dtype=np.float64).ravel()
    if g.shape != w.shape:
        return {"error": f"shapes {g.shape} and {w.shape} differ"}
    gc, wc = g - g.mean(), w - w.mean()
    denom = float(np.sqrt((gc ** 2).sum() * (wc ** 2).sum()))
    pcc = float((gc * wc).sum() / denom) if denom > 0 else float("nan")
    rel = float(np.linalg.norm(g - w) / max(np.linalg.norm(w), 1e-30))
    slope = float((gc * wc).sum() / max((wc ** 2).sum(), 1e-30))
    return {"pcc": round(pcc, 6), "rel_l2": round(rel, 6), "slope": round(slope, 6), "max_abs": float(np.abs(g - w).max())}


def log_mel_l1(torch, wave_a, wave_b, sr=44100, n_fft=2048, hop=512, n_mels=96):
    """Mean absolute difference of log-mel spectrograms (natural log of power plus 1e-6), averaged over channels. A small filter bank built here so
    the probe needs no extra package."""
    def mel_filters():
        def hz_to_mel(f):
            return 2595.0 * torch.log10(1.0 + f / 700.0)
        def mel_to_hz(m):
            return 700.0 * (10 ** (m / 2595.0) - 1.0)
        pts = mel_to_hz(torch.linspace(float(hz_to_mel(torch.tensor(0.0))), float(hz_to_mel(torch.tensor(sr / 2.0))), n_mels + 2))
        bins = torch.floor((n_fft + 1) * pts / sr).long()
        fb = torch.zeros(n_mels, n_fft // 2 + 1)
        for i in range(n_mels):
            l, c, r = int(bins[i]), int(bins[i + 1]), int(bins[i + 2])
            for k in range(l, c):
                fb[i, k] = (k - l) / max(c - l, 1)
            for k in range(c, r):
                fb[i, k] = (r - k) / max(r - c, 1)
        return fb
    fb = mel_filters()
    win = torch.hann_window(n_fft)

    def spec(w):
        s = torch.stft(w, n_fft, hop, window=win, return_complex=True).abs() ** 2
        return torch.log(fb @ s + 1e-6)
    a, b = torch.as_tensor(wave_a).float(), torch.as_tensor(wave_b).float()
    return float((spec(a) - spec(b)).abs().mean())


def first_divergence_summary(np, layer_ref, layer_bf, layer_chip, wave_ref, wave_bf, wave_chip) -> dict:
    """The first-divergence probe of orchard/first_divergence.py on the decoder's six or twelve layers, the chip against the CPU float32 reference, with the CPU
    decoder in bfloat16 as the floor (the error a bfloat16 decoder shows with no chip involved). Two tolerances are reported: 3 times that floor (the probe's
    default margin) and 5 percent relative L2 (a stated bound)."""
    sys.path.insert(0, str(REPO))
    from orchard import first_divergence as fd

    def run(layers, wave):
        arrays = {("layer", f"layer_{i:02d}"): v.numpy() for i, v in layers.items()}
        arrays[("waveform", "audio")] = wave.numpy()
        return fd.MemoryRun(arrays)
    ref, bf, chip = run(layer_ref, wave_ref), run(layer_bf, wave_bf), run(layer_chip, wave_chip)
    out = {}
    for label, tol in (("three_times_the_bfloat16_floor", fd.Tolerance()), ("rel_l2_0.05", fd.Tolerance(rel_l2=0.05))):
        rep = fd.find_first_divergence(ref, chip, floor=(ref, bf), tolerance=tol, cross_precision=True)
        out[label] = {"located": rep["located"], "first_block": None if rep["first_block"] is None else rep["first_block"]["name"],
                      "other_arrays_leaving": rep["other_arrays_leaving"],
                      "blocks": [{"name": b["name"], "rel_l2": round(float(b["rel_l2"]), 5), "floor_rel_l2": (None if b.get("floor") is None else round(float(b["floor"]["rel_l2"]), 5)),
                                  "leaves": b["leaves"]} for b in rep["blocks"]]}
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", required=True)
    p.add_argument("--latent", required=True, help="a .npy latent [1, 256, T], the input of the decoder")
    p.add_argument("--out", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-chip", action="store_true")
    p.add_argument("--device-id", type=int, default=0)
    p.add_argument("--weights-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    args = p.parse_args(argv)
    mk = load_sibling(REPO / "orchard" / "skills" / "audio-reference-templates" / "make_reference.py", "make_reference_for_probe")
    mk.offline_environment()
    import numpy as np
    import torch
    out_dir = Path(args.out)
    if out_dir.exists():
        raise SystemExit(f"{out_dir} exists; a run is never overwritten")
    out_dir.mkdir(parents=True)
    cdec = load_sibling(HERE / "chip_decoder.py", "chip_decoder_runtime")
    model, config_path, ckpt_path = mk.load_model(args.model)
    problems = cdec.config_problems(json.loads(Path(config_path).read_text(encoding="utf-8")))
    if problems:
        raise SystemExit("the chip decoder does not implement this model:\n  " + "\n  ".join(problems))
    pre = model.model.pretransform
    ae = pre.model
    latent = torch.from_numpy(np.load(args.latent)).float()
    result = {"model": args.model, "latent_shape": list(latent.shape), "seed": args.seed, "weights_dtype": args.weights_dtype}

    # CPU reference with per-layer hooks
    layer_ref = {}
    handles = []
    for i, layer in enumerate(ae.decoder.layers[3].transformers):
        handles.append(layer.register_forward_hook(lambda m, inp, out, i=i: layer_ref.__setitem__(i, out.detach().float().clone())))
    cpu_times = []
    wave_ref = None
    for r in range(max(args.repeats, 1)):
        torch.manual_seed(args.seed)
        t0 = time.time()
        with torch.inference_mode():
            w = pre.decode(latent)
        cpu_times.append(round(time.time() - t0, 3))
        if wave_ref is None:
            wave_ref = w.detach().float().clone()
    for h in handles:
        h.remove()
    result["cpu_decode_s"] = cpu_times
    result["waveform_shape"] = list(wave_ref.shape)
    np.save(out_dir / "wave_cpu_float32.npy", wave_ref.numpy())

    # CPU floor: the same decoder with bfloat16 weights
    import copy
    ae_bf = copy.deepcopy(ae.decoder).to(torch.bfloat16)
    orig_dec = ae.decoder
    ae.decoder = ae_bf
    layer_bf = {}
    bf_handles = [layer.register_forward_hook(lambda m, inp, out, i=i: layer_bf.__setitem__(i, out.detach().float().clone()))
                  for i, layer in enumerate(ae_bf.layers[3].transformers)]
    try:
        class Cast(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x, **kw):
                return self.m(x.to(torch.bfloat16), **kw).float()
        ae.decoder = Cast(ae_bf)
        torch.manual_seed(args.seed)
        with torch.inference_mode():
            wave_bf = pre.decode(latent).float()
    finally:
        ae.decoder = orig_dec
        for h in bf_handles:
            h.remove()
    result["cpu_bfloat16_floor"] = {"waveform": stats(np, wave_bf.numpy(), wave_ref.numpy()),
                                    "log_mel_l1": round(log_mel_l1(torch, wave_bf[0], wave_ref[0]), 5)}

    if not args.no_chip:
        import ttnn
        device = ttnn.open_device(device_id=args.device_id)
        try:
            chip = cdec.ChipDecoder(ttnn, torch, device, ae.decoder, dtype=getattr(ttnn, args.weights_dtype))
            cdec.install(ae, chip)
            chip.probe = {}
            torch.manual_seed(args.seed)
            with torch.inference_mode():
                wave_chip = pre.decode(latent).float()
            result["layers"] = {}
            chip_layers = dict(chip.probe)
            for i in sorted(chip.probe):
                ref = layer_ref[i]
                got = chip.probe[i]
                # hooks see [(b nc), 34, D]; the probe is the same
                result["layers"][f"layer_{i}"] = stats(np, got.numpy(), ref.numpy())
            chip.probe = None
            result["first_divergence"] = first_divergence_summary(np, layer_ref, layer_bf, dict(chip_layers), wave_ref, wave_bf, wave_chip)
            result["chip_waveform"] = {"vs_cpu_float32": stats(np, wave_chip.numpy(), wave_ref.numpy()),
                                       "log_mel_l1_vs_cpu_float32": round(log_mel_l1(torch, wave_chip[0], wave_ref[0]), 5)}
            np.save(out_dir / "wave_chip.npy", wave_chip.numpy())
            # timed repeats, no probing
            chip.timings.update({"forward_s": 0.0, "layers_s": 0.0, "calls": 0})
            times = []
            for r in range(args.repeats):
                torch.manual_seed(args.seed)
                t0 = time.time()
                with torch.inference_mode():
                    pre.decode(latent)
                times.append(round(time.time() - t0, 3))
            result["chip_decode_s"] = times
            result["chip_timings"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in chip.timings.items()}
        finally:
            ttnn.close_device(device)
        try:
            import soundfile
            soundfile.write(str(out_dir / "chip_decoder.flac"), wave_chip[0].numpy().T, 44100)
            soundfile.write(str(out_dir / "cpu_decoder.flac"), wave_ref[0].numpy().T, 44100)
        except Exception as exc:
            result["clip_error"] = repr(exc)
    (out_dir / "decode_probe.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
