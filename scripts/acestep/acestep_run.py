# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Runs ACE-Step 1.5 turbo (text to music) on the CPU or with the diffusion transformer and the audio VAE on one Tenstorrent chip, and records
the run in the layout of `orchard/audio_reference.py`, so `python -m orchard.audio_reference check --cross-precision REF CANDIDATE` compares them.

The model code is the community bundle `sunandbean/acestep-1.5-turbo-p100a` (its `code/` directory: ACE-Step itself, tt-metal's `models/` and
the port `tt_acestep`). Nothing here comes from the Hub at run time: the bundle's code directory, the weights snapshot and an ACE-Step project
root are all local paths given on the command line.

    # reference (CPU, float32)
    python acestep_run.py --device cpu --bundle-code CODE_DIR --snapshot SNAPSHOT_DIR --root WORK_DIR --out RUN_DIR --prompt "..." --duration 30 --seed 42
    # chip (run under a gozer lease; importing ttnn opens a device)
    python acestep_run.py --device chip ... [--warmup 1 --repeats 3]

The bundle's `LocalWorker` puts the diffusion transformer and the Oobleck VAE decoder on the chip. The text and lyric encoders, the sampler and
the post-processing stay on the CPU. The planning language model is not loaded (`thinking=False`), as in the bundle's own server.

What is recorded, per run (kind / name):
    noise / x0                     the first input to the diffusion transformer (identical on CPU and chip: same seed, same draws)
    conditioning / encoder_states  the encoder output the transformer attends to (CPU code on both sides)
    conditioning / context         the context latents
    step_denoised / v<k>           the velocity the transformer returned at step k (k = 0..7)
    step_latent / x<k>             the input to step k (differs after step 0 because the sampler feeds back the device's velocities)
    latent / final                 the latent the VAE decodes, [64, T]
    waveform / audio               the decoded waveform, [2, N] float32, before the pipeline's loudness post-processing is applied to the file
Timings go in `chip.json` (or `cpu.json`) beside the run, and a repeat k of a run is recorded in a sibling directory `<name>-r<k>`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
# In the repository this file is scripts/acestep/acestep_run.py and `orchard/` is two levels up; in the package it is runtime/acestep_run.py and
# `orchard/` (the few modules it needs) sits beside it.
REPO = HERE if (HERE / "orchard").is_dir() else HERE.parents[1]


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--device", choices=("cpu", "chip"), required=True)
    p.add_argument("--bundle-code", required=True, help="the bundle's code/ directory (acestep, tt_acestep, models)")
    p.add_argument("--snapshot", required=True, help="the ACE-Step/Ace-Step1.5 snapshot directory (holds acestep-v15-turbo, vae, Qwen3-Embedding-0.6B)")
    p.add_argument("--root", required=True, help="a writable directory for ACE-Step's project root (it gets a `checkpoints` link to the snapshot)")
    p.add_argument("--out", required=True, help="a new run directory (repeats go to siblings named <name>-r<k>)")
    p.add_argument("--prompt", required=True)
    p.add_argument("--lyrics", default="")
    p.add_argument("--duration", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--warmup", type=int, default=0, help="unrecorded generate calls before the recorded ones")
    p.add_argument("--repeats", type=int, default=1, help="recorded generate calls on the single loaded process")
    p.add_argument("--cpu-dtype", choices=("float32", "bfloat16"), default="float32",
                   help="cpu: the dtype of the diffusion transformer and the VAE (bfloat16 is a floor for the chip's precision)")
    p.add_argument("--sweep", default=None, help="comma-separated durations in seconds: for each, --warmup discarded and --repeats timed generate "
                   "calls on the single loaded process. Writes <out>.sweep.json (timings and host load only; no arrays are recorded). Needs --device chip or cpu.")
    p.add_argument("--stepcheck", default=None, help="chip: a recorded CPU run directory. Feeds its recorded inputs of every step to the chip's "
                   "transformer (so errors do not compound) and its final latent to the chip's VAE; writes <out>.stepcheck.json")
    p.add_argument("--own-conditioning", action="store_true", help="with --stepcheck: first make one clip, and give the chip's transformer the conditioning this process made "
                   "(text encoder, lyric encoder and cross-attention keys and values, with any --fast chip steps) instead of the recorded float32 one. The inputs x_k and "
                   "the reference velocities stay the recorded ones, so the per-step bars then cover the moved encoders as well as the transformer.")
    p.add_argument("--teacher-forced", default=None, help="cpu: a recorded CPU float32 run directory. Runs one generate call in which every transformer "
                   "call receives that run's recorded input of the same step (and the VAE its recorded final latent), in the dtype given by --cpu-dtype, "
                   "and writes <out>.teacher.json with the per-step errors against the recorded float32 velocities. With --cpu-dtype bfloat16 this "
                   "is the precision floor of the chip's per-step check (issue 262): the same comparison as --stepcheck with the chip replaced by "
                   "a bfloat16 CPU run.")
    p.add_argument("--fast", default="none", help="request-level savings of acestep_fast.py that leave the outputs unchanged: all, none or a comma-separated list "
                   "(cond_cache, timbre_cache, gc_freeze, and the chip-path steps listed there). Recorded reference runs use none.")
    p.add_argument("--no-vae-warm", action="store_true", help="chip: skip the VAE warm-up at load (a first decode then compiles)")
    return p.parse_args(argv)


def fast_steps_for(text: str) -> tuple:
    """The steps named by --fast. `none` does not import acestep_fast, so a runtime copy without that file still runs the default path."""
    if (text or "none").strip() == "none":
        return ()
    sys.path.insert(0, str(HERE))
    import acestep_fast
    return acestep_fast.parse_steps(text)


def repeat_dir(out: Path, k: int) -> Path:
    return out if k == 0 else out.with_name(f"{out.name}-r{k}")


class Tap(torch.nn.Module):
    """Wraps the decoder (the diffusion transformer, local or on the chip) and records each call's input and output. It is a module because the
    handler assigns it to `model.decoder`. Attribute reads it does not have (`seconds`, `reset`, config) go to the wrapped object."""

    def __init__(self, inner, cast=None, force=None):
        super().__init__()
        self.inner = inner
        self.cast = cast          # a dtype: the inner module is in that dtype, so inputs are cast to it and the velocity back to float32
        self.force = force        # teacher forcing: a list of [T, 64] float32 tensors; call k runs on force[k] instead of the sampler's own input
        self.reset_taps()

    def reset_taps(self):
        self.calls = []          # (x, v) pairs, torch tensors on the CPU
        self.enc = self.ctx = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.inner, name)

    def forward(self, hidden_states, timestep, timestep_r, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                context_latents=None, use_cache=None, past_key_values=None, **kw):
        if self.force is not None:
            hidden_states = self.force[len(self.calls)][None].to(hidden_states.dtype)
        if self.cast is not None:
            c = lambda t: t.to(self.cast) if torch.is_tensor(t) and t.is_floating_point() else t
            out = self.inner(hidden_states=c(hidden_states), timestep=c(timestep), timestep_r=c(timestep_r), attention_mask=attention_mask,
                             encoder_hidden_states=c(encoder_hidden_states), encoder_attention_mask=encoder_attention_mask,
                             context_latents=c(context_latents), use_cache=use_cache, past_key_values=past_key_values, **kw)
            out = (out[0].float(),) + tuple(out[1:])
        else:
            out = self.inner(hidden_states=hidden_states, timestep=timestep, timestep_r=timestep_r, attention_mask=attention_mask,
                             encoder_hidden_states=encoder_hidden_states, encoder_attention_mask=encoder_attention_mask,
                             context_latents=context_latents, use_cache=use_cache, past_key_values=past_key_values, **kw)
        if self.enc is None:
            self.enc = encoder_hidden_states[0].detach().float().cpu().clone()
            self.ctx = context_latents[0].detach().float().cpu().clone()
        self.calls.append((hidden_states[0].detach().float().cpu().clone(), out[0][0].detach().float().cpu().clone()))
        return out


class VaeTap:
    """Wraps the VAE decode (`handler.tiled_decode`) and records the latent in and the waveform out."""

    def __init__(self, inner, cast=None, force=None):
        self.inner = inner
        self.cast = cast
        self.force = force        # teacher forcing: a [64, T] float32 latent that replaces the sampler's own final latent
        self.latent = self.wave = None
        self.seconds = []

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def __call__(self, latents, *a, **kw):
        if self.force is not None:
            latents = self.force[None].to(latents.dtype)
        t0 = time.perf_counter()
        wave = self.inner(latents.to(self.cast) if self.cast is not None else latents, *a, **kw)
        wave = wave.float()
        self.seconds.append(time.perf_counter() - t0)
        self.latent = latents[0].detach().float().cpu().clone()
        self.wave = wave[0].detach().float().cpu().clone()
        return wave


def _manifest_digest(run_dir) -> str:
    import hashlib
    return hashlib.sha256((Path(run_dir) / "manifest.json").read_bytes()).hexdigest()


def stepcheck_conditioning(rec_enc, rec_ctx, tap, own: bool):
    """The encoder states and context latents the step check prepares the chip with: the recorded float32 ones, or (own) the ones `tap` saw in this process's last clip."""
    if not own:
        return rec_enc, rec_ctx
    if tap.enc is None or tap.ctx is None:
        raise SystemExit("--own-conditioning needs a clip made first: this process has no conditioning of its own to use")
    return tap.enc.float(), tap.ctx.float()


OWN_ENCODER_BOUND = 0.25      # relative L2 of the encoder states against the reference run's: a gross-mismatch check (a bfloat16 encoder differs by about 0.02)
OWN_CONTEXT_BOUND = 1e-4      # the context latents do not depend on the precision of the encoders


def check_own_conditioning(ref_run, args, tap) -> None:
    """Refuses `--stepcheck --own-conditioning` when the conditioning this process made is not for the reference run's request: the prompt, duration and seed of the command line must equal the
    reference run's settings, the arrays must have its shapes, the context latents must equal its context latents, and the encoder states must be near its encoder states. The lyrics are not
    in the manifest; a different lyric shows as a different encoder-state shape or distance."""
    import numpy as np
    s = ref_run.settings
    for key, got in (("prompt", args.prompt), ("duration", args.duration), ("seed", args.seed)):
        if s.get(key) != got:
            raise SystemExit(f"--own-conditioning: --{key} {got!r} differs from the reference run's {key} {s.get(key)!r}; use the request the reference run was made with")
    for name, got, bound, label in (("encoder_states", tap.enc, OWN_ENCODER_BOUND, "encoder states"), ("context", tap.ctx, OWN_CONTEXT_BOUND, "context latents")):
        ref = np.asarray(ref_run.array("conditioning", name))
        if tuple(got.shape) != ref.shape:
            raise SystemExit(f"--own-conditioning: the {label} have shape {tuple(got.shape)}, the reference run's {ref.shape}; use the prompt, lyrics and duration of the reference run")
        err = float(np.linalg.norm(got.numpy() - ref) / max(np.linalg.norm(ref), 1e-30))
        if err > bound:
            raise SystemExit(f"--own-conditioning: the {label.split()[0] if name == 'context' else 'encoder'} differ from the reference run's by relative L2 {err:.3g} (bound {bound:g}); "
                             "this is not the reference run's request")


def stepcheck(worker, ref_dir: Path, out_json: Path, tap=None, own: bool = False, args=None) -> None:
    """Feeds the recorded CPU inputs of each step (not the chip's own earlier outputs) to the chip's transformer, and the CPU run's final
    latent to the chip's VAE. Each result is then the error of one chip call against the CPU's call on the same input, with nothing compounded."""
    import numpy as np
    from orchard import audio_reference as ar
    from tt_acestep import wire
    from tt_acestep.acestep_dit import TURBO_SCHEDULE
    from tt_acestep.remote import RemoteVae
    run = ar.load_run(ref_dir)
    enc = torch.from_numpy(np.asarray(run.array("conditioning", "encoder_states")))
    ctx = torch.from_numpy(np.asarray(run.array("conditioning", "context")))
    enc, ctx = stepcheck_conditioning(enc, ctx, tap, own)
    if own and args is not None:
        check_own_conditioning(run, args, tap)
    worker.call({"op": "prepare", "enc": wire.encode(enc.float()), "ctx": wire.encode(ctx.float())})
    steps = []
    for k, t in enumerate(TURBO_SCHEDULE):
        x = np.asarray(run.array("step_latent", f"x{k}")).T.copy()
        v_ref = np.asarray(run.array("step_denoised", f"v{k}"))
        reply = worker.call({"op": "forward", "x": wire.encode(torch.from_numpy(x)), "t": float(t), "t_r": float(t)})
        v = wire.decode(reply["v"]).numpy().T
        m = ar.metrics(v_ref, v)
        steps.append({"step": k, "t": t, **{key: m[key] for key in ("pcc", "rel_l2", "slope", "max_abs_error")}})
    z = torch.from_numpy(np.asarray(run.array("latent", "final")))
    vae = RemoteVae(worker, upsample=worker.upsample)
    wave = vae(z[None].float())[0].numpy()
    ref_wave = np.asarray(run.array("waveform", "audio"))
    mw = ar.metrics(ref_wave, wave)
    out_json.write_text(json.dumps({"ref_run": str(ref_dir.name), "ref_manifest_sha256": _manifest_digest(ref_dir), "dit_steps": steps,
                                    "vae_on_cpu_final_latent": {**{k: mw[k] for k in ("pcc", "rel_l2", "slope", "max_abs_error")},
                                                                "log_spectrum_distance": ar.log_spectrum_distance(ref_wave, wave)}}, indent=1) + "\n")
    print("stepcheck:", json.dumps(steps), flush=True)
    print("stepcheck vae:", {k: mw[k] for k in ("pcc", "rel_l2", "slope")}, flush=True)
    run.close()


def teacher_forced(tap, vtap, run, out_json: Path, dtype_name: str) -> None:
    """Compares the taps of a teacher-forced generate call with the recorded float32 run: the per-step velocity errors and the decoder's waveform
    error, each with equal inputs. This is the file `orchard.step_bars` reads as the floor."""
    import numpy as np
    from orchard import audio_reference as ar
    for name, got, kind in (("encoder_states", tap.enc, "conditioning"), ("context", tap.ctx, "conditioning")):
        ref = np.asarray(run.array(kind, name))
        err = float(np.linalg.norm(got.numpy() - ref) / max(np.linalg.norm(ref), 1e-30)) if tuple(got.shape) == ref.shape else float("inf")
        if err > 1e-4:
            raise SystemExit(f"teacher forcing refused: the {name} made from this prompt and lyrics differ from the reference run's (relative L2 {err:.3g}); "
                             "use the prompt, lyrics and seed the reference run was made with")
    steps = []
    for k, (x, v) in enumerate(tap.calls):
        v_ref = np.asarray(run.array("step_denoised", f"v{k}"))
        m = ar.metrics(v_ref, v.T.numpy())
        steps.append({"step": k, **{key: m[key] for key in ("pcc", "rel_l2", "slope", "max_abs_error")}})
    ref_wave = np.asarray(run.array("waveform", "audio"))
    mw = ar.metrics(ref_wave, vtap.wave.numpy())
    out_json.write_text(json.dumps({"ref_run": run.root.name, "ref_manifest_sha256": _manifest_digest(run.root), "dtype": dtype_name, "dit_steps": steps,
                                    "vae_on_cpu_final_latent": {**{k: mw[k] for k in ("pcc", "rel_l2", "slope", "max_abs_error")},
                                                                "log_spectrum_distance": ar.log_spectrum_distance(ref_wave, vtap.wave.numpy())}}, indent=1) + "\n")
    print("teacher-forced:", json.dumps(steps), flush=True)
    print("teacher-forced vae:", {k: mw[k] for k in ("pcc", "rel_l2", "slope")}, flush=True)


def sweep(one, tmp: Path, durations, args, out_json: Path, load: dict) -> None:
    """For each duration: `warmup` discarded and `repeats` timed generate calls. The warm-up matters because the first call at a new size compiles."""
    rows = []
    for d in durations:
        for w in range(args.warmup):
            _, t = one(tmp / f"sw{int(d)}w{w}", d)
            print(f"sweep {d:g} s warm-up {w}: {t}", flush=True)
        runs = []
        for r in range(args.repeats):
            wav_path, t = one(tmp / f"sw{int(d)}r{r}", d)
            t["host_loadavg_1m"] = round(os.getloadavg()[0], 2)
            runs.append(t)
            print(f"sweep {d:g} s repeat {r}: {t}", flush=True)
        keep = Path(str(out_json).replace(".sweep.json", f".{int(d)}s.wav"))
        shutil.copy2(wav_path, keep)
        rows.append({"duration_s": d, "warmup": args.warmup, "runs": runs, "last_wav": str(keep)})
        out_json.write_text(json.dumps({"device": args.device, "cpu_dtype": args.cpu_dtype, "seed": args.seed, "prompt": args.prompt, "load": load,
                                        "rows": rows}, indent=1) + "\n")


class Session:
    """One loaded ACE-Step process: the handler, the taps and (on the chip) the worker. `main` records runs through it and `acestep_serve.py` keeps one
    open for as long as the server runs. Nothing is read from the Hub: the bundle's code directory, the weights snapshot and the project root are local
    paths. `force_x` and `force_z` are the teacher-forcing inputs of `Tap` and `VaeTap`."""

    CHECKPOINT = "acestep-v15-turbo"
    WEIGHTS_REVISION = "19671f406d603126926c1b7e2adc169acbcade22"

    def __init__(self, *, device, bundle_code, snapshot, root, cpu_dtype="float32", no_vae_warm=False, force_x=None, force_z=None, fast=()):
        code = Path(bundle_code).resolve()
        snapshot = Path(snapshot).resolve()
        sys.path.insert(0, str(code))          # acestep, tt_acestep and the bundle's models/ before the host tree's
        sys.path.insert(1, str(REPO))
        os.environ.setdefault("ACESTEP_DISABLE_TQDM", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ["HF_HUB_OFFLINE"] = "1"      # the weights are the local snapshot; nothing is fetched
        os.environ["ACESTEP_CHECKPOINTS"] = str(snapshot)
        os.environ["ACESTEP_ROOT"] = str(Path(root).resolve())
        self.device = device
        t_start = time.perf_counter()
        # ACE-Step finds its own model code with `Path(__file__).resolve().parent / "models"`. When the bundle's files are links (a Hugging Face cache
        # snapshot), resolve() leaves the directory, nothing is found, the sync of the model code is skipped, and the Hub's copy of the model code runs.
        # That gives different music without an error (measured: waveform correlation 0.65 against the run with real files).
        import acestep.model_downloader as _md
        if not (_md._get_models_source_dir() / "turbo").is_dir():
            raise SystemExit(f"ACE-Step cannot find its model code ({_md._get_models_source_dir() / 'turbo'}): the bundle code directory {code} holds links, not files. "
                             "Copy it to a real directory (hf download --local-dir) and pass that one.")
        # The bundle's server module is not imported: `tt_acestep/__init__` imports ttnn, which opens a device, and the CPU reference must not.
        # The three pieces of it this script needs (the project-root link, the handler class with the narrowed preflight, the constants) are
        # written out here with the same behaviour.
        root = Path(root).resolve()
        self.root = root
        # ACE-Step's handler rewrites the model's .py files inside the checkpoint directory ("Model code mismatch ... auto-syncing"), and a write
        # through a link into the Hugging Face cache would change the cached files. So `checkpoints` is a private tree: small files are copied
        # (they may be rewritten), the large weight files are links (they are only read).
        link = root / "checkpoints"
        if link.is_symlink():
            link.unlink()
        elif link.exists():
            shutil.rmtree(link)
        for src in snapshot.rglob("*"):
            if src.is_dir():
                continue
            dst = link / src.relative_to(snapshot)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.stat().st_size > (8 << 20):
                dst.symlink_to(src)
            else:
                shutil.copy2(src.resolve(), dst)
        from acestep.handler import AceStepHandler
        required = {self.CHECKPOINT: ("config.json", "silence_latent.pt"), "vae": ("config.json",),
                    "Qwen3-Embedding-0.6B": ("config.json", "tokenizer_config.json", "tokenizer.json")}

        class LocalCheckpointHandler(AceStepHandler):
            def _ensure_models_present(self, *, checkpoint_path, config_path, prefer_source, vae_variant=None):
                problems = [f"{c}/{n}" for c, names in required.items() for n in names if not (Path(checkpoint_path) / c / n).is_file()]
                problems += [f"{c}/weights" for c in required if not any((Path(checkpoint_path) / c).glob("*.safetensors"))]
                return ("ERROR: these ACE-Step assets are missing: " + ", ".join(problems), False) if problems else None

        self.worker = None
        self.load = {}
        if device == "chip":
            from tt_acestep.local import LocalWorker      # imports ttnn: the caller holds a lease
            t0 = time.perf_counter()
            self.worker = LocalWorker(root / "checkpoints" / self.CHECKPOINT, warm=not no_vae_warm).start()
            self.load["chip_open_upload_vae_warm_s"] = round(time.perf_counter() - t0, 2)
            self.load["worker"] = dict(self.worker.wait_ready())
        from acestep.llm_inference import LLMHandler
        t0 = time.perf_counter()
        self.handler = handler = LocalCheckpointHandler()
        status, ok = handler.initialize_service(project_root=str(root), config_path=self.CHECKPOINT, device="cpu", use_flash_attention=False,
                                                compile_model=False, offload_to_cpu=False, offload_dit_to_cpu=False, quantization=None)
        if not ok:
            raise SystemExit(f"initialize_service failed: {status}")
        self.load["handler_init_s"] = round(time.perf_counter() - t0, 2)
        if self.worker is not None:
            from tt_acestep.remote import RemoteDecoder, RemoteVae, skip_unused_lm_hints
            handler.model.decoder = RemoteDecoder(self.worker)
            handler.tiled_decode = RemoteVae(self.worker, upsample=self.worker.upsample)
            skip_unused_lm_hints(handler.model)
        bf16 = device == "cpu" and cpu_dtype == "bfloat16"
        if bf16:
            # Only the two stages that run on the chip change precision: the transformer and the VAE decoder. The encoders and the sampler stay float32.
            handler.model.decoder.to(torch.bfloat16)
            handler.vae.to(torch.bfloat16)
        self.tap = Tap(handler.model.decoder, cast=torch.bfloat16 if bf16 else None, force=force_x)
        handler.model.decoder = self.tap
        self.vtap = VaeTap(handler.tiled_decode, cast=torch.bfloat16 if bf16 else None, force=force_z)
        handler.tiled_decode = self.vtap
        self.llm = LLMHandler()
        if fast:
            sys.path.insert(0, str(HERE))
            import acestep_fast
            self.fast = acestep_fast.apply(handler, fast, worker=self.worker)
        else:
            self.fast = ()
        self.load["fast_steps"] = list(self.fast)
        self.load["total_load_s"] = round(time.perf_counter() - t_start, 2)

    def one(self, save_dir, duration, prompt, lyrics="", seed=42):
        """One generate call (8 steps, no planner). Returns (path of the written WAV, timings)."""
        from acestep.inference import GenerationConfig, GenerationParams, generate_music
        if self.fast:
            import acestep_fast
            acestep_fast.new_request(self.handler)
        tap, vtap = self.tap, self.vtap
        tap.reset_taps()
        vtap.seconds.clear()
        if hasattr(tap.inner, "seconds"):
            tap.inner.seconds.clear()
        if hasattr(tap.inner, "reset"):
            tap.inner.reset()
        params = GenerationParams(caption=prompt, lyrics=lyrics if lyrics.strip() else "[Instrumental]",
                                  instrumental=not lyrics.strip(), vocal_language="en", duration=float(duration), inference_steps=8,
                                  seed=seed, thinking=False, use_cot_metas=False, use_cot_caption=False, use_cot_language=False)
        config = GenerationConfig(batch_size=1, audio_format="wav", use_random_seed=False, seeds=[seed])
        t0 = time.perf_counter()
        result = generate_music(self.handler, self.llm, params, config, save_dir=str(save_dir))
        total = time.perf_counter() - t0
        if not result.success or not result.audios:
            raise RuntimeError(f"generation failed: {result.error or result.status_message}")
        timings = {"generate_and_decode_s": round(total, 4), "vae_s": round(sum(vtap.seconds), 4)}
        if hasattr(tap.inner, "seconds"):
            timings["dit_s"] = round(sum(tap.inner.seconds), 4)
            timings["dit_calls"] = len(tap.inner.seconds)
        else:
            timings["dit_calls"] = len(tap.calls)
        return Path(result.audios[0]["path"]), timings

    def close(self):
        if self.worker is not None:
            self.worker.close()


def main(argv=None) -> int:
    args = parse(argv)
    import numpy as np
    out = Path(args.out)
    for k in range(args.repeats):
        if repeat_dir(out, k).exists():
            raise SystemExit(f"{repeat_dir(out, k)} exists; a run is never overwritten")
    sys.path.insert(0, str(REPO))
    from orchard import audio_reference as ar
    force_x = force_z = teacher_run = None
    if args.teacher_forced:
        if args.device != "cpu":
            raise SystemExit("--teacher-forced is a CPU mode; use --stepcheck for the chip")
        teacher_run = ar.load_run(Path(args.teacher_forced))
        force_x = [torch.from_numpy(np.asarray(teacher_run.array("step_latent", f"x{k}")).T.copy()).float() for k in range(8)]
        force_z = torch.from_numpy(np.asarray(teacher_run.array("latent", "final"))).float()
    fast_steps = fast_steps_for(args.fast)
    sess = Session(device=args.device, bundle_code=args.bundle_code, snapshot=args.snapshot, root=args.root, cpu_dtype=args.cpu_dtype,
                   no_vae_warm=args.no_vae_warm, force_x=force_x, force_z=force_z, fast=fast_steps)
    srv, worker, load, tap, vtap = sess, sess.worker, sess.load, sess.tap, sess.vtap

    def one(save_dir, duration=None):
        return sess.one(save_dir, args.duration if duration is None else duration, args.prompt, args.lyrics, args.seed)

    tmp = Path(args.root) / "gen"
    tmp.mkdir(parents=True, exist_ok=True)
    if args.teacher_forced:
        one(tmp / "teacher")
        teacher_forced(tap, vtap, teacher_run, Path(str(out) + ".teacher.json"), args.cpu_dtype)
        teacher_run.close()
        return 0
    if args.stepcheck:
        if args.own_conditioning:
            one(tmp / "own-conditioning")         # this clip's conditioning is what the check prepares the chip with
        stepcheck(worker, Path(args.stepcheck), Path(str(out) + ".stepcheck.json"), tap=tap, own=args.own_conditioning, args=args)
    if args.sweep:
        sweep(one, tmp, [float(x) for x in args.sweep.split(",")], args, Path(str(out) + ".sweep.json"), load)
        if worker is not None:
            worker.close()
        return 0
    for w in range(args.warmup):
        _, t = one(tmp / f"warm{w}")
        print(f"warm-up {w}: {t}", flush=True)
    summary = []
    for k in range(args.repeats):
        wav_path, timings = one(tmp / f"rep{k}")
        rd = repeat_dir(out, k)
        loadavg = os.getloadavg()[0]
        sample_size = int(round(args.duration * 48000))
        rec = ar.Recorder(rd, model={"name": "ACE-Step/Ace-Step1.5", "checkpoint": srv.CHECKPOINT, "revision": srv.WEIGHTS_REVISION},
                          library={"name": "ACE-Step-1.5 via sunandbean/acestep-1.5-turbo-p100a", "commit": "ca1e85fe9430179831e6bc6be790c332190a3866"},
                          settings={"prompt": args.prompt, "duration": args.duration, "steps": 8, "cfg_scale": 1.0, "sampler": "turbo-ode-shift1",
                                    "seed": args.seed, "dtype": "float32" if args.device == "cpu" else "bfloat16", "sample_size": sample_size,
                                    "device": args.device})
        calls = tap.calls
        rec.record("noise", "x0", calls[0][0].T.numpy())
        rec.record("conditioning", "encoder_states", tap.enc.numpy())
        rec.record("conditioning", "context", tap.ctx.numpy())
        for i, (x, v) in enumerate(calls):
            rec.record("step_latent", f"x{i}", x.T.numpy())
            rec.record("step_denoised", f"v{i}", v.T.numpy())
        rec.record("latent", "final", vtap.latent.numpy())
        rec.record("waveform", "audio", vtap.wave.numpy())
        rec.timings = {k2: v2 for k2, v2 in timings.items() if isinstance(v2, float)}
        rec.finish()
        clips = rd / "clips"
        clips.mkdir()
        shutil.copy2(wav_path, clips / "audio.wav")
        info = {"device": args.device, "timings": timings, "load": load if k == 0 else None, "host_loadavg_1m": loadavg,
                "duration": args.duration, "seed": args.seed, "prompt": args.prompt, "warmup": args.warmup, "repeats": args.repeats,
                "output_wav": str(clips / "audio.wav")}
        (rd / f"{args.device}.json").write_text(json.dumps(info, indent=1) + "\n")
        summary.append(timings)
        print(f"repeat {k}: {timings} loadavg {loadavg:.1f} -> {rd}", flush=True)
    if worker is not None:
        worker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
