# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Targeted tests for scripts/acestep/: the recorder taps of acestep_run.py and the sweep table of acestep_report.py. No chip, no ACE-Step code and no weights."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts" / "acestep"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


run = _load("acestep_run")
report = _load("acestep_report")


class FakeDecoder(torch.nn.Module):
    """Returns (hidden_states * 2, None) like the transformer returns (velocity, past_key_values), and remembers the dtype it was called with."""

    def __init__(self):
        super().__init__()
        self.seconds = [0.5]
        self.seen = None

    def forward(self, hidden_states, timestep, timestep_r, attention_mask=None, encoder_hidden_states=None, encoder_attention_mask=None,
                context_latents=None, use_cache=None, past_key_values=None, **kw):
        self.seen = hidden_states.dtype
        return hidden_states * 2, past_key_values


def _call(tap):
    x = torch.ones(1, 4, 64)
    return tap(hidden_states=x, timestep=torch.tensor([1.0]), timestep_r=torch.tensor([1.0]), encoder_hidden_states=torch.zeros(1, 3, 8),
               context_latents=torch.zeros(1, 4, 128))


def test_tap_records_input_output_and_conditioning():
    tap = run.Tap(FakeDecoder())
    _call(tap)
    _call(tap)
    assert len(tap.calls) == 2
    x, v = tap.calls[0]
    assert x.shape == (4, 64) and float(v.mean()) == 2.0
    assert tuple(tap.enc.shape) == (3, 8) and tuple(tap.ctx.shape) == (4, 128)
    assert tap.seconds == [0.5], "attributes the tap does not have come from the wrapped decoder"
    tap.reset_taps()
    assert tap.calls == [] and tap.enc is None


def test_tap_cast_runs_inner_in_bfloat16_and_returns_float32():
    inner = FakeDecoder()
    tap = run.Tap(inner, cast=torch.bfloat16)
    out = _call(tap)
    assert inner.seen == torch.bfloat16
    assert out[0].dtype == torch.float32
    assert tap.calls[0][0].dtype == torch.float32      # what is recorded is the float32 input the sampler made


def test_vae_tap_records_latent_and_wave_and_casts():
    seen = {}

    def decode(latents, *a, **kw):
        seen["dtype"] = latents.dtype
        return torch.ones(1, 2, 10, dtype=latents.dtype)

    vt = run.VaeTap(decode, cast=torch.bfloat16)
    out = vt(torch.zeros(1, 64, 5))
    assert seen["dtype"] == torch.bfloat16 and out.dtype == torch.float32
    assert vt.latent.shape == (64, 5) and vt.wave.shape == (2, 10) and len(vt.seconds) == 1


def _sweep_file(tmp_path, warmup=1, n=3):
    runs = [{"generate_and_decode_s": 4.0 + 0.1 * i, "dit_s": 1.0, "vae_s": 1.0, "dit_calls": 8, "host_loadavg_1m": 1.0} for i in range(n)]
    p = tmp_path / "s.sweep.json"
    p.write_text(json.dumps({"device": "chip", "seed": 1, "load": {"chip_open_upload_vae_warm_s": 8.0},
                             "rows": [{"duration_s": 60.0, "warmup": warmup, "runs": runs}]}))
    return p


def test_sweep_table_labels_follow_the_rule(tmp_path, capsys):
    assert report.sweep(_sweep_file(tmp_path)) == 0
    assert "benchmark" in capsys.readouterr().out
    assert report.sweep(_sweep_file(tmp_path, warmup=0)) == 0
    assert "unwarmed" in capsys.readouterr().out
    assert report.sweep(_sweep_file(tmp_path, n=2)) == 0
    assert "single sample" in capsys.readouterr().out


def test_sweep_table_real_time_factor(tmp_path, capsys):
    report.sweep(_sweep_file(tmp_path))
    row = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip().startswith("60")][0]
    assert row.split()[-2] == "14.6"          # 60 s of audio over a median of 4.1 s


def test_sound_checks_use_48khz(tmp_path, capsys):
    from orchard import audio_reference as ar
    rec = ar.Recorder(tmp_path / "r", model={"n": 1}, library={"n": 1},
                      settings={"prompt": "p", "duration": 2.0, "steps": 8, "cfg_scale": 1.0, "sampler": "s", "seed": 1, "dtype": "float32", "sample_size": 96000})
    t = np.arange(96000) / 48000
    rec.record("waveform", "audio", np.stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 2))
    rec.finish()
    assert report.sound([str(tmp_path / "r")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["ok"] and out["duration_s"] == 2.0 and out["sample_rate"] == 48000


def test_tap_teacher_forcing_replaces_the_samplers_input_and_records_the_forced_one():
    forced = [torch.full((4, 64), 5.0), torch.full((4, 64), 7.0)]
    inner = FakeDecoder()
    tap = run.Tap(inner, force=forced)
    _call(tap)
    _call(tap)
    assert [float(v.mean()) for _, v in tap.calls] == [10.0, 14.0], "the inner module ran on the forced inputs"
    assert [float(x.mean()) for x, _ in tap.calls] == [5.0, 7.0], "the recorded input is the forced one"


def test_vae_tap_teacher_forcing_decodes_the_recorded_latent():
    seen = []
    vt = run.VaeTap(lambda z, *a, **k: (seen.append(z.clone()), z.sum(dim=1, keepdim=True).repeat(1, 2, 1))[1], force=torch.full((64, 6), 2.0))
    vt(torch.zeros(1, 64, 6))
    assert float(seen[0].mean()) == 2.0 and float(vt.latent.mean()) == 2.0
