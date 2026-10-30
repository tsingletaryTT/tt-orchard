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


# ---- the server (scripts/acestep/acestep_serve.py) with a stand-in session ----------------------------------------
serve = _load("acestep_serve")


class FakeSession:
    def __init__(self, wav, gate=None):
        import threading
        self.wav, self.gate, self.calls, self.entered = wav, gate, [], threading.Event()

    def one(self, save_dir, duration, prompt, lyrics, seed):
        self.calls.append((duration, prompt, lyrics, seed))
        self.entered.set()
        if self.gate:
            self.gate.wait(5)
        return self.wav, {"generate_and_decode_s": 0.1}


@pytest.fixture
def server(tmp_path):
    import threading
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"RIFFfake")
    sess = FakeSession(wav)
    srv = serve.make_server(sess, "127.0.0.1", 0, tmp_path)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv, sess, tmp_path
    srv.shutdown()
    srv.server_close()


def _post(srv, body, raw=None):
    import urllib.request, urllib.error
    data = raw if raw is not None else json.dumps(body).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/v1/music", data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def test_serve_returns_the_wav_with_timings_and_removes_the_clip_directory(server):
    srv, sess, tmp = server
    code, body, headers = _post(srv, {"prompt": "lofi", "duration": 10, "seed": 3})
    assert code == 200 and body == b"RIFFfake" and headers["X-Sample-Rate"] == "48000"
    assert json.loads(headers["X-Timings"])["generate_and_decode_s"] == 0.1
    assert sess.calls == [(10.0, "lofi", "", 3)]
    assert not list(tmp.glob("clip-*")), "the directory a request used is removed"


@pytest.mark.parametrize("body", [{}, {"prompt": ""}, {"prompt": "x", "duration": 1}, {"prompt": "x", "duration": 9999}, {"prompt": "x", "duration": True},
                                  {"prompt": "x", "seed": -1}, {"prompt": "x", "seed": 1.5}, {"prompt": "x", "extra": 1}, {"prompt": "x" * 2001},
                                  {"prompt": "x", "lyrics": 5}, [1]])
def test_serve_refuses_bad_input_with_400(server, body):
    srv, sess, _ = server
    code, _, _ = _post(srv, body)
    assert code == 400 and sess.calls == []


def test_serve_refuses_malformed_and_oversized_bodies(server):
    srv, _, _ = server
    assert _post(srv, None, raw=b"{not json")[0] == 400
    assert _post(srv, None, raw=b" " * (serve.MAX_BODY + 1))[0] == 400


def test_serve_second_request_while_busy_gets_503(tmp_path):
    import threading
    gate = threading.Event()
    wav = tmp_path / "x.wav"
    wav.write_bytes(b"RIFFfake")
    sess = FakeSession(wav, gate)
    srv = serve.make_server(sess, "127.0.0.1", 0, tmp_path)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    first = {}
    t = threading.Thread(target=lambda: first.update(r=_post(srv, {"prompt": "a"})))
    t.start()
    assert sess.entered.wait(5), "the first request reached the session"
    code, _, headers = _post(srv, {"prompt": "b"})
    gate.set()
    t.join(5)
    srv.shutdown()
    srv.server_close()
    assert code == 503 and headers["Retry-After"] == "5" and first["r"][0] == 200


def test_serve_a_failed_generate_is_a_500_that_leaks_no_text(tmp_path):
    import threading
    class Bad:
        def one(self, *a):
            raise RuntimeError("secret path /x/y")
    srv = serve.make_server(Bad(), "127.0.0.1", 0, tmp_path)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    code, body, _ = _post(srv, {"prompt": "a"})
    srv.shutdown()
    srv.server_close()
    assert code == 500 and b"secret" not in body and b"RuntimeError" in body


# ---- pkgsrc/acestep/tools/check_bundle.py ---------------------------------------------------------------------------
def _check_bundle():
    spec = importlib.util.spec_from_file_location("check_bundle", SCRIPTS.parents[1] / "pkgsrc" / "acestep" / "tools" / "check_bundle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(tmp_path):
    code = tmp_path / "code"
    (code / "acestep").mkdir(parents=True)
    (code / "acestep" / "a.py").write_text("x = 1\n")
    (code / "tt_acestep").mkdir()
    (code / "tt_acestep" / "b.py").write_text("y = 2\n")
    snap = tmp_path / ("a" * 40)
    (snap / "m").mkdir(parents=True)
    (snap / "m" / "w.safetensors").write_bytes(b"12345")
    cb = _check_bundle()
    h, n = cb.tree_hash(code)
    pins = {"bundle_tree_sha256": h, "bundle_files": n, "bundle_revision": "r", "model_revision": "a" * 40,
            "weights": {"m/w.safetensors": {"size": 5, "sha256": cb.sha256_file(snap / "m" / "w.safetensors")}}}
    return cb, code, snap, pins


def test_check_bundle_passes_on_the_pinned_tree_and_ignores_pycache(tmp_path):
    cb, code, snap, pins = _tree(tmp_path)
    (code / "acestep" / "__pycache__").mkdir()
    (code / "acestep" / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"x")
    assert cb.problems(code, snap, pins, hash_weights=True) == []


def test_check_bundle_names_a_changed_file_a_wrong_revision_and_a_changed_weight(tmp_path):
    cb, code, snap, pins = _tree(tmp_path)
    (code / "acestep" / "a.py").write_text("x = 2\n")
    assert any("not the pinned revision" in x for x in cb.problems(code, snap, pins))
    cb, code, snap, pins = _tree(tmp_path / "b")
    snap.rename(snap.with_name("b" * 40))
    assert any("pinned revision" in x for x in cb.problems(code, snap.with_name("b" * 40), pins))
    cb, code, snap, pins = _tree(tmp_path / "c")
    (snap / "m" / "w.safetensors").write_bytes(b"12346")
    assert any("sha256 differs" in x for x in cb.problems(code, snap, pins, hash_weights=True))
    (snap / "m" / "w.safetensors").write_bytes(b"1234")
    assert any("size 4" in x for x in cb.problems(code, snap, pins))


def test_check_bundle_refuses_a_code_directory_of_links(tmp_path):
    cb, code, snap, pins = _tree(tmp_path)
    real = tmp_path / "blob.py"
    real.write_text("y = 2\n")
    (code / "tt_acestep" / "b.py").unlink()
    (code / "tt_acestep" / "b.py").symlink_to(real)
    assert any("link" in x for x in cb.problems(code, snap, pins))
def test_teacher_forcing_refuses_conditioning_that_differs_from_the_reference(tmp_path):
    class R:
        root = tmp_path / "ref"

        def array(self, kind, name):
            return np.ones((3, 8), np.float32) if name == "encoder_states" else np.ones((4, 128), np.float32)
    (tmp_path / "ref").mkdir()
    (tmp_path / "ref" / "manifest.json").write_text("{}")
    tap = run.Tap(FakeDecoder())
    _call(tap)
    tap.enc = torch.ones(3, 8) * 1.5            # another prompt gives other encoder states
    vt = run.VaeTap(lambda z: z)
    with pytest.raises(SystemExit, match="teacher forcing refused"):
        run.teacher_forced(tap, vt, R(), tmp_path / "o.json", "bfloat16")
