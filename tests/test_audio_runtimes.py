# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The resident runtimes (orchard/skills/audio-chip-templates/audio_runtimes.py) and a one-shot script that calls them (chip_generate.py).

No model and no chip: the library's generation is replaced by a stand-in that draws a waveform from the seed, so the tests show what the contract's
code does around it: one load, a recorder for each generation, the same arrays for the same settings on any call, a warm-up that leaves nothing behind,
the load's timings in repeat 0 only, and a one-shot script that is a caller of the contract."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from orchard import resident
from orchard.audio_reference import load_run

TEMPLATES = Path(__file__).resolve().parents[1] / "orchard" / "skills" / "audio-chip-templates"


def load(name):
    spec = importlib.util.spec_from_file_location(name + "_under_test", TEMPLATES / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rts = load("audio_runtimes")
cg = load("chip_generate")


def settings(**over):
    return {"prompt": "p", "duration": 10.0, "steps": 8, "cfg_scale": 1.0, "seed": 42, "sampler": "pingpong", "sample_size": 5292032, **over}


class StubMk:
    """Stands in for make_reference.py: the same names the runtime uses."""
    def __init__(self, calls):
        self.calls = calls

    def offline_environment(self):
        pass

    def file_facts(self, path, hash_it=True):
        return {"sha256": "0" * 64, "bytes": 1}

    def library_commit(self):
        return {"name": "stable-audio-3", "commit": "unknown", "path_hash": "0" * 12}

    def _numpy(self, v):
        return np.asarray(v)

    def run_reference(self, model, run_settings, recorder, torch, *, layers=None):
        """A waveform that depends on the seed, the duration and the steps and on nothing else, plus one layer array."""
        self.calls.append(dict(run_settings))
        rng = np.random.default_rng([run_settings["seed"], int(run_settings["duration"] * 10), run_settings["steps"]])
        recorder.record("layer", "layer_00", rng.standard_normal((4, 3)))
        recorder.record("waveform", "audio", rng.standard_normal((2, 64)))
        recorder.timings["conditioning_s"] = 0.01
        recorder.timings["generate_and_decode_s"] = 0.02


def loaded_runtime(tmp_path):
    rt = rts.LibraryRuntime("small-music")
    rt.calls = []
    rt.mk = StubMk(rt.calls)
    rt.torch = None
    from orchard.audio_reference import Recorder
    rt.Recorder = Recorder
    rt.model = SimpleNamespace(model=SimpleNamespace(diffusion_objective="rf_denoiser"))
    snap = tmp_path / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    rt.config_path, rt.ckpt_path = snap / "model_config.json", snap / "model.safetensors"
    rt.load_timings = {"model_load_s": 5.0}
    return rt


def waveform(path):
    with load_run(path) as run:
        return np.asarray(run.array("waveform", "audio")).copy()


# ---- the contract on the real class -------------------------------------------------------------------------------------------------

def test_generate_before_load_and_after_close_is_refused(tmp_path):
    rt = rts.LibraryRuntime("small-music")
    with pytest.raises(resident.ResidentError, match="load"):
        rt.generate(settings(), tmp_path / "a")
    rt = loaded_runtime(tmp_path)
    rt.close()
    with pytest.raises(resident.ResidentError, match="closed"):
        rt.generate(settings(), tmp_path / "b")


def test_a_second_load_is_refused(tmp_path):
    with pytest.raises(resident.ResidentError, match="already"):
        loaded_runtime(tmp_path).load()


def test_a_bad_setting_is_refused_before_anything_is_recorded(tmp_path):
    rt = loaded_runtime(tmp_path)
    for bad in ({"seed": -1}, {"seed": True}, {"steps": 0}, {"duration": float("nan")}, {"duration": 0}, {"cfg_scale": float("inf")}, {"sample_size": 0}):
        with pytest.raises(resident.ResidentError):
            rt.generate(settings(**bad), tmp_path / "x")
    assert not (tmp_path / "x").exists() and rt.calls == []


def test_a_missing_setting_is_refused(tmp_path):
    rt = loaded_runtime(tmp_path)
    s = settings()
    del s["prompt"]
    with pytest.raises(resident.ResidentError, match="prompt"):
        rt.generate(s, tmp_path / "x")


# ---- the same seed gives the same arrays ---------------------------------------------------------------------------------------------

def test_later_calls_give_the_arrays_of_the_first_call_for_the_same_settings(tmp_path):
    rt = loaded_runtime(tmp_path)
    first = rt.generate(settings(), tmp_path / "first")
    rt.generate(settings(seed=1), tmp_path / "other")           # another run in between
    rt.generate(settings(duration=30.0), tmp_path / "other2")
    last = rt.generate(settings(), tmp_path / "last")
    assert first["output_sha256"] == last["output_sha256"] and first["output_sha256"] is not None
    assert np.array_equal(waveform(tmp_path / "first"), waveform(tmp_path / "last"))
    assert waveform(tmp_path / "first").tobytes() != waveform(tmp_path / "other").tobytes()


def test_the_output_hash_is_the_hash_of_the_recorded_waveform(tmp_path):
    import hashlib
    rt = loaded_runtime(tmp_path)
    res = rt.generate(settings(), tmp_path / "a")
    assert res["output_sha256"] == hashlib.sha256(np.ascontiguousarray(waveform(tmp_path / "a")).tobytes()).hexdigest()


def test_the_run_records_the_settings_and_the_device(tmp_path):
    rt = loaded_runtime(tmp_path)
    rt.generate(settings(sampler=None), tmp_path / "a")
    m = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert m["settings"]["sampler"] == "pingpong" and m["settings"]["device"] == "cpu" and m["settings"]["dtype"] == "float32"
    assert m["model"]["revision"] == "abc123" and m["seeds"]["torch"] == 42


# ---- warm-up, extras, load average ---------------------------------------------------------------------------------------------------

def test_a_run_with_no_directory_leaves_nothing_behind_and_writes_no_clip(tmp_path, monkeypatch):
    import tempfile
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    rt = loaded_runtime(tmp_path)
    res = rt.generate(settings())
    assert res["dir"] is None
    assert [p for p in tmp_path.iterdir() if p.name.startswith("audio-runtime-")] == []
    assert len(rt.calls) == 1
    kept = rt.generate(settings(), tmp_path / "kept")
    assert res["output_sha256"] == kept["output_sha256"] and res["output_sha256"] is not None


def test_same_output_works_on_the_real_runtime_class_without_a_directory(tmp_path):
    rt = loaded_runtime(tmp_path)
    out = resident.same_output(rt, settings(), calls=3)
    assert out["same"] is True and len(set(out["sha256"])) == 1


def test_the_cpu_runtime_hooks_the_layers_only_when_asked_and_the_chip_never(tmp_path):
    seen = []
    for device, record_layers, want in (("cpu", True, None), ("cpu", False, []), ("tt", True, []), ("tt", False, [])):
        rt = loaded_runtime(tmp_path / f"{device}{record_layers}")
        rt.device_label, rt.record_layers = device, record_layers
        run_reference = rt.mk.run_reference
        rt.mk.run_reference = lambda *a, layers="unset", _f=run_reference, **k: (seen.append(layers), _f(*a, **k))[1]
        if device == "tt":
            rt._before_generate = lambda *a: None                       # the chip's own recording needs a chip
            rt._after_generate = lambda *a: None
        rt.generate(settings())
        assert seen[-1] == want, (device, record_layers, seen[-1])


def test_extra_timings_go_into_the_record_and_not_into_the_returned_timings(tmp_path):
    rt = loaded_runtime(tmp_path)
    res = rt.generate(settings(), tmp_path / "a", extra_timings={"model_load_s": 5.0, "chip_weights_upload_s": 1.2})
    m = json.loads((tmp_path / "a" / "manifest.json").read_text())
    assert m["timings"]["model_load_s"] == 5.0 and m["timings"]["chip_weights_upload_s"] == 1.2
    assert "model_load_s" not in res["timings"] and set(res["timings"]) == {"conditioning_s", "generate_and_decode_s"}


def test_the_host_load_average_is_recorded_next_to_the_timings(tmp_path, monkeypatch):
    monkeypatch.setattr(resident, "host_loadavg_1m", lambda loadavg=None: 3.25)
    rt = loaded_runtime(tmp_path)
    rt.generate(settings(), tmp_path / "a")
    assert json.loads((tmp_path / "a" / "manifest.json").read_text())["timings"]["host_loadavg_1m"] == 3.25


def test_a_load_average_that_cannot_be_read_is_left_out_not_written_as_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(resident, "host_loadavg_1m", lambda loadavg=None: None)
    rt = loaded_runtime(tmp_path)
    rt.generate(settings(), tmp_path / "a")
    assert "host_loadavg_1m" not in json.loads((tmp_path / "a" / "manifest.json").read_text())["timings"]


def test_a_run_directory_is_never_overwritten(tmp_path):
    rt = loaded_runtime(tmp_path)
    rt.generate(settings(), tmp_path / "a")
    with pytest.raises(Exception):
        rt.generate(settings(), tmp_path / "a")


# ---- a one-shot script is a thin caller ---------------------------------------------------------------------------------------------

ARGS = ["--model", "small-music", "--prompt", "p", "--duration", "10", "--steps", "8", "--cfg-scale", "1.0", "--seed", "42"]


class CountingRuntime(resident.FakeRuntime):
    pass


def run_main(tmp_path, extra, rt=None):
    rt = rt or CountingRuntime()
    code = cg.main(ARGS + ["--out", str(tmp_path / "small-music")] + extra, runtime_factory=lambda args: rt)
    return code, rt


def test_the_one_shot_script_loads_once_and_generates_once(tmp_path):
    code, rt = run_main(tmp_path, [])
    assert code == 0 and rt.loads == 1 and rt.generate_calls == 1 and rt.closed is True
    assert rt.out_dirs == [tmp_path / "small-music"]
    assert rt.extras == [rt.load_timings]


def test_repeats_and_warmup_run_on_the_one_loaded_process(tmp_path):
    code, rt = run_main(tmp_path, ["--warmup", "1", "--repeats", "3"])
    assert rt.loads == 1 and rt.generate_calls == 4
    assert rt.out_dirs == [None] + [tmp_path / n for n in ("small-music", "small-music-r1", "small-music-r2")]
    assert rt.extras == [None, rt.load_timings, None, None]          # the load is reported once, in repeat 0
    summary = json.loads((tmp_path / "small-music.resident.json").read_text())
    assert summary["label"] == "benchmark" and summary["repeats"] == 3 and summary["warmup_runs"] == 1
    assert summary["parts"]["generate_and_decode_s"]["n"] == 3


def test_two_recorded_runs_are_labelled_a_single_sample(tmp_path):
    run_main(tmp_path, ["--repeats", "2"])
    assert json.loads((tmp_path / "small-music.resident.json").read_text())["label"] == "single sample"


def test_a_one_shot_run_writes_nothing_outside_its_run_directory(tmp_path):
    """The job server that calls the script removes only the run directory; a file beside it would stay."""
    run_main(tmp_path, [])
    assert not (tmp_path / "small-music.resident.json").exists()


def test_a_warmup_alone_writes_the_summary(tmp_path):
    run_main(tmp_path, ["--warmup", "1"])
    assert (tmp_path / "small-music.resident.json").exists()


def test_the_script_refuses_an_existing_directory_before_loading(tmp_path):
    (tmp_path / "small-music-r1").mkdir()
    rt = CountingRuntime()
    with pytest.raises(SystemExit, match="exists"):
        run_main(tmp_path, ["--repeats", "2"], rt)
    assert rt.loads == 0


@pytest.mark.parametrize("extra", [["--repeats", "0"], ["--warmup", "-1"]])
def test_the_script_refuses_bad_repeat_counts(tmp_path, extra):
    with pytest.raises(SystemExit):
        run_main(tmp_path, extra)


def test_the_runtime_is_closed_when_a_generation_fails(tmp_path):
    class Failing(resident.FakeRuntime):
        def generate(self, *a, **k):
            raise RuntimeError("boom")
    rt = Failing()
    with pytest.raises(RuntimeError):
        run_main(tmp_path, [], rt)
    assert rt.closed is True


def test_the_script_does_not_name_the_model_or_the_device_itself():
    """A thin caller: the load, the recording and the chip belong to the runtime module."""
    src = (TEMPLATES / "chip_generate.py").read_text()
    assert "ttnn" not in src.replace("(it imports ttnn)", "") and "run_reference" not in src and "Recorder(" not in src
