# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The CPU reference producer (orchard/speech_reference_run.py), run in the reference interpreter by stage 1.

The tests put fake `transformers` and `torch` modules in sys.modules and call `main`, so they check the plumbing (what is loaded, what is passed, what is
written) and not the model. No weights are read. The first run on real weights is owed (docs/run-logs/2026-11-23-speech-route.md)."""
import json
import os
import sys
import types

import pytest

from orchard import speech_reference_run as srr


class FakeProcessor:
    calls = []

    @classmethod
    def from_pretrained(cls, repo, **kw):
        cls.calls.append(("processor", repo))
        return cls()

    def apply_transcription_request(self, **kw):
        FakeProcessor.calls.append(("request", kw))
        return FakeInputs(kw["audio"])

    def batch_decode(self, ids, skip_special_tokens=True):
        FakeProcessor.calls.append(("decode", skip_special_tokens))
        return [f" heard {ids[0]} "]


class FakeInputs(dict):
    def __init__(self, audio):
        super().__init__(input_ids=FakeIds([[1, 2, 3]]))
        self.audio = audio
        self.input_ids = self["input_ids"]

    def to(self, *a, **kw):
        FakeProcessor.calls.append(("to", a, kw))
        return self


class FakeIds(list):
    shape = (1, 3)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return ["ids-after-prompt"]
        return list.__getitem__(self, key)


class FakeModel:
    calls = []

    @classmethod
    def from_pretrained(cls, repo, **kw):
        FakeProcessor.calls.append(("model", repo, kw))
        return cls()

    def generate(self, **kw):
        FakeProcessor.calls.append(("generate", {k: v for k, v in kw.items() if k != "input_ids"}))
        return FakeIds([[1, 2, 3, 4]])


@pytest.fixture
def fake_libs(monkeypatch):
    FakeProcessor.calls = []
    tf = types.ModuleType("transformers")
    tf.__version__ = "5.24.0"
    tf.AutoProcessor = FakeProcessor
    tf.VoxtralForConditionalGeneration = FakeModel

    def pipeline(task, model=None, device=None, **kw):
        FakeProcessor.calls.append(("pipeline", task, model, device))
        return lambda path, **k: {"text": f" scorer heard {path.rsplit('/', 1)[-1]} "}
    tf.pipeline = pipeline
    torch = types.ModuleType("torch")
    torch.__version__ = "2.15.0+cpu"
    torch.bfloat16 = "bf16"

    class Ctx:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    torch.inference_mode = lambda: Ctx()
    monkeypatch.setitem(sys.modules, "transformers", tf)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return FakeProcessor.calls


def cfg(tmp_path, **over):
    files = [{"name": "a.wav", "path": str(tmp_path / "a.wav")}, {"name": "b.wav", "path": str(tmp_path / "b.wav")}]
    for f in files:
        open(f["path"], "wb").write(b"x")
    doc = {"schema": 1, "model_id": "mistralai/Voxtral-Mini-3B-2507", "model_dir": str(tmp_path / "snap"), "files": files, "language": None, "max_new_tokens": 64, **over}
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps(doc))
    return p, tmp_path / "out.json"


def run(tmp_path, mode, fake_libs, **over):
    c, out = cfg(tmp_path, **over)
    code = srr.main(["--mode", mode, "--config", str(c), "--out", str(out)])
    return code, (json.loads(out.read_text()) if out.exists() else None)


def test_the_asr_mode_loads_the_model_from_the_snapshot_and_transcribes_each_file_greedily(tmp_path, fake_libs):
    code, out = run(tmp_path, "asr", fake_libs)
    assert code == 0
    assert out["texts"] == {"a.wav": "heard ids-after-prompt", "b.wav": "heard ids-after-prompt"}
    assert ("processor", str(tmp_path / "snap")) in fake_libs
    gen = [c for c in fake_libs if c[0] == "generate"]
    assert len(gen) == 2 and all(g[1]["do_sample"] is False and g[1]["max_new_tokens"] == 64 for g in gen)
    model_call = next(c for c in fake_libs if c[0] == "model")
    assert model_call[1] == str(tmp_path / "snap") and model_call[2]["device_map"] == "cpu"


def test_the_asr_mode_asks_for_english_unless_told_otherwise_and_names_the_model(tmp_path, fake_libs):
    run(tmp_path, "asr", fake_libs)
    req = [c[1] for c in fake_libs if c[0] == "request"]
    assert all(r["language"] == "en" and r["model_id"] == "mistralai/Voxtral-Mini-3B-2507" for r in req)


def test_a_language_in_the_config_is_passed_on(tmp_path, fake_libs):
    run(tmp_path, "asr", fake_libs, language="fr")
    assert all(c[1]["language"] == "fr" for c in fake_libs if c[0] == "request")


def test_the_record_names_the_versions_and_the_decoding(tmp_path, fake_libs):
    code, out = run(tmp_path, "asr", fake_libs)
    e = out["engine"]
    assert e["transformers"] == "5.24.0" and e["torch"] == "2.15.0+cpu" and e["model_id"] == "mistralai/Voxtral-Mini-3B-2507"
    assert e["decoding"] == "greedy" and e["python"] == sys.executable and out["schema"] == 1


def test_the_score_mode_runs_the_speech_to_text_pipeline_on_cpu(tmp_path, fake_libs):
    code, out = run(tmp_path, "score", fake_libs, model_id="openai/whisper-large-v3-turbo", model_dir=None)
    assert code == 0
    assert out["texts"] == {"a.wav": "scorer heard a.wav", "b.wav": "scorer heard b.wav"}
    assert ("pipeline", "automatic-speech-recognition", "openai/whisper-large-v3-turbo", "cpu") in fake_libs
    assert out["engine"]["model_id"] == "openai/whisper-large-v3-turbo"


def test_missing_libraries_end_with_code_3_and_a_sentence(tmp_path, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "transformers", None)
    code, out = run(tmp_path, "asr", None)
    assert code == 3 and out is None
    assert "transformers" in capsys.readouterr().err


@pytest.mark.parametrize("bad", ["schema", "mode", "files", "file name", "file path", "model_id", "max_new_tokens", "not json"])
def test_a_bad_config_ends_with_code_2_and_writes_nothing(tmp_path, fake_libs, bad):
    c, out = cfg(tmp_path)
    doc = json.loads(c.read_text())
    mode = "asr"
    if bad == "schema":
        doc["schema"] = 9
    elif bad == "mode":
        mode = "other"
    elif bad == "files":
        doc["files"] = []
    elif bad == "file name":
        doc["files"][0]["name"] = "../x"
    elif bad == "file path":
        doc["files"][0]["path"] = 5
    elif bad == "model_id":
        doc["model_id"] = ""
    elif bad == "max_new_tokens":
        doc["max_new_tokens"] = 0
    c.write_text("{nope" if bad == "not json" else json.dumps(doc))
    assert srr.main(["--mode", mode, "--config", str(c), "--out", str(out)]) == 2
    assert not out.exists() and fake_libs == []


def test_a_file_that_the_model_cannot_transcribe_ends_with_code_1_and_names_it(tmp_path, fake_libs, monkeypatch, capsys):
    def refuse(self, **kw):
        raise RuntimeError("cannot read audio")
    monkeypatch.setattr(FakeProcessor, "apply_transcription_request", refuse)
    code, out = run(tmp_path, "asr", fake_libs)
    assert code == 1 and out is None
    assert "a.wav" in capsys.readouterr().err


# ---- reading the config ---------------------------------------------------------------------------------------------------


def test_a_config_that_is_a_link_is_refused(tmp_path):
    real = tmp_path / "real.json"
    real.write_text("{}")
    (tmp_path / "link.json").symlink_to(real)
    with pytest.raises(ValueError, match="cannot be opened"):
        srr.read_config(str(tmp_path / "link.json"))


def test_a_config_that_is_a_pipe_is_refused_without_blocking(tmp_path):
    fifo = tmp_path / "pipe.json"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="regular file"):
        srr.read_config(str(fifo))


def test_a_config_over_the_size_limit_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(srr, "CONFIG_LIMIT", 10)
    p = tmp_path / "c.json"
    p.write_text('{"a": "' + "x" * 50 + '"}')
    with pytest.raises(ValueError, match="larger"):
        srr.read_config(str(p))


@pytest.mark.parametrize("text", ["[" * 40 + "]" * 40, '{"a": NaN}', '{"a": 1e999}', '{"a": ' + "9" * 40 + "}", '"\\ud800"', "\xff"])
def test_a_config_with_hostile_content_is_refused(tmp_path, text):
    p = tmp_path / "c.json"
    p.write_bytes(text.encode("utf-8", "surrogatepass") if text != "\xff" else b"\xff\xfe")
    with pytest.raises(ValueError):
        srr.read_config(str(p))


def test_the_output_file_is_never_written_over_a_link(tmp_path, fake_libs):
    c, out = cfg(tmp_path)
    target = tmp_path / "victim.txt"
    target.write_text("keep")
    (tmp_path / f"out.json.tmp{os.getpid()}").symlink_to(target)
    with pytest.raises(FileExistsError):
        srr.main(["--mode", "asr", "--config", str(c), "--out", str(out)])
    assert target.read_text() == "keep"
