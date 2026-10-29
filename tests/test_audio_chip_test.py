# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The hardware test of stage 2 (orchard/skills/audio-chip-templates/audio_chip_test.py) starts one process for each model, so the repeats of a
model share one load. A stand-in `chip_generate.py` beside a copy of the test records how it was called."""
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest

TEMPLATES = Path(__file__).resolve().parents[1] / "orchard" / "skills" / "audio-chip-templates"

FAKE_GENERATE = '''
import json, sys
from pathlib import Path
argv = sys.argv[1:]
get = lambda name: argv[argv.index(name) + 1]
out = Path(get("--out"))
log = out.parent / "calls.jsonl"
out.parent.mkdir(parents=True, exist_ok=True)
with open(log, "a") as f:
    f.write(json.dumps(argv) + "\\n")
if get("--model") == "crashes":
    raise SystemExit(3)
n = int(get("--repeats"))
for k in range(n):
    d = out if k == 0 else out.with_name(out.name + "-r%d" % k)
    d.mkdir()
    (d / "manifest.json").write_text("{}")
'''


@pytest.fixture
def rig(tmp_path):
    d = tmp_path / "tpl"
    d.mkdir()
    shutil.copy(TEMPLATES / "audio_chip_test.py", d / "audio_chip_test.py")
    (d / "chip_generate.py").write_text(FAKE_GENERATE)
    spec = importlib.util.spec_from_file_location("audio_chip_test_under_test", d / "audio_chip_test.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, tmp_path


def config(tmp_path, runs, **over):
    cfg = {"prompt": "p", "duration": 10.0, "seed": 42, "sample_size": 5292032, "src_dir": str(tmp_path), "hf_home": str(tmp_path), "out_dir": "chip",
           "repeats": 3, "warmup": 1, "runs": runs, **over}
    path = tmp_path / "stage" / "audio_chip_config.json"
    path.parent.mkdir()
    path.write_text(json.dumps(cfg))
    return path


def runs_for(model, n, steps=8):
    return [{"name": model if k == 0 else f"{model}-r{k}", "model": model, "steps": steps, "cfg_scale": 1.0, "sampler": "pingpong"} for k in range(n)]


def calls(tmp_path):
    return [json.loads(line) for line in (tmp_path / "stage" / "chip" / "calls.jsonl").read_text().splitlines()]


def test_one_process_for_each_model_repeats_all_its_runs(rig):
    mod, tmp_path = rig
    path = config(tmp_path, runs_for("small-music", 3) + runs_for("small-base", 3, steps=50))
    assert mod.main(["--config", str(path)]) == 0
    seen = calls(tmp_path)
    assert len(seen) == 2                                                    # six runs, two processes
    for argv, model in zip(seen, ("small-music", "small-base")):
        assert argv[argv.index("--model") + 1] == model and argv[argv.index("--repeats") + 1] == "3" and argv[argv.index("--warmup") + 1] == "1"
        assert argv[argv.index("--out") + 1].endswith("/" + model)
    result = json.loads((tmp_path / "stage" / "chip_test.json").read_text())
    assert [r["name"] for r in result["runs"]] == ["small-music", "small-music-r1", "small-music-r2", "small-base", "small-base-r1", "small-base-r2"]
    assert all(r["returncode"] == 0 and r["manifest"] for r in result["runs"])


def test_the_warmup_comes_from_the_config_and_defaults_to_none(rig):
    mod, tmp_path = rig
    path = config(tmp_path, runs_for("small-music", 1), warmup=2)
    mod.main(["--config", str(path)])
    assert calls(tmp_path)[0][calls(tmp_path)[0].index("--warmup") + 1] == "2"


def test_a_crash_of_one_model_does_not_stop_the_other_and_fails_the_test(rig):
    mod, tmp_path = rig
    path = config(tmp_path, runs_for("crashes", 2) + runs_for("small-base", 2))
    assert mod.main(["--config", str(path)]) == 1
    result = {r["name"]: r for r in json.loads((tmp_path / "stage" / "chip_test.json").read_text())["runs"]}
    assert result["crashes"]["returncode"] == 3 and not result["crashes"]["manifest"] and result["crashes-r1"]["returncode"] == 3
    assert result["small-base-r1"]["returncode"] == 0 and result["small-base-r1"]["manifest"]


def test_a_repeat_that_left_no_manifest_fails_the_test_even_when_the_process_exited_zero(rig):
    mod, tmp_path = rig
    (tmp_path / "tpl" / "chip_generate.py").write_text(FAKE_GENERATE.replace("for k in range(n):", "for k in range(n - 1):"))
    path = config(tmp_path, runs_for("small-music", 3))
    assert mod.main(["--config", str(path)]) == 1


@pytest.mark.parametrize("runs", [
    [{"name": "small-music", "model": "small-music", "steps": 8, "cfg_scale": 1.0, "sampler": "pingpong"},
     {"name": "other", "model": "small-music", "steps": 8, "cfg_scale": 1.0, "sampler": "pingpong"}],              # not the repeat name
    [{"name": "small-music", "model": "small-music", "steps": 8, "cfg_scale": 1.0, "sampler": "pingpong"},
     {"name": "small-music-r1", "model": "small-music", "steps": 9, "cfg_scale": 1.0, "sampler": "pingpong"}],      # settings differ
])
def test_runs_that_one_process_cannot_repeat_are_refused_before_anything_starts(rig, runs):
    mod, tmp_path = rig
    path = config(tmp_path, runs)
    with pytest.raises(SystemExit):
        mod.main(["--config", str(path)])
    assert not (tmp_path / "stage" / "chip" / "calls.jsonl").exists()


def test_the_timeout_covers_the_repeats_and_the_warmup(rig, monkeypatch):
    mod, tmp_path = rig
    seen = []
    real = mod.run_group
    monkeypatch.setattr(mod, "run_group", lambda cfg, group, out_root, log_dir, timeout_s: seen.append(timeout_s) or real(cfg, group, out_root, log_dir, timeout_s))
    path = config(tmp_path, runs_for("small-music", 3), per_run_timeout_s=100)
    mod.main(["--config", str(path)])
    assert seen == [400.0]
