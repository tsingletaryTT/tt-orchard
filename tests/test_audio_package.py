# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/audio_package.py: the package builder for a Stable Audio 3 small-music run, its card and its gate (issue 169).

Everything here is a fake run in a temporary directory: a ledger line, an audio result, a benchmark file, a snapshot with a license, and a demo
repository. No model, chip, network or Hugging Face cache is used, and nothing is published.
"""
import io
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import audio_package as ap

REV = "8" * 40
LIB = "7" * 40
SETTINGS = {"prompt": "a prompt", "steps": 8, "cfg_scale": 1.0, "sampler": "pingpong", "seed": 42, "sample_size": 5292032, "duration": 10.0,
            "device": "tt", "dtype": "bfloat16"}
ATTRIBUTION = "This Stability AI Model is licensed under the Stability AI Community License, Copyright ©  Stability AI Ltd. All Rights Reserved"
LICENSE = (
    "STABILITY AI COMMUNITY LICENSE AGREEMENT\n\n2. RESEARCH & NON-COMMERCIAL USE LICENSE\n\n"
    "Subject to the terms of this Agreement, Stability AI grants You a non-exclusive, worldwide license to use the Stability AI Materials "
    "for any Research or Non-Commercial Purpose.\n\n3. COMMERCIAL USE LICENSE\n\n"
    "If You are using or distributing the Stability AI Materials for a Commercial Purpose, You must register with Stability AI at "
    "(https://stability.ai/community-license).\nIf at any time You or Your Affiliate(s), either individually or in aggregate, generate more than "
    "USD $1,000,000 in annual revenue, any licenses granted to You under this Agreement shall terminate as of such date. You must request a "
    "license from Stability AI at (https://stability.ai/enterprise) , which Stability AI may grant to You in its sole discretion.\n\n"
    f'4. a. (ii) retain the following attribution notice within a "Notice" text file: "{ATTRIBUTION}”, and (iii) display "Powered by Stability AI".\n')
SNAPSHOT_NOTICE = "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms"
PAGE = 'import gradio\nPAGE_NOTE = ("Describe the music and the page asks the model for a clip. "\n             "Only 10 second clips.")\n'


def timings(g):
    return {"generate_and_decode_s": g, "chip_transformer_call_s": g / 5, "conditioning_s": 0.04, "chip_passes": 8}


def size(seconds, gens, warmup=1, label=None):
    parts = {}
    for name in ("generate_and_decode_s", "chip_transformer_call_s", "conditioning_s"):
        vals = sorted(timings(g)[name] for g in gens)
        n = len(vals)
        med = vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2
        parts[name] = {"n": n, "min": vals[0], "median": round(med, 3), "max": vals[-1]}
    return {"seconds": seconds, "label": label or ap.label_for(len(gens), warmup), "warmup_runs": warmup, "repeats": len(gens), "parts": parts,
            "runs": [{"timings": timings(g), "host_loadavg_1m": 1.0, "output_sha256": None} for g in gens]}


def bench_file(sizes=None, **over):
    sizes = sizes if sizes is not None else [size(10.0, [1.5, 1.6, 1.7]), size(120.0, [10.5, 10.6, 10.8])]
    out = {"device": "chip", "model": "small-music", "settings": {k: SETTINGS[k] for k in ("prompt", "steps", "cfg_scale", "sampler", "seed", "sample_size")},
           "warmup_runs": 1, "repeats": 3, "load": {"model_load_s": 5.5, "chip_weights_upload_s": 1.25}, "sizes": sizes}
    out.update(over)
    return out


def arr(pcc, rel, **extra):
    return {"ok": True, "comparable": True, "bars": "provisional: a choice", "latent_final": {"pcc": pcc, "rel_l2": rel, "slope": 1.0},
            "waveform": {"pcc": pcc - 0.01, "rel_l2": rel + 0.05, "log_spectrum_distance": rel - 0.02}, **extra}


def audio_result(**over):
    out = {"runs": {"chip": {"model": "small-music", "revision": REV, "library_commit": LIB, "settings": SETTINGS, "device": "one chip"}},
           "agreement": {"chip_vs_reference": arr(0.9905134, 0.1384541)}, "environment_floor": arr(0.9963038, 0.0860654),
           "control_discrimination": {"ok": True}, "control_difference": {"ok": True}}
    out.update(over)
    return out


class World:
    """A fake run, snapshot and demo repository in tmp_path."""

    def __init__(self, tmp):
        self.tmp = tmp
        self.run = tmp / "run"
        (self.run / "stages" / "2").mkdir(parents=True)
        self.snap = tmp / "snapshots" / REV
        self.snap.mkdir(parents=True)
        (self.snap / "LICENSE.md").write_text(LICENSE, encoding="utf-8")
        (self.snap / "NOTICE").write_text(SNAPSHOT_NOTICE, encoding="utf-8")
        self.demo = tmp / "demo-repo"
        for rel, text in (("demo/audio_client.py", "# client\n"), ("demo/audio_app.py", PAGE), ("serve/serve_audio.py", "# server\n"),
                          ("package/audio_package.py", 'GRADIO_REQUIREMENT = "gradio>=6.20,<7"\n')):
            (self.demo / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.demo / rel).write_text(text, encoding="utf-8")
        self.write_ledger()
        self.write_result(audio_result())
        self.bench = tmp / "bench.json"
        self.bench.write_text(json.dumps(bench_file()), encoding="utf-8")

    def write_ledger(self, model=ap.MODEL_ID, decisions=("ready for operator review",)):
        """A real, chained ledger: a run_start entry and then one decision entry per name in `decisions`."""
        from orchard.ledger import Ledger
        for name in ("ledger.jsonl", "ledger.jsonl.lock"):
            (self.run / name).unlink(missing_ok=True)
        with Ledger(self.run / "ledger.jsonl") as led:
            led.append("run_start", model=model, inputs={"model": str(self.snap)})
            for d in decisions:
                led.append("decision", decision=d)

    def write_result(self, result):
        (self.run / "stages" / "2" / "audio_result.json").write_text(json.dumps(result))

    def write_bench(self, bench):
        self.bench.write_text(json.dumps(bench))

    def stage(self, **kw):
        return ap.stage_package(self.run, bench_path=self.bench, demo_repo=self.demo, snapshot=self.snap, namespace="someone", **kw)

    @property
    def pkg(self):
        return self.run / "stages" / "7" / "package" / ap.PACKAGE_NAME


@pytest.fixture
def world(tmp_path):
    return World(tmp_path)


@pytest.fixture
def staged(world):
    world.stage()
    return world.pkg


def gate(pkg, **kw):
    return ap.package_problems(pkg, hostname="some-host", home="/home/someone", **kw)


def card_of(pkg):
    return (pkg / "README.md").read_text(encoding="utf-8")


# ---- the staged package passes, and has what it should ----------------------------------------------------------------------

def test_a_staged_package_passes_the_gate(staged):
    assert gate(staged) == []


def test_the_package_has_the_files_the_issue_lists_and_ships_no_weights(staged):
    names = {p.relative_to(staged).as_posix() for p in staged.rglob("*") if p.is_file()}
    for want in ("README.md", "LICENSE.md", "NOTICE", "tt-model.yaml", "run.sh", "requirements.txt", "docs/running.md", "demo/serve_audio.py", "demo/audio_app.py",
                 "demo/audio_client.py", "demo/requirements.txt", "runtime/orchard/skills/audio-chip-templates/chip_generate.py",
                 "runtime/orchard/skills/audio-chip-templates/chip_dit.py", "runtime/orchard/audio_reference.py", "evidence/audio_result.json",
                 "evidence/bench-chip.json", ap.RECORD_FILE):
        assert want in names, want
    assert ap.weights_problems(staged) == []


def test_the_license_is_copied_byte_for_byte_and_the_notice_has_the_exact_attribution(world, staged):
    assert (staged / "LICENSE.md").read_bytes() == (world.snap / "LICENSE.md").read_bytes()
    notice = (staged / "NOTICE").read_text(encoding="utf-8")
    assert ATTRIBUTION in notice                      # with the agreement's own two spaces after the copyright sign
    assert "Copyright ©  Stability AI Ltd." in notice


def test_the_copied_files_match_their_sources_and_the_page_has_one_edit(world, staged):
    run_sh = (staged / "run.sh")
    assert run_sh.stat().st_mode & 0o111
    assert (staged / "demo" / "serve_audio.py").read_text() == "# server\n"
    page = (staged / "demo" / "audio_app.py").read_text()
    assert page == PAGE.replace('PAGE_NOTE = ("Describe the music', 'PAGE_NOTE = ("Powered by Stability AI. Describe the music')
    record = json.loads((staged / ap.RECORD_FILE).read_text())
    edit = record["copied_with_one_edit"]
    assert edit["source_sha256"] != edit["staged_sha256"]
    assert ap.sha256_bytes(page.encode()) == edit["staged_sha256"]
    for rel in ("audio_reference.py", "pinned_files.py"):       # the chip runtime is copied unmodified
        assert (staged / "runtime" / "orchard" / rel).read_bytes() == (Path(ap.__file__).parent / rel).read_bytes()


def test_run_sh_is_valid_bash_and_names_the_attribution(staged):
    assert subprocess.run(["bash", "-n", str(staged / "run.sh")]).returncode == 0
    assert "Powered by Stability AI" in (staged / "run.sh").read_text()


def test_the_manifest_is_a_draft_http_server_manifest_and_loads_in_tt_model_when_it_is_installed(staged):
    text = (staged / "tt-model.yaml").read_text()
    assert "DRAFT. NOT BUILT." in text and "kind: http-server" in text
    assert ap.kernel_manifest_problems(staged / "tt-model.yaml") == []


def test_the_card_numbers_are_the_evidence_rounded_and_labelled(staged):
    card = card_of(staged)
    assert "| 10 s clip, generate and decode, median of 3 | 1.6 s | benchmark |" in card
    assert "| 120 s clip, generate and decode, max of 3 | 10.8 s | benchmark |" in card
    assert "| chip against the CPU reference, final latent PCC | 0.9905 | measured (provisional bars) |" in card
    assert "| model load, once per process (first run) | 5.5 s | single sample |" in card
    assert "| boot check of this package on a chip | TODO | TODO | - |" in card
    assert "No quality evaluation was run" in card


def test_every_staged_python_file_compiles(staged):
    import py_compile
    for f in staged.rglob("*.py"):
        py_compile.compile(str(f), cfile=str(staged.parent / "x.pyc"), doraise=True)


def test_the_publish_commands_are_text_with_no_live_line():
    text = ap.publish_commands(namespace="someone")
    assert text.strip() and all(ln.startswith("#") for ln in text.splitlines() if ln.strip())
    assert "hf upload" in text and "tt-model publish" in text


# ---- the builder refuses -----------------------------------------------------------------------------------------------------

def test_a_run_whose_benchmark_has_too_few_runs_is_refused(world):
    world.write_bench(bench_file([size(10.0, [1.5, 1.6]), size(120.0, [10.5])]))
    with pytest.raises(ap.AudioPackageError, match="no size of the benchmark file is a benchmark"):
        world.stage()
    assert not world.pkg.exists()


def test_a_run_whose_benchmark_has_no_warm_up_is_refused(world):
    world.write_bench(bench_file([size(10.0, [1.5, 1.6, 1.7], warmup=0)]))
    with pytest.raises(ap.AudioPackageError, match="no size of the benchmark file is a benchmark"):
        world.stage()


def test_a_label_the_counts_do_not_support_is_refused(world):
    world.write_bench(bench_file([size(10.0, [1.5, 1.6], label="benchmark")]))
    with pytest.raises(ap.AudioPackageError, match="is labelled 'benchmark' and its 2 runs"):
        world.stage()


def test_a_spread_the_runs_do_not_give_is_refused(world):
    b = bench_file()
    b["sizes"][0]["parts"]["generate_and_decode_s"]["median"] = 0.5
    world.write_bench(b)
    with pytest.raises(ap.AudioPackageError, match="is not what its runs give"):
        world.stage()


@pytest.mark.parametrize("change,message", [
    (lambda r: r["runs"]["chip"].pop("revision"), "no identity pin"),
    (lambda r: r["runs"]["chip"].update(revision="main"), "no identity pin"),
    (lambda r: r["runs"]["chip"].pop("library_commit"), "no identity pin"),
    (lambda r: r["agreement"]["chip_vs_reference"].update(ok=False), "did not agree"),
    (lambda r: r["control_discrimination"].update(ok=False), "control did not show"),
    (lambda r: r["agreement"]["chip_vs_reference"].update(bars="informational"), "provisional"),
    (lambda r: r["runs"]["chip"].update(model="small-music-base"), "this builder packages"),
])
def test_a_run_that_cannot_back_a_package_is_refused(world, change, message):
    result = audio_result()
    change(result)
    world.write_result(result)
    with pytest.raises(ap.AudioPackageError, match=message):
        world.stage()


def test_a_snapshot_of_another_revision_is_refused(world, tmp_path):
    other = tmp_path / "snapshots" / ("9" * 40)
    other.mkdir()
    (other / "LICENSE.md").write_text(LICENSE)
    (other / "NOTICE").write_text("x")
    with pytest.raises(ap.AudioPackageError, match="license snapshot is revision"):
        ap.stage_package(world.run, bench_path=world.bench, demo_repo=world.demo, snapshot=other, namespace="someone")


def test_a_run_of_another_model_is_refused(world):
    world.write_ledger(model="someone/else")
    with pytest.raises(ap.AudioPackageError, match="ledger names the model"):
        world.stage()


@pytest.mark.parametrize("key,value", [("model", "small-music-base"), ("device", "cpu")])
def test_a_benchmark_of_another_model_or_device_is_refused(world, key, value):
    world.write_bench(bench_file(**{key: value}))
    with pytest.raises(ap.AudioPackageError, match="the benchmark file is for"):
        world.stage()


def test_a_benchmark_made_with_other_settings_is_refused(world):
    b = bench_file()
    b["settings"]["steps"] = 50
    world.write_bench(b)
    with pytest.raises(ap.AudioPackageError, match="not comparable"):
        world.stage()


def test_a_size_over_the_models_maximum_is_refused(world):
    world.write_bench(bench_file([size(180.0, [1.0, 1.0, 1.0])]))
    with pytest.raises(ap.AudioPackageError, match="at most 120"):
        world.stage()


def test_weights_in_the_package_directory_are_refused_by_name(world):
    world.pkg.mkdir(parents=True)
    (world.pkg / "model.safetensors").write_bytes(b"x")
    with pytest.raises(ap.AudioPackageError, match="already holds weights"):
        world.stage()
    assert (world.pkg / "model.safetensors").exists() and not (world.pkg / "README.md").exists()


def test_a_non_empty_package_directory_is_never_staged_over(world):
    world.pkg.mkdir(parents=True)
    (world.pkg / "notes.txt").write_text("x")
    with pytest.raises(ap.AudioPackageError, match="is not empty"):
        world.stage()


def test_a_changed_demo_page_refuses_instead_of_staging_a_page_without_the_attribution(world):
    (world.demo / "demo" / "audio_app.py").write_text("import gradio\n")
    with pytest.raises(ap.AudioPackageError, match="no longer has the line"):
        world.stage()
    assert not world.pkg.exists()


def test_a_linked_source_file_is_refused(world, tmp_path):
    target = world.demo / "serve" / "serve_audio.py"
    real = tmp_path / "real.py"
    real.write_text("x")
    target.unlink()
    target.symlink_to(real)
    with pytest.raises(ap.AudioPackageError, match="is a link"):
        world.stage()


def test_a_snapshot_whose_license_lacks_a_quoted_sentence_is_refused(world):
    (world.snap / "LICENSE.md").write_text(LICENSE.replace("in its sole discretion.", "in its discretion."))
    with pytest.raises(ap.AudioPackageError, match="no longer has the threshold sentence"):
        world.stage()


def test_a_snapshot_whose_license_changed_the_revenue_threshold_is_refused(world):
    (world.snap / "LICENSE.md").write_text(LICENSE.replace("USD $1,000,000", "USD $2,000,000"))
    with pytest.raises(ap.AudioPackageError, match="revenue threshold"):
        world.stage()


@pytest.mark.parametrize("namespace", ["", "a b", "../x", "-x"])
def test_a_bad_namespace_is_refused(world, namespace):
    with pytest.raises(ap.AudioPackageError, match="namespace"):
        ap.stage_package(world.run, bench_path=world.bench, demo_repo=world.demo, snapshot=world.snap, namespace=namespace)


# ---- the gate: each license element -------------------------------------------------------------------------------------------

def edited(staged, old, new, count=-1):
    card = card_of(staged)
    assert old in card, old
    (staged / "README.md").write_text(card.replace(old, new, count), encoding="utf-8")
    return gate(staged)


LICENSE_CASES = [
    ("the license is named", "Stability AI Community License", "the license", "must name the Stability AI Community License"),
    ("the license is linked", "](LICENSE.md)", "](elsewhere.md)", "must link the license"),
    ("research use is said to be permitted", "Research and non-commercial use is permitted", "Some use is fine", "research and non-commercial use is permitted"),
    ("Powered by Stability AI", "Powered by Stability AI", "Made with care", "must display 'Powered by Stability AI'"),
    ("weights are not included", "does not include the model weights", "has a few files", "that the weights are not included"),
    ("weights are gated", "The weights are gated", "The weights are on the Hub", "that the weights are gated"),
    ("the Gemma terms are named", "Gemma Terms of Use", "Gemma notes", "Gemma Terms of Use"),
    ("the Gemma link is given", "https://ai.google.dev/gemma/terms", "https://example.com", "Gemma Terms of Use"),
    ("the runtime code is Tenstorrent's", "The runtime code in this package is Tenstorrent's", "The code is here", "runtime code is Tenstorrent's"),
    ("the reader is told to read the license", "Read the license", "Skim the text", "read the license"),
    ("registration is stated", "must register with Stability AI", "may register", "requires registering"),
    ("the revenue threshold is given", "USD $1,000,000", "USD $5,000,000", "revenue threshold"),
    ("the research sentence is quoted as written", "for any Research or Non-Commercial Purpose.", "for any purpose.", "research sentence"),
    ("the threshold sentence is quoted as written", "in its sole discretion.", "in its discretion.", "threshold sentence"),
]


@pytest.mark.parametrize("what,old,new,message", LICENSE_CASES, ids=[c[0] for c in LICENSE_CASES])
def test_the_gate_refuses_a_card_that_lacks_a_license_element(staged, what, old, new, message):
    problems = edited(staged, old, new)
    assert any(message.lower() in p.lower() for p in problems), (what, problems)


@pytest.mark.parametrize("old,new,message", [
    ("license: other", "license: unknown", "license: other"),
    ("license: other", "license: apache-2.0", "open license"),
    ("license: other", "license: mit", "open license"),
    ("license_name: stability-ai-community-license\n", "", "license_name"),
    ("license_link: LICENSE.md", "license_link: https://example.com", "license_link"),
])
def test_the_gate_refuses_a_front_matter_that_claims_another_license(staged, old, new, message):
    assert any(message in p for p in edited(staged, old, new))


@pytest.mark.parametrize("claim", ["Commercial use is permitted.", "It is free for commercial use.", "An open-source model.", "It is production-ready.",
                                   "A permissive license.", "You can resell the clips.", "Use it for anything."])
def test_the_gate_refuses_an_open_or_commercial_claim(staged, claim):
    problems = edited(staged, "## Intended use\n", f"## Intended use\n\n{claim}\n")
    assert any("open or commercial license" in p for p in problems), problems


def test_non_commercial_wording_is_not_a_commercial_claim(staged):
    assert gate(staged) == [] and "non-commercial use is permitted" in card_of(staged).lower()


@pytest.mark.parametrize("claim", ["The clips sound great.", "A high-quality model.", "State-of-the-art music.", "Professional results."])
def test_the_gate_refuses_a_quality_claim(staged, claim):
    problems = edited(staged, "## Intended use\n", f"## Intended use\n\n{claim}\n")
    assert any("sound quality" in p for p in problems), problems


@pytest.mark.parametrize("claim", ["The official package.", "Endorsed by the model's authors.", "![logo](https://example.com/stability-logo.png)", "<img src=x>"])
def test_the_gate_refuses_use_of_a_name_or_mark_beyond_what_the_license_requires(staged, claim):
    problems = edited(staged, "## Intended use\n", f"## Intended use\n\n{claim}\n")
    assert any("name or mark" in p for p in problems), problems


def test_the_gate_refuses_a_card_without_the_no_quality_evaluation_sentence(staged):
    assert any("no quality evaluation" in p for p in edited(staged, "No quality evaluation was run", "Nothing was run"))


# ---- the gate: NOTICE, LICENSE.md, the page and the docs ----------------------------------------------------------------------

def test_the_gate_refuses_a_missing_notice(staged):
    (staged / "NOTICE").unlink()
    assert any("no NOTICE file" in p for p in gate(staged))


@pytest.mark.parametrize("bad", [ATTRIBUTION.replace("©  S", "© S"), ATTRIBUTION.replace("©", "(c)"), "Licensed under a license.\n",
                                 ATTRIBUTION.replace("Community License", "License")])
def test_the_gate_refuses_a_notice_that_is_not_the_exact_sentence(staged, bad):
    (staged / "NOTICE").write_text(bad + "\n", encoding="utf-8")
    assert any("NOTICE must contain the attribution sentence exactly" in p for p in gate(staged))


def test_the_gate_refuses_a_missing_license_copy(staged):
    (staged / "LICENSE.md").unlink()
    assert any("no LICENSE.md" in p for p in gate(staged))


def test_the_gate_refuses_a_license_copy_that_is_another_text(staged):
    (staged / "LICENSE.md").write_text("MIT License\n")
    problems = gate(staged)
    assert any("is not the Stability AI Community License Agreement" in p for p in problems)


def test_the_gate_refuses_a_page_without_the_attribution(staged):
    (staged / "demo" / "audio_app.py").write_text("import gradio\n")
    assert any("the page (demo/audio_app.py) must display" in p for p in gate(staged))


def test_the_gate_refuses_a_documentation_file_without_the_attribution(staged):
    (staged / "docs" / "running.md").write_text("# Running\n")
    assert any("docs/running.md must display" in p for p in gate(staged))
    (staged / "docs" / "running.md").unlink()
    assert any("no documentation file" in p for p in gate(staged))


# ---- the gate: weights, scrub and files ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("name", ["model.safetensors", "demo/weights.bin", "x.ckpt", "a/b.gguf"])
def test_the_gate_refuses_weights_in_the_package(staged, name):
    (staged / name).parent.mkdir(parents=True, exist_ok=True)
    (staged / name).write_bytes(b"x")
    assert any("must not contain weights" in p or "weights file" in p for p in gate(staged))


def test_the_gate_refuses_a_large_file(staged, monkeypatch):
    monkeypatch.setattr(ap, "WEIGHT_MAX_BYTES", 10)
    assert any("no file in the package is over" in p for p in gate(staged))


def test_the_gate_refuses_a_home_path_and_a_host_name(staged):
    (staged / "docs" / "running.md").write_text("Powered by Stability AI\nsee /home/someone/x and some-host\n")
    problems = gate(staged)
    assert any("home path" in p for p in problems) and any("some-host" in p for p in problems)


@pytest.mark.parametrize("name", ["tt-model.yaml", "run.sh", "evidence/bench-chip.json", "demo/serve_audio.py", ap.RECORD_FILE])
def test_the_gate_names_a_missing_file(staged, name):
    (staged / name).unlink()
    assert any(f"no {name}" in p for p in gate(staged))


def test_the_gate_refuses_a_package_that_does_not_exist(tmp_path):
    assert ap.package_problems(tmp_path / "nope") != []


# ---- the gate: numbers and labels ---------------------------------------------------------------------------------------------

def test_the_gate_refuses_a_benchmark_label_on_numbers_whose_runs_do_not_earn_it(staged):
    b = json.loads((staged / "evidence" / "bench-chip.json").read_text())
    b["sizes"][0]["warmup_runs"] = 0
    (staged / "evidence" / "bench-chip.json").write_text(json.dumps(b))
    problems = gate(staged)
    assert any("is labelled benchmark and its runs earn 'unwarmed repeats'" in p for p in problems)


def test_the_gate_refuses_a_value_that_is_not_in_the_evidence(staged):
    problems = edited(staged, "| 1.6 s | benchmark |", "| 0.9 s | benchmark |")
    assert any("must be 1.6 s (benchmark) as the evidence gives; the card has 0.9 s" in p for p in problems)


def test_the_gate_refuses_a_value_taken_from_another_field_of_the_evidence(staged):
    """0.04 s is in the evidence (the conditioning time), so a search for the value in the file would pass it."""
    problems = edited(staged, "| 10 s clip, generate and decode, median of 3 | 1.6 s |", "| 10 s clip, generate and decode, median of 3 | 0.04 s |")
    assert any("must be 1.6 s" in p for p in problems)


def test_the_gate_refuses_an_extra_row_and_a_missing_row(staged):
    problems = edited(staged, "| boot check of this package on a chip | TODO | TODO | - |\n", "| 5 s clip, generate and decode, median of 3 | 1 s | benchmark | `evidence/bench-chip.json` |\n")
    assert any("a row the evidence does not give: '5 s clip" in p for p in problems)
    assert any("'boot check of this package on a chip' must be TODO" in p for p in problems)


def test_a_median_of_four_runs_is_shown_with_its_own_digits_and_passes(world):
    world.write_bench(bench_file([size(10.0, [1.5, 1.6, 1.7, 1.8]), size(120.0, [10.5, 10.6, 10.8])]))
    world.stage()
    assert "| 10 s clip, generate and decode, median of 4 | 1.65 s | benchmark |" in card_of(world.pkg)
    assert gate(world.pkg) == []


@pytest.mark.parametrize("evidence", ["README.md", "evidence/other.txt", "../README.md"])
def test_the_gate_refuses_evidence_that_is_not_a_json_file_in_evidence(staged, evidence):
    problems = edited(staged, "| 1.6 s | benchmark | `evidence/bench-chip.json` |", f"| 1.6 s | benchmark | `{evidence}` |")
    assert any("must be a JSON file in evidence/" in p for p in problems)


def test_the_gate_refuses_a_package_whose_evidence_cannot_back_a_card(staged):
    result = json.loads((staged / "evidence" / "audio_result.json").read_text())
    result["agreement"]["chip_vs_reference"]["ok"] = False
    (staged / "evidence" / "audio_result.json").write_text(json.dumps(result))
    assert any("the package's evidence cannot back a card" in p for p in gate(staged))
    (staged / "evidence" / "audio_result.json").write_text("[1]")
    assert any("the package's evidence cannot back a card" in p for p in gate(staged))


def test_the_gate_refuses_an_unknown_label_and_a_row_with_no_evidence(staged):
    assert any("has label 'verified'" in p for p in edited(staged, "| 1.6 s | benchmark |", "| 1.6 s | verified |"))
    assert any("names no evidence" in p for p in edited(staged, "| single sample | `evidence/bench-chip.json` |\n| weight", "| single sample | - |\n| weight"))


def test_the_gate_refuses_a_todo_row_that_shows_a_value(staged):
    assert any("TODO and shows a value" in p for p in edited(staged, "| boot check of this package on a chip | TODO |", "| boot check of this package on a chip | 3 s |"))


def test_the_gate_refuses_the_word_benchmark_when_no_row_earns_it(staged):
    for ev in ("evidence/bench-chip.json",):
        pass
    card = card_of(staged).replace("| benchmark |", "| single sample |")
    (staged / "README.md").write_text(card)
    assert any("uses the word benchmark and has no row that earns the label" in p for p in gate(staged))


def test_the_gate_refuses_a_rate_in_prose(staged):
    problems = edited(staged, "## Intended use\n", "## Intended use\n\nIt runs at 80 tok/s.\n")
    assert any("a number outside the Numbers table" in p for p in problems)


def test_the_gate_refuses_a_timing_in_prose(staged):
    problems = edited(staged, "## Intended use\n", "## Intended use\n\nIt makes a clip in 2.5 seconds.\n")
    assert any("a timing outside the Numbers table" in p for p in problems)


def test_the_gate_refuses_agreement_numbers_that_do_not_say_the_bars_are_provisional(staged):
    edited(staged, "their pass bars are provisional choices", "their pass bars are fixed")
    problems = edited(staged, "The pass bars for agreement are provisional.", "The pass bars for agreement are fixed.")
    assert any("does not say the bars are provisional" in p for p in problems)


# ---- nothing is published; the command ------------------------------------------------------------------------------------------

def args(world, **kw):
    base = dict(run_dir=str(world.run), namespace="someone", bench=str(world.bench), demo_repo=str(world.demo), snapshot=str(world.snap), check=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_the_command_stages_gates_and_never_starts_a_process_or_touches_the_network(world, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("the builder started a process or opened a connection")
    monkeypatch.setattr(subprocess, "run", boom)
    monkeypatch.setattr(subprocess, "Popen", boom)
    monkeypatch.setattr("os.system", boom)
    import socket
    monkeypatch.setattr(socket, "socket", boom)
    out = io.StringIO()
    assert ap.main(args(world), out=out, hostname="some-host") == 0
    assert "passes the gate" in out.getvalue() and "Nothing was uploaded or published" in out.getvalue()
    text = (world.run / "stages" / "7" / "AUDIO_PUBLISH_COMMANDS.txt").read_text()
    assert all(ln.startswith("#") for ln in text.splitlines() if ln.strip())


def test_the_command_exits_1_when_the_gate_fails_and_2_when_the_run_is_refused(world):
    out = io.StringIO()
    assert ap.main(args(world), out=out, hostname="some-host") == 0
    (world.pkg / "NOTICE").unlink()
    out = io.StringIO()
    assert ap.main(args(world, check=True), out=out, hostname="some-host") == 1
    assert "no NOTICE file" in out.getvalue()
    assert ap.main(args(world), out=io.StringIO(), hostname="some-host") == 2      # a second stage over the first is refused


def test_the_command_is_reachable_from_the_front_door(world, capsys):
    from orchard import cli
    code = cli.main(["audio-package", str(world.run), "--namespace", "someone", "--bench", str(world.bench), "--demo-repo", str(world.demo),
                     "--snapshot", str(world.snap)])
    assert code in (0, 1)
    assert "staged" in capsys.readouterr().out


def test_a_hostile_result_file_is_an_audio_package_error(tmp_path):
    (tmp_path / "x.json").write_text("[1, 2]")
    with pytest.raises(ap.AudioPackageError):
        ap.read_json_file(tmp_path / "x.json", "a test file")
    (tmp_path / "y.json").write_text('{"a": NaN}')
    with pytest.raises(ap.AudioPackageError):
        ap.read_json_file(tmp_path / "y.json", "a test file")


# ---- review of PR 193: the revision pin, hostile shapes and the required files ------------------------------------------------

@pytest.mark.parametrize("bench_change,message", [
    (lambda b: b["sizes"][0].update(runs=[], repeats=0), "no measured runs"),
    (lambda b: b["sizes"][0].update(warmup_runs=-1), "warmup_runs"),
    (lambda b: b["sizes"][0].update(warmup_runs=1.5), "warmup_runs"),
    (lambda b: b["sizes"][0].update(warmup_runs="x"), "warmup_runs"),
    (lambda b: b["sizes"][0].update(runs=["x", "x", "x"]), "has no generate_and_decode_s"),
    (lambda b: b["sizes"][0].update(parts=[]), "is missing"),
])
def test_a_benchmark_of_an_unexpected_shape_is_a_refusal_and_the_command_exits_2(world, bench_change, message):
    b = bench_file()
    bench_change(b)
    world.write_bench(b)
    with pytest.raises(ap.AudioPackageError, match=message):
        world.stage()
    out = io.StringIO()
    assert ap.main(args(world), out=out, hostname="some-host") == 2
    assert out.getvalue().startswith("refused:")


@pytest.mark.parametrize("change", [
    lambda r: r["agreement"]["chip_vs_reference"].update(latent_final=[]),
    lambda r: r["agreement"]["chip_vs_reference"].update(waveform=None),
    lambda r: r.update(environment_floor=[1]),
    lambda r: r["runs"].update(chip=[1]),
    lambda r: r["runs"]["chip"].update(settings="x"),
])
def test_an_audio_result_of_an_unexpected_shape_is_a_refusal_and_the_command_exits_2(world, change):
    result = audio_result()
    change(result)
    world.write_result(result)
    out = io.StringIO()
    assert ap.main(args(world), out=out, hostname="some-host") == 2


@pytest.mark.parametrize("name", list(ap.REQUIRED_FILES))
def test_the_gate_names_every_required_file_when_it_is_missing(staged, name):
    (staged / name).unlink()
    assert any(f"no {name}" in p for p in gate(staged)), name


def test_the_required_files_include_every_copied_runtime_and_demo_file_and_the_generated_ones():
    for _, dest in ap.RUNTIME_FILES:
        assert f"runtime/{dest}" in ap.REQUIRED_FILES
    for _, name in ap.DEMO_FILES:
        assert f"demo/{name}" in ap.REQUIRED_FILES
    for generated in ("runtime/orchard/__init__.py", "requirements.txt", "docs/running.md", "demo/requirements.txt"):
        assert generated in ap.REQUIRED_FILES


def test_the_pin_is_in_the_launcher_the_manifest_the_card_and_the_docs(staged):
    run_sh, manifest = (staged / "run.sh").read_text(), (staged / "tt-model.yaml").read_text()
    assert f'REVISION="{REV}"' in run_sh and "check_revision.py" in run_sh and f'revision: "{REV}"' in manifest
    assert f"--revision {REV}" in card_of(staged) and f"--revision {REV}" in (staged / "docs" / "running.md").read_text()


@pytest.mark.parametrize("name,old,new,message", [
    ("run.sh", f'REVISION="{REV}"', 'REVISION="main"', "run.sh must carry"),
    ("run.sh", "check_revision.py", "other.py", "run.sh must carry"),
    ("tt-model.yaml", f'revision: "{REV}"', 'revision: "main"', "tt-model.yaml must pin"),
    ("README.md", f"`{REV}`", "`main`", "the card must name the evidence's model revision"),
])
def test_the_gate_refuses_a_pin_that_differs_from_the_evidence(staged, name, old, new, message):
    f = staged / name
    f.write_text(f.read_text().replace(old, new))
    assert any(message in p for p in gate(staged)), gate(staged)


def run_check(tmp_path, pin, *, present=True, main="same", model="small-music", with_library=True, private=None, break_cache=False):
    """check_revision.py in a child process, with stand-ins for `stable_audio_3` and `huggingface_hub` and a real cache directory tree.
    `present`: the user's cache holds the pinned snapshot. `main`: what the user's refs/main holds ("same": the pin; None: no refs/main; or a revision)."""
    lib = tmp_path / "lib"
    (lib / "stable_audio_3").mkdir(parents=True, exist_ok=True)
    (lib / "huggingface_hub").mkdir(exist_ok=True)
    user = tmp_path / "user cache" / "models--x--y"
    if present:
        (user / "snapshots" / pin).mkdir(parents=True, exist_ok=True)
        (user / "snapshots" / pin / "model.safetensors").write_text("weights")
    if main is not None:
        (user / "refs").mkdir(parents=True, exist_ok=True)
        (user / "refs" / "main").write_text(pin if main == "same" else main)
    (lib / "stable_audio_3" / "__init__.py").write_text("")
    (lib / "huggingface_hub" / "__init__.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "def try_to_load_from_cache(repo_id, filename, revision=None, cache_dir=None):\n"
        f"    base = Path(cache_dir or {str(tmp_path / 'user cache')!r}) / 'models--x--y'\n"
        "    if revision is None:\n"
        "        ref = base / 'refs' / 'main'\n"
        "        if not ref.exists():\n            return None\n"
        "        revision = ref.read_text().strip()\n"
        f"    if cache_dir is not None and {break_cache!r}:\n        revision = 'f' * 40\n"
        "    f = base / 'snapshots' / revision / filename\n"
        "    return str(f) if f.exists() else None\n")
    if with_library:
        (lib / "stable_audio_3" / "model_configs.py").write_text("class M:\n    repo_id = 'x/y'\n    ckpt_path = 'model.safetensors'\n"
                                                                 "all_models = {'small-music': M()}\n")
    script = Path(ap.__file__).parent / "skills" / "audio-chip-templates" / "check_revision.py"
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(lib)}
    priv = private or (tmp_path / "private cache" / "hub")
    r = subprocess.run([sys.executable, str(script), pin, model, str(priv)], env=env, capture_output=True, text=True, cwd=str(tmp_path))
    r.private = priv
    return r


def test_the_private_cache_resolves_to_the_pin_and_links_the_users_snapshot(tmp_path):
    r = run_check(tmp_path, REV)
    assert r.returncode == 0 and REV in r.stdout
    repo = r.private / "models--x--y"
    assert (repo / "refs" / "main").read_text() == REV
    link = repo / "snapshots" / REV
    assert link.is_symlink() and (link / "model.safetensors").read_text() == "weights"


def test_the_private_cache_pins_the_revision_when_the_users_main_points_at_another_one(tmp_path):
    r = run_check(tmp_path, REV, main="9" * 40)
    assert r.returncode == 0 and (r.private / "models--x--y" / "refs" / "main").read_text() == REV
    assert (tmp_path / "user cache" / "models--x--y" / "refs" / "main").read_text() == "9" * 40     # the user's own cache is not written


def test_the_private_cache_works_for_a_cache_that_holds_only_the_pinned_snapshot(tmp_path):
    """A download of one revision sets no refs/ at all. The private cache does not need the user's."""
    r = run_check(tmp_path, REV, main=None)
    assert r.returncode == 0 and not (tmp_path / "user cache" / "models--x--y" / "refs").exists()


def test_the_private_cache_is_rebuilt_over_an_earlier_one_for_another_revision(tmp_path):
    first = run_check(tmp_path, REV)
    assert first.returncode == 0
    (first.private / "models--x--y" / "refs" / "main").write_text("0" * 40)
    again = run_check(tmp_path, REV, private=first.private)
    assert again.returncode == 0 and (first.private / "models--x--y" / "refs" / "main").read_text() == REV


def test_a_stale_link_in_the_private_cache_is_replaced_and_a_real_directory_there_is_refused(tmp_path):
    first = run_check(tmp_path, REV)
    link = first.private / "models--x--y" / "snapshots" / REV
    other = tmp_path / "elsewhere"
    other.mkdir()
    link.unlink()
    link.symlink_to(other, target_is_directory=True)
    again = run_check(tmp_path, REV, private=first.private)
    assert again.returncode == 0 and (link / "model.safetensors").read_text() == "weights"
    link.unlink()
    link.mkdir()                                       # a real directory where the link belongs: the script does not delete it
    assert run_check(tmp_path, REV, private=first.private).returncode == 2


def test_the_revision_check_stops_when_the_pinned_snapshot_is_not_in_the_cache(tmp_path):
    r = run_check(tmp_path, REV, present=False, main="9" * 40)
    assert r.returncode == 2 and f"hf download x/y --revision {REV}" in r.stderr and not r.private.exists()


def test_the_revision_check_stops_when_the_private_cache_does_not_resolve_to_the_pin(tmp_path):
    r = run_check(tmp_path, REV, break_cache=True)
    assert r.returncode == 3 and "Not starting" in r.stderr


def test_the_revision_check_stops_when_the_model_or_the_library_is_not_found(tmp_path):
    assert run_check(tmp_path, REV, model="other").returncode == 2
    assert run_check(tmp_path / "b", REV, with_library=False).returncode == 2


def test_the_revision_check_refuses_a_private_cache_directory_it_cannot_write(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("a file where a directory is needed")
    assert run_check(tmp_path, REV, private=blocker / "hub").returncode == 2


# ---- review of PR 193, round 2: whole rows, special files, a sizes value that is not a list -----------------------------------

def test_the_gate_refuses_a_second_row_with_the_same_name_and_an_invented_value(staged):
    row = "| 10 s clip, generate and decode, median of 3 | 0.5 s | benchmark | `evidence/bench-chip.json` |\n"
    problems = edited(staged, "| boot check of this package on a chip | TODO | TODO | - |\n", "| boot check of this package on a chip | TODO | TODO | - |\n" + row)
    assert any("a second row for '10 s clip, generate and decode, median of 3' that the evidence does not give: 0.5 s" in p for p in problems), problems


def test_the_gate_refuses_a_row_that_appears_twice(staged):
    row = "| 10 s clip, generate and decode, median of 3 | 1.6 s | benchmark | `evidence/bench-chip.json` |\n"
    problems = edited(staged, row, row + row)
    assert any("appears 2 times" in p for p in problems), problems


def test_the_gate_does_not_hang_on_a_fifo_in_the_package(staged):
    import os
    import signal
    os.mkfifo(staged / "extra.txt")
    signal.alarm(10)                                  # a gate that opens the FIFO blocks for ever; the alarm turns that into a failure
    try:
        problems = gate(staged)
    finally:
        signal.alarm(0)
    assert any("extra.txt: not a regular file" in p for p in problems), problems


def test_a_file_swapped_for_a_fifo_after_the_directory_check_does_not_hang_the_scrub(staged, monkeypatch):
    """The directory check passes (it is patched to see nothing), and then a FIFO is where a text file was: the scrub's read must not block."""
    import os
    import signal
    (staged / "docs" / "extra.md").unlink(missing_ok=True)
    os.mkfifo(staged / "extra.txt")
    monkeypatch.setattr(ap, "unreadable_entries", lambda pkg: [])
    signal.alarm(10)
    try:
        problems = gate(staged)
    finally:
        signal.alarm(0)
    assert any("extra.txt: could not be read as a small UTF-8 text file" in p for p in problems), problems


def test_the_scrub_finds_a_token_a_cache_directory_and_a_weights_suffix_in_a_package(staged):
    (staged / ".cache").mkdir()
    (staged / "docs" / "running.md").write_text("Powered by Stability AI\n" + "hf_" + "a" * 34 + "\n")
    hits = ap.scrub_files(staged, hostname="some-host", home="/home/someone")
    assert ".cache: a runtime cache" in hits and any("a Hugging Face token" in h for h in hits)
    (staged / "x.bin").write_bytes(b"\xff\xfe")
    assert "x.bin: a weights or cache file" in ap.scrub_files(staged, hostname="some-host", home="/home/someone")


def test_the_gate_refuses_a_link_in_the_package_and_reads_nothing_through_it(staged, tmp_path):
    target = tmp_path / "outside.txt"
    target.write_text("x")
    (staged / "link.txt").symlink_to(target)
    assert any("link.txt: a symbolic link" in p for p in gate(staged))


def test_the_gate_names_a_card_that_is_not_text_and_a_directory_in_its_place(staged):
    (staged / "README.md").write_bytes(b"\xff\xfe")
    assert any("no README.md card" in p for p in gate(staged))
    (staged / "README.md").unlink()
    (staged / "README.md").mkdir()
    assert any("no README.md card" in p for p in gate(staged))


@pytest.mark.parametrize("sizes", [1, "x", None, {"a": 1}, [1], [[]]])
def test_the_gate_reports_a_benchmark_file_whose_sizes_is_not_a_list_of_objects(staged, sizes):
    b = json.loads((staged / "evidence" / "bench-chip.json").read_text())
    b["sizes"] = sizes
    (staged / "evidence" / "bench-chip.json").write_text(json.dumps(b))
    problems = gate(staged)                            # no exception, and a problem
    assert any("the package's evidence cannot back a card" in p for p in problems)


@pytest.mark.parametrize("exc", [OSError("x"), TypeError("x"), ValueError("x"), AttributeError("x"), KeyError("x"), UnicodeDecodeError("utf-8", b"", 0, 1, "x")])
def test_an_unexpected_exception_while_reading_the_package_is_a_problem_and_not_a_traceback(staged, monkeypatch, exc):
    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(ap, "scrub_files", boom)
    problems = gate(staged)
    assert len(problems) == 1 and problems[0].startswith("the gate could not read the package")


# ---- review of PR 193, round 3: a finished run, and a cache that holds only the pin ---------------------------------------------

@pytest.mark.parametrize("decisions", [(), ("abort",), ("ready for operator review", "abort"), ("continue",)])
def test_a_run_that_did_not_finish_or_was_aborted_is_refused(world, decisions):
    world.write_ledger(decisions=decisions)
    with pytest.raises(ap.AudioPackageError, match="has not finished"):
        world.stage()
    assert not world.pkg.exists()
    out = io.StringIO()
    assert ap.main(args(world), out=out, hostname="some-host") == 2 and "has not finished" in out.getvalue()


def test_a_ledger_that_is_not_json_is_a_refusal(world):
    (world.run / "ledger.jsonl").write_text('{"event": "run_start"}\nnot json\n')
    with pytest.raises(ap.AudioPackageError, match="no readable ledger.jsonl"):
        world.stage()


def test_a_ledger_whose_hash_chain_is_broken_is_a_refusal(world):
    (world.run / "ledger.jsonl").write_text('{"event": "run_start", "data": {}, "seq": 1, "prev": "wrong", "stage": null}\n')
    with pytest.raises(ap.AudioPackageError, match="ledger.jsonl is damaged"):
        world.stage()


def test_a_ledger_with_no_run_start_names_no_model(world):
    from orchard.ledger import Ledger
    for name in ("ledger.jsonl", "ledger.jsonl.lock"):
        (world.run / name).unlink()
    with Ledger(world.run / "ledger.jsonl") as led:
        led.append("decision", decision="ready for operator review")
    with pytest.raises(ap.AudioPackageError, match="ledger names the model None"):
        world.stage()


def test_a_first_ledger_entry_that_is_not_run_start_does_not_name_the_model(world):
    """A `notice` entry that carries a model field is not the run's start."""
    from orchard.ledger import Ledger
    for name in ("ledger.jsonl", "ledger.jsonl.lock"):
        (world.run / name).unlink()
    with Ledger(world.run / "ledger.jsonl") as led:
        led.append("notice", model=ap.MODEL_ID)
        led.append("decision", decision="ready for operator review")
    with pytest.raises(ap.AudioPackageError, match="ledger names the model None"):
        world.stage()


def test_run_sh_points_the_runtime_at_a_cache_that_holds_only_the_pinned_revision(staged):
    run_sh = (staged / "run.sh").read_text()
    assert 'export HF_HUB_CACHE="$PRIVATE_CACHE"' in run_sh
    assert run_sh.index("check_revision.py") < run_sh.index("export HF_HUB_CACHE") < run_sh.index("exec ")
    assert '"$REVISION" small-music "$PRIVATE_CACHE"' in run_sh
    assert "AUDIO_PKG_CACHE" in run_sh and "XDG_CACHE_HOME" in run_sh


def test_the_gate_refuses_a_launcher_that_does_not_export_the_private_cache(staged):
    f = staged / "run.sh"
    f.write_text(f.read_text().replace('export HF_HUB_CACHE="$PRIVATE_CACHE"', "true"))
    assert any("export HF_HUB_CACHE" in p for p in gate(staged))


# ---- review of PR 193, round 4: the ledger is read once, and a decision that is not text -----------------------------------------

@pytest.mark.parametrize("name", ["evidence/audio_result.json", "evidence/bench-chip.json", "LICENSE.md", "NOTICE"])
def test_the_gate_refuses_a_license_or_evidence_file_changed_after_staging(staged, name):
    f = staged / name
    f.write_text(f.read_text() + " ")
    assert any(f"{name} changed after it was staged" in p for p in gate(staged))


def test_torch_and_torchaudio_are_pinned_as_the_library_pins_them(staged):
    lines = (staged / "requirements.txt").read_text().splitlines()
    assert "torch==2.7.1" in lines and "torchaudio==2.7.1" in lines and "torch" not in lines and "torchaudio" not in lines


def test_the_gate_refuses_a_missing_or_malformed_record(staged):
    (staged / ap.RECORD_FILE).write_text('{"files": 3}')
    assert any("has no file list" in p for p in gate(staged))
    (staged / ap.RECORD_FILE).write_text('{"files": []}')
    assert any("has no file list" in p for p in gate(staged))
    (staged / ap.RECORD_FILE).write_text('{"files": [{"path": "x", "sha256": "y"}]}')
    assert any("does not list LICENSE.md" in p for p in gate(staged))


@pytest.mark.parametrize("dep", list(ap.HOST_REQUIREMENTS) + ["gradio"])
def test_the_gate_names_a_missing_host_requirement(staged, dep):
    name = re.split(r"[<>=]", dep, maxsplit=1)[0]
    f = staged / "requirements.txt"
    f.write_text("\n".join(ln for ln in f.read_text().splitlines() if not ln.startswith(name)) + "\n")
    assert any(f"requirements.txt must name {name}" in p for p in gate(staged)), name


def test_a_failed_write_of_the_publish_commands_is_reported_and_exits_1(world, monkeypatch):
    def refuse(*a, **k):
        raise OSError("disk full")
    monkeypatch.setattr(ap.safeio, "atomic_write", refuse)
    out = io.StringIO()
    # staging writes through the same function, so stage first with the real one and then fail only the commands file
    monkeypatch.undo()
    world.stage()
    real = ap.safeio.atomic_write
    monkeypatch.setattr(ap.safeio, "atomic_write", lambda path, *a, **k: refuse() if Path(path).name == "AUDIO_PUBLISH_COMMANDS.txt" else real(path, *a, **k))
    assert ap.main(args(world, check=True), out=out, hostname="some-host") == 1
    assert "could not write stages/7/AUDIO_PUBLISH_COMMANDS.txt" in out.getvalue()


def _ledger_with(world, *decisions):
    from orchard.ledger import Ledger
    for name in ("ledger.jsonl", "ledger.jsonl.lock"):
        (world.run / name).unlink()
    with Ledger(world.run / "ledger.jsonl") as led:
        led.append("run_start", model=ap.MODEL_ID, inputs={"model": str(world.snap)})
        for d in decisions:
            led.append("decision", decision=d)


@pytest.mark.parametrize("odd", [[1], {"a": 1}, 7, None, True])
def test_a_decision_that_is_not_text_is_a_refusal_and_not_a_type_error(world, odd):
    _ledger_with(world, odd)
    with pytest.raises(ap.AudioPackageError, match="has not finished"):
        world.stage()
    assert ap.main(args(world), out=io.StringIO(), hostname="some-host") == 2


def test_a_decision_that_is_not_text_next_to_the_real_one_does_not_stop_the_run(world):
    _ledger_with(world, ["a", "list"], "ready for operator review")
    world.stage()


def test_the_ledger_is_opened_once_and_the_chain_is_verified_over_the_bytes_that_were_checked(world, monkeypatch):
    """A file replaced after the guarded read is not read again: the read below returns the real bytes and then removes the file."""
    real = ap.safeio.read_bytes

    def read_then_remove(path, **kw):
        data = real(path, **kw)
        if Path(path).name == "ledger.jsonl":
            Path(path).unlink()
        return data
    monkeypatch.setattr(ap.safeio, "read_bytes", read_then_remove)
    entries = ap.ledger_entries(world.run)
    assert [e["event"] for e in entries] == ["run_start", "decision"]
