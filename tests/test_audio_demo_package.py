# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What an audio package carries from the demo: the files, the requirements and the card text. The card text passes the card's own gates."""
import hashlib
import re
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from orchard import audio_demo
from orchard.package_card import card_problems

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "orchard" / "audio_demo"


def minimal_card(section: str, license_id: str = "apache-2.0") -> str:
    """The smallest card the card gates accept for a model, with the demo section added."""
    nc = "Non-commercial use only. Licensed under CC BY-NC 4.0.\n\n" if license_id == "cc-by-nc-4.0" else ""
    return (f"---\nlicense: {license_id}\n---\n# A model\n\n{nc}## Intended use\nMusic clips from a text prompt.\n\n"
            "## Expected performance\n\n| Number | Value | Label | Evidence |\n|---|---|---|---|\n| time to a clip | TODO | TODO | - |\n\n"
            "## Limitations\nLimits go here.\n\n## Risks and safety considerations\nRisks go here.\n\n" + section)


@pytest.mark.parametrize("license_id", ["apache-2.0", "cc-by-nc-4.0"])
def test_the_card_section_passes_the_card_gates(license_id, tmp_path):
    assert card_problems(minimal_card("", license_id), license_id=license_id, run_dir=tmp_path) == []        # the control: the card is clean without it
    problems = card_problems(minimal_card(audio_demo.card_section(), license_id), license_id=license_id, run_dir=tmp_path)
    assert problems == []


@pytest.mark.parametrize("sentence", ["The server makes a clip in 5 ms.", "It runs at 40% of the speed.", "About 12 tok/s.",
                                      "Use it for commercial use.", "It is production-ready."])
def test_the_gates_would_catch_a_number_or_a_commercial_claim_in_the_section(sentence, tmp_path):
    """The check on the check: the same gates go red when the section carries a performance number or a commercial claim."""
    section = audio_demo.card_section() + "\n" + sentence + "\n"
    assert card_problems(minimal_card(section, "cc-by-nc-4.0"), license_id="cc-by-nc-4.0", run_dir=tmp_path)


def test_the_section_says_how_to_start_the_page_and_what_it_does_not_cover():
    text = audio_demo.card_section()
    assert text.startswith("## Try it in a browser")
    for phrase in ("demo/requirements.txt", "gozer run --chips 1", "audio_server.py", "--ui", "--generator fake", "http://127.0.0.1:7872",
                   "queue is full", "loads the model again for each clip", "Only 10 second clips"):
        assert phrase in text, phrase
    assert not re.search(r"/home/|/mnt/", text)


def test_stage_demo_copies_the_files_and_writes_the_requirements(tmp_path):
    records = audio_demo.stage_demo(tmp_path)
    names = sorted(r["path"] for r in records)
    assert names == ["demo/audio_app.py", "demo/audio_client.py", "demo/audio_server.py", "demo/requirements.txt"]
    for r in records:
        data = (tmp_path / r["path"]).read_bytes()
        assert r["sha256"] == hashlib.sha256(data).hexdigest()
    for name in audio_demo.FILES:
        assert (tmp_path / "demo" / name).read_bytes() == (DEMO / name).read_bytes()
    assert (tmp_path / "demo" / "requirements.txt").read_text().strip() == audio_demo.GRADIO_REQUIREMENT


def test_the_staged_copy_runs_on_its_own(tmp_path):
    """The package is flat, with no orchard beside it: the staged server and client import and the server answers --help."""
    audio_demo.stage_demo(tmp_path)
    run = subprocess.run([sys.executable, "-I", str(tmp_path / "demo" / "audio_server.py"), "--help"], capture_output=True, text=True, cwd=tmp_path,
                         timeout=60)
    assert run.returncode == 0 and "--generator" in run.stdout and "--ui" in run.stdout
    run = subprocess.run([sys.executable, "-I", "-c", "import sys; sys.path.insert(0, 'demo'); import audio_client, audio_server"], capture_output=True,
                         text=True, cwd=tmp_path, timeout=60)
    assert run.returncode == 0, run.stderr


def test_stage_demo_refuses_a_missing_or_file_target(tmp_path):
    with pytest.raises(audio_demo.DemoError, match="not a directory"):
        audio_demo.stage_demo(tmp_path / "nope")
    f = tmp_path / "file"
    f.write_text("x")
    with pytest.raises(audio_demo.DemoError, match="not a directory"):
        audio_demo.stage_demo(f)


def test_stage_demo_will_not_write_through_a_link(tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "demo").symlink_to(elsewhere)
    with pytest.raises(audio_demo.DemoError, match="link"):
        audio_demo.stage_demo(pkg)
    assert list(elsewhere.iterdir()) == []


def test_gradio_is_an_optional_extra_and_not_a_core_dependency():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert meta["project"]["optional-dependencies"]["audio-demo"] == [audio_demo.GRADIO_REQUIREMENT]
    assert not meta["project"].get("dependencies")


def test_nothing_the_core_imports_needs_gradio():
    """Importing the helper and the two standard-library modules must not import gradio."""
    code = ("import sys; sys.path[:0] = ['.', 'orchard/audio_demo']; import orchard.audio_demo, audio_client, audio_server; "
            "assert 'gradio' not in sys.modules, 'gradio was imported'")
    run = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=60)
    assert run.returncode == 0, run.stderr


def test_no_machine_paths_in_the_shipped_files():
    for name in audio_demo.FILES:
        text = (DEMO / name).read_text()
        assert not re.search(r"/home/|/mnt/", text), name
