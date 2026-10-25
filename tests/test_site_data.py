# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The site's replay data (review of PR 71): the scrub covers the whole message, wrapped lines included."""
import importlib.util
import socket
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("make_site_data", Path(__file__).resolve().parent.parent / "scripts" / "make_site_data.py")
msd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(msd)

USER = Path.home().name
PAD = " " * 24


def rows(*lines, run="/runs/x"):
    return msd.assemble(list(lines), Path(run))


def test_a_wrapped_continuation_line_is_scrubbed_with_its_message():
    out = rows("10:00:00  orchardist   reading the file", f"{PAD}owned by {USER} in {Path.home()}/cache and on {socket.gethostname()}")
    text = out[0]["text"]
    assert USER not in text.replace("user", "") or "owned by user" in text
    assert str(Path.home()) not in text and socket.gethostname().lower() not in text.lower()
    assert "owned by user" in text and "~/cache" in text and "this-box" in text


def test_the_user_name_inside_another_word_is_left_alone():
    out = rows(f"10:00:00  orchardist   a {USER}x b")
    assert f"{USER}x" in out[0]["text"]


def test_continuation_text_joins_its_line_and_times_are_seconds_since_the_first():
    out = rows("23:59:50  orchardist   first", "00:00:10  grafter      second", f"{PAD}and more")
    assert [r["t"] for r in out] == [0, 20] and out[1]["text"] == "second and more" and out[1]["actor"] == "grafter"


def test_ansi_colour_is_removed():
    out = rows("10:00:00  \x1b[38;5;73morchardist   \x1b[0m hello")
    assert out[0]["actor"] == "orchardist" and out[0]["text"] == "hello"


def test_the_run_directory_becomes_a_label():
    out = rows("10:00:00  orchardist   wrote /runs/x/stages/1/f", run="/runs/x")
    assert "<run>/stages/1/f" in out[0]["text"]
