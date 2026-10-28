# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The supervisor and the stage table import with the standard library only. The audio stages need numpy and soundfile, and load them when an audio
stage runs (orchard/audio_stages.py), so a text run on an installation without numpy still starts."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(code):
    return subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)


def test_the_supervisor_and_the_stage_table_import_without_numpy():
    r = run("import sys; sys.modules['numpy'] = None; sys.modules['soundfile'] = None; import orchard.stages, orchard.supervisor; "
            "assert 'orchard.audio_stages' not in sys.modules; print('ok')")
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-400:]


def test_the_audio_gates_never_import_the_numeric_module_in_the_supervisor_process():
    """The gates run in the configured reference interpreter (orchard/audio_remote.py); with no selection in the ledger they fail without importing it."""
    r = run("import sys; import orchard.stages as s; res = s.gate_audio_reference('/nonexistent/stage', '/nonexistent/run'); "
            "assert not res.ok; assert 'orchard.audio_stages' not in sys.modules and 'numpy' not in sys.modules; print('ok')")
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr[-400:]
