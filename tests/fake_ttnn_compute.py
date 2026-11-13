# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The torch-backed fake ttnn for the tests. The file itself ships with the port check template as
orchard/skills/full-port-templates/cpu_ttnn.py (issue 697: the supervisor runs a port against it before any board check). This module loads
that file and stands in for it under the name the tests import (`import fake_ttnn_compute as fake`), so the tests and the supervisor's CPU run
use one stand-in."""
import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[1] / "orchard" / "skills" / "full-port-templates" / "cpu_ttnn.py"
_spec = importlib.util.spec_from_file_location(__name__, _PATH)
_mod = importlib.util.module_from_spec(_spec)
sys.modules[__name__] = _mod          # `import fake_ttnn_compute` returns the shipped module itself
_spec.loader.exec_module(_mod)
