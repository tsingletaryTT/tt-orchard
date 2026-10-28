#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Thin command over orchard/version_cli.py (also reachable as `python3 -m orchard.version`).

    python3 scripts/version.py show | check | new <slug> | release [--dry-run] | migrate-pr --from-ref <branch>

See docs/releases.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from orchard import version_cli  # noqa: E402

if __name__ == "__main__":
    sys.exit(version_cli.main(sys.argv[1:]))
