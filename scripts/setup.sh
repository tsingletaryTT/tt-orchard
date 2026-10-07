#!/usr/bin/env bash
# Prepare this machine for tt-orchard: check every prerequisite, install what can be installed safely
# (tt-gozer with its own installer, the reference venv, the coder package, the CPU-tier model), and write a
# first configuration. Idempotent. `--check` only reports. `--yes` asks nothing. `--help` lists the options.
#
# It never publishes, pushes, resets a chip or clears a lease, and it never overwrites a file that exists.
# The work is in orchard/setup_machine.py; this wrapper only finds Python 3.12 and the checkout.
set -euo pipefail

HERE="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
REPO="$(dirname "$HERE")"
PY="${PYTHON:-python3}"

if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' 2>/dev/null; then
    echo "tt-orchard needs Python 3.12 or newer, and '$PY' is older or missing." >&2
    echo "Install Python 3.12 (for example with: uv python install 3.12) and run this again, or point" >&2
    echo "PYTHON at it: PYTHON=/path/to/python3.12 $0" >&2
    exit 2
fi

cd "$REPO"
exec "$PY" -m orchard.setup_machine "$@"
