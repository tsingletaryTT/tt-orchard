#!/bin/sh
# Summarize one finished run: the safe way to start collect.py.
#
# usage: collect.sh RUN_DIR WIDTH HEIGHT FPS MIN_DUR MAX_DUR [IMAGE]
#
# collect.py reads files the model wrote, so it must not run any code the model (or anything on a path the model could write) could have
# planted: this wrapper unsets every PYTHON* variable (PYTHONPATH, PYTHONSTARTUP, PYTHONUSERBASE, PYTHONHOME and the rest), leaves the
# caller's working directory (RUN_DIR is made absolute first), and starts the interpreter with `-I -S`: no current directory, no user site,
# no site-packages on the import path. collect.py refuses to run when it was not started that way.
# env: CVBENCH_PYTHON  the interpreter (default python3); the other CVBENCH_* and SKILL_SRC variables go through to collect.py.
set -eu
HERE=$(cd "$(dirname "$0")" && pwd -P)
[ "$#" -ge 6 ] || { echo "usage: collect.sh RUN_DIR WIDTH HEIGHT FPS MIN_DUR MAX_DUR [IMAGE]" >&2; exit 2; }
RUN=$(cd "$1" 2>/dev/null && pwd -P) || { echo "collect.sh: no such run directory: $1" >&2; exit 2; }
shift
for v in $(env | sed -n 's/^\(PYTHON[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
cd /
exec "${CVBENCH_PYTHON:-python3}" -I -S "$HERE/collect.py" "$RUN" "$@"
