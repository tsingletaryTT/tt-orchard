#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Record `ttl.build_info()` of the venv this script runs in, as the JSON that
`tt-orchard ttlang-consistency --build-info-file FILE` reads.

Run it INSIDE the pair venv and UNDER A GOZER LEASE: `ttl.build_info()` initialises the tt-lang runtime, which opens the
device. It only reads; it makes no kernel and opens no device itself. Standard library only, so it needs nothing the venv
lacks. Example (the pair exports come from `tt-orchard kernel-env --pair`, see `hw_command` in orchard/ttlang_pair.py):

    gozer run --chips 1 --exact 2 --who "me" --reason "build_info record" -- \\
        env TT_METAL_HOME=... PYTHONPATH=... LD_LIBRARY_PATH=... <pair-venv>/bin/python scripts/ttlang_build_info.py --out FILE

The real keys of `ttl.build_info()` (read from docs/run-logs/evidence/ttlang-pair/results/pair-chip.jsonl) are `version`,
`ttlang` (the tt-lang commit), `tt_metal_tag` and `tt_metal` (the tt-metal commit). They are renamed to the keys the
consistency check reads: `tt_lang_commit`, `tt_metal_commit`, `version`, `tt_metal_tag`. A key the call did not give is
written as null, and the check then says "unknown" for the row.
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys

KEYS = {"tt_lang_commit": "ttlang", "tt_metal_commit": "tt_metal", "version": "version", "tt_metal_tag": "tt_metal_tag"}


def normalize(raw: dict) -> dict:
    """The consistency check's record from the raw `ttl.build_info()` dict."""
    return {new: (raw.get(old) if isinstance(raw, dict) else None) for new, old in KEYS.items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="where to write the JSON record")
    args = ap.parse_args(argv)
    import ttl                                           # opens the runtime: only under a lease
    raw = ttl.build_info()
    record = normalize(raw)
    record["source"] = "ttl.build_info()"
    record["recorded_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(args.out, "w") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    sys.exit(main())
