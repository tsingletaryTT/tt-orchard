# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Rebuild the agreement set's skip masks for the gate (issue 406), under an interpreter that has the `tokenizers` package.

The supervisor uses only the standard library, and `tokenizers` lives in the reference interpreter (the run's `reference_python`). The gate
(orchard/stages.py, `_expected_masks`) runs this file under that interpreter with `python -I`:

    python3 -I orchard/skip_mask_helper.py PATH_TO_AGREEMENT_CORE      (the request is one JSON object on stdin)

request: {"model_dir": "<served model directory>", "lists": [[reference ids of one prompt], ...]}
reply:   one JSON line on stdout, {"ok": true, "masks": [[skipped positions of each list], ...]} or {"ok": false, "error": "<why>"}

It reads the same input files the template read: the served directory's tokenizer.json and the `eos_token_id` of its generation_config.json
and config.json. Each is opened by its resolved path, must be a regular file, and is read up to a size cap. The mask rule is
`agreement_core.expected_skipped`, loaded from the path given, so the gate and the template (which carries a byte-identical copy of that file)
use one rule. This file imports nothing from orchard.
"""
import importlib.util
import json
import os
import stat
import sys

REQUEST_MAX = 16 * 1024 * 1024
TOKENIZER_MAX = 256 * 1024 * 1024
CONFIG_MAX = 8 * 1024 * 1024
LISTS_MAX = 256
IDS_MAX = 100_000


def read_regular(path, limit):
    """The bytes of the regular file `path` (links followed once, by resolving it first), or an OSError for a device, a FIFO or a file over `limit`."""
    fd = os.open(os.path.realpath(path), os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise OSError(f"{os.path.basename(path)} is not a regular file of at most {limit} bytes")
        data = b""
        while True:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                return data
            data += chunk
            if len(data) > limit:
                raise OSError(f"{os.path.basename(path)} grew past {limit} bytes while it was read")
    finally:
        os.close(fd)


def answer(request, core_path):
    from tokenizers import Tokenizer
    spec = importlib.util.spec_from_file_location("agreement_core_for_the_gate", core_path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    model_dir, lists = request.get("model_dir"), request.get("lists")
    if not isinstance(model_dir, str) or not model_dir or "\0" in model_dir:
        return {"ok": False, "error": "no model_dir"}
    if (not isinstance(lists, list) or len(lists) > LISTS_MAX
            or not all(isinstance(x, list) and len(x) <= IDS_MAX and all(core._is_int(i) for i in x) for x in lists)):
        return {"ok": False, "error": "lists must be a short list of lists of whole numbers"}
    tokenizer = Tokenizer.from_str(read_regular(os.path.join(model_dir, "tokenizer.json"), TOKENIZER_MAX).decode("utf-8"))
    docs = []
    for name in ("generation_config.json", "config.json"):
        try:
            docs.append(json.loads(read_regular(os.path.join(model_dir, name), CONFIG_MAX).decode("utf-8")))
        except FileNotFoundError:
            continue
    stops = core.stop_ids_from(*docs)
    return {"ok": True, "masks": [core.expected_skipped(tokenizer, stops, ids) for ids in lists]}


def main():
    try:
        if len(sys.argv) != 2:
            raise ValueError("one argument: the path of agreement_core.py")
        raw = sys.stdin.buffer.read(REQUEST_MAX + 1)
        if len(raw) > REQUEST_MAX:
            raise ValueError("the request is too large")
        out = answer(json.loads(raw.decode("utf-8")), sys.argv[1])
    except BaseException as exc:             # noqa: BLE001 - whatever went wrong goes back as text; the gate decides what it means
        out = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    sys.stdout.write(json.dumps(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
