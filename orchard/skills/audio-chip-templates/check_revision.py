# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Refuses to start the server when the checkpoint the chip runtime would load is not the revision a package was staged for (issue 169).

    PYTHONPATH=<stable-audio-3 checkout> python check_revision.py REVISION MODEL_NAME

The chip runtime has no revision argument. It asks the model library for the checkpoint (`all_models[name].resolve()`), and the library resolves it
through the local Hugging Face cache's `main` reference (the runtime runs offline). This script makes the same call and compares the revision of the
snapshot it returns, which is the name of the snapshot directory, with the pin. It also looks the pinned snapshot up by its explicit revision, so the
message says which of the two things is wrong. No device is opened and nothing is downloaded.

Exit 0: the runtime would load the pinned revision. Exit 2: the library, the model or the pinned snapshot is not in the cache. Exit 3: the runtime would
load another revision. Exit 4: the pinned snapshot is in the cache and the runtime cannot resolve the model (no `main` reference). For 3 and 4 the message
gives the line that points `main` at the pin.
"""
import os
import sys
from pathlib import Path


def main(argv) -> int:
    if len(argv) != 3:
        print("usage: check_revision.py REVISION MODEL_NAME", file=sys.stderr)
        return 2
    pin, name = argv[1], argv[2]
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from huggingface_hub import try_to_load_from_cache
        from stable_audio_3.model_configs import all_models
        entry = all_models[name]
        pinned = try_to_load_from_cache(entry.repo_id, entry.ckpt_path, revision=pin)
    except Exception as exc:          # a missing library, an unknown model or an unreadable cache
        print(f"check_revision: cannot look up {name!r} in the local Hugging Face cache ({type(exc).__name__}).", file=sys.stderr)
        return 2
    if not isinstance(pinned, str):
        print(f"check_revision: revision {pin} of {entry.repo_id} is not in the local cache. Download that revision with your own account: "
              f"hf download {entry.repo_id} --revision {pin}", file=sys.stderr)
        return 2
    refs_main = Path(pinned).parents[2] / "refs" / "main"           # <repo cache>/snapshots/<revision>/<file>  ->  <repo cache>/refs/main
    fix = f"To point the runtime at the pinned revision: echo {pin} > {refs_main}"
    try:
        _, ckpt = entry.resolve()
    except Exception as exc:
        print(f"check_revision: the pinned snapshot is in the cache, and the runtime cannot resolve {name!r} through the cache's main reference "
              f"({type(exc).__name__}). {fix}", file=sys.stderr)
        return 4
    got = Path(ckpt).parent.name
    if got != pin:
        print(f"check_revision: the runtime would load revision {got} of {name!r}, and this package was staged for revision {pin}. Not starting. {fix}",
              file=sys.stderr)
        return 3
    print(f"check_revision: {name} is revision {pin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
