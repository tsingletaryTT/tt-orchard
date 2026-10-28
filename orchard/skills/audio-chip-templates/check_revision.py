# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Refuses to start the server when the checkpoint the chip runtime would load is not the revision a package was staged for (issue 169).

    PYTHONPATH=<stable-audio-3 checkout> python check_revision.py REVISION MODEL_NAME

The model library resolves the checkpoint from the local Hugging Face cache (the chip runtime runs offline), and this script makes the same call. The
revision is the name of the snapshot directory that holds the checkpoint, which is also what the chip runtime records for a run. Exit 0: the same revision.
Exit 2: the library, the model or the checkpoint was not found. Exit 3: another revision. No device is opened and nothing is downloaded.
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
        from stable_audio_3.model_configs import all_models
        _, ckpt = all_models[name].resolve()
    except Exception as exc:          # a missing library, an unknown model, or a checkpoint that is not in the cache
        print(f"check_revision: cannot find {name!r} in the local Hugging Face cache ({type(exc).__name__}). Download it with your own account first.",
              file=sys.stderr)
        return 2
    got = Path(ckpt).parent.name
    if got != pin:
        print(f"check_revision: the cache resolves {name!r} to revision {got}, and this package was staged for revision {pin}. Not starting.", file=sys.stderr)
        return 3
    print(f"check_revision: {name} is revision {pin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
