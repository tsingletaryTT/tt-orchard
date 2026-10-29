# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Runs chip_generate.py only when the checkpoint the model library resolves is the revision a package was staged for (issue 169).

    AUDIO_PIN_REVISION=<40 hex> PYTHONPATH=<stable-audio-3 checkout> python pinned_generate.py <the arguments of chip_generate.py>

The serving script starts this file once for each clip. chip_generate.py records the revision it loaded but does not compare it with anything, and the
library resolves the checkpoint through the Hugging Face cache, so a cache that changed would load other weights without a word. This wrapper resolves the
checkpoint the same way the library does, requires its snapshot name to equal AUDIO_PIN_REVISION, and only then runs chip_generate.py (from the same
directory) with the same arguments. Exit 2: no pin was given, the library or the model was not found, or both decoder flags were given. Exit 3: the resolved revision is not the pin.
Nothing is opened on a device before the check.

The decoder. The package's numbers were measured with the autoencoder decoder's transformer layers on the chip, so this wrapper adds `--chip-decoder` to
the arguments it passes on. `--host-decoder` (given here, or `AUDIO_HOST_DECODER=1` in the environment, which is how a server that cannot pass the flag
selects it) keeps the whole decoder on the host instead. chip_generate.py does not know `--host-decoder`; this wrapper removes it. Giving
`--chip-decoder` and `--host-decoder` together is refused (exit 2).
"""
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def model_name(argv) -> str:
    """The value of --model in the arguments (chip_generate.py's own default is small-music)."""
    for i, a in enumerate(argv):
        if a == "--model" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--model="):
            return a.split("=", 1)[1]
    return "small-music"


def decoder_args(argv, env) -> list:
    """The arguments for chip_generate.py: `--chip-decoder` added unless the host decoder was asked for (`--host-decoder`, or AUDIO_HOST_DECODER=1), and
    `--host-decoder` itself removed. Raises ValueError when both flags are given."""
    argv = list(argv)
    host = "--host-decoder" in argv or env.get("AUDIO_HOST_DECODER") == "1"
    if "--host-decoder" in argv and "--chip-decoder" in argv:
        raise ValueError("--chip-decoder and --host-decoder were both given; they ask for opposite things")
    argv = [a for a in argv if a != "--host-decoder"]
    if not host and "--chip-decoder" not in argv:
        argv.append("--chip-decoder")
    return argv


def main(argv) -> int:
    try:
        argv = decoder_args(argv, os.environ)
    except ValueError as exc:
        print(f"pinned_generate: {exc}. Not generating.", file=sys.stderr)
        return 2
    pin = os.environ.get("AUDIO_PIN_REVISION", "")
    if len(pin) != 40:
        print("pinned_generate: AUDIO_PIN_REVISION is not set to a 40-character revision. Start the server with run.sh. Not generating.", file=sys.stderr)
        return 2
    name = model_name(argv)
    os.environ["HF_HUB_OFFLINE"] = "1"
    try:
        from stable_audio_3.model_configs import all_models
        _, ckpt = all_models[name].resolve()
    except Exception as exc:          # a missing library, an unknown model or a checkpoint that is not in the cache
        print(f"pinned_generate: cannot resolve {name!r} ({type(exc).__name__}). Not generating.", file=sys.stderr)
        return 2
    got = Path(ckpt).parent.name
    if got != pin:
        print(f"pinned_generate: the library resolved {name!r} to revision {got}, and this package is pinned to revision {pin}. Not generating.", file=sys.stderr)
        return 3
    sys.argv = [str(HERE / "chip_generate.py"), *argv]
    runpy.run_path(str(HERE / "chip_generate.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
