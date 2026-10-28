# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Builds a Hugging Face cache directory that holds only the pinned revision of a model, so the chip runtime cannot load another one (issue 169).

    PYTHONPATH=<stable-audio-3 checkout> python check_revision.py REVISION MODEL_NAME PRIVATE_HUB_DIR
    export HF_HUB_CACHE=PRIVATE_HUB_DIR        # then start the server; every clip's process inherits it

The chip runtime has no revision argument. The model library resolves the checkpoint, and transformers resolves the text encoder (a sub-folder of the same
repository), through the cache's `main` reference, and the server starts a new process for each clip. A download of another revision after the server
started would move `main`, and the next clip would load other weights than the ones the card, the evidence and the license copy describe.

This script finds the pinned snapshot in the user's cache by its explicit revision, then makes `PRIVATE_HUB_DIR/models--ORG--NAME/` with a `refs/main` file
that holds the pin and a `snapshots/<pin>` link to that snapshot (its files stay where the download put them). It then asks the cache for `main` through
that directory and requires the answer to be the pin. Nothing is written to the user's cache, nothing is downloaded and no device is opened.

Exit 0: the private cache resolves the model to the pin. Exit 2: the library, the model or the pinned snapshot is not available (the message gives the
download command). Exit 3: the private cache did not resolve to the pin.
"""
import os
import sys
from pathlib import Path


def build_private_cache(private: Path, repo_dir_name: str, pin: str, snapshot: Path) -> None:
    """`private/<repo_dir_name>/refs/main` holds `pin`, and `snapshots/<pin>` links to `snapshot`. A link or a ref that is already there is replaced."""
    repo = private / repo_dir_name
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "snapshots").mkdir(parents=True, exist_ok=True)
    tmp = repo / "refs" / f".main.{os.getpid()}"
    tmp.write_text(pin, encoding="utf-8")
    os.replace(tmp, repo / "refs" / "main")
    link = repo / "snapshots" / pin
    if link.is_symlink() or link.exists():
        if link.is_symlink() and Path(os.readlink(link)) == snapshot:
            return
        if link.is_symlink():
            link.unlink()
        else:
            raise OSError(f"{link} exists and is not a link; remove it")
    link.symlink_to(snapshot, target_is_directory=True)


def main(argv) -> int:
    if len(argv) != 4:
        print("usage: check_revision.py REVISION MODEL_NAME PRIVATE_HUB_DIR", file=sys.stderr)
        return 2
    pin, name, private = argv[1], argv[2], Path(argv[3])
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
    depth = len(Path(entry.ckpt_path).parts)                      # <repo cache>/snapshots/<revision>/<file parts>
    snapshot = Path(pinned).parents[depth - 1]
    repo_dir_name = snapshot.parents[1].name
    try:
        build_private_cache(private, repo_dir_name, pin, snapshot)
        got = try_to_load_from_cache(entry.repo_id, entry.ckpt_path, cache_dir=str(private))
    except OSError as exc:
        print(f"check_revision: cannot build the private cache at {private} ({type(exc).__name__}: {exc})", file=sys.stderr)
        return 2
    if not isinstance(got, str) or Path(got).parents[depth - 1].name != pin:
        print(f"check_revision: the private cache at {private} did not resolve {name!r} to revision {pin}. Not starting.", file=sys.stderr)
        return 3
    print(f"check_revision: {name} resolves to revision {pin} through the private cache at {private}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
