"""A stand-in for `tt-model`, for tests/test_container_template.py.

It answers only `tt-model serve <package> ... --print`: it prints a note line and then the docker
command from tests/container_fakes.py, as tt-model's --print does, and starts nothing. Every call
is appended to the "calls" file named in the JSON config at $FAKE_TT_MODEL_CONFIG, with HOME and
HF_HOME, so a test can check what the template asked for. The config's "printed" dict passes
keyword arguments to printed_argv (for example {"whole_dir": true}).
"""
from __future__ import annotations

import json
import os
import shlex
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from container_fakes import printed_argv  # noqa: E402


def main() -> int:
    args = sys.argv[1:]
    with open(os.environ["FAKE_TT_MODEL_CONFIG"], encoding="utf-8") as f:
        cfg = json.load(f)
    with open(cfg["calls"], "a", encoding="utf-8") as f:
        f.write(json.dumps({"argv": args, "HOME": os.environ.get("HOME"),
                            "HF_HOME": os.environ.get("HF_HOME")}) + "\n")
    if args[:1] != ["serve"] or "--print" not in args:
        print("fake tt-model: only `serve <package> ... --print` is supported", file=sys.stderr)
        return 1
    opt = lambda name: args[args.index(name) + 1]      # noqa: E731
    ids = [int(i) for i in opt("--device-id").split(",")]
    print("• profile 'batch32' (the author's default)")
    print(shlex.join(printed_argv(hf=os.environ["HF_HOME"], pkg_cache=cfg["pkg_cache"],
                                  port=int(opt("--port")), device_ids=ids, **cfg.get("printed", {}))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
