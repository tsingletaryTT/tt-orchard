# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Helpers shared by the Cans view tests: a fake checkout with can specs, and staged bundles with records, all in a temporary folder."""
import json
import textwrap
from pathlib import Path

from orchard import can

REV = "d18cca4c121edb87d022116d281ce212c9136f57"

SPEC = textwrap.dedent('''
    [can]
    name = "%(name)s"
    kind = "%(kind)s"
    app = "toy_asgi:app"
    model_py = "pkg/toy_model.py"
    readme = "pkg/README.src.md"
    mesh = "%(mesh)s"
    device_count = %(devices)d
    port = 20123
    requirements = ["ttnn==0.80.0", "numpy==1.26.4", "fastapi==0.136.3", "uvicorn==0.45.0", "pydantic==2.12.5", "pillow==10.4.0"]
    %(profile)s
    [weights]
    repo = "%(repo)s"
    revision = "%(rev)s"
''')

CARD = textwrap.dedent('''
    # Toy

    ## Expected performance

    | Metric | Value |
    | --- | --- |
    | Decode | 31.4 tok/s |
    | Start | 212 s |

    ## Limitations

    Greedy decoding only.
    Context is 2048 tokens.
''')


def make_checkout(tmp_path: Path) -> Path:
    """A checkout with `config/` and the two source files every spec names."""
    root = tmp_path / "checkout"
    (root / "config").mkdir(parents=True)
    (root / "pkg").mkdir()
    (root / "pkg" / "toy_model.py").write_text("app = None\n")
    (root / "pkg" / "README.src.md").write_text(CARD)
    return root


def add_spec(root: Path, slug: str, *, name=None, kind="tt-dit-server", mesh="P150", devices=1, profile="catalog",
             repo="someone/toy-model", rev=REV) -> str:
    name = name or f"{slug}-blackhole"
    text = SPEC % {"name": name, "kind": kind, "mesh": mesh, "devices": devices, "repo": repo, "rev": rev,
                   "profile": f'card_profile = "{profile}"' if profile else ""}
    (root / "config" / f"can.{slug}.toml").write_text(text)
    return name


def stage_bundle(cans: Path, name: str, *, card=CARD, extra: dict | None = None) -> Path:
    """A staged bundle directory: the files the gate requires (contents do not matter to the view) and a SHA256SUMS that matches."""
    b = cans / name
    b.mkdir(parents=True)
    for rel, text in {"run.sh": "#!/bin/sh\n", "install.sh": "#!/bin/sh\n", "requirements.txt": "ttnn==0.80.0\n", "README.md": card,
                      "tt_kernel_manifest.json": "{}\n", "toy_model.py": "app = None\n", **(extra or {})}.items():
        p = b / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    can.write_sums(b)
    return b


def record(cans: Path, name: str, bundle: Path, *, gate=None, boot=None) -> Path:
    return can.write_state(cans, name, bundle, gate=gate, boot=boot)


def passed_boot(rate=31.4, ready=212.0, cache="empty"):
    return can.boot_record({"passed": True, "checks": {"health": {"ok": True}}, "rate_timings": {"decode_tok_s": rate}, "ready_s": ready}, cache)


def by_name(answer: dict, name: str) -> dict:
    return next(c for c in answer["cans"] if c["name"] == name)


def dump(answer) -> str:
    return json.dumps(answer)
