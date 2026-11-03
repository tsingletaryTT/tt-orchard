# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Fakes for the embedding kind (issue 376) and for media in a pair file (issue 377) of the scoring check template. They build on
scoring_check_fakes: a run directory whose stage 1 files hold vectors, and port modules that rebuild the reference vector from the id in the
prompt (an honest one) or change it (a noisy one). Nothing here touches a chip."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from orchard import scoring_check
from scoring_check_fakes import make_run

EMBED_DIM = 8
EMBED_KINDS = ["text", "multilingual", "code", "long", "document"]
N_EMBED = 30                                           # six inputs of each kind
MEDIA_REL = "stages/1/evidence/reference/media/"


def fake_vector(n: int) -> list:
    """The reference vector of input n: a fixed pseudo-random vector. The honest port rebuilds it from the id in the prompt."""
    r = random.Random(n)
    return [r.uniform(-1, 1) for _ in range(EMBED_DIM)]


def embed_rows(media=None) -> list[dict]:
    """The embedding inputs. `media` maps an input number to the list of media entries of that row."""
    rows = []
    for n in range(N_EMBED):
        row = {"id": f"e{n}", "kind": EMBED_KINDS[n % len(EMBED_KINDS)], "text": "t", "input_ids": [n + 1, 7, 7]}
        if media and n in media:
            row["media"] = media[n]
        rows.append(row)
    return rows


def media_entry(name: str, data: bytes, kind: str = "image") -> dict:
    return {"kind": kind, "path": MEDIA_REL + name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


HONEST_EMBED = '''
import ttnn, random
def load(model_dir, mesh_device):
    return {"dir": model_dir, "device": mesh_device}
def embed(model, ids):
    r = random.Random(ids[0] - 1)
    return [r.uniform(-1, 1) for _ in range(8)]
def close(model):
    pass
'''
# Every element is off by 0.5 in alternating directions: the direction of the vector changes and the cosine falls below the bound.
NOISY_EMBED = HONEST_EMBED.replace("return [r.uniform(-1, 1) for _ in range(8)]",
                                   "return [r.uniform(-1, 1) + (0.5 if k % 2 else -0.5) for k in range(8)]")
# A port that takes media: it checks that the bytes it is given are the file's, and encodes the length in the vector's sign so a wrong or
# missing argument shows. Its vector is the reference vector when the media is as the pair file says.
MEDIA_EMBED = '''
import ttnn, random, hashlib
def load(model_dir, mesh_device):
    return {}
def embed(model, ids, media=None):
    n = ids[0] - 1
    if n % 5 == 0:                       # the inputs the test gave a file
        assert media and media[0]["kind"] == "image" and hashlib.sha256(media[0]["data"]).hexdigest() == media[0]["sha256"], "bad media"
        assert media[0]["name"].endswith(".png")
    else:
        assert media is None or media == [], "unexpected media"
    r = random.Random(n)
    return [r.uniform(-1, 1) for _ in range(8)]
def close(model):
    pass
'''
MEDIA_SCORE = '''
import ttnn, hashlib
def load(model_dir, mesh_device):
    return {}
def score_pairs(model, pairs, media=None):
    assert media is not None and len(media) == len(pairs), "media must be a list with one entry for each prompt"
    out = []
    for ids, m in zip(pairs, media):
        if ids[1] == 2:                  # the second document of every group has a file
            assert m and hashlib.sha256(m[0]["data"]).hexdigest() == m[0]["sha256"], "bad media"
        out.append((ids[-1] - 100) / 10)
    return out
def close(model):
    pass
'''


def make_embed_run(tmp_path: Path, port_src: str = HONEST_EMBED, *, rows=None, vectors=None, config: dict | None = None,
                   media_files: dict | None = None):
    """A run directory for kind embedding: the stage 1 pair file and reference (vectors), the port, the template and its config.
    `media_files` maps a name to bytes written under stages/1/evidence/reference/media/. Returns (run_dir, stage_dir, snapshot_dir)."""
    run, stage, snap = make_run(tmp_path, port_src)
    ev = run / "stages" / "1" / "evidence" / "reference"
    rows = embed_rows() if rows is None else rows
    pairs_bytes = (json.dumps({"schema": "orchard-scoring-pairs/1", "model_kind": "embedding", "pairs": rows}) + "\n").encode()
    (ev / "scoring-pairs.json").write_bytes(pairs_bytes)
    vecs = [fake_vector(n) for n in range(len(rows))] if vectors is None else vectors
    ref = {"schema": "orchard-scoring-reference/1", "model_kind": "embedding", "precision": "fp32",
           "pairs": {"file": "scoring-pairs.json", "sha256": hashlib.sha256(pairs_bytes).hexdigest(), "count": len(rows)},
           "vectors": vecs}
    (ev / "scoring-reference.json").write_text(json.dumps(ref))
    if media_files:
        (ev / "media").mkdir()
        for name, data in media_files.items():
            (ev / "media" / name).write_bytes(data)
    (stage / scoring_check.CONFIG_NAME).write_text(json.dumps(config if config is not None else scoring_check.draft_config(str(snap), "embedding")))
    return run, stage, snap
