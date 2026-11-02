# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""A port module for the scoring check (issue 375) that scores through a reranker server running on a board next to the check.

The scoring check (orchard/skills/scoring-check-templates/scoring_check.py) asks a port module for one score per prompt, where a prompt is
a list of token ids. This module answers by sending the prompts to a local HTTP server that holds the model on a chip (the public bundle
raahemnabeel/qwen3-reranker-4b, started with `tt-model serve`, runs its own ttnn model on one chip of a board) and returning the raw scores
the server computes. The chip produces every number. The check computes the parity numbers from them.

Why text: the server's API takes a query and a document as text and builds the prompt itself, as the Qwen3-Reranker model card does. This
module turns each list of ids back into those two texts (and the instruction) with the model's tokenizer. Before anything is sent it
builds the prompt again from the texts, with the same prefix and suffix, and refuses the run if the ids differ from the ones it was given.
After each request it compares the server's token count with the number of ids it sent. So a text that tokenizes differently, or a server
that builds another prompt, is an error and not a score.

Interface (the template's): load(model_dir, mesh_device), score_pairs(model, pairs), close(model). `embed` is not provided. The server's
address is the environment variable ORCHARD_SCORE_SERVER_URL, for example http://127.0.0.1:20000. It must be http, on a loopback address
(127.0.0.1, ::1 or localhost) and name a port. The request does not use a proxy. The port also leaves the device the template opened
alone: the template opens a device of its own for the worker, and the server holds a different chip of the leased board.

What this does not show: the model runs in the server, which is another process. The template's check that a port module mentions ttnn is
met by this comment, and that check cannot tell a port that computes from one that forwards. Here forwarding is the design, and the run log
says which chip the server held. Prompts longer than the server's own limit are refused by the server.
"""
import http.client
import json
import math
import os
from urllib.parse import urlsplit

ENV_URL = "ORCHARD_SCORE_SERVER_URL"
LOOPBACK_HOSTS = ("127.0.0.1", "::1", "localhost")
MAX_PER_REQUEST = 32               # choice: the server accepts more; a group is 4 to 20 prompts
TIMEOUT_S = 900.0                  # one request: the server's own limit for a long prompt is far below this
REPLY_MAX = 8 * 1024 * 1024        # choice: a reply holds one small object per prompt
TOKENIZER_FILE_MAX = 64 * 1024 * 1024
DEFAULT_INSTRUCTION = "Given a web search query, retrieve relevant passages that answer the query"
PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
          'Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_QUERY, _DOC = "\n<Query>: ", "\n<Document>: "
_HEAD = "<Instruct>: "


def _server(url: str):
    """(host, port) of the server, or a RuntimeError that names the variable."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        parts = port = None
    if (parts is None or parts.scheme != "http" or parts.hostname not in LOOPBACK_HOSTS or port is None or parts.username is not None
            or parts.password is not None or parts.path not in ("", "/") or parts.query or parts.fragment):
        raise RuntimeError(f"{ENV_URL} must be http://127.0.0.1:PORT (or ::1 or localhost) with a port, got {url!r}")
    return parts.hostname, port


def _call(model: dict, method: str, path: str, body=None):
    """One request to the server without a proxy. Returns the parsed JSON object, or raises RuntimeError with the status."""
    conn = http.client.HTTPConnection(model["host"], model["port"], timeout=TIMEOUT_S)
    try:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        conn.request(method, path, body=data, headers={"content-type": "application/json"} if data else {})
        resp = conn.getresponse()
        raw = resp.read(REPLY_MAX + 1)
        if len(raw) > REPLY_MAX:
            raise RuntimeError(f"the server's reply to {path} is larger than {REPLY_MAX} bytes")
        if resp.status != 200:
            raise RuntimeError(f"the server answered {resp.status} to {method} {path}: {raw[:300].decode('utf-8', 'replace')}")
        try:
            return json.loads(raw.decode("utf-8"))
        except ValueError as exc:
            raise RuntimeError(f"the server's reply to {path} is not JSON: {exc}")
    except OSError as exc:
        raise RuntimeError(f"cannot reach the server for {method} {path}: {type(exc).__name__}: {exc}")
    finally:
        conn.close()


def load(model_dir: str, mesh_device):
    """Read the tokenizer from the snapshot, and check that the server is up. The model itself is already on the server's chip."""
    from tokenizers import Tokenizer
    host, port = _server(os.environ.get(ENV_URL, ""))
    try:
        with open(os.path.join(model_dir, "tokenizer.json"), "rb") as f:
            raw = f.read(TOKENIZER_FILE_MAX + 1)
        if len(raw) > TOKENIZER_FILE_MAX:
            raise RuntimeError("tokenizer.json is larger than the limit")
        tok = Tokenizer.from_str(raw.decode("utf-8"))
    except OSError as exc:
        raise RuntimeError(f"cannot read tokenizer.json in the snapshot: {exc}")
    model = {"host": host, "port": port, "tok": tok,
             "prefix": tok.encode(PREFIX, add_special_tokens=False).ids, "suffix": tok.encode(SUFFIX, add_special_tokens=False).ids}
    try:
        _call(model, "GET", "/health")
    except RuntimeError as exc:
        raise RuntimeError(f"the server's health check failed: {exc}")
    return model


def to_texts(model: dict, ids: list) -> tuple:
    """(instruction, query, document) for one prompt's ids. A RuntimeError when the ids are not a prompt of this model, or when the prompt
    built again from the three texts has other ids."""
    tok = model["tok"]
    text = tok.decode(ids, skip_special_tokens=False)
    if not (text.startswith(PREFIX) and text.endswith(SUFFIX) and len(text) > len(PREFIX) + len(SUFFIX)):
        raise RuntimeError("these ids are not a prompt: they do not start with the system prefix and end with the assistant suffix")
    body = text[len(PREFIX):len(text) - len(SUFFIX)]
    q_at = body.find(_QUERY)
    d_at = body.find(_DOC, q_at + len(_QUERY)) if q_at >= 0 else -1
    if not body.startswith(_HEAD) or q_at < 0 or d_at < 0:
        raise RuntimeError("these ids are not a prompt: the body has no <Instruct>, <Query> and <Document> parts")
    instruction, query, document = body[len(_HEAD):q_at], body[q_at + len(_QUERY):d_at], body[d_at + len(_DOC):]
    again = model["prefix"] + tok.encode(_HEAD + instruction + _QUERY + query + _DOC + document, add_special_tokens=False).ids + model["suffix"]
    if again != [int(i) for i in ids]:
        raise RuntimeError("the prompt does not come back as the same ids when built from its texts "
                           f"({len(again)} ids against {len(ids)}): the server would score another prompt")
    return instruction, query, document


def score_pairs(model: dict, pairs: list) -> list:
    """The server's raw score (logit yes minus logit no) for each prompt, in the order given."""
    items = [to_texts(model, ids) for ids in pairs]              # all refused before the first request
    out = [None] * len(items)
    by_instruction: dict = {}
    for i, (ins, _, _) in enumerate(items):
        by_instruction.setdefault(ins, []).append(i)
    for ins, idx in by_instruction.items():
        for start in range(0, len(idx), MAX_PER_REQUEST):
            chunk = idx[start:start + MAX_PER_REQUEST]
            reply = _call(model, "POST", "/v1/score", {"text_1": [items[i][1] for i in chunk], "text_2": [items[i][2] for i in chunk],
                                                       "instruction": ins, "raw_scores": True})
            data = reply.get("data") if isinstance(reply, dict) else None
            if not isinstance(data, list) or len(data) != len(chunk):
                raise RuntimeError(f"the server returned {len(data) if isinstance(data, list) else data!r} scores for {len(chunk)} prompts")
            for k, row in enumerate(data):
                score = row.get("raw_score") if isinstance(row, dict) else None
                if not isinstance(row, dict) or row.get("index") != k:
                    raise RuntimeError(f"the server's score {k} has index {row.get('index') if isinstance(row, dict) else row!r}")
                if type(score) not in (int, float) or not math.isfinite(score):
                    raise RuntimeError(f"the server's raw_score {k} is not a finite number: {score!r}")
                out[chunk[k]] = float(score)
            usage = reply.get("usage") if isinstance(reply.get("usage"), dict) else {}
            sent = sum(len(pairs[i]) for i in chunk)
            if usage.get("total_tokens") != sent:
                raise RuntimeError(f"the server counted {usage.get('total_tokens')!r} tokens and the request held {sent}: it built another prompt")
    return out


def close(model) -> None:
    """Nothing to release: the server and its chip belong to whoever started it."""
