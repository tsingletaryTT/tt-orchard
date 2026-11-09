# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Stage 5 of a full port: the supervisor's own serving check. Nobody edits this file and nobody runs it by hand.

The supervisor copies this file to stages/5/port_serve.py (next to a copy of the shipped stage 2 check, port_check.py, a copy of the port that
passed stage 2 or 3 under stages/5/port/, the acceptance corpus corpus.json and port_serve_config.json) and runs

    python3 -P -s stages/5/port_serve.py

under the hardware lease. The orchestration is the one scripts/minicpm3/serve.py uses (a standard-library OpenAI-style server, one request at a
time, greedy decoding, loopback only) with the port in a worker process as stage 2's check runs it. This file:

1. runs the port module in the worker of port_check.py (the copy next to this file; its sha256 must equal the config's), so the port's prints,
   exceptions and imports stay out of this process;
2. starts the server on 127.0.0.1 with an ephemeral port (never 8000 or 8100 to 8103) and replays the corpus through it:
   POST /v1/completions with a prompt of token ids. The teacher-forced entry adds `forced_tokens` (an extension: the ids fed in place of the
   port's own, as stage 2's check does), the free-run entry does not;
3. refuses to go on when any position of the replay differs from the corpus: it prints `FAILED: ...` and exits 2 without a readiness line;
4. prints the readiness line `READY <model> on 127.0.0.1:<port> pid <pid> corpus N/N positions`;
5. asks the passkey questions through POST /v1/chat/completions (each twice) and the canary through /v1/completions;
6. writes evidence/port-serve.json and prints `PORT_CHECK_SHA256 <sha256 of the record>` as the last line.

Endpoints: GET /v1/models; POST /v1/completions (prompt: text or a list of ids, max_tokens, ignore_eos, forced_tokens); POST /v1/chat/completions
(messages, max_tokens). Both POSTs return `choices[0].token_ids` (an extension, like the one vLLM has) and the usage counts. Sampling is greedy.

The chat template is the one in the snapshot's tokenizer_config.json when it has one and jinja2 imports; otherwise the messages are written as
`role: content` lines and the record says `plain`. What the record cannot show is in orchard/full_port_serve.py.
"""
import hashlib
import http.client
import json
import os
import re
import secrets
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SCHEMA = "orchard-port-serve/1"
HERE = Path(__file__).resolve().parent                    # stages/5
RUN_DIR = HERE.parents[1]
CONFIG_FILE = HERE / "port_serve_config.json"
CORPUS_FILE = HERE / "corpus.json"
RECORD = HERE / "evidence" / "port-serve.json"
WORKER_FILE = HERE / "port_check.py"
PORT_DIR = HERE / "port"
MODEL_ID = "orchard-port"
FORBIDDEN_PORTS = {8000, 8011, 8780}                      # other work on this machine; 8100 to 8103 are refused below
MAX_BODY = 8 << 20
CLIENT_TIMEOUT_S = 3000.0
CLOSE_WAIT_S = 60.0           # how long to wait for a running completion before the worker is closed or stopped
FILLER = "The grass is green. The sky is blue. The sun is warm. Here we go again. "
NEEDLE = "The pass key is {key}. Remember it. "
QUESTION = "What is the pass key? The pass key is"
PLAIN_ANSWER_TOKENS = 12      # the passkey answer limit for a model whose chat template does not think
OFF_ANSWER_TOKENS = 64        # with thinking switched off through the template's own switch
THINK_ANSWER_TOKENS = 512     # no switch: the reply is read after the closing think tag, so it needs room for the thinking before it (issue 641)
SWITCH_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,40}")
CACHE_ENTRY_CAP = 2_000_000                               # the cache directory walk stops counting here


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path) -> str:
    return sha_bytes(Path(path).read_bytes())  # hardening: ok the supervisor wrote this file before the test; no agent works in stage 5, and the gate compares its hash


class ServeFailed(Exception):
    """The check failed; the message is printed after FAILED: and the run exits 2."""


def load_worker(expected_sha: str):
    """The module of the shipped stage 2 check, executed from the bytes whose sha256 was compared (not read again). It is used for its worker and
    its Worker class; its main() is never called."""
    data = WORKER_FILE.read_bytes()  # hardening: ok the supervisor wrote this file before the test; no agent works in stage 5, and the gate compares its hash
    if sha_bytes(data) != expected_sha:
        raise ServeFailed("port_check.py in this directory is not the shipped file (its sha256 differs from the config's)")
    mod = types.ModuleType("port_check")
    mod.__file__ = str(WORKER_FILE)
    exec(compile(data, str(WORKER_FILE), "exec"), mod.__dict__)
    return mod


# ---- the server ----------------------------------------------------------------------------------------------------------------

class ServerClosed(Exception):
    """The engine has been closed: the worker is shutting down and takes no more requests."""


class Engine:
    """The port, one request at a time. `run` returns the ids the port chose for a prompt of ids."""

    def __init__(self, worker, max_len: int, tokenizer, stop_ids, render, vocab_size=None):
        self.w, self.max_len, self.tok, self.stop_ids, self.render = worker, max_len, tokenizer, set(stop_ids), render
        # The number of token ids the loaded model can look up (the tokenizer's vocabulary, added tokens included). An id at or beyond it is
        # refused before the port runs, since a lookup past the embedding table fails inside the port (review of PR 490, round 5). None: no check.
        self.vocab_size = vocab_size
        self.lock = threading.Lock()
        self.closed = False
        # Until the corpus replay has matched, the server answers only the replay's own requests, which carry this per-run secret in a header.
        # An ordinary loopback client gets 503, so it cannot get a completion from a port that is still on trial, and cannot change the worker's
        # state in the middle of the replay (review of PR 490, round 4).
        self.open = False
        self.replay_token = secrets.token_hex(16)

    def run(self, prompt, max_tokens, *, forced=None, ignore_eos=False):
        """(ids, finish): at most `max_tokens` ids. With `forced` (a list of max_tokens - 1 ids) each step after the first is fed the forced id in
        place of the port's own. The position of a decoded token is its index in the whole sequence, the prompt included."""
        if not prompt:
            raise ValueError("the prompt has no tokens")
        if len(prompt) + max_tokens > self.max_len:
            raise ValueError(f"the prompt is {len(prompt)} tokens and {max_tokens} more were asked; the server's context is {self.max_len}")
        if forced is not None and len(forced) != max_tokens - 1:
            raise ValueError("forced_tokens must hold max_tokens - 1 ids")
        # A negative id would index from the end of an embedding table, or fail inside the port. It is refused here, before the port runs.
        if any(i < 0 for i in prompt) or (forced is not None and any(i < 0 for i in forced)):
            raise ValueError("token ids must not be negative")
        if self.vocab_size is not None and (any(i >= self.vocab_size for i in prompt) or (forced is not None and any(i >= self.vocab_size for i in forced))):
            raise ValueError(f"token ids must be below the model's vocabulary size, {self.vocab_size}")
        with self.lock:
            if self.closed:                 # the lock was held by a running completion while the server was closing: the worker is gone
                raise ServerClosed("the server is shutting down")
            out, finish = [], "length"
            nxt = self.w.ask(op="prefill", ids=prompt)
            for k in range(max_tokens):
                if k:
                    prev = forced[k - 1] if forced is not None else out[-1]
                    nxt = self.w.ask(op="decode", token=prev, position=len(prompt) + k - 1)
                if not ignore_eos and nxt in self.stop_ids:
                    finish = "stop"
                    break
                out.append(nxt)
            return out, finish


    def quiesce(self, timeout: float) -> bool:
        """Wait for the completion that is running (the lock), then refuse every later request, so the worker can be closed or stopped while
        nothing uses its pipes. False when the running completion did not finish in `timeout` seconds; requests are refused then too."""
        self.closed = True              # a request that is not yet inside the lock sees this when it gets there
        got = self.lock.acquire(timeout=timeout)
        if got:
            self.lock.release()
        return got


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.rstrip("/") == "/v1/models":
                self._json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "orchard"}]})
            else:
                self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            path = self.path.rstrip("/")
            if path not in ("/v1/completions", "/v1/chat/completions"):
                return self._json(404, {"error": {"message": "not found"}})
            if not engine.open and self.headers.get("X-Orchard-Replay") != engine.replay_token:
                return self._json(503, {"error": {"message": "the server is replaying its acceptance corpus and takes no other request yet"}})
            try:
                n = int(self.headers.get("Content-Length", "0"))
                if not 0 < n <= MAX_BODY:
                    return self._json(400, {"error": {"message": f"request body must be 1 byte to {MAX_BODY} bytes"}})
                req = json.loads(self.rfile.read(n))
                if not isinstance(req, dict):
                    raise ValueError("the body must be a JSON object")
                max_tokens = req.get("max_tokens", 16)
                if type(max_tokens) is not int or not 1 <= max_tokens <= 4096:
                    raise ValueError("max_tokens must be a whole number from 1 to 4096")
                if path == "/v1/chat/completions":
                    messages = req.get("messages")
                    if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) and isinstance(m.get("content"), str) for m in messages):
                        raise ValueError("messages must be a non-empty list of {role, content} with string content")
                    kwargs = req.get("chat_template_kwargs", {})
                    if not (isinstance(kwargs, dict) and len(kwargs) <= 4 and all(isinstance(k, str) and SWITCH_NAME.fullmatch(k) and type(v) is bool
                                                                              for k, v in kwargs.items())):
                        raise ValueError("chat_template_kwargs must be an object of at most four boolean switches")
                    prompt = engine.tok.encode(engine.render(messages, **kwargs), add_special_tokens=False).ids
                else:
                    p = req.get("prompt")
                    if isinstance(p, str) and p:
                        prompt = engine.tok.encode(p, add_special_tokens=False).ids
                    elif isinstance(p, list) and p and all(type(i) is int for i in p):
                        prompt = p
                    else:
                        raise ValueError("prompt must be text or a non-empty list of token ids")
                forced = req.get("forced_tokens")
                if forced is not None and not (isinstance(forced, list) and all(type(i) is int for i in forced)):
                    raise ValueError("forced_tokens must be a list of token ids")
                ids, finish = engine.run(prompt, max_tokens, forced=forced, ignore_eos=req.get("ignore_eos") is True)
            except ServerClosed as exc:
                return self._json(503, {"error": {"message": str(exc)}})
            except (ValueError, KeyError, TypeError) as exc:
                return self._json(400, {"error": {"message": str(exc)[:300]}})
            except Exception as exc:  # noqa: BLE001  a failure in the port must reach the client and the log
                sys.stderr.write(f"serve: generation failed: {type(exc).__name__}\n")
                return self._json(500, {"error": {"message": f"generation failed: {type(exc).__name__}"}})
            text, raw_text = decode_reply(engine.tok, ids)
            choice = {"index": 0, "finish_reason": finish, "token_ids": ids, "raw_text": raw_text}
            if path == "/v1/chat/completions":
                choice["message"] = {"role": "assistant", "content": text}
            else:
                choice["text"] = text
            self._json(200, {"id": f"cmpl-{int(time.time() * 1000)}", "model": MODEL_ID, "choices": [choice],
                             "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(ids), "total_tokens": len(prompt) + len(ids)}})

        def log_message(self, fmt, *args):
            sys.stderr.write("serve: " + fmt % args + "\n")
    return Handler


def post(port: int, path: str, body: dict, token: str | None = None) -> dict:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=CLIENT_TIMEOUT_S)
    try:
        headers = {"Content-Type": "application/json"}
        if token is not None:
            headers["X-Orchard-Replay"] = token
        conn.request("POST", path, json.dumps(body), headers)
        resp = conn.getresponse()
        data = json.loads(resp.read())
    finally:
        conn.close()
    if resp.status != 200:
        raise ServeFailed(f"the server answered {resp.status} to {path}: {str(data)[:300]}")
    return data


def get(port: int, path: str) -> dict:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    try:
        conn.request("GET", path)
        resp = conn.getresponse()
        return json.loads(resp.read())
    finally:
        conn.close()


# ---- helpers: chat template, stop ids, cache, passkey prompts ------------------------------------------------------------------------

def read_json(path):
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))  # hardening: ok the supervisor wrote this file before the test; no agent works in stage 5, and the gate compares its hash
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def chat_template_text(snapshot: Path):
    """(template text, where it came from): `chat_template` in tokenizer_config.json, else the file chat_template.jinja beside it (newer Hugging Face
    snapshots keep it there), else (None, None)."""
    cfg = read_json(snapshot / "tokenizer_config.json")
    template = cfg.get("chat_template")
    if isinstance(template, str) and template:
        return template, "tokenizer_config"
    try:
        text = (snapshot / "chat_template.jinja").read_text(encoding="utf-8")  # hardening: ok a regular file of the model snapshot the supervisor chose, read once before the server starts
    except (OSError, ValueError):
        return None, None
    return (text, "chat_template.jinja") if text.strip() else (None, None)


def thinking_support(render) -> dict:
    """{"template_thinks", "switch"} from the ACTIVE generation rendering of a one-turn conversation, not from the template's source text (which also
    holds comments and branches that only format earlier assistant turns). The template thinks when the generation prompt ends inside an opened
    `<think>` block (the last `<think>` has no `</think>` after it). Its switch is `enable_thinking` when passing it changes the rendering."""
    try:
        base = render([{"role": "user", "content": "x"}])
        off = render([{"role": "user", "content": "x"}], enable_thinking=False)
    except Exception:  # noqa: BLE001  a template that cannot render the probe is treated as not thinking
        return {"template_thinks": False, "switch": None}
    opens, closes = base.rfind("<think>"), base.rfind("</think>")
    return {"template_thinks": opens != -1 and opens > closes, "switch": "enable_thinking" if off != base else None}


def passkey_plan(support: dict) -> dict:
    """How the passkey questions are asked for a model with these thinking facts: thinking off through the switch (a short answer limit), or left on
    and read after the closing tag (a longer limit), or as before for a model that does not think."""
    if support.get("template_thinks") and support.get("switch"):
        return {"off": True, "kwargs": {support["switch"]: False}, "max_tokens": OFF_ANSWER_TOKENS}
    if support.get("template_thinks"):
        return {"off": False, "kwargs": {}, "max_tokens": THINK_ANSWER_TOKENS}
    return {"off": False, "kwargs": {}, "max_tokens": PLAIN_ANSWER_TOKENS}


def renderer_and_thinking(snapshot: Path):
    """(render, name, support): the chat renderer and the thinking facts of the template that is actually used. When jinja cannot run the template the
    renderer is the plain form, and a template that was not used says nothing about how the prompts were rendered, so the facts are non-thinking."""
    render, name = chat_renderer(snapshot)
    return render, name, (thinking_support(render) if name != "plain" else {"template_thinks": False, "switch": None})


def decode_reply(tok, ids) -> tuple[str, str]:
    """(clean text, raw text) of generated ids: the clean text drops special tokens, the raw text keeps them, so a closing think tag the tokenizer
    registers as special is still there for the gate to read."""
    return tok.decode(ids, skip_special_tokens=True), tok.decode(ids, skip_special_tokens=False)


def answer_limit(tok, render, text: str, kwargs: dict, *, plan_max: int, max_len: int) -> int:
    """The answer limit for a passkey request: the plan's, clamped so the rendered prompt (measured after the chat template with the same switches the
    request carries) and the answer fit the context. The server checks the rendered count, so the filler's requested length is not the measure."""
    n = len(tok.encode(render([{"role": "user", "content": text}], **kwargs), add_special_tokens=False).ids)
    room = max_len - n - 1
    if room < 1:
        raise ServeFailed(f"the rendered passkey prompt is {n} tokens and the context is {max_len}: no room for an answer")
    return min(plan_max, room)


def chat_renderer(snapshot: Path):
    """(render, name): `render(messages, **extra)` gives the prompt text; `extra` are template variables such as `enable_thinking`. The snapshot's
    own template when jinja2 can run it, else plain lines."""
    template, origin = chat_template_text(snapshot)
    cfg = read_json(snapshot / "tokenizer_config.json")
    if isinstance(template, str) and template:
        try:
            from jinja2.sandbox import ImmutableSandboxedEnvironment
            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

            def raise_exception(message):
                raise ValueError(message)
            tpl = env.from_string(template)
            tokens = {k: (v.get("content") if isinstance(v, dict) else v) for k, v in cfg.items() if k.endswith("_token")}

            def render(messages, **extra):
                return tpl.render(messages=messages, add_generation_prompt=True, raise_exception=raise_exception, **tokens, **extra)
            render([{"role": "user", "content": "x"}])            # a template that cannot render a plain turn is not used
            return render, origin
        except Exception:  # noqa: BLE001  any template or jinja2 problem means the plain form
            pass
    return (lambda messages, **extra: "".join(f"{m.get('role', 'user')}: {m['content']}\n" for m in messages) + "assistant:"), "plain"


def stop_ids(snapshot: Path) -> list:
    out = []
    for doc in (read_json(snapshot / "generation_config.json"), read_json(snapshot / "config.json")):
        value = doc.get("eos_token_id")
        out += [v for v in (value if isinstance(value, list) else [value]) if type(v) is int]
    return sorted(set(out))


def vocab_bound(snapshot: Path, tokenizer_count: int) -> int:
    """The token-id bound for requests: the larger of the tokenizer's entry count and the snapshot's `vocab_size` (top level or `text_config`).
    A padded embedding table has valid ids above the tokenizer's count (review of PR 490, round 6); ids at or above the bound are still a 400."""
    config = read_json(snapshot / "config.json")
    sizes = [tokenizer_count]
    for doc in (config, config.get("text_config") if isinstance(config.get("text_config"), dict) else {}):
        value = doc.get("vocab_size")
        if type(value) is int and 0 < value <= 10_000_000:
            sizes.append(value)
    return max(sizes)


def kernel_cache() -> tuple[str, int]:
    """(directory, files in it): TT_METAL_CACHE, else ~/.cache/tt-metal-cache. A directory that does not exist has 0 entries."""
    home = os.path.expanduser("~")
    root = os.environ.get("TT_METAL_CACHE") or os.path.join(home, ".cache", "tt-metal-cache")
    count = 0
    for _, _, files in os.walk(root):
        count += len(files)
        if count >= CACHE_ENTRY_CAP:
            break
    shown = ("~" + root[len(home):]) if home != "~" and root.startswith(home + os.sep) else root
    return shown, count


def passkey_prompt(tok, length: int, key: str) -> str:
    """Filler text with the needle in the middle and the question at the end, measured by the tokenizer to about `length` tokens (never above)."""
    def count(text):
        return len(tok.encode(text, add_special_tokens=False).ids)
    needle, fixed = NEEDLE.format(key=key), None
    fixed = count(needle) + count(QUESTION)
    unit = max(1, count(FILLER))
    k = max(2, (length - fixed) // unit)
    while k > 2 and fixed + k * unit > length:
        k -= 1
    half = k // 2
    return FILLER * half + needle + FILLER * (k - half) + QUESTION


# ---- the run -------------------------------------------------------------------------------------------------------------------

def write_record(data: bytes) -> None:
    """Write the record through the evidence directory's descriptor. The port ran in this directory tree and could have left a link at the
    temporary name, or replaced `evidence` itself with a link, so a plain write would truncate a file outside the run (review of PR 490, round 3).
    The directory must be a real directory at its own place beneath the run, the temporary name is removed first (removing a link removes the
    link), and the new file is created exclusively with no following of links."""
    evidence = RECORD.parent
    # Descend from the run directory one name at a time, each open relative to the one before and with no following of links. A directory
    # swapped for a link at any moment after this point cannot redirect the write, because every descriptor already names the real directory
    # (a realpath comparison followed by a path open leaves a gap between the check and the use; review of PR 490, round 4).
    try:
        dfd = os.open(RUN_DIR, os.O_RDONLY | os.O_DIRECTORY)   # hardening: ok the run directory is the trusted root the descent starts from; every part below it is opened with no following
        for part in evidence.relative_to(RUN_DIR).parts:
            try:
                nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=dfd)
            finally:
                os.close(dfd)
            dfd = nxt
    except OSError as exc:
        raise ServeFailed(f"{evidence.relative_to(RUN_DIR)} is a link or has a linked parent: the record is not written through it ({exc.strerror})")
    try:
        tmp = RECORD.name + ".tmp"
        try:
            os.unlink(tmp, dir_fd=dfd)
        except FileNotFoundError:
            pass
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o644, dir_fd=dfd)
        try:
            view = memoryview(data)
            while view:
                view = view[os.write(fd, view):]
            os.fsync(fd)
        finally:
            os.close(fd)
        os.replace(tmp, RECORD.name, src_dir_fd=dfd, dst_dir_fd=dfd)
    finally:
        os.close(dfd)


def main() -> int:
    t_start = time.monotonic()
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))  # hardening: ok the supervisor wrote this file before the test; no agent works in stage 5, and the gate compares its hash
    chips = [c for c in os.environ.get("TT_VISIBLE_DEVICES", "").split(",") if c]
    if not chips:
        raise ServeFailed("TT_VISIBLE_DEVICES names no chip: this test runs only under the supervisor's lease")
    if sha_file(CORPUS_FILE) != config["corpus_sha256"]:
        raise ServeFailed("corpus.json is not the corpus the config names")
    corpus = json.loads(CORPUS_FILE.read_text(encoding="utf-8"))  # hardening: ok the supervisor wrote this file before the test; no agent works in stage 5, and the gate compares its hash
    pc = load_worker(config["worker_sha256"])
    if not (PORT_DIR / "port_model.py").is_file():
        raise ServeFailed("stages/5/port/port_model.py does not exist: the supervisor copies the port that passed stage 2 or 3 there")
    pc.PORT_DIR = PORT_DIR
    pc.refuse_links()
    snapshot = Path(config["model_snapshot"])
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(snapshot / "tokenizer.json"))
    render, chat_template, support = renderer_and_thinking(snapshot)
    cache_dir, cache_before = kernel_cache()
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    if RECORD.exists() or RECORD.is_symlink():
        RECORD.unlink()

    w = pc.Worker()
    server = engine = None
    try:
        w.ask(op="init", n_chips=len(chips))
        t0 = time.monotonic()
        w.ask(op="load", model_dir=str(snapshot))
        load_s = max(time.monotonic() - t0, 1e-6)
        print(f"loaded in {load_s}s", flush=True)
        engine = Engine(w, int(config["max_len"]), tok, stop_ids(snapshot), render, vocab_size=vocab_bound(snapshot, tok.get_vocab_size(with_added_tokens=True)))
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine))
        port = server.server_address[1]
        if port in FORBIDDEN_PORTS or 8100 <= port <= 8103:
            # serve_forever has not started, so shutdown() below would wait for a loop that never ran: close the socket and clear `server`
            # first (review of PR 490).
            server.server_close()
            server = None
            raise ServeFailed(f"the operating system gave port {port}, which belongs to other work on this machine")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"listening on 127.0.0.1:{port}; replaying the corpus before any client is served", flush=True)

        # Boot: the corpus replay. The server is refused when any position differs.
        replay, total, matched, first_bad = [], 0, 0, None
        for e in corpus["entries"]:
            body = {"prompt": e["prompt_ids"], "max_tokens": len(e["expected_ids"]), "ignore_eos": True}
            if e["kind"] == "teacher-forced":
                body["forced_tokens"] = e["forced_ids"]
            got = post(port, "/v1/completions", body, engine.replay_token)["choices"][0]["token_ids"]
            replay.append({"id": e["id"], "kind": e["kind"], "got_ids": got})
            total += len(e["expected_ids"])
            same = sum(1 for a, b in zip(got, e["expected_ids"]) if a == b) if len(got) == len(e["expected_ids"]) else 0
            matched += same
            if got != e["expected_ids"] and first_bad is None:
                at = next((i for i, (a, b) in enumerate(zip(got, e["expected_ids"])) if a != b), min(len(got), len(e["expected_ids"])))
                first_bad = f"entry {e['id']!r} differs at position {at} ({same} of {len(e['expected_ids'])} equal)"
        if matched != total:
            raise ServeFailed(f"the corpus replay matched {matched} of {total} positions: {first_bad}. The server is refused")
        engine.open = True                                   # every position matched: ordinary requests are served from here
        models = [m.get("id") for m in get(port, "/v1/models").get("data", []) if isinstance(m, dict)]
        ready_line = f"READY {MODEL_ID} on 127.0.0.1:{port} pid {os.getpid()} corpus {matched}/{total} positions"
        print(ready_line, flush=True)
        ready_s = max(time.monotonic() - t_start, load_s)
        _, cache_after = kernel_cache()

        # The passkey runs: the same prompt twice through the chat endpoint.
        runs = []
        plan = passkey_plan(support)
        for item in config["passkeys"]:
            text = passkey_prompt(tok, item["length"], item["key"])
            limit = answer_limit(tok, render, text, plan["kwargs"], plan_max=plan["max_tokens"], max_len=int(config["max_len"]))
            body = {"messages": [{"role": "user", "content": text}], "max_tokens": limit}
            if plan["kwargs"]:
                body["chat_template_kwargs"] = plan["kwargs"]
            first, second = post(port, "/v1/chat/completions", body), post(port, "/v1/chat/completions", body)
            runs.append({"length": item["length"], "key": item["key"], "prompt_tokens": first["usage"]["prompt_tokens"], "prompt_chars": len(text),
                         "prompt_sha256": sha_bytes(text.encode()), "answer_text": first["choices"][0]["message"]["content"], "answer_raw": first["choices"][0].get("raw_text", first["choices"][0]["message"]["content"]),
                         "finish_reason": first["choices"][0].get("finish_reason"), "limit": limit,
                         "ids_first": first["choices"][0]["token_ids"], "ids_second": second["choices"][0]["token_ids"]})
            print(f"passkey length {item['length']}: {first['usage']['prompt_tokens']} prompt tokens", flush=True)

        # The canary: one fixed prompt, a different request, the same prompt again.
        free = next(e for e in corpus["entries"] if e["kind"] == "free-run")
        n = min(int(config["canary_tokens"]), len(free["expected_ids"]))
        canary_body = {"prompt": free["prompt_ids"], "max_tokens": n, "ignore_eos": True}
        other = {"prompt": list(reversed(free["prompt_ids"])) + [free["prompt_ids"][0]], "max_tokens": n, "ignore_eos": True}
        c_first = post(port, "/v1/completions", canary_body)["choices"][0]["token_ids"]
        c_between = post(port, "/v1/completions", other)["choices"][0]["token_ids"]
        c_second = post(port, "/v1/completions", canary_body)["choices"][0]["token_ids"]
        # Nothing but this script is meant to be talking to the server now. A client that is mid-completion finishes first, and a later one is
        # refused, so the worker's pipes are used by one caller when `close` goes out (review of PR 490, round 5).
        if not engine.quiesce(CLOSE_WAIT_S):
            raise ServeFailed(f"a request was still running {CLOSE_WAIT_S} seconds after the checks ended; the worker was not closed under it")
        w.ask(op="close")
    finally:
        if engine is not None:
            engine.quiesce(CLOSE_WAIT_S)        # the failure paths too: stop no worker that a handler is still using
        if server is not None:
            server.shutdown()
            server.server_close()
        w.stop()
    pc.refuse_links()
    if pc.FORGED.is_set():
        raise ServeFailed(f"the port printed a line that mentions {pc.MARK}: only this file prints that line, so the run is refused")

    record = {"schema": SCHEMA, "template_sha256": sha_file(__file__), "worker_sha256": sha_file(WORKER_FILE),
              "config": {"path": str(CONFIG_FILE.relative_to(RUN_DIR)), "sha256": sha_file(CONFIG_FILE)},
              "corpus": {"path": str(CORPUS_FILE.relative_to(RUN_DIR)), "sha256": sha_file(CORPUS_FILE)},
              "source_stage": config["source_stage"], "chips": chips,
              "boot": {"ready_line": ready_line, "load_s": load_s, "ready_s": ready_s, "endpoint": f"127.0.0.1:{port}", "models": models,
                       "chat_template": chat_template,
                       "kernel_cache": {"dir": cache_dir, "entries_before": cache_before, "entries_after": cache_after, "warm": cache_before > 0,
                                        "basis": "files in the directory before the worker started; the runtime's own hit counts are not read"}},
              "replay": {"entries": replay},
              "passkey": {"runs": runs, "thinking": {"template_thinks": support["template_thinks"], "switch": support["switch"], "off": plan["off"],
                                                    "plan_max_tokens": plan["max_tokens"]}},
              "canary": {"prompt_ids": free["prompt_ids"], "between_prompt_ids": other["prompt"], "first_ids": c_first, "between_ids": c_between,
                         "second_ids": c_second},
              "port_sources": [{"path": str(f.relative_to(RUN_DIR)), "sha256": sha_file(f), "role": "model"} for f in sorted(PORT_DIR.rglob("*.py"))]}
    data = (json.dumps(record, indent=2) + "\n").encode("utf-8")
    write_record(data)
    print(f"corpus {matched}/{total}; passkey lengths {[r['length'] for r in runs]}; canary {'same' if c_first == c_second else 'DIFFERENT'}", flush=True)
    print("PORT_CHECK_SHA256", sha_bytes(data), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001  ServeFailed, and the port worker's failures (port_check.PortFailed), print one line and end with code 2
        print(f"FAILED: {exc if isinstance(exc, ServeFailed) else f'{type(exc).__name__}: {exc}'}", flush=True)
        sys.exit(2)
