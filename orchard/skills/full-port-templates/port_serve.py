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
FILLER = "The grass is green. The sky is blue. The sun is warm. Here we go again. "
NEEDLE = "The pass key is {key}. Remember it. "
QUESTION = "What is the pass key? The pass key is"
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

class Engine:
    """The port, one request at a time. `run` returns the ids the port chose for a prompt of ids."""

    def __init__(self, worker, max_len: int, tokenizer, stop_ids, render):
        self.w, self.max_len, self.tok, self.stop_ids, self.render = worker, max_len, tokenizer, set(stop_ids), render
        self.lock = threading.Lock()

    def run(self, prompt, max_tokens, *, forced=None, ignore_eos=False):
        """(ids, finish): at most `max_tokens` ids. With `forced` (a list of max_tokens - 1 ids) each step after the first is fed the forced id in
        place of the port's own. The position of a decoded token is its index in the whole sequence, the prompt included."""
        if not prompt:
            raise ValueError("the prompt has no tokens")
        if len(prompt) + max_tokens > self.max_len:
            raise ValueError(f"the prompt is {len(prompt)} tokens and {max_tokens} more were asked; the server's context is {self.max_len}")
        if forced is not None and len(forced) != max_tokens - 1:
            raise ValueError("forced_tokens must hold max_tokens - 1 ids")
        with self.lock:
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
                    prompt = engine.tok.encode(engine.render(messages), add_special_tokens=False).ids
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
            except (ValueError, KeyError, TypeError) as exc:
                return self._json(400, {"error": {"message": str(exc)[:300]}})
            except Exception as exc:  # noqa: BLE001  a failure in the port must reach the client and the log
                sys.stderr.write(f"serve: generation failed: {type(exc).__name__}\n")
                return self._json(500, {"error": {"message": f"generation failed: {type(exc).__name__}"}})
            text = engine.tok.decode(ids, skip_special_tokens=True)
            choice = {"index": 0, "finish_reason": finish, "token_ids": ids}
            if path == "/v1/chat/completions":
                choice["message"] = {"role": "assistant", "content": text}
            else:
                choice["text"] = text
            self._json(200, {"id": f"cmpl-{int(time.time() * 1000)}", "model": MODEL_ID, "choices": [choice],
                             "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(ids), "total_tokens": len(prompt) + len(ids)}})

        def log_message(self, fmt, *args):
            sys.stderr.write("serve: " + fmt % args + "\n")
    return Handler


def post(port: int, path: str, body: dict) -> dict:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=CLIENT_TIMEOUT_S)
    try:
        conn.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
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


def chat_renderer(snapshot: Path):
    """(render, name): `render(messages)` gives the prompt text. The snapshot's own template when jinja2 can run it, else plain lines."""
    cfg = read_json(snapshot / "tokenizer_config.json")
    template = cfg.get("chat_template")
    if isinstance(template, str) and template:
        try:
            from jinja2.sandbox import ImmutableSandboxedEnvironment
            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

            def raise_exception(message):
                raise ValueError(message)
            tpl = env.from_string(template)
            tokens = {k: (v.get("content") if isinstance(v, dict) else v) for k, v in cfg.items() if k.endswith("_token")}

            def render(messages):
                return tpl.render(messages=messages, add_generation_prompt=True, raise_exception=raise_exception, **tokens)
            render([{"role": "user", "content": "x"}])            # a template that cannot render a plain turn is not used
            return render, "tokenizer_config"
        except Exception:  # noqa: BLE001  any template or jinja2 problem means the plain form
            pass
    return (lambda messages: "".join(f"{m.get('role', 'user')}: {m['content']}\n" for m in messages) + "assistant:"), "plain"


def stop_ids(snapshot: Path) -> list:
    out = []
    for doc in (read_json(snapshot / "generation_config.json"), read_json(snapshot / "config.json")):
        value = doc.get("eos_token_id")
        out += [v for v in (value if isinstance(value, list) else [value]) if type(v) is int]
    return sorted(set(out))


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
    render, chat_template = chat_renderer(snapshot)
    cache_dir, cache_before = kernel_cache()
    RECORD.parent.mkdir(parents=True, exist_ok=True)
    if RECORD.exists() or RECORD.is_symlink():
        RECORD.unlink()

    w = pc.Worker()
    server = None
    try:
        w.ask(op="init", n_chips=len(chips))
        t0 = time.monotonic()
        w.ask(op="load", model_dir=str(snapshot))
        load_s = max(time.monotonic() - t0, 1e-6)
        print(f"loaded in {load_s}s", flush=True)
        engine = Engine(w, int(config["max_len"]), tok, stop_ids(snapshot), render)
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(engine))
        port = server.server_address[1]
        if port in FORBIDDEN_PORTS or 8100 <= port <= 8103:
            raise ServeFailed(f"the operating system gave port {port}, which belongs to other work on this machine")
        threading.Thread(target=server.serve_forever, daemon=True).start()
        print(f"listening on 127.0.0.1:{port}; replaying the corpus before any client is served", flush=True)

        # Boot: the corpus replay. The server is refused when any position differs.
        replay, total, matched, first_bad = [], 0, 0, None
        for e in corpus["entries"]:
            body = {"prompt": e["prompt_ids"], "max_tokens": len(e["expected_ids"]), "ignore_eos": True}
            if e["kind"] == "teacher-forced":
                body["forced_tokens"] = e["forced_ids"]
            got = post(port, "/v1/completions", body)["choices"][0]["token_ids"]
            replay.append({"id": e["id"], "kind": e["kind"], "got_ids": got})
            total += len(e["expected_ids"])
            same = sum(1 for a, b in zip(got, e["expected_ids"]) if a == b) if len(got) == len(e["expected_ids"]) else 0
            matched += same
            if got != e["expected_ids"] and first_bad is None:
                at = next((i for i, (a, b) in enumerate(zip(got, e["expected_ids"])) if a != b), min(len(got), len(e["expected_ids"])))
                first_bad = f"entry {e['id']!r} differs at position {at} ({same} of {len(e['expected_ids'])} equal)"
        if matched != total:
            raise ServeFailed(f"the corpus replay matched {matched} of {total} positions: {first_bad}. The server is refused")
        models = [m.get("id") for m in get(port, "/v1/models").get("data", []) if isinstance(m, dict)]
        ready_line = f"READY {MODEL_ID} on 127.0.0.1:{port} pid {os.getpid()} corpus {matched}/{total} positions"
        print(ready_line, flush=True)
        ready_s = max(time.monotonic() - t_start, load_s)
        _, cache_after = kernel_cache()

        # The passkey runs: the same prompt twice through the chat endpoint.
        runs = []
        for item in config["passkeys"]:
            text = passkey_prompt(tok, item["length"], item["key"])
            body = {"messages": [{"role": "user", "content": text}], "max_tokens": 12}
            first, second = post(port, "/v1/chat/completions", body), post(port, "/v1/chat/completions", body)
            runs.append({"length": item["length"], "key": item["key"], "prompt_tokens": first["usage"]["prompt_tokens"], "prompt_chars": len(text),
                         "prompt_sha256": sha_bytes(text.encode()), "answer_text": first["choices"][0]["message"]["content"],
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
        w.ask(op="close")
    finally:
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
              "passkey": {"runs": runs},
              "canary": {"prompt_ids": free["prompt_ids"], "between_prompt_ids": other["prompt"], "first_ids": c_first, "between_ids": c_between,
                         "second_ids": c_second},
              "port_sources": [{"path": str(f.relative_to(RUN_DIR)), "sha256": sha_file(f), "role": "model"} for f in sorted(PORT_DIR.rglob("*.py"))]}
    data = (json.dumps(record, indent=2) + "\n").encode("utf-8")
    tmp = RECORD.with_name(RECORD.name + ".tmp")
    tmp.write_bytes(data)  # hardening: ok the supervisor wrote this file before the test; no agent works in stage 5, and the gate compares its hash
    os.replace(tmp, RECORD)
    print(f"corpus {matched}/{total}; passkey lengths {[r['length'] for r in runs]}; canary {'same' if c_first == c_second else 'DIFFERENT'}", flush=True)
    print("PORT_CHECK_SHA256", sha_bytes(data), flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001  ServeFailed, and the port worker's failures (port_check.PortFailed), print one line and end with code 2
        print(f"FAILED: {exc if isinstance(exc, ServeFailed) else f'{type(exc).__name__}: {exc}'}", flush=True)
        sys.exit(2)
