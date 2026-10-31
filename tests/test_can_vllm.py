"""The canning step for kind `vllm` (orchard/can.py): fixed serving arguments, the weights wiring through a model directory, the
bundled wheels, the vLLM boot check. The first can of this kind is Qwen3.5-35B-A3B (config/can.qwen35.toml).

Staging tests run the installed `tt-model package-thin --out` on a toy spec (it opens no device and reaches no network) and are skipped when
`tt-model` is not installed. The boot check runs against a stand-in server that talks like vLLM, so no chip is used."""
import json
import os
import shutil
import sys
import textwrap
from pathlib import Path

import pytest

from orchard import can
from orchard.package import assert_no_publish

ROOT = Path(__file__).resolve().parent.parent
needs_tt_model = pytest.mark.skipif(shutil.which("tt-model") is None, reason="tt-model is not installed")
REV = "59d61f3ce65a6d9863b86d2e96597125219dc754"

SPEC = textwrap.dedent('''
    [can]
    name = "toy-moe-p300"
    kind = "vllm"
    model_py = "toy_model.py"
    metadata = "toy_metadata.json"
    readme = "README.src.md"
    mesh = "P300"
    device_count = 2
    port = 20161
    max_num_seqs = 1
    block_size = 64
    max_model_len = 4096
    mesh_fabric = "FABRIC_1D"
    trace_region_bytes = 805306368
    vllm_version = "0.26.0"
    plugin_wheel = "wheels/vllm_tt_plugin-0.1.0-py3-none-any.whl"
    models_wheels = ["wheels/ttnn-0.79.0.dev1+toy-cp312-cp312-linux_x86_64.whl", "wheels/tt_metal_models-0.79.0.dev1+toy-py3-none-any.whl"]
    extra_args = ["--enable-prefix-caching", "--enable-chunked-prefill", "--max-num-batched-tokens", "2048"]
    tt_config = { sample_on_device_mode = "all", l1_small_size = 24576 }
    requirements = ["ttnn==0.79.0.dev1+toy", "tt-metal-models==0.79.0.dev1+toy", "transformers==5.12.1"]
    verify = []
    [weights]
    repo = "someone/toy-moe"
    revision = "%s"
    [env]
    TOY_FLAG = "1"
    [model_dir]
    base_config = ["config.json"]
    [check]
    expect = "Paris"
    min_decode_tok_s = 20.0
    request_extra = { chat_template_kwargs = { enable_thinking = false } }
''' % REV)


@pytest.fixture
def base(tmp_path):
    b = tmp_path / "base"
    (b / "wheels").mkdir(parents=True)
    (b / "toy_model.py").write_text("class Toy: pass\n")
    (b / "toy_metadata.json").write_text(json.dumps({"arch": "ToyForCausalLM", "main_class": "toy_model:Toy"}))
    (b / "config.json").write_text(json.dumps({"architectures": ["ToyForCausalLM"], "model_type": "toy"}))
    (b / "README.src.md").write_text("# toy\n")
    for name in ("vllm_tt_plugin-0.1.0-py3-none-any.whl", "ttnn-0.79.0.dev1+toy-cp312-cp312-linux_x86_64.whl",
                 "tt_metal_models-0.79.0.dev1+toy-py3-none-any.whl"):
        (b / "wheels" / name).write_bytes(b"PK toy wheel " + name.encode())
    (b / "can.toml").write_text(SPEC)
    return b


def edit(base, old, new):
    p = base / "can.toml"
    text = p.read_text()
    assert old in text, old
    p.write_text(text.replace(old, new))
    return can.load_spec(p, base=base)


@pytest.fixture
def spec(base):
    return can.load_spec(base / "can.toml", base=base)


# ---- the spec

def test_the_serving_fields_load(spec):
    assert spec.extra_args == ("--enable-prefix-caching", "--enable-chunked-prefill", "--max-num-batched-tokens", "2048")
    assert spec.tt_config == {"sample_on_device_mode": "all", "l1_small_size": 24576}
    assert spec.trace_region_bytes == 805306368
    assert [d for _, d in spec.base_config] == ["config.json"]
    assert spec.check.request_extra == {"chat_template_kwargs": {"enable_thinking": False}}


@pytest.mark.parametrize("old,new,words", [
    ('extra_args = ["--enable-prefix-caching"', 'extra_args = [1, "--enable-prefix-caching"', "extra_args"),
    ('extra_args = ["--enable-prefix-caching"', 'extra_args = ["--port", "9", "--enable-prefix-caching"', "extra_args must not set --port"),
    ('extra_args = ["--enable-prefix-caching"', 'extra_args = ["--model", "x", "--enable-prefix-caching"', "extra_args must not set --model"),
    ('extra_args = ["--enable-prefix-caching"', 'extra_args = ["--max-model-len", "8192", "--enable-prefix-caching"', "extra_args must not set --max_model_len"),
    ('extra_args = ["--enable-prefix-caching"', 'extra_args = ["--max_num_seqs=4", "--enable-prefix-caching"', "extra_args must not set --max_num_seqs"),
    ('extra_args = ["--enable-prefix-caching"', 'extra_args = ["--block-size", "32", "--enable-prefix-caching"', "extra_args must not set --block_size"),
    ('extra_args = ["--enable-prefix-caching"', 'extra_args = ["--max_model_len=1", "--enable-prefix-caching"', "extra_args must not set --max_model_len"),
    ('base_config = ["config.json"]', 'base_config = ["toy_model.py"]', "config.json"),
    ("tt_config = { sample", "tt_config = { fabric_config = \"FABRIC_2D\", sample", "reserved"),
    ("tt_config = { sample", "tt_config = { trace_region_size = 5, sample", "reserved"),
    ('base_config = ["config.json"]', 'base_config = ["nope.json"]', "nope.json"),
    ('base_config = ["config.json"]', "base_config = []", "base_config"),
    ("[model_dir]", "[model_dir]\nsurprise = 1", "surprise"),
    ("trace_region_bytes = 805306368", 'trace_region_bytes = "big"', "trace_region_bytes"),
])
def test_a_bad_vllm_spec_is_refused_with_a_message_that_names_the_fault(base, old, new, words):
    with pytest.raises(can.CanError, match=words):
        edit(base, old, new)


def test_the_serving_fields_are_refused_for_the_dit_kind(base):
    text = (base / "can.toml").read_text()
    text = text.replace('kind = "vllm"', 'kind = "tt-dit-server"\napp = "m:app"').replace('metadata = "toy_metadata.json"\n', "")
    (base / "d.toml").write_text(text)
    with pytest.raises(can.CanError, match="for kind vllm"):
        can.load_spec(base / "d.toml", base=base)


# ---- the package-thin call

def test_the_call_carries_the_serving_arguments(spec, tmp_path):
    argv = can.thin_argv(spec, out=tmp_path / "o", requirements=tmp_path / "r.txt", extra_code=None)
    assert_no_publish(argv)
    extra = [argv[i + 1] for i, a in enumerate(argv) if a == "--extra-arg"]
    assert extra == ["--enable-prefix-caching", "--enable-chunked-prefill", "--max-num-batched-tokens", "2048"]
    assert json.loads(argv[argv.index("--tt-config-json") + 1]) == {"sample_on_device_mode": "all", "l1_small_size": 24576}
    assert argv[argv.index("--trace-region-bytes") + 1] == "805306368" and argv[argv.index("--mesh-fabric") + 1] == "FABRIC_1D"
    assert argv.count("--models-wheel") == 2 and argv[argv.index("--device-count") + 1] == "2"


# ---- the run.sh edits

VLLM_LINE = ('CMD=("$PYBIN" -m vllm.entrypoints.openai.api_server --model Some/Model --max_num_seqs 1 --block_size 64 '
             '--revision ' + "a" * 40 + ' --tokenizer-revision ' + "a" * 40 + ' --max_model_len 4096 --enable-chunked-prefill "$@")')


def test_the_vllm_run_sh_binds_localhost_and_the_spec_port_before_the_operators_arguments():
    out = can.fix_run_sh_vllm("a\n" + VLLM_LINE + "\nb\n", 20161)
    cmd = [ln for ln in out.splitlines() if ln.startswith("CMD=(")][0]
    assert cmd.endswith('--host "${HOST:-127.0.0.1}" --port "${PORT:-20161}" "$@")')
    assert "0.0.0.0" not in out


def test_the_vllm_run_sh_edit_refuses_a_line_it_does_not_recognise():
    with pytest.raises(can.CanError, match="command"):
        can.fix_run_sh_vllm("CMD=(vllm serve)\n", 20161)


# ---- staging

@pytest.fixture
def bundle(spec, tmp_path):
    out = tmp_path / "out" / "toy-moe-p300"
    return can.stage(spec, out), out


@needs_tt_model
def test_a_staged_vllm_bundle_has_the_wiring_the_wheels_and_the_serving_arguments(spec, bundle):
    rec, pkg = bundle
    assert can.gate(pkg, spec) == []
    run_sh = (pkg / "run.sh").read_text()
    cmd = [ln for ln in run_sh.splitlines() if ln.startswith("CMD=(")][0]
    assert '--model "$HERE/model-dir"' in cmd and "--revision" not in cmd
    for arg in ("--enable-prefix-caching", "--max-num-batched-tokens 2048", "--max_num_seqs 1", "--max_model_len 4096"):
        assert arg in cmd, arg
    assert '"sample_on_device_mode": "all"' in cmd and '"l1_small_size": 24576' in cmd and '"trace_region_size": 805306368' in cmd
    assert cmd.endswith('--host "${HOST:-127.0.0.1}" --port "${PORT:-20161}" "$@")')
    assert 'export HF_MODEL="$HERE/model-dir"' in run_sh and 'export MODEL_WEIGHTS_DIR="$HERE/model-dir"' in run_sh
    assert run_sh.index("prepare_model_dir.py") < run_sh.index('exec "${CMD[@]}"')
    assert (pkg / "base_config" / "config.json").read_text() == (spec.base_config[0][0]).read_text()
    assert os.access(pkg / "prepare_model_dir.py", os.X_OK)
    assert (pkg / "vllm_models" / "toy-moe-p300" / "vllm_metadata.json").is_file()
    assert sorted(p.name for p in (pkg / "wheels").iterdir()) == sorted(p.name for p in (spec.plugin_wheel, *spec.models_wheels))
    assert not (pkg / "model-dir").exists()
    assert rec["kind"] == "vllm"
    assert "PyPI" not in (pkg / "requirements.txt").read_text() and "wheels/" in (pkg / "requirements.txt").read_text()


# ---- the gate

def rewrite_sums(pkg):
    can.write_sums(pkg)


def edit_file(pkg, name, old, new):
    p = pkg / name
    text = p.read_text()
    assert old in text, (name, old)
    p.write_text(text.replace(old, new))
    rewrite_sums(pkg)


def drop(pkg, name):
    (pkg / name).unlink()
    rewrite_sums(pkg)


RUN_CMD_HEAD = '--model "$HERE/model-dir"'


@needs_tt_model
@pytest.mark.parametrize("what,change,words", [
    ("weights not wired to the model directory", lambda p: edit_file(p, "run.sh", 'export MODEL_WEIGHTS_DIR="$HERE/model-dir"', 'export MODEL_WEIGHTS_DIR="$resolved_hf_model"'), "MODEL_WEIGHTS_DIR"),
    ("a model id passed to vLLM", lambda p: edit_file(p, "run.sh", RUN_CMD_HEAD, "--model someone/toy-moe"), "--model"),
    ("a revision left on the command", lambda p: edit_file(p, "run.sh", " --max_model_len", " --revision " + REV + " --max_model_len"), "--revision"),
    ("no prepare step", lambda p: edit_file(p, "run.sh", '"$PYBIN" "$HERE/prepare_model_dir.py"\n', ""), "prepare_model_dir.py"),
    ("a missing prepare script", lambda p: drop(p, "prepare_model_dir.py"), "prepare_model_dir.py is missing"),
    ("a base config that is not the spec's", lambda p: edit_file(p, "base_config/config.json", "toy", "other"), "base_config/config.json"),
    ("a missing base config", lambda p: drop(p, "base_config/config.json"), "base_config/config.json is missing"),
    ("a lost serving argument", lambda p: edit_file(p, "run.sh", " --enable-prefix-caching", ""), "--enable-prefix-caching"),
    ("a changed serving limit", lambda p: edit_file(p, "run.sh", "--max_model_len 4096", "--max_model_len 8192"), "max_model_len"),
    ("a limit passed twice", lambda p: edit_file(p, "run.sh", " --max_model_len 4096", " --max_model_len 4096 --max_model_len 9999"), "max_model_len is passed 2 times"),
    ("a limit set again under another spelling", lambda p: edit_file(p, "run.sh", " --enable-prefix-caching", " --max-model-len 99 --enable-prefix-caching"), "max_model_len is passed 2 times"),
    ("every interface", lambda p: edit_file(p, "run.sh", '--host "${HOST:-127.0.0.1}" ', ""), "127.0.0.1"),
    ("the wrong default port", lambda p: edit_file(p, "run.sh", "${PORT:-20161}", "${PORT:-8000}"), "port 20161"),
    ("a changed additional-config key", lambda p: edit_file(p, "run.sh", '"l1_small_size": 24576', '"l1_small_size": 1'), "additional-config"),
    ("a lost trace region", lambda p: edit_file(p, "run.sh", '"trace_region_size": 805306368', '"trace_region_size": 1'), "trace_region_size"),
    ("a lost environment setting", lambda p: edit_file(p, "run.sh", 'export TOY_FLAG="1"', ""), "TOY_FLAG"),
    ("a pin that names another ttnn than the shipped wheel", lambda p: edit_file(p, "requirements.txt", "ttnn==0.79.0.dev1+toy", "ttnn==0.80.0"), "ttnn==0.80.0"),
    ("a changed metadata file", lambda p: edit_file(p, "vllm_models/toy-moe-p300/vllm_metadata.json", "toy_model:Toy", "toy_model:Other"), "vllm_metadata.json"),
    ("a changed wheel", lambda p: (p / "wheels" / "vllm_tt_plugin-0.1.0-py3-none-any.whl").write_bytes(b"other") or rewrite_sums(p), "byte-identical"),
])
def test_the_gate_reports_each_vllm_fault(spec, bundle, what, change, words):
    _, pkg = bundle
    change(pkg)
    problems = can.gate(pkg, spec)
    assert any(words in p for p in problems), f"{what}: the gate said {problems}"


# ---- the boot check against a stand-in that talks like vLLM

SERVER = textwrap.dedent('''
    import json, os, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    SEEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "requests.jsonl")
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            if self.path == "/health":
                self._send(200, b"", "text/plain")
            else:
                self._send(200, {"object": "list", "data": [{"id": "served-name"}]})
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n))
            open(SEEN, "a").write(json.dumps(req) + "\\n")
            if req.get("stream"):
                self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.send_header("Connection", "close"); self.end_headers()
                count = int(req.get("max_tokens", 4))
                for i in range(count):
                    self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": "w"}}]}).encode() + b"\\n\\n"); self.wfile.flush()
                    time.sleep(0.01)
                self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": {"completion_tokens": count * int(os.environ.get("TOY_TOKENS_PER_CHUNK", "1"))}}).encode() + b"\\n\\n")
                self.wfile.write(b"data: [DONE]\\n\\n"); self.wfile.flush()
                self.close_connection = True
            else:
                self._send(200, {"choices": [{"message": {"content": "Paris"}}], "usage": {"completion_tokens": 1}})
        def log_message(self, *a): pass
    print("Application startup complete.", "listening on", os.environ["PORT"], flush=True)
    HTTPServer(("127.0.0.1", int(os.environ["PORT"])), H).serve_forever()
''')


def fake_bundle(tmp_path):
    b = tmp_path / "fake-bundle"
    b.mkdir()
    (b / "server.py").write_text(SERVER)
    (b / "install.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (b / "run.sh").write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" "$(dirname "$0")/server.py"\n')
    return b


def seen(tmp_path):
    return [json.loads(line) for line in (tmp_path / "work" / "bundle" / "requests.jsonl").read_text().splitlines()]


def test_the_boot_check_sends_the_served_model_name_and_the_request_extras(spec, tmp_path):
    ev = can.boot_check(fake_bundle(tmp_path), tmp_path / "work", spec, install=False, health_timeout=30)
    assert ev["passed"], ev
    for req in seen(tmp_path):
        assert req["model"] == "served-name"
        assert req["chat_template_kwargs"] == {"enable_thinking": False}


def test_the_boot_check_measures_the_decode_rate_from_a_stream_when_the_server_reports_no_timings(spec, tmp_path):
    ev = can.boot_check(fake_bundle(tmp_path), tmp_path / "work", spec, install=False, health_timeout=30)
    assert ev["passed"], ev
    rate = ev["checks"]["decode_rate"]
    assert rate["ok"], rate
    assert ev["rate_timings"]["source"] == "stream" and ev["rate_timings"]["completion_tokens"] >= 32
    # the stand-in sleeps 10 ms per token, so the measured rate is at most about 100 tokens per second and well above zero
    assert 5 < ev["rate_timings"]["decode_tok_s"] <= 110


def test_the_boot_check_fails_a_rate_below_the_floor(spec, tmp_path, base):
    slow = edit(base, "min_decode_tok_s = 20.0", "min_decode_tok_s = 5000.0")
    ev = can.boot_check(fake_bundle(tmp_path), tmp_path / "work", slow, install=False, health_timeout=30)
    assert not ev["passed"] and not ev["checks"]["decode_rate"]["ok"]


def test_boot_serves_on_the_port_the_command_line_names(tmp_path, base):
    import subprocess
    port = can.free_port()
    bundle = fake_bundle(tmp_path)
    r = subprocess.run([sys.executable, str(ROOT / "bin" / "tt-orchard"), "can", "boot", str(bundle), str(base / "can.toml"), "--base", str(base),
                        "--work", str(tmp_path / "w"), "--port", str(port), "--health-timeout", "60"],
                       capture_output=True, text=True, timeout=300, env={**os.environ, "PORT": "1"})
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"listening on {port}" in (tmp_path / "w" / "logs" / "server.log").read_text()


def test_the_rate_helper_needs_a_stream_that_has_a_first_and_a_last_token():
    assert can.stream_rate([]) is None
    assert can.stream_rate([(1.0, 1)]) is None
    assert can.stream_rate([(1.0, 1), (2.0, 11)]) == pytest.approx(10.0)


def test_the_boot_check_counts_two_tokens_a_chunk_when_the_server_reports_them(spec, tmp_path, monkeypatch):
    monkeypatch.setenv("TOY_TOKENS_PER_CHUNK", "2")
    ev = can.boot_check(fake_bundle(tmp_path), tmp_path / "work", spec, install=False, health_timeout=30)
    rt = ev["rate_timings"]
    assert rt["completion_tokens"] == 128 and rt["streamed_chunks"] == 64
    assert rt["decode_tok_s"] == pytest.approx(127 / rt["stream_seconds"], rel=0.02)


def test_the_stream_rate_uses_the_reported_token_count_when_a_chunk_holds_more_than_one_token():
    # 3 chunks over 2 seconds, but the server says it generated 21 tokens: (21 - 1) tokens after the first one, over 2 s
    events = [(1.0, 1), (2.0, 2), (3.0, 3)]
    assert can.stream_rate(events, reported=21) == pytest.approx(10.0)
    assert can.stream_rate(events) == pytest.approx(1.0)        # no report: one token a chunk


def test_the_boot_check_records_the_reported_token_count_in_the_rate(spec, tmp_path):
    ev = can.boot_check(fake_bundle(tmp_path), tmp_path / "work", spec, install=False, health_timeout=30)
    assert ev["rate_timings"]["completion_tokens"] == 64 and ev["rate_timings"]["decode_tok_s"] == pytest.approx(
        63 / ev["rate_timings"]["stream_seconds"], rel=0.02)
