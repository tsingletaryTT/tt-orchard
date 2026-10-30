"""The canning step (orchard/can.py): the spec, the package-thin call, the edits, the gate, the boot check and the record.

Most tests stage a real bundle with the installed `tt-model package-thin --out` (it opens no device and reaches no network) from a spec built
in a temporary directory. They are skipped when `tt-model` is not installed. The boot check is tested against a small stand-in server, so no
chip is used and nothing is installed."""
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from orchard import can
from orchard.package import PackageError, assert_no_publish

ROOT = Path(__file__).resolve().parent.parent
HAVE_TT_MODEL = shutil.which("tt-model") is not None
needs_tt_model = pytest.mark.skipif(not HAVE_TT_MODEL, reason="tt-model is not installed")
REV = "d18cca4c121edb87d022116d281ce212c9136f57"

APP = textwrap.dedent('''
    async def app(scope, receive, send):
        pass
''')

SPEC = textwrap.dedent('''
    [can]
    name = "toy-blackhole"
    kind = "tt-dit-server"
    app = "toy_asgi:app"
    model_py = "toy_model.py"
    readme = "README.src.md"
    mesh = "P150"
    device_count = 1
    port = 20123
    requirements = ["ttnn==0.80.0", "numpy==1.26.4", "fastapi==0.136.3", "uvicorn==0.45.0", "pydantic==2.12.5", "pillow==10.4.0"]
    verify = ["import toy_asgi"]
    [weights]
    repo = "someone/toy-model"
    revision = "%s"
    [[extra_code]]
    src = "toy_core.py"
    dst = "toy_core.py"
    [[extra_code]]
    src = "toy_asgi.py"
    dst = "toy_asgi.py"
    [[bundle_file]]
    src = "NOTICE.src"
    dst = "NOTICE"
    [check]
    expect = "Paris"
    min_decode_tok_s = 30.0
    ready_contains = ['"ready": true']
''' % REV)


@pytest.fixture
def base(tmp_path):
    b = tmp_path / "base"
    b.mkdir()
    (b / "toy_asgi.py").write_text(APP)
    (b / "toy_model.py").write_text("from toy_asgi import app  # noqa\n")
    (b / "toy_core.py").write_text("VALUE = 1\n")
    (b / "README.src.md").write_text("# toy\n")
    (b / "NOTICE.src").write_text("notice\n")
    (b / "can.toml").write_text(SPEC)
    return b


def edit_spec(base, old, new):
    p = base / "can.toml"
    text = p.read_text()
    assert old in text
    p.write_text(text.replace(old, new))
    return can.load_spec(p, base=base)


@pytest.fixture
def spec(base):
    return can.load_spec(base / "can.toml", base=base)


# ---- the spec

def test_the_granite_spec_loads_and_every_file_it_names_exists():
    s = can.load_spec(ROOT / "config" / "can.granite.toml", base=ROOT)
    assert (s.kind, s.app, s.mesh, s.device_count, s.revision) == ("tt-dit-server", "granite_asgi:app", "P150", 1, REV)
    assert all(src.is_file() for src, _ in s.extra_code + s.bundle_files)
    assert any(p.startswith("ttnn==") for p in s.requirements)


@pytest.mark.parametrize("old,new,words", [
    ("kind = \"tt-dit-server\"", "kind = \"gradio\"", "kind must be one of"),
    ("app = \"toy_asgi:app\"", "app = \"toy_asgi\"", "module:attribute"),
    ("revision = \"%s\"" % REV, "revision = \"main\"", "40-character"),
    ("port = 20123", "port = 8000", "not usable"),
    ("port = 20123", "port = 8101", "not usable"),
    ("model_py = \"toy_model.py\"", "model_py = \"../toy_model.py\"", "relative path inside"),
    ("model_py = \"toy_model.py\"", "model_py = \"/etc/passwd\"", "relative path inside"),
    ("model_py = \"toy_model.py\"", "model_py = \"missing.py\"", "not a regular file"),
    ("device_count = 1", "device_count = 0", "device_count"),
    ("[weights]", "[surprise]\nx = 1\n[weights]", "unknown tables"),
    ("mesh = \"P150\"", "mesh = \"P150\"\nsurprise = 1", "unknown keys"),
    ('"numpy==1.26.4"', '"numpy>=1.26"', "not an exact pin"),
    ("dst = \"toy_core.py\"", "dst = \"../x.py\"", "no '..'"),
])
def test_a_bad_spec_is_refused_with_a_message_that_names_the_fault(base, old, new, words):
    with pytest.raises(can.CanError, match=words):
        edit_spec(base, old, new)


def test_vllm_kind_needs_metadata_and_dit_kind_refuses_it(base):
    text = (base / "can.toml").read_text()
    (base / "v.toml").write_text(text.replace('kind = "tt-dit-server"', 'kind = "vllm"'))
    with pytest.raises(can.CanError, match="vllm takes no"):
        can.load_spec(base / "v.toml", base=base)
    (base / "v.toml").write_text(text.replace('kind = "tt-dit-server"', 'kind = "vllm"').replace('app = "toy_asgi:app"\n', ""))
    with pytest.raises(can.CanError, match="needs \\[can\\] metadata"):
        can.load_spec(base / "v.toml", base=base)
    (base / "m.json").write_text("{}")
    with pytest.raises(can.CanError, match="serves no vLLM"):
        edit_spec(base, 'app = "toy_asgi:app"', 'app = "toy_asgi:app"\nmetadata = "m.json"')


def test_a_symbolic_link_as_a_source_is_refused(base):
    (base / "link.py").symlink_to(base / "toy_core.py")
    with pytest.raises(can.CanError, match="not a regular file"):
        edit_spec(base, 'src = "toy_core.py"', 'src = "link.py"')


def test_a_spec_that_is_not_toml_or_is_too_big_is_refused(base):
    (base / "bad.toml").write_text("this is = = not toml")
    with pytest.raises(can.CanError, match="cannot read the spec"):
        can.load_spec(base / "bad.toml", base=base)
    (base / "big.toml").write_bytes(b"#" * (can.MAX_SPEC_BYTES + 1))
    with pytest.raises(can.CanError, match="cannot read the spec"):
        can.load_spec(base / "big.toml", base=base)


# ---- the package-thin call

def test_the_package_thin_call_is_accepted_by_stage_7s_publish_guard_and_has_out_and_no_repo_id(spec, tmp_path):
    argv = can.thin_argv(spec, out=tmp_path / "o", requirements=tmp_path / "r.txt", extra_code=tmp_path / "e")
    assert_no_publish(argv)
    assert argv[:2] == ["tt-model", "package-thin"] and "--out" in argv
    assert argv[argv.index("--kind") + 1] == "tt-dit-server" and argv[argv.index("--app") + 1] == "toy_asgi:app" and "--no-vllm" in argv
    assert argv[argv.index("--weights-revision") + 1] == REV


def test_the_guard_still_refuses_a_repo_id_and_the_publish_flags(spec, tmp_path):
    argv = can.thin_argv(spec, out=tmp_path / "o", requirements=tmp_path / "r.txt", extra_code=None)
    for bad in (argv + ["someone/repo"], argv + ["--public"], argv + ["--publish"], [a for a in argv if a != "--out"]):
        with pytest.raises(PackageError):
            assert_no_publish(bad)


def test_stage_runs_only_package_thin_with_out(spec, tmp_path, monkeypatch):
    calls = []

    def fake_run(argv, *, log, timeout, env=None):
        calls.append([str(a) for a in argv])
        return 1
    monkeypatch.setattr(can, "run_logged", fake_run)
    with pytest.raises(can.CanError, match="exited 1"):
        can.stage(spec, tmp_path / "out")
    assert len(calls) == 1 and calls[0][:2] == ["tt-model", "package-thin"] and "--out" in calls[0]
    assert not any(a.count("/") == 1 and not a.startswith("-") and a == spec.repo for a in calls[0][2:] if a != calls[0][calls[0].index("--weights") + 1])


def test_stage_refuses_an_existing_directory(spec, tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(can.CanError, match="already exists"):
        can.stage(spec, tmp_path / "out")


# ---- run.sh

RUN_LINE = 'CMD=("$PYBIN" -m uvicorn --host 0.0.0.0 --port "${PORT:-8000}" --lifespan on a:app "$@")'


def test_fix_run_sh_binds_localhost_and_the_spec_port_and_leaves_overrides_working():
    fixed = can.fix_run_sh(RUN_LINE, 20110)
    assert '--host "${HOST:-127.0.0.1}"' in fixed and '"${PORT:-20110}"' in fixed and "0.0.0.0" not in fixed


def test_fix_run_sh_refuses_a_line_it_does_not_recognise():
    with pytest.raises(can.CanError, match="launcher changed"):
        can.fix_run_sh('CMD=("$PYBIN" -m uvicorn a:app)', 20110)


# ---- the pins

@pytest.mark.parametrize("line,ok", [
    ("ttnn==0.80.0", True), ("torch==2.7.0", True), ("uvicorn[standard]==0.45.0", True), ("numpy==1.26.4 ; python_version >= '3.12'", True),
    ("--extra-index-url https://download.pytorch.org/whl/cpu", True), ("# a comment", True),
    ("numpy", False), ("numpy>=1.26", False), ("numpy~=1.26", False), ("ttnn  # TODO pin", False), ("git+https://x/y.git", False),
])
def test_requirement_lines(line, ok):
    assert (can.requirement_problems(line + "\n") == []) is ok


# ---- a staged bundle and the gate (real tt-model)

@pytest.fixture
def bundle(spec, tmp_path):
    return can.stage(spec, tmp_path / "out" / "toy-blackhole"), tmp_path / "out" / "toy-blackhole"


@needs_tt_model
def test_a_staged_bundle_has_the_v6_layout_and_passes_the_gate(spec, bundle):
    rec, pkg = bundle
    assert can.gate(pkg, spec) == []
    names = {p.relative_to(pkg).as_posix() for p in pkg.rglob("*") if p.is_file()}
    assert {"tt_kernel_manifest.json", "run.sh", "install.sh", "requirements.txt", "README.md", "SHA256SUMS", "toy_model.py", "NOTICE",
            "extra_code/toy_core.py", "extra_code/toy_asgi.py"} <= names
    assert rec["kind"] == "tt-dit-server" and rec["weights"]["revision"] == REV
    assert os.access(pkg / "run.sh", os.X_OK) and os.access(pkg / "install.sh", os.X_OK)
    assert "0.0.0.0" not in (pkg / "run.sh").read_text() and "${PORT:-20123}" in (pkg / "run.sh").read_text()
    assert not (pkg.parent / ".toy-blackhole.can-").exists() and [p for p in pkg.parent.iterdir()] == [pkg]     # the work directory is gone


@needs_tt_model
def test_the_staged_manifest_loads_with_tt_models_own_reader(bundle):
    from tt_kernel.manifest import Manifest
    _, pkg = bundle
    m = Manifest.from_json((pkg / "tt_kernel_manifest.json").read_text())
    assert m.deps.kind == "tt-dit-server" and m.deps.app == "toy_asgi:app" and m.producer.hostname == "redacted"
    assert m.weights.revision == REV


def mutate(pkg: Path, spec, what: str, change, words: str, hostname="somehost"):
    """Apply `change` to a staged bundle and expect the gate to report `words`. Returns the problems."""
    change(pkg)
    problems = can.gate(pkg, spec, hostname=hostname)
    assert any(words in p for p in problems), f"{what}: the gate said {problems}"
    return problems


def rewrite_sums(pkg):
    can.write_sums(pkg)


@needs_tt_model
@pytest.mark.parametrize("what,change,words", [
    ("a weights file", lambda p: (p / "model.safetensors").write_bytes(b"x"), "a weights file"),
    ("a venv", lambda p: (p / "venv").mkdir(), "an installed venv"),
    ("a model directory", lambda p: (p / "model-dir").mkdir(), "a built model directory"),
    ("a cache", lambda p: (p / ".cache").mkdir(), "a runtime cache"),
    ("a home path", lambda p: ((p / "NOTICE").write_text("see /home/someone/x\n"), rewrite_sums(p)), "an absolute home path"),
    ("a host name", lambda p: ((p / "NOTICE").write_text("built on somehost\n"), rewrite_sums(p)), "the hostname"),
    ("a token", lambda p: ((p / "NOTICE").write_text("hf_" + "a" * 34 + "\n"), rewrite_sums(p)), "a Hugging Face token"),
    ("a changed file", lambda p: (p / "NOTICE").write_text("changed\n"), "changed after SHA256SUMS"),
    ("an unlisted file", lambda p: (p / "extra.txt").write_text("x\n"), "does not list extra.txt"),
    ("a lost file", lambda p: (p / "NOTICE").unlink(), "lists NOTICE"),
    ("a run.sh that is not executable", lambda p: (p / "run.sh").chmod(0o644), "run.sh is not executable"),
    ("a run.sh on every interface", lambda p: ((p / "run.sh").write_text((p / "run.sh").read_text().replace('"${HOST:-127.0.0.1}"', "0.0.0.0").replace("--host 0.0.0.0", "--host 0.0.0.0")), rewrite_sums(p)), "binds uvicorn to 0.0.0.0"),
    ("an unpinned requirement", lambda p: ((p / "requirements.txt").write_text((p / "requirements.txt").read_text() + "requests\n"), rewrite_sums(p)), "not an exact pin"),
    ("a TODO pin", lambda p: ((p / "requirements.txt").write_text((p / "requirements.txt").read_text() + "foo==1  # TODO\n"), rewrite_sums(p)), "TODO"),
    ("a missing ASGI module", lambda p: ((p / "extra_code" / "toy_asgi.py").unlink(), rewrite_sums(p)), "ASGI module 'toy_asgi' is not at"),
    ("a verify import of a module at the bundle root", lambda p: ((p / "toy_asgi.py").write_text("app = None\n"), (p / "extra_code" / "toy_asgi.py").unlink(), rewrite_sums(p)), "install.sh puts only extra_code/"),
    ("a file that does not compile", lambda p: ((p / "extra_code" / "toy_core.py").write_text("def (:\n"), rewrite_sums(p)), "does not compile"),
    ("a symbolic link", lambda p: (p / "link").symlink_to(p / "NOTICE"), "a symbolic link"),
])
def test_the_gate_reports_each_fault(spec, bundle, what, change, words):
    _, pkg = bundle
    mutate(pkg, spec, what, change, words)


def edit_manifest(pkg, key_path, value):
    import json
    f = pkg / "tt_kernel_manifest.json"
    m = json.loads(f.read_text())
    d = m
    for k in key_path[:-1]:
        d = d[k]
    d[key_path[-1]] = value
    f.write_text(json.dumps(m, indent=2))
    can.write_sums(pkg)


@needs_tt_model
@pytest.mark.parametrize("path,value,words", [
    (["producer", "hostname"], "somehost", "hostname is not redacted"),
    (["weights", "revision"], "0" * 40, "weights revision"),
    (["weights", "repo_id"], "other/model", "weights repo"),
    (["deps", "kind"], "vllm", "manifest kind"),
    (["deps", "app"], "other:app", "manifest app"),
    (["device_count"], 4, "device_count"),
    (["schema_version"], "5", "not '6'"),
])
def test_the_gate_compares_the_manifest_with_the_spec(spec, bundle, path, value, words):
    _, pkg = bundle
    edit_manifest(pkg, path, value)
    assert any(words in p for p in can.gate(pkg, spec)), can.gate(pkg, spec)


@needs_tt_model
def test_the_gate_catches_a_namespace_outside_the_card(spec, bundle):
    _, pkg = bundle
    (pkg / "NOTICE").write_text("see myorg/thing\n")
    can.write_sums(pkg)
    assert any("namespace" in p for p in can.gate(pkg, spec, namespace="myorg"))


# ---- the boot check, against a stand-in server

SERVER = textwrap.dedent('''
    import json, os, sys
    from http.server import BaseHTTPRequestHandler, HTTPServer
    ANSWER = os.environ.get("TOY_ANSWER", "Paris")
    class H(BaseHTTPRequestHandler):
        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data)
        def do_GET(self):
            self._send(200, {"status": "ok"} if self.path == "/health" else {"object": "list", "data": []})
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0)); req = json.loads(self.rfile.read(n))
            if req.get("stream"):
                self._send(200, b'data: {"x": 1}\\n\\ndata: [DONE]\\n\\n', "text/event-stream")
            else:
                long = "dragon" in req["messages"][0]["content"]      # only a long answer has a decode rate, as in the real server
                self._send(200, {"choices": [{"message": {"content": ANSWER}}], "timings": {"decode_tok_s": 40.0 if long else None, "completion_tokens": 64 if long else 1}})
        def log_message(self, *a): pass
    print(json.dumps({"ready": True}), flush=True)
    HTTPServer(("127.0.0.1", int(os.environ["PORT"])), H).serve_forever()
''')


def fake_bundle(tmp_path, run_sh_body=None, answer="Paris"):
    b = tmp_path / "fake-bundle"
    b.mkdir()
    (b / "server.py").write_text(SERVER)
    (b / "install.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (b / "run.sh").write_text(run_sh_body or f'#!/usr/bin/env bash\nexport TOY_ANSWER="{answer}"\nexec "{sys.executable}" "$(dirname "$0")/server.py"\n')
    return b


def test_the_boot_check_passes_a_healthy_server_and_stops_it(spec, tmp_path):
    ev = can.boot_check(fake_bundle(tmp_path), tmp_path / "work", spec, install=False, health_timeout=30)
    assert ev["passed"], ev
    assert {"health", "models", "chat", "stream", "health_after", 'log contains "ready": true'} <= set(ev["checks"])
    assert ev["rate_timings"]["decode_tok_s"] == 40.0 and ev["timings"]["decode_tok_s"] is None
    assert (tmp_path / "work" / "boot-check.json").is_file()
    assert not Path(f"/proc/{ev['server_pid']}").exists() or Path(f"/proc/{ev['server_pid']}/stat").read_text().split()[2] == "Z"


def test_the_boot_check_fails_a_wrong_answer(spec, tmp_path):
    ev = can.boot_check(fake_bundle(tmp_path, answer="Berlin"), tmp_path / "work", spec, install=False, health_timeout=30)
    assert not ev["passed"] and not ev["checks"]["chat"]["ok"]


def test_the_boot_check_fails_a_server_that_exits_and_reports_why(spec, tmp_path):
    ev = can.boot_check(fake_bundle(tmp_path, run_sh_body="#!/usr/bin/env bash\nexit 3\n"), tmp_path / "work", spec, install=False, health_timeout=30)
    assert not ev["passed"] and "exited with 3" in ev["checks"]["health"]["detail"]


def test_the_boot_check_fails_a_slow_decode_rate(spec, tmp_path, base):
    slow = edit_spec(base, "min_decode_tok_s = 30.0", "min_decode_tok_s = 99.0")
    ev = can.boot_check(fake_bundle(tmp_path), tmp_path / "work", slow, install=False, health_timeout=30)
    assert not ev["passed"] and not ev["checks"]["decode_rate"]["ok"]


def test_the_boot_check_fails_an_install_that_fails_and_never_touches_the_staged_directory(spec, tmp_path):
    b = fake_bundle(tmp_path)
    (b / "install.sh").write_text("#!/usr/bin/env bash\nexit 5\n")
    before = sorted(p.name for p in b.iterdir())
    ev = can.boot_check(b, tmp_path / "work", spec, health_timeout=5)
    assert not ev["passed"] and "exited 5" in ev["checks"]["install"]["detail"]
    assert sorted(p.name for p in b.iterdir()) == before


def test_the_boot_check_installs_into_a_copy_not_the_staged_directory(spec, tmp_path):
    b = fake_bundle(tmp_path)
    (b / "install.sh").write_text('#!/usr/bin/env bash\nmkdir -p "$(dirname "$0")/venv/bin" && : > "$(dirname "$0")/venv/bin/python"\n')
    before = sorted(p.name for p in b.iterdir())
    ev = can.boot_check(b, tmp_path / "work", spec, health_timeout=30)
    assert ev["passed"], ev
    assert sorted(p.name for p in b.iterdir()) == before and (tmp_path / "work" / "bundle" / "venv").is_dir()


# ---- text and the record

def test_publish_commands_are_all_comments_and_name_the_repo(spec):
    text = can.publish_commands(spec, "myorg")
    assert can.publish_problems(text) == [] and "myorg/toy-blackhole" in text
    assert can.publish_problems("hf upload x y\n# fine\n") == ["line 1 would run: hf upload x y"]


def test_record_writes_can_json_and_a_ledger_entry(tmp_path):
    from orchard.ledger import Ledger, read_entries
    run = tmp_path / "run"
    run.mkdir()
    Ledger(run / "ledger.jsonl").close()
    path = can.record(run, {"name": "toy", "kind": "tt-dit-server"}, {"passed": True})
    assert path == run / "stages" / "7" / "can.json"
    last = read_entries(run / "ledger.jsonl")[-1]
    assert (last["event"], last["stage"], last["data"]["what"], last["data"]["booted"]) == ("evidence", 7, "can staged", True)


@needs_tt_model
def test_the_command_line_stages_gates_and_prints_commands(base, tmp_path):
    cmd = [sys.executable, str(ROOT / "bin" / "tt-orchard"), "can"]
    out = tmp_path / "cli" / "toy"
    r = subprocess.run(cmd + ["stage", str(base / "can.toml"), "--base", str(base), "--out", str(out)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr
    r = subprocess.run(cmd + ["gate", str(out), str(base / "can.toml"), "--base", str(base)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "gate: passed" in r.stdout
    (out / "NOTICE").write_text("changed\n")
    r = subprocess.run(cmd + ["gate", str(out), str(base / "can.toml"), "--base", str(base)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 1 and "changed after SHA256SUMS" in r.stdout
    r = subprocess.run(cmd + ["commands", str(base / "can.toml"), "--base", str(base), "--namespace", "myorg"], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and "myorg/toy-blackhole" in r.stdout
    r = subprocess.run(cmd + ["stage", str(base / "can.toml"), "--base", str(base)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 2 and "needs --out" in r.stdout


# ---- kind vllm (the call is checked; no bundle of this kind was built on hardware)

def vllm_spec(base):
    (base / "m.json").write_text('{"main_class": "mod:Cls", "arch": "ArchForCausalLM"}')
    (base / "plugin-1.0-py3-none-any.whl").write_bytes(b"not a real wheel")
    text = (base / "can.toml").read_text()
    text = text.replace('kind = "tt-dit-server"', 'kind = "vllm"').replace('app = "toy_asgi:app"\n', 'metadata = "m.json"\n'
                        'plugin_wheel = "plugin-1.0-py3-none-any.whl"\nvllm_version = "0.25.1"\nmax_num_seqs = 1\nmax_model_len = 4096\n')
    (base / "v.toml").write_text(text)
    return can.load_spec(base / "v.toml", base=base)


def test_a_vllm_can_passes_the_plugin_wheel_metadata_and_serving_limits(base, tmp_path):
    argv = can.thin_argv(vllm_spec(base), out=tmp_path / "o", requirements=tmp_path / "r.txt", extra_code=None)
    assert_no_publish(argv)
    assert argv[argv.index("--kind") + 1] == "vllm" and "--app" not in argv and "--no-vllm" not in argv
    assert argv[argv.index("--plugin-wheel") + 1].endswith("plugin-1.0-py3-none-any.whl")
    assert argv[argv.index("--vllm-version") + 1] == "0.25.1" and argv[argv.index("--max-model-len") + 1] == "4096"


def test_wheel_options_are_refused_for_the_dit_kind(base):
    (base / "w.whl").write_bytes(b"x")
    with pytest.raises(can.CanError, match="are for kind vllm"):
        edit_spec(base, 'port = 20123', 'port = 20123\nplugin_wheel = "w.whl"')


# ---- review of PR 303

def test_a_directory_link_inside_base_that_points_outside_is_refused(base, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.py").write_text("SECRET = 1\n")
    (base / "dl").symlink_to(outside, target_is_directory=True)
    with pytest.raises(can.CanError, match="outside|not a regular file"):
        edit_spec(base, 'src = "toy_core.py"', 'src = "dl/secret.py"')


@pytest.mark.parametrize("line", ["numpy==1.*", "numpy==1.26.*", "numpy===*"])
def test_a_wildcard_is_not_an_exact_pin(line):
    assert can.requirement_problems(line + "\n")


@needs_tt_model
def test_a_dotted_asgi_module_is_looked_for_as_a_path(spec, bundle):
    _, pkg = bundle
    dotted = can.Spec(**{**spec.__dict__, "app": "server.api:app"})
    assert any("ASGI module" in p for p in can.gate(pkg, dotted))
    (pkg / "extra_code" / "server").mkdir()
    (pkg / "extra_code" / "server" / "api.py").write_text("app = None\n")
    can.write_sums(pkg)
    assert not any("ASGI module" in p for p in can.gate(pkg, dotted))
    (pkg / "extra_code" / "server" / "api.py").unlink()
    (pkg / "extra_code" / "server" / "__init__.py").write_text("")   # a package's __init__ holds `server`, not `server.api`
    can.write_sums(pkg)
    assert any("ASGI module" in p for p in can.gate(pkg, dotted))


def test_a_relative_work_directory_works(spec, tmp_path, monkeypatch):
    b = fake_bundle(tmp_path)
    monkeypatch.chdir(tmp_path)
    ev = can.boot_check(b, "rel-work", spec, install=False, health_timeout=30)
    assert ev["passed"], ev


def test_a_reused_work_directory_with_another_bundle_is_refused(spec, tmp_path):
    first = fake_bundle(tmp_path)
    assert can.boot_check(first, tmp_path / "work", spec, install=False, health_timeout=30)["passed"]
    other = tmp_path / "other"
    shutil.copytree(first, other)
    (other / "run.sh").write_text("#!/usr/bin/env bash\nexit 3\n")
    ev = can.boot_check(other, tmp_path / "work", spec, install=False, health_timeout=30)
    assert not ev["passed"] and not ev["checks"]["bundle_copy"]["ok"]


def test_an_old_log_does_not_satisfy_ready_contains(spec, tmp_path):
    work = tmp_path / "work"
    (work / "logs").mkdir(parents=True)
    (work / "logs" / "server.log").write_text('{"ready": true}\n')
    b = fake_bundle(tmp_path)
    (b / "server.py").write_text(SERVER.replace('print(json.dumps({"ready": True}), flush=True)', "pass"))
    ev = can.boot_check(b, work, spec, install=False, health_timeout=30)
    assert not ev["passed"] and not ev["checks"]['log contains "ready": true']["ok"]


def test_a_transport_failure_after_readiness_writes_failed_evidence(spec, tmp_path, monkeypatch):
    work = tmp_path / "work"
    work.mkdir()
    (work / "boot-check.json").write_text('{"passed": true}')        # an earlier successful run
    real = can._http

    def flaky(method, url, body=None, *, timeout=30.0):
        if method == "POST":
            raise TimeoutError("timed out")
        return real(method, url, body, timeout=timeout)
    monkeypatch.setattr(can, "_http", flaky)
    ev = can.boot_check(fake_bundle(tmp_path), work, spec, install=False, health_timeout=30)
    assert not ev["passed"] and "timed out" in str(ev["checks"]["transport"])
    assert '"passed": false' in (work / "boot-check.json").read_text()
