"""The two scripts the weights-swap-check skill copies: prepare_swap.py and serve_and_compare.py.

Everything here runs on fakes in tmp directories. The bundle's run.sh is replaced by one that
execs tests/fake_swap_server.py, a stdlib HTTP server that opens no device. Each test that starts
a server kills its process group in a fixture finalizer, so a failing test leaves nothing behind.
"""
import importlib.util
import json
import os
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

if importlib.util.find_spec("tokenizers") is None:
    pytest.skip("SKIPPED: the `tokenizers` package is not importable, so the weights-swap template "
                "tests did not run. serve_and_compare.py needs it. These templates are untested "
                "on this interpreter.", allow_module_level=True)

from tokenizers import Tokenizer  # noqa: E402
from tokenizers.models import WordLevel  # noqa: E402
from tokenizers.pre_tokenizers import Whitespace  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "orchard" / "skills" / "weights-swap-templates"
FAKE_SERVER = Path(__file__).resolve().parent / "fake_swap_server.py"
NEAREST = "Qwen/Qwen3.8-27B"
REV = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
TOK_REV = "abcdef0123456789abcdef0123456789abcdef01"

BUNDLE_RUN_SH = f"""#!/usr/bin/env bash
# Serve this model on TT hardware.
set -euo pipefail
HERE="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
VENV="${{VENV:-$HERE/venv}}"
PYBIN="$VENV/bin/python"
export HF_MODEL="${{HF_MODEL:-{NEAREST}}}"
CMD=("$PYBIN" -m vllm.entrypoints.openai.api_server --model "{NEAREST}" --max_num_seqs 4 --revision {REV} --tokenizer-revision {TOK_REV} --max_model_len 262144 "$@")
exec "${{CMD[@]}}"
"""

# ---- prepare_swap.py -----------------------------------------------------------------------------


def make_snapshot(root: Path, files: dict) -> Path:
    """An HF-cache-shaped snapshot: each file is a symlink into a blobs/ directory."""
    blobs, snap = root / "blobs", root / "snapshots" / ("0" * 40)
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    for name, content in files.items():
        blob = blobs / f"blob-{name}"
        blob.write_text(content)
        (snap / name).symlink_to(os.path.relpath(blob, snap))
    return snap


LINKED = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
          "model.safetensors.index.json", "model-00001-of-00002.safetensors",
          "model-00002-of-00002.safetensors", "model-mtp.safetensors"]


@pytest.fixture
def prep(tmp_path):
    """A stage dir with prepare_swap.py copied in, a fake bundle and two fake snapshots."""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "run.sh").write_text(BUNDLE_RUN_SH)
    base = make_snapshot(tmp_path / "base", {"config.json": '{"base": true}',
                                             "preprocessor_config.json": '{"pre": 1}'})
    new = make_snapshot(tmp_path / "new", {"config.json": '{"new": true}',
                                           **{n: f"content of {n}" for n in LINKED}})
    stage = tmp_path / "run" / "stages" / "2"
    stage.mkdir(parents=True)
    shutil.copy(TEMPLATES / "prepare_swap.py", stage)
    cfg = {"run_dir": str(tmp_path / "run"), "bundle_dir": str(bundle), "nearest_model_id": NEAREST,
           "base_snapshot": str(base), "new_snapshot": str(new), "new_model_id": "Altworld/Hemmingway-1",
           "tt_cache": str(tmp_path / "cache"), "hf_home": str(tmp_path / "hf"), "port": 8100}
    (stage / "swap_config.json").write_text(json.dumps(cfg))
    return {"stage": stage, "bundle": bundle, "base": base, "new": new}


def run_prepare(stage: Path):
    return subprocess.run([sys.executable, str(stage / "prepare_swap.py")], capture_output=True,
                          text=True, timeout=60)


def test_prepare_builds_the_model_dir(prep):
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    md = prep["stage"] / "model-dir"
    # Copied from the nearest model: real files with the base content. The absent video config
    # is skipped without an error.
    for name in ("config.json", "preprocessor_config.json"):
        assert (md / name).is_file() and not (md / name).is_symlink()
        assert (md / name).read_text() == (prep["base"] / name).read_text()
    assert not (md / "video_preprocessor_config.json").exists()
    # Linked from the new model: absolute links to the resolved blob, not to the snapshot link.
    for name in LINKED:
        link = md / name
        assert link.is_symlink(), name
        target = os.readlink(link)
        assert os.path.isabs(target) and target == os.path.realpath(prep["new"] / name)
        assert "/blobs/" in target
    assert sorted(p.name for p in md.iterdir()) == sorted(LINKED + ["config.json",
                                                                    "preprocessor_config.json"])


def test_prepare_edits_the_run_script(prep):
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    out = prep["stage"] / "run.sh"
    text = out.read_text()
    md = str(prep["stage"] / "model-dir")
    assert f'HERE="{prep["bundle"]}"\n' in text
    assert "BASH_SOURCE" not in text
    assert text.count("--model ") == 1 and f"--model {md} " in text
    assert "--revision" not in text and "--tokenizer-revision" not in text
    # The HF_MODEL line is set to the model-dir: the TT runtime takes its weights directory from
    # MODEL_WEIGHTS_DIR, then HF_MODEL, so the nearest model's id there serves the base weights.
    assert f'export HF_MODEL="{md}"\n' in text and NEAREST not in text
    assert "--max_num_seqs 4 --max_model_len 262144" in text
    assert out.stat().st_mode & stat.S_IXUSR
    # Everything except the three edited spots is unchanged.
    expect = (BUNDLE_RUN_SH
              .replace(f'export HF_MODEL="${{HF_MODEL:-{NEAREST}}}"', f'export HF_MODEL="{md}"')
              .replace('HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"', f'HERE="{prep["bundle"]}"')
              .replace(f'--model "{NEAREST}"', f"--model {md}")
              .replace(f" --revision {REV}", "").replace(f" --tokenizer-revision {TOK_REV}", ""))
    assert text == expect


def test_prepare_accepts_an_unquoted_model_and_reports_absent_revisions(prep):
    src = (BUNDLE_RUN_SH.replace(f'--model "{NEAREST}"', f"--model {NEAREST}")
           .replace(f" --revision {REV}", "").replace(f" --tokenizer-revision {TOK_REV}", ""))
    (prep["bundle"] / "run.sh").write_text(src)
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"--model {prep['stage'] / 'model-dir'} " in (prep["stage"] / "run.sh").read_text()
    assert "--revision" in r.stdout and "absent" in r.stdout
    assert "--tokenizer-revision" in r.stdout


def test_prepare_reports_an_absent_hf_model_line_and_goes_on(prep):
    src = BUNDLE_RUN_SH.replace(f'export HF_MODEL="${{HF_MODEL:-{NEAREST}}}"\n', "")
    (prep["bundle"] / "run.sh").write_text(src)
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "HF_MODEL" in r.stdout and "absent" in r.stdout


def test_prepare_runs_twice(prep):
    assert run_prepare(prep["stage"]).returncode == 0
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert (prep["stage"] / "model-dir" / "tokenizer.json").is_symlink()


@pytest.mark.parametrize("edit,named", [
    (lambda s: s.replace('HERE="$(cd', 'WHERE="$(cd'), "HERE="),
    (lambda s: s.replace(f'--model "{NEAREST}"', '--model "Other/Model"'), "--model"),
    (lambda s: s.replace(f'--model "{NEAREST}"', f'--model "{NEAREST}" --model "{NEAREST}"'), "--model"),
    (lambda s: s.replace(f"--revision {REV}", f"--revision {REV} --revision {REV}"), "--revision"),
])
def test_prepare_exits_2_when_an_edit_does_not_happen_exactly_once(prep, edit, named):
    (prep["bundle"] / "run.sh").write_text(edit(BUNDLE_RUN_SH))
    r = run_prepare(prep["stage"])
    assert r.returncode == 2, r.stdout + r.stderr
    assert named in r.stdout + r.stderr
    assert not (prep["stage"] / "run.sh").exists()


# ---- the drafter and the MTP head ----------------------------------------------------------------
# The nearest bundle may serve with a speculative-decoding drafter that needs the model's `mtp.*` tensors.
# Cloudflare/clef has none (it dropped them), so the bundle's engine died with "model has no MTP head" after
# the image processor error was fixed (2026-10-06). When the new model has no mtp.* tensor the script
# clears QWEN36_DRAFTER in its run.sh copy. When it cannot tell, it changes nothing.

DRAFTER_LINE = 'export QWEN36_DRAFTER="dflash2"\n'


def with_drafter(prep):
    run = prep["bundle"] / "run.sh"
    run.write_text(run.read_text().replace('CMD=(', DRAFTER_LINE + 'export DFLASH_WEIGHTS="incoai/x@abc"\nCMD=(', 1))


def new_index(prep, keys):
    """Replace the new snapshot's weight index with a real one that lists `keys`."""
    blob = (prep["new"] / "model.safetensors.index.json").resolve()
    blob.write_text(json.dumps({"metadata": {}, "weight_map": {k: "model-00001-of-00002.safetensors" for k in keys}}))


def drafter_lines(prep):
    return [l for l in (prep["stage"] / "run.sh").read_text().splitlines() if "QWEN36_DRAFTER" in l]


def test_a_model_without_mtp_tensors_gets_the_drafter_cleared(prep):
    with_drafter(prep)
    new_index(prep, ["model.embed_tokens.weight", "model.layers.0.mlp.up_proj.weight", "lm_head.weight"])
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert drafter_lines(prep) == ['export QWEN36_DRAFTER=""']
    assert "no mtp" in r.stdout.lower() and "drafter" in r.stdout.lower()


def test_a_model_with_mtp_tensors_keeps_the_drafter(prep):
    with_drafter(prep)
    new_index(prep, ["model.embed_tokens.weight", "mtp.fc.weight", "mtp.layers.0.mlp.up_proj.weight"])
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert drafter_lines(prep) == [DRAFTER_LINE.strip()]


def test_mtp_tensors_under_a_prefix_count_too(prep):
    with_drafter(prep)
    new_index(prep, ["model.embed_tokens.weight", "model.language_model.mtp.fc.weight"])
    assert run_prepare(prep["stage"]).returncode == 0
    assert drafter_lines(prep) == [DRAFTER_LINE.strip()]


def test_a_name_that_only_contains_the_letters_mtp_is_not_an_mtp_tensor(prep):
    with_drafter(prep)
    new_index(prep, ["model.embed_tokens.weight", "model.layers.0.attn_mtpx.weight"])
    assert run_prepare(prep["stage"]).returncode == 0
    assert drafter_lines(prep) == ['export QWEN36_DRAFTER=""']


def test_when_the_weights_cannot_be_read_nothing_is_changed_and_the_script_says_so(prep):
    with_drafter(prep)                                  # the fixture's index and shards are not parseable
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert drafter_lines(prep) == [DRAFTER_LINE.strip()]
    assert "could not tell" in r.stdout.lower()


def test_without_an_index_the_shard_headers_are_read(prep):
    import struct
    with_drafter(prep)
    (prep["new"] / "model.safetensors.index.json").unlink()
    header = json.dumps({"model.embed_tokens.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    for shard in sorted(prep["new"].glob("*.safetensors")):          # the fixture also has a model-mtp shard
        shard.resolve().write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")
    assert run_prepare(prep["stage"]).returncode == 0
    assert drafter_lines(prep) == ['export QWEN36_DRAFTER=""']


def test_a_bundle_with_no_drafter_line_is_left_alone_and_reported(prep):
    new_index(prep, ["model.embed_tokens.weight"])
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert drafter_lines(prep) == []
    assert "QWEN36_DRAFTER" in r.stdout and "absent" in r.stdout


def test_two_drafter_lines_are_an_error_because_one_cannot_be_chosen(prep):
    with_drafter(prep)
    run = prep["bundle"] / "run.sh"
    run.write_text(run.read_text().replace('CMD=(', DRAFTER_LINE + 'CMD=(', 1))
    new_index(prep, ["model.embed_tokens.weight"])
    r = run_prepare(prep["stage"])
    assert r.returncode == 2 and "QWEN36_DRAFTER" in r.stdout + r.stderr
    assert not (prep["stage"] / "run.sh").exists()


def test_the_other_lines_of_run_sh_are_unchanged_when_the_drafter_is_cleared(prep):
    with_drafter(prep)
    new_index(prep, ["model.embed_tokens.weight"])
    assert run_prepare(prep["stage"]).returncode == 0
    text = (prep["stage"] / "run.sh").read_text()
    assert 'export DFLASH_WEIGHTS="incoai/x@abc"' in text and 'export QWEN36_DRAFTER=""' in text
    assert text.count("QWEN36_DRAFTER") == 1


# ---- on-device sampling with the drafter off -----------------------------------------------------
# With the drafter on, the bundle's DFlash path samples on its own. With it off the model code asks for
# on-device sampling because the bundle's --additional-config sets "sample_on_device_mode", and refuses it on
# a 1x2 mesh ("requires a certified TP topology (1x4 or 1x8) ... Unset sample_on_device_mode for host
# sampling", Cloudflare/clef, 2026-10-06). So clearing the drafter also removes that key.

SAMPLING_CONFIG = ("--additional-config '{\"tt\": {\"l1_small_size\": 24576, \"fabric_config\": \"FABRIC_1D\", "
                   "\"sample_on_device_mode\": \"decode_only\", \"trace_region_size\": 1073741824}}'")


def with_sampling(prep, config=SAMPLING_CONFIG):
    run = prep["bundle"] / "run.sh"
    run.write_text(run.read_text().replace(" --max_num_seqs 4", f" --max_num_seqs 4 {config}", 1))


def additional_config(prep):
    import re
    text = (prep["stage"] / "run.sh").read_text()
    found = re.findall(r"--additional-config '([^']*)'", text)
    return [json.loads(x) for x in found]


def test_clearing_the_drafter_also_removes_on_device_sampling(prep):
    with_drafter(prep)
    with_sampling(prep)
    new_index(prep, ["model.embed_tokens.weight"])
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert additional_config(prep) == [{"tt": {"l1_small_size": 24576, "fabric_config": "FABRIC_1D",
                                               "trace_region_size": 1073741824}}]
    assert "sample_on_device_mode" in r.stdout


def test_on_device_sampling_stays_when_the_drafter_stays(prep):
    with_drafter(prep)
    with_sampling(prep)
    new_index(prep, ["model.embed_tokens.weight", "mtp.fc.weight"])
    assert run_prepare(prep["stage"]).returncode == 0
    assert additional_config(prep)[0]["tt"]["sample_on_device_mode"] == "decode_only"


def test_on_device_sampling_stays_when_the_weights_cannot_be_read(prep):
    with_drafter(prep)
    with_sampling(prep)
    assert run_prepare(prep["stage"]).returncode == 0
    assert additional_config(prep)[0]["tt"]["sample_on_device_mode"] == "decode_only"


def test_a_command_without_the_sampling_key_is_left_alone(prep):
    with_drafter(prep)
    with_sampling(prep, "--additional-config '{\"tt\": {\"l1_small_size\": 24576}}'")
    new_index(prep, ["model.embed_tokens.weight"])
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert additional_config(prep) == [{"tt": {"l1_small_size": 24576}}]


def test_a_command_without_additional_config_is_left_alone_and_reported(prep):
    with_drafter(prep)
    new_index(prep, ["model.embed_tokens.weight"])
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "additional-config" in r.stdout and "absent" in r.stdout


def test_two_additional_config_options_are_an_error_when_the_drafter_is_cleared(prep):
    with_drafter(prep)
    with_sampling(prep)
    run = prep["bundle"] / "run.sh"
    run.write_text(run.read_text().replace(" --max_num_seqs 4", f" --max_num_seqs 4 {SAMPLING_CONFIG}", 1))
    new_index(prep, ["model.embed_tokens.weight"])
    r = run_prepare(prep["stage"])
    assert r.returncode == 2 and "additional-config" in r.stdout + r.stderr
    assert not (prep["stage"] / "run.sh").exists()


def test_an_additional_config_that_is_not_json_is_left_alone_and_reported(prep):
    with_drafter(prep)
    with_sampling(prep, "--additional-config '{not json'")
    new_index(prep, ["model.embed_tokens.weight"])
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    assert "not valid JSON" in r.stdout


def test_the_sampling_key_is_removed_wherever_it_sits_in_the_tt_table(prep):
    with_drafter(prep)
    with_sampling(prep, "--additional-config '{\"tt\": {\"sample_on_device_mode\": \"all\", \"l1_small_size\": 1}}'")
    new_index(prep, ["model.embed_tokens.weight"])
    assert run_prepare(prep["stage"]).returncode == 0
    assert additional_config(prep) == [{"tt": {"l1_small_size": 1}}]


# ---- serve_and_compare.py ------------------------------------------------------------------------

VOCAB = [a + b for a in ("ba", "de", "ki", "lo", "mu", "ra", "so", "tu") for b in ("n", "l", "r", "s", "t")]
PROMPT_IDS = [1, 2, 3, 4, 5]
GENERATED = list(range(2, 34))          # 32 distinct words, so no 6-word phrase repeats


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@pytest.fixture
def swap(tmp_path):
    """A run dir with a stage 1 reference, a stage 2 dir holding serve_and_compare.py, a model-dir
    with a WordLevel tokenizer, and a run.sh that execs the fake server.

    The finalizer kills the fake server's process group if it is still there, whatever the test
    did, and reports it so a leak is visible."""
    run = tmp_path / "run"
    ref = run / "stages" / "1" / "evidence" / "reference"
    ref.mkdir(parents=True)
    (ref / "prompt-ids.json").write_text(json.dumps({"prompt_ids": PROMPT_IDS}))
    (ref / "generated-ids.json").write_text(json.dumps(
        {"generated_ids": GENERATED, "generated_text": " ".join(VOCAB[i] for i in GENERATED)}))
    stage = run / "stages" / "2"
    md = stage / "model-dir"
    md.mkdir(parents=True)
    tok = Tokenizer(WordLevel({w: i for i, w in enumerate(VOCAB)} | {"[UNK]": len(VOCAB)},
                              unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.save(str(md / "tokenizer.json"))
    shutil.copy(TEMPLATES / "serve_and_compare.py", stage)
    pid_file = tmp_path / "server-pid.json"
    server_cfg = tmp_path / "server.json"
    weights_env = tmp_path / "weights-env.txt"
    (stage / "run.sh").write_text(
        "#!/usr/bin/env bash\n"
        f'printf "%s\\n%s\\n" "${{MODEL_WEIGHTS_DIR-<unset>}}" "${{HF_MODEL-<unset>}}" > "{weights_env}"\n'
        f'exec "{sys.executable}" "{FAKE_SERVER}" --config "{server_cfg}" "$@"\n')
    cfg = {"run_dir": str(run), "bundle_dir": str(tmp_path / "bundle"), "nearest_model_id": NEAREST,
           "base_snapshot": str(tmp_path / "base"), "new_snapshot": str(tmp_path / "new"),
           "new_model_id": "Altworld/Hemmingway-1", "tt_cache": str(tmp_path / "cache" / "tt_cache"),
           "hf_home": str(tmp_path / "hf"), "port": free_port(), "health_timeout_s": 30}

    state = {"stage": stage, "run": run, "cfg": cfg, "pid_file": pid_file, "weights_env": weights_env,
             "server_cfg": server_cfg}

    def start(mode="perfect", **overrides):
        (stage / "swap_config.json").write_text(json.dumps(cfg | overrides))
        server_cfg.write_text(json.dumps({"mode": mode, "vocab": VOCAB, "prompt_ids": PROMPT_IDS,
                                          "generated_ids": GENERATED, "model": str(md),
                                          "pid_file": str(pid_file)}))
        # The test's own environment must not supply the variables under test.
        env = {k: v for k, v in os.environ.items() if k not in ("MODEL_WEIGHTS_DIR", "HF_MODEL")}
        return subprocess.run([sys.executable, str(stage / "serve_and_compare.py")],
                              capture_output=True, text=True, timeout=180, env=env)

    state["start"] = start
    try:
        yield state
    finally:
        if pid_file.exists():
            pgid = json.loads(pid_file.read_text())["pgid"]
            if group_alive(pgid):
                os.killpg(pgid, signal.SIGKILL)
                print(f"fixture killed a leaked fake server group {pgid}", file=sys.stderr)


def server_pgid(state) -> int:
    return json.loads(state["pid_file"].read_text())["pgid"]


def wait_gone(pgid: int, within: float = 5.0) -> bool:
    end = time.monotonic() + within
    while time.monotonic() < end:
        if not group_alive(pgid):
            return True
        time.sleep(0.1)
    return not group_alive(pgid)


def report(state) -> dict:
    return json.loads((state["stage"] / "evidence" / "swap-check.json").read_text())


def test_perfect_agreement(swap):
    r = swap["start"]("perfect")
    assert r.returncode == 0, r.stdout + r.stderr
    rep = report(swap)
    draft = rep["result_draft"]
    assert set(draft) == {"serves", "server_ready_s", "coherent", "free_run_text", "top1_agreement",
                          "n_tokens", "cache_dir", "evidence"}
    assert draft["serves"] is True and draft["coherent"] is True
    assert draft["top1_agreement"] == 1.0 and draft["n_tokens"] == 32
    assert draft["server_ready_s"] > 0
    assert draft["cache_dir"] == swap["cfg"]["tt_cache"]
    assert draft["evidence"] == ["stages/2/evidence/swap-check.json", "stages/2/evidence/server.log"]
    assert all((swap["run"] / p).is_file() for p in draft["evidence"])
    assert draft["free_run_text"].split() == [VOCAB[i] for i in GENERATED][:len(draft["free_run_text"].split())]
    assert rep["mismatches"] == []
    assert json.loads(r.stdout[r.stdout.index("{"):]) == draft
    # The server got the cache, HF home and offline flag; the cache was created with its marker.
    md = str(swap["stage"] / "model-dir")
    env = json.loads(swap["pid_file"].read_text())["env"]
    assert env == {"TT_CACHE_PATH": swap["cfg"]["tt_cache"], "TT_CACHE_HOME": swap["cfg"]["tt_cache"],
                   "HF_HOME": swap["cfg"]["hf_home"], "HF_HUB_OFFLINE": "1",
                   "MODEL_WEIGHTS_DIR": md, "HF_MODEL": md}
    marker = Path(swap["cfg"]["tt_cache"]) / ".orchard-model"
    assert marker.read_text() == "Altworld/Hemmingway-1"
    assert wait_gone(server_pgid(swap))


def test_every_fourth_token_wrong(swap):
    r = swap["start"]("every4")
    assert r.returncode == 0, r.stdout + r.stderr
    rep = report(swap)
    assert rep["result_draft"]["top1_agreement"] == 0.75
    assert [m["position"] for m in rep["mismatches"]] == list(range(3, 32, 4))
    m = rep["mismatches"][0]
    assert m["expected_text"].strip() == VOCAB[GENERATED[3]]
    assert m["got_text"].strip() == VOCAB[GENERATED[3] + 1]
    assert rep["result_draft"]["coherent"] is True


def test_an_http_error_exits_5_and_stops_the_server(swap):
    r = swap["start"]("http500")
    assert r.returncode == 5, r.stdout + r.stderr
    assert "fake server failure" in r.stdout + r.stderr
    assert wait_gone(server_pgid(swap)), "the server's process group is still running"


def test_an_error_in_the_script_still_stops_the_server(swap):
    r = swap["start"]("perfect", test_raise_after_ready=True)
    assert r.returncode not in (0, 3, 4, 5), r.stdout + r.stderr
    assert swap["pid_file"].exists()
    assert wait_gone(server_pgid(swap)), "the server's process group is still running"


def test_a_cache_without_the_marker_is_refused_and_no_server_starts(swap):
    cache = Path(swap["cfg"]["tt_cache"])
    cache.mkdir(parents=True)
    (cache / "layer0.bin").write_text("weights of some other model")
    r = swap["start"]("perfect")
    assert r.returncode == 3, r.stdout + r.stderr
    assert ".orchard-model" in r.stdout + r.stderr
    assert not swap["pid_file"].exists()


def test_a_cache_with_a_wrong_marker_is_refused(swap):
    cache = Path(swap["cfg"]["tt_cache"])
    cache.mkdir(parents=True)
    (cache / ".orchard-model").write_text("Someone/Else")
    (cache / "layer0.bin").write_text("x")
    assert swap["start"]("perfect").returncode == 3
    assert not swap["pid_file"].exists()


def test_a_cache_with_the_matching_marker_is_reused(swap):
    cache = Path(swap["cfg"]["tt_cache"])
    cache.mkdir(parents=True)
    (cache / ".orchard-model").write_text("Altworld/Hemmingway-1")
    (cache / "layer0.bin").write_text("converted earlier by this run")
    r = swap["start"]("perfect")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (cache / "layer0.bin").read_text() == "converted earlier by this run"


def test_a_server_that_exits_before_it_is_healthy_exits_4(swap):
    r = swap["start"]("die")
    assert r.returncode == 4, r.stdout + r.stderr
    out = r.stdout + r.stderr
    assert "fake server log line 69" in out and "fake server log line 10" in out
    assert "fake server log line 9\n" not in out          # only the last 60 lines
    assert wait_gone(server_pgid(swap))


def test_the_script_sends_only_the_fields_the_server_accepts():
    text = (TEMPLATES / "serve_and_compare.py").read_text()
    assert not re.search(r'["\']logprobs["\']\s*:', text)
    assert not re.search(r'["\']top_p["\']\s*:', text)


def test_the_server_is_told_to_load_the_model_dir_weights(swap):
    # The TT runtime takes its weights directory from MODEL_WEIGHTS_DIR, then HF_MODEL. Both must
    # name the model-dir, or the server loads the nearest model's weights from the HF cache.
    r = swap["start"]("perfect")
    assert r.returncode == 0, r.stdout + r.stderr
    md = str(swap["stage"] / "model-dir")
    assert swap["weights_env"].read_text().splitlines() == [md, md]
    rep = report(swap)
    assert rep["weights_dir_env"] == {"MODEL_WEIGHTS_DIR": md, "HF_MODEL": md}
    assert rep["hf_model_env"] == md


def test_prepare_then_serve_through_a_fake_bundle(swap, tmp_path):
    """The whole path: prepare_swap edits a bundle run.sh whose command execs the fake server."""
    bundle = tmp_path / "bundle"
    (bundle / "venv" / "bin").mkdir(parents=True)
    (bundle / "venv" / "bin" / "python").symlink_to(sys.executable)
    (bundle / "run.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'PYBIN="$HERE/venv/bin/python"\n'
        f'export HF_MODEL="${{HF_MODEL:-{NEAREST}}}"\n'
        f'CMD=("$PYBIN" "{FAKE_SERVER}" --config "{swap['server_cfg']}" --model "{NEAREST}" '
        f'--revision {REV} --tokenizer-revision {TOK_REV} "$@")\n'
        'exec "${CMD[@]}"\n')
    base = make_snapshot(tmp_path / "base", {"config.json": "{}"})
    new = make_snapshot(tmp_path / "new", {"tokenizer.json": (swap["stage"] / "model-dir" /
                                                               "tokenizer.json").read_text(),
                                           "model-00001-of-00001.safetensors": "w"})
    stage = swap["stage"]
    shutil.copy(TEMPLATES / "prepare_swap.py", stage)
    (stage / "swap_config.json").write_text(json.dumps(
        swap["cfg"] | {"bundle_dir": str(bundle), "base_snapshot": str(base), "new_snapshot": str(new)}))
    p = run_prepare(stage)
    assert p.returncode == 0, p.stdout + p.stderr
    r = swap["start"]("perfect", bundle_dir=str(bundle), base_snapshot=str(base),
                      new_snapshot=str(new))
    assert r.returncode == 0, r.stdout + r.stderr
    md = str(stage / "model-dir")
    seen = json.loads(swap["pid_file"].read_text())
    assert seen["model_arg"] == md
    assert seen["env"]["MODEL_WEIGHTS_DIR"] == md and seen["env"]["HF_MODEL"] == md
    assert report(swap)["result_draft"]["top1_agreement"] == 1.0



# ---- prepare_swap.py for a container package (stage 4) ------------------------------------------

def container_config(prep, **change):
    cfg = json.loads((prep["stage"] / "swap_config.json").read_text())
    del cfg["bundle_dir"]
    cfg.update(change)
    (prep["stage"] / "swap_config.json").write_text(json.dumps(cfg))


def test_prepare_for_a_container_package_builds_only_the_model_dir(prep):
    container_config(prep, package="changh95/qwen3.8-27b-p300x2")
    r = run_prepare(prep["stage"])
    assert r.returncode == 0, r.stdout + r.stderr
    md = prep["stage"] / "model-dir"
    assert (md / "config.json").read_text() == (prep["base"] / "config.json").read_text()
    assert os.readlink(md / "tokenizer.json") == os.path.realpath(prep["new"] / "tokenizer.json")
    assert not (prep["stage"] / "run.sh").exists()
    assert "changh95/qwen3.8-27b-p300x2 is a container package" in r.stdout


def test_prepare_with_neither_a_bundle_nor_a_package_exits_2(prep):
    container_config(prep)
    r = run_prepare(prep["stage"])
    assert r.returncode == 2, r.stdout + r.stderr
    assert "bundle_dir (or package" in r.stderr
    assert not (prep["stage"] / "model-dir").exists()


def test_the_rewritten_additional_config_stays_on_the_commands_one_line(prep):
    with_drafter(prep)
    with_sampling(prep)
    new_index(prep, ["model.embed_tokens.weight"])
    assert run_prepare(prep["stage"]).returncode == 0
    lines = [l for l in (prep["stage"] / "run.sh").read_text().splitlines() if "--additional-config" in l]
    assert len(lines) == 1 and lines[0].startswith("CMD=(") and lines[0].rstrip().endswith(")")
