# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""serve_and_compare_container.py: the stage 4 template that serves the swapped weights with a
container package. The argv edit is tested in-process; the whole script runs against a fake
`tt-model` and a fake `docker` (tests/fake_tt_model.py, tests/fake_docker.py) that start
tests/fake_swap_server.py, which opens no device."""
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from container_fakes import IMAGE, NEAREST, PACKAGE, REVISION, printed_argv

REPO = Path(__file__).resolve().parent.parent
TEMPLATES = REPO / "orchard" / "skills" / "weights-swap-templates"
LABEL = "orchard.test=0123456789ab"


def load_template():
    spec = importlib.util.spec_from_file_location("serve_and_compare_container",
                                                  TEMPLATES / "serve_and_compare_container.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


sac = load_template()

# ---- the argv edit ---------------------------------------------------------------------------------


@pytest.fixture
def paths(tmp_path):
    p = {"hf": tmp_path / "home" / ".cache" / "huggingface",
         "pkg": tmp_path / "home" / ".cache" / "tt-model" / "qwen3.8-27b-p300x2",
         "md": tmp_path / "configs" / "4" / "model-dir", "cache": tmp_path / "cache" / "4chip" / "tt_cache",
         "hf_dir": tmp_path / "configs" / "4" / "hf-isolated",
         "blobs": str(tmp_path / "home" / ".cache" / "huggingface" / "hub" / "models--A--B" / "blobs")}
    return p


def edit(paths, printed=None, ids=(0, 1, 2, 3), port=8100, **kw):
    argv = printed if printed is not None else printed_argv(hf=paths["hf"], pkg_cache=paths["pkg"],
                                                            port=port, device_ids=list(ids))
    return sac.edit_docker_argv(argv, image=IMAGE, nearest=NEAREST, model_dir=paths["md"],
                                tt_cache=paths["cache"], hf_dir=paths["hf_dir"],
                                name="orchard-4chip-8100", label=LABEL, device_ids=list(ids),
                                port=port, blobs=[paths["blobs"]], **kw)


def pairs(argv, flag):
    return [argv[i + 1] for i, a in enumerate(argv[:-1]) if a == flag]


def test_the_edit_points_every_weight_path_at_the_new_model(paths):
    out = edit(paths)
    md = str(paths["md"])
    k = out.index(IMAGE)
    assert out[:3] == ["docker", "run", "--detach"]
    vols = pairs(out[:k], "--volume")
    assert f"{paths['cache']}:/tensor-cache" in vols                 # a fresh cache for this model
    assert f"{paths['hf_dir']}:/hf" in vols                          # the base weights are not mounted
    assert f"{md}:{md}:ro" in vols and f"{paths['blobs']}:{paths['blobs']}:ro" in vols
    assert not any(v.startswith(f"{paths['hf']}:") or v.startswith(f"{paths['pkg']}/tensors")
                   for v in vols)
    envs = pairs(out[:k], "--env")
    assert envs.count(f"HF_MODEL={md}") == 1 and envs.count(f"MODEL_WEIGHTS_DIR={md}") == 1
    assert not any(e.startswith("HF_MODEL=Qwen") for e in envs)
    assert "TT_CACHE_PATH=/tensor-cache" in envs and "HF_HOME=/hf" in envs
    assert pairs(out[:k], "--label") == [LABEL] and pairs(out[:k], "--name") == ["orchard-4chip-8100"]
    assert pairs(out[:k], "--device") == [f"/dev/tenstorrent/{d}:/dev/tenstorrent/{d}" for d in range(4)]
    assert out[k + 1:k + 4] == ["vllm", "serve", md]
    assert "--revision" not in out and REVISION not in out
    assert pairs(out[k:], "--port") == ["8100"] and "--max-model-len" in out[k:]


def test_options_the_edit_does_not_touch_are_kept_in_order(paths):
    printed = printed_argv(hf=paths["hf"], pkg_cache=paths["pkg"], port=8100, device_ids=[0, 1, 2, 3])
    kept = ["--user", "--ipc", "--mount", "--publish"]
    before = [(f, v) for f, v in zip(printed, printed[1:]) if f in kept]
    out = edit(paths)
    assert [(f, v) for f, v in zip(out, out[1:]) if f in kept] == before
    assert f"{paths['pkg']}/cache:/cache" in pairs(out, "--volume")   # the JIT kernel cache stays


def test_an_existing_model_weights_dir_is_replaced_and_not_added_twice(paths):
    printed = printed_argv(hf=paths["hf"], pkg_cache=paths["pkg"], port=8100, device_ids=[0, 1, 2, 3],
                           extra_env={"MODEL_WEIGHTS_DIR": "/somewhere/else"})
    out = edit(paths, printed=printed)
    assert [e for e in pairs(out, "--env") if e.startswith("MODEL_WEIGHTS_DIR=")] == [
        f"MODEL_WEIGHTS_DIR={paths['md']}"]


def test_a_token_variable_is_not_passed_to_the_test_container(paths):
    printed = printed_argv(hf=paths["hf"], pkg_cache=paths["pkg"], port=8100, device_ids=[0, 1, 2, 3])
    k = printed.index(IMAGE)
    printed = printed[:k] + ["--env", "HF_TOKEN"] + printed[k:]
    assert "HF_TOKEN" not in edit(paths, printed=printed)


# ---- the drafter and the MTP head ------------------------------------------------------------------
# Same finding as in prepare_swap.py: a speculative drafter needs the model's mtp.* tensors.

def printed(paths, **kw):
    return printed_argv(hf=paths["hf"], pkg_cache=paths["pkg"], port=8100, device_ids=[0, 1, 2, 3], **kw)


def second_drafter(paths):
    """The usual printed command already sets QWEN36_DRAFTER=mtp; this adds a second option."""
    argv = printed(paths)
    k = argv.index(IMAGE)
    return argv[:k] + ["--env", "QWEN36_DRAFTER=dflash2"] + argv[k:]


def without_drafter(paths):
    argv, out, i = printed(paths), [], 0
    while i < len(argv):
        if argv[i] == "--env" and argv[i + 1].startswith("QWEN36_DRAFTER"):
            i += 2
            continue
        out.append(argv[i])
        i += 1
    return out


def drafter_env(argv):
    return [e for e in pairs(argv, "--env") if e.startswith("QWEN36_DRAFTER")]


def test_the_drafter_is_cleared_when_the_new_model_has_no_mtp_tensors(paths):
    assert drafter_env(edit(paths, drafter_off=True)) == ["QWEN36_DRAFTER="]


def test_the_drafter_is_kept_when_the_model_has_mtp_tensors(paths):
    assert drafter_env(edit(paths)) == ["QWEN36_DRAFTER=mtp"]
    assert drafter_env(edit(paths, drafter_off=False)) == ["QWEN36_DRAFTER=mtp"]


def test_a_command_without_a_drafter_variable_is_unchanged_by_drafter_off(paths):
    plain = without_drafter(paths)
    assert drafter_env(plain) == []
    assert edit(paths, printed=plain, drafter_off=True) == edit(paths, printed=plain)


def test_two_drafter_variables_cannot_be_cleared_because_one_cannot_be_chosen(paths):
    with pytest.raises(sac.EditError) as exc:
        edit(paths, printed=second_drafter(paths), drafter_off=True)
    assert "QWEN36_DRAFTER" in str(exc.value)


def test_two_drafter_variables_are_left_alone_when_the_drafter_is_kept(paths):
    assert len(drafter_env(edit(paths, printed=second_drafter(paths)))) == 2


def test_clearing_the_drafter_changes_nothing_else(paths):
    kept, cleared = edit(paths), edit(paths, drafter_off=True)
    assert [a for a in kept if not a.startswith("QWEN36_DRAFTER")] == [a for a in cleared if not a.startswith("QWEN36_DRAFTER")]


def write_shard(path, names):
    import json, struct
    header = json.dumps({n: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]} for n in names}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"\0\0")


def test_the_template_reads_mtp_tensors_from_the_model_directory(tmp_path):
    import json
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "x", "mtp.fc.weight": "x"}}))
    assert sac.has_mtp_tensors(tmp_path) is True
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"a": "x"}}))
    assert sac.has_mtp_tensors(tmp_path) is False
    (tmp_path / "model.safetensors.index.json").unlink()
    assert sac.has_mtp_tensors(tmp_path) is None                        # no index and no shards
    write_shard(tmp_path / "model-1.safetensors", ["a", "model.language_model.mtp.norm.weight"])
    assert sac.has_mtp_tensors(tmp_path) is True


def test_the_two_templates_use_the_same_mtp_check():
    """prepare_swap.py and serve_and_compare_container.py are copied into a stage directory on their own, so
    each carries the check. They must not drift."""
    import importlib.util
    import inspect
    spec = importlib.util.spec_from_file_location("prepare_swap_under_test", TEMPLATES / "prepare_swap.py")
    ps = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ps)
    for name in ("shard_tensor_names", "has_mtp_tensors"):
        assert inspect.getsource(getattr(ps, name)) == inspect.getsource(getattr(sac, name)), name
    assert ps.MTP_NAME.pattern == sac.MTP_NAME.pattern


def bad(paths, change):
    printed = printed_argv(hf=paths["hf"], pkg_cache=paths["pkg"], port=8100, device_ids=[0, 1, 2, 3])
    return change(printed, paths)


def swap(old, new, count=1):
    def change(argv, paths):
        text = "\x00".join(argv)
        assert text.count(old) >= 1, old
        return text.replace(old, new, count).split("\x00")
    return change


def insert_before_image(*tokens):
    def change(argv, paths):
        k = argv.index(IMAGE)
        return argv[:k] + list(tokens) + argv[k:]
    return change


@pytest.mark.parametrize("change,words", [
    (swap("/dev/tenstorrent/0:/dev/tenstorrent/0", "/dev/tenstorrent"), "not one /dev/tenstorrent"),
    (swap("/dev/tenstorrent/3:/dev/tenstorrent/3", "/dev/tenstorrent/5:/dev/tenstorrent/5"), "the lease gave"),
    (insert_before_image("--volume", "/x:/tensor-cache"), "/tensor-cache in the printed command, found 2"),
    (swap("\x00--env\x00HF_MODEL=Qwen/Qwen3.8-27B", ""), "HF_MODEL"),
    (insert_before_image("--volume", "/x:/hf-hub"), "/hf-hub"),
    (insert_before_image("--privileged"), "unknown option '--privileged'"),
    (swap("TT_CACHE_PATH=/tensor-cache", "TT_CACHE_PATH=/elsewhere"), "the tensor cache must be"),
    (swap(f"\x00{IMAGE}\x00", "\x00other/image:1\x00"), "image"),
    (insert_before_image(IMAGE), "image"),
    (swap(f"serve\x00{NEAREST}", "serve\x00Other/Model"), NEAREST),
    (swap(f"--revision\x00{REVISION}", f"--revision\x00{REVISION}\x00--revision\x00{REVISION}"),
     "at most one --revision"),
    (swap("--port\x008100", "--port\x008000"), "the config's port is 8100"),
    (lambda argv, p: ["docker", "create"] + argv[2:], "docker run"),
])
def test_a_printed_command_of_another_shape_is_refused(paths, change, words):
    with pytest.raises(sac.EditError) as exc:
        edit(paths, printed=bad(paths, change))
    assert words in str(exc.value)


def test_blob_dirs_lists_the_directories_the_links_point_into(tmp_path):
    md, blobs = tmp_path / "model-dir", tmp_path / "hub" / "models--A--B" / "blobs"
    md.mkdir()
    blobs.mkdir(parents=True)
    for name in ("tokenizer.json", "model-00001.safetensors"):
        (blobs / f"b-{name}").write_text("x")
        (md / name).symlink_to(blobs / f"b-{name}")
    (md / "config.json").write_text("{}")                 # a copied file is inside model-dir already
    assert sac.blob_dirs(md) == [str(blobs)]


# ---- the whole script, against fake tt-model and docker -------------------------------------------

TESTS = Path(__file__).resolve().parent
FAKE_SERVER, FAKE_TT_MODEL, FAKE_DOCKER = (TESTS / "fake_swap_server.py", TESTS / "fake_tt_model.py",
                                           TESTS / "fake_docker.py")


@pytest.fixture
def crig(tmp_path):
    """An operator home with the package installed and both models in its HF cache, a run with a
    stage 1 reference, one configuration directory prepared by prepare_swap.py, and fake tt-model
    and docker first on PATH. The finalizer kills any fake server a test left running."""
    pytest.importorskip("tokenizers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from test_weights_swap_templates import GENERATED, PROMPT_IDS, VOCAB, free_port, make_snapshot

    home = tmp_path / "operator-home"
    hf = home / ".cache" / "huggingface"
    pkg_cache = home / ".cache" / "tt-model" / "qwen3.8-27b-p300x2"
    for d in ("cache", "weights", "tensors"):
        (pkg_cache / d).mkdir(parents=True)
    (home / ".cache" / "tt-model" / "installed.json").write_text(json.dumps(
        {PACKAGE: {"repo_id": PACKAGE, "container": True, "image": IMAGE, "profile": "batch32"}}))
    tok = Tokenizer(WordLevel({w: i for i, w in enumerate(VOCAB)} | {"[UNK]": len(VOCAB)},
                              unk_token="[UNK]"))
    tok.pre_tokenizer = Whitespace()
    tok.save(str(tmp_path / "tokenizer.json"))
    base = make_snapshot(hf / "hub" / "models--Qwen--Qwen3.8-27B", {"config.json": '{"base": true}'})
    new = make_snapshot(hf / "hub" / "models--Altworld--Hemmingway-1",
                        {"tokenizer.json": (tmp_path / "tokenizer.json").read_text(),
                         "model-00001-of-00001.safetensors": "new weights"})
    run = tmp_path / "run"
    ref = run / "stages" / "1" / "evidence" / "reference"
    ref.mkdir(parents=True)
    (ref / "prompt-ids.json").write_text(json.dumps({"prompt_ids": PROMPT_IDS}))
    (ref / "generated-ids.json").write_text(json.dumps(
        {"generated_ids": GENERATED, "generated_text": " ".join(VOCAB[i] for i in GENERATED)}))
    cdir = run / "stages" / "4" / "configs" / "4"
    cdir.mkdir(parents=True)
    for name in ("prepare_swap.py", "serve_and_compare.py", "serve_and_compare_container.py"):
        shutil.copy(TEMPLATES / name, cdir)
    cfg = {"run_dir": str(run), "nearest_model_id": NEAREST, "base_snapshot": str(base),
           "new_snapshot": str(new), "new_model_id": "Altworld/Hemmingway-1",
           "tt_cache": str(tmp_path / "orchard-cache" / "hemmingway-1" / "4chip-p300x2" / "tt_cache"),
           "hf_home": str(hf), "operator_home": str(home), "port": free_port(), "package": PACKAGE,
           "profile": "batch32", "chips": 4, "health_timeout_s": 30}
    (cdir / "swap_config.json").write_text(json.dumps(cfg))
    prep = subprocess.run([sys.executable, str(cdir / "prepare_swap.py")], capture_output=True,
                          text=True, timeout=60)
    assert prep.returncode == 0, prep.stdout + prep.stderr
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, script in (("tt-model", FAKE_TT_MODEL), ("docker", FAKE_DOCKER)):
        (bin_dir / name).write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
        (bin_dir / name).chmod(0o755)
    state = tmp_path / "docker-state"
    state.mkdir()
    pid_file, server_cfg = tmp_path / "server-pid.json", tmp_path / "server.json"
    (state / "config.json").write_text(json.dumps({"image": IMAGE, "fake_server": str(FAKE_SERVER),
                                                   "server_config": str(server_cfg)}))
    tt_cfg, calls = tmp_path / "tt-model.json", tmp_path / "tt-model-calls.jsonl"
    rig = {"cdir": cdir, "cfg": cfg, "home": home, "hf": hf, "pkg_cache": pkg_cache, "state": state,
           "pid_file": pid_file, "calls": calls}

    def start(mode="perfect", printed=None, env_change=None, **overrides):
        (cdir / "swap_config.json").write_text(json.dumps(cfg | overrides))
        server_cfg.write_text(json.dumps({"mode": mode, "vocab": VOCAB, "prompt_ids": PROMPT_IDS,
                                          "generated_ids": GENERATED, "model": str(cdir / "model-dir"),
                                          "pid_file": str(pid_file)}))
        tt_cfg.write_text(json.dumps({"calls": str(calls), "pkg_cache": str(pkg_cache),
                                      "printed": printed or {}}))
        env = {k: v for k, v in os.environ.items()
               if k not in ("MODEL_WEIGHTS_DIR", "HF_MODEL", "HF_HUB_CACHE")}
        env.update(PATH=f"{bin_dir}:{env['PATH']}", FAKE_DOCKER_STATE=str(state),
                   FAKE_TT_MODEL_CONFIG=str(tt_cfg), ORCHARD_DEVICE_IDS="0,1,2,3",
                   ORCHARD_TEST_LABEL=LABEL)
        for k, v in (env_change or {}).items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
        return subprocess.run([sys.executable, str(cdir / "serve_and_compare_container.py")],
                              capture_output=True, text=True, timeout=180, env=env)

    rig["start"] = start
    try:
        yield rig
    finally:
        for rec in (state / "containers").glob("*.json"):
            pid = json.loads(rec.read_text())["pid"]
            try:
                os.killpg(pid, signal.SIGKILL)
                print(f"fixture killed a leaked fake container group {pid}", file=sys.stderr)
            except ProcessLookupError:
                pass


def docker_runs(rig) -> list[list[str]]:
    path = rig["state"] / "runs.jsonl"
    return [json.loads(ln) for ln in path.read_text().splitlines()] if path.exists() else []


def leftover(rig) -> list[str]:
    return sorted(p.stem for p in (rig["state"] / "containers").glob("*.json"))


def server_gone(rig, within=5.0) -> bool:
    pgid = json.loads(rig["pid_file"].read_text())["pgid"]
    end = time.monotonic() + within
    while time.monotonic() < end:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


def test_the_container_serves_the_new_weights_and_is_removed_afterwards(crig):
    r = crig["start"]()
    assert r.returncode == 0, r.stdout + r.stderr
    md = str(crig["cdir"] / "model-dir")
    seen = json.loads(crig["pid_file"].read_text())
    assert seen["model_arg"] == md
    assert seen["env"]["MODEL_WEIGHTS_DIR"] == md and seen["env"]["HF_MODEL"] == md
    assert seen["env"]["TT_CACHE_PATH"] == "/tensor-cache" and seen["env"]["HF_HOME"] == "/hf"
    [argv] = docker_runs(crig)
    vols = pairs(argv, "--volume")
    assert f"{crig['cfg']['tt_cache']}:/tensor-cache" in vols
    assert f"{crig['cdir'] / 'hf-isolated'}:/hf" in vols
    assert not any(v.split(":")[0] == str(crig["hf"]) for v in vols)
    assert pairs(argv, "--label") == [LABEL]
    rep = json.loads((crig["cdir"] / "evidence" / "swap-check.json").read_text())
    assert rep["result_draft"]["top1_agreement"] == 1.0 and rep["result_draft"]["coherent"] is True
    assert rep["result_draft"]["evidence"] == ["stages/4/configs/4/evidence/swap-check.json",
                                               "stages/4/configs/4/evidence/server.log"]
    assert rep["kind"] == "container" and rep["container_stopped"] is True and rep["device_ids"] == [0, 1, 2, 3]
    assert json.loads((crig["cdir"] / "evidence" / "docker-argv.json").read_text()) == ["docker", "run"] + argv
    [call] = [json.loads(ln) for ln in crig["calls"].read_text().splitlines()]
    assert call["HOME"] == str(crig["home"]) and call["HF_HOME"] == str(crig["hf"])
    assert call["argv"][:2] == ["serve", PACKAGE] and call["argv"][-1] == "--print"
    assert call["argv"][call["argv"].index("--device-id") + 1] == "0,1,2,3"
    assert (Path(crig["cfg"]["tt_cache"]) / ".orchard-model").read_text() == "Altworld/Hemmingway-1"
    assert leftover(crig) == [] and server_gone(crig)


@pytest.mark.parametrize("mode,overrides,code", [("http500", {}, 5), ("die", {}, 4),
                                                 ("perfect", {"test_raise_after_ready": True}, None)])
def test_the_container_is_stopped_and_removed_on_every_exit_path(crig, mode, overrides, code):
    r = crig["start"](mode, **overrides)
    if code is None:
        assert r.returncode not in (0, 2, 3, 4, 5), r.stdout + r.stderr
    else:
        assert r.returncode == code, r.stdout + r.stderr
    assert leftover(crig) == [], "docker still lists the test container"
    assert server_gone(crig), "the fake container's server is still running"


def test_a_server_that_exits_shows_the_end_of_its_log(crig):
    r = crig["start"]("die")
    assert "fake server log line 69" in r.stdout and "exited with code 1" in r.stdout


def test_a_cache_inside_the_package_caches_is_refused_before_anything_starts(crig):
    r = crig["start"](tt_cache=str(crig["pkg_cache"] / "tensors"))
    assert r.returncode == 3, r.stdout + r.stderr
    assert "where the packages keep their own tensor caches" in r.stderr
    assert docker_runs(crig) == [] and not crig["calls"].exists()


def test_a_cache_marked_for_another_model_is_refused(crig):
    cache = Path(crig["cfg"]["tt_cache"])
    cache.mkdir(parents=True)
    (cache / ".orchard-model").write_text("Qwen/Qwen3.8-27B")
    (cache / "layer0.bin").write_text("base model tensors")
    assert crig["start"]().returncode == 3
    assert docker_runs(crig) == []


def test_a_printed_command_that_maps_every_device_exits_2_and_starts_nothing(crig):
    r = crig["start"](printed={"whole_dir": True})
    assert r.returncode == 2, r.stdout + r.stderr
    assert "not one /dev/tenstorrent/<N> node" in r.stderr
    assert docker_runs(crig) == []


@pytest.mark.parametrize("env_change", [{"ORCHARD_TEST_LABEL": None}, {"ORCHARD_DEVICE_IDS": "0,1"},
                                        {"ORCHARD_DEVICE_IDS": None}])
def test_without_the_supervisors_variables_it_exits_2(crig, env_change):
    r = crig["start"](env_change=env_change)
    assert r.returncode == 2, r.stdout + r.stderr
    assert "ORCHARD_TEST_LABEL and ORCHARD_DEVICE_IDS" in r.stderr
    assert docker_runs(crig) == [] and not crig["calls"].exists()


def test_a_package_that_is_not_installed_exits_2(crig):
    r = crig["start"](package="someone/else-p300x2")
    assert r.returncode == 2 and "is not installed" in r.stderr
    assert docker_runs(crig) == []


# ---- the drafter, through the whole script ------------------------------------------------------

def run_env_values(rig, key):
    argv = docker_runs(rig)[0]
    return [argv[i + 1].split("=", 1)[1] for i, a in enumerate(argv[:-1])
            if a == "--env" and argv[i + 1].startswith(key + "=")]


def set_index(rig, keys):
    (rig["cdir"] / "model-dir" / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model-00001-of-00001.safetensors" for k in keys}}))


def test_a_model_without_mtp_tensors_is_served_with_the_drafter_cleared(crig):
    set_index(crig, ["model.embed_tokens.weight", "lm_head.weight"])
    r = crig["start"]()
    assert r.returncode == 0, r.stdout + r.stderr
    assert run_env_values(crig, "QWEN36_DRAFTER") == [""]


def test_a_model_with_mtp_tensors_is_served_with_the_drafter_the_package_sets(crig):
    set_index(crig, ["model.embed_tokens.weight", "mtp.fc.weight"])
    r = crig["start"]()
    assert r.returncode == 0, r.stdout + r.stderr
    assert run_env_values(crig, "QWEN36_DRAFTER") == ["mtp"]


def test_when_the_model_directory_cannot_say_the_drafter_is_left_as_the_package_sets_it(crig):
    r = crig["start"]()                              # the fixture's weights file is not a safetensors file
    assert r.returncode == 0, r.stdout + r.stderr
    assert run_env_values(crig, "QWEN36_DRAFTER") == ["mtp"]


# ---- what the container check records about how it served ---------------------------------------------

def argv_with(*pairs):
    return ["docker", "run", *[x for k, v in pairs for x in (k, v)], "image", "vllm", "serve", "m"]


def test_the_container_serving_record_names_the_drafter():
    assert sac.serving_from_argv(argv_with(("--env", "QWEN36_DRAFTER=")))["drafter"] == "off"
    assert sac.serving_from_argv(argv_with(("--env", "QWEN36_DRAFTER=mtp")))["drafter"] == "on (mtp)"
    assert sac.serving_from_argv(argv_with(("--env", "HF_HOME=/hf")))["drafter"] == "not set by the package"


def test_the_container_serving_record_names_where_sampling_happens():
    on = '{"tt": {"sample_on_device_mode": "all"}}'
    host = '{"tt": {"fabric_config": "FABRIC_1D"}}'
    assert sac.serving_from_argv(["--additional-config", on])["sampling"] == "on device (all)"
    assert sac.serving_from_argv(["--additional-config", host])["sampling"] == "host"
    assert sac.serving_from_argv(["--additional-config", "{no"])["sampling"] == "unknown"
    assert sac.serving_from_argv(["--port", "1"])["sampling"] == "not set by the package"


def test_the_container_report_carries_the_serving_record(crig):
    set_index(crig, ["model.embed_tokens.weight"])
    r = crig["start"]()
    assert r.returncode == 0, r.stdout + r.stderr
    rep = json.loads((crig["cdir"] / "evidence" / "swap-check.json").read_text())
    assert rep["serving"]["drafter"] == "off"
