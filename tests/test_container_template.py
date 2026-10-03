"""serve_and_compare_container.py: the stage 4 template that serves the swapped weights with a
container package. The argv edit is tested in-process; the whole script runs against a fake
`tt-model` and a fake `docker` (tests/fake_tt_model.py, tests/fake_docker.py) that start
tests/fake_swap_server.py, which opens no device."""
import importlib.util
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
