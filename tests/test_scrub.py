# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The bundle scrub check (spec section 10)."""
import pytest

from orchard.scrub import scrub_bundle, scrub_package, scrub_text

HOST, HOME = "quietbox-7", "/home/alice"


def test_clean_text_has_no_hits():
    assert scrub_text("Hemmingway-1 decodes at 80 tok/s on 2 chips.", hostname=HOST, home=HOME) == []


@pytest.mark.parametrize("text, what", [
    ("served from quietbox-7 last night", "the hostname 'quietbox-7'"),
    ("token hf_" + "a" * 34, "a Hugging Face token"),
    ("token ghp_" + "b" * 36, "a GitHub token"),
    ("token github_pat_" + "c" * 50, "a GitHub token"),
    ("-----BEGIN OPENSSH PRIVATE KEY-----", "a private key"),
    ("weights in /home/alice/models", "an absolute home path"),
    ("weights in /home/bob/models", "an absolute home path"),
    ("cache in /root/.cache", "an absolute home path"),
])
def test_each_kind_of_leak_is_found(text, what):
    assert scrub_text(text, hostname=HOST, home=HOME) == [what]


def test_a_hostname_inside_a_longer_name_is_not_a_hit():
    assert scrub_text("quietbox-70 is another machine", hostname=HOST, home=HOME) == []


def test_the_bundle_scrub_names_the_file_and_skips_the_operator_ledger(tmp_path):
    (tmp_path / "card.md").write_text("Built on quietbox-7.")
    (tmp_path / "ledger.jsonl").write_text('{"path": "/home/alice/run/evidence/x.txt"}\n')
    assert scrub_bundle(tmp_path, hostname=HOST, home=HOME) == ["card.md: the hostname 'quietbox-7'"]


# ---- the package scrub (plan 5, stage 7) ---------------------------------------------------------


def package_dir(root):
    """A clean staged thin bundle: text files, a wheel and a card that names the operator's repo."""
    root.mkdir()
    (root / "run.sh").write_text('HERE="$(pwd)"\nexport HF_MODEL="$HERE/model-dir"\n')
    (root / "tt_kernel_manifest.json").write_text('{"producer": {"hostname": "redacted"}}\n')
    (root / "README.md").write_text("Serve it: tt-model serve episod/hemmingway-1-p300\n")
    (root / "wheels").mkdir()
    (root / "wheels" / "ttnn-0.79.0-cp312-cp312-linux_x86_64.whl").write_bytes(
        b"PK\x03\x04 built in /home/alice/tt-metal on quietbox-7")
    return root


def test_a_clean_package_has_no_hits_and_wheel_contents_are_not_read(tmp_path):
    root = package_dir(tmp_path / "pkg")
    assert scrub_package(root, hostname=HOST, home=HOME, namespace="episod") == []


@pytest.mark.parametrize("rel, what", [
    (".tt_cache/layer0.tensorbin", "a tensor cache directory"),
    ("tensors/layer0.bin", "a tensor cache directory"),
    ("venv/bin/python", "an installed venv"),
    (".python/cpython/bin/python3", "an installed interpreter"),
    (".hf/hub/x", "a Hugging Face cache"),
    ("model-dir/config.json", "a built model directory"),
    ("layer0.tensorbin", "a tensor cache file"),
    ("model-00001-of-00002.safetensors", "a weights file"),
    ("pytorch_model.bin", "a weights or cache file"),
])
def test_caches_weights_and_install_output_are_hits(tmp_path, rel, what):
    root = package_dir(tmp_path / "pkg")
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text("x")
    assert scrub_package(root, hostname=HOST, home=HOME, namespace="episod") == [
        f"{rel.split('/')[0]}: {what}"]


def test_a_symbolic_link_is_a_hit(tmp_path):
    root = package_dir(tmp_path / "pkg")
    (root / "config.json").symlink_to(tmp_path / "elsewhere.json")
    assert scrub_package(root, hostname=HOST, home=HOME, namespace="episod") == [
        "config.json: a symbolic link"]


def test_the_hostname_tokens_and_home_paths_are_found_in_text_files(tmp_path):
    root = package_dir(tmp_path / "pkg")
    (root / "tt_kernel_manifest.json").write_text('{"producer": {"hostname": "quietbox-7"}}\n')
    (root / "run.sh").write_text("export HF_HOME=/home/alice/.cache/huggingface\n")
    assert scrub_package(root, hostname=HOST, home=HOME, namespace="episod") == [
        "run.sh: an absolute home path", "tt_kernel_manifest.json: the hostname 'quietbox-7'"]


def test_the_operator_namespace_is_allowed_only_in_the_card(tmp_path):
    root = package_dir(tmp_path / "pkg")
    (root / "run.sh").write_text('export HF_MODEL="episod/hemmingway-1-p300"\n')
    assert scrub_package(root, hostname=HOST, home=HOME, namespace="episod") == [
        "run.sh: the operator's namespace 'episod' outside README.md"]
