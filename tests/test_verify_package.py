# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard verify-package` (issue 24). The Hub is a fake that serves a directory, so nothing here uses the network or a device."""
from __future__ import annotations

import hashlib
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from orchard import verify_package as vp

REV = "d18cca4c121edb87d022116d281ce212c9136f57"
SHA = "1ed61f14ae051d889f6ca23c783a68b6a347900a"
REPO = "someone/thing-blackhole"
CARD = f"""---
license: apache-2.0
base_model: ibm/thing
tags:
- blackhole
---
# thing
Runs ibm/thing (revision `{REV}`). The weights are not in this repository.
"""


def h(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeHub(vp.Hub):
    """Serves `files` ({path: bytes}); `lfs` names the paths the Hub records a sha256 for."""

    def __init__(self, files, lfs=(), private=False, exists=True, tamper=None):
        self.files, self.lfs, self.private, self.exists, self.tamper = files, set(lfs), private, exists, tamper or {}
        self.calls = []

    def info(self, repo, revision=None):
        self.calls.append(("info", repo, revision))
        return vp.RepoInfo(SHA, self.private, [vp.RepoFile(p, len(b), h(b) if p in self.lfs else None)
                                               for p, b in self.files.items()])

    def download(self, repo, revision, dest, paths):
        self.calls.append(("download", repo, revision))
        assert revision == SHA, "the pull is pinned to the commit info returned"
        for p in paths:
            f = Path(dest) / p
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_bytes(self.tamper.get(p, self.files[p]))

    def revision_exists(self, repo, revision):
        self.calls.append(("exists", repo, revision))
        return self.exists


def good_files():
    files = {"README.md": CARD.encode(), "LICENSE": b"Apache License", "run.sh": b"#!/bin/sh\nexec python serve.py\n",
             "serve.py": b"print('hi')\n"}
    rec = {"package": "thing", "model": "ibm/thing", "revision": REV,
           "files": [{"path": p, "sha256": h(b)} for p, b in sorted(files.items())]}
    files["package-record.json"] = json.dumps(rec).encode()
    return files


def run(hub, tmp_path, **kw):
    args = SimpleNamespace(repo_id=REPO, scratch=str(tmp_path / "s"), staged=kw.pop("staged", None), revision=None,
                           plan=kw.pop("plan", False), check=kw.pop("check", False))
    buf = io.StringIO()
    code = vp.main(args, out=buf, hub=hub, hostname="testhost", home="/home/nobody-here")
    return code, buf.getvalue()


def line(text, name):
    return [l for l in text.splitlines() if l.strip().split(":")[0].split()[-1:] == [name]][0]


def test_a_good_package_passes_every_check_and_prints_a_boot_command(tmp_path):
    code, text = run(FakeHub(good_files()), tmp_path, check=True)
    assert code == 0, text
    assert "FAIL" not in text
    for name in ("hub-hashes", "scrub", "license", "record", "manifests", "revision"):
        assert line(text, name).strip().startswith("ok"), (name, text)
    assert f"gozer run --chips 1 --who \"claude:verify-package\" --reason \"boot check of {REPO}@{SHA[:12]}\" -- bash run.sh" in text
    assert "never opens a device" in text


def test_the_pull_is_pinned_to_the_commit_the_hub_reported(tmp_path):
    hub = FakeHub(good_files())
    run(hub, tmp_path)
    assert [c[0] for c in hub.calls][:2] == ["info", "download"]


def test_plan_pulls_nothing(tmp_path):
    hub = FakeHub(good_files())
    code, text = run(hub, tmp_path, plan=True)
    assert code == 0 and "nothing was pulled" in text
    assert all(c[0] != "download" for c in hub.calls) and not (tmp_path / "s").exists()


def test_a_pulled_file_that_differs_from_the_hubs_sha256_fails(tmp_path):
    files = good_files()
    files["big.bin.whl"] = b"wheel"
    hub = FakeHub(files, lfs=["big.bin.whl"], tamper={"big.bin.whl": b"wheeL"})
    code, text = run(hub, tmp_path, check=True)
    assert code == 1 and "FAIL" in line(text, "hub-hashes") and "big.bin.whl" in line(text, "hub-hashes")


def test_a_weights_file_fails_the_scrub(tmp_path):
    files = good_files()
    files["model.safetensors"] = b"x"
    code, text = run(FakeHub(files), tmp_path, check=True)
    assert code == 1 and "model.safetensors" in line(text, "scrub") and "weights" in line(text, "scrub")


def test_a_home_path_or_token_fails_the_scrub_and_the_token_is_not_echoed(tmp_path):
    files = good_files()
    secret = "hf_" + "a" * 34
    files["notes.txt"] = f"see /home/someone/x and {secret}".encode()
    code, text = run(FakeHub(files), tmp_path, check=True)
    assert code == 1 and "home path" in line(text, "scrub") and "token" in line(text, "scrub")
    assert secret not in text


def test_a_package_without_a_license_warns(tmp_path):
    files = good_files()
    del files["LICENSE"]
    files["README.md"] = CARD.replace("license: apache-2.0\n", "").encode()
    files.pop("package-record.json")
    _, text = run(FakeHub(files), tmp_path)
    assert line(text, "license").strip().startswith("warn")


def test_a_recorded_hash_that_no_longer_matches_fails(tmp_path):
    files = good_files()
    files["serve.py"] = b"print('changed')\n"           # the record still holds the old hash
    code, text = run(FakeHub(files), tmp_path, check=True)
    assert code == 1 and "serve.py differs from its recorded sha256" in line(text, "record")


def test_a_file_missing_from_the_record_fails_and_a_recorded_file_that_is_absent_fails(tmp_path):
    files = good_files()
    files["extra.txt"] = b"x"
    del files["serve.py"]
    _, text = run(FakeHub(files), tmp_path)
    assert "serve.py is recorded and absent" in line(text, "record") and "not in the record: extra.txt" in line(text, "record")


def test_sha256sums_is_read_when_there_is_no_package_record(tmp_path):
    files = good_files()
    files.pop("package-record.json")
    files["SHA256SUMS"] = "".join(f"{h(b)}  {p}\n" for p, b in sorted(files.items())).encode()
    _, text = run(FakeHub(files), tmp_path)
    assert line(text, "record").strip().startswith("ok") and "SHA256SUMS" in line(text, "record")


def test_no_record_at_all_is_a_warning(tmp_path):
    files = good_files()
    files.pop("package-record.json")
    _, text = run(FakeHub(files), tmp_path)
    assert line(text, "record").strip().startswith("warn")


def test_a_record_revision_that_the_card_does_not_name_fails(tmp_path):
    files = good_files()
    rec = json.loads(files["package-record.json"])
    rec["revision"] = "0" * 40
    files["package-record.json"] = json.dumps(rec).encode()
    code, text = run(FakeHub(files), tmp_path, check=True)
    assert code == 1 and "FAIL" in line(text, "revision")


def test_a_card_that_names_no_revision_warns(tmp_path):
    files = good_files()
    files["README.md"] = b"---\nlicense: mit\n---\nno revision here\n"
    _, text = run(FakeHub(files), tmp_path)
    assert line(text, "revision").strip().startswith("warn")


def test_a_weights_revision_the_hub_does_not_have_fails(tmp_path):
    code, text = run(FakeHub(good_files(), exists=False), tmp_path, check=True)
    assert code == 1 and "has no revision" in line(text, "revision")


def test_a_hub_that_cannot_answer_about_the_revision_only_warns(tmp_path):
    _, text = run(FakeHub(good_files(), exists=None), tmp_path)
    assert line(text, "revision").strip().startswith("warn") and "could not ask" in line(text, "revision")


def test_a_manifest_naming_another_repository_fails(tmp_path):
    files = good_files()
    files["tt-model.yaml"] = f"schema: \"5.1\"\nrepo: other/place\nweights:\n  repo: ibm/thing\n  revision: \"{REV}\"\n".encode()
    files.pop("package-record.json")
    _, text = run(FakeHub(files), tmp_path)
    assert "FAIL" in line(text, "manifests") and "other/place" in line(text, "manifests")


def test_a_broken_json_manifest_fails(tmp_path):
    files = good_files()
    files["tt_kernel_manifest.json"] = b"{not json"
    files.pop("package-record.json")
    _, text = run(FakeHub(files), tmp_path)
    assert "FAIL" in line(text, "manifests") and "tt_kernel_manifest.json" in line(text, "manifests")


def test_the_chip_count_comes_from_the_manifest_or_the_card_tag(tmp_path):
    files = good_files()
    files["README.md"] = CARD.replace("- blackhole", "- blackhole\n- p150x2").encode()
    files.pop("package-record.json")
    files["install.sh"] = b"echo hi\n"
    _, text = run(FakeHub(files), tmp_path)
    assert "--chips 2" in text and "bash install.sh" in text
    files2 = good_files()
    files2["tt-model.yaml"] = b"serve_profiles:\n  - name: p\n    mesh_device: P300\n"
    files2.pop("package-record.json")
    _, text2 = run(FakeHub(files2), tmp_path / "again")
    assert "--chips 2" in text2


def test_staged_comparison_names_each_kind_of_difference(tmp_path):
    files = good_files()
    staged = tmp_path / "staged"
    for p, b in files.items():
        (staged / p).parent.mkdir(parents=True, exist_ok=True)
        (staged / p).write_bytes(b)
    _, same = run(FakeHub(files), tmp_path, staged=str(staged))
    assert line(same, "staged").strip().startswith("ok")
    (staged / "serve.py").write_bytes(b"other")
    (staged / "only-local.txt").write_bytes(b"x")
    (staged / "run.sh").unlink()
    code, text = run(FakeHub(files), tmp_path / "two", staged=str(staged), check=True)
    ln = line(text, "staged")
    assert code == 1 and "only on the Hub: run.sh" in ln and "only in the staged directory: only-local.txt" in ln \
        and "different bytes: serve.py" in ln


def test_without_check_a_failure_still_exits_zero_and_with_it_exits_one(tmp_path):
    files = good_files()
    files["model.safetensors"] = b"x"
    assert run(FakeHub(files), tmp_path)[0] == 0
    assert run(FakeHub(files), tmp_path / "b", check=True)[0] == 1


def test_a_non_empty_scratch_directory_is_refused(tmp_path):
    (tmp_path / "s").mkdir()
    (tmp_path / "s" / "old").write_text("x")
    assert run(FakeHub(good_files()), tmp_path)[0] == 2


def test_a_bad_repo_id_and_a_missing_staged_directory_are_refused(tmp_path):
    buf = io.StringIO()
    assert vp.main(SimpleNamespace(repo_id="no-slash", scratch=None, staged=None, revision=None, plan=False, check=False),
                   out=buf, hub=FakeHub({})) == 2
    assert run(FakeHub(good_files()), tmp_path, staged=str(tmp_path / "nope"))[0] == 2


def test_a_hub_failure_is_a_refusal_and_never_echoes_the_token(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("HF_TOKEN", "hf_" + "b" * 34)

    class Down(FakeHub):
        def info(self, repo, revision=None):
            raise RuntimeError("401 for token hf_" + "b" * 34)
    code, _ = run(Down({}), tmp_path)
    err = capsys.readouterr().err
    assert code == 2 and "hf_" + "b" * 34 not in err and "<token>" in err


def test_a_token_that_crosses_the_truncation_boundary_is_not_partly_echoed(tmp_path, monkeypatch, capsys):
    token = "hf_" + "c" * 34
    monkeypatch.setenv("HF_TOKEN", token)

    class Down(FakeHub):
        def info(self, repo, revision=None):
            raise RuntimeError("x" * 185 + token)          # the token straddles character 200 of the message
    code, _ = run(Down({}), tmp_path)
    err = capsys.readouterr().err
    assert code == 2 and "hf_" + "c" * 10 not in err and "<token>" in err


def test_a_scratch_path_that_is_a_file_is_refused_not_a_traceback(tmp_path):
    a_file = tmp_path / "scratch-is-a-file"
    a_file.write_text("x")
    args = SimpleNamespace(repo_id=REPO, scratch=str(a_file), staged=None, revision=None, plan=False, check=False)
    assert vp.main(args, out=io.StringIO(), hub=FakeHub(good_files()), hostname="testhost", home="/home/nobody-here") == 2


def test_the_command_never_starts_a_process(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("verify-package started a process"))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("verify-package started a process"))
    assert run(FakeHub(good_files()), tmp_path)[0] == 0


def test_the_cli_registers_the_command_and_its_flags():
    from orchard import cli
    ns = cli._parser().parse_args(["verify-package", REPO, "--scratch", "d", "--staged", "s", "--plan", "--check", "--revision", "r"])
    assert (ns.cmd, ns.repo_id, ns.scratch, ns.staged, ns.plan, ns.check, ns.revision) == ("verify-package", REPO, "d", "s", True, True, "r")


def test_a_token_inside_printed_text_is_replaced(tmp_path, monkeypatch):
    secret = "hf_" + "c" * 34
    monkeypatch.setenv("HF_TOKEN", secret)
    files = good_files()
    files[f"{secret}.safetensors"] = b"x"          # the scrub prints the path, which here carries the token
    _, text = run(FakeHub(files), tmp_path)
    assert secret not in text and "<token>" in text


def test_the_download_librarys_bookkeeping_directory_is_not_judged_as_package_content(tmp_path):
    """huggingface_hub writes .cache/huggingface under local_dir. It is the pull's own bookkeeping, so it is removed before the checks."""
    class Bookkeeping(FakeHub):
        def download(self, repo, revision, dest, paths):
            super().download(repo, revision, dest, paths)
            (Path(dest) / ".cache" / "huggingface").mkdir(parents=True)
            (Path(dest) / ".cache" / "huggingface" / "x.metadata").write_text("etag")
    code, text = run(Bookkeeping(good_files()), tmp_path, check=True)
    assert code == 0 and line(text, "scrub").strip().startswith("ok") and not (tmp_path / "s" / ".cache").exists()


def test_record_with_a_path_to_sha256_map_is_read(tmp_path):
    """The Qwen3.5 recipe package writes `files` as {path: sha256}; every file in it must count as recorded."""
    import hashlib
    from orchard import verify_package as vp
    (tmp_path / "a.txt").write_text("x")
    (tmp_path / "package-record.json").write_text(json.dumps(
        {"files": {"a.txt": hashlib.sha256(b"x").hexdigest()}, "revision": "r"}))
    name, listed, rev = vp._read_record(tmp_path)
    assert listed == {"a.txt": hashlib.sha256(b"x").hexdigest()} and rev == "r"
    assert vp.check_record(tmp_path, (name, listed, rev)).status == vp.OK


def test_a_revision_named_next_to_another_repository_is_checked_against_that_repository(tmp_path):
    """A card can pin the weights and a second repository (a code bundle): each commit is looked up in the repository named just before it."""
    files = good_files()
    other = "b" * 40
    files["README.md"] = (CARD + f"\nCode: [org/bundle](https://huggingface.co/org/bundle), revision `{other}`.\n").encode()
    rec = json.loads(files["package-record.json"])
    rec["files"] = [{"path": p, "sha256": h(b)} for p, b in sorted(files.items()) if p != "package-record.json"]
    files["package-record.json"] = json.dumps(rec).encode()
    hub = FakeHub(files)
    run(hub, tmp_path, check=True)
    assert ("exists", "org/bundle", other) in hub.calls and ("exists", "ibm/thing", other) not in hub.calls


# ---- issue 251: pins in manifests and launchers, graft pins, scratch paths with spaces, the cache directory

OTHER = "b" * 40


def _manifest(**weights):
    return json.dumps({"weights": {"repo_id": "ibm/thing", "revision": REV, **weights}}).encode()


def test_a_kernel_manifest_that_pins_the_cards_revision_passes(tmp_path):
    files = good_files()
    files["tt_kernel_manifest.json"] = _manifest()
    files.pop("package-record.json")
    _, text = run(FakeHub(files), tmp_path)
    assert line(text, "manifests").strip().startswith("ok"), text


@pytest.mark.parametrize("weights,word", [
    ({"revision": "main"}, "40-character"),
    ({"revision": OTHER}, "card"),
    ({"repo_id": "not a repo"}, "repo_id"),
    ({"repo_id": "someone/else"}, "someone/else"),
])
def test_a_kernel_manifest_whose_weights_pin_is_not_the_cards_fails(tmp_path, weights, word):
    files = good_files()
    files["tt_kernel_manifest.json"] = _manifest(**weights)
    files.pop("package-record.json")
    _, text = run(FakeHub(files), tmp_path)
    assert "FAIL" in line(text, "manifests") and word in line(text, "manifests"), text


def test_a_kernel_manifest_with_no_weights_pin_fails(tmp_path):
    files = good_files()
    files["tt_kernel_manifest.json"] = json.dumps({"weights": {}}).encode()
    files.pop("package-record.json")
    _, text = run(FakeHub(files), tmp_path)
    assert "FAIL" in line(text, "manifests") and "weights" in line(text, "manifests"), text


def test_a_tt_model_yaml_weights_revision_the_card_does_not_name_fails(tmp_path):
    files = good_files()
    files["tt-model.yaml"] = f"schema: \"5.1\"\nrepo: {REPO}\nweights:\n  repo: ibm/thing\n  revision: \"{OTHER}\"\n".encode()
    files.pop("package-record.json")
    _, text = run(FakeHub(files), tmp_path)
    assert "FAIL" in line(text, "manifests") and OTHER in line(text, "manifests"), text


def test_a_launcher_that_pins_other_weights_fails_and_an_auxiliary_pin_is_only_a_note(tmp_path):
    files = good_files()
    files["run.sh"] = f"#!/bin/sh\nWEIGHTS_REVISION={OTHER}\nexec python serve.py\n".encode()
    files.pop("package-record.json")
    _, text = run(FakeHub(files), tmp_path)
    assert "FAIL" in line(text, "revision") and OTHER in line(text, "revision"), text
    files["run.sh"] = f"#!/bin/sh\nCODE_REVISION={OTHER}\nexec python serve.py\n".encode()
    _, text = run(FakeHub(files), tmp_path / "again")
    assert not line(text, "revision").strip().startswith("FAIL"), text
    assert OTHER in line(text, "revision")


def test_a_graft_pin_in_the_card_is_checked_against_the_repository_graft_json_names(tmp_path):
    files = good_files()
    files["README.md"] = (CARD + f"The MTP head is taken at revision `{OTHER}`.\n").encode()
    files["graft.json"] = json.dumps({"schema": 1, "repo": "ibm/thing-nearest", "revision": OTHER}).encode()
    files.pop("package-record.json")
    asked = []

    class Hub(FakeHub):
        def revision_exists(self, repo, revision):
            asked.append((repo, revision))
            return repo == "ibm/thing-nearest" or revision == REV
    run(Hub(files), tmp_path)
    assert ("ibm/thing-nearest", OTHER) in asked and ("ibm/thing", OTHER) not in asked, asked


def test_the_printed_boot_instructions_quote_a_scratch_path_with_spaces(tmp_path):
    scratch = tmp_path / "has some spaces" / "s"
    args = SimpleNamespace(repo_id=REPO, scratch=str(scratch), staged=None, revision=None, plan=False, check=False)
    buf = io.StringIO()
    vp.main(args, out=buf, hub=FakeHub(good_files()), hostname="testhost", home="/home/nobody-here")
    cd = [l for l in buf.getvalue().splitlines() if l.startswith("cd ")][0]
    assert cd == f"cd -- '{scratch}'", cd


def test_a_file_the_package_publishes_under_dot_cache_is_not_deleted_before_the_checks(tmp_path):
    files = good_files()
    files[".cache/notes.txt"] = b"published by the package"
    files.pop("package-record.json")
    _, text = run(FakeHub(files), tmp_path)
    assert (tmp_path / "s" / ".cache" / "notes.txt").is_file()
    assert "FAIL" in line(text, "scrub") and ".cache" in line(text, "scrub"), text
