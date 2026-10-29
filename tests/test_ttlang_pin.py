# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The tt-lang pin record (orchard/ttlang_pin.py), its consistency check (orchard/ttlang_consistency.py) and the
reference assembly that follows it (orchard/ttlang_reference.py). No test touches the operator's checkout, venv, skill
directory or tt-metal tree: every directory is made up in tmp_path and every command is a fake."""
from __future__ import annotations

import argparse
import hashlib
import pytest
from pathlib import Path

from orchard import defaults, ttlang_consistency as tc, ttlang_env, ttlang_pin, ttlang_reference as tr

PIN = ttlang_pin.SHA


def spec(version: str) -> str:
    return ("# TT-Lang specification\n\n## Specification Versions\n\n| *Version* | *Date* | *Description* |\n"
            f"| :---- | :---- | :---- |\n| 0.1 | 2026-01-01 | first |\n| {version} | 2026-10-01 | latest |\n\n## Intro\n")


def old_spec(version: str) -> str:
    return spec(version).replace("## Specification Versions", "## 0. Versions")


def make_checkout(root: Path, version: str = "0.20") -> Path:
    src = root / "src"
    (src / "examples").mkdir(parents=True)
    (src / "examples" / "eltwise_add.py").write_text("# checkout eltwise_add\n")
    (src / "examples" / "eltwise_broadcast_reduce.py").write_text("# checkout reduce\n")
    (src / "examples" / "README.md").write_text("# checkout examples readme\n")
    (src / "docs" / "sphinx" / "specs").mkdir(parents=True)
    (src / "docs" / "sphinx" / "specs" / "TTLangSpecification.md").write_text(spec(version))
    for rel in tr.GUIDE_DOCS + tr.EXAMPLES_DOCS:
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"# {rel}\nbody of {rel}\n")
    for rel in [f"examples/{r}" for r, _ in defaults.TTLANG_NEGATIVE_EXAMPLES] + [
            f"examples/{n}" for n in ttlang_env.VERIFY_SUPPORT_FILES]:     # the programs verify runs
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(f"# {rel}\n")
    (src / ".git").mkdir()
    return src


def make_skill(root: Path, version: str = "0.12", name: str = "skill") -> Path:
    d = root / name
    d.mkdir()
    (d / "guide.md").write_text("# skill guide\n")
    (d / "examples.md").write_text("# skill examples\n")
    (d / "TTLangSpecification.md").write_text(old_spec(version))
    return d


class Fake:
    """A KEnv whose commands are answered from a table: HEAD of the checkout, the simulator version, the tt-metal tree."""

    def __init__(self, tmp_path, *, head=PIN, sim=f"tt-lang-sim {ttlang_pin.SIM_VERSION}\n", version="0.20"):
        self.src = make_checkout(tmp_path, version)
        venv = tmp_path / "venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "bin" / "tt-lang-sim").write_text("#!/bin/sh\n")
        self.head, self.sim, self.tree_ref, self.calls = head, sim, "v0.80.0", []
        self.untracked, self.modified, self.git_works, self.envs = set(), set(), True, []
        self.env = ttlang_env.KEnv(src_dir=self.src, venv_dir=venv, scratch_dir=tmp_path / "scratch", run=self.run)

    def _tree(self):
        return sorted(p.relative_to(self.src).as_posix() for d in ("examples", "docs")
                      for p in (self.src / d).rglob("*") if p.is_file())

    def _tracked(self):
        """Paths the fake pinned commit has: what is on disk except `untracked`, and the negative examples verify runs
        (absent from the fake checkout, so they count as deleted, which does not block verify)."""
        needed = [f"examples/{rel}" for rel, _ in defaults.TTLANG_NEGATIVE_EXAMPLES]
        needed += [f"examples/{n}" for n in ttlang_env.VERIFY_SUPPORT_FILES]
        return sorted({p for p in self._tree() if p not in self.untracked} | set(needed))

    def _blob(self, rel):
        """The blob id the fake commit holds for `rel`: the disk content's, or another one for a `modified` path."""
        path = self.src / rel
        data = path.read_bytes() if path.is_file() else b"absent"
        if rel in self.modified:
            data = b"original:" + data
        return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()

    def run(self, argv, timeout, env=None, cwd=None):
        self.calls.append(list(argv))
        self.envs.append(env)
        if "ls-tree" in argv:               # what the pinned commit has: the files on disk, less the untracked ones
            if not self.git_works:
                return 128, "", "fatal: not a valid object name"
            return 0, "".join(f"100644 blob {self._blob(p)}\t{p}\0" for p in self._tracked()), ""
        if "cat-file" in argv and self.git_works:           # a file as the pinned commit has it
            return 0, "# the pinned commit's program\n", ""
        if "cat-file" in argv:
            return 128, "", "fatal"
        if argv[-1] == "--version":
            return 0, self.sim, ""
        if "rev-parse" in argv:
            return 0, self.head + "\n", ""
        if "describe" in argv:
            return 0, self.tree_ref + "\n", ""
        return 1, "", "not faked"


# ---- the pin record ------------------------------------------------------------------------------------------------

def test_the_pin_is_a_full_sha_and_the_simulator_version_follows_from_it():
    assert len(ttlang_pin.SHA) == 40 and set(ttlang_pin.SHA) <= set("0123456789abcdef")
    assert ttlang_pin.SIM_SOURCE_SHA == ttlang_pin.SHA
    assert ttlang_pin.SIM_VERSION == f"{ttlang_pin.VERSION_TAG}+g{ttlang_pin.SHA[:7]}"


def test_defaults_and_the_environment_builder_read_the_one_record():
    assert defaults.TTLANG_PIN == ttlang_pin.SHA and defaults.TTLANG_VERSION_TAG == ttlang_pin.VERSION_TAG
    assert defaults.TTLANG_URL == ttlang_pin.URL and defaults.TTLANG_DIAGNOSTICS_SHA256 == ttlang_pin.DIAGNOSTICS_SHA256
    assert ttlang_env.expected_version() == ttlang_pin.SIM_VERSION


def test_the_tt_metal_ref_is_left_open_until_issue_95_is_settled():
    assert ttlang_pin.TT_METAL_REF is None and "issue 95" in ttlang_pin.TT_METAL_NOTE


def test_the_spec_version_comes_from_the_checkout_not_the_pin_file(tmp_path):
    assert tc.checkout_spec_version(make_checkout(tmp_path, "0.31")) == "0.31"
    assert not hasattr(ttlang_pin, "SPEC_VERSION")


# ---- reference assembly --------------------------------------------------------------------------------------------

def by_name(files):
    return {f["name"]: f for f in files}


def test_equal_versions_ship_the_skills_guide_and_examples(tmp_path):
    f = Fake(tmp_path)
    files, problems = tr.assemble(f.env, make_skill(tmp_path, "0.20"))
    assert problems == []
    b = by_name(files)
    assert b["guide.md"]["source"] == ttlang_env.SKILL_SOURCE and b["guide.md"]["data"] == b"# skill guide\n"
    assert b["examples.md"]["source"] == ttlang_env.SKILL_SOURCE
    assert all(r["spec_version"] == "0.20" for r in files)
    assert not any(r["generated"] for r in files)


def test_differing_versions_build_the_guide_and_examples_from_the_checkout(tmp_path):
    f = Fake(tmp_path)
    files, problems = tr.assemble(f.env, make_skill(tmp_path, "0.12"))
    assert problems == []
    b = by_name(files)
    for name in ("guide.md", "examples.md"):
        r = b[name]
        assert r["source"] == ttlang_env.CHECKOUT_SOURCE and r["generated"] is True
        assert r["spec_version"] == "0.20" and r["sha256"] == hashlib.sha256(r["data"]).hexdigest()
        assert b"# skill guide" not in r["data"] and b"# skill examples" not in r["data"]
        assert r["generated_from"] and all(p.startswith(("docs/", "examples/")) for p in r["generated_from"])
    assert b"body of docs/sphinx/programming-guide.md" in b["guide.md"]["data"]
    assert b"body of docs/EXAMPLES.md" in b["examples.md"]["data"]
    assert b["guide.md"]["data"].startswith(b"<!-- generated by tt-orchard")
    assert [r["name"] for r in files] == ["guide.md", "examples.md", "TTLangSpecification.md", "eltwise_add.py",
                                          "eltwise_broadcast_reduce.py"]


def test_the_generated_guide_says_where_it_came_from_and_which_version_it_documents(tmp_path):
    f = Fake(tmp_path)
    files, _ = tr.assemble(f.env, make_skill(tmp_path, "0.12"))
    head = by_name(files)["guide.md"]["data"].decode().split("-->")[0]
    assert "0.20" in head and "docs/sphinx/programming-guide.md" in head and "skill documents 0.12" in head
    assert PIN in head


def test_an_unknown_version_keeps_the_skills_files_and_says_it_is_unknown(tmp_path):
    f = Fake(tmp_path)
    skill = make_skill(tmp_path, "0.12")
    (skill / "TTLangSpecification.md").write_text("# no versions table\n")
    files, problems = tr.assemble(f.env, skill)
    assert problems == []
    g = by_name(files)["guide.md"]
    assert g["source"] == ttlang_env.SKILL_SOURCE and g["generated"] is False and g["spec_version"] is None


def test_a_missing_checkout_doc_is_a_problem_that_names_it(tmp_path):
    f = Fake(tmp_path)
    (f.src / tr.GUIDE_DOCS[0]).unlink()
    _, problems = tr.assemble(f.env, make_skill(tmp_path, "0.12"))
    assert any(tr.GUIDE_DOCS[0] in p and "does not exist" in p for p in problems)


def test_a_linked_checkout_doc_is_refused_and_not_followed(tmp_path):
    f = Fake(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("token=abc\n")
    victim = f.src / tr.GUIDE_DOCS[0]
    victim.unlink()
    victim.symlink_to(secret)
    files, problems = tr.assemble(f.env, make_skill(tmp_path, "0.12"))
    assert any("is a link" in p for p in problems)
    assert all(b"token=abc" not in r["data"] for r in files)


def test_assembly_never_writes_into_the_checkout_or_the_skill(tmp_path):
    f = Fake(tmp_path)
    skill = make_skill(tmp_path, "0.12")

    def snapshot():
        return sorted((str(p), p.stat().st_mtime_ns) for d in (f.src, skill) for p in d.rglob("*"))
    before = snapshot()
    tr.assemble(f.env, skill)
    assert before == snapshot()


def test_the_checkouts_other_eltwise_broadcast_and_reduce_examples_go_in_as_full_files(tmp_path):
    f = Fake(tmp_path)
    ex = f.src / "examples"
    for name in ("broadcast.py", "general_broadcast.py", "eltwise_pipe.py", "my_reduce_sum.py", "matmul_1d.py", "multidevice_all_reduce.py"):
        (ex / name).write_text(f"# full text of {name}\n")
    (ex / "eltwise_dir").mkdir()                       # a directory is not an example file
    files, problems = tr.assemble(f.env, make_skill(tmp_path, "0.12"))
    assert problems == []
    names = [r["name"] for r in files]
    assert names[:5] == ["guide.md", "examples.md", "TTLangSpecification.md", "eltwise_add.py",
                         "eltwise_broadcast_reduce.py"]
    assert names[5:] == ["broadcast.py", "eltwise_pipe.py", "general_broadcast.py", "my_reduce_sum.py"]
    b = by_name(files)
    assert b["broadcast.py"]["data"] == b"# full text of broadcast.py\n"
    assert b["broadcast.py"]["source"] == ttlang_env.CHECKOUT_SOURCE and b["broadcast.py"]["spec_version"] == "0.20"
    assert b["broadcast.py"]["sha256"] == hashlib.sha256(b["broadcast.py"]["data"]).hexdigest()
    assert "matmul_1d.py" not in b and "multidevice_all_reduce.py" not in b


def test_a_linked_extra_example_is_refused_and_not_followed(tmp_path):
    f = Fake(tmp_path)
    secret = tmp_path / "secret.txt"
    secret.write_text("token=abc\n")
    (f.src / "examples" / "broadcast.py").symlink_to(secret)
    files, problems = tr.assemble(f.env, make_skill(tmp_path, "0.20"))
    assert any("is a link" in p for p in problems)
    assert all(b"token=abc" not in r["data"] for r in files)


# ---- report lines --------------------------------------------------------------------------------------------------

def test_the_report_says_plainly_when_guide_and_spec_versions_differ():
    skill_guide = {"name": "reference/guide.md", "source": "tt-lang skill", "path": "/s/guide.md", "sha256": "a" * 64,
                   "spec_version": "0.12", "generated": False}
    spec_ref = {"name": "reference/TTLangSpecification.md", "source": "tt-lang checkout", "path": "/c/spec.md",
                "sha256": "b" * 64, "spec_version": "0.20", "generated": False}
    text = "\n".join(tr.report_lines({"references": [skill_guide, spec_ref]}))
    assert "guide.md" in text and "spec version 0.12" in text and "sha256 " + "a" * 64 in text
    assert "The guide documents specification 0.12 and the specification documents 0.20" in text
    consistent = [dict(skill_guide, spec_version="0.20"), spec_ref]
    assert "The guide and the specification both document 0.20" in "\n".join(tr.report_lines({"references": consistent}))


def test_the_report_names_a_generated_file_and_what_it_was_built_from():
    gen = {"name": "reference/guide.md", "source": "tt-lang checkout", "path": "generated", "sha256": "c" * 64,
           "spec_version": "0.20", "generated": True, "generated_from": ["docs/sphinx/programming-guide.md"]}
    assert "generated from docs/sphinx/programming-guide.md" in "\n".join(tr.report_lines({"references": [gen]}))


# ---- consistency rows ----------------------------------------------------------------------------------------------

def rows_by_name(rows):
    return {r.name: r for r in rows}


def check(f, skill, tree=None, **kw):
    return tc.check(f.env, skill, tt_metal_dir=tree, **kw)


def test_a_consistent_machine_has_no_blocking_row(tmp_path):
    f = Fake(tmp_path)
    rows = rows_by_name(check(f, make_skill(tmp_path, "0.20")))
    assert {n: r.status for n, r in rows.items() if n != "hardware tree"} == {n: tc.OK for n in rows if n != "hardware tree"}
    assert rows["hardware tree"].status == tc.TODO and "issue 95" in rows["hardware tree"].detail
    assert tc.blocking(list(rows.values())) == []


def test_a_checkout_at_another_commit_blocks_and_the_fix_is_one_line(tmp_path):
    f = Fake(tmp_path, head="d" * 40)
    r = rows_by_name(check(f, make_skill(tmp_path)))["checkout"]
    assert r.status == tc.MISMATCH and r.blocks and "d" * 9 in r.detail and PIN[:9] in r.detail
    assert "\n" not in r.fix and "kernel-env" in r.fix


def test_a_simulator_built_from_another_commit_blocks(tmp_path):
    f = Fake(tmp_path, sim="tt-lang-sim 1.1.10+gabcdef0\n")
    r = rows_by_name(check(f, make_skill(tmp_path)))["simulator"]
    assert r.status == tc.MISMATCH and r.blocks and "abcdef0" in r.detail and "kernel-env" in r.fix


def test_a_missing_simulator_blocks(tmp_path):
    f = Fake(tmp_path, sim="")
    (f.env.venv_dir / "bin" / "tt-lang-sim").unlink()
    r = rows_by_name(check(f, make_skill(tmp_path)))["simulator"]
    assert r.status == tc.MISMATCH and r.blocks and "not installed" in r.detail


def test_an_older_skill_is_a_mismatch_that_does_not_block_because_the_probe_builds_from_the_checkout(tmp_path):
    f = Fake(tmp_path)
    r = rows_by_name(check(f, make_skill(tmp_path, "0.12")))["skill"]
    assert r.status == tc.MISMATCH and not r.blocks
    assert "0.12" in r.detail and "0.20" in r.detail and "checkout" in r.fix


def test_the_reference_row_blocks_when_a_shipped_file_documents_another_version(tmp_path):
    f = Fake(tmp_path)
    stale = [{"name": "guide.md", "spec_version": "0.12", "source": "tt-lang skill"},
             {"name": "TTLangSpecification.md", "spec_version": "0.20", "source": "tt-lang checkout"}]
    r = rows_by_name(check(f, make_skill(tmp_path, "0.20"), references=stale))["references"]
    assert r.status == tc.MISMATCH and r.blocks and "guide.md" in r.detail and "0.12" in r.detail


def test_the_reference_row_is_built_from_the_assembly_when_no_references_are_given(tmp_path):
    f = Fake(tmp_path)
    r = rows_by_name(check(f, make_skill(tmp_path, "0.12")))["references"]
    assert r.status == tc.OK and "0.20" in r.detail


def test_the_hardware_tree_row_compares_the_tree_with_the_pinned_ref(tmp_path, monkeypatch):
    f = Fake(tmp_path)
    tree = tmp_path / "tree"
    (tree / ".git").mkdir(parents=True)
    monkeypatch.setattr(ttlang_pin, "TT_METAL_REF", "v0.80.0")
    ok = rows_by_name(check(f, make_skill(tmp_path, "0.20"), tree))["hardware tree"]
    assert ok.status == tc.OK
    f.tree_ref = "v0.81.0"
    bad = rows_by_name(check(f, make_skill(tmp_path, "0.20", "skill2"), tree))["hardware tree"]
    assert bad.status == tc.MISMATCH and bad.blocks and "v0.81.0" in bad.detail and "v0.80.0" in bad.detail


def test_the_tt_metal_tree_is_only_read(tmp_path, monkeypatch):
    f = Fake(tmp_path)
    tree = tmp_path / "tree"
    (tree / ".git").mkdir(parents=True)
    monkeypatch.setattr(ttlang_pin, "TT_METAL_REF", "v0.80.0")
    check(f, make_skill(tmp_path, "0.20"), tree)
    git = [a for a in f.calls if a[0] == "git" and str(tree) in a]
    assert git and all(a[3] in {"describe", "rev-parse"} for a in git)


def test_render_prints_one_line_per_row_with_the_fix_on_a_mismatch(tmp_path):
    f = Fake(tmp_path, head="d" * 40)
    text = "\n".join(tc.render(check(f, make_skill(tmp_path))))
    assert "mismatch" in text and "fix:" in text and "ok" in text


def test_the_command_exits_2_on_a_blocking_row_and_0_otherwise(tmp_path, capsys):
    f = Fake(tmp_path)
    args = argparse.Namespace(src_dir=None, venv_dir=None, tt_lang_skill_dir=str(make_skill(tmp_path, "0.12")),
                              tt_metal_dir=None)
    assert tc.main(args, env=f.env) == 0
    f.head = "e" * 40
    assert tc.main(args, env=f.env) == 2
    assert "mismatch" in capsys.readouterr().out


# ---- Copilot review of PR 198 ----------------------------------------------------------------------------------------

def test_an_unreadable_skill_version_gives_an_unknown_references_row_never_ok(tmp_path):
    f = Fake(tmp_path)
    skill = make_skill(tmp_path, "0.12")
    (skill / "TTLangSpecification.md").write_text("# no versions table\n")
    refs, problems = tr.assemble(f.env, skill)
    assert problems == []
    refs = refs + [{"name": "library_add.py", "source": "orchard library", "spec_version": None}]
    r = rows_by_name(check(f, skill, references=refs))["references"]
    assert r.status == tc.UNKNOWN and not r.blocks and "guide.md" in r.detail and "library_add.py" not in r.detail


def test_the_worked_kernel_library_does_not_make_the_references_row_unknown(tmp_path):
    f = Fake(tmp_path)
    skill = make_skill(tmp_path, "0.20")
    refs, _ = tr.assemble(f.env, skill)
    refs = refs + [{"name": "library_add.py", "source": "orchard library", "spec_version": None}]
    assert rows_by_name(check(f, skill, references=refs))["references"].status == tc.OK


def test_a_check_goes_on_past_a_pin_mismatch_and_a_real_build_stops_at_it(tmp_path):
    f = Fake(tmp_path, head="d" * 40)
    ran = ttlang_env.ensure(f.env, check_only=True, say=lambda s: None, allow_pin_mismatch=True)
    assert ran[0].name == "clone" and ran[0].status == ttlang_env.FAIL and ran[0].pin_mismatch
    assert len(ran) > 1                                # with the override the steps after the mismatch are reported
    assert [r.name for r in ttlang_env.ensure(f.env, check_only=True, say=lambda s: None)] == ["clone"]
    built = ttlang_env.ensure(f.env, check_only=False, say=lambda s: None)
    assert [r.name for r in built] == ["clone"]        # a build never goes on from a checkout at another commit


def test_only_a_pin_mismatch_is_marked_overridable(tmp_path):
    f = Fake(tmp_path)
    assert ttlang_env.step_clone(f.env, True).pin_mismatch is False                       # at the pin
    (tmp_path / "b").mkdir()
    off = Fake(tmp_path / "b", head="d" * 40)
    assert ttlang_env.step_clone(off.env, True).pin_mismatch is True
    gone = ttlang_env.KEnv(src_dir=tmp_path / "nowhere", venv_dir=tmp_path / "v", scratch_dir=tmp_path / "s", run=f.run)
    assert ttlang_env.step_wheel(gone, True).pin_mismatch is False                        # no wheel at all


# ---- Copilot review of PR 198, round 2: what is executed or read from the checkout, and whether the pin has it ----------

def test_the_checkouts_untracked_examples_are_refused_not_given_to_the_coder(tmp_path):
    f = Fake(tmp_path)
    (f.src / "examples" / "broadcast.py").write_text("# tracked\n")
    (f.src / "examples" / "eltwise_scratch.py").write_text("print('operator scratch')\n")
    f.untracked = {"examples/eltwise_scratch.py"}
    files, problems = tr.assemble(f.env, make_skill(tmp_path, "0.20"))
    assert any("eltwise_scratch.py" in p and "not in the pinned commit" in p for p in problems)
    assert "eltwise_scratch.py" not in by_name(files) and "broadcast.py" in by_name(files)


def test_a_modified_file_taken_from_the_checkout_is_refused(tmp_path):
    f = Fake(tmp_path)
    f.modified = {"examples/eltwise_add.py", tr.SPEC_REL}
    _, problems = tr.assemble(f.env, make_skill(tmp_path, "0.20"))
    assert any("examples/eltwise_add.py" in p and "modified" in p for p in problems)
    assert any(tr.SPEC_REL in p for p in problems)


def test_a_modified_file_that_is_not_used_does_not_refuse(tmp_path):
    f = Fake(tmp_path)
    f.modified = {"docs/unrelated.md"}
    _, problems = tr.assemble(f.env, make_skill(tmp_path, "0.20"))
    assert problems == []


def test_when_git_cannot_say_what_is_tracked_no_extra_example_is_taken(tmp_path):
    f = Fake(tmp_path)
    (f.src / "examples" / "broadcast.py").write_text("# x\n")
    f.git_works = False
    files, problems = tr.assemble(f.env, make_skill(tmp_path, "0.20"))
    assert problems == [] and "broadcast.py" not in by_name(files)


def test_an_examples_directory_that_cannot_be_listed_is_a_problem_not_a_traceback(tmp_path, monkeypatch):
    f = Fake(tmp_path)

    def deny(path):
        raise PermissionError(13, "Permission denied")
    monkeypatch.setattr(tr.os, "scandir", deny)
    files, problems = tr.assemble(f.env, make_skill(tmp_path, "0.20"))
    assert any("cannot be listed" in p and "Permission denied" in p for p in problems)


def test_git_is_only_asked_to_read_and_never_to_refresh_the_index(tmp_path):
    f = Fake(tmp_path)
    tr.git_state(f.env)
    git = [a for a in f.calls if a[0] == "git"]
    assert {w for a in git for w in a if w in ("ls-tree", "status")} == {"ls-tree"}
    assert all("fetch" not in a and "checkout" not in a and "reset" not in a for a in git)
    assert all(e and e["GIT_OPTIONAL_LOCKS"] == "0" for e in f.envs)
    assert all("HF_TOKEN" not in e for e in f.envs)


def test_the_checkout_files_row_reports_untracked_and_modified_files(tmp_path):
    f = Fake(tmp_path)
    skill = make_skill(tmp_path, "0.20")
    assert rows_by_name(check(f, skill))["checkout files"].status == tc.OK
    (f.src / "examples" / "eltwise_scratch.py").write_text("x = 1\n")
    f.untracked = {"examples/eltwise_scratch.py"}
    f.modified = {"docs/sphinx/overview.md"}
    r = rows_by_name(check(f, skill))["checkout files"]
    assert r.status == tc.MISMATCH and r.blocks and "eltwise_scratch.py (untracked)" in r.detail
    assert "overview.md (modified)" in r.detail and "restore" in r.fix
    f.git_works = False
    u = rows_by_name(check(f, skill))["checkout files"]
    assert u.status == tc.UNKNOWN and not u.blocks


# ---- nothing from an off-pin checkout runs without the override, and what runs gets a minimal environment ---------------

def sim_env(tmp_path, *, head=PIN):
    """A KEnv for the real ensure: every step passes, and the negative examples fail with the text upstream prints."""
    f = Fake(tmp_path, head=head)
    diag = f.src / "python" / "ttl" / "diagnostics.py"
    diag.parent.mkdir(parents=True)
    diag.write_text("# diagnostics\n")
    lib = tmp_path / "purelib"
    (lib / "ttl" / "ttl").mkdir(parents=True)
    (lib / "ttl" / "ttl" / "diagnostics.py").write_bytes(diag.read_bytes())
    (f.env.venv_dir / "bin" / "python").write_text("#!/bin/sh\n")
    ran = []
    base = f.run

    def run(argv, timeout, env=None, cwd=None):
        if "sysconfig" in " ".join(map(str, argv)):
            return 0, f"{lib}\n", ""
        if "import torch" in " ".join(map(str, argv)):
            return 0, "", ""
        if str(argv[0]).endswith("tt-lang-sim") and argv[-1] != "--version":
            ran.append((list(argv), env))
            return 1, " ".join(n for _, needles in defaults.TTLANG_NEGATIVE_EXAMPLES for n in needles), ""
        return base(argv, timeout, env, cwd)
    f.env.run = run
    return f, ran, hashlib.sha256(diag.read_bytes()).hexdigest()


def test_no_example_from_an_off_pin_checkout_runs_without_the_override(tmp_path, monkeypatch):
    f, ran, sha = sim_env(tmp_path, head="d" * 40)
    monkeypatch.setattr(defaults, "TTLANG_DIAGNOSTICS_SHA256", sha)
    ttlang_env.ensure(f.env, check_only=True, say=lambda s: None)
    assert ran == []


def test_a_simulator_off_the_pin_also_runs_nothing_without_the_override(tmp_path, monkeypatch):
    f, ran, sha = sim_env(tmp_path)
    f.sim = "tt-lang-sim 1.1.10+gabcdef0\n"
    monkeypatch.setattr(defaults, "TTLANG_DIAGNOSTICS_SHA256", sha)
    results = ttlang_env.ensure(f.env, check_only=True, say=lambda s: None)
    assert ran == [] and results[-1].name == "verify" and results[-1].status == ttlang_env.TODO


def test_with_the_override_the_examples_run_in_an_allow_listed_environment(tmp_path, monkeypatch):
    f, ran, sha = sim_env(tmp_path, head="d" * 40)
    monkeypatch.setattr(defaults, "TTLANG_DIAGNOSTICS_SHA256", sha)
    f.env.environ = {"PATH": "/usr/bin", "HF_TOKEN": "hf_secret", "OPENAI_API_KEY": "k", "HOME": "/home/operator",
                     "LANG": "C.UTF-8"}
    results = ttlang_env.ensure(f.env, check_only=True, say=lambda s: None, allow_pin_mismatch=True)
    assert ran and results[-1].name == "verify" and results[-1].status == ttlang_env.OK
    for _, env in ran:
        assert env["PATH"] == "/usr/bin" and env["LANG"] == "C.UTF-8"
        assert "HF_TOKEN" not in env and "OPENAI_API_KEY" not in env
        assert env["HOME"] != "/home/operator" and env["HOME"].startswith(str(tmp_path))
        assert env["TT_VISIBLE_DEVICES"] == ttlang_env.NO_CHIP


def test_an_example_at_the_pin_runs_in_the_same_minimal_environment(tmp_path, monkeypatch):
    f, ran, sha = sim_env(tmp_path)
    monkeypatch.setattr(defaults, "TTLANG_DIAGNOSTICS_SHA256", sha)
    f.env.environ = {"PATH": "/usr/bin", "HF_TOKEN": "hf_secret"}
    ttlang_env.ensure(f.env, check_only=True, say=lambda s: None)
    assert ran and all("HF_TOKEN" not in env for _, env in ran)


def test_the_example_environment_allow_list_is_the_agent_shells():
    from orchard import agent
    assert ttlang_env.VERIFY_ENV_ALLOW == agent.ENV_ALLOW and ttlang_env.NO_CHIP == agent.NO_CHIP


# ---- verify() runs copies from the pinned commit, in a directory made for the call, after the integrity check -----------

import os
import subprocess
import textwrap


def make_repo(root: Path):
    """A real git repository holding the programs verify() runs, committed once; returns (KEnv, commit sha, marker)."""
    src = root / "tt-lang"
    (src / "examples" / "errors").mkdir(parents=True)
    (src / "examples" / "errors" / "eltwise_add_error.py").write_text("print('the pinned program')\n")
    (src / "examples" / "compiler_only_external_call.py").write_text("print('the pinned program')\n")
    (src / "examples" / "compiler_only_add.hpp").write_text("// header\n")
    git = ["git", "-C", str(src), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    for argv in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "x"]):
        subprocess.run(git + argv, check=True, capture_output=True)
    sha = subprocess.run(git + ["rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    venv = root / "venv"
    (venv / "bin").mkdir(parents=True)
    marker = root / "marker.txt"
    needles = " ".join(n for _, ns in defaults.TTLANG_NEGATIVE_EXAMPLES for n in ns)
    # A stand-in tt-lang-sim: it records its working directory and the program it was given, and runs the program.
    (venv / "bin" / "tt-lang-sim").write_text(textwrap.dedent(f"""\
        #!/bin/sh
        echo "$PWD $1" >> {marker}
        echo "$HOME $PATH" >> {marker}.env
        python3 "$1"
        echo "{needles}"
        exit 1
        """))
    (venv / "bin" / "tt-lang-sim").chmod(0o755)
    env = ttlang_env.KEnv(src_dir=src, venv_dir=venv, scratch_dir=root / "scratch", environ={"PATH": os.environ["PATH"]})
    return env, sha, marker


def test_verify_runs_a_copy_from_the_pinned_commit_in_a_fresh_directory_that_is_then_removed(tmp_path, monkeypatch):
    env, sha, marker = make_repo(tmp_path)
    monkeypatch.setattr(ttlang_pin, "SHA", sha)
    got = ttlang_env.verify(env)
    assert got.status == ttlang_env.OK, got.detail
    lines = marker.read_text().splitlines()
    assert len(lines) == len(defaults.TTLANG_NEGATIVE_EXAMPLES)
    for line in lines:
        cwd, program = line.split(" ", 1)
        assert Path(cwd).name.startswith("orchard-verify-") and program.startswith(cwd + "/examples/")
        assert not Path(cwd).exists()                        # removed after the call
        assert not str(env.src_dir) in program               # never the working tree's version
    assert subprocess.run(["git", "-C", str(env.src_dir), "status", "--porcelain"], capture_output=True,
                          text=True).stdout == ""


def test_a_modified_negative_example_that_would_write_a_file_is_never_run(tmp_path, monkeypatch):
    env, sha, marker = make_repo(tmp_path)
    monkeypatch.setattr(ttlang_pin, "SHA", sha)
    planted = tmp_path / "planted.txt"
    (env.src_dir / "examples" / "errors" / "eltwise_add_error.py").write_text(
        f"open({str(planted)!r}, 'w').write('ran')\n")
    for allow in (False, True):                               # the override does not reach a modified program
        got = ttlang_env.verify(env, allow_pin_mismatch=allow)
        assert got.status == ttlang_env.FAIL and "eltwise_add_error.py (modified)" in got.detail
    assert not planted.exists() and not marker.exists()


def test_an_untracked_support_file_next_to_the_programs_blocks_verify(tmp_path, monkeypatch):
    env, sha, marker = make_repo(tmp_path)
    monkeypatch.setattr(ttlang_pin, "SHA", sha)
    hpp = env.src_dir / "examples" / "compiler_only_add.hpp"
    hpp.write_text("// edited\n")
    got = ttlang_env.verify(env)
    assert got.status == ttlang_env.FAIL and "compiler_only_add.hpp (modified)" in got.detail and not marker.exists()


def test_verify_refuses_when_git_cannot_say_whether_the_programs_are_clean(tmp_path):
    f = Fake(tmp_path)
    f.git_works = False
    got = ttlang_env.verify(f.env)
    assert got.status == ttlang_env.FAIL and "git could not say" in got.detail


def test_verify_without_the_override_never_falls_back_to_another_commit(tmp_path, monkeypatch):
    env, sha, marker = make_repo(tmp_path)
    monkeypatch.setattr(ttlang_pin, "SHA", "0" * 40)        # the pinned commit is not in the checkout
    got = ttlang_env.verify(env)
    assert got.status == ttlang_env.FAIL and "pinned commit" in got.detail and not marker.exists()
    again = ttlang_env.verify(env, allow_pin_mismatch=True)  # with the override the checkout's own HEAD is used
    assert again.status == ttlang_env.OK and marker.exists()


def test_the_integrity_check_runs_before_any_program_in_ensure(tmp_path, monkeypatch):
    f, ran, sha = sim_env(tmp_path)
    monkeypatch.setattr(defaults, "TTLANG_DIAGNOSTICS_SHA256", sha)
    f.modified = {"examples/errors/eltwise_add_error.py"}
    results = ttlang_env.ensure(f.env, check_only=True, say=lambda s: None)
    assert ran == [] and results[-1].name == "verify" and results[-1].status == ttlang_env.FAIL
    order = [a for a in f.calls if a[0] == "git"]
    assert any("ls-tree" in a for a in order)


def test_kernel_env_check_runs_verify_through_the_same_guards(tmp_path, monkeypatch, capsys):
    f, ran, sha = sim_env(tmp_path)
    monkeypatch.setattr(defaults, "TTLANG_DIAGNOSTICS_SHA256", sha)
    f.modified = {"examples/compiler_only_add.hpp"}
    args = argparse.Namespace(check=True, src_dir=None, venv_dir=None, tt_lang_skill_dir=str(make_skill(tmp_path, "0.20")),
                              allow_stale_skills=True)
    monkeypatch.setattr(ttlang_env, "skills_status", lambda *a, **k: [])
    code = ttlang_env.main(args, env=f.env)
    assert code == 2 and ran == []
    assert "verify" in capsys.readouterr().out


def test_a_checkout_directory_that_cannot_be_searched_is_handled_wherever_it_is_looked_at(tmp_path):
    f = Fake(tmp_path)
    tree = tmp_path / "tree"
    tree.mkdir()
    os.chmod(f.src, 0)
    os.chmod(tree, 0)
    try:
        names, why = tr.extra_examples(f.src, set(), {"examples/broadcast.py"})
        assert names == [] and why and "cannot be listed" in why
        assert ttlang_env.head_sha(f.env) is None
        assert tc._tree_row(f.env, tree).name == "hardware tree"          # no PermissionError reaches the caller
        f.git_works = False                                                # the fake git would walk the tree itself
        files, problems = tr.assemble(f.env, make_skill(tmp_path, "0.20"))
        assert problems                                                    # refused with reasons, no traceback
    finally:
        os.chmod(f.src, 0o755)
        os.chmod(tree, 0o755)


def test_relative_venv_checkout_scratch_and_path_still_verify_from_the_temporary_directory(tmp_path, monkeypatch):
    """The programs run in a directory made for the call, so nothing handed to them may be relative to the caller's."""
    env, sha, marker = make_repo(tmp_path)
    monkeypatch.setattr(ttlang_pin, "SHA", sha)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "bin").mkdir()
    rel = ttlang_env.KEnv(src_dir=Path("tt-lang"), venv_dir=Path("venv"), scratch_dir=Path("scratch"),
                          environ={"PATH": "bin" + os.pathsep + os.environ["PATH"]})
    assert not rel.sim.is_absolute()
    got = ttlang_env.verify(rel)
    assert got.status == ttlang_env.OK, got.detail
    assert len(marker.read_text().splitlines()) == len(defaults.TTLANG_NEGATIVE_EXAMPLES)
    for line in Path(str(marker) + ".env").read_text().splitlines():
        home, path = line.split(" ", 1)
        assert os.path.isabs(home) and home.startswith(str(tmp_path))
        assert all(os.path.isabs(p) for p in path.split(os.pathsep) if p)
        assert path.split(os.pathsep)[0] == str(tmp_path / "bin")


def test_minimal_environment_paths_do_not_depend_on_the_callers_directory(tmp_path):
    env = ttlang_env.KEnv(src_dir=Path("s"), venv_dir=Path("v"), scratch_dir=Path("scratch"), environ={"PATH": "bin:/usr/bin"})
    out = ttlang_env.minimal_env(env)
    assert all(os.path.isabs(out[k]) for k in ("HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME", "HF_TOKEN_PATH"))
    assert all(os.path.isabs(p) for p in out["PATH"].split(os.pathsep))


# ---- inspecting a checkout must never run a program the checkout names (filters, fsmonitor, hooks) ---------------------

def make_hostile_repo(root: Path):
    """make_repo plus a clean filter, a textconv driver, an fsmonitor program and hooks in the checkout's own config
    and attributes, each of which would write a marker file if git ran it. The working tree has a modified example."""
    env, sha, marker = make_repo(root)
    src = env.src_dir
    mark = root / "FILTER-RAN"
    writer = root / "write_marker.sh"
    writer.write_text(f"#!/bin/sh\necho ran >> {mark}\ncat\n")
    writer.chmod(0o755)
    git = ["git", "-C", str(src)]
    for key, value in (("filter.evil.clean", str(writer)), ("filter.evil.smudge", str(writer)),
                       ("diff.evil.textconv", str(writer)), ("core.fsmonitor", str(writer)),
                       ("core.hooksPath", str(root / "hooks"))):
        subprocess.run(git + ["config", key, value], check=True, capture_output=True)
    (src / ".gitattributes").write_text("* filter=evil diff=evil\n")
    (src / ".git" / "info").mkdir(exist_ok=True)
    (src / ".git" / "info" / "attributes").write_text("* filter=evil diff=evil\n")
    # change a tracked example and its size, so any git command that hashes the working tree must run the clean filter
    (src / "examples" / "compiler_only_external_call.py").write_text("print('edited, and longer than before')\n")
    return env, sha, marker, mark, src


def test_the_control_git_status_really_runs_the_checkouts_filter_here(tmp_path):
    """The instrument check: in this repository `git status` does write the marker, so a marker that never appears
    under our code means something."""
    env, sha, marker, mark, src = make_hostile_repo(tmp_path)
    subprocess.run(["git", "-C", str(src), "status", "--porcelain"], capture_output=True)
    assert mark.exists()


def test_no_git_call_we_make_runs_a_filter_a_textconv_a_fsmonitor_or_a_hook(tmp_path, monkeypatch):
    env, sha, marker, mark, src = make_hostile_repo(tmp_path)
    monkeypatch.setattr(ttlang_pin, "SHA", sha)
    tracked, dirty = tr.git_state(env)
    assert dirty["examples/compiler_only_external_call.py"] == "modified"
    tr.assemble(env, make_skill(tmp_path, "0.20"))
    tc.check(env, make_skill(tmp_path, "0.20", "skill2"))
    ttlang_env.verify(env)
    ttlang_env.verify(env, allow_pin_mismatch=True)
    assert not mark.exists()


def test_every_git_call_is_made_with_the_safe_settings(tmp_path):
    f = Fake(tmp_path)
    tr.git_state(f.env)
    ttlang_env.verify(f.env)
    git = [(a, e) for a, e in zip(f.calls, f.envs) if a[0] == "git" and a[3:5] != ["rev-parse", "HEAD"]]
    assert git
    for argv, env in git:
        joined = " ".join(argv)
        assert "core.fsmonitor=false" in joined and "core.hooksPath=/dev/null" in joined
        assert "core.attributesFile=/dev/null" in joined
        assert not {"status", "diff", "add", "checkout", "reset", "fetch", "show"} & set(argv)
        assert env["GIT_CONFIG_NOSYSTEM"] == "1" and env["GIT_CONFIG_GLOBAL"] == "/dev/null"
        assert env["GIT_ATTR_NOSYSTEM"] == "1" and env["GIT_OPTIONAL_LOCKS"] == "0"
        assert "HF_TOKEN" not in env


def test_the_working_tree_is_compared_with_the_pinned_blobs_in_python(tmp_path, monkeypatch):
    env, sha, marker = make_repo(tmp_path)
    monkeypatch.setattr(ttlang_pin, "SHA", sha)
    src = env.src_dir
    assert tr.git_state(env)[1] == {}
    (src / "examples" / "compiler_only_add.hpp").write_text("// changed\n")
    (src / "examples" / "new_scratch.py").write_text("x = 1\n")
    (src / "docs").mkdir(exist_ok=True)
    (src / "docs" / "note.md").write_text("untracked doc\n")
    (src / "examples" / "__pycache__").mkdir()
    (src / "examples" / "__pycache__" / "x.pyc").write_bytes(b"\0")
    (src / "examples" / "__pycache__" / "cache.bin").write_bytes(b"\0")
    (src / "examples" / "compiler_only_external_call.py").unlink()
    (src / "examples" / "link.py").symlink_to("compiler_only_add.hpp")
    dirty = tr.git_state(env)[1]
    assert dirty == {"examples/compiler_only_add.hpp": "modified", "examples/new_scratch.py": "untracked",
                     "docs/note.md": "untracked", "examples/compiler_only_external_call.py": "deleted",
                     "examples/link.py": "untracked"}


# ---- things git or the filesystem can redirect: replace refs, links, sizes (round 6) -----------------------------------

def test_a_replace_ref_for_the_pinned_commit_does_not_redirect_what_we_read(tmp_path, monkeypatch):
    env, sha, marker = make_repo(tmp_path)
    monkeypatch.setattr(ttlang_pin, "SHA", sha)
    src = env.src_dir
    plain = {"GIT_NO_REPLACE_OBJECTS": "1", "PATH": os.environ["PATH"]}
    git = ["git", "-C", str(src), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    program = src / "examples" / "errors" / "eltwise_add_error.py"
    program.write_text("print('a different commit')\n")
    subprocess.run(git + ["commit", "-q", "-a", "-m", "other"], check=True, capture_output=True, env=plain)
    other = subprocess.run(git + ["rev-parse", "HEAD"], check=True, capture_output=True, text=True,
                           env=plain).stdout.strip()
    subprocess.run(git + ["reset", "-q", "--hard", sha], check=True, capture_output=True, env=plain)
    subprocess.run(git + ["replace", sha, other], check=True, capture_output=True, env=plain)
    # the control: without our settings, the pinned commit now reads as the other one
    shown = subprocess.run(git + ["cat-file", "blob", f"{sha}:examples/errors/eltwise_add_error.py"],
                           capture_output=True, text=True).stdout
    assert shown == "print('a different commit')\n"
    got = ttlang_env.verify(env)                       # the working tree is the pinned commit's, and still is
    assert got.status == ttlang_env.OK, got.detail
    assert tr.git_state(env)[1] == {}
    assert "a different commit" not in marker.read_text()


def test_every_git_call_switches_off_replace_refs(tmp_path):
    f = Fake(tmp_path)
    tr.git_state(f.env)
    ttlang_env.verify(f.env)
    git = [(a, e) for a, e in zip(f.calls, f.envs) if a[0] == "git"]
    assert git
    for argv, env in git:
        assert "core.useReplaceRefs=false" in " ".join(argv) and env["GIT_NO_REPLACE_OBJECTS"] == "1"


def make_link_repo(root: Path, which: str):
    """make_repo, then `examples` (or the checkout root) replaced by a link to the real directory."""
    env, sha, marker = make_repo(root)
    src = env.src_dir
    if which == "examples":
        (src / "examples").rename(src / "examples-real")
        (src / "examples").symlink_to("examples-real")
    else:
        moved = root / "tt-lang-real"
        src.rename(moved)
        src.symlink_to(moved)
    return env, sha, marker


def test_an_examples_directory_that_is_a_link_is_refused_before_anything_is_walked_or_run(tmp_path, monkeypatch):
    env, sha, marker = make_link_repo(tmp_path, "examples")
    monkeypatch.setattr(ttlang_pin, "SHA", sha)
    assert tr.git_state(env)[1].get("examples") == "link"
    got = ttlang_env.verify(env)
    assert got.status == ttlang_env.FAIL and "examples" in got.detail and not marker.exists()
    assert ttlang_env.verify(env, allow_pin_mismatch=True).status == ttlang_env.FAIL
    _, problems = tr.assemble(env, make_skill(tmp_path, "0.20"))
    assert any("is a link" in p for p in problems)
    row = rows_by_name(tc.check(env, make_skill(tmp_path, "0.20", "s2")))["checkout files"]
    assert row.status == tc.MISMATCH and row.blocks and "examples (link)" in row.detail


def test_a_checkout_root_that_is_a_link_is_refused(tmp_path, monkeypatch):
    env, sha, marker = make_link_repo(tmp_path, "root")
    monkeypatch.setattr(ttlang_pin, "SHA", sha)
    assert tr.git_state(env)[1] == {".": "link"}
    got = ttlang_env.verify(env)
    assert got.status == ttlang_env.FAIL and "link" in got.detail and not marker.exists()


def test_a_file_above_the_size_cap_is_not_read_and_is_reported_unchecked(tmp_path, monkeypatch):
    env, sha, marker = make_repo(tmp_path)
    monkeypatch.setattr(ttlang_pin, "SHA", sha)
    (env.src_dir / "docs").mkdir(exist_ok=True)
    big = env.src_dir / "docs" / "big.bin"
    big.write_bytes(b"x" * 100)
    git = ["git", "-C", str(env.src_dir), "-c", "user.name=t", "-c", "user.email=t@example.com"]
    subprocess.run(git + ["add", "-A"], check=True, capture_output=True)
    subprocess.run(git + ["commit", "-q", "-m", "big"], check=True, capture_output=True)
    sha = subprocess.run(git + ["rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    monkeypatch.setattr(ttlang_pin, "SHA", sha)
    monkeypatch.setattr(tr, "FILE_CAP", 50)
    opened = []
    real_open = os.open
    monkeypatch.setattr(tr.os, "open", lambda p, *a, **k: opened.append(str(p)) or real_open(p, *a, **k))
    assert tr.git_state(env)[1] == {"docs/big.bin": "unchecked"}
    assert not [p for p in opened if p.endswith("big.bin")]                 # never opened, so never read
    assert rows_by_name(tc.check(env, make_skill(tmp_path, "0.20"), references=[]))["checkout files"].status == tc.OK


def test_a_needed_program_above_the_size_cap_blocks_verify(tmp_path, monkeypatch):
    env, sha, marker = make_repo(tmp_path)
    monkeypatch.setattr(ttlang_pin, "SHA", sha)
    monkeypatch.setattr(tr, "FILE_CAP", 5)
    got = ttlang_env.verify(env)
    assert got.status == ttlang_env.FAIL and "unchecked" in got.detail and not marker.exists()


def test_files_are_hashed_in_chunks_and_give_git_s_blob_id(tmp_path, monkeypatch):
    path = tmp_path / "f.bin"
    path.write_bytes(bytes(range(256)) * 20)
    want = subprocess.run(["git", "hash-object", "--no-filters", str(path)], capture_output=True, text=True,
                          check=True).stdout.strip()
    monkeypatch.setattr(tr, "CHUNK", 7)
    sizes = []
    real = hashlib.sha1

    class Spy:
        def __init__(self, *a):
            self.h = real(*a)

        def update(self, data):
            sizes.append(len(data))
            self.h.update(data)

        def hexdigest(self):
            return self.h.hexdigest()
    monkeypatch.setattr(tr.hashlib, "sha1", Spy)
    assert tr._blob_id(path, 10_000)[0] == want
    assert max(sizes[1:]) <= 7 and len(sizes) > 100


def test_the_total_and_entry_caps_stop_the_scan_and_say_so(tmp_path, monkeypatch):
    env, sha, marker = make_repo(tmp_path)
    monkeypatch.setattr(ttlang_pin, "SHA", sha)
    monkeypatch.setattr(tr, "TOTAL_CAP", 30)                # the three programs are 22 + 30 + 22 bytes or so
    dirty = tr.git_state(env)[1]
    assert dirty and all(v == "unchecked" for v in dirty.values())
    monkeypatch.setattr(tr, "TOTAL_CAP", 10 ** 9)
    monkeypatch.setattr(tr, "MAX_ENTRIES", 2)
    assert tr.git_state(env)[1].get("(entries)") == "too many"


def test_a_file_that_grew_between_the_size_check_and_the_open_is_still_not_read(tmp_path, monkeypatch):
    path = tmp_path / "grew.bin"
    path.write_bytes(b"x" * 100)
    real = os.lstat

    def lstat(p, *a, **k):                      # the size check sees a small file
        st = real(p, *a, **k)
        return os.stat_result((st.st_mode, st.st_ino, st.st_dev, st.st_nlink, st.st_uid, st.st_gid, 1,
                               int(st.st_atime), int(st.st_mtime), int(st.st_ctime)))
    monkeypatch.setattr(tr.os, "lstat", lstat)
    monkeypatch.setattr(tr, "CHUNK", 1)
    assert tr._blob_id(path, 50) == (None, "unchecked")


def test_the_head_and_tree_reads_get_the_safe_environment_and_not_the_operators_git_variables(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_DIR", "/somewhere/else/.git")
    monkeypatch.setattr(ttlang_pin, "TT_METAL_REF", "v0.80.0")
    f = Fake(tmp_path)
    f.env.environ = {"PATH": "/usr/bin", "GIT_DIR": "/somewhere/else/.git", "GIT_WORK_TREE": "/x"}
    tree = tmp_path / "tree"
    (tree / ".git").mkdir(parents=True)
    ttlang_env.head_sha(f.env)
    tc.check(f.env, make_skill(tmp_path, "0.20"), tt_metal_dir=tree)
    reads = [(a, e) for a, e in zip(f.calls, f.envs) if a[0] == "git" and ("rev-parse" in a or "describe" in a)]
    assert len(reads) >= 3
    for argv, env in reads:
        assert "GIT_DIR" not in env and "GIT_WORK_TREE" not in env
        assert env["GIT_NO_REPLACE_OBJECTS"] == "1" and env["GIT_CONFIG_NOSYSTEM"] == "1"
        cfg = {env[f"GIT_CONFIG_KEY_{i}"]: env[f"GIT_CONFIG_VALUE_{i}"] for i in range(int(env["GIT_CONFIG_COUNT"]))}
        assert cfg["core.useReplaceRefs"] == "false" and cfg["core.fsmonitor"] == "false"


# ---- --reference-source: the pre-PR-198 reference set, for an A/B -------------------------------------------------------

def test_reference_source_skill_gives_the_old_set_skill_guide_and_examples_and_no_extra_kernels(tmp_path):
    f = Fake(tmp_path)
    (f.src / "examples" / "broadcast.py").write_text("# an extra kernel\n")
    skill = make_skill(tmp_path, "0.12")
    files, problems = tr.assemble(f.env, skill, reference_source="skill")
    assert problems == []
    b = by_name(files)
    assert [r["name"] for r in files] == ["guide.md", "examples.md", "TTLangSpecification.md", "eltwise_add.py",
                                          "eltwise_broadcast_reduce.py"]
    assert b["guide.md"]["data"] == b"# skill guide\n" and b["guide.md"]["source"] == ttlang_env.SKILL_SOURCE
    assert b["guide.md"]["generated"] is False and b["guide.md"]["spec_version"] == "0.12"
    assert b["TTLangSpecification.md"]["spec_version"] == "0.20"
    new, _ = tr.assemble(f.env, skill)                                  # the default is unchanged
    assert by_name(new)["guide.md"]["generated"] is True and "broadcast.py" in by_name(new)


def test_an_unknown_reference_source_is_refused(tmp_path):
    f = Fake(tmp_path)
    with pytest.raises(ValueError):
        tr.assemble(f.env, make_skill(tmp_path), reference_source="elsewhere")


def test_a_skill_source_that_documents_another_version_does_not_block_when_the_operator_chose_it(tmp_path):
    f = Fake(tmp_path)
    skill = make_skill(tmp_path, "0.12")
    refs, _ = tr.assemble(f.env, skill, reference_source="skill")
    blocked = rows_by_name(tc.check(f.env, skill, references=refs))["references"]
    assert blocked.status == tc.MISMATCH and blocked.blocks
    chosen = rows_by_name(tc.check(f.env, skill, references=refs, reference_source="skill"))["references"]
    assert chosen.status == tc.MISMATCH and not chosen.blocks and "--reference-source skill" in chosen.detail
