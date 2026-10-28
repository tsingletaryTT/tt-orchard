# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The tt-lang pin record (orchard/ttlang_pin.py), its consistency check (orchard/ttlang_consistency.py) and the
reference assembly that follows it (orchard/ttlang_reference.py). No test touches the operator's checkout, venv, skill
directory or tt-metal tree: every directory is made up in tmp_path and every command is a fake."""
from __future__ import annotations

import argparse
import hashlib
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
        self.env = ttlang_env.KEnv(src_dir=self.src, venv_dir=venv, scratch_dir=tmp_path / "scratch", run=self.run)

    def run(self, argv, timeout, env=None):
        self.calls.append(list(argv))
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
