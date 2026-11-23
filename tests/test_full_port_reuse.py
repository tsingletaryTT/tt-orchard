# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""What stages 3 and 4 of a full port tested (gate honesty, issue 793).

On the Aion run a bare "pass" for stage 3 hid that the port was the same file as stage 2's with every weight replicated, and a bare
"pass" for stage 4 hid that it ran no test. These tests build a small run directory with the records the stages leave and check that the
operator bundle's words say what each stage did, read from the records and the port source, and that a record that is missing,
malformed or hostile gives "not known" and never a claim.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

from orchard import full_port_reuse as reuse

REPO = Path(__file__).resolve().parents[1]
TEMPLATE = REPO / "orchard" / "skills" / "operator-bundle-templates" / "build_bundle.py"

PORT_A = "import ttnn\nw = ttnn.from_torch(x, mesh_mapper=ttnn.ReplicateTensorToMesh(m))\nv = ttnn.from_torch(y, mesh_mapper=ttnn.ReplicateTensorToMesh(m))\n"
PORT_B = PORT_A + "u = ttnn.from_torch(z, mesh_mapper=ttnn.ShardTensorToMesh(m, dim=0))\n"
IDS = [5, 6, 7, 8]


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def put(root: Path, rel: str, value) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(value if isinstance(value, str) else json.dumps(value))


def check_record(port: str, chips: int, ids=IDS, top1=0.96875) -> dict:
    return {"port_sources": [{"path": "port_model.py", "sha256": sha(port), "role": "model"}], "chips": [f"0000:0{i}:00.0" for i in range(chips)],
            "chip_ids": list(ids), "top1_agreement": top1}


def make_run(tmp_path: Path, *, port2=PORT_A, port3=PORT_A, ids3=IDS, chips2=2, chips3=4, covered=(("2", 2), ("4", 3))) -> Path:
    run = tmp_path / "run"
    put(run, "stages/2/evidence/port-check.json", check_record(port2, chips2))
    put(run, "stages/3/evidence/port-check.json", check_record(port3, chips3, ids=ids3))
    put(run, "stages/3/port/port_model.py", port3)
    put(run, "stages/4/result.json", {"required": [int(c) for c, _ in covered],
                                       "entries": [{"chips": int(c), "covered_by": by, "top1_agreement": 0.96875} for c, by in covered]})
    return run


def text(info) -> str:
    return "\n".join(reuse.section(info))


# ---- stage 3 -------------------------------------------------------------------------------------------------------------------------

def test_a_stage_3_that_reran_the_same_port_says_so_and_what_that_shows(tmp_path):
    info = reuse.describe(make_run(tmp_path))
    s3 = info["stage3"]
    assert s3["same_port"] is True and s3["same_ids"] is True
    assert (s3["chips_before"], s3["chips_after"]) == (2, 4)
    assert (s3["replicated"], s3["sharded"]) == (2, 0)
    out = text(info)
    assert "byte-identical to stage 2's port" in out
    assert "no weight is split over the chips" in out
    assert "output ids equal stage 2's" in out


def test_a_stage_3_port_that_differs_from_stage_2s_is_not_called_identical(tmp_path):
    info = reuse.describe(make_run(tmp_path, port3=PORT_B))
    assert info["stage3"]["same_port"] is False
    out = text(info)
    assert "byte-identical" not in out and "differs from stage 2's port" in out


def test_a_port_that_shards_a_weight_is_not_said_to_replicate_everything(tmp_path):
    info = reuse.describe(make_run(tmp_path, port2=PORT_B, port3=PORT_B))
    assert info["stage3"]["sharded"] == 1
    out = text(info)
    assert "no weight is split over the chips" not in out
    assert "1 place" in out          # the number of sharding places the source holds


def test_output_ids_that_differ_from_stage_2s_are_reported_as_different(tmp_path):
    info = reuse.describe(make_run(tmp_path, ids3=[5, 6, 7, 9]))
    assert info["stage3"]["same_ids"] is False
    assert "differ from stage 2's" in text(info)


def test_a_missing_stage_2_record_gives_not_known_and_no_claim_of_identity(tmp_path):
    run = make_run(tmp_path)
    (run / "stages/2/evidence/port-check.json").unlink()
    info = reuse.describe(run)
    assert info["stage3"]["same_port"] is None and info["stage3"]["same_ids"] is None
    out = text(info)
    assert "byte-identical" not in out and "not known" in out


# ---- stage 4 -------------------------------------------------------------------------------------------------------------------------

def test_stage_4_entries_covered_by_earlier_stages_say_that_no_new_test_ran(tmp_path):
    info = reuse.describe(make_run(tmp_path))
    assert info["stage4"] == {"covered": [{"chips": 2, "covered_by": 2}, {"chips": 4, "covered_by": 3}], "tested": []}
    out = text(info)
    assert "Stage 4 ran no new test" in out
    assert "2-chip entry is covered by the stage 2 test" in out and "4-chip entry is covered by the stage 3 test" in out


def test_a_stage_4_entry_with_no_covering_stage_is_listed_as_tested_and_not_as_covered(tmp_path):
    run = make_run(tmp_path, covered=(("2", 2), ("4", None)))
    info = reuse.describe(run)
    assert info["stage4"]["tested"] == [4] and [c["chips"] for c in info["stage4"]["covered"]] == [2]
    out = text(info)
    assert "Stage 4 ran no new test" not in out and "4-chip entry ran its own test" in out


@pytest.mark.parametrize("bad", [None, [], "x", 3, {"entries": "x"}, {"entries": [None]}, {"entries": [{"chips": "2", "covered_by": 2}]},
                                 {"entries": [{"chips": 2, "covered_by": "2"}]}, {"entries": [{"chips": 2, "covered_by": True}]},
                                 {"entries": [{"chips": 10 ** 400, "covered_by": 2}]}])
def test_a_malformed_stage_4_result_never_gives_a_covered_claim(tmp_path, bad):
    run = make_run(tmp_path)
    put(run, "stages/4/result.json", bad if isinstance(bad, str) else json.dumps(bad))
    info = reuse.describe(run)
    assert not (info or {}).get("stage4", {}).get("covered")
    assert "Stage 4 ran no new test" not in (text(info) if info else "")


# ---- hostile and absent inputs ------------------------------------------------------------------------------------------------------

def test_a_run_with_no_stage_3_records_gives_none_so_other_runs_keep_their_bundle(tmp_path):
    assert reuse.describe(tmp_path) is None


@pytest.mark.parametrize("junk", ["{", "[]", "null", '{"port_sources": 3}', '{"port_sources": [null], "chip_ids": "x", "chips": 2}',
                                  '{"port_sources": [{"sha256": 1}], "chip_ids": [1, "a"], "chips": {}}', "\x00" * 8])
def test_a_junk_port_check_record_gives_not_known(tmp_path, junk):
    run = make_run(tmp_path)
    put(run, "stages/3/evidence/port-check.json", junk)
    info = reuse.describe(run)
    s3 = (info or {}).get("stage3") or {}
    assert s3.get("same_port") is not True and s3.get("same_ids") is not True


def test_a_port_source_that_is_a_link_or_a_fifo_is_not_read(tmp_path):
    run = make_run(tmp_path)
    src = run / "stages/3/port/port_model.py"
    src.unlink()
    os.mkfifo(src)                                     # a read of this would block forever
    info = reuse.describe(run)
    assert info["stage3"]["replicated"] is None and info["stage3"]["sharded"] is None
    src.unlink()
    (tmp_path / "elsewhere.py").write_text(PORT_A)
    src.symlink_to(tmp_path / "elsewhere.py")
    assert reuse.describe(run)["stage3"]["replicated"] is None


def test_an_oversized_port_source_is_not_counted(tmp_path):
    run = make_run(tmp_path)
    (run / "stages/3/port/port_model.py").write_text(PORT_A + "#" * (reuse.MAX_SOURCE_BYTES + 1))
    assert reuse.describe(run)["stage3"]["replicated"] is None


# ---- the bundle ------------------------------------------------------------------------------------------------------------------------

def load_builder():
    spec = importlib.util.spec_from_file_location("build_bundle_reuse_under_test", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class StubRun:
    """The parts of `Run` that `summary` reads, for a full-port run whose stages 0 to 6 passed and stage 7 was skipped."""

    def __init__(self, directory: Path):
        self.dir = directory
        self.path, self.cls, self.package = "full-port", "full-port", None

    def result(self, n):
        return "skipped" if n == 7 else "pass"

    def model(self):
        return "org/model", "rev"

    def profiles(self):
        return []


def test_the_summary_names_the_stage_that_reused_another_stages_test(tmp_path):
    bb = load_builder()
    s = bb.summary(StubRun(make_run(tmp_path)), 0, 0)
    assert "Stages that passed: 0, 1, 2, 3, 4, 5, 6." in s          # the verdict list keeps its words
    assert "Stage 4 ran no new test" in s and "stages 2 and 3" in s
    assert "Stage 3 re-ran stage 2's port unchanged" in s


def test_the_summary_of_a_run_with_no_such_records_is_what_it_was_before(tmp_path):
    bb = load_builder()
    s = bb.summary(StubRun(tmp_path), 0, 0)
    assert "ran no new test" not in s and "re-ran" not in s


def test_the_stage_3_and_4_bodies_of_a_full_port_run_carry_the_section(tmp_path):
    bb = load_builder()
    body = bb.reuse_lines(StubRun(make_run(tmp_path)))
    assert set(body) == {3, 4}
    assert any("byte-identical" in x for x in body[3]) and any("ran no new test" in x for x in body[4])


def test_a_source_hash_that_is_not_a_sha256_gives_not_known_and_not_identical(tmp_path):
    run = make_run(tmp_path)
    for stage in (2, 3):
        rec = json.loads((run / f"stages/{stage}/evidence/port-check.json").read_text())
        rec["port_sources"][0]["sha256"] = "zz" * 32          # equal in both records, and not a hash
        put(run, f"stages/{stage}/evidence/port-check.json", rec)
    assert reuse.describe(run)["stage3"]["same_port"] is None


def test_a_comment_that_names_a_mapper_is_not_counted(tmp_path):
    run = make_run(tmp_path, port3=PORT_A + "# was ttnn.ShardTensorToMesh(m, dim=0) before\n")
    assert reuse.describe(run)["stage3"]["sharded"] == 0


def test_a_chip_count_that_is_a_bool_or_huge_is_not_a_count(tmp_path):
    run = make_run(tmp_path)
    put(run, "stages/4/result.json", {"entries": [{"chips": True, "covered_by": 2}, {"chips": 10 ** 6, "covered_by": 3}]})
    assert reuse.describe(run)["stage4"] == {"covered": [], "tested": []}


def test_an_entry_covered_by_a_stage_that_does_not_cover_entries_is_not_counted_as_covered(tmp_path):
    info = reuse.describe(make_run(tmp_path, covered=(("2", 2), ("4", 5))))
    assert info["stage4"] == {"covered": [{"chips": 2, "covered_by": 2}], "tested": [4]}


def test_the_summary_does_not_say_stage_4_ran_nothing_when_one_entry_ran_its_own_test(tmp_path):
    bb = load_builder()
    s = bb.summary(StubRun(make_run(tmp_path, covered=(("2", 2), ("4", None)))), 0, 0)
    assert "ran no new test" not in s


def test_a_run_on_another_path_gets_no_such_sentences_even_when_the_records_exist(tmp_path):
    bb = load_builder()
    stub = StubRun(make_run(tmp_path))
    stub.path = "weights-only"
    assert "ran no new test" not in bb.summary(stub, 0, 0) and bb.reuse_lines(stub) == {}
