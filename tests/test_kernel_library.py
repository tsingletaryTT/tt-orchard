"""The worked-kernel library: kernels that pass the simulator, given to the coder in each attempt's reference/."""
import hashlib
import pytest

from orchard import kernelprobe as kp

LIBRARY = kp.TEMPLATES / "library"
OPS_WITH_A_KERNEL = ["center", "rmsnorm", "silu_mul", "softmax", "tile_scan"]


def test_the_library_holds_one_kernel_per_op_and_nothing_else():
    assert sorted(p.stem for p in LIBRARY.glob("*.py")) == OPS_WITH_A_KERNEL


def _names(refs):
    return sorted(r["name"] for r in refs)


def test_mode_all_offers_every_library_kernel():
    refs = kp.library_references("rmsnorm", "all")
    assert _names(refs) == [f"library_{op}.py" for op in OPS_WITH_A_KERNEL]
    for r in refs:
        data = (LIBRARY / r["name"].replace("library_", "")).read_bytes()
        assert r["data"] == data and r["sha256"] == hashlib.sha256(data).hexdigest()
        assert r["source"] == kp.LIBRARY_SOURCE and r["path"].endswith(r["name"].replace("library_", ""))


def test_mode_others_leaves_out_the_op_under_test_only():
    assert _names(kp.library_references("rmsnorm", "others")) == ["library_center.py", "library_silu_mul.py",
                                                                    "library_softmax.py", "library_tile_scan.py"]
    assert len(kp.library_references("add", "others")) == 5          # an op with no kernel of its own loses nothing


def test_mode_none_offers_nothing_and_an_unknown_mode_is_refused():
    assert kp.library_references("rmsnorm", "none") == []
    with pytest.raises(ValueError, match="unknown library mode"):
        kp.library_references("rmsnorm", "some")


def test_the_report_says_when_the_library_held_the_op_under_test():
    assert kp.library_note("rmsnorm", "all").startswith("The library held a worked kernel for rmsnorm")
    assert "adapt" in kp.library_note("rmsnorm", "all")
    assert kp.library_note("rmsnorm", "others") is None
    assert kp.library_note("rmsnorm", "none") is None
    assert kp.library_note("linear_scan", "all") is None             # no kernel for it, so nothing was copyable


def test_the_skill_points_at_the_library_and_says_how_to_use_it():
    flat = " ".join(kp.SKILL.read_text().split())
    assert "`reference/library_*.py`" in flat
    assert "adapt" in flat

