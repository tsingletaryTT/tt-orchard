"""Checks on the archived Granite evidence scripts that need no model and no chip.

Issue 225: `bf16_cpu_control.py` compared the teacher-forced logits (the full fp32 sequence) with reference logits cut to the length of the
bf16 free run. When bf16 stopped early at an end-of-sequence token (8 of 24 tokens, for example) the two shapes differed and the report
was never written. The script needs the model to run, so this test reads the script and confirms it keeps the reference logits whole."""
import ast
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parent.parent / "docs" / "run-logs" / "evidence" / "granite-real-port" / "reference" / "bf16_cpu_control.py")


def _ref_lg_value() -> ast.expr:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    values = [n.value for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "ref_lg" for t in n.targets)]
    assert len(values) == 1, "the script must assign ref_lg exactly once"
    return values[0]


def test_the_reference_logits_are_not_sliced_to_the_free_run_length():
    value = _ref_lg_value()
    assert not any(isinstance(n, ast.Slice) for n in ast.walk(value)), ast.unparse(value)
    assert "gen" not in {n.id for n in ast.walk(value) if isinstance(n, ast.Name)}, ast.unparse(value)
