"""CLAUDE.md is read by Copilot's code review as custom instructions. At about 390 KB it left no room for the diff (a review
of pull request 109 failed with "exceed the maximum prompt size"; the reviews of 106 and 108 saw only the version files).
The dated log lives in docs/project-log.md, which no review reads."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIMIT_BYTES = 30_000


def test_claude_md_stays_small_enough_for_a_code_review_to_read_the_diff():
    size = (ROOT / "CLAUDE.md").stat().st_size
    assert size <= LIMIT_BYTES, f"CLAUDE.md is {size} bytes; keep it under {LIMIT_BYTES} and put dated entries in docs/project-log.md"


def test_the_dated_log_has_a_home_and_claude_md_says_where_it_is():
    log = ROOT / "docs" / "project-log.md"
    assert log.is_file() and log.stat().st_size > 100_000
    assert "docs/project-log.md" in (ROOT / "CLAUDE.md").read_text()


def test_the_readme_points_at_the_log_and_not_at_claude_md_for_it():
    text = (ROOT / "README.md").read_text()
    assert "The log in `CLAUDE.md`" not in text and "see the log in CLAUDE.md" not in text
    assert "docs/project-log.md" in text
