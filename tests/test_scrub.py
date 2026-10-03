"""The bundle scrub check (spec section 10)."""
import pytest

from orchard.scrub import scrub_bundle, scrub_text

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
