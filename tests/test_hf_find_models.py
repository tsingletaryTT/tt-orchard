# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""scripts/hf_find_models.py against a fake Hub (no network).

The script searches the Hugging Face model API for models derived from named parent models (finetunes, quantizations, adapters and
merges) and for names that match a pattern. These tests pin the query URLs, the de-duplication, the date filter, the name filter, the
exit codes, and that the access token is sent but never printed.
"""
import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "hf_find_models.py"
SPEC = importlib.util.spec_from_file_location("hf_find_models_under_test", SCRIPT)
fm = importlib.util.module_from_spec(SPEC)
sys.modules["hf_find_models_under_test"] = fm
SPEC.loader.exec_module(fm)

NOW = datetime(2026, 11, 6, 12, 0, tzinfo=timezone.utc)


def iso(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def model(mid, days_ago=1.0, downloads=5, tags=(), gated=False):
    return {"id": mid, "createdAt": iso(days_ago), "lastModified": iso(days_ago / 2), "downloads": downloads, "tags": list(tags), "gated": gated}


class FakeHub:
    """Answers each API URL from a table keyed by (filter or search) text and records every call with its headers."""

    def __init__(self, by_filter=None, by_search=None, fail=()):
        self.by_filter = by_filter or {}
        self.by_search = by_search or {}
        self.fail = set(fail)
        self.calls = []

    def __call__(self, url, headers):
        self.calls.append((url, dict(headers)))
        q = parse_qs(urlparse(url).query)
        key = (q.get("filter") or q.get("search") or [""])[0]
        if key in self.fail:
            raise OSError("boom " + key)
        table = self.by_filter if "filter" in q else self.by_search
        return json.dumps(table.get(key, [])).encode()


def test_a_parent_is_searched_under_every_derivation_kind():
    hub = FakeHub()
    fm.find(parents=["Qwen/Qwen3.8-27B"], searches=[], authors=[], since_days=None, limit=10, fetch=hub, now=NOW, token=None)
    filters = {parse_qs(urlparse(u).query)["filter"][0] for u, _ in hub.calls}
    assert filters == {"base_model:Qwen/Qwen3.8-27B", "base_model:finetune:Qwen/Qwen3.8-27B", "base_model:quantized:Qwen/Qwen3.8-27B",
                       "base_model:adapter:Qwen/Qwen3.8-27B", "base_model:merge:Qwen/Qwen3.8-27B"}


def test_results_are_deduplicated_across_queries_and_sorted_newest_first():
    a, b = model("org/a", 3), model("org/b", 1)
    hub = FakeHub(by_filter={"base_model:Qwen/X": [a, b], "base_model:finetune:Qwen/X": [a], "base_model:quantized:Qwen/X": [b]})
    found, errors = fm.find(parents=["Qwen/X"], searches=[], authors=[], since_days=None, limit=10, fetch=hub, now=NOW, token=None)
    assert [m["id"] for m in found] == ["org/b", "org/a"] and errors == []


def test_each_result_says_which_query_found_it():
    hub = FakeHub(by_filter={"base_model:finetune:Qwen/X": [model("org/concept-a")]},
                  by_search={"concept": [model("org/concept-a"), model("org/concept-c")]})
    found, _ = fm.find(parents=["Qwen/X"], searches=["concept"], authors=[], since_days=None, limit=10, fetch=hub, now=NOW, token=None)
    by_id = {m["id"]: m for m in found}
    assert set(by_id["org/concept-a"]["found_by"]) == {"parent:Qwen/X (finetune)", "search:concept"}
    assert by_id["org/concept-c"]["found_by"] == ["search:concept"]


def test_since_days_drops_older_models():
    hub = FakeHub(by_search={"org": [model("org/new", 2), model("org/old", 40)]})
    found, _ = fm.find(parents=[], searches=["org"], authors=[], since_days=7, limit=10, fetch=hub, now=NOW, token=None)
    assert [m["id"] for m in found] == ["org/new"]


def test_a_model_with_no_creation_date_is_kept_when_filtering_by_date():
    m = model("org/undated", 1)
    del m["createdAt"]
    hub = FakeHub(by_search={"org": [m]})
    found, _ = fm.find(parents=[], searches=["org"], authors=[], since_days=7, limit=10, fetch=hub, now=NOW, token=None)
    assert [x["id"] for x in found] == ["org/undated"]


def test_a_search_result_must_contain_every_word_of_the_search():
    hub = FakeHub(by_search={"qwen": [model("a/Qwen3.8-27B-x"), model("a/Qwen3.8-7B-x"), model("a/other-27b")]})
    found, _ = fm.find(parents=[], searches=["qwen 27b"], authors=[], since_days=None, limit=10, fetch=hub, now=NOW, token=None)
    assert [m["id"] for m in found] == ["a/Qwen3.8-27B-x"]


def test_the_search_query_uses_the_first_word_and_the_filter_checks_all():
    hub = FakeHub()
    fm.find(parents=[], searches=["qwen 27b"], authors=[], since_days=None, limit=10, fetch=hub, now=NOW, token=None)
    assert parse_qs(urlparse(hub.calls[0][0]).query)["search"] == ["qwen"]


def test_an_author_search_lists_that_authors_models():
    hub = FakeHub()
    fm.find(parents=[], searches=[], authors=["ccore"], since_days=None, limit=10, fetch=hub, now=NOW, token=None)
    q = parse_qs(urlparse(hub.calls[0][0]).query)
    assert q["author"] == ["ccore"]


def test_one_failing_query_is_reported_and_the_others_still_run():
    hub = FakeHub(by_filter={"base_model:quantized:Qwen/X": [model("org/q")]}, fail={"base_model:finetune:Qwen/X"})
    found, errors = fm.find(parents=["Qwen/X"], searches=[], authors=[], since_days=None, limit=10, fetch=hub, now=NOW, token=None)
    assert [m["id"] for m in found] == ["org/q"] and len(errors) == 1 and "finetune" in errors[0]


def test_the_token_is_sent_as_a_header_and_never_in_the_url_or_the_output(capsys):
    hub = FakeHub(by_search={"org": [model("org/a")]}, fail={"y"})
    found, errors = fm.find(parents=[], searches=["org", "y"], authors=[], since_days=None, limit=10, fetch=hub, now=NOW, token="hf_SECRET123")
    assert all(h.get("Authorization") == "Bearer hf_SECRET123" for _, h in hub.calls)
    assert all("hf_SECRET123" not in u for u, _ in hub.calls)
    text = fm.render_table(found) + "\n".join(errors) + json.dumps(found)
    assert "hf_SECRET123" not in text


def test_the_limit_is_passed_to_the_api():
    hub = FakeHub()
    fm.find(parents=[], searches=["x"], authors=[], since_days=None, limit=7, fetch=hub, now=NOW, token=None)
    assert parse_qs(urlparse(hub.calls[0][0]).query)["limit"] == ["7"]


def test_the_table_lists_id_dates_downloads_and_the_tenstorrent_tags():
    found = [dict(model("org/a", 2, downloads=12, tags=["blackhole", "gguf", "license:apache-2.0", "other"], gated=True), found_by=["search:x"])]
    out = fm.render_table(found)
    assert "org/a" in out and "2026-11-04" in out and "12" in out and "blackhole" in out and "gguf" in out and "gated" in out
    assert "license:apache-2.0" in out and "other" not in out.split("org/a", 1)[1]


def test_an_empty_result_renders_a_plain_sentence():
    assert fm.render_table([]).strip() == "No matching models."


def test_main_exits_one_with_fail_on_empty_and_zero_otherwise(monkeypatch, capsys):
    monkeypatch.setattr(fm, "http_get", FakeHub())
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert fm.main(["--search", "nothing-here"]) == 0
    assert fm.main(["--search", "nothing-here", "--fail-on-empty"]) == 1


def test_main_json_prints_one_valid_document(monkeypatch, capsys):
    monkeypatch.setattr(fm, "http_get", FakeHub(by_search={"org": [model("org/a")]}))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert fm.main(["--search", "org", "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert [m["id"] for m in doc["models"]] == ["org/a"] and doc["errors"] == []


def test_main_needs_at_least_one_query(capsys):
    with pytest.raises(SystemExit) as exc:
        fm.main([])
    assert exc.value.code == 2


def test_the_default_watchlist_is_used_when_asked(monkeypatch):
    hub = FakeHub()
    monkeypatch.setattr(fm, "http_get", hub)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    fm.main(["--default-parents"])
    seen = {parse_qs(urlparse(u).query)["filter"][0] for u, _ in hub.calls}
    assert all(any(p in s for s in seen) for p in fm.DEFAULT_PARENTS)


def test_a_parent_name_is_checked_before_it_goes_into_a_url():
    with pytest.raises(ValueError):
        fm.find(parents=["Qwen/X&limit=999999"], searches=[], authors=[], since_days=None, limit=10, fetch=FakeHub(), now=NOW, token=None)
    with pytest.raises(ValueError):
        fm.find(parents=["no-slash"], searches=[], authors=[], since_days=None, limit=10, fetch=FakeHub(), now=NOW, token=None)


def test_an_exception_message_that_quotes_the_token_never_reaches_the_errors():
    secret = "hf_SECRET456"

    def leaky(url, headers):
        raise ValueError(f"Invalid header value b'Bearer {secret}\\n'")

    found, errors = fm.find(parents=[], searches=["org"], authors=[], since_days=None, limit=10, fetch=leaky, now=NOW, token=secret)
    assert len(errors) == 1 and secret not in "\n".join(errors) and "ValueError" in errors[0]


def test_non_object_rows_are_skipped_and_later_queries_still_run():
    hub = FakeHub(by_search={"org": [None, 5, "x", model("org/a")], "two": [model("two/b")]})
    found, errors = fm.find(parents=[], searches=["org", "two"], authors=[], since_days=None, limit=10, fetch=hub, now=NOW, token=None)
    assert sorted(m["id"] for m in found) == ["org/a", "two/b"] and errors == []
