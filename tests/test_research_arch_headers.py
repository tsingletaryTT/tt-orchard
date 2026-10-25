# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Architecture matching and the header-only class estimate (arch_match.py and headers.py)."""
import json
import struct

import pytest

pytest.importorskip("tokenizers", reason="the class rules live in the delta-triage template")

from orchard.research import arch_match, headers  # noqa: E402
from orchard.research.http import FetchError  # noqa: E402

CFG = {"model_type": "gemma4", "architectures": ["Gemma4ForCausalLM"], "num_hidden_layers": 30,
       "hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 8, "vocab_size": 262144}


def test_the_same_family_and_decisive_values_are_inferred():
    r = arch_match.compare({**CFG, "model_type": "gemma4_unified"}, CFG)
    assert r["confidence"] == "inferred" and r["differing"] == {} and r["same_family"] is True


def test_the_same_family_with_a_different_size_is_weak_and_names_the_difference():
    r = arch_match.compare(CFG, {**CFG, "num_hidden_layers": 48})
    assert r["confidence"] == "weak"
    assert r["differing"] == {"num_hidden_layers": {"new": 30, "candidate": 48}}


def test_another_family_is_not_a_candidate():
    assert arch_match.compare(CFG, {**CFG, "model_type": "qwen3"})["confidence"] is None


def test_a_nested_text_config_is_read_through_the_templates_rules():
    nested = {"model_type": "gemma4", "text_config": {k: v for k, v in CFG.items() if k != "model_type"}}
    assert arch_match.compare(nested, CFG)["confidence"] == "inferred"


class FakeFetch:
    def __init__(self, routes):
        self.routes = routes

    def json(self, url, **kw):
        got = self.routes.get(url)
        if got is None or isinstance(got, Exception):
            raise got or FetchError("HTTP 404")
        return got

    def now(self):
        return "2026-10-12T00:00:00Z"


def test_the_config_reader_keeps_the_reason_a_config_could_not_be_read():
    """report._config is the one config reader (arch_match.fetch_config, which dropped the reason, is gone)."""
    from orchard.research import report
    from orchard.research.types import Result
    url = "https://huggingface.co/g/it/resolve/main/config.json"
    assert not hasattr(arch_match, "fetch_config")
    res = Result()
    assert report._config(FakeFetch({url: CFG}), "g/it", res) == CFG and res.unavailable == []
    assert report._config(FakeFetch({url: FetchError("HTTP 401 (needs a login)")}), "g/it", res) is None
    assert report._config(FakeFetch({url: [1]}), "g/it", res) is None
    assert report._config(FakeFetch({}), "bad id", res) is None
    assert [u.reason for u in res.unavailable] == ["g/it: HTTP 401 (needs a login)", "g/it: config.json is not an object"]


# ---- headers ---------------------------------------------------------------------------------------

def header_bytes(tensors: dict) -> bytes:
    blob = json.dumps({k: {"dtype": v[0], "shape": v[1], "data_offsets": [0, 0]} for k, v in tensors.items()}
                      | {"__metadata__": {}}).encode()
    return struct.pack("<Q", len(blob)) + blob


class RangeFetch:
    """Serves a file's first 8 bytes, then its header, by byte range, and a JSON index."""

    def __init__(self, files: dict, indexes=None):
        self.files, self.indexes, self.ranges = files, indexes or {}, []

    def get(self, url, *, byte_range=None, max_bytes=0):
        name = url.rsplit("/", 1)[1]
        self.ranges.append((name, byte_range))
        if name not in self.files:
            raise FetchError("HTTP 404")
        data = self.files[name]
        start, end = byte_range
        return data[start:end + 1]

    def json(self, url, **kw):
        name = url.rsplit("/", 1)[1]
        if name not in self.indexes:
            raise FetchError("HTTP 404")
        return self.indexes[name]

    def now(self):
        return "2026-10-12T00:00:00Z"


T = {"model.embed_tokens.weight": ("BF16", [100, 64]), "model.layers.0.q.weight": ("BF16", [64, 64])}


def test_a_header_is_read_with_two_range_requests_and_no_weights():
    f = RangeFetch({"model.safetensors": header_bytes(T) + b"\0" * 1000})
    r = headers.read_model(f, "g/it", ["config.json", "model.safetensors"])
    assert r.data["tensors"]["model.layers.0.q.weight"] == {"dtype": "BF16", "shape": [64, 64],
                                                           "shard": "model.safetensors"}
    assert [rng for _, rng in f.ranges] == [(0, 7), (8, len(header_bytes(T)) - 1)]
    assert r.data["errors"] == []


def test_the_index_decides_the_backbone_and_other_safetensors_are_possible_sidecars():
    f = RangeFetch({"model-00001-of-00001.safetensors": header_bytes(T), "joint_head.safetensors": header_bytes(T)},
                   indexes={"model.safetensors.index.json": {"weight_map": {"a": "model-00001-of-00001.safetensors"}}})
    files = ["model.safetensors.index.json", "model-00001-of-00001.safetensors", "joint_head.safetensors"]
    r = headers.read_model(f, "g/it", files)
    assert r.data["shards"] == ["model-00001-of-00001.safetensors"]
    assert r.data["extra_safetensors"] == ["joint_head.safetensors"]
    assert all(name != "joint_head.safetensors" for name, _ in f.ranges)


def test_the_index_wins_over_the_file_name_pattern():
    f = RangeFetch({"weights-a.safetensors": header_bytes(T), "model.safetensors": header_bytes(T)},
                   indexes={"model.safetensors.index.json": {"weight_map": {"a": "weights-a.safetensors"}}})
    r = headers.read_model(f, "g/it", ["model.safetensors.index.json", "weights-a.safetensors", "model.safetensors"])
    assert r.data["shards"] == ["weights-a.safetensors"] and r.data["extra_safetensors"] == ["model.safetensors"]


@pytest.mark.parametrize("index", [{"weight_map": ["a", "b"]}, {"weight_map": "x"}, ["not", "an", "object"]])
def test_an_index_with_a_weight_map_that_is_not_an_object_is_an_error_row_and_the_pattern_is_used(index):
    f = RangeFetch({"model-00001-of-00001.safetensors": header_bytes(T)}, indexes={"model.safetensors.index.json": index})
    r = headers.read_model(f, "g/it", ["model.safetensors.index.json", "model-00001-of-00001.safetensors"])
    assert any("index" in u.reason for u in r.unavailable)
    assert r.data["shards"] == ["model-00001-of-00001.safetensors"]
    assert any("index" in e for e in r.data["errors"])          # so the estimate is unknown, not a guess


@pytest.mark.parametrize("name", ["../evil.safetensors", "a/b.safetensors", "x y.safetensors", ".safetensors"])
def test_an_unsafe_shard_name_is_never_fetched(name):
    f = RangeFetch({})
    r = headers.read_model(f, "g/it", [name])
    assert f.ranges == [] and r.unavailable and r.data["tensors"] == {}


def test_an_implausible_header_length_is_an_error_row_not_a_crash():
    f = RangeFetch({"model.safetensors": struct.pack("<Q", 1 << 60) + b"x" * 100})
    r = headers.read_model(f, "g/it", ["model.safetensors"])
    assert r.data["errors"] and "not plausible" in r.data["errors"][0]


def test_a_header_larger_than_the_research_reads_is_an_error_and_its_bytes_are_never_asked_for():
    """The template accepts headers up to MAX_HEADER (100 MiB) from a local file. Over the network a repo
    anyone can upload must not make the research read that much, so the claimed length is checked first."""
    big = headers.MAX_HEADER_READ + 1
    f = RangeFetch({"model.safetensors": struct.pack("<Q", big) + b"x" * 100})
    r = headers.read_model(f, "g/it", ["model.safetensors"])
    assert f.ranges == [("model.safetensors", (0, 7))]
    assert r.data["errors"] and "larger than" in r.data["errors"][0]
    other = headers.read_model(RangeFetch({"model.safetensors": header_bytes(T)}), "g/base", ["model.safetensors"])
    assert headers.estimate_class(CFG, r.data, CFG, other.data)["class"] == "unknown"


def test_the_research_header_cap_is_the_fetchers_body_cap_and_below_the_templates_limit():
    from orchard.research import http, triage_rules
    assert headers.MAX_HEADER_READ == http.MAX_BODY < triage_rules.load().MAX_HEADER


def test_no_safetensors_is_unavailable_and_too_many_shards_is_refused():
    assert headers.read_model(RangeFetch({}), "g/it", ["config.json"]).unavailable[0].reason.endswith("no safetensors files")
    many = [f"model-{i:05d}-of-00100.safetensors" for i in range(100)]
    r = headers.read_model(RangeFetch({}), "g/it", many)
    assert "more than" in r.unavailable[0].reason


def test_a_failed_range_is_recorded_per_shard():
    f = RangeFetch({"model-00001-of-00002.safetensors": header_bytes(T)})
    r = headers.read_model(f, "g/it", ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"])
    assert len(r.data["errors"]) == 1 and "00002" in r.data["errors"][0]


def model_data(tensors, extra=()):
    return {"tensors": {k: {"dtype": v[0], "shape": v[1], "shard": "m"} for k, v in tensors.items()},
            "shards": ["m"], "extra_safetensors": list(extra), "errors": []}


def test_the_estimate_follows_the_template_rules():
    same = headers.estimate_class(CFG, model_data(T), CFG, model_data(T))
    assert same["class"] == "weights-only" and same["shared_tensors"] == 2 and same["sidecar_possible"] is False
    other = dict(T, **{"model.layers.0.q.weight": ("BF16", [64, 32])})
    assert headers.estimate_class(CFG, model_data(other), CFG, model_data(T))["class"] == "full-port"


def test_a_changed_dtype_that_is_not_a_small_float_is_full_port():
    other = dict(T, **{"model.layers.0.q.weight": ("I8", [64, 64])})
    r = headers.estimate_class(CFG, model_data(other), CFG, model_data(T))
    assert r["class"] == "full-port" and any("dtype" in x for x in r["reasons"])


def test_a_small_float_dtype_change_is_weights_only_as_in_stage_0():
    other = dict(T, **{"model.layers.0.q.weight": ("F32", [64, 64])})
    r = headers.estimate_class(CFG, model_data(other), CFG, model_data(T))
    assert r["class"] == "weights-only" and r["reasons"] == []


def test_a_repo_with_extra_safetensors_files_flags_a_possible_sidecar():
    r = headers.estimate_class(CFG, model_data(T, extra=["joint_head.safetensors"]), CFG, model_data(T))
    assert r["class"] == "weights-only" and r["sidecar_possible"] is True


def test_reasons_are_cleaned_and_limited():
    other = {f"model.layers.{i}.q.weight": ("BF16", [1, 1]) for i in range(20)}
    r = headers.estimate_class(CFG, model_data(other), CFG, model_data(
        {f"model.layers.{i}.q.weight": ("BF16", [2, 2]) for i in range(20)}))
    assert len(r["reasons"]) <= 5


@pytest.mark.parametrize("side", ["new", "candidate"])
def test_a_header_read_error_on_either_side_makes_the_estimate_unknown(side):
    """read_model records a failed shard in `errors`; the estimate must not call a model with missing
    tensors weights-only or full-port."""
    new, cand = model_data(T), model_data(T)
    (new if side == "new" else cand)["errors"] = ["model-00002-of-00002.safetensors: HTTP 404"]
    r = headers.estimate_class(CFG, new, CFG, cand)
    assert r["class"] == "unknown"
    assert r["reasons"] and any("00002" in x or "read" in x.lower() or "header" in x.lower() for x in r["reasons"])


class BrokenIndexFetch(RangeFetch):
    def json(self, url):
        raise FetchError("HTTP 500 at " + url)


def test_an_index_that_cannot_be_read_is_an_error_not_a_silent_fallback_to_file_names():
    f = BrokenIndexFetch({"model-00001-of-00001.safetensors": header_bytes(T), "joint_head.safetensors": header_bytes(T)})
    r = headers.read_model(f, "g/it", ["model.safetensors.index.json", "model-00001-of-00001.safetensors",
                                       "joint_head.safetensors"])
    assert any("index" in e for e in r.data["errors"])


def test_an_index_that_lists_none_of_the_repos_safetensors_is_an_error():
    f = RangeFetch({"model-00001-of-00001.safetensors": header_bytes(T)},
                   indexes={"model.safetensors.index.json": {"weight_map": {"a": "elsewhere.safetensors"}}})
    r = headers.read_model(f, "g/it", ["model.safetensors.index.json", "model-00001-of-00001.safetensors"])
    assert any("index" in e for e in r.data["errors"])


def test_a_broken_index_makes_the_class_estimate_unknown():
    f = BrokenIndexFetch({"model-00001-of-00001.safetensors": header_bytes(T)})
    new = headers.read_model(f, "g/it", ["model.safetensors.index.json", "model-00001-of-00001.safetensors"])
    other = headers.read_model(RangeFetch({"model-00001-of-00001.safetensors": header_bytes(T)}), "g/base",
                               ["model-00001-of-00001.safetensors"])
    est = headers.estimate_class({"model_type": "g"}, new.data, {"model_type": "g"}, other.data)
    assert est["class"] == "unknown"


def test_an_index_that_lists_a_shard_the_repo_does_not_have_is_an_error_and_the_estimate_is_unknown():
    """A partial tensor set would make the estimate a confident guess (missing tensors read as full-port)."""
    f = RangeFetch({"model-00001-of-00002.safetensors": header_bytes(T)},
                   indexes={"model.safetensors.index.json": {"weight_map": {
                       "a": "model-00001-of-00002.safetensors", "b": "model-00002-of-00002.safetensors"}}})
    new = headers.read_model(f, "g/it", ["model.safetensors.index.json", "model-00001-of-00002.safetensors"])
    assert any("model-00002-of-00002.safetensors" in e for e in new.data["errors"])
    other = headers.read_model(RangeFetch({"model.safetensors": header_bytes(T)}), "g/base", ["model.safetensors"])
    assert headers.estimate_class(CFG, new.data, CFG, other.data)["class"] == "unknown"
