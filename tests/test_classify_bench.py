# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""orchard/classify_bench.py: stage 0's class decision on a corpus of models from their Hub headers, set beside independent labels.

The bench has to agree with the classifier it measures. So the central tests are differential: a fake Hub serves the files of small synthetic
models by range request, the bench classifies them from headers, and its class must equal what the real stage 0 template functions
(`compare_configs`, `compare_tensors`, `compare_sidecars`, `detect_restricted_head`, `decide_path`, `decide_class`) give on the same files
read from disk. The rest pin what a labeler sees (no tool verdict in a packet), how the corpus is read and how the comparison is reported.
"""
from __future__ import annotations

import functools
import importlib.util
import json
import struct
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

from orchard import classify_bench as cb                         # noqa: E402
from orchard.research.http import FetchError                      # noqa: E402
from orchard.research import triage_rules                         # noqa: E402

CLASSES = ("weights-only", "weights+sidecar", "restricted-head", "full-port", "unknown")
H, V, LAYERS = 16, 64, 3


def bf16(arr) -> bytes:
    return (np.asarray(arr, dtype="<f4").view("<u4") >> 16).astype("<u2").tobytes()


def write_st(path: Path, tensors: dict) -> None:
    header, blobs, off = {}, [], 0
    for name, arr in tensors.items():
        dtype = "BF16"
        if isinstance(arr, tuple):                          # ("F32", array): a tensor stored in full precision
            dtype, arr = arr
        arr = np.asarray(arr, dtype="<f4")
        data = arr.tobytes() if dtype == "F32" else bf16(arr)
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [off, off + len(data)]}
        blobs.append(data)
        off += len(data)
    raw = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(blobs))


def text_config(**change):
    cfg = {"model_type": "qwen3_5_text", "num_hidden_layers": LAYERS, "hidden_size": H, "num_attention_heads": 2, "num_key_value_heads": 1,
           "head_dim": 8, "layer_types": ["linear_attention", "full_attention", "linear_attention"], "vocab_size": V}
    cfg.update(change)
    return cfg


def backbone(prefix="model.", rng=None, hidden=H):
    rng = rng or np.random.default_rng(0)
    t = {f"{prefix}embed_tokens.weight": rng.normal(size=(V, hidden)), f"{prefix}norm.weight": rng.normal(size=(hidden,))}
    for i in range(LAYERS):
        t[f"{prefix}layers.{i}.mlp.gate_proj.weight"] = rng.normal(size=(32, hidden))
        t[f"{prefix}layers.{i}.input_layernorm.weight"] = rng.normal(size=(hidden,))
    return t


def make_repo(root: Path, repo: str, tensors: dict, *, config=None, extra: dict | None = None, shards=2, meta=None) -> Path:
    d = root / repo
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(config if config is not None else text_config()))
    names = sorted(tensors)
    if names and shards:
        half = max(1, len(names) // shards)
        parts = [names[i:i + half] for i in range(0, len(names), half)]
        weight_map = {}
        for n, part in enumerate(parts, 1):
            fn = f"model-{n:05d}-of-{len(parts):05d}.safetensors"
            write_st(d / fn, {k: tensors[k] for k in part})
            weight_map.update({k: fn for k in part})
        (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    for name, content in (extra or {}).items():
        (d / name).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, dict):                       # a safetensors sidecar: {tensor: array}
            write_st(d / name, content)
        else:
            (d / name).write_bytes(content)
    (d / "_hub.json").write_text(json.dumps(meta or {"license": "apache-2.0", "base_model": "org/base"}))
    return d


class FakeFetch:
    """The two things the bench uses of research.http.Fetcher, over a directory tree standing in for the Hub."""

    def __init__(self, root: Path, gated=()):
        self.root, self.gated, self.requests = Path(root), set(gated), 0

    def _file(self, url):
        rest = url.split("huggingface.co/", 1)[1]
        repo = "/".join(rest.split("/")[:2])
        name = rest.split("/resolve/main/", 1)[1]
        return repo, self.root / repo / name

    def get(self, url, *, byte_range=None, max_bytes=8 << 20):
        self.requests += 1
        if "/api/models/" in url:
            return self._api(url)
        repo, path = self._file(url)
        if repo in self.gated:
            raise FetchError("HTTP 401")
        if not path.is_file():
            raise FetchError("HTTP 404")
        data = path.read_bytes()
        if byte_range is not None:
            data = data[byte_range[0]:byte_range[1] + 1]
        if len(data) > max_bytes:
            raise FetchError(f"response is larger than {max_bytes} bytes")
        return data

    def json(self, url, **kw):
        try:
            return json.loads(self.get(url, **kw).decode("utf-8"))
        except ValueError as exc:
            raise FetchError(f"not JSON: {exc}")

    def _api(self, url):
        repo = url.split("/api/models/", 1)[1].split("?", 1)[0]
        d = self.root / repo
        if not d.is_dir():
            raise FetchError("HTTP 404")
        meta = json.loads((d / "_hub.json").read_text())
        files = [{"rfilename": p.relative_to(d).as_posix(), "size": p.stat().st_size} for p in sorted(d.rglob("*")) if p.is_file() and p.name != "_hub.json"]
        return json.dumps({"id": repo, "sha": "a" * 40, "gated": repo in self.gated and "auto", "downloads": 7, "pipeline_tag": "text-generation",
                           "tags": ["safetensors"], "cardData": {"license": meta.get("license"), "base_model": meta.get("base_model")},
                           "siblings": files}).encode()


@functools.lru_cache(maxsize=1)
def tmpl():
    return triage_rules.load()


def oracle_class(model_dir: Path, base_dir: Path) -> str:
    """What stage 0's own functions say about the same two directories."""
    t = tmpl()
    c = t.compare_configs(model_dir, base_dir)
    tt = t.compare_tensors(model_dir, base_dir)
    names = set(t.read_tensors(model_dir)[1])
    sc = t.compare_sidecars(model_dir, base_dir, names)
    head = t.detect_restricted_head(model_dir, base_dir, sc)
    tt = {**tt, "head": head}
    reasons = t.decide_path(c, tt) + sc["problems"]
    return t.decide_class(c, tt, reasons, sc)


@pytest.fixture
def world(tmp_path):
    rng = np.random.default_rng(1)
    head = rng.normal(size=(V, H)).astype("float32")
    base_t = {**backbone("model.", rng), "lm_head.weight": head}
    hub = tmp_path / "hub"
    hub.mkdir()
    base = make_repo(hub, "org/base", base_t)
    return SimpleNamespace(tmp=tmp_path, hub=hub, base=base, head=head, rng=rng, fetch=FakeFetch(hub))


from types import SimpleNamespace                                  # noqa: E402


def classify(world, repo):
    base = cb.read_base(world.base)
    return cb.collect_one(world.fetch, repo, base)


CASES = {}


def case(name, expect):
    def deco(fn):
        CASES[name] = (fn, expect)
        return fn
    return deco


@case("fine-tune with the same names", "weights-only")
def _ft(w):
    t = {**backbone("model.", np.random.default_rng(5)), "lm_head.weight": w.head * 1.001}
    return make_repo(w.hub, "org/ft", t)


@case("renamed prefixes, readout rows copied from the head, no lm_head", "restricted-head")
def _rh(w):
    ids = [3, 9, 20, 41]
    return make_repo(w.hub, "org/decider", backbone("language_model.", np.random.default_rng(6)), extra={"readout.safetensors": {"weight": w.head[ids]}})


@case("renamed prefixes and no lm_head and no sidecar", "full-port")
def _nolm(w):
    return make_repo(w.hub, "org/nohead", backbone("language_model.", np.random.default_rng(7)))


@case("a sidecar whose rows are not the head's", "full-port")
def _badrows(w):
    return make_repo(w.hub, "org/unrelated", backbone("language_model.", np.random.default_rng(8)),
                     extra={"readout.safetensors": {"weight": np.random.default_rng(9).normal(size=(4, H))}})


@case("same backbone plus an extra head file", "weights+sidecar")
def _side(w):
    t = {**backbone("model.", np.random.default_rng(10)), "lm_head.weight": w.head}
    return make_repo(w.hub, "org/sidecar", t, extra={"task_head.safetensors": {"cls.dense.weight": np.zeros((4, 4))}})


@case("a different hidden size", "full-port")
def _wide(w):
    t = {**backbone("model.", np.random.default_rng(11), hidden=32), "lm_head.weight": np.random.default_rng(12).normal(size=(V, 32))}
    return make_repo(w.hub, "org/wide", t, config=text_config(hidden_size=32))


@case("a gguf and no safetensors", "unknown")
def _gguf(w):
    return make_repo(w.hub, "org/gguf", {}, extra={"model-q4.gguf": b"GGUF" + b"\0" * 64})


@case("an adapter-only repo", "unknown")
def _lora(w):
    return make_repo(w.hub, "org/lora", {}, extra={"adapter_model.safetensors": {"base_model.model.layers.0.q.lora_A.weight": np.zeros((4, H)),
                                                                                  "base_model.model.layers.0.q.lora_B.weight": np.zeros((H, 4))}})


@case("a full-height sidecar and no lm_head", "full-port")
def _tall(w):
    return make_repo(w.hub, "org/tall", backbone("language_model.", np.random.default_rng(13)), extra={"readout.safetensors": {"weight": w.head}})


@case("its own lm_head and a readout-shaped extra file", "weights+sidecar")
def _own(w):
    t = {**backbone("model.", np.random.default_rng(14)), "lm_head.weight": w.head}
    return make_repo(w.hub, "org/ownhead", t, extra={"readout.safetensors": {"weight": w.head[[1, 2, 3, 4]]}})


@case("an extra file as wide as the head is not wider", "weights+sidecar")
def _narrow(w):
    t = {**backbone("model.", np.random.default_rng(15)), "lm_head.weight": w.head}
    return make_repo(w.hub, "org/narrow", t, extra={"aux.safetensors": {"weight": np.zeros((4, H + 1))}})


@case("small tensors stored in float32", "full-port")
def _f32(w):
    t = {**backbone("model.", np.random.default_rng(23)), "lm_head.weight": w.head}
    t["model.norm.weight"] = ("F32", t["model.norm.weight"])
    return make_repo(w.hub, "org/f32norm", t)


@case("a drafter in a subdirectory", "weights-only")
def _subdir(w):
    t = {**backbone("model.", np.random.default_rng(22)), "lm_head.weight": w.head}
    return make_repo(w.hub, "org/withdrafter", t, extra={"dflash2/model.safetensors": {"fc.weight": np.zeros((4, 4))}})


@case("no lm_head and a sidecar of another width", "full-port")
def _width(w):
    return make_repo(w.hub, "org/oddwidth", backbone("language_model.", np.random.default_rng(21)),
                     extra={"readout.safetensors": {"weight": np.zeros((4, H + 1))}})


@case("a config.json that is not JSON", "unknown")
def _badcfg(w):
    d = make_repo(w.hub, "org/badcfg", {**backbone("model."), "lm_head.weight": w.head})
    (d / "config.json").write_text("{not json")
    return d


@case("no config.json", "unknown")
def _noconfig(w):
    d = make_repo(w.hub, "org/noconfig", {**backbone("model."), "lm_head.weight": w.head})
    (d / "config.json").unlink()
    return d


@case("an extra file whose names overlap the backbone", "unknown")
def _overlap(w):
    t = {**backbone("model.", np.random.default_rng(16)), "lm_head.weight": w.head}
    return make_repo(w.hub, "org/overlap", t, extra={"copy.safetensors": {"model.norm.weight": np.zeros((H,))}})


@case("an extra file that is not safetensors", "unknown")
def _pt(w):
    t = {**backbone("model.", np.random.default_rng(17)), "lm_head.weight": w.head}
    return make_repo(w.hub, "org/ptfile", t, extra={"extra_head.pt": b"PK\x03\x04" + b"\0" * 32})


@case("pytorch .bin copies of the weights beside the safetensors", "weights-only")
def _bins(w):
    t = {**backbone("model.", np.random.default_rng(18)), "lm_head.weight": w.head}
    return make_repo(w.hub, "org/withbins", t, extra={"pytorch_model-00001-of-00002.bin": b"x", "pytorch_model-00002-of-00002.bin": b"y"})


@pytest.mark.parametrize("name", sorted(CASES))
def test_the_bench_gives_the_class_stage_0s_own_functions_give_on_the_same_files(world, name):
    build, expect = CASES[name]
    d = build(world)
    repo = f"org/{d.name}"
    got = classify(world, repo)
    assert got["tool"]["class"] == oracle_class(d, world.base), got["tool"]
    assert got["tool"]["class"] == expect, got["tool"]


@pytest.mark.parametrize("name", ["a full-height sidecar and no lm_head", "its own lm_head and a readout-shaped extra file",
                                  "an extra file as wide as the head is not wider", "no lm_head and a sidecar of another width",
                                  "same backbone plus an extra head file", "a gguf and no safetensors"])
def test_a_sidecar_that_cannot_be_a_cut_down_head_costs_no_download(world, name):
    d = CASES[name][0](world)
    got = classify(world, f"org/{d.name}")
    assert got["tool"]["downloaded_bytes"] == 0 and got["tool"]["head"]["restricted"] is False


def test_the_reasons_say_why_a_config_or_a_sidecar_could_not_be_read(world):
    d = CASES["a config.json that is not JSON"][0](world)
    reasons = " ".join(classify(world, f"org/{d.name}")["tool"]["reasons"])
    assert "config.json" in reasons and "not JSON" in reasons
    d = CASES["an extra file that is not safetensors"][0](world)
    reasons = " ".join(classify(world, f"org/{d.name}")["tool"]["reasons"])
    assert "extra_head.pt is not a safetensors file" in reasons


def test_a_weights_file_in_a_subdirectory_is_not_a_sidecar_because_stage_0_looks_only_at_the_top_level(world):
    """bytkim/Qwen3.8-27B-pi ships the DFlash2 drafter as dflash2/model.safetensors. Stage 0 scans the snapshot's top-level files, so it is not
    a sidecar; the bench listed every path in the repo and called the model weights+sidecar."""
    d = CASES["a drafter in a subdirectory"][0](world)
    got = classify(world, f"org/{d.name}")
    assert got["tool"]["sidecars"] == [] and got["tool"]["class"] == "weights-only" == oracle_class(d, world.base)


def test_a_repo_with_more_shards_than_the_preflight_estimator_reads_is_read_in_full_by_the_bench(world):
    """The official Qwen3.8-27B-FP8 ships 66 per-layer files and the estimator reads at most 64: the first bench said `unknown` for what was only
    an unread repo. The bench raises its own limit and says when evidence is still incomplete."""
    t = {**backbone("model.", np.random.default_rng(24)), "lm_head.weight": world.head}
    for i in range(80):
        t[f"model.layers.{10 + i}.mlp.gate_proj.weight"] = np.zeros((4, 4))
    d = make_repo(world.hub, "org/manyshards", t, shards=75)
    got = classify(world, "org/manyshards")
    assert len([p for p in d.iterdir() if p.name.startswith("model-")]) > 64
    assert got["tool"]["class"] == oracle_class(d, world.base) == "full-port" and got["tool"]["evidence_complete"] is True
    assert got["packet"]["tensors"]["model_count"] == len(t) and got["packet"]["tensors"]["read_errors"] == []


def test_a_config_larger_than_the_default_body_cap_is_read(world):
    cfg = text_config()
    cfg["quantization_config"] = {"targets": ["layer.%d.mixer.proj" % i for i in range(500000)]}
    assert len(json.dumps(cfg)) > 8 * 1024 * 1024
    t = {**backbone("model.", np.random.default_rng(25)), "lm_head.weight": world.head}
    d = make_repo(world.hub, "org/bigconfig", t, config=cfg)
    got = classify(world, "org/bigconfig")
    assert got["tool"]["class"] == oracle_class(d, world.base) == "weights-only" and got["tool"]["evidence_complete"] is True


def test_evidence_that_could_not_be_read_is_flagged_in_the_decision_and_the_packet(world):
    t = {**backbone("model.", np.random.default_rng(26)), "lm_head.weight": world.head}
    d = make_repo(world.hub, "org/lostshard", t)
    sorted(d.glob("model-0000*-of-*.safetensors"))[1].unlink()         # the index still lists it
    lost = [x for x in d.glob("model-0000*-of-*.safetensors")]
    got = classify(world, "org/lostshard")
    assert got["tool"]["evidence_complete"] is False and "safetensors" in " ".join(got["tool"]["incomplete_because"])
    assert "evidence is incomplete" in cb.render_packet(got["packet"])


def test_files_the_bench_cannot_read_by_design_are_not_a_gap_in_its_evidence(world):
    """A gguf or a .pt cannot be read from a header at all. That is a fact about the repo, and the class is unknown because of it. `evidence_complete`
    is for what the bench could have read and did not (a failed fetch, a repo over its limits)."""
    for name in ("a gguf and no safetensors", "an extra file that is not safetensors"):
        d = CASES[name][0](world)
        got = classify(world, f"org/{d.name}")
        assert got["tool"]["evidence_complete"] is True, (name, got["tool"]["incomplete_because"])
        assert "evidence is incomplete" not in cb.render_packet(got["packet"])


def test_a_sidecar_header_that_failed_to_fetch_is_a_gap(world):
    d = make_repo(world.hub, "org/badside", {**backbone("model."), "lm_head.weight": world.head}, extra={"aux.safetensors": b"\x01\x02"})
    got = classify(world, "org/badside")
    assert got["tool"]["evidence_complete"] is False and "aux.safetensors cannot be read:" in " ".join(got["tool"]["incomplete_because"])


def test_a_file_the_nearest_model_also_has_is_not_a_sidecar(world):
    (world.base / "vision_tower.safetensors").write_bytes(b"x")
    t = {**backbone("model.", np.random.default_rng(19)), "lm_head.weight": world.head}
    d = make_repo(world.hub, "org/withvision", t, extra={"vision_tower.safetensors": b"x"})
    got = classify(world, "org/withvision")
    assert got["tool"]["class"] == "weights-only" == oracle_class(d, world.base) and got["tool"]["sidecars"] == []


def test_a_sidecar_over_the_download_cap_is_not_downloaded_and_the_head_is_left_unconfirmed(world):
    ids = [3, 9, 20, 41]
    d = make_repo(world.hub, "org/bigside", backbone("language_model.", np.random.default_rng(20)),
                  extra={"readout.safetensors": {"weight": world.head[ids], "pad": np.zeros((cb.SIDECAR_DOWNLOAD_MAX,), dtype="float32")}})
    assert (d / "readout.safetensors").stat().st_size > cb.SIDECAR_DOWNLOAD_MAX
    got = classify(world, "org/bigside")
    assert got["tool"]["downloaded_bytes"] == 0 and got["tool"]["head"]["restricted"] is None
    assert "cap" in got["tool"]["head"]["note"] and got["tool"]["class"] == "full-port"


def test_the_restricted_head_is_confirmed_from_the_sidecar_alone_with_the_base_head_local(world):
    d = CASES["renamed prefixes, readout rows copied from the head, no lm_head"][0](world)
    got = classify(world, f"org/{d.name}")
    head = got["tool"]["head"]
    assert head["restricted"] is True and head["rows"] == 4 and head["token_ids"] == [3, 9, 20, 41]
    # nothing but the headers, the small sidecar and the config was requested: no shard body
    assert got["tool"]["downloaded_bytes"] < 64 * 1024


def test_a_repo_the_hub_will_not_serve_is_skipped_not_classified(world):
    make_repo(world.hub, "org/private", backbone("model."))
    world.fetch.gated.add("org/private")
    got = classify(world, "org/private")
    assert got["tool"]["class"] is None and got["tool"]["status"] == "skipped" and "401" in got["tool"]["why"]
    assert got["packet"]["status"] == "skipped"


def test_an_unknown_repo_is_skipped(world):
    got = classify(world, "org/missing")
    assert got["tool"]["status"] == "skipped" and got["tool"]["class"] is None


# ---- what a labeler is given -------------------------------------------------------------------------------------------------------

FORBIDDEN_KEYS = {"class", "verdict", "decision", "tool", "estimate", "restricted"}


def walk_keys(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield k
            yield from walk_keys(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_keys(v)


def test_a_packet_carries_no_verdict_of_the_tool(world):
    d = CASES["renamed prefixes, readout rows copied from the head, no lm_head"][0](world)
    got = classify(world, f"org/{d.name}")
    assert not (set(walk_keys(got["packet"])) & FORBIDDEN_KEYS), set(walk_keys(got["packet"])) & FORBIDDEN_KEYS
    text = cb.render_packet(got["packet"])
    for word in ("restricted-head", "weights+sidecar", "full-port"):
        assert word not in text
    assert "weights-only" not in text


def test_a_packet_states_the_evidence_a_person_would_look_at(world):
    d = CASES["renamed prefixes, readout rows copied from the head, no lm_head"][0](world)
    p = classify(world, f"org/{d.name}")["packet"]
    text = cb.render_packet(p)
    assert "org/decider" in text and "language_model" in text and "readout.safetensors" in text
    assert "lm_head" in text and "weight [4, 16]" in text
    assert p["tensors"]["model_count"] == len(backbone("language_model.")) and p["tensors"]["base_count"] == len(backbone("model.")) + 1
    assert p["tensors"]["aligned_rules"] and p["tensors"]["only_in_base"] == ["lm_head.weight"]


def test_a_packet_shows_which_dtypes_differ_from_the_nearest_models(world):
    """The first pilot's labelers could not tell whether 96 float32 tensors matched the nearest model: the packet said they were F32 and not
    what the nearest model stores. It now names the tensors and both dtypes."""
    d = CASES["small tensors stored in float32"][0](world)
    p = classify(world, f"org/{d.name}")["packet"]
    assert p["tensors"]["dtype_differences"] == ["model.norm.weight F32 here, BF16 in the nearest model"]
    assert p["tensors"]["dtype_difference_count"] == 1
    text = cb.render_packet(p)
    assert "1 tensors differ in dtype" in text and "model.norm.weight F32 here, BF16 in the nearest model" in text


def test_a_packet_is_small_enough_to_read(world):
    for name, (build, _) in CASES.items():
        d = build(world)
        text = cb.render_packet(classify(world, f"org/{d.name}")["packet"])
        assert len(text) < 9000, (name, len(text))


# ---- the corpus ------------------------------------------------------------------------------------------------------------------

def test_the_corpus_is_validated(tmp_path):
    good = [{"id": "org/a", "stratum": "fine-tune"}, {"id": "org/b", "stratum": "decider-like", "truth": "restricted-head", "note": "known"}]
    p = tmp_path / "c.json"
    p.write_text(json.dumps(good))
    assert [m["id"] for m in cb.load_corpus(p)] == ["org/a", "org/b"]
    for bad in ([{"id": "not an id"}], [{"id": "org/a"}, {"id": "org/a"}], [{"id": "org/a", "truth": "nonsense"}], [], {"id": "org/a"}):
        p.write_text(json.dumps(bad))
        with pytest.raises(cb.CorpusError):
            cb.load_corpus(p)


def test_slugs_are_filesystem_safe_and_reversible_enough_to_read():
    assert cb.slug("perplexity-ai/pplx-decider-v1.2") == "perplexity-ai--pplx-decider-v1.2"


# ---- the comparison --------------------------------------------------------------------------------------------------------------

def tool_of(**classes):
    return {s: {"class": c, "status": "classified" if c else "skipped", "reasons": [f"{s} reason"]} for s, c in classes.items()}


def labels_of(**classes):
    return {s: {"class": c, "confidence": "high", "why": f"{s} why"} for s, c in classes.items()}


def test_agreement_is_counted_per_labeler_and_against_the_majority():
    tool = tool_of(a="weights-only", b="full-port", c="restricted-head", d=None)
    l1 = labels_of(a="weights-only", b="weights-only", c="restricted-head", d="unknown")
    l2 = labels_of(a="weights-only", b="weights-only", c="full-port", d="unknown")
    l3 = labels_of(a="weights-only", b="full-port", c="restricted-head", d="unknown")
    rep = cb.compare(tool, {"x": l1, "y": l2, "z": l3}, truth={"c": "restricted-head", "b": "weights-only"})
    assert rep["n_compared"] == 3 and rep["n_skipped"] == 1                           # d was skipped by the tool
    assert rep["tool_vs_majority"] == {"agree": 2, "of": 3}                             # a and c agree; b: majority weights-only (2 of 3) against tool full-port
    by = {r["id"]: r for r in rep["rows"]}
    assert by["b"]["majority"] == "weights-only" and by["b"]["tool"] == "full-port" and by["b"]["flag"] == "tool differs from the majority"
    assert by["c"]["flag"] == "labelers disagree" and by["a"]["flag"] == ""
    assert rep["truth"]["tool"] == {"n": 2, "correct": 1}                              # right on c, wrong on b
    assert rep["truth"]["labelers"] == {"x": {"n": 2, "correct": 2}, "y": {"n": 2, "correct": 1}, "z": {"n": 2, "correct": 1}}
    assert rep["confusion"]["weights-only"]["full-port"] == 1


def test_a_tie_among_labelers_has_no_majority_and_is_flagged():
    tool = tool_of(a="weights-only")
    rep = cb.compare(tool, {"x": labels_of(a="weights-only"), "y": labels_of(a="full-port")})
    row = rep["rows"][0]
    assert row["majority"] is None and row["flag"] == "labelers disagree"


def test_labels_with_an_unknown_class_or_for_an_unknown_model_are_rejected(tmp_path):
    tool = tool_of(a="weights-only")
    with pytest.raises(cb.CorpusError):
        cb.compare(tool, {"x": {"a": {"class": "nonsense", "confidence": "high", "why": "x"}}})
    with pytest.raises(cb.CorpusError):
        cb.compare(tool, {"x": {"zzz": {"class": "unknown", "confidence": "high", "why": "x"}}})


def test_the_report_names_every_disagreement_with_both_reasons():
    tool = tool_of(a="weights-only", b="full-port")
    rep = cb.compare(tool, {"x": labels_of(a="weights-only", b="weights-only")})
    text = cb.render_report(rep)
    assert "b" in text and "b reason" in text and "b why" in text and "full-port" in text
    assert "a why" not in text                                                        # agreements are counted, not listed
