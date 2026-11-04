# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Issue 440 (a slice of 439): stage 0 reads the header of each GGUF file it ignores and writes evidence/gguf-header.json.

The reader lives in the stage 0 template (orchard/skills/delta-triage-templates/delta_triage.py), which uses only the standard library
because it is copied into a run directory. Everything here builds small GGUF files in tmp directories: a header, a tensor table and a
few zero bytes of data. No real weights are used, and the tests need no `tokenizers` package.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import struct
import threading
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
TEMPLATE = REPO / "orchard" / "skills" / "delta-triage-templates" / "delta_triage.py"

T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
F32, F16, Q8_0, Q4_K, Q6_K, BF16 = 0, 1, 8, 12, 14, 30


def module():
    spec = importlib.util.spec_from_file_location("delta_triage_gguf_under_test", TEMPLATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- a GGUF builder -------------------------------------------------------------------------------

def s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def value(typ: int, v) -> bytes:
    fmt = {T_U8: "<B", T_I8: "<b", T_U16: "<H", T_I16: "<h", T_U32: "<I", T_I32: "<i", T_F32: "<f", T_BOOL: "<?",
           T_U64: "<Q", T_I64: "<q", T_F64: "<d"}
    if typ in fmt:
        return struct.pack(fmt[typ], v)
    if typ == T_STR:
        return s(v)
    if typ == T_ARR:
        elem, items = v
        return struct.pack("<IQ", elem, len(items)) + b"".join(value(elem, x) for x in items)
    raise AssertionError(typ)


def kv(key: str, typ: int, v) -> bytes:
    return s(key) + struct.pack("<I", typ) + value(typ, v)


def tensor_info(name: str, dims, type_id: int, offset: int) -> bytes:
    return s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", type_id, offset)


def gguf(kvs=(), tensors=(), *, version=3, align=32, data=True) -> bytes:
    """kvs: ready-made kv byte strings. tensors: (name, dims, type_id) or (name, dims, type_id, offset). The data section holds 32 zero
    bytes per tensor, and a tensor's offset is 32 times its index unless it is given."""
    head = b"GGUF" + struct.pack("<I", version) + struct.pack("<QQ", len(tensors), len(kvs))
    head += b"".join(kvs) + b"".join(tensor_info(x[0], x[1], x[2], x[3] if len(x) > 3 else i * 32) for i, x in enumerate(tensors))
    pad = (-len(head)) % align
    return head + b"\0" * pad + (b"\0" * (32 * len(tensors)) if data else b"")


TEMPLATE_TEXT = "{% for m in messages %}{{ m.content }}{% endfor %}"


def hybrid_kvs(**over):
    """The header of a small hybrid-attention model, in the shape a real Qwen3.5 header has (issue 440's real file)."""
    d = {
        "general.architecture": (T_STR, "qwen35"), "general.name": (T_STR, "Tiny-Hybrid"), "general.file_type": (T_U32, 15),
        "qwen35.block_count": (T_U32, 8), "qwen35.context_length": (T_U32, 262144), "qwen35.embedding_length": (T_U32, 64),
        "qwen35.feed_forward_length": (T_U32, 128), "qwen35.attention.head_count": (T_U32, 4),
        "qwen35.attention.head_count_kv": (T_U32, 2), "qwen35.attention.key_length": (T_U32, 16),
        "qwen35.attention.value_length": (T_U32, 16), "qwen35.rope.freq_base": (T_F32, 10000000.0),
        "qwen35.rope.dimension_count": (T_U32, 4), "qwen35.rope.dimension_sections": (T_ARR, (T_I32, [1, 1, 0, 0])),
        "qwen35.attention.layer_norm_rms_epsilon": (T_F32, 1e-06), "qwen35.nextn_predict_layers": (T_U32, 0),
        "qwen35.ssm.conv_kernel": (T_U32, 4), "qwen35.ssm.state_size": (T_U32, 16), "qwen35.ssm.group_count": (T_U32, 2),
        "qwen35.ssm.time_step_rank": (T_U32, 4), "qwen35.ssm.inner_size": (T_U32, 64),
        "qwen35.full_attention_interval": (T_U32, 4),
        "qwen35.attention.recurrent_layers": (T_ARR, (T_BOOL, [True, True, True, False, True, True, True, False])),
        "tokenizer.ggml.model": (T_STR, "gpt2"), "tokenizer.ggml.pre": (T_STR, "qwen35"),
        "tokenizer.ggml.tokens": (T_ARR, (T_STR, [f"t{i}" for i in range(300)])),
        "tokenizer.ggml.token_type": (T_ARR, (T_I32, [1] * 300)),
        "tokenizer.chat_template": (T_STR, TEMPLATE_TEXT),
    }
    d.update(over)
    return [kv(k, t, v) for k, (t, v) in d.items() if t is not None]


def hybrid_tensors():
    out = [("token_embd.weight", [64, 300], Q6_K), ("output_norm.weight", [64], F32)]
    for i in range(8):
        out += [(f"blk.{i}.attn_norm.weight", [64], F32), (f"blk.{i}.ffn_up.weight", [64, 128], Q4_K),
                (f"blk.{i}.ffn_down.weight", [128, 64], Q6_K if i % 2 == 0 else Q4_K)]
        if i % 4 != 3:
            out += [(f"blk.{i}.ssm_a", [4], F32), (f"blk.{i}.ssm_out.weight", [64, 64], Q4_K)]
    return out


def write(tmp_path: Path, data: bytes, name="m.gguf") -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


# ---- the reader -----------------------------------------------------------------------------------

def test_a_valid_header_is_read(tmp_path):
    m = module()
    h = m.read_gguf_header(write(tmp_path, gguf(hybrid_kvs(), hybrid_tensors())))
    assert h["version"] == 3 and h["kv"]["general.architecture"] == "qwen35" and h["kv"]["qwen35.block_count"] == 8
    assert h["kv"]["qwen35.attention.recurrent_layers"][:4] == [True, True, True, False]
    assert h["kv"]["qwen35.rope.dimension_sections"] == [1, 1, 0, 0]
    assert h["kv"]["tokenizer.ggml.tokens"] == {"array_len": 300}          # a long array is counted and not kept
    assert h["kv"]["tokenizer.chat_template"] == TEMPLATE_TEXT
    assert len(h["tensors"]) == len(hybrid_tensors()) and h["tensors"][0] == ("token_embd.weight", [64, 300], "Q6_K")


def test_a_short_numeric_array_is_kept_and_a_long_one_or_a_string_array_is_counted(tmp_path):
    m = module()
    kvs = [kv("a.short", T_ARR, (T_I32, list(range(10)))), kv("a.long", T_ARR, (T_I32, [1] * (m.GGUF_KEEP_ARRAY + 1))),
           kv("a.strings", T_ARR, (T_STR, ["x", "y"]))]
    h = m.read_gguf_header(write(tmp_path, gguf(kvs, [])))
    assert h["kv"]["a.short"] == list(range(10)) and h["kv"]["a.long"] == {"array_len": m.GGUF_KEEP_ARRAY + 1}
    assert h["kv"]["a.strings"] == {"array_len": 2}


def test_version_2_is_read_and_other_versions_are_refused(tmp_path):
    m = module()
    assert m.read_gguf_header(write(tmp_path, gguf(hybrid_kvs(), hybrid_tensors(), version=2)))["version"] == 2
    for v in (0, 1, 4, 99):
        with pytest.raises(m.GgufError, match="version"):
            m.read_gguf_header(write(tmp_path, gguf(hybrid_kvs(), hybrid_tensors(), version=v), f"v{v}.gguf"))


def test_an_unknown_tensor_type_is_named_not_refused(tmp_path):
    m = module()
    h = m.read_gguf_header(write(tmp_path, gguf(hybrid_kvs(), [("a", [4], 777)])))
    assert h["tensors"] == [("a", [4], "type_777")]


GOOD = gguf(hybrid_kvs(), hybrid_tensors())


def mutate(at: int, new: bytes) -> bytes:
    return GOOD[:at] + new + GOOD[at + len(new):]


MALFORMED = {
    "not gguf": (b"GGUX" + GOOD[4:], "GGUF"),
    "empty": (b"", "GGUF"),
    "shorter than the fixed lead": (GOOD[:10], "short"),
    "cut inside a key": (GOOD[:60], "short"),
    "cut inside the tensor table": (GOOD[:GOOD.index(b"token_embd.weight") + 8], "short"),
    "kv count above the cap": (GOOD[:16] + struct.pack("<Q", 10**9) + GOOD[24:], "key-value count"),
    "tensor count above the cap": (GOOD[:8] + struct.pack("<Q", 10**9) + GOOD[16:], "tensor count"),
    "a key longer than the cap": (b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 10**6) + b"k" * 10**6, "too long"),
    "a string length beyond the file": (b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + s("general.name") + struct.pack("<I", T_STR)
                                        + struct.pack("<Q", 2**40) + b"x", "string"),
    "a string length that wraps": (b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + s("general.name") + struct.pack("<I", T_STR)
                                    + struct.pack("<Q", 2**64 - 1) + b"x", "string"),
    "an array count above the cap": (b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + s("a") + struct.pack("<I", T_ARR)
                                      + struct.pack("<IQ", T_U32, 10**12), "more than"),
    "an array of 5 million strings, above the cap and inside the element budget": (
        b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + s("a") + struct.pack("<I", T_ARR) + struct.pack("<IQ", T_STR, 5_000_000), "more than"),
    "a numeric array that runs past the file": (b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + s("a") + struct.pack("<I", T_ARR)
                                                 + struct.pack("<IQ", T_U32, 3_000_000), "array"),
    "an array of arrays": (b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + s("a") + struct.pack("<I", T_ARR)
                            + struct.pack("<IQ", T_ARR, 1), "array"),
    "an unknown value type": (b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + s("a") + struct.pack("<I", 99), "value type"),
    "a repeated key": (gguf([kv("a", T_U32, 1), kv("a", T_U32, 2)]), "repeated"),
    "a repeated tensor name": (gguf([], [("t", [4], F32), ("t", [4], F32)]), "repeated"),
    "a tensor with no dimensions": (gguf([], [("t", [], F32)]), "dimension"),
    "a tensor with too many dimensions": (gguf([], [("t", [1] * 9, F32)]), "dimension"),
    "a tensor name over the cap": (gguf([], [("t" * 5000, [4], F32)]), "too long"),
    "a tensor offset past the data": (gguf([], [("t", [4], F32, 10**9)]), "offset"),
    "a zero alignment": (gguf([kv("general.alignment", T_U32, 0)], [("t", [4], F32)]), "alignment"),
    "an alignment that is not a power of two": (gguf([kv("general.alignment", T_U32, 24)], [("t", [4], F32)]), "alignment"),
    "a key that is not UTF-8": (b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + struct.pack("<Q", 2) + b"\xff\xfe" + struct.pack("<II", T_U32, 1), "UTF-8"),
}


@pytest.mark.parametrize("label", MALFORMED)
def test_a_malformed_header_is_one_named_refusal(tmp_path, label):
    m = module()
    data, word = MALFORMED[label]
    with pytest.raises(m.GgufError) as exc:
        m.read_gguf_header(write(tmp_path, data))
    assert word.lower() in str(exc.value).lower(), str(exc.value)


def test_a_header_over_the_size_cap_is_refused_without_reading_past_it(tmp_path):
    m = module()
    big = gguf([kv("general.name", T_STR, "x" * 5000)], [])
    with pytest.raises(m.GgufError, match="larger than"):
        m.read_gguf_header(write(tmp_path, big), max_header=2000)


def test_a_string_over_the_string_cap_is_refused_even_inside_the_header_cap(tmp_path):
    m = module()
    with pytest.raises(m.GgufError, match="string"):
        m.read_gguf_header(write(tmp_path, gguf([kv("general.name", T_STR, "x" * (m.GGUF_MAX_STRING + 1))], [])))


def test_the_reader_never_reads_past_the_header(tmp_path, monkeypatch):
    """A header and a sparse 8 GiB of 'tensor data': every physical read ends inside the parsed header (review of PR 441, round 1: a
    read-ahead of 1 MiB used to pull in tensor bytes after a small header)."""
    m = module()
    p = write(tmp_path, gguf(hybrid_kvs(), hybrid_tensors()))
    os.truncate(p, 8 << 30)
    seen = []
    real = os.pread
    monkeypatch.setattr(os, "pread", lambda fd, n, off: (seen.append((off, n)), real(fd, n, off))[1])
    out = m.read_gguf_header(p)
    assert out["version"] == 3 and seen
    assert max(off + n for off, n in seen) <= out["header_bytes"], (max(off + n for off, n in seen), out["header_bytes"])
    assert sum(n for _, n in seen) <= out["header_bytes"]          # no byte is asked for twice


def test_no_read_goes_past_a_small_header_cap_even_when_the_file_is_large(tmp_path, monkeypatch):
    """The cap bounds the bytes asked of the file, not only the fields that are used (review of PR 441, round 1)."""
    m = module()
    p = write(tmp_path, gguf([kv("general.name", T_STR, "x" * 5000)], []))
    os.truncate(p, 8 << 30)
    seen = []
    real = os.pread
    monkeypatch.setattr(os, "pread", lambda fd, n, off: (seen.append((off, n)), real(fd, n, off))[1])
    with pytest.raises(m.GgufError, match="larger than"):
        m.read_gguf_header(p, max_header=2000)
    assert seen and max(off + n for off, n in seen) <= 2000


def test_a_hostile_length_is_refused_before_any_large_read(tmp_path, monkeypatch):
    m = module()
    data = MALFORMED["a string length beyond the file"][0]
    seen = []
    real = os.pread
    monkeypatch.setattr(os, "pread", lambda fd, n, off: (seen.append(n), real(fd, n, off))[1])
    with pytest.raises(m.GgufError):
        m.read_gguf_header(write(tmp_path, data))
    assert max(seen) <= 4096


def test_a_link_to_a_regular_file_is_followed_because_a_snapshot_is_made_of_links(tmp_path):
    """A Hugging Face snapshot directory holds links into the cache's blob store: every real GGUF file there is a link."""
    m = module()
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    (blobs / "abc123").write_bytes(GOOD)
    snap = tmp_path / "snapshots" / "rev"
    snap.mkdir(parents=True)
    (snap / "m.gguf").symlink_to("../../blobs/abc123")
    assert m.read_gguf_header(snap / "m.gguf")["version"] == 3
    e = m.gguf_file_evidence(snap / "m.gguf", None, None)
    assert e["status"] == "ok" and e["size"] == len(GOOD)


def test_a_link_to_a_device_a_directory_a_fifo_and_a_missing_file_are_refused(tmp_path):
    m = module()
    link = tmp_path / "link.gguf"
    link.symlink_to(tmp_path / "dir.gguf")
    zero = tmp_path / "zero.gguf"
    zero.symlink_to("/dev/zero")
    fifo = tmp_path / "fifo.gguf"
    os.mkfifo(fifo)
    (tmp_path / "dir.gguf").mkdir()
    outcome = {}

    def attempt(p):
        try:
            m.read_gguf_header(p)
            outcome[p.name] = "read"
        except m.GgufError as exc:
            outcome[p.name] = str(exc)

    ts = [threading.Thread(target=attempt, args=(p,), daemon=True) for p in (link, zero, fifo, tmp_path / "dir.gguf", tmp_path / "none.gguf")]
    [t.start() for t in ts]
    [t.join(10) for t in ts]
    assert not any(t.is_alive() for t in ts), "a FIFO blocked the reader"
    assert set(outcome) == {"link.gguf", "zero.gguf", "fifo.gguf", "dir.gguf", "none.gguf"}
    assert all(v != "read" for v in outcome.values()), outcome


def test_refusal_reasons_never_hold_the_files_content(tmp_path):
    m = module()
    secret = "hf_" + "A" * 30
    data = gguf([kv(secret, T_U32, 1), kv(secret, T_U32, 2)])
    with pytest.raises(m.GgufError) as exc:
        m.read_gguf_header(write(tmp_path, data))
    assert secret not in str(exc.value)


# ---- the evidence for one file --------------------------------------------------------------------

def entry_of(tmp_path, kvs=None, tensors=None, name="m.gguf", config=None, template=None, mtp_tensors=None):
    m = module()
    p = write(tmp_path, gguf(hybrid_kvs() if kvs is None else kvs, hybrid_tensors() if tensors is None else tensors), name)
    return m, m.gguf_file_evidence(p, config, template, mtp_tensors)


def test_the_evidence_summarizes_the_header(tmp_path):
    m, e = entry_of(tmp_path)
    assert e["status"] == "ok" and e["file"] == "m.gguf" and e["role"] == "language model" and e["gguf_version"] == 3
    assert e["architecture"] == "qwen35" and e["tensor_count"] == len(hybrid_tensors()) and e["file_type"] == 15
    assert e["hyperparameters"]["qwen35.block_count"] == 8 and e["hyperparameters"]["qwen35.ssm.state_size"] == 16
    assert e["tokenizer"] == {"model": "gpt2", "pre": "qwen35", "tokens": 300, "merges": None,
                              "chat_template_sha256": hashlib.sha256(TEMPLATE_TEXT.encode()).hexdigest(),
                              "chat_template_chars": len(TEMPLATE_TEXT)}
    assert TEMPLATE_TEXT not in json.dumps(e)                               # the template is hashed, not copied
    assert e["nextn_predict_layers"] == 0 and e["tied_embeddings"] is True


def test_the_quantization_histogram_folds_layer_numbers(tmp_path):
    m, e = entry_of(tmp_path)
    q = e["quantization"]
    assert q["by_type"] == {"F32": 15, "Q4_K": 18, "Q6_K": 5}                  # 38 tensors: 1 + 8 + 6 norms and state, 8 + 4 + 6, 1 + 4
    kinds = q["by_kind"]
    assert kinds["blk.N.ffn_down.weight"] == {"Q4_K": 4, "Q6_K": 4} and kinds["blk.N.ffn_up.weight"] == {"Q4_K": 8}
    assert kinds["token_embd.weight"] == {"Q6_K": 1} and kinds["blk.N.ssm_a"] == {"F32": 6}
    assert sum(sum(v.values()) for v in kinds.values()) == e["tensor_count"]


def test_full_precision_kinds_are_listed_with_their_type_and_count(tmp_path):
    m, e = entry_of(tmp_path)
    fp = {(x["kind"], x["type"]): x["count"] for x in e["quantization"]["full_precision"]}
    assert fp == {("output_norm.weight", "F32"): 1, ("blk.N.attn_norm.weight", "F32"): 8, ("blk.N.ssm_a", "F32"): 6}
    assert "quantized" in e["quantization"]["note"] and "hint" in e["quantization"]["note"]


def test_a_kind_that_is_half_full_precision_is_listed_for_the_precision_it_has(tmp_path):
    m, e = entry_of(tmp_path, tensors=[("blk.0.x.weight", [4], F16), ("blk.1.x.weight", [4], Q4_K)])
    assert e["quantization"]["by_kind"]["blk.N.x.weight"] == {"F16": 1, "Q4_K": 1}
    assert [(x["type"], x["count"]) for x in e["quantization"]["full_precision"]] == [("F16", 1)]


def test_an_output_tensor_means_the_embeddings_are_not_tied(tmp_path):
    m, e = entry_of(tmp_path, tensors=hybrid_tensors() + [("output.weight", [64, 300], Q6_K)])
    assert e["tied_embeddings"] is False


def test_a_mtp_count_is_reported_and_a_missing_one_is_none(tmp_path):
    _, e = entry_of(tmp_path, kvs=hybrid_kvs(**{"qwen35.nextn_predict_layers": (T_U32, 1)}))
    assert e["nextn_predict_layers"] == 1
    _, e = entry_of(tmp_path, kvs=hybrid_kvs(**{"qwen35.nextn_predict_layers": (None, None)}), name="b.gguf")
    assert e["nextn_predict_layers"] is None


def test_a_vision_projector_is_named_and_keeps_its_clip_keys(tmp_path):
    kvs = [kv("general.architecture", T_STR, "clip"), kv("general.type", T_STR, "mmproj"), kv("clip.vision.block_count", T_U32, 24),
           kv("clip.vision.embedding_length", T_U32, 1024), kv("clip.projector_type", T_STR, "qwen3vl_merger")]
    _, e = entry_of(tmp_path, kvs=kvs, tensors=[("v.blk.0.ln1.weight", [4], F32), ("mm.0.weight", [4, 4], BF16)])
    assert e["status"] == "ok" and e["role"] == "vision projector" and e["architecture"] == "clip"
    assert e["hyperparameters"]["clip.vision.block_count"] == 24 and e["hyperparameters"]["clip.projector_type"] == "qwen3vl_merger"


def test_an_unreadable_header_is_recorded_with_a_reason_and_does_not_raise(tmp_path):
    m = module()
    p = write(tmp_path, b"GGUF" + bytes([1]) * 50)
    e = m.gguf_file_evidence(p, None, None)
    assert e["status"] == "unreadable" and e["file"] == "m.gguf" and e["size"] == p.stat().st_size and e["reason"]
    assert set(e) == {"file", "size", "status", "reason"}


def test_hyperparameter_keys_are_capped(tmp_path):
    m = module()
    kvs = hybrid_kvs() + [kv(f"qwen35.extra{i}", T_U32, i) for i in range(m.GGUF_MAX_HYPERPARAMETERS + 50)]
    _, e = entry_of(tmp_path, kvs=kvs)
    assert len(e["hyperparameters"]) == m.GGUF_MAX_HYPERPARAMETERS and e["hyperparameters_cut"] is True


def test_a_long_string_value_in_the_hyperparameters_is_cut(tmp_path):
    m = module()
    _, e = entry_of(tmp_path, kvs=hybrid_kvs(**{"qwen35.note": (T_STR, "x" * 5000)}))
    assert len(e["hyperparameters"]["qwen35.note"]) <= m.GGUF_MAX_VALUE_CHARS


# ---- the agreement table --------------------------------------------------------------------------

def hybrid_config(**over):
    """The config.json that goes with hybrid_kvs: the newer layout with a nested text_config and rope_parameters."""
    text = {
        "num_hidden_layers": 8, "hidden_size": 64, "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2,
        "head_dim": 16, "rms_norm_eps": 1e-06, "tie_word_embeddings": True, "mtp_num_hidden_layers": 0, "vocab_size": 300,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 3 + ["full_attention"],
        "full_attention_interval": 4, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4,
        "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 0]},
    }
    text.update(over)
    return {"architectures": ["Qwen3_5ForConditionalGeneration"], "text_config": text, "tie_word_embeddings": True}


def rows_of(tmp_path, config, template=TEMPLATE_TEXT, kvs=None, tensors=None):
    _, e = entry_of(tmp_path, kvs=kvs, tensors=tensors, config=config, template=template)
    return {r["item"]: r for r in e["agreement"]}, e


ALL_ITEMS = {"block count", "hidden size", "feed-forward size", "attention heads", "key-value heads", "head dimension", "rope base",
             "rope dimension", "rope sections", "rms norm epsilon", "layer pattern", "full attention interval", "linear conv kernel",
             "linear state size (key head dimension)", "linear key heads", "linear value heads", "linear inner size", "tied embeddings",
             "multi-token prediction layers", "vocabulary size", "chat template"}


def test_a_gguf_that_matches_its_config_agrees_on_every_row(tmp_path):
    rows, e = rows_of(tmp_path, hybrid_config())
    assert set(rows) == ALL_ITEMS, set(rows) ^ ALL_ITEMS
    assert {r["verdict"] for r in rows.values()} == {"agree"}, {k: v for k, v in rows.items() if v["verdict"] != "agree"}
    assert e["agreement_summary"] == {"agree": len(ALL_ITEMS), "differ": 0, "not comparable": 0}


@pytest.mark.parametrize("item,over", [
    ("block count", {"num_hidden_layers": 9}), ("hidden size", {"hidden_size": 65}), ("feed-forward size", {"intermediate_size": 1}),
    ("attention heads", {"num_attention_heads": 8}), ("key-value heads", {"num_key_value_heads": 1}), ("head dimension", {"head_dim": 32}),
    ("rope base", {"rope_parameters": {"rope_theta": 10000, "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 0]}}),
    ("rope dimension", {"rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.5, "mrope_section": [1, 1, 0]}}),
    ("rope sections", {"rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 0.25, "mrope_section": [1, 2, 0]}}),
    ("rms norm epsilon", {"rms_norm_eps": 1e-5}),
    ("layer pattern", {"layer_types": ["full_attention"] * 8}), ("full attention interval", {"full_attention_interval": 2}),
    ("linear conv kernel", {"linear_conv_kernel_dim": 3}), ("linear state size (key head dimension)", {"linear_key_head_dim": 32}),
    ("linear key heads", {"linear_num_key_heads": 4}), ("linear value heads", {"linear_num_value_heads": 8}),
    ("linear inner size", {"linear_value_head_dim": 32}), ("tied embeddings", {"tie_word_embeddings": False}),
    ("multi-token prediction layers", {"mtp_num_hidden_layers": 1}), ("vocabulary size", {"vocab_size": 301}),
])
def test_each_row_says_differ_when_the_config_differs_and_shows_both_values(tmp_path, item, over):
    rows, e = rows_of(tmp_path, hybrid_config(**over))
    assert rows[item]["verdict"] == "differ", rows[item]
    assert rows[item]["gguf"] is not None and rows[item]["config"] is not None
    assert e["agreement_summary"]["differ"] >= 1


def test_a_layer_pattern_of_another_length_differs(tmp_path):
    rows, _ = rows_of(tmp_path, hybrid_config(layer_types=["linear_attention"] * 3))
    assert rows["layer pattern"]["verdict"] == "differ"


def test_the_float32_epsilon_of_a_gguf_agrees_with_the_config_value(tmp_path):
    kvs = hybrid_kvs(**{"qwen35.attention.layer_norm_rms_epsilon": (T_F32, 1e-06)})
    rows, _ = rows_of(tmp_path, hybrid_config(rms_norm_eps=1e-06), kvs=kvs)
    assert rows["rms norm epsilon"]["verdict"] == "agree" and abs(rows["rms norm epsilon"]["gguf"] - 1e-06) < 1e-12


@pytest.mark.parametrize("a,b,same", [
    (1_000_000, 1_000_000, True),
    (1_000_000, 1_000_001, False),                      # review of PR 441, round 1: the relative tolerance made this agree
    (3_000_000_000, 3_000_000_001, False),
    (248_320, 248_319, False),
    (10 ** 400, 10 ** 400, True),                       # an integer too large for a float must not raise
    (10 ** 400, 10 ** 400 + 1, False),
    (10 ** 400, 1e300, False),                          # int against float: no OverflowError, and they differ
    (1e300, 10 ** 400, False),
    (10 ** 7, 1e7, True),                               # int against float keeps the tolerance (a rope base written either way)
    (1e-6, 1.0000004e-6, True),                         # floats keep the tolerance (a float32 epsilon)
    (1.0, 1.1, False),
])
def test_integers_compare_exactly_and_a_huge_integer_cannot_abort_stage_0(a, b, same):
    m = module()
    assert m._same(a, b) is same and m._same(b, a) is same
    assert m._same([a, 1], [b, 1]) is same


@pytest.mark.parametrize("key", ["rms_norm_eps", "rope_theta"])
def test_a_float_setting_written_as_a_huge_integer_is_not_comparable_and_cannot_abort_stage_0(tmp_path, key):
    """Same class as review round 1's finding on integer comparison: `math.isfinite` raises OverflowError on an integer too large for a float."""
    cfg = hybrid_config(rms_norm_eps=10 ** 400) if key == "rms_norm_eps" else hybrid_config(
        rope_parameters={"rope_theta": 10 ** 400, "partial_rotary_factor": 0.25, "mrope_section": [1, 1, 0]})
    rows, _ = rows_of(tmp_path, cfg)
    assert all(r["verdict"] in ("agree", "differ", "not comparable") for r in rows.values())
    assert any(r["config"] is not None and r["verdict"] == "not comparable" for r in rows.values())


def test_number_and_whole_refuse_what_is_not_a_finite_number_without_raising():
    m = module()
    assert m._number(10 ** 400) is None and m._number(-10 ** 400) is None
    assert m._number(float("inf")) is None and m._number(True) is None and m._number("1") is None
    assert m._number(3) == 3 and m._number(2.5) == 2.5


def test_a_row_of_whole_counts_that_differ_by_one_differs_even_when_the_count_is_large(tmp_path):
    kvs = hybrid_kvs(**{"qwen35.embedding_length": (T_U32, 3_000_000_000)})
    rows, _ = rows_of(tmp_path, hybrid_config(hidden_size=3_000_000_001), kvs=kvs)
    assert rows["hidden size"]["verdict"] == "differ"
    rows, _ = rows_of(tmp_path, hybrid_config(hidden_size=3_000_000_000), kvs=kvs)
    assert rows["hidden size"]["verdict"] == "agree"


def untied_tensors():
    return hybrid_tensors() + [("output.weight", [64, 300], Q6_K)]


@pytest.mark.parametrize("split", [
    {"split.no": (T_U16, 0), "split.count": (T_U16, 2)},            # a shard of a two-file model
    {"split.tensors.count": (T_I32, 9999)},                         # the model holds more tensors than this file does
    {"split.count": (T_STR, "two")},                                # a split count that is not a number: the inventory is unknown
], ids=["split.count 2", "tensors.count above this file", "split.count not a number"])
@pytest.mark.parametrize("tie", [True, False])
def test_a_missing_output_tensor_in_one_shard_of_a_split_file_does_not_show_tying(tmp_path, split, tie):
    """Review of PR 441, round 1: a split GGUF's tensor table lists one shard. An untied model whose output.weight is in another shard was
    reported as tied, a false difference from config.json."""
    kvs = hybrid_kvs(**split)
    rows, e = rows_of(tmp_path, hybrid_config(tie_word_embeddings=tie), kvs=kvs)
    assert e["tied_embeddings"] is None and e["tensor_inventory_complete"] is False
    assert rows["tied embeddings"]["verdict"] == "not comparable"
    assert "split" in rows["tied embeddings"]["note"].lower()


def test_an_output_tensor_that_is_present_shows_untied_even_in_a_shard(tmp_path):
    kvs = hybrid_kvs(**{"split.no": (T_U16, 1), "split.count": (T_U16, 2)})
    rows, e = rows_of(tmp_path, hybrid_config(tie_word_embeddings=False), kvs=kvs, tensors=untied_tensors())
    assert e["tied_embeddings"] is False and rows["tied embeddings"]["verdict"] == "agree"
    rows, _ = rows_of(tmp_path, hybrid_config(tie_word_embeddings=True), kvs=kvs, tensors=untied_tensors())
    assert rows["tied embeddings"]["verdict"] == "differ"


def test_a_file_that_is_not_split_still_reads_its_missing_output_tensor_as_tied(tmp_path):
    rows, e = rows_of(tmp_path, hybrid_config(tie_word_embeddings=True))
    assert e["tied_embeddings"] is True and e["tensor_inventory_complete"] is True and rows["tied embeddings"]["verdict"] == "agree"
    kvs = hybrid_kvs(**{"split.no": (T_U16, 0), "split.count": (T_U16, 1), "split.tensors.count": (T_I32, len(hybrid_tensors()))})
    rows, e = rows_of(tmp_path, hybrid_config(tie_word_embeddings=False), kvs=kvs)
    assert e["tied_embeddings"] is True and e["tensor_inventory_complete"] is True and rows["tied embeddings"]["verdict"] == "differ"


def test_a_row_with_a_missing_side_is_not_comparable_and_never_agree(tmp_path):
    cfg = hybrid_config()
    del cfg["text_config"]["intermediate_size"]
    rows, _ = rows_of(tmp_path, cfg)
    assert rows["feed-forward size"]["verdict"] == "not comparable" and rows["feed-forward size"]["config"] is None
    kvs = hybrid_kvs(**{"qwen35.feed_forward_length": (None, None)})
    rows, _ = rows_of(tmp_path, hybrid_config(), kvs=kvs)
    assert rows["feed-forward size"]["verdict"] == "not comparable" and rows["feed-forward size"]["gguf"] is None


@pytest.mark.parametrize("bad", [None, "x", 3.5, True, [], {"a": 1}, 10**30])
def test_a_config_value_of_the_wrong_type_is_not_comparable_not_a_crash(tmp_path, bad):
    rows, _ = rows_of(tmp_path, hybrid_config(num_hidden_layers=bad))
    assert rows["block count"]["verdict"] in ("not comparable", "differ")
    assert rows["block count"]["verdict"] != "agree" or bad == 8


def test_a_config_that_is_not_an_object_makes_every_row_not_comparable(tmp_path):
    for bad in (None, [], "x", 3):
        _, e = entry_of(tmp_path, config=bad, name=f"{type(bad).__name__}.gguf")
        assert e["status"] == "ok" and {r["verdict"] for r in e["agreement"]} == {"not comparable"}


def test_a_flat_config_is_read_too(tmp_path):
    flat = hybrid_config()["text_config"]
    rows, _ = rows_of(tmp_path, flat)
    assert rows["block count"]["verdict"] == "agree" and rows["rope base"]["verdict"] == "agree"


def test_the_mtp_row_shows_a_difference_when_the_config_declares_a_head_the_file_lacks(tmp_path):
    rows, _ = rows_of(tmp_path, hybrid_config(mtp_num_hidden_layers=1))
    r = rows["multi-token prediction layers"]
    assert r["verdict"] == "differ" and r["gguf"] == 0 and r["config"] == 1 and "converter" in r["note"]
    assert "compare with the mtp tensor count" in r["note"]                      # the safetensors' count was not given


def test_the_mtp_row_states_the_safetensors_own_mtp_tensor_count_when_it_is_known(tmp_path):
    """On the real Qwen3.5-4B-3D-Coder repository the config declares one head and neither the GGUF nor the safetensors has it."""
    for n in (0, 15):
        _, e = entry_of(tmp_path, config=hybrid_config(mtp_num_hidden_layers=1), name=f"n{n}.gguf", mtp_tensors=n)
        [r] = [x for x in e["agreement"] if x["item"] == "multi-token prediction layers"]
        assert r["verdict"] == "differ" and f"the safetensors carry {n} mtp tensors" in r["note"]
    _, e = entry_of(tmp_path, config=hybrid_config(mtp_num_hidden_layers=1), name="bool.gguf", mtp_tensors=True)
    [r] = [x for x in e["agreement"] if x["item"] == "multi-token prediction layers"]
    assert "safetensors carry" not in r["note"]                                   # a bool is not a count


def test_the_chat_template_row_compares_hashes_and_ignores_trailing_whitespace(tmp_path):
    rows, _ = rows_of(tmp_path, hybrid_config(), template=TEMPLATE_TEXT + "\n")
    r = rows["chat template"]
    assert r["verdict"] == "agree" and r["gguf"] == hashlib.sha256(TEMPLATE_TEXT.encode()).hexdigest()
    rows, _ = rows_of(tmp_path, hybrid_config(), template=TEMPLATE_TEXT + "{# edited #}")
    assert rows["chat template"]["verdict"] == "differ"
    rows, _ = rows_of(tmp_path, hybrid_config(), template=None)
    assert rows["chat template"]["verdict"] == "not comparable" and "no chat_template" in rows["chat template"]["note"]


def test_a_projector_is_not_compared_with_the_language_model_config(tmp_path):
    kvs = [kv("general.architecture", T_STR, "clip"), kv("clip.vision.block_count", T_U32, 24)]
    cfg = hybrid_config()
    cfg["vision_config"] = {"depth": 24}
    _, e = entry_of(tmp_path, kvs=kvs, tensors=[("v.a", [4], F32)], config=cfg)
    assert [r["item"] for r in e["agreement"]] == ["vision block count"] and e["agreement"][0]["verdict"] == "agree"
    cfg["vision_config"] = {"depth": 24, "hidden_size": 1024}               # a value on one side only is listed, as not comparable
    _, e = entry_of(tmp_path, kvs=kvs, tensors=[("v.a", [4], F32)], config=cfg, name="b.gguf")
    assert {r["item"]: r["verdict"] for r in e["agreement"]} == {"vision block count": "agree", "vision hidden size": "not comparable"}


def test_a_projector_that_disagrees_with_the_vision_config_differs(tmp_path):
    kvs = [kv("general.architecture", T_STR, "clip"), kv("clip.vision.block_count", T_U32, 24)]
    cfg = hybrid_config()
    cfg["vision_config"] = {"depth": 12}
    _, e = entry_of(tmp_path, kvs=kvs, tensors=[("v.a", [4], F32)], config=cfg)
    assert e["agreement"][0]["verdict"] == "differ"


# ---- the whole evidence file ----------------------------------------------------------------------

def test_the_evidence_file_lists_the_ignored_gguf_files_and_only_those(tmp_path):
    m = module()
    write(tmp_path, GOOD, "a.gguf")
    write(tmp_path, b"junk", "b.gguf")
    write(tmp_path, GOOD, "c.bin")
    ignored = [{"file": "a.gguf", "size": 1, "reason": "x"}, {"file": "b.gguf", "size": 4, "reason": "x"}]
    out = m.gguf_evidence(tmp_path, ignored, hybrid_config(), TEMPLATE_TEXT)
    assert out["schema"] == 1 and [f["file"] for f in out["files"]] == ["a.gguf", "b.gguf"]
    assert [f["status"] for f in out["files"]] == ["ok", "unreadable"] and "hint" in out["note"]


def test_no_ignored_gguf_files_gives_no_evidence(tmp_path):
    m = module()
    assert m.gguf_evidence(tmp_path, [], hybrid_config(), None) is None
    assert m.gguf_evidence(tmp_path, [{"file": "x.bin", "size": 1, "reason": "x"}], hybrid_config(), None) is None


@pytest.mark.parametrize("name", ["../a.gguf", "sub/a.gguf", "/etc/a.gguf", "", None, 5, "a\0.gguf"])
def test_a_listed_name_that_is_not_a_plain_file_name_is_refused_not_opened(tmp_path, name):
    m = module()
    sub = tmp_path / "sub"
    sub.mkdir()
    write(tmp_path, GOOD, "a.gguf")
    out = m.gguf_evidence(sub, [{"file": name, "size": 1, "reason": "x"}], None, None)
    assert out is not None and out["files"][0]["status"] == "unreadable" and "plain file name" in out["files"][0]["reason"]


def test_the_number_of_files_read_is_capped(tmp_path):
    m = module()
    names = [f"f{i}.gguf" for i in range(m.GGUF_MAX_FILES + 3)]
    for n in names:
        write(tmp_path, GOOD, n)
    out = m.gguf_evidence(tmp_path, [{"file": n, "size": 1, "reason": "x"} for n in names], None, None)
    assert len(out["files"]) == m.GGUF_MAX_FILES and out["files_cut"] == 3


# ---- the repository's own chat template -----------------------------------------------------------

def test_read_small_text_reads_a_file_and_a_link_and_refuses_the_rest(tmp_path):
    m = module()
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "l.txt").symlink_to("a.txt")
    (tmp_path / "big.txt").write_text("x" * 100)
    (tmp_path / "bad.txt").write_bytes(b"\xff\xfe")
    (tmp_path / "zero.txt").symlink_to("/dev/zero")
    os.mkfifo(tmp_path / "fifo.txt")
    (tmp_path / "dir.txt").mkdir()
    assert m.read_small_text(tmp_path / "a.txt") == "hello" and m.read_small_text(tmp_path / "l.txt") == "hello"
    assert m.read_small_text(tmp_path / "big.txt", cap=50) is None and m.read_small_text(tmp_path / "big.txt", cap=100) == "x" * 100
    for name in ("bad.txt", "zero.txt", "fifo.txt", "dir.txt", "none.txt"):
        assert m.read_small_text(tmp_path / name) is None, name


def test_the_repository_template_is_the_jinja_file_else_the_tokenizer_config_string(tmp_path):
    m = module()
    assert m.repo_chat_template(tmp_path) is None
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": "from-config"}))
    assert m.repo_chat_template(tmp_path) == "from-config"
    (tmp_path / "chat_template.jinja").write_text("from-file")
    assert m.repo_chat_template(tmp_path) == "from-file"
    (tmp_path / "chat_template.jinja").write_text("")                     # an empty file does not win
    assert m.repo_chat_template(tmp_path) == "from-config"
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": [1, 2]}))
    assert m.repo_chat_template(tmp_path) is None


# ---- hostile headers in a memory-limited child ----------------------------------------------------
# The reader must stay small whatever a header declares. These run it in a child with RLIMIT_AS of 2 GiB and report the child's peak size
# (a /dev/zero test once ran a machine out of memory: tests/hostile_inputs.py, and the note about hostile-input tests in the project log).

CHILD = r"""
import importlib.util, json, resource, sys
resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
spec = importlib.util.spec_from_file_location("t", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
out = {}
for p in sys.argv[2:]:
    try:
        h = m.read_gguf_header(p)
        out[p] = "read:%d" % len(h["tensors"])
    except m.GgufError as e:
        out[p] = "refused: " + str(e)
print(json.dumps({"results": out, "peak_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
"""


def child_run(paths):
    import subprocess
    import sys
    r = subprocess.run([sys.executable, "-c", CHILD, str(TEMPLATE), *map(str, paths)], capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-2000:]
    return json.loads(r.stdout)


def big_strings_header(path: Path, count: int, each: int) -> None:
    """A sparse file that declares `count` string values of `each` bytes: the header is far larger than the file's real content."""
    with open(path, "wb") as f:
        f.write(b"GGUF" + struct.pack("<IQQ", 3, 0, count))
        for i in range(count):
            f.write(s(f"k{i}") + struct.pack("<I", T_STR) + struct.pack("<Q", each))
            f.seek(each, os.SEEK_CUR)
        f.truncate(f.tell() + 1)


def test_hostile_headers_are_refused_by_a_reader_limited_to_2_gib_and_it_stays_small(tmp_path):
    m = module()
    files = {}
    files["a string of 2**40 bytes"] = write(tmp_path, MALFORMED["a string length beyond the file"][0], "s1.gguf")
    files["an array of 4 million u32 in a tiny file"] = write(tmp_path, b"GGUF" + struct.pack("<IQQ", 3, 0, 1) + s("a") + struct.pack("<I", T_ARR)
                                                              + struct.pack("<IQ", T_U32, 4_000_000), "a1.gguf")
    files["an array count of 10**12"] = write(tmp_path, MALFORMED["an array count above the cap"][0], "a2.gguf")
    files["a kv count of 10**9"] = write(tmp_path, MALFORMED["kv count above the cap"][0], "k1.gguf")
    big = tmp_path / "big.gguf"
    big_strings_header(big, 200, m.GGUF_MAX_STRING)            # 200 MiB of declared header
    files["a header of 200 MiB"] = big
    huge = write(tmp_path, gguf(hybrid_kvs(), hybrid_tensors()), "huge.gguf")
    os.truncate(huge, 8 << 30)
    files["a valid header in an 8 GiB sparse file"] = huge
    (tmp_path / "zero.gguf").symlink_to("/dev/zero")
    os.mkfifo(tmp_path / "fifo.gguf")
    files["a link to /dev/zero"], files["a FIFO"] = tmp_path / "zero.gguf", tmp_path / "fifo.gguf"
    out = child_run(files.values())
    by_label = {label: out["results"][str(p)] for label, p in files.items()}
    assert by_label["a valid header in an 8 GiB sparse file"].startswith("read:")
    for label, v in by_label.items():
        if label != "a valid header in an 8 GiB sparse file":
            assert v.startswith("refused: "), (label, v)
    assert "larger than" in by_label["a header of 200 MiB"]
    assert "regular file" in by_label["a link to /dev/zero"] and "regular file" in by_label["a FIFO"]
    assert out["peak_kb"] < 600_000, out["peak_kb"]                                   # well under the 2 GiB limit


def test_the_element_budget_stops_a_header_of_many_arrays(tmp_path, monkeypatch):
    m = module()
    monkeypatch.setattr(m, "GGUF_MAX_ELEMENTS", 1000)
    kvs = [kv(f"a{i}", T_ARR, (T_STR, ["x"] * 400)) for i in range(3)]
    with pytest.raises(m.GgufError, match="too many array elements"):
        m.read_gguf_header(write(tmp_path, gguf(kvs, [])))


def test_a_non_finite_float_is_stored_as_none_so_the_evidence_is_valid_json(tmp_path):
    kvs = hybrid_kvs(**{"qwen35.rope.freq_base": (T_F32, float("nan")), "qwen35.extra": (T_ARR, (T_F32, [1.0, float("inf")]))})
    _, e = entry_of(tmp_path, kvs=kvs)
    assert e["hyperparameters"]["qwen35.rope.freq_base"] is None and e["hyperparameters"]["qwen35.extra"] == [1.0, None]
    json.dumps(e, allow_nan=False)
