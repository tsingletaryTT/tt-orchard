# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""Residency estimate from safetensors headers (issue 327): how many chips a checkpoint needs, before any boot.

A port for a model that does not fit the box fails after the coder has spent a prepare step and a lease. The checkpoint already says
how big it is: each `.safetensors` file starts with an 8-byte little-endian length and a JSON header that lists every tensor's dtype,
shape and byte range. This module reads those headers (never the tensor data), sorts the tensors into categories by name, applies a
quantization policy per category and divides what lands on the chips by the chip count.

Two kinds of figure, kept apart in the record:

  * `measured`: read from the safetensors headers. Byte counts, element counts and tensor counts per category.
  * `estimated`: arithmetic on the measured figures and on the assumptions. Stored bytes after quantization, bytes per chip, whether a
    row fits, chips needed, the first-boot disk and host RAM. Every assumption is listed in `assumptions` with its value and its
    source: `caller` when the caller stated it, `default (choice)` when it comes from orchard/defaults.py. No default is an
    allocator reading. `not_measured` says what a real reading would replace.

Categories (`CATEGORIES`, decided by `categorize`): embedding (input table and LM head), attention (full attention and its indexer), gdn
(gated delta net and other linear or state-space mixers), dense_mlp, routed_experts, shared_experts, ngram_ple (the n-gram or
per-layer embedding table), vision, mtp (everything under the multi-token-prediction layer, its own experts included), norms_other
(norms, one-dimensional tensors, routers and the small gated-residual modules).

Quantization policy: one of bf16, bfp8, bfp4 per category, or `host` for a category that stays in host memory in its checkpoint dtype
(the n-gram table in the default policy). A block format applies to floating-point tensors with two or more dimensions; a norm vector or
an integer index tensor keeps the bytes it has in the checkpoint. Widths come from orchard/decode_floor.py (`DTYPE_BYTES`).

Where it is used: `tt-orchard residency TARGET` (a snapshot directory, or a repo id whose headers come from
`huggingface_hub.get_safetensors_metadata`), and stage 0 of a run (`write_residency`), which writes `stages/0/residency.json`. The
supervisor reads that record before a full-port stage 2 (`read_record`, `shortfall`) and blocks when the box has fewer chips than the
estimate needs.

Data parallel (issue 359). When the weights per chip fit one chip with the stated reserve and KV budget (the one-chip row of the table),
the record's `estimated.data_parallel` block says that the configured chips can each hold a full copy: the number of copies (one per
chip), the chips that gives a copy each, and that this is replication with no collective operations. It estimates no throughput. The one
measured example (`MEASURED_EXAMPLE`) is the author's figure for one reranker on one kind of die; measured scaling is needed before a
card or a plan says anything about speed. Stage 0 also notes a model that fits one chip but is configured for tensor parallel across
several (`sharded_chips`): a hazard note, never a block.

Precision options and the refusal. The record lists the whole model at each of bf16, bfp8 and bfp4 (`estimated.precision_options`), and
`shortfall` ends its sentence with the highest listed precision that fits or says that none does. A model with routed experts is marked
(`estimated.mixture_of_experts`): every expert stays resident. `machine_shortfall` compares the first boot's disk and host RAM with two
readings (`machine_facts`), and the supervisor blocks on it before a full-port stage 2.

Totals only. `totals_bound` takes the Hub's parameter counts by dtype (one model-info call, `totals_from_hub`) and gives a lower bound by
precision with no tensor name or shape. It exists for a repository whose headers are too many to read (Hydrogen-1.0 has 303 shards), and
for the preflight's `fit` row, which refuses a model no listed precision can fit before the run downloads anything.

Limits. Even spread of the weights over the chips, no replication of any tensor in the table's rows, no padding to tiles, one reserve per
chip. The data-parallel block counts the host RAM and disk of one copy. A shard whose data is shorter than its header says is not
noticed, because the data is never read.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import shutil
import stat
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

from fractions import Fraction

from orchard import decode_floor, defaults, safeio, sizing

FILE = "residency.json"
INDEX_FILE = "model.safetensors.index.json"
MAX_HEADER_BYTES = 64 << 20          # the largest real headers are a few megabytes (a shard of 50,000 tensors); this bound keeps a hostile length from allocating
MAX_TOTAL_HEADER_BYTES = 512 << 20   # header bytes held for a whole snapshot: a choice, far above any header set seen (a few megabytes per shard)
MAX_FILES = 10_000                   # shard files in one snapshot
MAX_TENSORS = 1_000_000              # tensors in one checkpoint
MAX_NDIM = 8
MAX_RECORD_BYTES = 4 << 20
GIB = 1 << 30
MESH_SIZES = (1, 2, 4, 8)            # the chip counts the table always shows

CATEGORIES = ("embedding", "attention", "gdn", "dense_mlp", "routed_experts", "shared_experts", "ngram_ple", "vision", "mtp", "norms_other")
POLICY_DTYPES = ("bf16", "bfp8", "bfp4", "host")
# Choice (nobody measured these): the Flash-Next packages store routed experts as BF4 and dense weights as BF8, and keep the vision tower in
# BF16 and the n-gram table on the host (docs/run-logs/2026-11-01-flash-next-graft-study.md, sections 3 and 5). The embedding, MTP and
# shared-expert rows follow "dense weights go to BF8" in the same section.
DEFAULT_POLICY = {"embedding": "bfp8", "attention": "bfp8", "gdn": "bfp8", "dense_mlp": "bfp8", "routed_experts": "bfp4",
                  "shared_experts": "bfp8", "ngram_ple": "host", "vision": "bf16", "mtp": "bfp8", "norms_other": "bf16"}

# Bytes per value in the safetensors dtype names, and which of them are floating point (a block format can replace those).
DTYPE_WIDTH = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1,
               "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U64": 8, "U32": 4, "U16": 2, "U8": 1, "BOOL": 1}
FLOAT_DTYPES = frozenset({"F64", "F32", "F16", "BF16", "F8_E4M3", "F8_E5M2", "F8_E8M0"})

LABELS = {"measured": "read from the safetensors headers (no tensor data was read)",
          "estimated": "arithmetic on the measured figures and the listed assumptions; not an allocator reading"}
NOT_MEASURED = (
    "usable DRAM per chip: an allocator reading after the runtime's own buffers (the DRAM figure here is an assumption)",
    "the per-chip reserve for trace regions, program binaries and circular buffers, and the KV and recurrent-state budget (assumptions)",
    "the real size of the converted cache and the first boot's peak disk and host RAM (read them from a first boot)",
    "padding to 32 by 32 tiles, replication of tensors across chips, and any tensor that is split unevenly",
    "tensor data: only headers were read, so a shard that is shorter than its header says is not noticed")

# The one measured example of data-parallel scaling we have. It is the author's own measurement on p300c dies. Orchard did not measure it, and it covers one model.
MEASURED_EXAMPLE = {
    "model": "raahemnabeel/qwen3-reranker-4b", "unit": "pairs/s", "clients": 4, "by_board": {"p150": 21.7, "p300": 42.0, "p300x2": 83.0},
    "basis": "the model card author's measurement on p300c dies with four clients; orchard has not measured it"}
SHARDED_BASES = ("stated", "required chip counts")
DP_NOTE = "replication: one full copy of the weights on each chip, with no collective operations between the chips"
DP_THROUGHPUT = ("not estimated: measured scaling is needed. The one measured example is " + MEASURED_EXAMPLE["model"] + ": 21.7, 42.0 and 83.0 pairs/s "
                 "on a p150, a p300 and a p300x2 with four clients (" + MEASURED_EXAMPLE["basis"] + ")")


# The whole-model precisions the record lists next to the policy it uses, highest first. Each applies one block format (or bf16) to every
# floating-point weight of two or more dimensions; a norm vector or an integer tensor keeps the bytes it has in the checkpoint.
PRECISIONS = ("bf16", "bfp8", "bfp4")
MAX_TOTAL_PARAMS = 10 ** 15          # a count above this is not a model: a hostile answer must not turn into a huge integer sum
MOE_NOTE = ("a mixture-of-experts model keeps every expert's weights resident on the chips although only a few experts run for each token, "
            "so the chip count follows all the experts. The number of experts used per token is not in the safetensors "
            "headers and was not read")
MOE_NOTE_HOST = ("a mixture-of-experts model whose policy keeps the routed experts in host memory puts them on no chip: they take no chip memory "
                 "and the host RAM figure includes them. The number of experts used per token is not in the safetensors headers and was "
                 "not read")
BOUND_POLICY_TEXT = ("if every weight were stored at bfp4, a lower bound: tensor names and shapes were not read, so norms and other "
                     "one-dimensional tensors are counted at the block size too")
BOUND_NOT_MEASURED = (
    "tensor names and shapes: the Hub's counts are totals by dtype, so no category, expert count or one-dimensional tensor (a norm or a bias, "
    "which no block format shrinks) is known; every figure is therefore a lower bound on the bytes a real checkpoint holds, and on the "
    "device bytes only when no n-gram or per-layer table is among them: the default policy keeps such a table in host memory, which would "
    "lower the device figure")
BOUND_HOST_NOTE = ("the counts include every parameter, so a table that the default policy keeps on the host (an n-gram or per-layer "
                   "embedding) is counted as if it sat on the chips; the device figure of a model with such a table is lower")
MAX_CONFIG_KEYS, MAX_CONFIG_DEPTH = 5000, 6        # a config.json is a few dozen keys; these bounds keep a hostile one from costing time


def _table_key(key: str) -> bool:
    """Whether a config key names the n-gram or per-layer table that `categorize` puts in `ngram_ple`: a word that starts with `ngram`, the word
    `ple`, or `per_layer`. A word is a part between underscores, so `multiple_of` and `sample_rate` do not match."""
    low = key.lower()
    parts = low.split("_")
    return "per_layer" in low or "ple" in parts or any(p.startswith("ngram") for p in parts)


def config_has_host_table(config) -> bool | None:
    """Whether a model's config.json names a table that the default policy keeps on the host (an n-gram or per-layer embedding), looking at
    every key down to MAX_CONFIG_DEPTH levels, through lists too. None when the config is not an object, or is too large or too deep to read to
    the end without finding one: no claim either way, and the row that asks cannot block on it."""
    if not isinstance(config, dict):
        return None
    seen = 0
    stack = [(config, 0)]
    while stack:
        node, depth = stack.pop()
        items = node.items() if isinstance(node, dict) else enumerate(node) if isinstance(node, list) else ()
        for key, value in items:
            seen += 1
            if seen > MAX_CONFIG_KEYS:
                return None
            if isinstance(key, str) and _table_key(key):
                return True
            if isinstance(value, (dict, list)):
                if depth + 1 > MAX_CONFIG_DEPTH:
                    return None
                stack.append((value, depth + 1))
    return False


def index_has_host_table(names) -> bool | None:
    """Whether the tensor names of a weight index (`model.safetensors.index.json`'s `weight_map` keys) include a table that the default policy
    keeps on the host: the rule of `categorize`, applied to each name. None when there are no names to read. A name that is not text is
    skipped."""
    if not isinstance(names, (list, tuple)) or not names:
        return None
    return any(isinstance(n, str) and categorize(n, (2, 2)) == "ngram_ple" for n in names)


def host_table_from(config, names) -> bool | None:
    """Whether the Hub's parameter counts may include a table that the default policy keeps on the host, from the model's config.json and the
    tensor names of its weight index. True when either names one. False only when placement is established: the config was read and names no
    table, and the index's tensor names were read and none is a table. None otherwise (a single-file model has no index, a gated repository
    gives neither file without a token, a file was too large or too deep): nothing is established, and the row that asks stays advisory."""
    by_config, by_names = config_has_host_table(config), index_has_host_table(names)
    if by_config is True or by_names is True:
        return True
    if by_config is False and by_names is False:
        return False
    return None


class ResidencyError(ValueError):
    """An input is unusable: a header that is malformed or too large, a bad number, a bad policy, a target that is not a snapshot or a repo id.
    The message names the file or the option and never holds file content or a token."""


@dataclass(frozen=True)
class Tensor:
    name: str
    dtype: str
    shape: tuple
    bytes: int
    elements: int
    file: str


@dataclass(frozen=True)
class HeaderSet:
    tensors: tuple
    files: int
    source: str
    local: bool

    @property
    def checkpoint_bytes(self) -> int:
        return sum(t.bytes for t in self.tensors)


# ---- reading headers --------------------------------------------------------------------------------------------------------

def _tensor(name, entry, where: str) -> Tensor:
    """One validated header entry. `where` names the file in every message."""
    if not isinstance(name, str) or not isinstance(entry, dict):
        raise ResidencyError(f"{where}: a tensor entry is not an object")
    dtype, shape, offsets = entry.get("dtype"), entry.get("shape"), entry.get("data_offsets")
    if not isinstance(dtype, str) or dtype not in DTYPE_WIDTH:
        raise ResidencyError(f"{where}: a tensor has a dtype the estimator has no width for")
    if (not isinstance(shape, (list, tuple)) or len(shape) > MAX_NDIM
            or not all(isinstance(d, int) and not isinstance(d, bool) and d >= 0 for d in shape)):
        raise ResidencyError(f"{where}: a tensor has a shape that is not a short list of whole numbers of at least 0")
    if (not isinstance(offsets, (list, tuple)) or len(offsets) != 2
            or not all(isinstance(o, int) and not isinstance(o, bool) and o >= 0 for o in offsets) or offsets[1] < offsets[0]):
        raise ResidencyError(f"{where}: a tensor has data offsets that are not an ordered pair of whole numbers")
    if any(d > safeio.MAX_NUMEL for d in shape) or (0 not in shape and math.prod(shape) > safeio.MAX_NUMEL):
        raise ResidencyError(f"{where}: a tensor claims too many values")
    elements = math.prod(shape)
    size = elements * DTYPE_WIDTH[dtype]
    if offsets[1] - offsets[0] != size:
        raise ResidencyError(f"{where}: a tensor's data offsets disagree with its shape and dtype")
    return Tensor(name, dtype, tuple(shape), size, elements, where)


def _read_exact(fd: int, n: int, where: str) -> bytes:
    chunks, left = [], n
    while left:
        try:
            piece = os.read(fd, min(left, 1 << 20))
        except OSError:
            raise ResidencyError(f"{where}: the file cannot be read") from None
        if not piece:
            raise ResidencyError(f"{where}: the file ended inside its header")
        chunks.append(piece)
        left -= len(piece)
    return b"".join(chunks)


def _read_file_header(path: str, where: str, budget: list) -> dict:
    """The JSON header object of one shard. Reads 8 bytes, then the header, and nothing else. `budget` is a one-item list holding the
    header bytes still allowed across the whole snapshot; a header larger than what is left is refused before it is read."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except (OSError, ValueError):
        raise ResidencyError(f"{where}: the file cannot be opened") from None
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            raise ResidencyError(f"{where}: not a regular file")
        (length,) = struct.unpack("<Q", _read_exact(fd, 8, where))
        if length > MAX_HEADER_BYTES:
            raise ResidencyError(f"{where}: the header is longer than the {MAX_HEADER_BYTES}-byte cap")
        if 8 + length > st.st_size:
            raise ResidencyError(f"{where}: the header is longer than the file")
        budget[0] -= length
        if budget[0] < 0:
            raise ResidencyError(f"{where}: the headers of this snapshot together are longer than the {MAX_TOTAL_HEADER_BYTES}-byte budget")
        raw = _read_exact(fd, length, where)
    finally:
        os.close(fd)
    try:
        header = safeio.parse_json_bytes(raw, max_bytes=MAX_HEADER_BYTES, expect=dict)
    except safeio.SafeReadError as exc:
        raise ResidencyError(f"{where}: the header is not usable JSON ({exc.reason})") from None
    return header


def _check_coverage(weight_map, file_names, tensor_names: set) -> None:
    """Refuse a snapshot that its own index says is incomplete: a shard the index names that is not there, or a tensor it names that no
    header lists. A missing shard would otherwise understate every figure and could give a false fit. `weight_map` is the index's
    tensor-to-file map. Names from the index are not repeated in the message, only counted."""
    if not isinstance(weight_map, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in weight_map.items()):
        raise ResidencyError("the safetensors index does not hold a map from tensor names to file names")
    files = set(weight_map.values())
    if files - set(file_names):
        raise ResidencyError(f"the snapshot is incomplete: the index names {len(files - set(file_names))} shard file(s) that are not here")
    if set(weight_map) - tensor_names:
        raise ResidencyError(f"the snapshot is incomplete: the index names {len(set(weight_map) - tensor_names)} tensor(s) that no file header lists")


def _collect(per_file, source: str, local: bool, weight_map=None) -> HeaderSet:
    """Validate the tensors of each (file name, header dict) from `per_file`, one file at a time, so a limit (tensor count, header budget)
    stops the read before the later files are opened and no raw header is kept after its tensors are made. `per_file` may be a generator."""
    tensors, seen, names = [], set(), []
    for where, header in per_file:
        names.append(where)
        for name, entry in header.items():
            if name == "__metadata__":
                continue
            t = _tensor(name, entry, where)
            if name in seen:
                raise ResidencyError(f"{where}: the tensor name appears twice in the checkpoint")
            seen.add(name)
            tensors.append(t)
            if len(tensors) > MAX_TENSORS:
                raise ResidencyError(f"the checkpoint has more than {MAX_TENSORS} tensors")
        header = None                                  # the raw header is no longer needed
    if weight_map:
        _check_coverage(weight_map, names, seen)
    return HeaderSet(tuple(tensors), len(names), source, local)


def read_headers(directory) -> HeaderSet:
    """The tensors in every top-level `.safetensors` file of a snapshot directory, from the headers only.

    A Hugging Face cache snapshot holds each file as a link into blobs/, so each path is resolved before the guarded open. Raises
    ResidencyError for a missing directory, no shard, too many shards or tensors, and any shard whose header is not usable."""
    try:
        base = Path(directory)
        names = sorted(n for n in os.listdir(base) if n.endswith(".safetensors"))
    except (OSError, ValueError, TypeError):
        raise ResidencyError("the model directory cannot be listed") from None
    if not names:
        raise ResidencyError("the directory holds no .safetensors file")
    if len(names) > MAX_FILES:
        raise ResidencyError(f"the directory holds more than {MAX_FILES} .safetensors files")
    weight_map = None
    if os.path.lexists(base / INDEX_FILE):
        try:
            index = safeio.read_json(safeio.resolved(base / INDEX_FILE), max_bytes=MAX_HEADER_BYTES, expect=dict)
        except safeio.SafeReadError as exc:
            raise ResidencyError(f"{INDEX_FILE}: the index is not usable ({exc.reason})") from None
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict):
            raise ResidencyError(f"{INDEX_FILE}: the index has no weight_map object")
    budget = [MAX_TOTAL_HEADER_BYTES]
    per_file = ((n, _read_file_header(safeio.resolved(base / n), n, budget)) for n in names)
    return _collect(per_file, f"local:{base.name}", True, weight_map)


def headers_from_metadata(meta, *, source: str) -> HeaderSet:
    """A HeaderSet from a `huggingface_hub` SafetensorsRepoMetadata (or anything shaped like it)."""
    try:
        files = sorted(meta.files_metadata.items())
        weight_map = getattr(meta, "weight_map", None)
        if not files:
            raise ResidencyError("the repository lists no .safetensors file")
        if len(files) > MAX_FILES:
            raise ResidencyError(f"the repository lists more than {MAX_FILES} .safetensors files")
        per_file = ((fname, {n: {"dtype": i.dtype, "shape": list(i.shape), "data_offsets": list(i.data_offsets)} for n, i in fm.tensors.items()})
                    for fname, fm in files)
        return _collect(per_file, source, False, weight_map if isinstance(weight_map, dict) and weight_map else None)
    except (AttributeError, TypeError):
        raise ResidencyError("the Hugging Face metadata is not in the expected shape") from None


_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")


def _hub_fetch(repo_id, revision=None):
    import huggingface_hub
    return huggingface_hub.get_safetensors_metadata(repo_id, revision=revision)


def read_target(target, *, revision=None, fetch=None) -> HeaderSet:
    """Headers for a snapshot directory or a repo id. A repo id goes through `fetch(repo_id, revision)`, which defaults to
    `huggingface_hub.get_safetensors_metadata` (headers only: it uses HTTP range requests and never downloads the weights)."""
    if isinstance(target, (str, os.PathLike)) and os.path.isdir(target):
        return read_headers(target)
    if not isinstance(target, str) or not _REPO_ID.match(target) or ".." in target.split("/"):
        raise ResidencyError("the target is neither a model directory nor a repo id like owner/name")
    if revision is not None and not (isinstance(revision, str) and re.match(r"^[A-Za-z0-9._/-]{1,200}$", revision)):
        raise ResidencyError("the revision is not a branch name, tag or commit hash")
    try:
        meta = (fetch or _hub_fetch)(target, revision)
    except ResidencyError:
        raise
    except ImportError:
        raise ResidencyError("huggingface_hub is not installed, so a repo id cannot be read") from None
    except Exception as exc:                  # a network, auth or format failure: the type is enough, the text may hold a token
        raise ResidencyError(f"the Hugging Face metadata call for {target} failed ({type(exc).__name__})") from None
    return headers_from_metadata(meta, source=f"hub:{target}" + (f"@{revision}" if revision else ""))


# ---- categories -------------------------------------------------------------------------------------------------------------

_VISION_PARTS = frozenset({"visual", "vision_tower", "vision_model", "vision_encoder", "multi_modal_projector", "mm_projector", "image_encoder"})
_EMBED_PARTS = frozenset({"embed_tokens", "lm_head", "wte", "word_embeddings", "tok_embeddings", "embed_out"})
_GDN_PARTS = frozenset({"linear_attn", "linear_attention", "gated_delta_net", "gdn", "mamba", "ssm"})
_ATTN_PARTS = frozenset({"self_attn", "attn", "attention", "cross_attn"})
_MLP_PARTS = frozenset({"mlp", "feed_forward", "ffn"})
_EXPERT_PARTS = frozenset({"experts", "routed_experts", "local_experts"})
_ROUTER_PARTS = frozenset({"gate", "router"})
_LAYER_PREFIX = re.compile(r"^(.*?\blayers\.\d+)\.")
_MAX_PATH_PARTS = 12               # a name is read down this many dotted parts; more is not a block layout
_MAX_PARENTS = 200_000             # distinct parents of numbered parts remembered; a hostile header cannot make the memory grow past this
GROUPING_RULE = ("a tensor is grouped with the other tensors of its numbered block (`layers.N`, or the first numbered path component that repeats "
                 "across the checkpoint's tensors, such as `h.N` or `blocks.N`); a tensor with no numbered block is a unit on its own")


def _repeated_parents(names) -> frozenset:
    """The dotted prefixes under which a numbered part takes at least two different values across `names` (`model.layers` for `model.layers.0.`
    and `model.layers.1.`; `transformer.h` for `transformer.h.0.` and so on). Those numbers are block indexes. Only the first _MAX_PATH_PARTS parts of a
    name are read, a number has at most six digits, and at most _MAX_PARENTS distinct prefixes are remembered."""
    seen: dict[str, set] = {}
    for name in names:
        parts = name.split(".", _MAX_PATH_PARTS)[:_MAX_PATH_PARTS]
        for i, part in enumerate(parts):
            if part.isdigit() and len(part) <= 6:
                parent = ".".join(parts[:i])
                bucket = seen.get(parent)
                if bucket is None:
                    if len(seen) >= _MAX_PARENTS:
                        continue
                    bucket = seen[parent] = set()
                bucket.add(part)
    return frozenset(parent for parent, numbers in seen.items() if len(numbers) >= 2)


def _block_key(name: str, repeated) -> str | None:
    """The key of the numbered block a tensor sits in, or None when it sits in none. A `layers.<n>` prefix is the block as before. Otherwise the
    block is the name up to and including its first numbered part whose prefix is in `repeated`, so `transformer.h.3.mlp.experts.7.w1.weight` is in
    block `transformer.h.3` and not in an expert of its own."""
    m = _LAYER_PREFIX.match(name)
    if m:
        return m.group(1)
    if not repeated:
        return None
    parts = name.split(".", _MAX_PATH_PARTS)[:_MAX_PATH_PARTS]
    for i, part in enumerate(parts):
        if part.isdigit() and len(part) <= 6 and ".".join(parts[:i]) in repeated:
            return ".".join(parts[:i + 1])
    return None


def _is_table(parts: list, name: str) -> bool:
    """Whether a tensor belongs to the n-gram or per-layer embedding table: a part that starts with `ngram` or has `embed` or `table` in it,
    inside a PLE module or anywhere, or a per-layer embedding (`..._per_layer`). The PLE module's own compute weights (projection, gate,
    convolution) are not table tensors: they run on the mesh, and they fall through to the later rules (norms_other)."""
    if "per_layer" in name or any(p.startswith("ngram") for p in parts):
        return True
    return "ple" in parts and any("embed" in p or "table" in p for p in parts)


def categorize(name: str, shape) -> str:
    """The category of one tensor, from its name and shape. The first rule that matches wins, in the order of the module text:
    vision, MTP, n-gram or per-layer table, embedding, norms and one-dimensional tensors, shared experts, routed experts, routers,
    gated delta net, attention, dense MLP; anything else is norms_other."""
    parts = name.split(".")
    if _VISION_PARTS & set(parts):
        return "vision"
    if "mtp" in parts or any(p.startswith("mtp_") for p in parts):
        return "mtp"
    if _is_table(parts, name):
        return "ngram_ple"
    if _EMBED_PARTS & set(parts):
        return "embedding"
    if len(shape) <= 1 or any("norm" in p for p in parts):
        return "norms_other"
    if any(p.startswith("shared_expert") for p in parts):
        return "shared_experts"
    if _EXPERT_PARTS & set(parts):
        return "routed_experts"
    if len(parts) >= 2 and parts[-2] in _ROUTER_PARTS:
        return "norms_other"
    if _GDN_PARTS & set(parts):
        return "gdn"
    if _ATTN_PARTS & set(parts):
        return "attention"
    if _MLP_PARTS & set(parts):
        return "dense_mlp"
    return "norms_other"


# ---- the estimate -----------------------------------------------------------------------------------------------------------

def _number(value, what: str, *, lo: float, allow_equal: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < lo or (value == lo and not allow_equal):
        raise ResidencyError(f"{what} must be a finite number {'of at least' if allow_equal else 'above'} {lo:g}")
    return float(value)


def _chip_list(values, what: str) -> list[int]:
    out = []
    for v in values or ():
        if isinstance(v, bool) or not isinstance(v, int) or v < 1:
            raise ResidencyError(f"{what} must be whole numbers of at least 1")
        out.append(v)
    return out


def _policy(override) -> tuple[dict, str]:
    policy = dict(DEFAULT_POLICY)
    if override:
        if not isinstance(override, dict):
            raise ResidencyError("the policy must map a category to a dtype")
        for cat, dtype in override.items():
            if cat not in CATEGORIES:
                raise ResidencyError(f"the policy names an unknown category {cat!r} (known: {', '.join(CATEGORIES)})")
            if dtype not in POLICY_DTYPES:
                raise ResidencyError(f"the policy gives {cat} the dtype {dtype!r} (known: {', '.join(POLICY_DTYPES)})")
            policy[cat] = dtype
    return policy, "caller" if override else "default (choice)"


# Name markers of packed quantized checkpoints (GPTQ and AWQ `qweight`, `qzeros`, `g_idx`; MXFP4 `..._blocks`; bitsandbytes `absmax`,
# `quant_map`, `quant_state`). In these, one stored element holds several logical weights, so element counts and byte counts do not give
# the weights a block format would store. The list is not exhaustive; a packed format it does not know is counted by stored elements.
_PACKED_LAST = frozenset({"qweight", "qzeros", "g_idx", "absmax", "quant_map", "SCB"})


def _packed_marker(t: Tensor) -> str | None:
    parts = t.name.split(".")
    if parts[-1] in _PACKED_LAST:
        return parts[-1]
    if "quant_state" in parts or any(p.startswith("quant_state") for p in parts):
        return "quant_state"
    if t.dtype == "U8" and (parts[-1].endswith("_blocks") or parts[-1] == "blocks"):
        return "blocks"
    return None


def _stored_bytes(t: Tensor, dtype: str) -> int:
    """Bytes of one tensor on its way to the chips under `dtype` (a block format or bf16)."""
    if t.dtype in FLOAT_DTYPES and len(t.shape) >= 2:
        return int(t.elements * decode_floor.DTYPE_BYTES[dtype])
    return t.bytes


def _fits(device: int, n: int, dram: float, reserve: float, kv: float) -> tuple[float, float, bool]:
    used = device / n + reserve * GIB + kv * GIB / n
    free = dram * GIB - used
    return used, free, free >= 0


def _chips_needed(device: int, dram: float, reserve: float, kv: float) -> int:
    """The smallest chip count whose rows fit, by exact arithmetic on the floats' values (no search, so any size ends at once)."""
    from fractions import Fraction
    room = (Fraction(dram) - Fraction(reserve)) * GIB
    return max(1, math.ceil((Fraction(device) + Fraction(kv) * GIB) / room))


def _gib(n: float) -> str:
    return f"{n / GIB:.1f} GiB"


def _precision_options(by_precision: dict, dram: float, reserve: float, kv: float) -> dict:
    """The whole model at each listed precision: the bytes on the chips and the chips they need (the same arithmetic as the table's rows)."""
    return {"label": "estimate", "rows": [{"precision": p, "device_bytes": int(by_precision[p]),
                                           "chips_needed": _chips_needed(int(by_precision[p]), dram, reserve, kv)} for p in PRECISIONS],
            "note": "one block format (or bf16) for every floating-point weight of two or more dimensions; categories the policy keeps on the host stay there"}


def shortfall(record, chips, what: str = "this box") -> str | None:
    """A sentence when the record's estimate needs more chips than `chips`, else None (also None when `chips` is unknown or the record is
    unusable). `what` names the limit in the sentence: "this box", or "stage 2's lease" when the stage gets fewer chips than the box has."""
    if not usable(record) or chips is None:
        return None
    need = record["estimated"]["chips_needed"]
    if need <= chips:
        return None
    a = {x["name"]: x for x in record.get("assumptions", [])}
    dram = a.get("dram_gib_per_chip", {}).get("value", "?")
    policy = record["estimated"].get("policy_text")
    text = (f"the model does not fit {what}: the estimate needs at least {need} chips and {what} has {chips} chips "
            f"({_gib(record['estimated']['device_bytes'])} of weights on the chips {policy if isinstance(policy, str) else 'under the stated policy'}, "
            f"DRAM {dram} GiB per chip; the DRAM figure is a choice and no allocator reading has been taken)")
    tail = _options_sentence(record, chips, what)
    return f"{text}. {tail}" if tail else text


def _options_sentence(record, chips: int, what: str) -> str:
    """The precision options of `record` against `chips`: every listed precision with the chips it needs, and either the highest one
    that fits or that none does. An empty string when the record has no options (an older record) or they are damaged."""
    opts = (record.get("estimated") or {}).get("precision_options")
    rows = opts.get("rows") if isinstance(opts, dict) else None
    whole = lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 1      # noqa: E731
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) and isinstance(r.get("precision"), str) and whole(r.get("chips_needed"))
                                                         for r in rows):
        return ""
    listing = ", ".join(f"{r['precision']} needs {r['chips_needed']} chips" for r in rows)
    best = next((r for r in rows if r["chips_needed"] <= chips), None)
    head = f"for the whole model at each listed precision: {listing}"
    if best is None:
        return f"{head}; the model does not fit {what} at any listed precision"
    return f"{head}; the highest listed precision that fits is {best['precision']}, which needs {best['chips_needed']} chips"


def usable(record) -> bool:
    """Whether `record` holds every field `shortfall` and the supervisor's check read: a ran scan, a whole `chips_needed` of at least 1, a
    non-negative whole `device_bytes`, a text label and assumptions that are objects with a text name. A damaged record is no claim."""
    if not isinstance(record, dict) or record.get("scan") != "ran" or not isinstance(record.get("label"), str):
        return False
    est = record.get("estimated")
    if not isinstance(est, dict):
        return False
    whole = lambda v: isinstance(v, int) and not isinstance(v, bool)      # noqa: E731
    if not whole(est.get("chips_needed")) or est["chips_needed"] < 1 or not whole(est.get("device_bytes")) or est["device_bytes"] < 0:
        return False
    a = record.get("assumptions", [])
    return isinstance(a, list) and all(isinstance(x, dict) and isinstance(x.get("name"), str) for x in a)


def machine_facts(path) -> tuple[int | None, int | None]:
    """(free disk bytes on the filesystem that holds `path`, or on its nearest existing parent when `path` is not made yet, and the host's
    MemAvailable in bytes). Either is None when it cannot be read; a reading that is not made is no claim, and a MemAvailable of zero is a
    reading. Never raises."""
    free = None
    if isinstance(path, (str, os.PathLike)):
        probe = Path(path)
        for _ in range(64):                              # a cache root that does not exist yet is measured at its nearest existing parent
            try:
                free = int(shutil.disk_usage(probe).free)
                break
            except (FileNotFoundError, NotADirectoryError):
                if probe.parent == probe:
                    break
                probe = probe.parent
            except (OSError, ValueError, OverflowError):
                break
    try:
        mem = sizing.mem_available_bytes()
    except Exception:                                    # a damaged /proc/meminfo is not a finding about the model
        mem = None
    # A MemAvailable of zero is a reading (no host RAM is available), and the RAM check must still run on it; only a figure that is not a
    # whole non-negative number is no reading.
    return free, mem if isinstance(mem, int) and not isinstance(mem, bool) and mem >= 0 else None


def machine_check(record, *, free_disk_bytes, mem_available_bytes, cache_declared: bool = False) -> tuple[str | None, str | None]:
    """(a blocking sentence, a notice) about the record's first boot against the machine's readings; each is None when there is nothing to say.
    A reading that is None, a damaged record and a record that was skipped are no claim. The record's `first_boot` figures are estimates (the
    converted cache, the download when the snapshot is not local, and the conversion's host RAM); the machine's figures are readings.

    Blocking: the host RAM of the conversion, and the download of the checkpoint (it needs the disk whatever the port does). The converted-weight
    cache is a notice, unless the caller says the port writes one (`cache_declared`): a port's `load(model_dir, mesh_device)` may convert straight
    onto the chips and write no cache, and the check runs before stage 2 has written the port, so nothing a port declares exists yet (review of
    PR 435, round 3)."""
    if not usable(record):
        return None, None
    fb = record["estimated"].get("first_boot")
    whole = lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 0      # noqa: E731
    if not isinstance(fb, dict) or not whole(fb.get("disk_bytes")) or not whole(fb.get("host_ram_bytes")):
        return None, None
    parts, notice = [], None
    if whole(mem_available_bytes) and fb["host_ram_bytes"] > mem_available_bytes:
        grouping = fb.get("conversion_grouping")
        detail = ""
        if isinstance(grouping, dict) and whole(grouping.get("blocks")) and whole(grouping.get("tensors_alone")) and whole(fb.get("conversion_in_flight_bytes")):
            detail = (f" (an estimate: its in-flight part, {_gib(fb['conversion_in_flight_bytes'])}, is the largest of {grouping['blocks']} numbered blocks "
                      f"and {grouping['tensors_alone']} single tensors; {GROUPING_RULE})")
        parts.append(f"the model does not fit this box: the first boot's conversion needs about {_gib(fb['host_ram_bytes'])} of host RAM "
                     f"and {_gib(mem_available_bytes)} is available{detail}")
    if whole(free_disk_bytes) and fb["disk_bytes"] > free_disk_bytes:
        download = fb.get("checkpoint_bytes_to_download")
        if whole(download) and download > free_disk_bytes:
            parts.append(f"the checkpoint download needs about {_gib(download)} of free disk and the cache disk has {_gib(free_disk_bytes)} free")
        elif cache_declared:
            extra = " and the download of the checkpoint" if whole(download) and download > 0 else ""
            parts.append(f"the first boot needs about {_gib(fb['disk_bytes'])} of free disk for the converted weights{extra}, "
                         f"and the cache disk has {_gib(free_disk_bytes)} free")
        else:
            notice = (f"the first boot would need about {_gib(fb['disk_bytes'])} of free disk if the port writes a converted-weight cache, and the "
                      f"cache disk has {_gib(free_disk_bytes)} free. Nothing declares that this port writes a cache (a loader that converts "
                      "straight onto the chips needs none), so this is a notice and not a block; the estimate was not measured")
    block = ("; ".join(parts) + " (estimates of the first boot against the machine's readings; the converted size was not measured)") if parts else None
    return block, notice


def machine_shortfall(record, *, free_disk_bytes, mem_available_bytes, cache_declared: bool = False) -> str | None:
    """The blocking sentence of `machine_check` (the host RAM of the conversion, the download, and the converted cache when `cache_declared`)."""
    return machine_check(record, free_disk_bytes=free_disk_bytes, mem_available_bytes=mem_available_bytes, cache_declared=cache_declared)[0]


def _validate_totals(parameters) -> dict:
    if not isinstance(parameters, dict) or not parameters:
        raise ResidencyError("the parameter counts must be a non-empty mapping from a dtype name to a count")
    out = {}
    for dtype, count in parameters.items():
        if not isinstance(dtype, str) or dtype not in DTYPE_WIDTH:
            raise ResidencyError("the parameter counts name a dtype that is not a safetensors dtype (known: " + ", ".join(sorted(DTYPE_WIDTH)) + ")")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0 or count > MAX_TOTAL_PARAMS:
            raise ResidencyError(f"the count for {dtype} must be a whole number from 0 to {MAX_TOTAL_PARAMS}")
        out[dtype] = count
    if sum(out.values()) == 0:
        raise ResidencyError("the parameter counts hold no parameters")
    return out


def totals_bound(parameters, *, box_chips=None, dram_gib=None, kv_gib=None, reserve_gib=None, chips=(), source="hub totals") -> dict:
    """A lower bound on the chips a model needs from the Hub's parameter counts by dtype alone (no tensor name, shape or header was read).

    Every floating-point parameter is counted at the smaller of the bytes it is stored with (its dtype's width) and the bytes the precision
    gives, so each listed precision gets the smallest bytes any shape of that parameter could take: a weight of two or more dimensions is
    converted to the precision, and a vector keeps its stored bytes (an FP8 vector stays at one byte under bf16); integer dtypes (a packed MXFP4, NVFP4 or GPTQ repository reports its packed bytes as U8) are counted at the
    bytes they store, and no block format shrinks them. A real checkpoint holds norms and biases that no format shrinks, so the true figure is
    at least the bound. The record carries the same `shortfall` fields as an estimate, so a model that cannot fit even at the lowest listed
    precision is refused with the numbers and the precision that would fit."""
    counts = _validate_totals(parameters)
    dram_src, kv_src, res_src = ("caller" if v is not None else "default (choice)" for v in (dram_gib, kv_gib, reserve_gib))
    dram = _number(defaults.CHIP_DRAM_GIB if dram_gib is None else dram_gib, "dram_gib", lo=0, allow_equal=False)
    kv = _number(defaults.KV_BUDGET_GIB if kv_gib is None else kv_gib, "kv_gib", lo=0, allow_equal=True)
    reserve = _number(defaults.CHIP_RESERVE_GIB if reserve_gib is None else reserve_gib, "reserve_gib", lo=0, allow_equal=True)
    if reserve >= dram:
        raise ResidencyError(f"the reserve ({reserve:g} GiB per chip) must be below the DRAM per chip ({dram:g} GiB)")
    extra = _chip_list(chips, "chips")
    if box_chips is not None:
        box_chips = _chip_list([box_chips], "box_chips")[0]
    fixed = sum(n * DTYPE_WIDTH[d] for d, n in counts.items() if d not in FLOAT_DTYPES)
    # With no shape known, a floating-point parameter sits either in a tensor of two or more dimensions, which the estimator converts to the
    # precision (`_stored_bytes`), or in a vector that keeps the bytes it has in the checkpoint. The smaller of the two widths is a bound for
    # both: an FP8 vector stays at one byte under bf16, and an FP32 weight is converted down.
    by_precision = {p: fixed + int(sum(min(Fraction(DTYPE_WIDTH[d]), Fraction(decode_floor.DTYPE_BYTES[p])) * n
                                       for d, n in counts.items() if d in FLOAT_DTYPES)) for p in PRECISIONS}
    options = _precision_options(by_precision, dram, reserve, kv)
    options["note"] = ("a lower bound for each precision: every floating-point parameter is counted at the smaller of the bytes it is stored with "
                       "and the bytes the precision gives, because a parameter of unknown shape may be a weight that is converted or a vector "
                       "that is not")
    need_by = {r["precision"]: r["chips_needed"] for r in options["rows"]}
    lowest = PRECISIONS[-1]
    device = by_precision[lowest]
    need = need_by[lowest]
    table = [{"chips": n, "fits": {p: need_by[p] <= n for p in PRECISIONS}} for n in sorted(set(MESH_SIZES) | set(extra) | ({box_chips} if box_chips else set()))]
    best = next((p for p in PRECISIONS if box_chips and need_by[p] <= box_chips), None)
    notes = ["the counts are the Hub's totals by dtype; no tensor name or shape was read", BOUND_HOST_NOTE]
    if any(d not in FLOAT_DTYPES and n for d, n in counts.items()):
        notes.append("integer dtypes (U8, I8 and the like) are counted at their stored bytes: a packed checkpoint (MXFP4, NVFP4, GPTQ) reports its "
                     "packed bytes this way and no block format shrinks them")
    record = {
        "label": "bound", "scan": "ran", "source": source, "labels": {
            "measured": "the Hub's parameter counts by dtype (no safetensors header, tensor name or tensor data was read)",
            "estimated": "arithmetic on those counts and the listed assumptions: a lower bound, and no allocator reading backs it"},
        "measured": {"parameters_by_dtype": counts, "total_parameters": sum(counts.values()), "notes": notes},
        "estimated": {"device_bytes": device, "chips_needed": need, "smallest_mesh": 1 << (need - 1).bit_length(), "policy_text": BOUND_POLICY_TEXT,
                      "precision_options": options, "rows": table},
        "assumptions": [
            {"name": "dram_gib_per_chip", "value": dram, "source": dram_src, "note": "DRAM per chip; usable space is lower and was not measured"},
            {"name": "reserve_gib_per_chip", "value": reserve, "source": res_src, "note": "per chip, for trace regions, binaries and circular buffers"},
            {"name": "kv_budget_gib", "value": kv, "source": kv_src, "note": "KV cache and recurrent state for the whole model, spread evenly over the chips"},
            {"name": "placement", "value": "even spread", "source": "default (choice)", "note": "weights are divided evenly over the chips; no replication, no tile padding"}],
        "not_measured": [BOUND_NOT_MEASURED, *NOT_MEASURED[:4]], "hazards": [],
        "verdict": {"box_chips": box_chips, "fits_box": None if box_chips is None else best is not None, "chips_needed": need, "best_fit": best}}
    if box_chips is not None:
        why = shortfall(record, box_chips)
        if why:
            record["hazards"].append({"name": "model_does_not_fit_box", "reason": why})
    return record


def _hub_model_info(repo_id, revision=None):
    import huggingface_hub
    return huggingface_hub.model_info(repo_id, revision=revision)


def totals_from_hub(repo_id, revision=None, *, fetch=None) -> dict:
    """The Hub's parameter counts by dtype for a repo id: one model-info call, no safetensors header and no weight. `fetch(repo_id, revision)`
    replaces the call in tests. A failure names its type and never its text, which may hold a token."""
    if not isinstance(repo_id, str) or not _REPO_ID.match(repo_id):
        raise ResidencyError("the target is not a repo id like owner/name (--totals-only reads the Hub's parameter counts for a repo id)")
    if revision is not None and not (isinstance(revision, str) and re.match(r"^[A-Za-z0-9._/-]{1,200}$", revision)):
        raise ResidencyError("the revision is not a branch name, tag or commit hash")
    try:
        info = (fetch or _hub_model_info)(repo_id, revision)
    except ResidencyError:
        raise
    except ImportError:
        raise ResidencyError("huggingface_hub is not installed, so a repo id cannot be read") from None
    except Exception as exc:
        raise ResidencyError(f"the Hugging Face model info call for {repo_id} failed ({type(exc).__name__})") from None
    params = getattr(getattr(info, "safetensors", None), "parameters", None)
    if not isinstance(params, dict) or not params:
        raise ResidencyError(f"the Hugging Face model info lists no safetensors parameter counts for {repo_id}")
    return dict(params)


def _data_parallel(fits_one: bool, box_chips, extra: list) -> dict:
    """The data-parallel block. A copy per chip needs the whole model, the reserve and the whole KV budget on one chip, which is the table's
    one-chip row (`fits_one`). The configured chips are the box's when it is known, else the largest stated chip count, else unknown."""
    if box_chips is not None:
        configured, source = box_chips, "box"
    elif extra:
        configured, source = max(extra), "stated chip counts (the largest)"
    else:
        configured, source = None, None
    if not fits_one:
        return {"applies": False, "configured_chips": configured, "configured_from": source, "copies": 0, "chips_used": 0,
                "note": "the weights per chip do not fit one chip with the stated reserve and KV budget, so the model does not fit one chip "
                        "and a copy per chip is not possible"}
    return {"applies": True, "configured_chips": configured, "configured_from": source, "copies": configured, "chips_used": configured,
            "note": DP_NOTE + (": one copy per chip" if configured is None else f": {configured} copies, one per chip, on {configured} chips"),
            "throughput": DP_THROUGHPUT, "measured_example": copy.deepcopy(MEASURED_EXAMPLE)}


def estimate(headers: HeaderSet, *, dram_gib=None, kv_gib=None, reserve_gib=None, policy=None, chips=(), box_chips=None, sharded_chips=None, sharded_basis="stated") -> dict:
    """The residency record for `headers`. `dram_gib`, `kv_gib` and `reserve_gib` fall back to orchard/defaults.py and the record says so.
    `chips` adds rows to the table (1, 2, 4 and 8 are always there); `box_chips` is the chip count of the machine, or None when unknown.
    `sharded_chips` is the chip count the run is configured to shard the model across (tensor parallel), or None when not stated; a model
    that fits one chip and is sharded across more than one gets a hazard note. `sharded_basis` is "stated" when the caller knows the layout
    is tensor parallel, or "required chip counts" when only the counts a run must test are known (a data-parallel run needs those too, so
    the note is then conditional)."""
    if sharded_basis not in SHARDED_BASES:
        raise ResidencyError(f"sharded_basis must be one of {', '.join(SHARDED_BASES)}")
    dram_src, kv_src, res_src = ("caller" if v is not None else "default (choice)" for v in (dram_gib, kv_gib, reserve_gib))
    dram = _number(defaults.CHIP_DRAM_GIB if dram_gib is None else dram_gib, "dram_gib", lo=0, allow_equal=False)
    kv = _number(defaults.KV_BUDGET_GIB if kv_gib is None else kv_gib, "kv_gib", lo=0, allow_equal=True)
    reserve = _number(defaults.CHIP_RESERVE_GIB if reserve_gib is None else reserve_gib, "reserve_gib", lo=0, allow_equal=True)
    if reserve >= dram:
        raise ResidencyError(f"the reserve ({reserve:g} GiB per chip) must be below the DRAM per chip ({dram:g} GiB)")
    extra = _chip_list(chips, "chips")
    if box_chips is not None:
        box_chips = _chip_list([box_chips], "box_chips")[0]
    if sharded_chips is not None:
        sharded_chips = _chip_list([sharded_chips], "sharded_chips")[0]
    pol, pol_src = _policy(policy)
    packed = [m for m in map(_packed_marker, headers.tensors) if m]
    if packed:
        raise ResidencyError(f"the checkpoint holds packed quantized weights ({len(packed)} tensors, such as `{packed[0]}`): the estimate counts "
                             "logical weights from tensor shapes and cannot read packed ones, so no estimate is made")

    measured = {c: {"tensors": 0, "elements": 0, "bytes": 0} for c in CATEGORIES}
    stored = {c: 0 for c in CATEGORIES}
    uniform = {p: 0 for p in PRECISIONS}         # the chips' bytes if one precision applied to the whole model (categories on the host stay there)
    layer_groups: dict[str, int] = {}
    repeated = _repeated_parents(t.name for t in headers.tensors)
    alone_max, alone_count = 0, 0                 # tensors with no numbered block: each is a unit, counted without a table entry per tensor
    for t in headers.tensors:
        cat = categorize(t.name, t.shape)
        m = measured[cat]
        m["tensors"] += 1
        m["elements"] += t.elements
        m["bytes"] += t.bytes
        if pol[cat] == "host":
            stored[cat] += t.bytes
            continue
        stored[cat] += _stored_bytes(t, pol[cat])
        for p in PRECISIONS:
            uniform[p] += _stored_bytes(t, p)
        if t.dtype in FLOAT_DTYPES and len(t.shape) >= 2:
            unit = t.bytes + max(t.bytes, _stored_bytes(t, pol[cat]))
            key = _block_key(t.name, repeated)
            if key is None:
                alone_max, alone_count = max(alone_max, unit), alone_count + 1
            else:
                layer_groups[key] = layer_groups.get(key, 0) + unit
    by_measured = {c: v for c, v in measured.items() if v["tensors"]}
    host_cats = [c for c in CATEGORIES if pol[c] == "host" and c in by_measured]
    device = sum(stored[c] for c in by_measured if pol[c] != "host")
    host = sum(stored[c] for c in host_cats)

    counts = sorted(set(MESH_SIZES) | set(extra) | ({box_chips} if box_chips else set()))
    rows = []
    for n in counts:
        used, free, ok = _fits(device, n, dram, reserve, kv)
        rows.append({"chips": n, "per_chip_bytes": int(device / n), "used_bytes_per_chip": int(used), "free_bytes_per_chip": int(free), "fits": ok})
    need = _chips_needed(device, dram, reserve, kv)
    mesh = 1 << (need - 1).bit_length()
    routed = by_measured.get("routed_experts")
    # Where the routed experts sit follows the policy (review of PR 435, round 3): under `routed_experts=host` they stay in host memory, take no chip
    # memory, and the note and the text must not say they occupy the chips. `routed_expert_stored_bytes` is what the policy stores, wherever it sits.
    experts_on_host = pol["routed_experts"] == "host"
    moe = ({"present": True, "placement": "host" if experts_on_host else "device", "routed_expert_tensors": routed["tensors"],
            "routed_expert_checkpoint_bytes": routed["bytes"], "routed_expert_stored_bytes": stored["routed_experts"],
            "routed_expert_device_bytes": 0 if experts_on_host else stored["routed_experts"],
            "routed_expert_host_bytes": stored["routed_experts"] if experts_on_host else 0,
            "note": MOE_NOTE_HOST if experts_on_host else MOE_NOTE} if routed else {"present": False})

    flight = max(max(layer_groups.values(), default=0), alone_max)    # the largest block or lone tensor: each tensor's source plus the larger of its source and converted size
    to_download = 0 if headers.local else headers.checkpoint_bytes
    first_boot = {"snapshot_local": headers.local, "checkpoint_bytes_to_download": to_download, "cache_bytes": device,
                  "disk_bytes": device + to_download, "conversion_in_flight_bytes": flight, "host_ram_bytes": host + flight,
                  "conversion_grouping": {"rule": GROUPING_RULE, "blocks": len(layer_groups), "tensors_alone": alone_count}}
    record = {
        "label": "estimate", "scan": "ran", "source": headers.source, "labels": dict(LABELS),
        "measured": {"files": headers.files, "tensors": len(headers.tensors), "checkpoint_bytes": headers.checkpoint_bytes, "by_category": by_measured},
        "estimated": {
            "policy": pol,
            "by_category": {c: {"placement": "host" if pol[c] == "host" else "device", "stored_bytes": stored[c]} for c in by_measured},
            "host_resident_categories": host_cats, "device_bytes": device, "host_bytes": host,
            "rows": rows, "chips_needed": need, "smallest_mesh": mesh, "first_boot": first_boot,
            "precision_options": _precision_options(uniform, dram, reserve, kv), "mixture_of_experts": moe,
            "data_parallel": _data_parallel(next(r for r in rows if r["chips"] == 1)["fits"], box_chips, extra)},
        "assumptions": [
            {"name": "dram_gib_per_chip", "value": dram, "source": dram_src, "note": "DRAM per chip; usable space is lower and was not measured"},
            {"name": "reserve_gib_per_chip", "value": reserve, "source": res_src, "note": "per chip, for trace regions, binaries and circular buffers"},
            {"name": "kv_budget_gib", "value": kv, "source": kv_src, "note": "KV cache and recurrent state for the whole model, spread evenly over the chips"},
            {"name": "policy", "value": pol, "source": pol_src, "note": "bytes per value from orchard/decode_floor.py DTYPE_BYTES; block formats apply to floating-point tensors with two or more dimensions"},
            {"name": "placement", "value": "even spread", "source": "default (choice)", "note": "weights are divided evenly over the chips; no replication, no tile padding"}],
        "not_measured": list(NOT_MEASURED), "hazards": [], "verdict": {"box_chips": box_chips, "fits_box": None, "chips_needed": need}}
    if box_chips is not None:
        record["verdict"]["fits_box"] = need <= box_chips
        why = shortfall(record, box_chips)
        if why:
            record["hazards"].append({"name": "model_does_not_fit_box", "reason": why})
    if sharded_chips and sharded_chips > 1 and need == 1:
        asks = (f"the run is configured to shard it across {sharded_chips} chips" if sharded_basis == "stated" else
                f"the run's required chip counts ask for {sharded_chips} chips, and if the port shards the model across them (tensor parallel)")
        record["hazards"].append({"name": "fits_one_chip_but_sharded", "basis": sharded_basis, "reason": (
            f"the model fits one chip by this estimate (weights, reserve and KV budget; no allocator reading), and {asks}. "
            f"Tensor parallel adds collective operations between the chips. A data parallel layout (a full copy on each chip, "
            f"no collective operations) is an option for throughput; measured scaling is needed before expecting any gain. This is a note and "
            f"blocks nothing")})
    return record


# ---- stage 0 and the supervisor -----------------------------------------------------------------------------------------------

def _skipped(why: str) -> dict:
    return {"label": "estimate", "scan": "skipped", "reason": why, "hazards": []}


def write_residency(stage_dir, model_dir=None, box_chips=None, **kw) -> dict:
    """Build the record for stage 0 and write it to `stage_dir/residency.json` (all or nothing). Never raises for an input it cannot count:
    that is a record with `scan: skipped` and the reason. Reads a local directory only; stage 0 never makes a network call here.
    `model_dir` falls back to the stage's triage_config.json `model_snapshot`."""
    stage_dir = Path(stage_dir)
    if not model_dir:
        from orchard import stage0_scan
        model_dir = stage0_scan._triage_dirs(stage_dir)[0]
    if not model_dir:
        rec = _skipped("no model directory is known for this stage")
    elif not os.path.isdir(model_dir):
        rec = _skipped("the model path is not a directory")
    else:
        try:
            rec = estimate(read_headers(model_dir), box_chips=box_chips, **kw)
        except ResidencyError as exc:
            rec = _skipped(f"the estimate could not be made: {exc}")
    safeio.atomic_write(stage_dir / FILE, json.dumps(rec, indent=1, sort_keys=True) + "\n")
    return rec


def read_record(run_dir) -> dict | None:
    """Stage 0's residency record when it ran and holds a usable `chips_needed`, else None."""
    try:
        rec = safeio.read_json(Path(run_dir) / "stages" / "0" / FILE, max_bytes=MAX_RECORD_BYTES, expect=dict)
    except safeio.SafeReadError:
        return None
    return rec if usable(rec) else None


# ---- the command ------------------------------------------------------------------------------------------------------------

def parser(add_help=True) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="tt-orchard residency", add_help=add_help,
                                 description="Estimate how many chips a checkpoint needs from its safetensors headers (no weights are read).")
    ap.add_argument("target", help="a model snapshot directory, or a repo id like owner/name (headers only, over the network)")
    ap.add_argument("--revision", help="the branch, tag or commit of a repo id")
    ap.add_argument("--totals-only", action="store_true",
                    help="for a repo id: read only the Hub's parameter counts by dtype (one call; no safetensors header) and print a lower bound by "
                         "precision. Use it for a repository too large to read header by header")
    ap.add_argument("--chips", type=int, nargs="+", metavar="N", help="chip counts to show; the exit status is 1 when none of them holds the model")
    ap.add_argument("--dram-gib", type=float, help=f"DRAM per chip in GiB (default {defaults.CHIP_DRAM_GIB:g}, a choice)")
    ap.add_argument("--reserve-gib", type=float, help=f"per-chip reserve in GiB (default {defaults.CHIP_RESERVE_GIB:g}, a choice)")
    ap.add_argument("--kv-gib", type=float, help=f"KV cache and state budget for the whole model in GiB (default {defaults.KV_BUDGET_GIB:g}, a choice)")
    ap.add_argument("--policy", action="append", default=[], metavar="CATEGORY=DTYPE",
                    help=f"override the quantization policy for one category; dtype is {', '.join(POLICY_DTYPES)}. Repeat for more")
    ap.add_argument("--json", action="store_true", help="print the record as JSON")
    return ap


def _text(rec: dict, stated: list[int]) -> list[str]:
    m, e = rec["measured"], rec["estimated"]
    lines = [f"residency estimate for {rec['source']}  (label: {rec['label']})",
             f"  measured: {m['tensors']} tensors in {m['files']} files, {_gib(m['checkpoint_bytes'])} ({LABELS['measured']})",
             f"  estimated: {LABELS['estimated']}", "",
             f"  {'category':<16}{'tensors':>8}{'checkpoint':>13}{'policy':>8}{'on chips':>13}{'on host':>13}"]
    for c in CATEGORIES:
        if c not in m["by_category"]:
            continue
        mc, ec = m["by_category"][c], e["by_category"][c]
        dev, host = (ec["stored_bytes"], 0) if ec["placement"] == "device" else (0, ec["stored_bytes"])
        lines.append(f"  {c:<16}{mc['tensors']:>8}{_gib(mc['bytes']):>13}{e['policy'][c]:>8}{_gib(dev):>13}{_gib(host):>13}")
    lines += ["", f"  on the chips {_gib(e['device_bytes'])}; on the host {_gib(e['host_bytes'])}"
              + (f" ({', '.join(e['host_resident_categories'])})" if e["host_resident_categories"] else ""),
              f"  {'chips':>6}{'weights/chip':>14}{'used/chip':>12}{'free/chip':>12}  fits"]
    for r in e["rows"]:
        lines.append(f"  {r['chips']:>6}{_gib(r['per_chip_bytes']):>14}{_gib(r['used_bytes_per_chip']):>12}{_gib(r['free_bytes_per_chip']):>12}  {'yes' if r['fits'] else 'no'}")
    fb = e["first_boot"]
    d = e["data_parallel"]
    if d["applies"]:
        copies = "the configured chip count is not known, so the number of copies is not stated" if d["copies"] is None else (
            f"{d['copies']} copies on {d['chips_used']} chips (configured chips: {d['configured_chips']}, from {d['configured_from']})")
        lines += ["", f"  data parallel: the model fits one chip; {copies}", f"    {d['note']}", f"    throughput: {d['throughput']}"]
    else:
        lines += ["", "  data parallel: not available; the model does not fit one chip with the stated reserve and KV budget"]
    lines += ["", "  the whole model at one precision (categories the policy keeps on the host stay there):"]
    lines += [f"    {r['precision']:<6}{_gib(r['device_bytes']):>13} on the chips  needs {r['chips_needed']} chips" for r in e["precision_options"]["rows"]]
    moe = e["mixture_of_experts"]
    if moe["present"]:
        where = (f"{_gib(moe['routed_expert_host_bytes'])} in host memory" if moe.get("placement") == "host"
                 else f"{_gib(moe['routed_expert_stored_bytes'])} on the chips")
        lines += ["", f"  mixture of experts: {moe['routed_expert_tensors']} routed-expert tensors, {where} under the policy", f"    {moe['note']}"]
    lines += ["", f"  chips needed: at least {e['chips_needed']} (smallest mesh {e['smallest_mesh']})",
              f"  first boot: free disk {_gib(fb['disk_bytes'])} ({'snapshot already local' if fb['snapshot_local'] else 'includes the download'}), "
              f"host RAM {_gib(fb['host_ram_bytes'])}", "", "  assumptions:"]
    for a in rec["assumptions"]:
        value = "see the table above" if a["name"] == "policy" else a["value"]
        lines.append(f"    {a['name']} = {value}  [{a['source']}] {a['note']}")
    lines.append("  not measured:")
    lines += [f"    {s}" for s in rec["not_measured"]]
    if stated:
        ok = [r["chips"] for r in e["rows"] if r["chips"] in stated and r["fits"]]
        lines += ["", f"  stated chip counts {stated}: " + (f"the model fits at {ok}" if ok else "the model does not fit any of them")]
    return lines


def _text_bound(rec: dict, stated: list[int]) -> list[str]:
    m, e = rec["measured"], rec["estimated"]
    lines = [f"residency bound for {rec['source']}  (label: {rec['label']})",
             f"  measured: {m['total_parameters'] / 1e9:.1f} billion parameters ({', '.join(f'{d} {n:,}' for d, n in sorted(m['parameters_by_dtype'].items()))})",
             f"    {rec['labels']['measured']}", f"  estimated: {rec['labels']['estimated']}"]
    lines += [f"  note: {n}" for n in m["notes"]]
    lines += ["", "  the whole model at one precision (a lower bound):"]
    lines += [f"    {r['precision']:<6}{_gib(r['device_bytes']):>13} on the chips  needs at least {r['chips_needed']} chips" for r in e["precision_options"]["rows"]]
    lines += ["", f"  {'chips':>6}  " + "  ".join(f"{p:<6}" for p in PRECISIONS)]
    lines += [f"  {r['chips']:>6}  " + "  ".join(f"{('fits' if r['fits'][p] else 'no'):<6}" for p in PRECISIONS) for r in e["rows"]]
    lines += ["", f"  chips needed: at least {e['chips_needed']} (the lowest listed precision, {PRECISIONS[-1]}; smallest mesh {e['smallest_mesh']})", "", "  assumptions:"]
    lines += [f"    {a['name']} = {a['value']}  [{a['source']}] {a['note']}" for a in rec["assumptions"]]
    lines.append("  not measured:")
    lines += [f"    {s}" for s in rec["not_measured"]]
    for h in rec["hazards"]:
        lines += ["", f"  {h['name']}: {h['reason']}"]
    if stated:
        ok = [r["chips"] for r in e["rows"] if r["chips"] in stated and any(r["fits"].values())]
        lines += ["", f"  stated chip counts {stated}: " + (f"the bound allows the model at {ok}" if ok else "the model does not fit any of them at any listed precision")]
    return lines


def run(args, out=None) -> int:
    """The command body. Exit 0 when it printed an estimate (and a stated chip count holds the model, if any was stated), 1 when none does,
    2 for an input it refused."""
    out = out or sys.stdout
    try:
        overrides = {}
        for item in args.policy or []:
            cat, sep, dtype = item.partition("=")
            if not sep:
                raise ResidencyError(f"--policy needs CATEGORY=DTYPE, got {item!r}")
            overrides[cat] = dtype
        stated = list(args.chips or [])
        if getattr(args, "totals_only", False):
            if overrides:
                raise ResidencyError("--policy has no effect with --totals-only (the bound lists every whole-model precision)")
            rec = totals_bound(totals_from_hub(args.target, args.revision), dram_gib=args.dram_gib, kv_gib=args.kv_gib, reserve_gib=args.reserve_gib,
                               chips=stated, box_chips=max(stated) if stated else None, source=f"hub totals:{args.target}")
            if args.json:
                print(json.dumps(rec, indent=2, sort_keys=True), file=out)
            else:
                print("\n".join(_text_bound(rec, stated)), file=out)
            if stated and not any(any(r["fits"].values()) for r in rec["estimated"]["rows"] if r["chips"] in stated):
                if args.json:
                    print("residency: the model does not fit any stated chip count at any listed precision", file=sys.stderr)
                return 1
            return 0
        rec = estimate(read_target(args.target, revision=args.revision), dram_gib=args.dram_gib, kv_gib=args.kv_gib, reserve_gib=args.reserve_gib,
                       policy=overrides or None, chips=stated)
    except ResidencyError as exc:
        print(f"residency: {exc}", file=out)
        return 2
    if args.json:
        print(json.dumps(rec, indent=2, sort_keys=True), file=out)
    else:
        print("\n".join(_text(rec, stated)), file=out)
    if stated and not any(r["fits"] for r in rec["estimated"]["rows"] if r["chips"] in stated):
        if args.json:
            print("residency: the model does not fit any stated chip count", file=sys.stderr)
        return 1
    return 0


def main(argv=None, out=None) -> int:
    return run(parser().parse_args(argv), out=out)


if __name__ == "__main__":
    raise SystemExit(main())
