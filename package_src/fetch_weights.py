#!/usr/bin/env python3
"""Fetch the text weights of Qwen/Qwen3.5-35B-A3B at a pinned revision into a local model directory for this package.

Reads only the tensors the text model needs (embeddings, 40 decoder layers, final norm, lm_head) by byte ranges from the Hub, in
parallel, and writes one `model.safetensors` plus an index, the tokenizer files, the license and an edited `config.json`.
The vision tower (`model.visual.*`) and the multi-token-prediction head (`mtp.*`) are not fetched. About 69.3 GB.

Why ranges: `hf download` of the same files ran at 4 to 6 MB/s on the machine that built this package; ranged reads ran at 35 to
50 MB/s. `hf download Qwen/Qwen3.5-35B-A3B --revision <sha>` into the Hugging Face cache also works with this package's patched
loader (it skips the vision and MTP keys), but then run.sh needs a model directory with the edited config; use this script.

The config edits (all other keys are unchanged): `architectures` becomes ["Qwen3NextForCausalLM"] (the class the bundle code registers),
`mtp_num_hidden_layers` becomes 0, and `rope_parameters.mrope_section` / `mrope_interleaved` are removed (with them the vLLM plugin
expects a rotary delta from prefill that the bundle's generator does not return; text-only decoding does not use them).

Usage: fetch_weights.py OUTDIR [--revision SHA]   (set HF_TOKEN only if your network needs it; the model is not gated)
"""
import argparse, concurrent.futures as cf, hashlib, json, os, re, shutil, struct, sys

REPO = "Qwen/Qwen3.5-35B-A3B"
REVISION = "59d61f3ce65a6d9863b86d2e96597125219dc754"
TOKENIZER_FILES = ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json", "chat_template.jinja", "generation_config.json", "LICENSE")
BATCH = 16     # tensors in flight; bounds memory (a layer's fused experts are about 0.8 GB per tensor)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("outdir")
    ap.add_argument("--revision", default=REVISION)
    a = ap.parse_args()
    from huggingface_hub import HfFileSystem, hf_hub_download
    os.makedirs(a.outdir, exist_ok=True)
    fs = HfFileSystem()
    base = f"{REPO}@{a.revision}"
    for name in TOKENIZER_FILES + ("config.json", "model.safetensors.index.json"):
        shutil.copyfile(hf_hub_download(REPO, name, revision=a.revision), os.path.join(a.outdir, name + (".orig" if name.endswith(".index.json") else "")))
    idx = json.load(open(os.path.join(a.outdir, "model.safetensors.index.json.orig")))["weight_map"]
    os.remove(os.path.join(a.outdir, "model.safetensors.index.json.orig"))
    want = {}
    for k, sh in idx.items():
        if k.startswith("mtp.") or k.startswith("model.visual."):
            continue
        want.setdefault(sh, []).append(k)
    tensors = []
    for sh, keys in sorted(want.items()):
        with fs.open(f"{base}/{sh}", "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr = json.loads(f.read(n))
        for k in keys:
            h = hdr[k]
            tensors.append((k, h["dtype"], h["shape"], sh, 8 + n + h["data_offsets"][0], 8 + n + h["data_offsets"][1]))
    total = sum(t[5] - t[4] for t in tensors)
    print(f"{len(tensors)} tensors, {total / 1e9:.1f} GB", flush=True)

    def fetch(t):
        k, dt, shp, sh, lo, hi = t
        with fs.open(f"{base}/{sh}", "rb") as f:
            f.seek(lo)
            buf = f.read(hi - lo)
        if len(buf) != hi - lo:
            raise RuntimeError(f"short read for {k}")
        return buf

    hdr, off = {}, 0
    for k, dt, shp, sh, lo, hi in tensors:
        hdr[k] = {"dtype": dt, "shape": shp, "data_offsets": [off, off + hi - lo]}
        off += hi - lo
    hb = json.dumps(hdr).encode()
    hb += b" " * (-len(hb) % 8)
    done = 0
    with open(os.path.join(a.outdir, "model.safetensors"), "wb") as w:
        w.write(struct.pack("<Q", len(hb)))
        w.write(hb)
        with cf.ThreadPoolExecutor(8) as ex:
            for i in range(0, len(tensors), BATCH):
                for buf in ex.map(fetch, tensors[i:i + BATCH]):
                    w.write(buf)
                    done += len(buf)
                print(f"{done / 1e9:.1f} / {total / 1e9:.1f} GB", flush=True)
    json.dump({"metadata": {}, "weight_map": {k: "model.safetensors" for k in hdr}}, open(os.path.join(a.outdir, "model.safetensors.index.json"), "w"))
    cp = os.path.join(a.outdir, "config.json")
    c = json.load(open(cp))
    c["architectures"] = ["Qwen3NextForCausalLM"]
    t = c["text_config"]
    t["mtp_num_hidden_layers"] = 0
    rp = t.get("rope_parameters", {})
    rp.pop("mrope_section", None)
    rp.pop("mrope_interleaved", None)
    json.dump(c, open(cp, "w"), indent=1)
    json.dump({"repo": REPO, "revision": a.revision, "tensors": len(tensors), "bytes": total,
               "header_sha256": hashlib.sha256(hb).hexdigest()}, open(os.path.join(a.outdir, "fetch-record.json"), "w"), indent=1)
    print("done", flush=True)


if __name__ == "__main__":
    main()
