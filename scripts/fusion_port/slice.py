"""Build a reduced Qwen3.5-35B-A3B checkpoint (first N layers + embeddings + head) by range-reading only those tensors."""  Usage: slice.py N OUTDIR INDEX_JSON
import json, re, struct, sys, os, concurrent.futures as cf
from huggingface_hub import HfFileSystem
N = int(sys.argv[1]); out = sys.argv[2]
REPO = "Qwen/Qwen3.5-35B-A3B"
fs = HfFileSystem()
idx = json.load(open(sys.argv[3]))["weight_map"]    # the checkpoint's model.safetensors.index.json (hf download of that one file)
want = {}
for k, sh in idx.items():
    if k.startswith("mtp.") or k.startswith("model.visual"): continue
    m = re.match(r"model\.language_model\.layers\.(\d+)\.", k)
    if m and int(m.group(1)) >= N: continue
    want.setdefault(sh, []).append(k)
tensors = []   # (key, dtype, shape, shard, start, end)
for sh, keys in want.items():
    with fs.open(f"{REPO}/{sh}", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n
    for k in keys:
        h = hdr[k]; tensors.append((k, h["dtype"], h["shape"], sh, base + h["data_offsets"][0], base + h["data_offsets"][1]))
total = sum(t[5] - t[4] for t in tensors); print("tensors", len(tensors), "bytes", total / 1e9, flush=True)
def fetch(t):
    k, dt, shp, sh, a, b = t
    with fs.open(f"{REPO}/{sh}", "rb") as f:
        f.seek(a); buf = f.read(b - a)
    assert len(buf) == b - a
    return k, buf
hdr = {}; off = 0
for k, dt, shp, sh, a, b in tensors:
    hdr[k] = {"dtype": dt, "shape": shp, "data_offsets": [off, off + b - a]}; off += b - a
hb = json.dumps(hdr).encode(); hb += b" " * (-len(hb) % 8)
done = 0
with open(os.path.join(out, "model.safetensors"), "wb") as w:
    w.write(struct.pack("<Q", len(hb))); w.write(hb)
    with cf.ThreadPoolExecutor(8) as ex:
        for i in range(0, len(tensors), 16):     # batches bound the memory held in flight
            for k, buf in ex.map(fetch, tensors[i:i + 16]):     # map keeps order, matching the header offsets
                w.write(buf); done += len(buf)
            print(f"{done/1e9:.1f} GB", flush=True)
json.dump({"metadata": {}, "weight_map": {k: "model.safetensors" for k in hdr}}, open(os.path.join(out, "model.safetensors.index.json"), "w"))
print("done", flush=True)
