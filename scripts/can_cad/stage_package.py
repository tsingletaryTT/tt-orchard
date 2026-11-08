#!/usr/bin/env python3
"""Stage the Berkelium CadQuery 2-chip container package from the base bundle's own files (no rebuild, no upload).

Copies ~/orchard-runs/can-cad/base-clone (a download of mando2222/qwen3-32b-blackhole-v51: image/, code/, requirements.lock, manifest) into
`staged/`, then edits only the manifest and the card: the weights pointer (the fine-tune at a pinned revision), the serve profiles
(a 2-chip p300 profile, the default, and the 4-chip p150x4 profile the earlier stage 2 check proved) and the card sections. The image is the
base bundle's image, byte for byte (same digest), and the card says so. Run with the tt-model interpreter:
    python stage_package.py   # the interpreter that has tt_kernel (tt-model's) --profile-json profile.json --card-json card.json --repo OWNER/NAME
"""
import argparse, json, os, shutil, subprocess, sys
HERE = os.environ.get("CAN_CAD_WORK", os.getcwd())      # the work directory: caches, base-clone/, staged/
ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--staged", default=os.path.join(HERE, "staged"))
ap.add_argument("--profile-json", required=True)     # {"p300": {...}, "p150x4": {...}}
ap.add_argument("--card-json", required=True)        # fields of the manifest card
ap.add_argument("--readme", required=True)
ap.add_argument("--weights-repo", default="Mhaquehaque/berkelium-qwen3-32b-cadquery-v1-merged")
ap.add_argument("--weights-revision", default="73df255658c8b469b0b6081f989b95eff6730892")
a = ap.parse_args()
from tt_kernel.manifest import Manifest
base = os.path.join(HERE, "base-clone")
if os.path.exists(a.staged):
    shutil.rmtree(a.staged)
subprocess.check_call(["cp", "-al", base, a.staged])                 # hard links: the image blobs are not copied twice
shutil.rmtree(os.path.join(a.staged, ".cache"), ignore_errors=True)
mp = os.path.join(a.staged, "tt_kernel_manifest.json")
os.unlink(mp)                                                          # the staged manifest is a new file, not the base's through a link
m = json.load(open(os.path.join(base, "tt_kernel_manifest.json")))
m["name"] = a.repo.split("/")[1]
m["producer"]["hostname"] = "tt-orchard"
m["weights"] = {"repo_id": a.weights_repo, "revision": a.weights_revision, "allow_patterns": None, "ignore_patterns": ["eval/*", ".gitattributes"], "repo_type": "model"}
profiles = json.load(open(a.profile_json))
c = m["container"]
c["serve_profiles"] = [dict(name=k, **v) for k, v in profiles.items()]
c["default_profile"] = "p300"
# The base manifest's serve defaults ask for on-device sampling (tt.sample_on_device_mode = "all"). On first decode that fails here with
# "ttnn_decode_forward got on_device_logits=True but no on-device sampling module exists". The 4-chip stage 2 check ran without it, so it is
# removed from the defaults of this package.
c["serve"]["additional_config"].get("tt", {}).pop("sample_on_device_mode", None)
m["device_count"] = 2
c["card"] = json.load(open(a.card_json))
Manifest.from_json(json.dumps(m))                                      # refuses a manifest the installed tt-model would refuse
open(mp, "w").write(json.dumps(m, indent=2) + "\n")
os.unlink(os.path.join(a.staged, "README.md"))
shutil.copy(a.readme, os.path.join(a.staged, "README.md"))
print("staged", a.staged)
