"""Issue 119: a gated model the operator already downloaded passes the hub row (with a warning naming the revision),
and a snapshot is local by what the repo needs, not by a config.json alone. The harness never uses a token."""
from orchard import preflight as pf
from test_preflight import cfg, payload, real, signals


def _gated_hub(files, **over):
    sib = [{"rfilename": f, "size": 1} for f in files] + [{"rfilename": "LICENSE.md", "size": 1}]
    return pf.parse_hub(payload(**{"gated": "manual", "siblings": sib, **over}))


def _audio_snap(tmp_path, rev="8bdfb53c6b0f"):
    snap = tmp_path / "hf" / "hub" / "models--stabilityai--audio" / "snapshots" / rev
    (snap / "t5").mkdir(parents=True)
    (snap / "model_config.json").write_text('{"model_type": "diffusion_cond", "model": {}}')      # what stage 0's detector accepts
    (snap / "model.safetensors").write_text("w")
    (snap / "t5" / "model.safetensors").write_text("w")
    return snap


def test_a_gated_model_that_is_complete_on_disk_passes_with_a_warning_that_names_the_revision(tmp_path):
    snap = _audio_snap(tmp_path)
    h = _gated_hub(["model.safetensors", "model_config.json", "t5/model.safetensors"])
    c = pf.check_hub(h, None, local=True, local_path=snap)
    assert c.status == "warn" and c.reason is None
    assert "8bdfb53c6b0f" in c.detail and "on disk" in c.detail
    assert c.data is h


def test_a_gated_model_missing_a_listed_weight_file_still_blocks(tmp_path):
    snap = _audio_snap(tmp_path)
    h = _gated_hub(["model.safetensors", "t5/model.safetensors", "extra/model-00002.safetensors"])
    assert pf.check_hub(h, None, local=True, local_path=snap).reason == "credentials-needed"


def test_a_gated_model_not_on_disk_blocks_with_the_existing_message():
    c = pf.check_hub(_gated_hub(["model.safetensors"]), None, local=False, local_path=None)
    assert (c.status, c.reason) == ("block", "credentials-needed") and "Download it yourself" in c.detail


def test_a_gated_local_model_with_no_license_still_blocks_for_review(tmp_path):
    snap = _audio_snap(tmp_path)
    h = pf.parse_hub(payload(gated="auto", cardData={}, siblings=[{"rfilename": "model.safetensors", "size": 1}]))
    assert pf.check_hub(h, None, local=True, local_path=snap).reason == "license-needs-review"


def test_a_private_model_on_disk_is_still_a_block(tmp_path):
    snap = _audio_snap(tmp_path)
    h = pf.parse_hub(payload(private=True, siblings=[{"rfilename": "model.safetensors", "size": 1}]))
    assert pf.check_hub(h, None, local=True, local_path=snap).reason == "credentials-needed"


def test_a_gated_model_whose_code_files_are_local_keeps_the_code_warning(tmp_path):
    snap = _audio_snap(tmp_path)
    (snap / "run.py").write_text("x")
    h = _gated_hub(["model.safetensors", "run.py"])
    c = pf.check_hub(h, None, local=True, local_path=snap)
    assert c.status == "warn" and "run.py" in c.detail and "on disk" in c.detail


def test_model_config_json_marks_a_snapshot_as_local(tmp_path):
    snap = _audio_snap(tmp_path)
    assert real(tmp_path).local_snapshot("stabilityai/audio") == snap


def test_a_snapshot_with_no_config_file_of_any_kind_is_not_local(tmp_path):
    snap = _audio_snap(tmp_path)
    (snap / "model_config.json").unlink()
    assert real(tmp_path).local_snapshot("stabilityai/audio") is None


def test_a_snapshot_whose_index_lists_a_missing_shard_is_not_local(tmp_path):
    snap = _audio_snap(tmp_path)
    (snap / "model.safetensors.index.json").write_text(
        '{"weight_map": {"a": "model-00002.safetensors", "b": "model.safetensors"}}')
    assert real(tmp_path).local_snapshot("stabilityai/audio") is None
    (snap / "model-00002.safetensors").write_text("w")
    assert real(tmp_path).local_snapshot("stabilityai/audio") == snap


def test_a_snapshot_with_one_dangling_weight_link_is_not_local(tmp_path):
    snap = _audio_snap(tmp_path)
    (snap / "t5" / "gone.safetensors").symlink_to(tmp_path / "nowhere")
    assert real(tmp_path).local_snapshot("stabilityai/audio") is None


def test_a_malformed_index_makes_the_snapshot_not_local(tmp_path):
    snap = _audio_snap(tmp_path)
    (snap / "model.safetensors.index.json").write_text("not json")
    assert real(tmp_path).local_snapshot("stabilityai/audio") is None


def test_the_run_passes_the_disk_snapshot_to_the_hub_check(tmp_path):
    snap = _audio_snap(tmp_path)
    h = _gated_hub(["model.safetensors"])
    s = signals(hub_info=lambda m: (h, None), local_snapshot=lambda m: snap)
    out = pf.run_preflight(cfg(tmp_path), "stabilityai/audio", accept_credentials=False, signals=s)
    row = next(c for c in out if c.name == "hub")
    assert row.status == "warn" and "8bdfb53c6b0f" in row.detail


def test_a_copy_that_is_not_the_hubs_current_revision_says_so(tmp_path):
    snap = _audio_snap(tmp_path, rev="1111111111aa")
    h = _gated_hub(["model.safetensors"], sha="2222222222bb")
    c = pf.check_hub(h, None, local=True, local_path=snap)
    assert c.status == "warn" and "1111111111aa" in c.detail and "222222222222"[:10] in c.detail and "not the newest" in c.detail


def test_a_copy_at_the_hubs_current_revision_has_no_newest_note(tmp_path):
    snap = _audio_snap(tmp_path, rev="2222222222bb")
    h = _gated_hub(["model.safetensors"], sha="2222222222bb")
    assert "not the newest" not in pf.check_hub(h, None, local=True, local_path=snap).detail


def test_a_listed_pytorch_bin_weight_that_is_missing_blocks_a_gated_model(tmp_path):
    snap = _audio_snap(tmp_path)
    h = _gated_hub(["model.safetensors", "pytorch_model.bin"])
    assert pf.check_hub(h, None, local=True, local_path=snap).reason == "credentials-needed"
    (snap / "pytorch_model.bin").write_text("w")
    assert pf.check_hub(h, None, local=True, local_path=snap).status == "warn"


def test_a_weight_index_in_a_component_directory_with_a_missing_shard_makes_it_not_local(tmp_path):
    snap = _audio_snap(tmp_path)
    (snap / "t5" / "model.safetensors.index.json").write_text('{"weight_map": {"a": "part-2.safetensors"}}')
    assert real(tmp_path).local_snapshot("stabilityai/audio") is None


def test_the_newest_complete_snapshot_is_used(tmp_path):
    old = _audio_snap(tmp_path, rev="aaaa")
    new = _audio_snap(tmp_path, rev="zzzz")
    assert real(tmp_path).local_snapshot("stabilityai/audio") == new != old
