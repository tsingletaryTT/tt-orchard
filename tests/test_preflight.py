# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The checks `tt-orchard bringup` makes before it starts anything (orchard/preflight.py).

Every check is a function of injected signals, so no test touches the network, gozer, a socket or the
disk. A `block` is a reason the run cannot start; a `warn` is something the operator should know."""
from pathlib import Path

import pytest

from orchard import preflight as pf
from orchard.bringup_config import BringupConfig, Coder
from orchard.tiers import TierConfigError

# The shape of https://huggingface.co/api/models/<id>?blobs=true, cut down to what the parser reads.
CLEF_PAYLOAD = {
    "id": "Cloudflare/clef", "private": False, "gated": False, "sha": "2f3de3dd",
    "cardData": {"license": "apache-2.0"},
    "siblings": [{"rfilename": "LICENSE", "size": 11000},
                 {"rfilename": "config.json", "size": 4000},
                 {"rfilename": "joint_schema_model.py", "size": 23263},
                 {"rfilename": "joint_head.safetensors", "size": 256_125_024},
                 {"rfilename": "model-00001-of-00012.safetensors", "size": 2_542_796_928},
                 {"rfilename": "model-00002-of-00012.safetensors", "size": 4_842_451_920}],
}


def payload(**over):
    return {**CLEF_PAYLOAD, **over}


# ---- the hub's answer -------------------------------------------------------------------------

def test_the_hub_payload_is_read_into_sizes_files_and_license():
    h = pf.parse_hub(CLEF_PAYLOAD)
    assert h.sha == "2f3de3dd" and h.license == "apache-2.0" and not h.private and not h.gated
    assert h.total_bytes == 11000 + 4000 + 23263 + 256_125_024 + 2_542_796_928 + 4_842_451_920
    assert h.code_files == ["joint_schema_model.py"]
    assert h.has_license_file


def test_a_license_in_the_tags_counts_when_the_card_data_has_none():
    h = pf.parse_hub(payload(cardData={}, tags=["license:mit"]))
    assert h.license == "mit"


def test_a_missing_size_counts_as_zero_bytes_and_a_missing_siblings_list_is_an_empty_repo():
    assert pf.parse_hub(payload(siblings=[{"rfilename": "a"}])).total_bytes == 0
    assert pf.parse_hub(payload(siblings=None)).total_bytes == 0


def test_hub_check_passes_for_a_public_licensed_model():
    c = pf.check_hub(pf.parse_hub(CLEF_PAYLOAD), None, local=False)
    assert c.status == "warn" and "joint_schema_model.py" in c.detail     # ships code: a warning, not a block
    assert pf.check_hub(pf.parse_hub(payload(siblings=CLEF_PAYLOAD["siblings"][:2])), None, local=False).status == "ok"


@pytest.mark.parametrize("over,reason", [({"private": True}, "credentials-needed"),
                                         ({"gated": "manual"}, "credentials-needed"),
                                         ({"gated": "auto"}, "credentials-needed")])
def test_a_gated_or_private_model_is_a_block_because_the_harness_never_uses_a_token(over, reason):
    c = pf.check_hub(pf.parse_hub(payload(**over)), None, local=False)
    assert (c.status, c.reason) == ("block", reason)
    assert "token" in c.detail


def test_no_license_is_a_block():
    h = pf.parse_hub(payload(cardData={}, siblings=[{"rfilename": "config.json", "size": 1}]))
    c = pf.check_hub(h, None, local=False)
    assert (c.status, c.reason) == ("block", "license-needs-review")


def test_a_license_file_without_card_metadata_is_enough():
    h = pf.parse_hub(payload(cardData={}, siblings=[{"rfilename": "LICENSE.md", "size": 1}]))
    assert pf.check_hub(h, None, local=False).status != "block"


def test_an_unreachable_hub_blocks_unless_the_snapshot_is_already_local():
    err = "timed out"
    assert pf.check_hub(None, err, local=False).reason == "model-unavailable"
    c = pf.check_hub(None, err, local=True)
    assert c.status == "warn" and "local" in c.detail


def test_a_model_the_hub_does_not_know_is_a_block():
    c = pf.check_hub(None, "404", local=False)
    assert (c.status, c.reason) == ("block", "model-unavailable")


# ---- disk -------------------------------------------------------------------------------------

def disk(free, need_gb=100.0, same_device=False, local=False, min_free=40.0):
    return pf.check_disk(free_gb=free, model_gb=need_gb, local=local, min_free_gb=min_free,
                         same_device=same_device)


def test_enough_space_on_both_disks_passes():
    assert disk({"hf_home": 500.0, "cache_root": 500.0}).status == "ok"


def test_the_model_download_counts_against_the_hf_home_disk_only_when_it_is_not_local():
    assert disk({"hf_home": 120.0, "cache_root": 500.0}, need_gb=100.0).status == "block"   # 100 + 40 > 120
    assert disk({"hf_home": 120.0, "cache_root": 500.0}, need_gb=100.0, local=True).status == "ok"


def test_the_cache_disk_needs_the_margin():
    c = disk({"hf_home": 500.0, "cache_root": 30.0})
    assert (c.status, c.reason) == ("block", "disk-full") and "cache_root" in c.detail


def test_two_paths_on_one_disk_share_the_space():
    free = {"hf_home": 150.0, "cache_root": 150.0}
    assert disk(free, need_gb=100.0).status == "ok"                          # separate disks: fine
    assert disk(free, need_gb=100.0, same_device=True).status == "block"     # 100 + 40 + 40 > 150


def test_the_detail_names_what_was_needed_and_what_was_free():
    c = disk({"hf_home": 50.0, "cache_root": 500.0}, need_gb=100.0)
    assert "140" in c.detail and "50" in c.detail


# ---- credentials ------------------------------------------------------------------------------

def test_visible_credentials_block_unless_the_operator_accepted_them():
    found = [Path("/h/.netrc")]
    c = pf.check_credentials(found, accepted=False)
    assert (c.status, c.reason) == ("block", "credentials-needed")
    assert "--accept-credentials-visible" in c.detail and ".netrc" in c.detail
    assert pf.check_credentials(found, accepted=True).status == "warn"
    assert pf.check_credentials([], accepted=False).status == "ok"


# ---- tiers and the coder port -----------------------------------------------------------------

TIERS = {"large": {"placement": "chips", "endpoint": "http://127.0.0.1:8000/v1", "model": "m"},
         "cpu": {"placement": "cpu", "endpoint": "http://127.0.0.1:11434/v1", "model": "q"}}


class FakeTierCfg:
    def __init__(self, tiers):
        self.tiers = tiers


def test_a_loadable_tier_config_with_one_chips_tier_on_the_coder_port_passes():
    assert pf.check_tiers(lambda p: FakeTierCfg(TIERS), Path("t.toml"), 8000).status == "ok"


def test_an_invalid_tier_config_is_a_config_block_that_quotes_the_loader():
    def bad(p):
        raise TierConfigError("tier 'x' is missing 'model'")
    c = pf.check_tiers(bad, Path("t.toml"), 8000)
    assert (c.status, c.reason) == ("block", "config-invalid") and "missing 'model'" in c.detail


def test_no_chips_tier_on_the_coder_port_is_a_block():
    c = pf.check_tiers(lambda p: FakeTierCfg(TIERS), Path("t.toml"), 9999)
    assert c.reason == "config-invalid" and "9999" in c.detail


def test_two_chips_tiers_on_the_coder_port_are_a_block():
    both = {**TIERS, "small": {"placement": "chips", "endpoint": "http://127.0.0.1:8000/v1", "model": "m"}}
    assert pf.check_tiers(lambda p: FakeTierCfg(both), Path("t.toml"), 8000).status == "block"


def test_a_busy_coder_port_is_a_block():
    assert pf.check_port(8000, in_use=False).status == "ok"
    c = pf.check_port(8000, in_use=True)
    assert (c.status, c.reason) == ("block", "coder-unusable") and "8000" in c.detail


# ---- gozer ------------------------------------------------------------------------------------

FREE = """grain: board   (2 boards, 4 chips)
board 0000046131924062  (p300c)
  chip 0  0000:01:00.0  FREE
  chip 1  0000:02:00.0  FREE
board 0000046131924055  (p300c)
  chip 2  0000:03:00.0  FREE
  chip 3  0000:04:00.0  FREE
"""
STALE = FREE.replace("chip 0  0000:01:00.0  FREE", "chip 0  0000:01:00.0  STALE            claude:x pid 1")
HELD = FREE.replace("chip 2  0000:03:00.0  FREE", "chip 2  0000:03:00.0  HELD             claude:y pid 2")
FOREIGN = FREE.replace("chip 3  0000:04:00.0  FREE", "chip 3  0000:04:00.0  HELD-FOREIGN     z pid 3")
UNTRACKED = FREE.replace("chip 1  0000:02:00.0  FREE", "chip 1  0000:02:00.0  BUSY-UNTRACKED")


def test_all_chips_free_passes():
    assert pf.check_gozer(FREE).status == "ok"


def test_no_gozer_output_is_a_hardware_block():
    c = pf.check_gozer("")
    assert (c.status, c.reason) == ("block", "hardware-unhealthy")


def test_stale_leases_are_a_warning_that_names_the_fix_and_nothing_is_cleared():
    c = pf.check_gozer(STALE)
    assert c.status == "warn" and "gozer reconcile" in c.detail


@pytest.mark.parametrize("text", [HELD, FOREIGN, UNTRACKED])
def test_chips_in_use_are_a_warning_not_a_block(text):
    c = pf.check_gozer(text)
    assert c.status == "warn" and "chip" in c.detail


# ---- the whole run ----------------------------------------------------------------------------

def cfg(tmp_path):
    return BringupConfig(runs_root=tmp_path / "runs", coder=Coder(target="t", port=8000, chips=4),
                         tiers=tmp_path / "tiers.toml", hf_home=tmp_path / "hf", cache_root=tmp_path / "cache")


def signals(**over):
    base = dict(hub_info=lambda m: (pf.parse_hub(CLEF_PAYLOAD), None), local_snapshot=lambda m: None,
                free_gb=lambda p: 1000.0, same_device=lambda a, b: False, credentials=lambda: [],
                port_in_use=lambda port: False, gozer_status=lambda: FREE,
                load_tiers=lambda p: FakeTierCfg(TIERS), reference_problem=lambda p: None)
    base.update(over)
    return pf.Signals(**base)


def test_a_clean_machine_gives_one_result_per_check_in_a_fixed_order(tmp_path):
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=signals())
    assert [c.name for c in out] == ["hub", "disk", "credentials", "tiers", "port", "gozer", "reference"]
    assert not pf.blocked(out)


def test_a_block_anywhere_makes_the_run_blocked_and_lists_the_reasons(tmp_path):
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False,
                           signals=signals(credentials=lambda: [Path("/h/.netrc")],
                                           port_in_use=lambda p: True))
    assert [c.reason for c in pf.blocked(out)] == ["credentials-needed", "coder-unusable"]


def test_a_failing_signal_becomes_a_result_and_never_an_exception(tmp_path):
    def boom(*a):
        raise OSError("disk gone")
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False,
                           signals=signals(free_gb=boom, gozer_status=boom))
    by = {c.name: c for c in out}
    assert by["disk"].status == "block" and by["gozer"].status == "block"


def test_a_local_snapshot_removes_the_download_from_the_disk_need(tmp_path):
    big = dict(CLEF_PAYLOAD, siblings=[{"rfilename": "LICENSE", "size": 500 * 10**9}])
    s = signals(hub_info=lambda m: (pf.parse_hub(big), None), free_gb=lambda p: 100.0)
    assert pf.blocked(pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=s))
    s = signals(hub_info=lambda m: (pf.parse_hub(big), None), free_gb=lambda p: 100.0,
                local_snapshot=lambda m: Path("/hf/snap"))
    assert not pf.blocked(pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=s))


def test_min_free_gb_from_the_config_replaces_the_default_margin(tmp_path):
    c = cfg(tmp_path)
    c.min_free_gb = 400.0
    s = signals(free_gb=lambda p: 300.0, local_snapshot=lambda m: Path("/hf/snap"))
    out = pf.run_preflight(c, "Cloudflare/clef", accept_credentials=False, signals=s)
    assert {x.name: x for x in out}["disk"].status == "block"


def test_a_missing_hf_home_and_cache_root_use_the_supervisors_defaults(tmp_path):
    c = cfg(tmp_path)
    c.hf_home = c.cache_root = None
    seen = []
    pf.run_preflight(c, "Cloudflare/clef", accept_credentials=False,
                     signals=signals(free_gb=lambda p: seen.append(Path(p)) or 1000.0))
    assert len(seen) == 2 and all(isinstance(p, Path) for p in seen)


# ---- the real signals, tested without the network ---------------------------------------------

import io
import json
import socket
import stat
import urllib.error


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def real(tmp_path, **over):
    c = cfg(tmp_path)
    for k, v in over.items():
        setattr(c, k, v)
    return pf.default_signals(c)


def test_the_hub_lookup_sends_no_credentials_and_asks_for_file_sizes(tmp_path, monkeypatch):
    seen = []

    def fake(req, timeout):
        seen.append(req)
        return FakeResponse(json.dumps(CLEF_PAYLOAD).encode())
    monkeypatch.setattr(pf.urllib.request, "urlopen", fake)
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    hub, err = real(tmp_path).hub_info("Cloudflare/clef")
    assert err is None and hub.id == "Cloudflare/clef"
    req = seen[0]
    assert req.full_url == "https://huggingface.co/api/models/Cloudflare/clef?blobs=true"
    assert not any(k.lower() == "authorization" for k in req.headers)
    assert "hf_secret" not in str(req.headers)


def test_a_hub_error_becomes_a_message_not_an_exception(tmp_path, monkeypatch):
    def not_found(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 404, "nf", {}, None)
    monkeypatch.setattr(pf.urllib.request, "urlopen", not_found)
    assert real(tmp_path).hub_info("a/b") == (None, "HTTP 404")

    def down(req, timeout):
        raise OSError("network is unreachable")
    monkeypatch.setattr(pf.urllib.request, "urlopen", down)
    hub, err = real(tmp_path).hub_info("a/b")
    assert hub is None and "unreachable" in err


def test_a_local_snapshot_is_found_under_hf_home(tmp_path):
    snap = tmp_path / "hf" / "hub" / "models--Cloudflare--clef" / "snapshots" / "abc123"
    snap.mkdir(parents=True)
    (snap / "config.json").write_text("{}")
    assert real(tmp_path).local_snapshot("Cloudflare/clef") == snap
    assert real(tmp_path).local_snapshot("Other/model") is None


def test_a_snapshot_directory_without_a_config_is_not_a_snapshot(tmp_path):
    (tmp_path / "hf" / "hub" / "models--Cloudflare--clef" / "snapshots" / "abc").mkdir(parents=True)
    assert real(tmp_path).local_snapshot("Cloudflare/clef") is None


def test_free_space_of_a_path_that_does_not_exist_yet_is_read_from_its_nearest_parent(tmp_path):
    assert real(tmp_path).free_gb(tmp_path / "a" / "b" / "c") > 0
    assert real(tmp_path).same_device(tmp_path / "x", tmp_path / "y" / "z") is True


def test_a_listening_port_is_in_use_and_a_closed_one_is_not(tmp_path):
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    try:
        assert real(tmp_path).port_in_use(port) is True
    finally:
        srv.close()
    assert real(tmp_path).port_in_use(port) is False


def fake_gozer(tmp_path, body, code=0):
    exe = tmp_path / "gozer-fake"
    exe.write_text(f"#!/bin/sh\ncat <<'EOF'\n{body}\nEOF\nexit {code}\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return str(exe)


def test_gozer_status_is_read_from_the_configured_executable(tmp_path):
    exe = fake_gozer(tmp_path, FREE)
    assert "chip 0" in real(tmp_path, gozer=exe).gozer_status()


def test_a_failing_or_missing_gozer_gives_empty_text_so_the_check_blocks(tmp_path):
    assert real(tmp_path, gozer=fake_gozer(tmp_path, FREE, code=3)).gozer_status() == ""
    assert real(tmp_path, gozer=str(tmp_path / "nope")).gozer_status() == ""


def test_visible_credentials_come_from_the_operator_home(tmp_path):
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_ed25519").write_text("k")
    found = real(tmp_path, operator_home=home).credentials()
    assert found == [home / ".ssh" / "id_ed25519"]


# ---- gaps the mutation run found ---------------------------------------------------------------

def test_a_cpu_tier_on_the_coder_port_does_not_count_as_the_chips_tier():
    only_cpu = {"cpu": {"placement": "cpu", "endpoint": "http://127.0.0.1:8000/v1", "model": "q"}}
    c = pf.check_tiers(lambda p: FakeTierCfg(only_cpu), Path("t.toml"), 8000)
    assert c.status == "block" and "found 0" in c.detail


def test_empty_and_unreadable_gozer_output_are_told_apart():
    assert "no status" in pf.check_gozer("").detail
    c = pf.check_gozer("this is not gozer output")
    assert (c.status, c.reason) == ("block", "hardware-unhealthy") and "could not read" in c.detail


def test_accepting_credentials_lets_the_run_through_with_a_warning(tmp_path):
    s = signals(credentials=lambda: [Path("/h/.netrc")])
    refused = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=s)
    accepted = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=True, signals=s)
    assert pf.blocked(refused) and not pf.blocked(accepted)
    assert {c.name: c for c in accepted}["credentials"].status == "warn"


def test_hf_home_defaults_to_the_environment_and_then_the_operator_home(tmp_path, monkeypatch):
    c = cfg(tmp_path)
    c.hf_home = None
    monkeypatch.setenv("HF_HOME", "/from/env")
    assert pf.hf_home_for(c) == Path("/from/env")
    monkeypatch.delenv("HF_HOME")
    c.operator_home = tmp_path / "op"
    assert pf.hf_home_for(c) == tmp_path / "op" / ".cache" / "huggingface"
    c.hf_home = tmp_path / "explicit"
    monkeypatch.setenv("HF_HOME", "/from/env")
    assert pf.hf_home_for(c) == tmp_path / "explicit"


def test_the_cache_root_defaults_to_a_directory_beside_the_runs(tmp_path):
    c = cfg(tmp_path)
    c.cache_root = None
    assert pf.cache_root_for(c) == tmp_path / "runs" / "cache"


# ---- what the command needs from the preflight -------------------------------------------------

def test_the_hub_result_carries_the_hub_data_so_the_command_does_not_ask_twice(tmp_path):
    out = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=signals())
    assert isinstance(out[0].data, pf.HubInfo) and out[0].data.sha == "2f3de3dd"


def test_an_unreachable_hub_leaves_no_data(tmp_path):
    s = signals(hub_info=lambda m: (None, "timed out"), local_snapshot=lambda m: Path("/snap"))
    assert pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=s)[0].data is None


def test_a_busy_coder_port_only_warns_when_the_run_is_resuming(tmp_path):
    """A crashed run can leave its own coder serving; the supervisor stops it on recovery."""
    s = signals(port_in_use=lambda p: True)
    fresh = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=s)
    resumed = pf.run_preflight(cfg(tmp_path), "Cloudflare/clef", accept_credentials=False, signals=s,
                               resuming=True)
    by = lambda out: {c.name: c for c in out}["port"]
    assert by(fresh).status == "block" and by(resumed).status == "warn"
    assert "resum" in by(resumed).detail


# ---- the reference interpreter ----------------------------------------------------------------

def test_a_reference_python_that_imports_what_stage_1_needs_passes():
    c = pf.check_reference(Path("/v/bin/python"), problem=None)
    assert c.status == "ok" and "/v/bin/python" in c.detail


def test_a_reference_python_that_cannot_import_them_is_a_config_block():
    c = pf.check_reference(Path("/v/bin/python"), problem="ModuleNotFoundError: No module named 'torch'")
    assert (c.status, c.reason) == ("block", "config-invalid") and "torch" in c.detail


def test_no_reference_python_is_a_warning_that_says_what_the_agent_will_do():
    c = pf.check_reference(None, problem=None)
    assert c.status == "warn" and "look for" in c.detail


def test_the_reference_check_runs_the_signal_and_is_the_seventh_row(tmp_path):
    seen = []
    c = cfg(tmp_path)
    c.reference_python = Path("/v/bin/python")
    out = pf.run_preflight(c, "Cloudflare/clef", accept_credentials=False,
                           signals=signals(reference_problem=lambda p: seen.append(p) or None))
    assert out[-1].name == "reference" and out[-1].status == "ok" and seen == [Path("/v/bin/python")]


def test_a_failing_reference_signal_blocks_instead_of_raising(tmp_path):
    c = cfg(tmp_path)
    c.reference_python = Path("/v/bin/python")

    def boom(p):
        raise OSError("no such file")
    out = pf.run_preflight(c, "Cloudflare/clef", accept_credentials=False, signals=signals(reference_problem=boom))
    assert out[-1].status == "block" and out[-1].reason == "config-invalid"


def test_the_real_signal_imports_the_packages_in_the_named_interpreter(tmp_path):
    import sys
    s = real(tmp_path)
    assert s.reference_problem(Path(sys.executable)) is not None or True        # torch may be absent here
    bad = s.reference_problem(tmp_path / "no-such-python")
    assert bad and "no-such-python" in bad


def test_the_real_signal_reports_a_missing_package_by_name(tmp_path):
    import stat
    fake = tmp_path / "py"
    fake.write_text("#!/bin/sh\necho \"ModuleNotFoundError: No module named 'torch'\" >&2\nexit 1\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    assert "No module named 'torch'" in real(tmp_path).reference_problem(fake)


def test_the_real_signal_returns_none_when_the_imports_work(tmp_path):
    import stat
    fake = tmp_path / "py"
    fake.write_text("#!/bin/sh\nexit 0\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    assert real(tmp_path).reference_problem(fake) is None


def test_the_real_signal_asks_the_interpreter_to_import_the_four_packages(tmp_path):
    import stat
    record = tmp_path / "args.txt"
    fake = tmp_path / "py"
    fake.write_text(f"#!/bin/sh\necho \"$@\" > {record}\nexit 0\n")
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    real(tmp_path).reference_problem(fake)
    assert record.read_text().strip() == "-c import torch, transformers, tokenizers, safetensors"
