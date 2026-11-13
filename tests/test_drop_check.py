# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The release-day check (orchard/drop_check.py).

`tt-orchard drop-check` reads only a release's config, tokenizer and weight index and says which route a bring-up takes: the
Qwen4 package (covered), the Qwen3.8 bundle (same architecture as Qwen3.8), or the full-port fallback (not covered). It never
downloads weights, never sends a token, and refuses any organisation other than the allowed one. The network is injected, so no
test touches it."""
import copy
import http.client
import io
import json
import urllib.error
from pathlib import Path

import pytest

from orchard import drop_check as dc

FIX = Path(__file__).parent / "fixtures" / "model-configs"
REV = "0123456789abcdef0123456789abcdef01234567"


def load(name):
    return json.loads((FIX / name).read_text())


def qwen4_dense():
    """The provisional dense Qwen4 config with the keys a buildable config needs (indexer, rope, eps, act, gate, tying)."""
    cfg = load("qwen4-dense-provisional-buildable.json")
    cfg.pop("_source", None)
    return cfg


def caps():
    return copy.deepcopy(dc.EXAMPLE_CAPABILITIES)


def files_for(cfg, **extra):
    out = {"config.json": json.dumps(cfg).encode()}
    out.update(extra)
    return out


def verdict(cfg, **kw):
    return dc.classify("Qwen/Qwen4-27B", REV, files_for(cfg), **kw)


# ---- repo ids and organisations -------------------------------------------------------------------------------------

@pytest.mark.parametrize("repo", ["Qwen/Qwen4-27B", "Qwen/Qwen4-27B-FP8", "Qwen/Qwen3.8-27B"])
def test_allowed_repo_ids(repo):
    assert dc.repo_problem(repo) is None


@pytest.mark.parametrize("repo", ["Esteban143/Qwen4-32B", "qwen/Qwen4-27B", "QwenLM/Qwen4-27B", "Qwen", "Qwen/", "/Qwen4",
                                  "Qwen/../x", "Qwen/a b", "Qwen/x/y", "Qwen/x\x00", "", None, 5, ["Qwen/Qwen4-27B"],
                                  "Qwen/" + "a" * 200, "Qwen/.hidden", "Qwen/x?y=1", "Qwen/x#y"])
def test_refused_repo_ids(repo):
    assert dc.repo_problem(repo)


@pytest.mark.parametrize("repo", ["Qwen/a..b", "Qwen/x.."])
def test_double_dot_names_are_refused(repo):
    """The Hub forbids '..' in repository names, so a name with it is not a real release."""
    assert dc.repo_problem(repo)


def test_classify_refuses_other_org_without_reading_files():
    out = dc.classify("Esteban143/Qwen4-32B", REV, {"config.json": b"{not json"})
    assert out["verdict"] == "refused" and out["route"] is None


@pytest.mark.parametrize("rev", ["main", "0123", "G" * 40, None, 7, "0123456789abcdef0123456789abcdef0123456\x00"])
def test_bad_revision_refused(rev):
    assert dc.classify("Qwen/Qwen4-27B", rev, files_for(qwen4_dense()))["verdict"] == "refused"


# ---- routes ---------------------------------------------------------------------------------------------------------

def test_same_architecture_as_qwen38_is_covered_by_the_qwen38_bundle():
    out = verdict(load("qwen3.8-27b.json"))
    assert out["verdict"] == "covered" and out["route"] == "qwen3.8-bundle"
    assert "--base" in out["command"] and dc.QWEN38_BUNDLE in out["command"]


def test_qwen35_type_with_other_widths_is_not_covered():
    cfg = load("qwen3.8-27b.json")
    (cfg.get("text_config") or cfg)["intermediate_size"] = 14336
    out = verdict(cfg)
    assert out["verdict"] == "not-covered" and out["route"] == "full-port"
    assert any("intermediate_size" in r for r in out["reasons"])


def test_flash_next_shape_is_not_covered_and_names_experts_and_ngram():
    out = verdict(load("qwen3.8-flash-next.json"), capabilities=caps())
    assert out["verdict"] == "not-covered" and out["route"] == "full-port"
    text = " ".join(out["reasons"])
    assert "mixture-of-experts" in text and "n-gram" in text
    assert "--full-port" in out["command"]


def test_dense_qwen4_without_capabilities_is_not_covered_yet():
    out = verdict(qwen4_dense(), capabilities=None)
    assert out["verdict"] == "not-covered" and out["route"] == "full-port"
    assert any("declares no capabilities" in r for r in out["reasons"])


def test_dense_qwen4_with_matching_capabilities_is_covered_with_notes():
    out = verdict(qwen4_dense(), capabilities=caps())
    assert out["verdict"] == "covered-with-notes", out["reasons"]
    assert out["route"] == "qwen4-package"
    assert any("2048" in n for n in out["notes"])            # indexer present: context capped where it is exact
    assert dc.QWEN4_BUNDLE in out["command"]


def test_capability_rule_violation_names_the_key():
    cfg = qwen4_dense()
    cfg["text_config"]["hc_lowrank"] = 512
    c = caps()
    c["keys"]["hc_lowrank"] = {"one_of": [320, 640]}
    out = verdict(cfg, capabilities=c)
    assert out["verdict"] == "not-covered" and any("hc_lowrank" in r for r in out["reasons"])


@pytest.mark.parametrize("key", dc.REQUIRED_QWEN4_KEYS)
def test_each_missing_required_key_is_named(key):
    cfg = qwen4_dense()
    tc = cfg["text_config"]
    if key in tc:
        del tc[key]
    elif key in cfg:
        del cfg[key]
    out = verdict(cfg, capabilities=caps())
    assert out["verdict"] == "not-covered", key
    assert any(key in r for r in out["reasons"]), (key, out["reasons"])


def test_missing_rope_theta_is_named():
    cfg = qwen4_dense()
    cfg["text_config"]["rope_parameters"].pop("rope_theta")
    out = verdict(cfg, capabilities=caps())
    assert out["verdict"] == "not-covered" and any("rope_theta" in r for r in out["reasons"])


def test_partial_indexer_keys_are_refused():
    cfg = qwen4_dense()
    cfg["text_config"].pop("indexer_budget")
    out = verdict(cfg, capabilities=caps())
    assert out["verdict"] == "not-covered"
    assert any("indexer keys are incomplete" in r and "indexer_budget" in r for r in out["reasons"]), out["reasons"]


def bare_caps():
    """Capabilities with no per-key rules, so the structural checks are seen on their own."""
    c = caps()
    c["keys"] = {}
    return c


@pytest.mark.parametrize("field,value,words", [("num_attention_heads", 26, "does not divide over 4 chips"),
                                               ("linear_num_value_heads", 50, "does not divide over 4 chips"),
                                               ("num_key_value_heads", 3, "nor replicates"),
                                               ("intermediate_size", 14400, "whole 32-wide tiles on 4 chips")])
def test_split_checks_without_capability_rules(field, value, words):
    cfg = qwen4_dense()
    cfg["text_config"][field] = value
    out = verdict(cfg, capabilities=bare_caps())
    assert out["verdict"] == "not-covered"
    assert any(field in r and words in r for r in out["reasons"]), out["reasons"]


@pytest.mark.parametrize("field", ["hc_count", "hc_lowrank", "intermediate_size", "num_key_value_heads"])
def test_a_boolean_is_not_a_count(field):
    cfg = qwen4_dense()
    cfg["text_config"][field] = True
    out = verdict(cfg, capabilities=bare_caps())
    assert any(r.startswith(f"{field} is true; it must be a whole number") for r in out["reasons"]), out["reasons"]


def test_no_indexer_means_no_context_cap():
    cfg = qwen4_dense()
    for k in dc.INDEXER_KEYS:
        cfg["text_config"].pop(k)
    out = verdict(cfg, capabilities=caps())
    assert out["verdict"] in ("covered", "covered-with-notes"), out["reasons"]
    assert not any("2048" in n for n in out["notes"])


@pytest.mark.parametrize("field,value", [("num_attention_heads", 26), ("linear_num_value_heads", 50),
                                         ("num_key_value_heads", 3), ("intermediate_size", 14400)])
def test_shapes_that_do_not_split_over_two_and_four_chips(field, value):
    cfg = qwen4_dense()
    cfg["text_config"][field] = value
    out = verdict(cfg, capabilities=caps())
    assert out["verdict"] == "not-covered" and any(field in r for r in out["reasons"])


def test_dense_rule_counts_experts_only_on_scheduled_layers():
    cfg = qwen4_dense()
    tc = cfg["text_config"]
    tc.update(num_experts=8, decoder_sparse_step=1, mlp_only_layers=list(range(tc["num_hidden_layers"])))
    assert verdict(cfg, capabilities=caps())["route"] == "qwen4-package"      # every layer listed as dense
    tc["mlp_only_layers"] = [0]
    assert verdict(cfg, capabilities=caps())["route"] == "full-port"


def test_new_model_type_needs_a_reference_interpreter_with_the_class():
    cfg = qwen4_dense()
    cfg["model_type"], cfg["architectures"] = "qwen4", ["Qwen4ForConditionalGeneration"]
    c = caps()
    c["model_types"].append("qwen4")
    c["architectures"].append("Qwen4ForConditionalGeneration")
    missing = verdict(cfg, capabilities=c, class_probe=lambda classes: {k: False for k in classes})
    assert missing["verdict"] == "needs-operator"
    assert any("scripts/reference_venv.py" in r for r in missing["reasons"])
    present = verdict(cfg, capabilities=c, class_probe=lambda classes: {k: True for k in classes})
    assert present["route"] == "qwen4-package"


def test_unknown_class_probe_is_not_a_pass():
    cfg = qwen4_dense()
    cfg["model_type"], cfg["architectures"] = "qwen4", ["Qwen4ForConditionalGeneration"]
    c = caps()
    c["model_types"].append("qwen4")
    c["architectures"].append("Qwen4ForConditionalGeneration")
    out = verdict(cfg, capabilities=c, class_probe=None)
    assert out["verdict"] == "needs-operator"


def test_unknown_model_type_is_full_port():
    cfg = {"model_type": "llama", "architectures": ["LlamaForCausalLM"], "hidden_size": 4096}
    out = verdict(cfg, capabilities=caps())
    assert out["verdict"] == "not-covered" and out["route"] == "full-port"


# ---- hostile and malformed input ------------------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, [], {}, "5120", "", "x\x00", True, -1, 0, 10 ** 20, 1.5])
@pytest.mark.parametrize("field", ["hidden_size", "num_hidden_layers", "num_key_value_heads", "intermediate_size",
                                   "hc_count", "hc_lowrank"])
def test_bad_field_values_give_a_named_reason(field, value):
    cfg = qwen4_dense()
    cfg["text_config"][field] = value
    out = verdict(cfg, capabilities=caps())
    assert out["verdict"] == "not-covered" and any(field in r for r in out["reasons"])


@pytest.mark.parametrize("raw", [b"", b"[]", b"null", b"5", b"{bad", "{}".encode() + b"\xff", b"{" * 5000])
def test_unreadable_config_is_a_reason(raw):
    out = dc.classify("Qwen/Qwen4-27B", REV, {"config.json": raw}, capabilities=caps())
    assert out["verdict"] == "not-covered" and any("config.json" in r for r in out["reasons"])


def test_missing_config_is_a_reason():
    out = dc.classify("Qwen/Qwen4-27B", REV, {}, capabilities=caps())
    assert out["verdict"] == "not-covered" and any("config.json" in r for r in out["reasons"])


@pytest.mark.parametrize("bad", [None, [], "x", {"schema": 2}, {"schema": 1, "keys": []}, {"schema": 1, "keys": {"x": 5}},
                                 {"schema": 1, "keys": {"hc_lowrank": {"one_of": "640"}}}])
def test_malformed_capabilities_file_is_not_a_pass(bad):
    out = verdict(qwen4_dense(), capabilities=bad)
    assert out["verdict"] == "not-covered" and any("capabilities" in r for r in out["reasons"])


# ---- tokenizer and chat template notes ------------------------------------------------------------------------------

def test_tokenizer_and_template_notes():
    files = files_for(qwen4_dense(), **{"tokenizer.json": b"other tokenizer", "chat_template.jinja": b"{{ messages }}"})
    out = dc.classify("Qwen/Qwen4-27B", REV, files, capabilities=caps())
    notes = " ".join(out["notes"])
    assert "tokenizer differs from Qwen3.8" in notes and "no enable_thinking" in notes


# ---- the Hub client -------------------------------------------------------------------------------------------------

class FakeResponse(io.BytesIO):
    """A response body and the URL it came from after any redirect (urllib's geturl())."""

    def __init__(self, body, url):
        super().__init__(body)
        self.url = url

    def geturl(self):
        return self.url

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeOpener:
    """Routes by URL suffix. A route's value is a body, an HTTP status, or (body, final URL) for a request the Hub redirected."""

    def __init__(self, routes):
        self.routes, self.requests = routes, []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        for suffix, body in self.routes.items():
            if req.full_url.endswith(suffix):
                if isinstance(body, int):
                    raise urllib.error.HTTPError(req.full_url, body, "x", {}, None)
                if isinstance(body, tuple):
                    return FakeResponse(*body)
                return FakeResponse(body, req.full_url)
        raise urllib.error.HTTPError(req.full_url, 404, "not found", {}, None)


def test_hub_client_sends_no_token_and_reads_files():
    op = FakeOpener({"/resolve/" + REV + "/config.json": b"{}"})
    hub = dc.HubClient(opener=op)
    assert hub.get("Qwen/Qwen4-27B", REV, "config.json", max_bytes=100) == b"{}"
    req = op.requests[0]
    assert req.full_url == f"https://huggingface.co/Qwen/Qwen4-27B/resolve/{REV}/config.json"
    assert not any(k.lower() == "authorization" for k in req.headers)


def test_hub_client_missing_file_is_none_and_oversize_raises():
    op = FakeOpener({"/config.json": b"x" * 101})
    hub = dc.HubClient(opener=op)
    assert hub.get("Qwen/Qwen4-27B", REV, "tokenizer.json", max_bytes=100) is None
    with pytest.raises(dc.TooLarge):
        hub.get("Qwen/Qwen4-27B", REV, "config.json", max_bytes=100)


@pytest.mark.parametrize("code", [401, 403, 404])
def test_hub_info_of_missing_or_gated_repo_is_none(code):
    hub = dc.HubClient(opener=FakeOpener({"?blobs=true": code}))
    assert hub.info("Qwen/Qwen4-27B") is None


def test_hub_client_refuses_bad_ids_before_any_request():
    op = FakeOpener({})
    hub = dc.HubClient(opener=op)
    with pytest.raises(ValueError):
        hub.get("Qwen/../x", REV, "config.json", max_bytes=10)
    with pytest.raises(ValueError):
        hub.get("Qwen/Qwen4-27B", "main", "config.json", max_bytes=10)
    with pytest.raises(ValueError):
        hub.get("Qwen/Qwen4-27B", REV, "../config.json", max_bytes=10)
    assert op.requests == []


@pytest.mark.parametrize("body", [b'{"sha": NaN}', b'{"\\udc00": 1}', b'{"a": ' * 70 + b'1' + b'}' * 70, b'{"a": 1, "a": 2}'],
                         ids=["nan", "lone-surrogate-key", "nested-70", "repeated-key"])
def test_a_hub_record_that_is_not_strict_json_is_refused(body):
    """NaN, a lone surrogate, deep nesting and a repeated key are refused with the record (issue 676)."""
    hub = dc.HubClient(opener=FakeOpener({"?blobs=true": body}))
    assert hub.info("Qwen/Qwen4-27B") is None


def test_a_config_with_nan_is_not_valid_json():
    raw = json.dumps(qwen4_dense())[:-1].encode() + b', "extra": NaN}'
    out = dc.classify("Qwen/Qwen4-27B", REV, {"config.json": raw}, capabilities=caps())
    assert out["verdict"] == "not-covered" and any("config.json is not valid JSON" in r for r in out["reasons"])


def test_a_chat_template_with_a_lone_surrogate_gives_no_template_note():
    files = files_for(qwen4_dense(), **{"tokenizer_config.json": b'{"chat_template": "\\ud800"}'})
    out = dc.classify("Qwen/Qwen4-27B", REV, files, capabilities=caps())
    assert not any("chat template" in n for n in out["notes"])


# ---- a Hub that does not answer (issue 666) ---------------------------------------------------------------------------

class DownOpener:
    """urlopen that fails as a network outage does."""

    def __init__(self, exc):
        self.exc = exc

    def __call__(self, req, timeout=None):
        exc = self.exc(req) if callable(self.exc) and not isinstance(self.exc, type) else self.exc
        raise exc


OUTAGES = {
    "refused": urllib.error.URLError(ConnectionRefusedError(111, "Connection refused")),
    "http-500": lambda req: urllib.error.HTTPError(req.full_url, 500, "server error", {}, None),
    "http-429": lambda req: urllib.error.HTTPError(req.full_url, 429, "too many requests", {}, None),
    "timeout": TimeoutError("timed out"),
    "reset": ConnectionResetError(104, "Connection reset by peer"),
    "incomplete-read": http.client.IncompleteRead(b"partial"),
}


@pytest.mark.parametrize("name", sorted(OUTAGES))
def test_a_hub_that_does_not_answer_is_unreachable(name):
    hub = dc.HubClient(opener=DownOpener(OUTAGES[name]))
    with pytest.raises(dc.HubUnreachable):
        hub.info("Qwen/Qwen4-27B")
    with pytest.raises(dc.HubUnreachable):
        hub.get("Qwen/Qwen4-27B", REV, "config.json", max_bytes=100)


def test_an_outage_is_not_a_missing_or_unreadable_file():
    """fetch_metadata turns an unreadable file into a reason; an outage must not become reasons that send the release down the
    full-port route."""
    assert not issubclass(dc.HubUnreachable, dc.Unreadable)
    with pytest.raises(dc.HubUnreachable):
        dc.fetch_metadata(dc.HubClient(opener=DownOpener(OUTAGES["refused"])), "Qwen/Qwen4-27B", REV)


class HalfDownHub:
    """Answers the API record, then fails on the files."""

    def __init__(self, opener=None):
        pass

    def info(self, repo, revision=None):
        return {"sha": REV, "siblings": []}

    def get(self, repo, revision, name, *, max_bytes):
        raise dc.HubUnreachable("the Hub did not answer (URLError)")


class DownHub(HalfDownHub):
    def info(self, repo, revision=None):
        raise dc.HubUnreachable("the Hub did not answer (URLError)")


@pytest.mark.parametrize("client", [DownHub, HalfDownHub], ids=["record", "files"])
def test_cli_reports_an_unreachable_hub_with_its_own_exit_code(client, monkeypatch, capsys):
    monkeypatch.setattr(dc, "HubClient", client)
    code = dc.main(["Qwen/Qwen4-27B", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == dc.EXIT_UNREACHABLE and out["verdict"] == "unreachable" and out["command"] == []
    assert any("did not answer" in r for r in out["reasons"])


# ---- the host a redirect ends on (issue 667) --------------------------------------------------------------------------

FILE = f"/resolve/{REV}/tokenizer.json"


@pytest.mark.parametrize("final", [
    "https://cdn-lfs.hf.co/repos/aa/bb/tokenizer.json?X-Amz=1",
    "https://cas-bridge.xethub.hf.co/xet-bridge-us/abc?sig=1",
    "https://hf.co/x",
    f"https://huggingface.co/Qwen/Qwen4-27B-Renamed/resolve/{REV}/tokenizer.json",
    f"https://huggingface.co/api/resolve-cache/models/Qwen/Qwen4-27B/{REV}/tokenizer.json?etag=1",
    "https://HUGGINGFACE.CO/Qwen/Qwen4-27B/resolve/main/tokenizer.json",
], ids=["cdn-lfs", "xet-bridge", "hf.co", "renamed-in-org", "resolve-cache", "upper-case-host"])
def test_a_redirect_to_the_content_hosts_or_within_an_allowed_organisation_is_read(final):
    hub = dc.HubClient(opener=FakeOpener({FILE: (b"tok", final)}))
    assert hub.get("Qwen/Qwen4-27B", REV, "tokenizer.json", max_bytes=100) == b"tok"


@pytest.mark.parametrize("final", [
    "https://example.com/tokenizer.json",
    "https://hf.co.example.com/tokenizer.json",
    "https://evilhf.co/tokenizer.json",
    "https://huggingface.co.example.com/Qwen/x/resolve/main/tokenizer.json",
    "http://cdn-lfs.hf.co/repos/aa/bb/tokenizer.json",
    "https://user:pw@cdn-lfs.hf.co/repos/aa/bb/tokenizer.json",
    "https://cdn-lfs.hf.co:8443/repos/aa/bb/tokenizer.json",
    f"https://huggingface.co/someone/Qwen4-27B/resolve/{REV}/tokenizer.json",
    f"https://huggingface.co/api/resolve-cache/models/someone/Qwen4-27B/{REV}/tokenizer.json",
    "https://huggingface.co/api/models/someone/Qwen4-27B",
    "https://huggingface.co/api/whoami",
    "https://huggingface.co/Qwen",
    f"https://huggingface.co/Qwen/%2e%2e/someone/resolve/{REV}/tokenizer.json",
    "https://spaces.huggingface.co/x",
    "file:///etc/passwd",
], ids=["other-host", "hf.co-prefix", "hf.co-suffix", "hub-prefix", "plain-http", "userinfo", "port", "other-org",
        "resolve-cache-other-org", "api-other-org", "api-no-org", "org-only", "encoded-dots", "hub-subdomain", "file"])
def test_a_redirect_that_leaves_the_hub_or_the_organisation_is_refused(final):
    hub = dc.HubClient(opener=FakeOpener({FILE: (b"tok", final), "?blobs=true": (b'{"sha": "x"}', final)}))
    with pytest.raises(dc.RedirectRefused):
        hub.get("Qwen/Qwen4-27B", REV, "tokenizer.json", max_bytes=100)
    with pytest.raises(dc.RedirectRefused):
        hub.info("Qwen/Qwen4-27B")


def test_a_redirect_of_the_api_record_within_the_organisation_is_read():
    final = "https://huggingface.co/api/models/Qwen/Qwen4-27B-Renamed?blobs=true"
    hub = dc.HubClient(opener=FakeOpener({"?blobs=true": (b'{"sha": "x"}', final)}))
    assert hub.info("Qwen/Qwen4-27B") == {"sha": "x"}


def test_a_refused_redirect_is_not_a_reason_about_the_files():
    """fetch_metadata turns an unreadable file into a reason that can send the release down the full-port route; a file served from
    somewhere else stops the check."""
    hub = dc.HubClient(opener=FakeOpener({"/config.json": (b"{}", "https://example.com/config.json")}))
    with pytest.raises(dc.RedirectRefused):
        dc.fetch_metadata(hub, "Qwen/Qwen4-27B", REV)


class RedirectedHub(HalfDownHub):
    def get(self, repo, revision, name, *, max_bytes):
        raise dc.RedirectRefused("config.json was redirected to example.com, outside the Hub and its content hosts")


class RedirectedRecordHub(HalfDownHub):
    def info(self, repo, revision=None):
        raise dc.RedirectRefused("the Hub record was redirected to /api/models/someone/x, outside the allowed organisations")


@pytest.mark.parametrize("client", [RedirectedRecordHub, RedirectedHub], ids=["record", "files"])
def test_cli_refuses_a_release_whose_request_was_redirected_away(client, monkeypatch, capsys):
    monkeypatch.setattr(dc, "HubClient", client)
    code = dc.main(["Qwen/Qwen4-27B", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == dc.EXIT_REFUSED and out["verdict"] == "refused" and out["command"] == []
    assert any("redirected" in r for r in out["reasons"])


# ---- command line ---------------------------------------------------------------------------------------------------

def write_release(tmp_path, cfg, sha=REV):
    d = tmp_path / "release"
    d.mkdir()
    (d / "config.json").write_text(json.dumps(cfg))
    (d / "info.json").write_text(json.dumps({"sha": sha, "siblings": [{"rfilename": "config.json", "size": 10}]}))
    return d


def test_cli_from_dir_exit_codes(tmp_path, capsys):
    d = write_release(tmp_path, load("qwen3.8-flash-next.json"))
    code = dc.main(["Qwen/Qwen4-27B", "--from-dir", str(d), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == dc.EXIT_NOT_COVERED and out["verdict"] == "not-covered" and out["revision"] == REV


def test_cli_refused_org_exit_code(tmp_path, capsys):
    d = write_release(tmp_path, qwen4_dense())
    assert dc.main(["Esteban143/Qwen4-32B", "--from-dir", str(d), "--json"]) == dc.EXIT_REFUSED


def test_cli_with_capabilities_file(tmp_path, capsys):
    d = write_release(tmp_path, qwen4_dense())
    cap = tmp_path / "caps.json"
    cap.write_text(json.dumps(caps()))
    code = dc.main(["Qwen/Qwen4-27B", "--from-dir", str(d), "--capabilities", str(cap), "--json"])
    assert code == dc.EXIT_COVERED, capsys.readouterr().out


def test_cli_writes_out_file(tmp_path, capsys):
    d = write_release(tmp_path, load("qwen3.8-27b.json"))
    out = tmp_path / "result.json"
    assert dc.main(["Qwen/Qwen4-27B", "--from-dir", str(d), "--out", str(out)]) == dc.EXIT_COVERED
    assert json.loads(out.read_text())["route"] == "qwen3.8-bundle"


def test_tt_orchard_cli_runs_drop_check_without_a_config(tmp_path):
    """`tt-orchard drop-check` works on a box with no bringup.toml: the class probe is then "not checked"."""
    from orchard import cli
    d = write_release(tmp_path, load("qwen3.8-27b.json"))
    buf = io.StringIO()
    code = cli.main(["drop-check", "Qwen/Qwen4-27B", "--from-dir", str(d), "--json"],
                    env={"HOME": str(tmp_path / "none")}, stdout=buf)
    assert code == dc.EXIT_COVERED
    assert json.loads(buf.getvalue())["route"] == "qwen3.8-bundle"


# ---- rule semantics shared with stage 0's capability reader (track C1) ----------------------------------------------

def test_a_dotted_key_reaches_a_nested_value():
    cfg = qwen4_dense()
    c = caps()
    c["keys"]["rope_parameters.rope_theta"] = {"equals": 10000000}
    assert verdict(cfg, capabilities=c)["route"] == "qwen4-package"
    cfg["text_config"]["rope_parameters"]["rope_theta"] = 5000000
    out = verdict(cfg, capabilities=c)
    assert out["verdict"] == "not-covered" and any("rope_parameters.rope_theta" in r for r in out["reasons"])


@pytest.mark.parametrize("key,rule,value", [("tie_word_embeddings", {"equals": False}, 0),
                                            ("hc_count", {"one_of": [1, 4]}, True),
                                            ("hc_count", {"equals": True}, 1)])
def test_rules_compare_json_values(key, rule, value):
    """true is not 1 and false is not 0, as in JSON (stage 0 compares the same way)."""
    cfg = qwen4_dense()
    cfg["text_config"][key] = value
    c = bare_caps()
    c["keys"][key] = rule
    out = verdict(cfg, capabilities=c)
    assert any(f"text config {key} is " in r and "outside the rule" in r for r in out["reasons"]), out["reasons"]


@pytest.mark.parametrize("value,why", [(float("nan"), "NaN is not allowed"), (10 ** 400, "too many digits"),
                                       (-(10 ** 400), "too many digits")], ids=["nan", "huge", "minus-huge"])
def test_a_nan_or_a_huge_integer_makes_the_config_unreadable(value, why):
    """The release's config.json is parsed strictly (safeio.parse_json_bytes): a NaN or an integer of hundreds of digits refuses the
    file, and the reason says why. (Before strict parsing, math.isfinite(10**400) in a rule check raised OverflowError.)"""
    cfg = qwen4_dense()
    cfg["text_config"]["hc_lowrank"] = value
    c = bare_caps()
    c["keys"]["hc_lowrank"] = {"min": 1, "max": 4096}
    out = verdict(cfg, capabilities=c)
    assert out["verdict"] == "not-covered"
    assert any(r.startswith("config.json is not valid JSON") and why in r for r in out["reasons"]), out["reasons"]


# ---- stage 0's capability reader decides the config side (orchard/bundle_capabilities.config_reasons, PR 659) ----------

def test_the_config_rules_are_stage_0s(monkeypatch):
    """drop-check takes the config verdict from the same function stage 0 uses, with the bundle's own weights config."""
    seen = {}

    def fake(capabilities, release_config, dev_config=None):
        seen.update(caps=capabilities, dev=dev_config)
        return ["text config sentinel_key is 1, the reason stage 0 gives"]
    monkeypatch.setattr(dc.bundle_capabilities, "config_reasons", fake)
    dev = {"model_type": "qwen4_exp"}
    out = verdict(qwen4_dense(), capabilities=caps(), dev_config=dev)
    assert out["verdict"] == "not-covered" and "text config sentinel_key is 1, the reason stage 0 gives" in out["reasons"]
    assert seen["dev"] is dev and seen["caps"]["schema"] == 1


def dev_of(cfg, layers=20):
    d = json.loads(json.dumps(cfg))
    d["text_config"]["num_hidden_layers"] = layers
    d["text_config"]["layer_types"] = d["text_config"]["layer_types"][:layers]
    return d


def test_an_undeclared_difference_from_the_dev_weights_is_not_covered():
    cfg = qwen4_dense()
    out = verdict(cfg, capabilities=caps(), dev_config=dev_of(cfg))
    assert out["verdict"] == "not-covered"
    assert any("num_hidden_layers differs: 64 vs 20" in r for r in out["reasons"]), out["reasons"]


def test_a_declared_layer_count_is_covered():
    cfg = qwen4_dense()
    c = caps()
    c["keys"]["num_hidden_layers"] = {"min": 4, "max": 128}
    c["layers"] = {"pattern": r"^model\.layers\.(?P<layer>\d+)\.", "count_key": "num_hidden_layers", "types_key": "layer_types",
                   "type_values": ["linear_attention", "full_attention", "indexed_attention"]}
    out = verdict(cfg, capabilities=c, dev_config=dev_of(cfg))
    assert out["route"] == "qwen4-package", out["reasons"]


def test_without_the_dev_config_the_differences_are_left_to_stage_0():
    out = verdict(qwen4_dense(), capabilities=caps(), dev_config=None)
    assert out["route"] == "qwen4-package"
    assert any("not checked here" in n for n in out["notes"]), out["notes"]


def lay_out_bundle(tmp_path, caps_obj=None, rev=REV, with_snapshot=True, dev=None):
    bundle = tmp_path / "models" / "mando2222" / "qwen4-27b-dense-v6-thin"
    bundle.mkdir(parents=True)
    if caps_obj is not None:
        (bundle / "orchard-capabilities.json").write_text(json.dumps(caps_obj))
    (bundle / "tt_kernel_manifest.json").write_text(json.dumps({"schema_version": "6", "weights": {
        "repo_id": "mando2222/qwen4-27b-franken-k20", "revision": rev}}))
    hf = tmp_path / "hf"
    if with_snapshot:
        snap = hf / "hub" / "models--mando2222--qwen4-27b-franken-k20" / "snapshots" / rev
        snap.mkdir(parents=True)
        (snap / "config.json").write_text(json.dumps(dev or {"model_type": "qwen4_exp"}))
    return bundle, hf


def test_bundle_inputs_reads_the_file_and_the_dev_snapshot(tmp_path):
    cfg = qwen4_dense()
    bundle, hf = lay_out_bundle(tmp_path, caps_obj=caps(), dev=dev_of(cfg))
    got_caps, dev, notes = dc.bundle_inputs(bundle, hf)
    assert got_caps["schema"] == 1 and dev["text_config"]["num_hidden_layers"] == 20 and notes == []


def test_bundle_inputs_without_a_file_or_a_snapshot(tmp_path):
    bundle, hf = lay_out_bundle(tmp_path, caps_obj=None, with_snapshot=False)
    got_caps, dev, notes = dc.bundle_inputs(bundle, hf)
    assert got_caps is None and dev is None
    assert any("mando2222/qwen4-27b-franken-k20" in n for n in notes)


@pytest.mark.parametrize("manifest", ["{bad", "[]", json.dumps({"weights": {"repo_id": "../x", "revision": REV}}),
                                      json.dumps({"weights": {"repo_id": "a/b", "revision": "main"}}),
                                      json.dumps({"weights": {"repo_id": "mando2222/qwen4-27b-franken-k20", "revision": "../../.."}})])
def test_bundle_inputs_with_a_bad_manifest_finds_no_dev_config(tmp_path, manifest):
    bundle, hf = lay_out_bundle(tmp_path, caps_obj=caps())
    (bundle / "tt_kernel_manifest.json").write_text(manifest)
    got_caps, dev, notes = dc.bundle_inputs(bundle, hf)
    assert got_caps["schema"] == 1 and dev is None
    assert any("cannot be read" in n or "names no weights repository at a 40-character revision" in n for n in notes), notes


def test_cli_reads_the_bundle_directory(tmp_path, capsys):
    cfg = qwen4_dense()
    d = write_release(tmp_path, cfg)
    bundle, hf = lay_out_bundle(tmp_path, caps_obj=caps(), dev=dev_of(cfg))
    code = dc.main(["Qwen/Qwen4-27B", "--from-dir", str(d), "--bundle-dir", str(bundle), "--hf-home", str(hf), "--json"])
    out = json.loads(capsys.readouterr().out)
    assert code == dc.EXIT_NOT_COVERED and any("num_hidden_layers differs" in r for r in out["reasons"])


def test_bundle_inputs_follows_a_cache_link_into_blobs(tmp_path):
    """A Hugging Face snapshot holds config.json as a link into the model's blobs/ directory; that link is read."""
    cfg = qwen4_dense()
    bundle, hf = lay_out_bundle(tmp_path, caps_obj=caps(), with_snapshot=False)
    model = hf / "hub" / "models--mando2222--qwen4-27b-franken-k20"
    (model / "blobs").mkdir(parents=True)
    (model / "blobs" / "abc123").write_text(json.dumps(dev_of(cfg)))
    snap = model / "snapshots" / REV
    snap.mkdir(parents=True)
    (snap / "config.json").symlink_to("../../blobs/abc123")
    _, dev, notes = dc.bundle_inputs(bundle, hf)
    assert dev["text_config"]["num_hidden_layers"] == 20 and notes == []


def test_bundle_inputs_refuses_a_link_out_of_the_cache(tmp_path):
    bundle, hf = lay_out_bundle(tmp_path, caps_obj=caps(), with_snapshot=False)
    outside = tmp_path / "elsewhere.json"
    outside.write_text(json.dumps({"model_type": "qwen4_exp"}))
    snap = hf / "hub" / "models--mando2222--qwen4-27b-franken-k20" / "snapshots" / REV
    snap.mkdir(parents=True)
    (snap / "config.json").symlink_to(outside)
    _, dev, notes = dc.bundle_inputs(bundle, hf)
    assert dev is None and any("outside" in n for n in notes)


def test_no_guidance_names_a_script_that_does_not_exist():
    """Issue 665: `scripts/reference_venv.sh` was named in guidance; the script is `scripts/reference_venv.py`. The project log may quote the old name."""
    root = Path(__file__).resolve().parent.parent
    assert (root / "scripts" / "reference_venv.py").is_file() and not (root / "scripts" / "reference_venv.sh").exists()
    offenders = [str(p.relative_to(root)) for pattern in ("orchard/**/*.py", "tests/**/*.py", "docs/**/*.md", "README.md")
                 for p in root.glob(pattern)
                 if p.name not in ("project-log.md", "test_drop_check.py") and "reference_venv.sh" in p.read_text(encoding="utf-8", errors="replace")]
    assert offenders == []
