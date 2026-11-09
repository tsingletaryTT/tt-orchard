# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The release-day check (orchard/drop_check.py).

`tt-orchard drop-check` reads only a release's config, tokenizer and weight index and says which route a bring-up takes: the
Qwen4 package (covered), the Qwen3.8 bundle (same architecture as Qwen3.8), or the full-port fallback (not covered). It never
downloads weights, never sends a token, and refuses any organisation other than the allowed one. The network is injected, so no
test touches it."""
import copy
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
    assert any("reference_venv.sh" in r for r in missing["reasons"])
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

@pytest.mark.parametrize("value", [None, [], {}, "5120", "", "x\x00", True, -1, 0, 10 ** 20, 1.5, float("nan")])
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
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeOpener:
    def __init__(self, routes):
        self.routes, self.requests = routes, []

    def __call__(self, req, timeout=None):
        self.requests.append(req)
        for suffix, body in self.routes.items():
            if req.full_url.endswith(suffix):
                if isinstance(body, int):
                    raise urllib.error.HTTPError(req.full_url, body, "x", {}, None)
                return FakeResponse(body)
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
    assert any(r.startswith(f"{key} is ") and "the package serves" in r for r in out["reasons"]), out["reasons"]
