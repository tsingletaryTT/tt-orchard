# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""config/bringup.toml, the run directory name and the `supervisor run` argument list
(orchard/bringup_config.py). Everything here is a pure function of a config file and a model id."""
from pathlib import Path

import pytest

from orchard import bringup_config as bc
from orchard import supervisor

MINIMAL = """
runs_root = "runs"

[coder]
target = "mando2222/qwen3.8-27b-dflash2-p300x2-q4kv"
port = 8000
chips = 4
"""


def write(tmp_path, text=MINIMAL, name="bringup.toml"):
    p = tmp_path / name
    p.write_text(text)
    return p


def load(tmp_path, text=MINIMAL):
    return bc.load(write(tmp_path, text))


# ---- loading ----------------------------------------------------------------------------------

def test_a_minimal_config_loads_with_documented_defaults(tmp_path):
    cfg = load(tmp_path)
    assert cfg.coder.kind == "container" and cfg.coder.profile == "default"
    assert cfg.coder.port == 8000 and cfg.coder.chips == 4
    assert cfg.gozer == "gozer" and cfg.package_format is None and cfg.skills_dirs == []
    assert cfg.tiers == tmp_path / "tiers.toml"          # beside the config file unless it says otherwise


def test_relative_paths_resolve_against_the_config_files_directory(tmp_path):
    cfg = load(tmp_path, MINIMAL + "")
    assert cfg.runs_root == tmp_path / "runs"
    cfg = load(tmp_path, 'runs_root = "/abs/runs"\ncache_root = "cache"\n[coder]\ntarget="t"\nport=1\nchips=2\n')
    assert cfg.runs_root == Path("/abs/runs") and cfg.cache_root == tmp_path / "cache"


@pytest.mark.parametrize("missing,text", [
    ("runs_root", '[coder]\ntarget="t"\nport=8000\nchips=2\n'),
    ("coder.target", 'runs_root="r"\n[coder]\nport=8000\nchips=2\n'),
    ("coder.port", 'runs_root="r"\n[coder]\ntarget="t"\nchips=2\n'),
    ("coder.chips", 'runs_root="r"\n[coder]\ntarget="t"\nport=8000\n'),
    ("coder", 'runs_root="r"\n'),
])
def test_a_missing_required_key_is_named(tmp_path, missing, text):
    with pytest.raises(bc.BringupConfigError) as e:
        load(tmp_path, text)
    assert missing in str(e.value)


def test_an_unknown_key_is_refused_with_a_suggestion(tmp_path):
    with pytest.raises(bc.BringupConfigError, match="did you mean 'runs_root'"):
        load(tmp_path, 'runs_roots = "x"\n' + MINIMAL)


def test_an_unknown_key_in_the_coder_table_is_refused(tmp_path):
    with pytest.raises(bc.BringupConfigError, match="did you mean 'profile'"):
        load(tmp_path, MINIMAL.replace("chips = 4", "chips = 4\nprofil = 'x'"))


def test_an_unknown_table_is_refused(tmp_path):
    with pytest.raises(bc.BringupConfigError, match="unknown"):
        load(tmp_path, MINIMAL + "\n[extras]\na = 1\n")


def test_change_me_values_are_refused(tmp_path):
    with pytest.raises(bc.BringupConfigError, match="CHANGE-ME"):
        load(tmp_path, MINIMAL.replace('"runs"', '"CHANGE-ME/runs"'))


def test_the_example_file_is_refused_until_edited_and_loads_once_it_is():
    example = Path(__file__).resolve().parent.parent / "config" / "bringup.example.toml"
    with pytest.raises(bc.BringupConfigError, match="CHANGE-ME"):
        bc.load(example)


def test_the_example_file_loads_when_its_placeholders_are_filled(tmp_path):
    example = Path(__file__).resolve().parent.parent / "config" / "bringup.example.toml"
    filled = example.read_text().replace("CHANGE-ME", "filled")
    assert bc.load(write(tmp_path, filled)).coder.port


@pytest.mark.parametrize("text", [
    'runs_root="r"\n[coder]\ntarget="t"\nport="8000"\nchips=2\n',     # port as a string
    'runs_root="r"\n[coder]\ntarget="t"\nport=0\nchips=2\n',          # port out of range
    'runs_root="r"\n[coder]\ntarget="t"\nport=8000\nchips=0\n',       # no chips
    'runs_root="r"\n[coder]\ntarget="t"\nport=8000\nchips=2\nkind="vm"\n',
    'runs_root=3\n[coder]\ntarget="t"\nport=8000\nchips=2\n',
])
def test_wrong_types_and_values_are_refused(tmp_path, text):
    with pytest.raises(bc.BringupConfigError):
        load(tmp_path, text)


def test_required_chips_must_be_a_list_of_chip_counts(tmp_path):
    assert load(tmp_path, 'required_chips="2,4"\n' + MINIMAL).required_chips == "2,4"
    with pytest.raises(bc.BringupConfigError, match="required_chips"):
        load(tmp_path, 'required_chips="two"\n' + MINIMAL)


def test_a_package_format_needs_a_namespace(tmp_path):
    with pytest.raises(bc.BringupConfigError, match="package_namespace"):
        load(tmp_path, 'package_format="v6"\n' + MINIMAL)
    assert load(tmp_path, 'package_format="v6"\npackage_namespace="me"\n' + MINIMAL).package_format == "v6"


def test_an_env_name_that_looks_like_a_credential_is_refused(tmp_path):
    with pytest.raises(bc.BringupConfigError, match="HF_TOKEN"):
        load(tmp_path, MINIMAL + '\n[env]\nHF_TOKEN = "x"\n')
    assert load(tmp_path, MINIMAL + '\n[env]\nHF_HUB_OFFLINE = "1"\n').env == {"HF_HUB_OFFLINE": "1"}


def test_a_missing_file_and_bad_toml_are_config_errors(tmp_path):
    with pytest.raises(bc.BringupConfigError, match="cannot read"):
        bc.load(tmp_path / "nope.toml")
    with pytest.raises(bc.BringupConfigError, match="cannot parse"):
        load(tmp_path, "runs_root = ")


# ---- the run directory ------------------------------------------------------------------------

@pytest.mark.parametrize("model,slug", [
    ("Cloudflare/clef", "cloudflare--clef"),
    ("Altworld/Hemmingway-1", "altworld--hemmingway-1"),
    ("iapp/openthai2.0-qwen3.8-27b", "iapp--openthai2.0-qwen3.8-27b"),
])
def test_the_slug_is_lower_case_with_a_double_dash(model, slug):
    assert bc.slug(model) == slug


@pytest.mark.parametrize("bad", ["clef", "../x", "a/b/c", "-x/y", "a b/c", "org/", "/name", "org/..",
                                 "org/na;me", "", "org/.hidden", "a/b\n"])
def test_a_model_id_that_could_escape_the_runs_root_is_refused(bad):
    with pytest.raises(bc.BringupConfigError):
        bc.slug(bad)


def test_the_default_run_directory_is_under_the_runs_root(tmp_path):
    cfg = load(tmp_path)
    assert bc.run_dir(cfg, "Cloudflare/clef") == tmp_path / "runs" / "cloudflare--clef"


# ---- the supervisor argument list -------------------------------------------------------------

FULL = """
runs_root = "/r"
tiers = "/etc/tiers.toml"
cache_root = "/c"
hf_home = "/h"
operator_home = "/o"
gozer = "/bin/gozer"
required_chips = "2,4"
skills_dirs = ["/s1", "/s2"]
package_format = "v6"
package_namespace = "me"
package_models_root = "/m"

[env]
HF_HUB_OFFLINE = "1"
HF_HOME = "/h"

[coder]
target = "pkg/name"
kind = "container"
profile = "batch8"
port = 8000
chips = 4
image_id = "abc123"
"""


def test_the_argument_list_matches_the_documented_run_command(tmp_path):
    cfg = load(tmp_path, FULL)
    argv = bc.supervisor_argv(cfg, "Cloudflare/clef")
    assert argv == [
        "run", "--model", "Cloudflare/clef", "--run-dir", "/r/cloudflare--clef", "--tiers", "/etc/tiers.toml",
        "--coder-target", "pkg/name", "--coder-kind", "container", "--coder-profile", "batch8",
        "--coder-port", "8000", "--coder-chips", "4", "--coder-image-id", "abc123",
        "--required-chips", "2,4", "--cache-root", "/c", "--hf-home", "/h", "--operator-home", "/o",
        "--package-format", "v6", "--package-namespace", "me", "--package-models-root", "/m",
        "--gozer", "/bin/gozer", "--skills-dir", "/s1", "--skills-dir", "/s2",
        "--env", "HF_HUB_OFFLINE=1", "--env", "HF_HOME=/h", "--unattended",
    ]


def test_the_argument_list_parses_with_the_real_supervisor_parser(tmp_path):
    ns = supervisor.parse(bc.supervisor_argv(load(tmp_path, FULL), "Cloudflare/clef"))
    assert (ns.model, ns.coder_port, ns.coder_chips, ns.package_format) == ("Cloudflare/clef", 8000, 4, "v6")
    assert ns.env == ["HF_HUB_OFFLINE=1", "HF_HOME=/h"] and ns.skills_dir == ["/s1", "/s2"]


def test_the_minimal_argument_list_leaves_optional_flags_out(tmp_path):
    argv = bc.supervisor_argv(load(tmp_path), "Cloudflare/clef")
    for flag in ("--coder-image-id", "--required-chips", "--cache-root", "--hf-home", "--package-format",
                 "--skills-dir", "--env"):
        assert flag not in argv
    supervisor.parse(argv)


def test_the_builder_never_accepts_visible_credentials_on_its_own(tmp_path):
    assert "--accept-credentials-visible" not in bc.supervisor_argv(load(tmp_path, FULL), "Cloudflare/clef")


def test_an_explicit_run_dir_replaces_the_default(tmp_path):
    argv = bc.supervisor_argv(load(tmp_path), "Cloudflare/clef", run_dir=Path("/x/y"))
    assert argv[argv.index("--run-dir") + 1] == "/x/y"


@pytest.mark.parametrize("coder", ["port = 70000", "port = true", "chips = true", "port = -1"])
def test_ports_and_chip_counts_must_be_real_numbers_in_range(tmp_path, coder):
    key = coder.split()[0]
    text = MINIMAL.replace(next(l for l in MINIMAL.splitlines() if l.startswith(key)), coder)
    with pytest.raises(bc.BringupConfigError):
        load(tmp_path, text)


@pytest.mark.parametrize("value", ["0", "-5", "true", '"40"'])
def test_min_free_gb_must_be_a_positive_number(tmp_path, value):
    with pytest.raises(bc.BringupConfigError, match="min_free_gb"):
        load(tmp_path, f"min_free_gb = {value}\n" + MINIMAL)


def test_min_free_gb_is_kept_when_valid(tmp_path):
    assert load(tmp_path, "min_free_gb = 60\n" + MINIMAL).min_free_gb == 60.0


def test_the_snapshot_input_is_passed_to_the_supervisor_as_an_input_flag(tmp_path):
    argv = bc.supervisor_argv(load(tmp_path), "Cloudflare/clef", inputs={"model": "/hf/snap"})
    assert argv[argv.index("--input") + 1] == "model=/hf/snap"
    supervisor.parse(argv)


def test_accepting_credentials_is_added_only_when_the_operator_asked(tmp_path):
    cfg = load(tmp_path)
    assert "--accept-credentials-visible" in bc.supervisor_argv(cfg, "Cloudflare/clef", accept_credentials=True)
    assert "--accept-credentials-visible" not in bc.supervisor_argv(cfg, "Cloudflare/clef")
    supervisor.parse(bc.supervisor_argv(cfg, "Cloudflare/clef", accept_credentials=True))


# ---- the reference interpreter ----------------------------------------------------------------

def test_a_reference_python_is_read_and_resolved_against_the_config_directory(tmp_path):
    cfg = load(tmp_path, 'reference_python = "venvs/ref/bin/python"\n' + MINIMAL)
    assert cfg.reference_python == tmp_path / "venvs" / "ref" / "bin" / "python"
    assert load(tmp_path, 'reference_python = "/abs/py"\n' + MINIMAL).reference_python == Path("/abs/py")
    assert load(tmp_path).reference_python is None


def test_the_reference_python_is_handed_to_agents_as_an_input(tmp_path):
    argv = bc.supervisor_argv(load(tmp_path, 'reference_python = "/v/bin/python"\n' + MINIMAL), "Cloudflare/clef",
                              inputs={"model": "/hf/snap"})
    inputs = [argv[i + 1] for i, a in enumerate(argv) if a == "--input"]
    assert inputs == ["model=/hf/snap", "reference_python=/v/bin/python"]
    assert supervisor.parse(argv).input == inputs


def test_without_a_reference_python_no_such_input_is_passed(tmp_path):
    argv = bc.supervisor_argv(load(tmp_path), "Cloudflare/clef", inputs={"model": "/hf/snap"})
    assert [argv[i + 1] for i, a in enumerate(argv) if a == "--input"] == ["model=/hf/snap"]


def test_an_empty_reference_python_is_refused(tmp_path):
    with pytest.raises(bc.BringupConfigError, match="reference_python"):
        load(tmp_path, 'reference_python = ""\n' + MINIMAL)


# ---- a lab box (`[lab]`) ------------------------------------------------------------------------

LAB = """
runs_root = "/srv/orchard/runs"
cache_root = "/srv/orchard/cache"
hf_home = "/srv/orchard/hf"
tt_model_root = "/srv/orchard/tt-model/models"
required_chips = "1,2"

[coder]
target = "raahemnabeel/qwen3-coder-next-blackhole"
profile = "p300"
port = 8001
chips = 2

[lab]
host = "node4"
root = "/srv/orchard"
path = ["~/.local/bin", "~/.tenstorrent-venv/bin"]
test_python = "/srv/orchard/venvs/reference/bin/python"
"""


def test_a_lab_table_loads_and_becomes_the_supervisors_lab_flags(tmp_path):
    cfg = load(tmp_path, LAB)
    assert cfg.lab.host == "node4" and cfg.lab.root == Path("/srv/orchard")
    assert cfg.lab.gozer == "gozer" and cfg.lab.python == "python3"
    assert cfg.tt_model_root == Path("/srv/orchard/tt-model/models")
    argv = bc.supervisor_argv(cfg, "Altworld/Hemmingway-1")
    pairs = list(zip(argv, argv[1:]))
    assert ("--lab", "node4") in pairs and ("--lab-root", "/srv/orchard") in pairs
    assert ("--tt-model-root", "/srv/orchard/tt-model/models") in pairs
    assert ("--lab-path", "~/.local/bin") in pairs and ("--lab-path", "~/.tenstorrent-venv/bin") in pairs
    assert ("--lab-test-python", "/srv/orchard/venvs/reference/bin/python") in pairs
    args = supervisor.parse(argv)                       # the supervisor accepts every flag it is given
    assert args.lab == "node4" and args.lab_path == ["~/.local/bin", "~/.tenstorrent-venv/bin"]


def test_without_a_lab_table_no_lab_flag_is_passed(tmp_path):
    argv = bc.supervisor_argv(load(tmp_path), "Altworld/Hemmingway-1")
    assert not [a for a in argv if a.startswith("--lab")]


@pytest.mark.parametrize("table,message", [
    ('[lab]\nroot = "/srv/orchard"\n', "host"),
    ('[lab]\nhost = "node4"\n', "root"),
    ('[lab]\nhost = "node4"\nroot = "relative/dir"\n', "absolute"),
    ('[lab]\nhost = "node4"\nroot = "/srv/orchard"\nhostname = "x"\n', "hostname"),
    ('[lab]\nhost = "-oProxyCommand=evil"\nroot = "/srv/orchard"\n', "host"),
    ('[lab]\nhost = "node4"\nroot = "/srv/orchard"\npath = "~/.local/bin"\n', "list"),
])
def test_a_bad_lab_table_is_refused(tmp_path, table, message):
    with pytest.raises(bc.BringupConfigError, match=message):
        load(tmp_path, MINIMAL + table)
