"""The `tt-orchard` command (orchard/cli.py).

The supervisor, the download and the outside signals are all injected. The tests that matter most prove
what must never happen: a blocked preflight reaching the supervisor, a dry run starting anything, a
download when the snapshot is already local, and credentials being accepted without the operator's flag.
"""
import io
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from orchard import cli, fetch
from orchard import preflight as pf
from orchard import supervisor

REPO = Path(__file__).resolve().parent.parent
REV = "2f3de3dd85f379784083b0814d997ab627200f0c"
FREE = """grain: board   (2 boards, 4 chips)
board 0000046131924062  (p300c)
  chip 0  0000:01:00.0  FREE
  chip 1  0000:02:00.0  FREE
"""
PAYLOAD = {"id": "Cloudflare/clef", "private": False, "gated": False, "sha": REV,
           "cardData": {"license": "apache-2.0"},
           "siblings": [{"rfilename": "LICENSE", "size": 1000}, {"rfilename": "config.json", "size": 1000},
                        {"rfilename": "model-00001-of-00002.safetensors", "size": 5_000_000_000}]}
TIERS = {"large": {"placement": "chips", "endpoint": "http://127.0.0.1:8000/v1", "model": "m"}}


class Tty(io.StringIO):
    encoding = "utf-8"

    def isatty(self):
        return True


class Pipe(io.StringIO):
    encoding = "utf-8"

    def isatty(self):
        return False


class Tiers:
    tiers = TIERS


def signals(**over):
    base = dict(hub_info=lambda m: (pf.parse_hub(PAYLOAD), None), local_snapshot=lambda m: None,
                free_gb=lambda p: 1000.0, same_device=lambda a, b: False, credentials=lambda: [],
                port_in_use=lambda port: False, gozer_status=lambda: FREE, load_tiers=lambda p: Tiers())
    base.update(over)
    return pf.Signals(**base)


@pytest.fixture
def conf(tmp_path):
    p = tmp_path / "bringup.toml"
    p.write_text(f'runs_root = "{tmp_path}/runs"\nhf_home = "{tmp_path}/hf"\ncache_root = "{tmp_path}/cache"\n'
                 '[coder]\ntarget = "pkg/coder"\nport = 8000\nchips = 4\n')
    return p


class Recorder:
    """A supervisor that records its argv, and a downloader that records what it was asked for."""

    def __init__(self, code=0):
        self.code, self.argv, self.fetched = code, None, []

    def supervisor(self, argv):
        self.argv = argv
        return self.code

    def fetcher(self, model_id, revision, hf_home, *, files, **kw):
        self.fetched.append((model_id, revision, Path(hf_home), list(files)))
        return fetch.snapshot_dir(hf_home, model_id, revision)


def run(conf, args, rec=None, sig=None, out=None, env=None):
    rec = rec or Recorder()
    out = out if out is not None else Pipe()
    code = cli.main(["--config", str(conf), *args], env=env or {}, stdout=out, signals=sig or signals(),
                    supervisor_main=rec.supervisor, fetcher=rec.fetcher)
    return code, out.getvalue(), rec


# ---- bringup: the paths that must not start anything ------------------------------------------

def test_a_dry_run_prints_the_checks_and_the_command_and_starts_nothing(conf):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--dry-run"])
    assert code == 0 and rec.argv is None and rec.fetched == []
    for name in ("hub", "disk", "credentials", "tiers", "port", "gozer"):
        assert name in out
    assert "python3 -m orchard.supervisor run --model Cloudflare/clef" in out
    assert "cloudflare--clef" in out


def test_a_blocked_preflight_never_reaches_the_supervisor_or_the_download(conf, capsys):
    s = signals(credentials=lambda: [Path("/h/.netrc")], port_in_use=lambda p: True)
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"], sig=s)
    assert code == 2 and rec.argv is None and rec.fetched == []
    assert "credentials-needed" in out and "coder-unusable" in out and "Nothing was started" in out


def test_a_dry_run_that_is_blocked_still_exits_2(conf):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--dry-run"],
                         sig=signals(gozer_status=lambda: ""))
    assert code == 2 and rec.argv is None


def test_an_invalid_model_id_is_refused_before_anything_else(conf, capsys):
    code, out, rec = run(conf, ["bringup", "../etc/passwd"])
    assert code == 2 and rec.argv is None and "org/name" in capsys.readouterr().err


def test_a_bad_config_is_refused_with_its_message(tmp_path, capsys):
    bad = tmp_path / "bad.toml"
    bad.write_text("runs_root = 3\n")
    code, out, rec = run(bad, ["bringup", "Cloudflare/clef"])
    assert code == 2 and rec.argv is None
    assert "refused:" in capsys.readouterr().err


# ---- bringup: the run ------------------------------------------------------------------------

def test_the_snapshot_is_fetched_at_the_revision_the_preflight_saw_and_handed_to_the_supervisor(conf, tmp_path):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"])
    model, revision, hf_home, files = rec.fetched[0]
    assert (model, revision, hf_home) == ("Cloudflare/clef", REV, tmp_path / "hf")
    assert files == ["LICENSE", "config.json", "model-00001-of-00002.safetensors"]
    snap = fetch.snapshot_dir(tmp_path / "hf", "Cloudflare/clef", REV)
    assert rec.argv[rec.argv.index("--input") + 1] == f"model={snap}"
    assert rec.argv[0] == "run" and rec.argv[rec.argv.index("--run-dir") + 1] == str(tmp_path / "runs" / "cloudflare--clef")


def test_a_snapshot_that_is_already_local_is_used_and_nothing_is_downloaded(conf, tmp_path):
    local = tmp_path / "hf" / "hub" / "models--Cloudflare--clef" / "snapshots" / "oldrev"
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"], sig=signals(local_snapshot=lambda m: local))
    assert rec.fetched == [] and rec.argv[rec.argv.index("--input") + 1] == f"model={local}"


def test_a_local_snapshot_works_when_the_hub_cannot_be_reached(conf, tmp_path):
    local = tmp_path / "snap"
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"],
                         sig=signals(hub_info=lambda m: (None, "timed out"), local_snapshot=lambda m: local))
    assert code == 0 and rec.argv is not None and rec.fetched == []


def test_no_fetch_with_no_local_snapshot_is_refused(conf, capsys):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--no-fetch"])
    assert code == 2 and rec.argv is None and rec.fetched == []
    assert "--no-fetch" in capsys.readouterr().err


def test_a_failed_download_is_a_refusal_because_nothing_was_started(conf, capsys):
    def boom(*a, **k):
        raise fetch.FetchError("hf download exited 1")
    rec = Recorder()
    code = cli.main(["--config", str(conf), "bringup", "Cloudflare/clef"], env={}, stdout=Pipe(),
                    signals=signals(), supervisor_main=rec.supervisor, fetcher=boom)
    assert code == 2 and rec.argv is None and "hf download exited 1" in capsys.readouterr().err


def test_credentials_are_accepted_only_with_the_operator_flag(conf):
    s = signals(credentials=lambda: [Path("/h/.netrc")])
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--accept-credentials-visible"], sig=s)
    assert code == 0 and "--accept-credentials-visible" in rec.argv
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"], sig=s)
    assert code == 2 and rec.argv is None


def test_without_the_flag_the_supervisor_is_never_told_to_accept_credentials(conf):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"])
    assert "--accept-credentials-visible" not in rec.argv


def test_an_explicit_run_dir_replaces_the_default(conf, tmp_path):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--run-dir", str(tmp_path / "mine")])
    assert rec.argv[rec.argv.index("--run-dir") + 1] == str(tmp_path / "mine")


def test_an_existing_ledger_means_resuming_and_a_busy_port_only_warns(conf, tmp_path):
    run_dir = tmp_path / "runs" / "cloudflare--clef"
    run_dir.mkdir(parents=True)
    (run_dir / "ledger.jsonl").write_text("")
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"], sig=signals(port_in_use=lambda p: True))
    assert code == 0 and rec.argv is not None and "resuming" in out


def test_supervisor_exit_codes_pass_through_and_each_end_state_has_a_closing_line(conf):
    for code, word in ((0, "ripe"), (4, "fallen"), (3, "same command again")):
        got, out, rec = run(conf, ["bringup", "Cloudflare/clef"], rec=Recorder(code))
        assert got == code and word in out, (code, out)


# ---- status and control forward ---------------------------------------------------------------

def test_status_by_model_forwards_the_run_directory_and_options(conf, tmp_path):
    code, out, rec = run(conf, ["status", "Cloudflare/clef", "--json", "--style", "plain"])
    assert rec.argv == ["status", "--run-dir", str(tmp_path / "runs" / "cloudflare--clef"), "--json",
                        "--style", "plain"]


def test_status_with_a_run_dir_needs_no_config(tmp_path):
    rec = Recorder()
    code = cli.main(["status", "--run-dir", str(tmp_path)], env={"HOME": str(tmp_path / "none")},
                    stdout=Pipe(), signals=signals(), supervisor_main=rec.supervisor, fetcher=rec.fetcher)
    assert rec.argv == ["status", "--run-dir", str(tmp_path)]


@pytest.mark.parametrize("word", ["pause", "resume", "abort"])
def test_control_words_forward_to_the_supervisors_control_command(conf, tmp_path, word):
    code, out, rec = run(conf, [word, "Cloudflare/clef"])
    assert rec.argv == ["control", "--run-dir", str(tmp_path / "runs" / "cloudflare--clef"), word]


def test_status_without_a_model_or_run_dir_is_refused(conf, capsys):
    code, out, rec = run(conf, ["status"])
    assert code == 2 and rec.argv is None and "--run-dir" in capsys.readouterr().err


def test_the_arguments_forwarded_to_the_real_supervisor_parser_are_valid(conf, tmp_path):
    for args in (["status", "Cloudflare/clef", "--style", "plain"], ["pause", "Cloudflare/clef"]):
        code, out, rec = run(conf, args)
        supervisor.parse(rec.argv)


# ---- finding the config -----------------------------------------------------------------------

def test_the_config_comes_from_the_flag_then_the_environment_then_the_checkout_then_the_home(tmp_path):
    checkout, home = tmp_path / "co", tmp_path / "home"
    (checkout / "config").mkdir(parents=True)
    (home / ".config" / "tt-orchard").mkdir(parents=True)
    files = {k: p for k, p in (("flag", tmp_path / "flag.toml"), ("env", tmp_path / "env.toml"),
                               ("checkout", checkout / "config" / "bringup.toml"),
                               ("home", home / ".config" / "tt-orchard" / "bringup.toml"))}
    for p in files.values():
        p.write_text("")
    find = lambda **kw: cli.find_config(kw.get("explicit"), kw.get("env", {}), checkout, home)
    assert find(explicit=str(files["flag"]), env={"ORCHARD_BRINGUP_CONFIG": str(files["env"])}) == files["flag"]
    assert find(env={"ORCHARD_BRINGUP_CONFIG": str(files["env"])}) == files["env"]
    assert find() == files["checkout"]
    files["checkout"].unlink()
    assert find() == files["home"]
    files["home"].unlink()
    with pytest.raises(cli.NoConfig) as e:
        find()
    assert "bringup.toml" in str(e.value) and "ORCHARD_BRINGUP_CONFIG" in str(e.value)


def test_a_missing_explicit_config_is_not_silently_replaced_by_another(tmp_path):
    other = tmp_path / "co" / "config"
    other.mkdir(parents=True)
    (other / "bringup.toml").write_text("")
    with pytest.raises(cli.NoConfig):
        cli.find_config(str(tmp_path / "missing.toml"), {}, tmp_path / "co", tmp_path)


# ---- the screen ---------------------------------------------------------------------------------

def test_piped_output_has_no_escape_codes_and_no_emoji(conf):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--dry-run"])
    assert "\x1b" not in out and all(ord(c) < 0x2000 or c in "║╔╚═╠─" for c in out)


def test_style_pretty_adds_emoji_and_style_plain_removes_them_on_a_terminal(conf):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--dry-run", "--style", "pretty"],
                         env={"NO_COLOR": "1"})
    assert "🍎" in out and "\x1b" not in out
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--dry-run", "--style", "plain"], out=Tty(),
                         env={"TERM": "xterm-256color", "COLORTERM": "truecolor"})
    assert "🍎" not in out and "\x1b" not in out


def test_every_preflight_line_stays_inside_80_columns_without_a_right_border(conf):
    from orchard import ui
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--dry-run", "--style", "pretty"],
                         sig=signals(credentials=lambda: [Path("/home/someone/" + "x" * 90 + "/.netrc")]))
    for line in out.splitlines():
        if line.startswith(("║", "╔", "╚", " ")):
            assert ui.visible_width(line) <= 80, line


# ---- version and the shim -----------------------------------------------------------------------

def test_the_version_matches_pyproject():
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]
    import orchard
    assert orchard.__version__ == pyproject


def test_the_version_flag_prints_it(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"], env={}, stdout=Pipe())
    assert e.value.code == 0 and "tt-orchard" in capsys.readouterr().out


def test_the_shim_runs_through_a_symlink_from_any_directory(tmp_path):
    link = tmp_path / "tt-orchard"
    link.symlink_to(REPO / "bin" / "tt-orchard")
    done = subprocess.run([str(link), "--version"], cwd=tmp_path, capture_output=True, text=True,
                          env={"PATH": os.environ["PATH"], "HOME": str(tmp_path)})
    assert done.returncode == 0 and done.stdout.startswith("tt-orchard ")


def test_the_label_column_fits_every_check_name_with_its_icon_and_a_space(conf):
    from orchard import orchard_view, ui
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--dry-run"])
    names = ["hub", "disk", "credentials", "tiers", "port", "gozer"]
    assert all(ui.visible_width("⛔ " + n) + 1 <= orchard_view.PREFLIGHT_LABEL for n in names)


def test_the_plain_page_says_each_checks_status_in_words(conf):
    s = signals(credentials=lambda: [Path("/h/.netrc")],
                gozer_status=lambda: FREE.replace("FREE", "HELD", 1))
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--dry-run"], sig=s)
    rows = {line.split()[1]: line for line in out.splitlines() if line.startswith("║") and len(line.split()) > 2}
    assert " BLOCK " in rows["credentials"] and " warn " in rows["gozer"] and " ok " in rows["disk"]


def test_a_bringup_is_always_an_unattended_supervisor_run(conf):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"])
    assert "--unattended" in rec.argv


def test_a_blocked_exit_prints_the_reason_and_where_the_bundle_is(conf, tmp_path):
    run_dir = tmp_path / "runs" / "cloudflare--clef"
    run_dir.mkdir(parents=True)

    class Blocking(Recorder):
        def supervisor(self, argv):
            self.argv = argv
            (run_dir / "blocked.json").write_text('{"code": "needs-new-model-code", "reason": "full port"}')
            return 5
    got, out, rec = run(conf, ["bringup", "Cloudflare/clef"], rec=Blocking())
    assert got == 5 and "frost" in out and "needs-new-model-code" in out and "BLOCKED.md" in out


def test_a_blocked_exit_without_a_bundle_still_says_blocked(conf):
    got, out, rec = run(conf, ["bringup", "Cloudflare/clef"], rec=Recorder(5))
    assert got == 5 and "blocked" in out
