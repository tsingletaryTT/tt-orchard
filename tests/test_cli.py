# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
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
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--refuse-credentials-visible"], sig=s)
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


def test_visible_credentials_are_accepted_by_default_and_the_supervisor_is_told_so(conf):
    s = signals(credentials=lambda: [Path("/h/.netrc")])
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"], sig=s)
    assert code == 0 and "--accept-credentials-visible" in rec.argv
    assert "accepted" in out and ".netrc" in out            # the page still names what is visible


def test_the_old_flag_is_still_accepted(conf):
    s = signals(credentials=lambda: [Path("/h/.netrc")])
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--accept-credentials-visible"], sig=s)
    assert code == 0 and "--accept-credentials-visible" in rec.argv


def test_refusing_visible_credentials_restores_the_block(conf):
    s = signals(credentials=lambda: [Path("/h/.netrc")])
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--refuse-credentials-visible"], sig=s)
    assert code == 2 and rec.argv is None and "credentials-needed" in out


def test_with_no_credentials_visible_the_flag_is_still_passed_so_the_ledger_is_consistent(conf):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef"])
    assert "--accept-credentials-visible" in rec.argv


def test_refusing_means_the_supervisor_is_not_told_to_accept(conf):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--refuse-credentials-visible"])
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
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--dry-run", "--style", "pretty",
                                   "--refuse-credentials-visible"],
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
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--dry-run", "--refuse-credentials-visible"], sig=s)
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


# ---- bringup and watch: say what is happening -------------------------------------------------

class Writer(Recorder):
    """A supervisor that writes ledger entries into the run directory it is given, as the real one does."""

    def __init__(self, code=0, script=None):
        super().__init__(code)
        self.script = script or []

    def supervisor(self, argv):
        from orchard.ledger import Ledger
        self.argv = argv
        run_dir = Path(argv[argv.index("--run-dir") + 1])
        run_dir.mkdir(parents=True, exist_ok=True)
        with Ledger(run_dir / "ledger.jsonl") as led:
            for ev, stage, data in self.script:
                led.append(ev, stage, **data)
        if self.code == 5:
            from orchard import blocked
            from orchard.ledger import read_entries
            blocked.write_bundle(run_dir, read_entries(run_dir / "ledger.jsonl"), "stage-failed",
                                 "stage 1 failed after escalation", now=0)
        return self.code


SCRIPT = [("run_start", None, {"model": "Cloudflare/clef", "coder": {"target": "pkg/coder"}}),
          ("stage_start", 0, {"escalated": False, "resumed": False}),
          ("stage_end", 0, {"result": "pass"})]


def test_bringup_prints_what_the_supervisor_records_as_it_goes(conf):
    code, out, _ = run(conf, ["bringup", "Cloudflare/clef"], rec=Writer(0, SCRIPT))
    assert "stage 0 (survey): starting" in out and "stage 0 (survey): pass" in out
    assert "orchardist" in out


def test_quiet_turns_the_narration_off(conf):
    code, out, _ = run(conf, ["bringup", "Cloudflare/clef", "--quiet"], rec=Writer(0, SCRIPT))
    assert "stage 0 (survey)" not in out


def test_a_blocked_run_ends_with_what_was_tried_and_how_to_unblock(conf):
    script = SCRIPT[:2] + [("decision", 1, {"decision": "pause", "reason": "stage 1 failed after escalation"}),
                           ("decision", None, {"decision": "blocked", "code": "stage-failed",
                                               "reason": "stage 1 failed after escalation"})]
    code, out, _ = run(conf, ["bringup", "Cloudflare/clef"], rec=Writer(5, script))
    tail = out.split("frost")[-1]
    assert code == 5 and "What was tried" in tail and "How to unblock" in tail
    assert "tt-orchard bringup Cloudflare/clef" in tail


def test_watch_once_shows_the_recent_history_and_stops(conf, tmp_path):
    rec = Writer(0, SCRIPT)
    run(conf, ["bringup", "Cloudflare/clef", "--quiet"], rec=rec)
    code, out, _ = run(conf, ["watch", "Cloudflare/clef", "--once"])
    assert code == 0 and "stage 0 (survey): pass" in out


def test_watch_without_a_run_is_refused(conf, capsys):
    code, out, _ = run(conf, ["watch", "Cloudflare/clef", "--once"])
    assert code == 2 and "no run" in capsys.readouterr().err


def test_watch_stops_by_itself_when_the_run_has_ended(conf):
    import threading
    script = SCRIPT + [("decision", None, {"decision": "ready for operator review", "bundle": "stages/8/bundle"})]
    run(conf, ["bringup", "Cloudflare/clef", "--quiet"], rec=Writer(0, script))
    result = {}
    t = threading.Thread(target=lambda: result.update(r=run(conf, ["watch", "Cloudflare/clef"])), daemon=True)
    t.start()
    t.join(10)
    assert not t.is_alive() and result["r"][0] == 0 and "ready for operator review" in result["r"][1]


def test_the_narrator_stops_and_flushes_even_when_the_supervisor_raises(conf):
    class Boom(Writer):
        def supervisor(self, argv):
            super().supervisor(argv)
            raise RuntimeError("supervisor died")
    with pytest.raises(RuntimeError):
        out = Pipe()
        cli.main(["--config", str(conf), "bringup", "Cloudflare/clef"], env={}, stdout=out, signals=signals(),
                 supervisor_main=Boom(0, SCRIPT).supervisor, fetcher=Recorder().fetcher)
    assert "stage 0 (survey): pass" in out.getvalue()


def test_a_resumed_run_starts_by_showing_where_it_left_off(conf):
    first = Writer(0, SCRIPT)
    run(conf, ["bringup", "Cloudflare/clef", "--quiet"], rec=first)
    code, out, _ = run(conf, ["bringup", "Cloudflare/clef"], rec=Writer(0, []))
    assert "stage 0 (survey): pass" in out


def test_a_fresh_run_does_not_replay_anything(conf):
    code, out, _ = run(conf, ["bringup", "Cloudflare/clef"], rec=Writer(0, []))
    assert "orchardist" not in out


def test_watch_all_starts_from_the_first_entry(conf):
    script = [("run_start", None, {"model": "Cloudflare/clef", "coder": {}})] + [
        ("stage_start", 0, {"escalated": False, "resumed": False})] * 20
    run(conf, ["bringup", "Cloudflare/clef", "--quiet"], rec=Writer(0, script))
    _, recent, _ = run(conf, ["watch", "Cloudflare/clef", "--once"])
    _, everything, _ = run(conf, ["watch", "Cloudflare/clef", "--once", "--all"])
    assert "run started" not in recent and "run started" in everything


# ---- bringup: choosing the model to base the run on -------------------------------------------

from orchard import nearest  # noqa: E402

GEMMA = "google/gemma-4-12B"
GEMMA_IT = "google/gemma-4-12B-it"
BUNDLE = "stisiTT/gemma-4-12b-it-p150"
GEMMA_REV = "d" * 40


def hub_payload(model_id):
    if model_id == "jialinyyzz/humanizer":
        return {**PAYLOAD, "id": model_id, "cardData": {"license": "apache-2.0", "base_model": GEMMA}}
    return {"id": model_id, "private": False, "gated": False, "sha": GEMMA_REV, "cardData": {"license": "gemma"},
            "siblings": [{"rfilename": "config.json", "size": 1}, {"rfilename": "w.safetensors", "size": 2_000_000_000}]}


class BaseWorld:
    """The outside world for the base-model tests: bundles that are installed, what a search finds, and
    what `tt-model pull` does."""

    def __init__(self, installed=(), found=(), pull_ok=True, local=None, pull_installs=(BUNDLE, GEMMA_IT)):
        self.installed = list(installed)
        self.found = list(found)
        self.pull_ok, self.local, self.pull_installs = pull_ok, local or {}, pull_installs
        self.pulled, self.asked, self.chose = [], [], []

    def signals(self, **over):
        return signals(hub_info=lambda m: (pf.parse_hub(hub_payload(m)), None),
                       installed_bundles=lambda: list(self.installed),
                       search_bundles=lambda q: self.asked.append(q) or list(self.found),
                       local_snapshot=lambda m: self.local.get(m), **over)

    def puller(self, bundle):
        self.pulled.append(bundle)
        if self.pull_ok:
            name, repo = self.pull_installs
            self.installed.append(nearest.Installed(name, repo, "6", 1))
            return True, ""
        return False, "no such bundle"


def run_base(conf, args, world, chooser=None, out=None):
    rec = Recorder()
    out = out if out is not None else Pipe()
    code = cli.main(["--config", str(conf), "bringup", "jialinyyzz/humanizer", *args], env={}, stdout=out,
                    signals=world.signals(), supervisor_main=rec.supervisor, fetcher=rec.fetcher,
                    chooser=chooser, puller=world.puller)
    return code, out.getvalue(), rec


FOUND = [{"name": BUNDLE, "installed": False}]


def test_without_a_base_bundle_a_script_run_is_refused_with_the_candidates_and_the_flag(conf):
    w = BaseWorld(found=FOUND)
    code, out, rec = run_base(conf, [], w)
    assert code == 2 and rec.argv is None and rec.fetched == [] and w.pulled == []
    assert "nearest-model-missing" in out and BUNDLE in out
    assert f"tt-orchard bringup jialinyyzz/humanizer --base {BUNDLE}" in out
    assert w.asked == ["gemma"]


def test_the_search_is_never_asked_when_the_base_is_already_served(conf, tmp_path):
    inst = [nearest.Installed("episod/gemma-p300", GEMMA, "6", 2)]
    w = BaseWorld(installed=inst, found=FOUND, local={GEMMA: tmp_path / "g"})
    code, out, rec = run_base(conf, [], w)
    assert code == 0 and w.asked == []


def test_at_a_terminal_the_operator_is_asked_to_pick_and_the_pick_is_installed_and_used(conf, tmp_path):
    w = BaseWorld(found=FOUND)

    def chooser(candidates, model, base):
        w.chose.append(([c["name"] for c in candidates], model, base))
        return BUNDLE
    code, out, rec = run_base(conf, [], w, chooser=chooser)
    assert w.chose == [([BUNDLE], "jialinyyzz/humanizer", GEMMA)]
    assert w.pulled == [BUNDLE]
    assert code == 0 and rec.argv is not None
    base_fetch = [f for f in rec.fetched if f[0] == GEMMA_IT]
    assert base_fetch and base_fetch[0][1] == GEMMA_REV
    snap = fetch.snapshot_dir(tmp_path / "hf", GEMMA_IT, GEMMA_REV)
    assert f"base={snap}" in rec.argv


def test_choosing_nothing_stops_the_run_and_installs_nothing(conf):
    w = BaseWorld(found=FOUND)
    code, out, rec = run_base(conf, [], w, chooser=lambda c, m, b: None)
    assert code == 2 and rec.argv is None and w.pulled == []


def test_an_already_installed_pick_is_not_pulled_again(conf, tmp_path):
    inst = [nearest.Installed(BUNDLE, GEMMA_IT, "6", 1)]
    w = BaseWorld(installed=inst, found=[{"name": BUNDLE, "installed": True}], local={GEMMA_IT: tmp_path / "g"})
    code, out, rec = run_base(conf, [], w, chooser=lambda c, m, b: BUNDLE)
    assert code == 0 and w.pulled == []


def test_base_names_a_bundle_to_install_and_needs_no_prompt(conf, tmp_path):
    w = BaseWorld(found=FOUND)
    code, out, rec = run_base(conf, ["--base", BUNDLE], w)
    assert code == 0 and w.pulled == [BUNDLE] and rec.argv is not None
    assert any(f"base={fetch.snapshot_dir(tmp_path / 'hf', GEMMA_IT, GEMMA_REV)}" == a for a in rec.argv)


def test_base_naming_an_installed_bundle_uses_its_weights_repo_without_pulling(conf, tmp_path):
    inst = [nearest.Installed(BUNDLE, GEMMA_IT, "6", 1)]
    w = BaseWorld(installed=inst, local={GEMMA_IT: tmp_path / "g"})
    code, out, rec = run_base(conf, ["--base", BUNDLE], w)
    assert code == 0 and w.pulled == [] and w.asked == []
    assert f"base={tmp_path / 'g'}" in rec.argv


def test_base_naming_a_weights_repo_with_a_bundle_installed_works(conf, tmp_path):
    inst = [nearest.Installed("x/y-p150", GEMMA_IT, "6", 1)]
    w = BaseWorld(installed=inst, local={GEMMA_IT: tmp_path / "g"})
    code, out, rec = run_base(conf, ["--base", GEMMA_IT], w)
    assert code == 0 and w.pulled == []


def test_a_failed_install_stops_the_run_with_the_reason(conf, capsys):
    w = BaseWorld(found=FOUND, pull_ok=False)
    code, out, rec = run_base(conf, ["--base", BUNDLE], w)
    err = capsys.readouterr().err
    assert code == 2 and rec.argv is None and "no such bundle" in err and BUNDLE in err


def test_a_local_base_snapshot_is_handed_to_the_supervisor_and_nothing_is_downloaded(conf, tmp_path):
    inst = [nearest.Installed("episod/gemma-p300", GEMMA, "6", 2)]
    w = BaseWorld(installed=inst, local={GEMMA: tmp_path / "g"})
    code, out, rec = run_base(conf, [], w)
    assert f"base={tmp_path / 'g'}" in rec.argv and [f[0] for f in rec.fetched] == ["jialinyyzz/humanizer"]


def test_a_dry_run_neither_asks_nor_installs(conf):
    w = BaseWorld(found=FOUND)
    code, out, rec = run_base(conf, ["--dry-run"], w, chooser=lambda c, m, b: pytest.fail("asked"))
    assert code == 2 and w.pulled == [] and rec.argv is None


def test_the_preflight_page_shows_the_base_row(conf):
    code, out, rec = run_base(conf, [], BaseWorld(found=FOUND))
    flat = " ".join(out.replace("║", " ").split())
    assert "BLOCK no installed bundle serves google/gemma-4-12B" in flat and "[nearest-model-missing]" in flat


def test_a_search_that_found_nothing_says_how_to_install_a_bundle_by_hand(conf):
    code, out, rec = run_base(conf, [], BaseWorld(found=[]))
    assert code == 2 and "found no bundle" in out.replace("\n", " ").replace("║", "")


def answers_from(items):
    it = iter(items)
    return lambda _prompt: next(it)


def test_the_prompt_lists_the_candidates_and_returns_the_one_chosen():
    cands = [{"name": "a/installed", "installed": True}, {"name": "b/other", "installed": False}]
    out = Pipe()
    got = cli.pick_base(cands, "x/new", "g/base", out=out, ask=answers_from(["2"]))
    text = out.getvalue()
    assert got == "b/other" and "1. a/installed  [installed]" in text and "2. b/other  [not installed" in text
    assert "x/new is based on g/base" in text


@pytest.mark.parametrize("answers, expected, bad", [(["0"], None, 0), ([""], None, 0),
                                                    (["x", "9", "1"], "a/installed", 2)])
def test_the_prompt_stops_on_zero_or_enter_and_asks_again_on_a_bad_answer(answers, expected, bad):
    cands = [{"name": "a/installed", "installed": True}]
    out = Pipe()
    assert cli.pick_base(cands, "m", "b", out=out, ask=answers_from(answers)) == expected
    assert out.getvalue().count("not a choice") == bad


def test_the_prompt_gives_up_at_the_end_of_input():
    def eof(_):
        raise EOFError
    assert cli.pick_base([{"name": "a/b", "installed": True}], "m", "b", out=Pipe(), ask=eof) is None


def test_a_piped_run_never_prompts(conf, monkeypatch):
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    code, out, rec = run_base(conf, [], BaseWorld(found=FOUND))       # stdout is a pipe
    assert code == 2 and "choose a number" not in out


def test_a_dry_run_with_base_does_not_install_the_bundle(conf):
    w = BaseWorld(found=FOUND)
    code, out, rec = run_base(conf, ["--dry-run", "--base", BUNDLE], w)
    assert w.pulled == [] and rec.argv is None


def test_a_base_flag_that_still_blocks_is_not_followed_by_a_prompt(conf):
    w = BaseWorld(found=FOUND)
    code, out, rec = run_base(conf, ["--base", GEMMA], w, chooser=lambda c, m, b: pytest.fail("asked"))
    assert code == 2 and rec.argv is None and w.pulled == []


# ---- --mode: one box or the lab, per run -------------------------------------------------------

@pytest.fixture
def lab_conf(tmp_path):
    p = tmp_path / "bringup.toml"
    p.write_text(f'mode = "lab"\nruns_root = "{tmp_path}/runs"\nhf_home = "{tmp_path}/hf"\n'
                 f'cache_root = "{tmp_path}/cache"\n'
                 '[coder]\ntarget = "pkg/coder"\nport = 8000\nchips = 4\n'
                 '[lab]\nhost = "node4"\nroot = "/srv/orchard"\n')
    return p


def test_a_lab_config_runs_on_the_lab_and_says_so(lab_conf):
    code, out, rec = run(lab_conf, ["bringup", "Cloudflare/clef", "--dry-run"])
    assert code == 0 and "--lab node4" in out and "mode: lab (node4)" in out


def test_mode_local_overrides_a_lab_config_for_one_run(lab_conf):
    code, out, rec = run(lab_conf, ["bringup", "Cloudflare/clef", "--dry-run", "--mode", "local"])
    assert code == 0 and "--lab" not in out.split("command", 1)[1] and "mode: local" in out


def test_mode_lab_without_a_lab_table_is_refused(conf, capsys):
    code, out, rec = run(conf, ["bringup", "Cloudflare/clef", "--dry-run", "--mode", "lab"])
    assert code == 2 and rec.argv is None and "[lab]" in capsys.readouterr().err



def test_caches_lists_the_tensor_caches_under_the_cache_root(conf, tmp_path):
    d = tmp_path / "cache" / "org--m" / "1chip-x" / "tt_cache"
    d.mkdir(parents=True)
    (d / ".orchard-model").write_text("org/m@abc")
    code, out, rec = run(conf, ["caches"])
    assert code == 0 and "org/m@abc" in out and rec.argv is None



def test_caches_takes_its_own_flags(conf, tmp_path):
    code, out, rec = run(conf, ["caches", "--prune"])
    assert code == 2                                   # --prune needs --older-than; the flag reached caches


def test_coder_status_with_nothing_kept(conf, tmp_path):
    code, out, rec = run(conf, ["coder", "status"])
    assert code == 0 and "no coder is kept up" in out and rec.argv is None
