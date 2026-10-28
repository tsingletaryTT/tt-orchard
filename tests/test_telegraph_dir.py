"""The ledger directory is a `[telegraph]` key (so the web page and the command read the same ledger), and the ledger
reader survives bytes that are not UTF-8."""
import json
from pathlib import Path

import pytest

from ledger_io import seed_ledger
from orchard import telegraph as tg
from test_telegraph import table


def test_dir_is_a_key_and_a_relative_one_is_against_the_config_directory(tmp_path):
    cfg = tg.load(table(dir="spend/ledger"), base=tmp_path)
    assert cfg.dir == (tmp_path / "spend" / "ledger")
    assert tg.load(table(dir=str(tmp_path / "abs")), base=Path("/elsewhere")).dir == tmp_path / "abs"
    assert tg.load(table(), base=tmp_path).dir is None


def test_dir_expands_the_home_marker(tmp_path):
    assert tg.load(table(dir="~/x"), base=tmp_path).dir == Path("~/x").expanduser()


@pytest.mark.parametrize("bad", [3, True, "", ["a"]])
def test_a_dir_that_is_not_a_path_is_refused(tmp_path, bad):
    with pytest.raises(tg.ConfigError, match="telegraph.dir"):
        tg.load(table(dir=bad), base=tmp_path)


def test_a_disabled_table_keeps_its_dir_so_past_spend_can_still_be_shown(tmp_path):
    assert tg.load({"enabled": False, "dir": "old"}, base=tmp_path).dir == tmp_path / "old"
    assert tg.load({"enabled": False, "dir": 7}, base=tmp_path).dir is None      # not checked further when off


def test_the_command_uses_the_configured_dir_when_dir_is_not_given(tmp_path):
    import io
    from orchard import cli, telegraph_cmd as cmd
    cfg = tmp_path / "tg.toml"
    cfg.write_text('[telegraph]\nenabled = true\nendpoint = "https://llm.example.org/v1"\nmodels = ["a"]\n'
                   'dir = "ledger-here"\n')
    seed_ledger(tmp_path / "ledger-here",
                [{"call_id": "c1", "outcome": "ok", "tokens_in": 1, "tokens_out": 1, "model": "a", "problem": "p"}])
    out = io.StringIO()
    args = cli._parser().parse_args(["telegraph", "usage", "--telegraph-config", str(cfg)])
    assert cmd.main(args, out=out, environ={}, home=tmp_path / "home") == 0
    assert "calls: 1" in out.getvalue()
    out = io.StringIO()
    other = cli._parser().parse_args(["telegraph", "usage", "--telegraph-config", str(cfg), "--dir", str(tmp_path / "x")])
    cmd.main(other, out=out, environ={}, home=tmp_path / "home")
    assert "calls: 0" in out.getvalue()                      # --dir still wins

