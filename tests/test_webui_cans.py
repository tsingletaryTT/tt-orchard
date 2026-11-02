# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The Cans route of the web page (GET /api/cans, orchard/webui.py, orchard/cans_view.py).

The route is read only, answers on loopback names only, returns no path of this machine, takes no input that reaches the file system, and
gives a row (never a 500) for a can with a missing or broken piece."""
import json
from pathlib import Path

import pytest

from cans_fixture import add_spec, by_name, dump, make_checkout, passed_boot, record, stage_bundle
from orchard import can, webui
from orchard import cans_view
from test_webui import HOME, Served, make_app


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "orchard-runs"
    r.mkdir()
    return r


def app_for(root, tmp_path, *, cans=None, environ=None):
    """A WebApp whose specs come from a temporary checkout (not the real config folder) and whose cans folder is `cans`."""
    app = make_app(root)
    checkout = make_checkout(tmp_path)
    app.environ = environ if environ is not None else {}
    app.cans_specs_dir, app.cans_base = checkout / "config", checkout
    if cans is not None:
        app.cans_dir = Path(cans)
    return app, checkout


def get(app, path="/api/cans"):
    with Served(app) as s:
        code, _, body = s.request("GET", path)
    return code, json.loads(body)


def test_the_route_lists_the_specs_with_their_facts(root, tmp_path):
    app, checkout = app_for(root, tmp_path)
    add_spec(checkout, "toy")
    code, data = get(app)
    assert code == 200 and data["read_only"] is True
    row = by_name(data, "toy-blackhole")
    assert row["kind"] == "tt-dit-server" and row["bundle"]["state"] == "not-staged"
    assert data["hub"]["text"] == "not checked here"


def test_the_default_cans_folder_is_cans_under_the_runs_folder(root, tmp_path):
    app, checkout = app_for(root, tmp_path)
    name = add_spec(checkout, "toy")
    stage_bundle(root / "cans", name)
    assert by_name(get(app)[1], name)["bundle"]["state"] == "staged"


def test_the_environment_variable_names_the_cans_folder(root, tmp_path):
    other = tmp_path / "elsewhere"
    app, checkout = app_for(root, tmp_path, environ={"ORCHARD_CANS_DIR": str(other)})
    name = add_spec(checkout, "toy")
    stage_bundle(other, name)
    stage_bundle(root / "cans", name, extra={"only-in-default.txt": "x"})
    assert by_name(get(app)[1], name)["sums"]["text"] == "all 6 files match"


def test_the_answer_has_no_path_and_redacts_the_home_directory(root, tmp_path):
    app, checkout = app_for(root, tmp_path)
    name = add_spec(checkout, "toy")
    b = stage_bundle(root / "cans", name)
    record(root / "cans", name, b, gate=can.gate_record([f"scrub: {HOME}/x/run.sh holds something"]))
    code, data = get(app)
    text = dump(data)
    assert HOME not in text and "~/x/run.sh" in text
    assert str(tmp_path) not in text and str(root) not in text


def test_a_query_string_cannot_choose_a_file(root, tmp_path):
    app, checkout = app_for(root, tmp_path)
    name = add_spec(checkout, "toy")
    stage_bundle(root / "cans", name)
    code, data = get(app, "/api/cans?name=../../etc&dir=/etc&cans=/etc")
    assert code == 200 and by_name(data, name)["bundle"]["state"] == "staged"


def test_a_can_with_everything_broken_is_a_row_not_an_error_page(root, tmp_path):
    app, checkout = app_for(root, tmp_path)
    name = add_spec(checkout, "toy")
    b = stage_bundle(root / "cans", name)
    (b / can.SUMS_FILE).unlink()
    (b / "README.md").unlink()
    (root / "cans" / f"{name}{can.STATE_SUFFIX}").write_text("{broken")
    (checkout / "config" / "can.bad.toml").write_text("[[[")
    code, data = get(app)
    assert code == 200
    row = by_name(data, name)
    assert (row["sums"]["state"], row["card"]["state"], row["gate"]["state"]) == ("missing", "missing", "unreadable")
    assert by_name(data, "bad")["spec"]["state"] == "refused"


def test_the_route_answers_loopback_hosts_only_and_takes_no_post(root, tmp_path):
    app, checkout = app_for(root, tmp_path)
    add_spec(checkout, "toy")
    with Served(app) as s:
        code, _, _ = s.request("GET", "/api/cans", host="evil.example")
        assert code == 403
        assert s.post("/api/cans", {})[0] in (404, 405)


def test_the_route_does_not_change_the_cans_folder(root, tmp_path):
    app, checkout = app_for(root, tmp_path)
    name = add_spec(checkout, "toy")
    stage_bundle(root / "cans", name)

    def snap():
        return sorted((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in tmp_path.rglob("*"))
    before = snap()
    get(app)
    assert snap() == before


def test_one_view_serves_every_request_and_is_built_once(root, tmp_path):
    app, checkout = app_for(root, tmp_path)
    add_spec(checkout, "toy")
    first = app.cans_view()
    assert app.cans_view() is first and isinstance(first, cans_view.CansView)


def test_the_real_config_folder_is_the_default_source_of_specs(root):
    app = make_app(root)
    assert app.cans_specs_dir == webui.CHECKOUT / "config" and app.cans_base == webui.CHECKOUT
    code, data = get(app)
    assert code == 200 and {"acestep", "fiona", "granite", "minicpm3"} <= {c["slug"] for c in data["cans"]}
