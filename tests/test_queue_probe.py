# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""The queue and concurrency probe (orchard/queue_probe.py) against a fake server that speaks the audio job API: POST /v1/audio/jobs answers 202 with an
id, 429 when the queue is full and 503 while the model loads; GET /v1/audio/jobs/<id> gives the status and the queue position; DELETE drops a queued job;
GET /v1/health gives the queue. The fake has one worker thread (or, to show the probe notices, several) and a fixed service time. No chip and no model."""
import json
import os
import threading
import time
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from orchard import queue_probe as qp

SETTINGS = {"prompt": "p", "duration": 10.0, "steps": 8, "cfg_scale": 1.0, "seed": 42}


class FakeJobs:
    """The state behind the fake: jobs, a queue and `workers` threads that each take `service_s` seconds per job."""

    def __init__(self, service_s=0.15, max_waiting=2, workers=1, state="ok", reply="normal", never_finish=False, lifo=False, start_delay=0.0):
        self.service_s, self.max_waiting, self.state, self.reply, self.never_finish = service_s, max_waiting, state, reply, never_finish
        self.lifo, self.start_delay = lifo, start_delay
        self.stall = {}                                       # request kind -> seconds the handler sleeps before it answers ("get", "post", "health")
        self.lock = threading.Lock()
        self.queue: deque = deque()
        self.jobs: dict = {}
        self.running = 0
        self.deleted: list = []
        self.posts: list = []
        self.stop = False
        self.threads = [threading.Thread(target=self.work, daemon=True) for _ in range(workers)]
        for t in self.threads:
            t.start()

    def work(self):
        time.sleep(self.start_delay)
        while not self.stop:
            with self.lock:
                job = (self.queue.pop() if self.lifo else self.queue.popleft()) if self.queue else None
                if job:
                    job["status"] = "running"
                    self.running += 1
            if job is None:
                time.sleep(0.005)
                continue
            t0 = time.monotonic()
            while self.never_finish and not self.stop:
                time.sleep(0.01)
            time.sleep(self.service_s)
            with self.lock:
                job["status"], job["total_s"] = "done", round(time.monotonic() - t0, 3)
                self.running -= 1

    def submit(self, body):
        with self.lock:
            self.posts.append(body)
            if self.state != "ok":
                return 503, {"detail": "the model is loading"}
            if len(self.queue) + self.running >= self.max_waiting + 1:
                return 429, {"detail": "the queue is full"}
            job = {"id": uuid.uuid4().hex, "status": "queued", "settings": body}
            self.jobs[job["id"]] = job
            self.queue.append(job)
            return 202, {"id": job["id"], "status": "queued"}

    def view(self, job):
        with self.lock:
            pos = (list(self.queue).index(job) + 1) if job["status"] == "queued" else 0
            v = {"id": job["id"], "status": job["status"], "queue_position": pos, "stage": "", "settings": job["settings"], "error": "",
                 "cancel_requested": False, "stats": None}
            if job["status"] == "done":
                v["stats"] = {"audio_s": job["settings"]["duration"], "total_s": job["total_s"], "audio_format": "flac", "stages": {}}
            return v

    def health(self):
        with self.lock:
            return {"status": self.state, "model": "fake", "device_name": "fake", "queue_depth": len(self.queue), "queue_max": self.max_waiting,
                    "busy": self.running > 0, "limits": {"max_seconds": 120.0}, "defaults": {"duration": 10.0}}


def serve(fake):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, payload, raw=None):
            data = raw if raw is not None else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            time.sleep(fake.stall.get("health" if self.path == "/v1/health" else "get", 0))
            if fake.reply == "nan":
                return self.send(200, None, raw=b'{"status": NaN}')
            if fake.reply == "array":
                return self.send(200, None, raw=b"[1, 2]")
            if fake.reply == "huge":
                return self.send(200, None, raw=b'{"x": "' + b"a" * (3 << 20) + b'"}')
            if fake.reply == "garbage":
                return self.send(200, None, raw=b"<html>not json</html>")
            if self.path == "/v1/health":
                return self.send(200, fake.health())
            job = fake.jobs.get(self.path.rsplit("/", 1)[-1])
            return self.send(200, fake.view(job)) if job else self.send(404, {"detail": "no such job"})

        def do_POST(self):
            time.sleep(fake.stall.get("post", 0))
            if self.path != "/v1/audio/jobs" or self.headers.get("Content-Type") != "application/json":
                return self.send(415, {"detail": "send the job as application/json"})
            code, payload = fake.submit(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            if fake.reply == "nan" and code == 202:
                return self.send(202, None, raw=b'{"id": NaN}')
            if fake.reply == "badid" and code == 202:
                return self.send(202, {"id": "../../etc/passwd", "status": "queued"})
            self.send(code, payload)

        def do_DELETE(self):
            time.sleep(fake.stall.get("delete", 0))
            job = fake.jobs.get(self.path.rsplit("/", 1)[-1])
            if not job:
                return self.send(404, {"detail": "no such job"})
            with fake.lock:
                if job["status"] == "queued":
                    fake.queue.remove(job)
                    job["status"] = "cancelled"
                fake.deleted.append(job["id"])
            self.send(200, {"id": job["id"], "status": job["status"]})

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


@pytest.fixture
def make_server():
    started = []

    def make(**kw):
        fake = FakeJobs(**kw)
        httpd = serve(fake)
        started.append((fake, httpd))
        return fake, f"http://127.0.0.1:{httpd.server_address[1]}"
    yield make
    for fake, httpd in started:
        fake.stop = True
        httpd.shutdown()
        httpd.server_close()


def level(url, n, **kw):
    return qp.probe_level(url, n, SETTINGS, poll_s=0.02, timeout_s=kw.pop("timeout_s", 10.0), **kw)


# ---- what a level records -------------------------------------------------------------------------------------------------------------

def test_one_request_is_the_baseline_with_no_queue(make_server):
    fake, url = make_server()
    r = level(url, 1)
    assert r["concurrency"] == 1 and r["accepted"] == 1 and r["rejected"] == {} and r["max_queue_position"] <= 1
    assert r["jobs"][0]["final_status"] == "done" and r["jobs"][0]["server_total_s"] >= 0.15


def test_two_requests_at_once_queue_one_and_run_one_at_a_time_in_order(make_server):
    fake, url = make_server(service_s=0.25)
    r = level(url, 2)
    assert r["accepted"] == 2 and r["rejected"] == {} and r["max_queue_position"] in (1, 2)   # a job not yet taken by the worker still counts in the queue
    assert r["ran_one_at_a_time"] is True and "first_in_first_out" not in r
    waits = sorted(j["wait_s"] for j in r["jobs"])
    assert waits[0] < 0.2 and waits[1] >= 0.2                      # the second waited about one service time
    assert {j["final_status"] for j in r["jobs"]} == {"done"} and r["makespan_s"] >= 0.45


def test_four_requests_against_a_queue_of_two_get_one_429(make_server):
    fake, url = make_server(service_s=0.25, max_waiting=2)
    r = level(url, 4)
    assert r["accepted"] == 3 and r["rejected"] == {"429": 1} and r["max_queue_position"] in (2, 3)
    assert len(r["jobs"]) == 3 and {j["final_status"] for j in r["jobs"]} == {"done"}
    assert sorted(x["http_status"] for x in r["requests"]) == [202, 202, 202, 429]


def test_a_server_that_runs_jobs_in_parallel_is_reported_as_not_one_at_a_time(make_server):
    fake, url = make_server(service_s=0.4, workers=4, max_waiting=4)
    r = level(url, 4)
    assert r["accepted"] == 4 and r["ran_one_at_a_time"] is False


def test_a_server_still_loading_rejects_every_request_with_its_code_and_nothing_is_polled(make_server):
    fake, url = make_server(state="loading")
    r = level(url, 2)
    assert r["accepted"] == 0 and r["rejected"] == {"503": 2} and r["jobs"] == []


def test_the_request_is_the_documented_json_with_the_stated_settings(make_server):
    fake, url = make_server()
    level(url, 1)
    assert fake.posts == [SETTINGS]


def test_jobs_that_never_finish_time_out_and_the_queued_ones_are_cancelled(make_server):
    fake, url = make_server(never_finish=True, service_s=0.0)
    r = level(url, 3, timeout_s=0.6)
    assert r["timed_out"] is True
    assert len(fake.deleted) >= 1                                       # the queued ones were dropped so the server is left clean
    finals = {j["final_status"] for j in r["jobs"]}
    assert "done" not in finals


def test_a_proxy_named_in_the_environment_is_not_used(make_server, monkeypatch):
    fake, url = make_server()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    assert level(url, 1)["accepted"] == 1


@pytest.mark.parametrize("reply", ["nan", "array", "huge", "garbage"])
def test_a_reply_that_is_not_a_json_object_is_recorded_as_a_problem_and_does_not_crash(make_server, reply):
    fake, url = make_server(reply=reply)
    r = level(url, 1, timeout_s=0.5)
    assert r["problems"], r
    assert r["accepted"] == 0 or r["jobs"][0]["final_status"] in ("unknown", "timeout")


def test_a_reply_over_the_size_cap_is_named_as_too_large(make_server):
    fake, url = make_server(reply="huge")
    r = level(url, 1, timeout_s=5.0)
    assert any("larger than" in p for p in r["problems"]), r["problems"]


def test_a_job_whose_status_cannot_be_read_is_called_unknown_after_a_few_tries_and_does_not_wait_for_the_timeout(make_server):
    fake, url = make_server(reply="garbage")
    r = level(url, 1, timeout_s=30.0)
    assert r["timed_out"] is False and r["jobs"][0]["final_status"] == "unknown"


def test_an_id_that_could_climb_out_of_the_path_is_not_followed(make_server):
    fake, url = make_server(reply="badid")
    r = level(url, 1, timeout_s=2.0)
    assert r["accepted"] == 0 and r["rejected"] == {"202 unusable": 1} and r["problems"]


def test_requests_sent_at_once_are_not_judged_for_order(make_server):
    """Their order of arrival is not known, and the positions seen after a job starts cannot give one."""
    fake, url = make_server(lifo=True, start_delay=0.3, service_s=0.1, max_waiting=3)
    r = level(url, 3)
    assert r["accepted"] == 3 and "first_in_first_out" not in r


def order(url, **kw):
    return qp.probe_order(url, SETTINGS, poll_s=0.02, timeout_s=kw.pop("timeout_s", 10.0), **kw)


def test_jobs_sent_one_after_another_that_start_in_that_order_are_first_in_first_out(make_server):
    fake, url = make_server(start_delay=0.3, service_s=0.1, max_waiting=3)
    r = order(url)
    assert r["accepted"] == 3 and r["first_in_first_out"] is True and r["refused"] == {}
    assert len(r["submit_order"]) == 3 and r["start_s_in_submit_order"] == sorted(r["start_s_in_submit_order"])


def test_a_last_in_first_out_queue_is_reported_as_not_first_in_first_out(make_server):
    fake, url = make_server(lifo=True, start_delay=0.3, service_s=0.1, max_waiting=3)
    r = order(url)
    assert r["accepted"] == 3 and r["first_in_first_out"] is False


def test_the_order_is_unknown_when_the_jobs_do_not_run_to_the_end(make_server):
    fake, url = make_server(never_finish=True, service_s=0.0, max_waiting=3)
    r = order(url, timeout_s=0.5)
    assert r["first_in_first_out"] is None and r["timed_out"] is True


def test_the_order_is_unknown_when_only_some_of_the_jobs_ended(make_server):
    fake, url = make_server(service_s=0.3, max_waiting=3)
    r = order(url, timeout_s=0.75)                    # jobs end at about 0.3, 0.6 and 0.9 s
    assert r["timed_out"] is True and r["first_in_first_out"] is None


def test_a_stalled_submission_in_the_order_test_is_bounded_by_its_deadline(make_server):
    fake, url = make_server()
    fake.stall["post"] = 8.0
    t0 = time.monotonic()
    r = order(url, timeout_s=0.8)
    assert time.monotonic() - t0 < 4.0 and r["accepted"] == 0 and r["refused"] == {"no reply": 3}


def test_the_order_is_unknown_with_one_job(make_server):
    fake, url = make_server()
    assert order(url, n=1)["first_in_first_out"] is None


def test_the_order_test_records_a_refused_job_by_its_code(make_server):
    fake, url = make_server(max_waiting=0, service_s=0.4)
    r = order(url)
    assert r["accepted"] == 1 and r["refused"] == {"429": 2}


# ---- deadlines ----------------------------------------------------------------------------------------------------------------------------

def test_a_server_that_stalls_every_status_request_cannot_hold_the_level_past_its_deadline(make_server):
    fake, url = make_server()
    fake.stall["get"] = 8.0
    t0 = time.monotonic()
    r = level(url, 4, timeout_s=0.8)
    assert time.monotonic() - t0 < 4.0, "the level waited on stalled requests past its deadline"
    assert r["timed_out"] is True or all(j["final_status"] == "unknown" for j in r["jobs"])


def test_a_server_that_stalls_the_submission_does_not_hold_the_level_past_its_deadline(make_server):
    fake, url = make_server()
    fake.stall["post"] = 8.0
    t0 = time.monotonic()
    r = level(url, 2, timeout_s=0.8)
    assert time.monotonic() - t0 < 4.0 and r["accepted"] == 0 and r["rejected"] == {"no reply": 2}


def test_waiting_for_idle_is_bounded_by_its_timeout_even_when_the_health_check_stalls(make_server):
    fake, url = make_server()
    fake.stall["health"] = 8.0
    t0 = time.monotonic()
    with pytest.raises(qp.ProbeError):
        qp.wait_idle(url, 0.8, 0.02)
    assert time.monotonic() - t0 < 4.0


def test_the_cleanup_of_queued_jobs_has_its_own_small_timeout_and_budget(make_server, monkeypatch):
    fake, url = make_server(never_finish=True, service_s=0.0, max_waiting=6)
    fake.stall["delete"] = 8.0
    monkeypatch.setattr(qp, "CLEANUP_TIMEOUT_S", 0.5)
    monkeypatch.setattr(qp, "CLEANUP_BUDGET_S", 0.6)
    t0 = time.monotonic()
    r = level(url, 6, timeout_s=0.4)                  # one job runs, five wait; five stalled DELETEs of 0.5 s each would take 2.5 s
    assert r["timed_out"] is True
    assert time.monotonic() - t0 < 2.0, "the clean-up went past its budget"


# ---- refusals ---------------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("url", ["http://example.com:20100", "http://10.0.0.5:20100", "ftp://127.0.0.1:20100", "http://127.0.0.1:8000", "http://127.0.0.1:8100",
                                 "http://127.0.0.1:8103", "http://127.0.0.1:8780", "http://localhost", "127.0.0.1:20100", "http://user:pw@127.0.0.1:20100"])
def test_a_url_that_is_not_a_local_server_on_an_allowed_port_is_refused(url):
    with pytest.raises(qp.ProbeError):
        qp.check_url(url)


@pytest.mark.parametrize("url", ["http://127.0.0.1:20100", "http://localhost:20100", "http://[::1]:20100", "http://127.0.0.1:20100/"])
def test_a_local_url_on_a_free_port_is_accepted(url):
    assert qp.check_url(url).startswith("http://")


@pytest.mark.parametrize("n", [0, -1, 17, True, 1.5])
def test_a_bad_concurrency_is_refused(n):
    with pytest.raises(qp.ProbeError):
        qp.check_levels([n])


def test_levels_default_to_one_two_and_four():
    assert qp.DEFAULT_LEVELS == (1, 2, 4)


# ---- the whole probe and the file ------------------------------------------------------------------------------------------------------

def test_the_probe_runs_each_level_waits_for_idle_between_them_and_writes_the_file(make_server, tmp_path):
    fake, url = make_server(service_s=0.1, max_waiting=2)
    out = tmp_path / "probe.json"
    rec = qp.run_probe(url, SETTINGS, levels=(1, 2, 4), poll_s=0.02, timeout_s=10.0, out=out)
    assert [lv["concurrency"] for lv in rec["levels"]] == [1, 2, 4]
    assert rec["health_before"]["queue_max"] == 2 and rec["health_after"]["queue_depth"] == 0
    assert rec["poll_s"] == 0.02 and "resolution" in rec["limits"][0]
    assert rec["order"]["first_in_first_out"] is True and rec["order"]["accepted"] == 3
    assert json.loads(out.read_text())["levels"][2]["rejected"] == {"429": 1}
    assert rec["levels"][1]["accepted"] == 2                    # the earlier level did not leave the server busy for the next


def test_the_probe_refuses_a_server_that_says_it_is_not_ready(make_server):
    fake, url = make_server(state="loading")
    with pytest.raises(qp.ProbeError, match="loading"):
        qp.run_probe(url, SETTINGS, levels=(1,), poll_s=0.02, timeout_s=5.0)
    assert fake.posts == []


def test_the_probe_waits_for_a_busy_server_before_the_first_level(make_server):
    fake, url = make_server(service_s=0.6)
    fake.submit(SETTINGS)                                  # work that is already running when the probe starts
    time.sleep(0.1)
    rec = qp.run_probe(url, SETTINGS, levels=(1,), poll_s=0.02, timeout_s=10.0)
    assert rec["levels"][0]["jobs"][0]["wait_s"] < 0.4, rec["levels"][0]["jobs"][0]


def test_a_server_that_never_goes_idle_is_a_probe_error(make_server):
    fake, url = make_server(never_finish=True, service_s=0.0)
    fake.submit(SETTINGS)
    time.sleep(0.1)
    with pytest.raises(qp.ProbeError, match="idle"):
        qp.run_probe(url, SETTINGS, levels=(1,), poll_s=0.02, timeout_s=0.5)
    assert len(fake.posts) == 1                            # the probe sent nothing of its own


def test_the_probe_refuses_to_overwrite_a_result_file(make_server, tmp_path):
    fake, url = make_server()
    out = tmp_path / "probe.json"
    out.write_text("{}")
    with pytest.raises(qp.ProbeError, match="exists"):
        qp.run_probe(url, SETTINGS, levels=(1,), poll_s=0.02, timeout_s=5.0, out=out)
    assert out.read_text() == "{}" and fake.posts == []


def test_a_server_that_is_down_is_a_probe_error_naming_the_health_check():
    with pytest.raises(qp.ProbeError, match="health"):
        qp.run_probe("http://127.0.0.1:1", SETTINGS, levels=(1,), poll_s=0.02, timeout_s=1.0)


def test_the_host_load_average_is_recorded_for_each_level(make_server):
    fake, url = make_server()
    r = level(url, 1)
    assert "host_loadavg_1m_at_start" in r and "host_loadavg_1m_at_end" in r


def test_the_command_line_writes_the_file(make_server, tmp_path, capsys):
    fake, url = make_server(service_s=0.05)
    out = tmp_path / "p.json"
    code = qp.main(["--url", url, "--out", str(out), "--levels", "1,2", "--poll-s", "0.02", "--timeout-s", "10", "--prompt", "p", "--duration", "10",
                    "--steps", "8", "--cfg-scale", "1.0", "--seed", "42"])
    assert code == 0 and [lv["concurrency"] for lv in json.loads(out.read_text())["levels"]] == [1, 2]
    assert "concurrency 2" in capsys.readouterr().out
