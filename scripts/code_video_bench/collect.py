"""Summarize one finished run of run_bench.sh.

usage: collect.py RUN_DIR WIDTH HEIGHT FPS MIN_DUR MAX_DUR [IMAGE]

Reads the run's logs (Qwen Code's stream-json output, the server metric samples) and checks out/final.mp4 with ffprobe, ffmpeg and the
skill's own check_video.sh, both inside a read-only container that has no network. Writes RUN_DIR/logs/summary.json and prints it.
Every figure comes from a file in RUN_DIR/logs or RUN_DIR/task; the summary names which.

The checker is run from a trusted copy of the skill, mounted read-only: the directory in the SKILL_SRC environment variable, or else the one
run_bench.sh recorded in RUN_DIR/logs/skill_src (the logs directory is never mounted into the model's container). The task's own copy of
the skill is not used for the check, because the model can edit it; the summary lists the skill files the run changed (hashes recorded by
run_bench.sh in RUN_DIR/logs/skill.sha256). A figure that is not measured is null, never zero.
Isolation: start it with collect.sh, which scrubs every PYTHON* variable, leaves the caller's working directory and runs `python3 -I -S`
(no current directory, no PYTHON* variable, no user site and no site-packages on the import path). The script refuses to run when it was not
started isolated and does not try to re-run itself: code that python runs at startup (a sitecustomize or usercustomize module on a path the
model could write, for example) has already run by the time the script's first line does. Inside the containers the checker runs with its
working directory set to the trusted copy of the skill (the unchanged files, mounted read-only), the video named by its absolute path, and
PYTHONSAFEPATH, PYTHONNOUSERSITE and an empty PYTHONPATH, so a file the model wrote into the task directory (a json.py, say) cannot be
imported by the checker. Every other container call runs from an empty read-only scratch directory.
Reading the model's files: the task directory is the model's. Every read of it goes through one anchored reader (class Beneath) that opens
each directory with O_NOFOLLOW from a descriptor of the task directory, looks at each name with lstat first, and takes only regular files. A
link anywhere on a path is refused and listed in `refused_paths` (path and reason), never followed and never reported by its target's size. A
file that is a FIFO, a device or a dangling link is skipped the same way; nothing the model left is opened unless it is a regular file.
Bounds: the agent's stream and the metric samples are read line by line to a byte cap (CVBENCH_STREAM_CAP_BYTES, default 512 MiB), a line
over CVBENCH_LINE_CAP_BYTES (4 MiB) is skipped and counted, and the summary says what was read (`stream.read`). A container call is read while
it runs and stopped at its output cap (stdout CVBENCH_OUTPUT_CAP_BYTES, 64 MiB by default; stderr 1 MiB at most) and at its deadline; the
client's process group is killed and the container removed by its id. A removal that fails or times out (CVBENCH_CLEANUP_TIMEOUT_S, 60) is
listed in `cleanup_failures` with the container id and the reason, and collection goes on.
Failures: a measurement that could not be taken (a container that timed out or that docker refused, ffmpeg that failed or wrote anything but
framemd5 records, a hash count that disagrees with ffprobe's frame count) is reported as null with the reason in an `error` field, and
`measurement_failures` lists and counts them. A failed measurement is never a figure.
Environment: CVBENCH_DOCKER_TIMEOUT_S (default 900) bounds each container; a container that overruns is removed by its id."""
import os, sys
if not sys.flags.isolated:
    sys.stderr.write("collect.py refuses to run without isolation: start it with scripts/code_video_bench/collect.sh, or with `python3 -I -S collect.py ...`. "
                     "Startup code the interpreter ran before this line (sitecustomize, usercustomize, PYTHONSTARTUP, a PYTHONPATH module) cannot be undone from "
                     "inside the script.\n")
    raise SystemExit(2)
import hashlib, json, re, selectors, shlex, signal, stat, subprocess, tempfile, time, datetime

def env_number(name, default):
    """A positive finite number from the environment, else the default (a bad value is not an error here: the bound stays)."""
    try:
        v = float(os.environ[name])
    except (KeyError, ValueError):
        return default
    return v if 0 < v < float("inf") else default


STREAM_CAP = int(env_number("CVBENCH_STREAM_CAP_BYTES", 512 << 20))     # the most of the agent's stream (and of the metric samples) read
LINE_CAP = int(env_number("CVBENCH_LINE_CAP_BYTES", 4 << 20))           # a longer line is skipped and counted
run, W, H, FPS, DMIN, DMAX = sys.argv[1:7]
run = os.path.abspath(run)                           # docker needs absolute mount destinations and working directories
image = sys.argv[7] if len(sys.argv) > 7 else "cvbench:0.1"
task, logs = os.path.join(run, "task"), os.path.join(run, "logs")
out = {"run": os.path.basename(run)}


def iso(path):
    return datetime.datetime.strptime(rd(path) or "", "%Y-%m-%dT%H:%M:%SZ")


def rd(path):
    """A small file of the run's own logs directory (written by run_bench.sh, never mounted into the model's container), at most 1 MiB."""
    try:
        with open(path) as f:
            return f.read(1 << 20).strip()
    except OSError:
        return None


def read_lines(path, info, total_cap=None):
    """Yield the lines of `path` one at a time, without ever holding the file: at most `total_cap` bytes (default STREAM_CAP) are read, and a
    line longer than LINE_CAP is skipped and counted. `info` is filled in: bytes_read, truncated (the file had more), long_lines_skipped.
    The file's content comes from the model (its words are in the stream), so its size is bounded and its text is only parsed as JSON."""
    total_cap = STREAM_CAP if total_cap is None else total_cap
    info.update(present=False, bytes_read=0, truncated=False, long_lines_skipped=0)
    try:
        f = open(path, "rb")
    except OSError:
        return
    info["present"] = True
    with f:
        while info["bytes_read"] < total_cap:
            raw = f.readline(LINE_CAP + 1)
            if not raw:
                return
            info["bytes_read"] += len(raw)
            if len(raw) > LINE_CAP and not raw.endswith(b"\n"):
                info["long_lines_skipped"] += 1
                while info["bytes_read"] < total_cap:               # drop the rest of this line, a megabyte at a time
                    more = f.readline(1 << 20)
                    info["bytes_read"] += len(more)
                    if not more or more.endswith(b"\n"):
                        break
                continue
            yield raw.decode("utf-8", "replace").rstrip("\n")
        info["truncated"] = bool(f.read(1))


# ---- wall time and exit ------------------------------------------------------------------------------------------------------
if rd(f"{logs}/started_at") and rd(f"{logs}/ended_at"):
    out["wall_s"] = int((iso(f"{logs}/ended_at") - iso(f"{logs}/started_at")).total_seconds())
out["exit_code"] = rd(f"{logs}/exit_code")

# ---- the agent's stream ------------------------------------------------------------------------------------------------------
st = {"assistant_messages": 0, "assistant_messages_with_usage": 0, "tool_calls": {}, "tool_errors": 0, "tool_results": 0, "compaction_events": 0, "thinking_chars": 0,
      "text_chars": 0, "usage_sum": {"input_tokens": 0, "output_tokens": 0}, "result": None, "web_fetch_calls": 0, "image_tool_calls": {}}
errors_sample = []
last_in = 0
stream_info = {}
for line in read_lines(f"{logs}/stream.jsonl", stream_info):
    try:
        d = json.loads(line)
    except (ValueError, RecursionError):
        continue
    if not isinstance(d, dict):
        continue
    t = d.get("type")
    kinds = " ".join(str(x).lower() for x in (t, d.get("subtype"), (d.get("event") or {}).get("type")))
    if t != "system" and re.search(r"compact|compress", kinds):       # an event of that kind; the init event only lists slash commands
        st["compaction_events"] += 1
    if t == "assistant":
        m = d.get("message", {})
        st["assistant_messages"] += 1
        u = m.get("usage") or {}
        n_in = u.get("input_tokens", 0) or 0
        if n_in and last_in and n_in < 0.7 * last_in:                 # the prompt shrank by 30% or more between two calls
            st["context_drops"] = st.get("context_drops", 0) + 1
        if n_in:
            last_in = n_in
            st["max_prompt_tokens"] = max(st.get("max_prompt_tokens", 0), n_in)
        if u:
            st["assistant_messages_with_usage"] = st.get("assistant_messages_with_usage", 0) + 1
        st["usage_sum"]["input_tokens"] += n_in
        st["usage_sum"]["output_tokens"] += u.get("output_tokens", 0) or 0
        for b in m.get("content", []):
            if b.get("type") == "tool_use":
                n = b.get("name"); st["tool_calls"][n] = st["tool_calls"].get(n, 0) + 1
                if n == "web_fetch":
                    st["web_fetch_calls"] += 1
                if n in ("zoom_image", "read_file"):
                    inp = json.dumps(b.get("input"))
                    if re.search(r"\.(png|jpg|jpeg|webp)", inp):
                        st["image_tool_calls"][n] = st["image_tool_calls"].get(n, 0) + 1
            elif b.get("type") == "thinking":
                st["thinking_chars"] += len(b.get("thinking", ""))
            elif b.get("type") == "text":
                st["text_chars"] += len(b.get("text", ""))
    elif t == "user":
        for b in (d.get("message", {}).get("content") or []):
            if isinstance(b, dict) and b.get("type") == "tool_result":
                st["tool_results"] += 1
                if b.get("is_error"):
                    st["tool_errors"] += 1
                    if len(errors_sample) < 8:
                        errors_sample.append(str(b.get("content"))[:200])
    elif t == "result":
        st["result"] = {k: d.get(k) for k in ("subtype", "is_error", "duration_ms", "num_turns", "usage", "result")}
        if isinstance(st["result"].get("result"), str):
            st["result"]["result"] = st["result"]["result"][:400]
st["tool_error_samples"] = errors_sample
st["read"] = stream_info
out["stream"] = st

# ---- server load (metric samples every 15 s) ------------------------------------------------------------------------------------
def js(path):
    text = rd(path)
    try:
        return json.loads(text) if text else None
    except (ValueError, RecursionError):
        return None

s0, s1 = js(f"{logs}/metrics-start.json"), js(f"{logs}/metrics-end.json")
samples = []
for x in read_lines(f"{logs}/metrics.jsonl", {}, total_cap=64 << 20):
    try:
        s_ = json.loads(x) if x.strip() else None
    except (ValueError, RecursionError):
        s_ = None
    if isinstance(s_, dict):
        samples.append(s_)
def gauge(name):                                      # only samples that carry the gauge count; a sample without it is no measurement
    return [s[name] for s in samples if "error" not in s and isinstance(s.get(name), (int, float)) and not isinstance(s.get(name), bool)]

run_vals, wait_vals = gauge("vllm:num_requests_running"), gauge("vllm:num_requests_waiting")
srv = {"samples": len(samples), "samples_with_error": sum(1 for s in samples if "error" in s),
       "samples_with_running_gauge": len(run_vals), "samples_with_waiting_gauge": len(wait_vals),
       "max_requests_running": max(run_vals) if run_vals else None,
       "samples_with_more_than_one_running": sum(1 for v in run_vals if v > 1),
       "max_requests_waiting": max(wait_vals) if wait_vals else None}
if s0 and s1:
    for k in ("vllm:prompt_tokens_total", "vllm:generation_tokens_total"):
        if k in s0 and k in s1:
            srv[k.split(":")[1] + "_delta"] = int(s1[k] - s0[k])
out["server"] = srv

# ---- reading the model's files: one anchored reader that never follows a link ------------------------------------------------------
O_DIR = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
O_FILE = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC       # NONBLOCK: opening a FIFO the model left must not wait for a writer
refused_paths = []                                       # {"path", "reason"}: what was not read and why (at most 20)


class Refused(Exception):
    def __init__(self, path, reason):
        super().__init__(f"{path}: {reason}")
        self.path, self.reason = path, reason


def note_refused(path, reason):
    entry = {"path": path, "reason": reason}
    if len(refused_paths) < 20 and entry not in refused_paths:
        refused_paths.append(entry)


class Beneath:
    """Reads under one directory (the task directory, which the model controls) without following a link anywhere. Every directory on a path
    is opened with O_NOFOLLOW relative to the descriptor of the one before it, after an lstat that tells a link or a non-directory from a
    directory; the last name is looked at with lstat or opened with O_NOFOLLOW|O_NONBLOCK and then checked with fstat. A path is never resolved
    and then opened. What is refused is listed through note_refused. A missing name is not a refusal: it returns None."""

    def __init__(self, root):
        self.fd = os.open(root, O_DIR)                    # OSError when the root is missing or is itself a link: the caller reads nothing

    def close(self):
        os.close(self.fd)

    @staticmethod
    def parts(rel):
        ps = [p for p in rel.split("/") if p]
        if any(p in (".", "..") for p in ps):
            raise Refused(rel, "a path part that is not a plain name")
        return ps

    def _step(self, fd, name, rel):
        st_ = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if stat.S_ISLNK(st_.st_mode):
            raise Refused(rel, "a symbolic link")
        if not stat.S_ISDIR(st_.st_mode):
            raise Refused(rel, "not a directory")
        try:
            return os.open(name, O_DIR, dir_fd=fd)
        except OSError:
            raise Refused(rel, "changed while it was opened") from None

    def _dir(self, names, prefix=""):
        """(fd, owned) of the directory below the root named by `names`; the caller closes it when owned."""
        fd, owned = self.fd, False
        try:
            for i, n in enumerate(names):
                nfd = self._step(fd, n, prefix + "/".join(names[:i + 1]))
                if owned:
                    os.close(fd)
                fd, owned = nfd, True
            return fd, owned
        except BaseException:
            if owned:
                os.close(fd)
            raise

    def _parent(self, rel):
        """(parent fd, owned, last name) of rel, or None when something on the way is missing or refused (refusals are noted)."""
        try:
            ps = self.parts(rel)
            if not ps:
                return None
            fd, owned = self._dir(ps[:-1])
            return fd, owned, ps[-1]
        except FileNotFoundError:
            return None
        except Refused as e:
            note_refused(e.path, e.reason)
            return None
        except OSError as e:
            note_refused(rel, type(e).__name__)
            return None

    def regular(self, rel):
        """The lstat result of `rel` when it is a regular file, else None; a link, a FIFO or a device is refused and noted."""
        got = self._parent(rel)
        if got is None:
            return None
        fd, owned, name = got
        try:
            st_ = os.stat(name, dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError:
            return None
        except OSError as e:
            note_refused(rel, type(e).__name__)
            return None
        finally:
            if owned:
                os.close(fd)
        if stat.S_ISLNK(st_.st_mode):
            note_refused(rel, "a symbolic link")
            return None
        if not stat.S_ISREG(st_.st_mode):
            note_refused(rel, "not a regular file")
            return None
        return st_

    def digest(self, rel, cap=16 << 20):
        """sha256 of a regular file of at most `cap` bytes, or None for anything else (missing, a link, a FIFO, too large)."""
        got = self._parent(rel)
        if got is None:
            return None
        pfd, owned, name = got
        try:
            fd = os.open(name, O_FILE, dir_fd=pfd)
        except FileNotFoundError:
            return None
        except OSError as e:
            note_refused(rel, "a symbolic link" if e.errno == 40 else type(e).__name__)
            return None
        finally:
            if owned:
                os.close(pfd)
        try:
            st_ = os.fstat(fd)
            if not stat.S_ISREG(st_.st_mode):
                note_refused(rel, "not a regular file")
                return None
            if st_.st_size > cap:
                note_refused(rel, "larger than the read cap")
                return None
            h = hashlib.sha256()
            with os.fdopen(os.dup(fd), "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()
        finally:
            os.close(fd)

    def entries(self, rel, limit=5000):
        """[(name, lstat result)] for the names directly under `rel`, links and all (lstat, never stat), at most `limit`; [] when missing."""
        try:
            ps = self.parts(rel)
            fd, owned = self._dir(ps)
        except FileNotFoundError:
            return []
        except Refused as e:
            note_refused(e.path, e.reason)
            return []
        except OSError as e:
            note_refused(rel, type(e).__name__)
            return []
        found = []
        try:
            with os.scandir(fd) as it:
                for entry in it:
                    if len(found) >= limit:
                        note_refused(rel, f"more than {limit} entries; the rest are not listed")
                        break
                    try:
                        found.append((entry.name, entry.stat(follow_symlinks=False)))
                    except OSError:
                        continue                          # it vanished between the listing and the stat
        finally:
            if owned:
                os.close(fd)
        return found

    def walk(self, rel, max_depth=8, max_entries=5000):
        """[(path relative to `rel`, kind)] for everything below `rel` that is not a directory, kind "file", "link" or "other"; directories are
        opened by descriptor and never entered through a link; the walk stops at max_depth levels and max_entries names (both noted)."""
        out_ = []
        try:
            ps = self.parts(rel)
            top, top_owned = self._dir(ps)
        except FileNotFoundError:
            return out_
        except Refused as e:
            note_refused(e.path, e.reason)
            return out_
        except OSError as e:
            note_refused(rel, type(e).__name__)
            return out_
        stack = [(top, top_owned, "", 0)]
        count = 0
        try:
            while stack:
                fd, owned, prefix, depth = stack.pop()
                try:                                          # the descriptor stays open until the subdirectories below it have been opened
                    # At most the names the entry budget still allows, plus one to show that the directory has more: a model-made directory
                    # may hold millions of names, and sorting a full listing would use the host's memory before the limit is checked.
                    budget = max_entries - count + 1
                    listing = []
                    try:
                        with os.scandir(fd) as it:
                            for e in it:
                                if len(listing) >= budget:
                                    break
                                try:
                                    listing.append((e.name, e.stat(follow_symlinks=False)))
                                except OSError:
                                    continue                  # it vanished between the listing and the stat
                    except OSError:
                        pass                                  # keep what was listed before the failure
                    listing.sort(key=lambda x: x[0])
                    for name, st_ in listing:
                        count += 1
                        if count > max_entries:
                            note_refused(rel, f"more than {max_entries} entries; the rest are not listed")
                            return out_
                        path_ = prefix + name
                        if stat.S_ISLNK(st_.st_mode):
                            note_refused(f"{rel}/{path_}", "a symbolic link")
                            out_.append((path_, "link"))
                        elif stat.S_ISDIR(st_.st_mode):
                            if depth + 1 > max_depth:
                                note_refused(f"{rel}/{path_}", f"deeper than {max_depth} levels")
                                continue
                            try:
                                stack.append((self._step(fd, name, path_), True, path_ + "/", depth + 1))
                            except (Refused, OSError):
                                note_refused(f"{rel}/{path_}", "changed while it was opened")
                        elif stat.S_ISREG(st_.st_mode):
                            out_.append((path_, "file"))
                        else:
                            note_refused(f"{rel}/{path_}", "not a regular file")
                            out_.append((path_, "other"))
                finally:
                    if owned:
                        os.close(fd)
        finally:
            for fd, owned, *_ in stack:
                if owned:
                    os.close(fd)
        return out_


try:
    B = Beneath(task)
except OSError:
    B = None                                              # no task directory: nothing of the model's is read


# ---- the trusted copy of the skill and what the run changed in its own copy ------------------------------------------------------
def trusted_skill():
    """The directory of the unchanged skill: $SKILL_SRC, else the path run_bench.sh recorded in logs/skill_src. None when neither names a
    directory that holds scripts/check_video.sh. Both come from the operator, never from the task directory."""
    for cand in (os.environ.get("SKILL_SRC"), rd(f"{logs}/skill_src")):
        if cand and os.path.isfile(os.path.join(cand, "scripts", "check_video.sh")):
            return os.path.realpath(cand)
    return None


recorded = rd(f"{logs}/skill.sha256")
if recorded is None:
    out["skill_files_changed_by_run"] = None            # no record, so it is unknown and not "none"
else:
    copy_rel = ".qwen/skills/code-video"
    known, changed = set(), []
    for ln in recorded.splitlines():
        h, _, name = ln.partition("  ")
        name = name[2:] if name.startswith("./") else name
        if not h or not name:
            continue
        known.add(name)
        if B is None or ".." in name.split("/") or B.digest(f"{copy_rel}/{name}") != h:
            changed.append(name)
    out["skill_files_changed_by_run"] = sorted(changed)
    out["skill_files_added_by_run"] = sorted(p_ for p_, _kind in (B.walk(copy_rel) if B else []) if p_ not in known)

# ---- running things in a container, read while they run and bounded -----------------------------------------------------------
DOCKER_TIMEOUT_S = env_number("CVBENCH_DOCKER_TIMEOUT_S", 900.0)
CLEANUP_TIMEOUT_S = env_number("CVBENCH_CLEANUP_TIMEOUT_S", 60.0)
SCRATCH = "/scratch"                                   # an empty read-only tmpfs: the working directory of every call that needs none
PY_SCRUB = ("PYTHONSAFEPATH=1", "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=")
DOCKER_ERROR_CODES = (125, 126, 127)                    # docker's own failures: the daemon refused, the command could not be run or was not found
OUTPUT_CAP = int(env_number("CVBENCH_OUTPUT_CAP_BYTES", 64 << 20))      # the most one call may write on stdout (the frame records of ffmpeg)
PROBE_CAP = min(OUTPUT_CAP, 8 << 20)                    # ffprobe's JSON and the checker's text are small; a hostile file must not make them large
ERR_CAP = min(OUTPUT_CAP, 1 << 20)
cleanup_failures = []                                   # {"container", "reason"}: removals that failed or timed out, listed in the summary


def run_capped(argv, timeout, out_cap, err_cap):
    """Run `argv` like subprocess.run(capture_output=True), but read both pipes WHILE it runs and stop at a cap or a deadline: past `out_cap` bytes
    on stdout or `err_cap` on stderr, or after `timeout` seconds, the process group is killed (it was started in a group of its own). Returns
    {"rc", "stdout", "stderr", "timed_out", "over"} with `over` naming the stream that passed its cap (the text kept is cut at the cap). A
    descendant that keeps a pipe open after the process ended is bounded by the same deadline."""
    res = {"rc": None, "stdout": "", "stderr": "", "timed_out": False, "over": None}
    p = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    out_fd, err_fd = p.stdout.fileno(), p.stderr.fileno()
    streams = {out_fd: ("stdout", out_cap, bytearray()), err_fd: ("stderr", err_cap, bytearray())}
    sel = selectors.DefaultSelector()
    for f in (p.stdout, p.stderr):
        sel.register(f, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout
    try:
        while sel.get_map() and not res["over"]:
            left = deadline - time.monotonic()
            if left <= 0:
                res["timed_out"] = True
                break
            for key, _ in sel.select(min(left, 1.0)):
                name, cap, buf = streams[key.fd]
                chunk = os.read(key.fd, 1 << 16)
                if not chunk:
                    sel.unregister(key.fileobj)
                    continue
                room = cap - len(buf)
                buf += chunk[:max(room, 0)]
                if len(chunk) > room:
                    res["over"] = name
                    break
        if not (res["timed_out"] or res["over"]):
            try:                                          # both pipes are closed: the process should end now; the deadline still holds
                p.wait(timeout=max(deadline - time.monotonic(), 0.01))
            except subprocess.TimeoutExpired:
                res["timed_out"] = True
    finally:
        if p.poll() is None or res["timed_out"] or res["over"]:
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        sel.close()
        for f in (p.stdout, p.stderr):
            f.close()
        p.wait()
    res["rc"] = p.returncode
    res["stdout"] = bytes(streams[out_fd][2]).decode("utf-8", "replace")
    res["stderr"] = bytes(streams[err_fd][2]).decode("utf-8", "replace")
    return res


def remove_container(cid):
    """`docker rm -f <id>`, bounded by CLEANUP_TIMEOUT_S. A failure or a timeout is recorded in cleanup_failures with the id and the reason and
    never stops the collection. True when the container is gone."""
    try:
        r = run_capped(["docker", "rm", "-f", cid], CLEANUP_TIMEOUT_S, 64 << 10, 64 << 10)
    except OSError as exc:
        reason = f"could not run docker rm: {type(exc).__name__}"
    else:
        if r["timed_out"]:
            reason = f"docker rm timed out after {CLEANUP_TIMEOUT_S:g} s"
        elif r["over"]:
            reason = f"docker rm wrote more than 64 KiB on {r['over']}"
        elif r["rc"] != 0:
            reason = f"docker rm exited {r['rc']}: " + " ".join(r["stderr"].split())[:150]
        else:
            return True
    cleanup_failures.append({"container": cid[:12], "reason": reason})
    return False


def in_container(cmd, extra_mounts=(), workdir=SCRATCH, out_cap=PROBE_CAP, err_cap=ERR_CAP):
    """Run `sh -c cmd` in a read-only container with no network, the task directory mounted read-only, an empty scratch directory and a Python
    environment that imports nothing from the working directory. The container id is written to a --cidfile. The client's output is read while
    it runs and bounded (out_cap on stdout, err_cap on stderr, DOCKER_TIMEOUT_S in all); when the call overruns either, the client's process
    group is killed and the container is removed by that id, so the bound also covers the container's own life.
    Returns {"ok", "rc", "stdout", "stderr", "error", "text"}: `ok` is false when no command result exists (a timeout, an output over its cap, a
    docker error, or docker missing) and `error` says why; `rc` is the command's exit status when it ran, and a nonzero one is the caller's to judge."""
    res = {"ok": False, "rc": None, "stdout": "", "stderr": "", "error": None, "text": ""}
    with tempfile.TemporaryDirectory(prefix="cvbench-collect-") as tmp:
        cidfile = os.path.join(tmp, "cid")
        argv = ["docker", "run", "--rm", "--cidfile", cidfile, "--network", "none", "--read-only", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges", "--user", f"{os.getuid()}:{os.getgid()}", "--memory", "4g",
                "--tmpfs", f"{SCRATCH}:ro,size=1m", *[a for kv in PY_SCRUB for a in ("-e", kv)],
                "-v", f"{task}:{task}:ro", *[a for m in extra_mounts for a in ("-v", m)], "-w", workdir, image, "sh", "-c", cmd]
        try:
            r = run_capped(argv, DOCKER_TIMEOUT_S, out_cap, err_cap)
        except OSError as exc:
            res["error"] = f"could not run docker: {type(exc).__name__}"
            res["text"] = res["error"]
            return res
        if r["timed_out"] or r["over"]:
            cid = rd(cidfile)
            gone = remove_container(cid) if cid else False
            what = (f"timed out after {DOCKER_TIMEOUT_S:g} s" if r["timed_out"] else
                    f"{r['over']} exceeded {out_cap if r['over'] == 'stdout' else err_cap} bytes")
            res["error"] = what + (f"; container {cid[:12]} " + ("removed" if gone else "could not be removed (see cleanup_failures)") if cid
                                   else "; no container id was recorded")
            res["text"] = res["error"]
            return res
    res.update(rc=r["rc"], stdout=r["stdout"], stderr=r["stderr"], text=(r["stdout"] + r["stderr"]).strip())
    if r["rc"] in DOCKER_ERROR_CODES and r["stderr"].lstrip().startswith("docker"):
        res["error"] = f"docker error (exit {r['rc']}): " + " ".join(r["stderr"].split())[:200]
    else:
        res["ok"] = True
    return res


failures = []                                           # the measurements that could not be taken, in the order they were tried

# ---- the deliverables: only regular files that sit under the task directory without a link on the way ------------------------------
delivered = B is not None and B.regular("out/final.mp4") is not None
final_rel = "out/final.mp4" if delivered else None
if not delivered and B is not None:                    # the agent never delivered out/final.mp4: measure the newest regular mp4 in out/ and say so
    cands = [(st_.st_mtime_ns, name) for name, st_ in B.entries("out") if name.endswith(".mp4")]
    cands = [(m, n) for m, n in cands if B.regular(f"out/{n}") is not None]       # links, FIFOs and dangling names are refused and noted here
    if cands:
        final_rel = "out/" + sorted(cands)[-1][1]
out["final_mp4_delivered"] = delivered
files = {}
for name in ("SPEC.md", "VERIFY.md", "STRUCTURE.md", "out/final.mp4"):
    st_ = B.regular(name) if B is not None else None
    files[name] = st_.st_size if st_ is not None else None
out["files"] = files
final = os.path.join(task, final_rel) if final_rel else None
out["measured_file"] = final_rel
FRAMEMD5 = re.compile(r"^\d+,\s*-?\d+,\s*-?\d+,\s*\d+,\s*\d+,\s*([0-9a-fA-F]{32})$")      # stream, dts, pts, duration, size, md5


def frame_hash_figures(r, probe_frames):
    """The frame-hash measurement from one container result: the figures, or every figure null with the reason in `error`. Only framemd5
    records count as frames; any other line, any ffmpeg error output, a nonzero exit, a container that did not run, or a record count that
    disagrees with the frame count ffprobe reported makes the whole measurement a failure."""
    fh = {"frames": None, "distinct": None, "identical_to_previous": None, "longest_identical_run_frames": None, "invalid_lines": 0, "error": None}
    if not r["ok"]:
        fh["error"] = r["error"]
        return fh
    if r["rc"] != 0:
        fh["error"] = f"ffmpeg exited {r['rc']}: " + (" ".join(r["stderr"].split())[:200] or "no message")
        return fh
    if r["stderr"].strip():
        fh["error"] = "ffmpeg reported errors: " + " ".join(r["stderr"].split())[:200]
        return fh
    if len(r["stdout"]) > OUTPUT_CAP:
        fh["error"] = f"ffmpeg wrote more than {OUTPUT_CAP} bytes of frame records"
        return fh
    hashes, bad = [], []
    for ln in r["stdout"].splitlines():
        s_ = ln.strip()
        if not s_ or s_.startswith("#"):
            continue
        m = FRAMEMD5.match(s_)
        if m:
            hashes.append(m.group(1).lower())
        else:
            bad.append(s_)
    fh["invalid_lines"] = len(bad)
    if bad:
        fh["error"] = f"{len(bad)} invalid line(s) in the ffmpeg output, which are not framemd5 records (first: {bad[0][:80]!r})"
    elif not hashes:
        fh["error"] = "ffmpeg wrote no frame records"
    elif probe_frames is not None and probe_frames != len(hashes):
        fh["error"] = f"{len(hashes)} frame hashes against {probe_frames} frames counted by ffprobe"
    else:
        longest = cur = 0
        for i in range(1, len(hashes)):
            cur = cur + 1 if hashes[i] == hashes[i - 1] else 0
            longest = max(longest, cur)
        fh.update(frames=len(hashes), distinct=len(set(hashes)), longest_identical_run_frames=longest + 1,
                  identical_to_previous=sum(1 for i in range(1, len(hashes)) if hashes[i] == hashes[i - 1]))
    return fh


if final:
    qa = shlex.quote(final)                           # absolute, and quoted once: a file name may hold a space or a shell metacharacter
    probe = in_container(f"ffprobe -v error -count_frames -show_streams -show_format -of json {qa}")
    if not probe["ok"]:
        out["video"] = {"probe_error": probe["error"], "raw": ""}
        failures.append("video")
    else:
        try:
            pj = json.loads(probe["stdout"])
            v = [s for s in pj["streams"] if s["codec_type"] == "video"][0]
            out["video"] = {"codec": v["codec_name"], "size": f"{v['width']}x{v['height']}", "pix_fmt": v.get("pix_fmt"), "avg_frame_rate": v["avg_frame_rate"],
                            "frames": int(v["nb_read_frames"]) if str(v.get("nb_read_frames", "")).isdigit() else None, "duration_s": float(pj["format"]["duration"]),
                            "audio_streams": sum(1 for s in pj["streams"] if s["codec_type"] == "audio"), "bytes": int(pj["format"]["size"])}
        except Exception as exc:
            out["video"] = {"probe_error": str(exc)[:160], "raw": probe["text"][:300]}
            failures.append("video")
    trusted = trusted_skill()
    if trusted:
        out["check_video_source"] = f"trusted copy of the skill, read-only mount ({os.path.basename(trusted)})"
        chk = in_container("sh /trusted-skill/scripts/check_video.sh " + " ".join(shlex.quote(x) for x in (final, W, H, FPS, DMIN, DMAX, "0")),
                           extra_mounts=(f"{trusted}:/trusted-skill:ro",), workdir="/trusted-skill", out_cap=ERR_CAP)
        if chk["ok"]:
            out["check_video"] = chk["text"]
            out["check_video_exit"] = chk["rc"]
            if chk["rc"] != 0:                            # a checker that crashed or was killed may print nothing; its exit status is kept and counted
                out["check_video"] = f"exited {chk['rc']}: " + (chk["text"] or "no output")
                failures.append("check_video_exit")
        else:
            out["check_video"] = "not measured: " + chk["error"]
            failures.append("check_video")
    else:
        out["check_video_source"] = None
        out["check_video"] = ("not run: no trusted copy of the skill (set SKILL_SRC, or use a run made by run_bench.sh, which records "
                              "logs/skill_src); the task's own copy can be edited by the model")
    md5 = in_container(f"ffmpeg -v error -i {qa} -f framemd5 -an -", out_cap=OUTPUT_CAP)
    out["frame_hashes"] = frame_hash_figures(md5, out["video"].get("frames") if isinstance(out.get("video"), dict) else None)
    if out["frame_hashes"]["error"]:
        failures.append("frame_hashes")
out["measurement_failures"] = {"count": len(failures), "names": failures}
out["cleanup_failures"] = cleanup_failures
out["refused_paths"] = refused_paths
if B is not None:
    B.close()
json.dump(out, open(f"{logs}/summary.json", "w"), indent=1)
print(json.dumps(out, indent=1)[:6000])
