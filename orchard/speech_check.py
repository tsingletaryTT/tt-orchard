# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
"""`tt-orchard speech-check`: check a RUNNING speech server (orchard/speech_models.py says which models are speech models).

    tt-orchard speech-check URL --task speech-to-text --clip FILE --expect TEXT [--clip FILE --expect TEXT ...] [--language CODE] [--max-wer F]
    tt-orchard speech-check URL --task text-to-speech [--text TEXT] [--voice NAME ...] [--stream] [--asr-url URL] [--max-wer F] [--sample-rate HZ]

The command sends requests to a server that someone else started (`tt-model serve`, under a gozer lease). It opens no device, starts no
server and takes no lease. The address is given with its port and is never guessed. Only a server on this machine is accepted, and ports 8000 and
8100 to 8103 are refused because other services use them. A redirect is a failure and is never followed.

What each check rests on:

  health, models        GET /health answers 200; GET /v1/models lists at least one model id. (Both Voxtral bundles answer these.)
  transcribe CLIP       the clip is sent to POST /v1/audio/transcriptions as the multipart field `file`; the answer must be JSON with a string `text`,
                        and the word error rate against the expected text must be at most --max-wer (default 0.10). The rule is `wer` below.
  bad request           the same route without a `file` must answer 4xx. The bundle's code answers 400; a 200 or a 5xx fails.
  speech [VOICE]        POST /v1/audio/speech with `input` (and `voice` when asked) must answer audio/* that opens as a WAV at the expected sample
                        rate (default 24000), 16-bit, mono and audible (a peak of at least 300 of 32767), with a duration between 0.02 and 0.4
                        seconds per character of the text. The duration bounds are wide on purpose: the Voxtral card's own clips run from about
                        0.06 seconds per character in English to about 0.2 in Hindi. They catch an empty, truncated or run-on answer. They say
                        nothing about what was said.
  transcribe speech     with --asr-url, the WAV is sent to that speech-to-text server and the transcript is compared with the text: word error
                        rate at most --max-wer (default 0.15). Without it the check says the words were not checked.
  bad voice             a voice the server cannot have must answer 4xx. A 200 or a 5xx fails.
  missing input         a request with no `input` must answer 4xx.
  stream                with --stream, a request with stream true and response_format pcm must answer a non-empty body of whole 16-bit samples
                        whose duration is in the same bounds.

`wer`: both texts are put in Unicode NFKC form and lower case; every character that is not a letter, a digit, a combining mark or an apostrophe
is a separator; an apostrophe at the start or end of a word is dropped; the words are compared by edit distance (a substitution, an insertion
and a deletion cost one each) and the distance is divided by the number of words in the expected text. A reference with no words is refused.

Not checked, and printed as such: questions about audio through /v1/chat/completions, languages other than the clips' own, concurrency,
long audio, and anything about speed. This command shows that a server answers correctly for the inputs it was given. It does not repeat the
bundle's own accuracy tests.
"""
from __future__ import annotations

import argparse
import array
import http.client
import io
import ipaddress
import json
import os
import re
import sys
import unicodedata
import wave
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from orchard import safeio

FORBIDDEN_PORTS = frozenset({8000, 8100, 8101, 8102, 8103})
DEFAULT_MAX_WER_TEXT = 0.10          # speech to text: the card's longest clip differed in 2 of 521 words
DEFAULT_MAX_WER_SPEECH = 0.15        # text to speech through an ASR server: two models in the chain, and the card's Hindi voices reach 0.3
SECONDS_PER_CHARACTER = (0.02, 0.4)
DEFAULT_SAMPLE_RATE = 24000
DEFAULT_TEXT = "The quick brown fox jumps over the lazy dog."
SILENCE_PEAK = 300                   # of 32767; real speech peaks in the thousands
NO_SUCH_VOICE = "no-such-voice-for-speech-check"
JSON_LIMIT = 1 << 20
AUDIO_LIMIT = 64 << 20
CLIP_LIMIT = 64 << 20
TEXT_LIMIT = 4000
TASKS = ("speech-to-text", "text-to-speech")

WER_RULE = ("word error rate = edit distance between the words of the expected text and of the answer, divided by the words of the expected text; "
            "both texts in NFKC form and lower case, anything but letters, digits, combining marks and apostrophes separates words, "
            "apostrophes at the edge of a word are dropped")


class SpeechCheckError(ValueError):
    """A refusal before anything is sent: a bad address, option, clip or text. `run` prints it and exits 2."""


@dataclass
class Result:
    name: str
    status: str                       # "pass" or "fail"
    detail: str
    data: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail, "data": self.data}


class _Failed(Exception):
    """A request that did not give an answer to judge (no connection, a timeout, a redirect, an answer over the limit)."""


# ---- the address --------------------------------------------------------------------------------------------------


def parse_base_url(url) -> tuple[str, int]:
    """(host, port) of a server address, or SpeechCheckError. Accepted: `http://HOST:PORT` with an optional `/` or `/v1` after it, where HOST
    is localhost or a loopback address. Refused: another scheme, credentials, a query or fragment, any other path, no port, a port outside 1 to 65535,
    a port in FORBIDDEN_PORTS, a host that is not on this machine, and any control character."""
    if not isinstance(url, str) or not url:
        raise SpeechCheckError("the server address must be text such as http://127.0.0.1:20000")
    if any(ord(c) < 32 or ord(c) == 127 for c in url):
        raise SpeechCheckError("the server address holds a control character")
    try:
        parts = urlsplit(url)
        port = parts.port
        host = parts.hostname
    except ValueError:
        raise SpeechCheckError(f"the server address {url!r} has a port or host that cannot be read") from None
    if parts.scheme != "http":
        raise SpeechCheckError("the server address must start with http:// (a local server does not use TLS)")
    if "@" in parts.netloc:
        raise SpeechCheckError("the server address must not hold a user name or password")
    if parts.query or parts.fragment or parts.path not in ("", "/", "/v1", "/v1/"):
        raise SpeechCheckError("the server address is http://HOST:PORT with nothing after it except an optional /v1")
    if not host:
        raise SpeechCheckError("the server address has no host")
    if not _is_local(host):
        raise SpeechCheckError(f"{host} is not on this machine. Only a server on this machine is checked, because the clips and text are sent to it")
    if port is None:
        raise SpeechCheckError("the server address needs its port (for example http://127.0.0.1:20000); no port is assumed")
    if not 1 <= port <= 65535:
        raise SpeechCheckError(f"port {port} is outside 1 to 65535")
    if port in FORBIDDEN_PORTS:
        raise SpeechCheckError(f"port {port} belongs to another service on this machine; the checker refuses 8000 and 8100 to 8103")
    return host, port


def _is_local(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


# ---- word error rate ----------------------------------------------------------------------------------------------


def words(text: str) -> list[str]:
    """The words of `text` under the rule in WER_RULE."""
    norm = unicodedata.normalize("NFKC", text).casefold()
    chars = [c if (unicodedata.category(c)[0] in "LNM" or c in "'’") else " " for c in norm]
    out = []
    for w in "".join(chars).replace("’", "'").split():
        w = w.strip("'")
        if w:
            out.append(w)
    return out


def wer(reference: str, hypothesis: str) -> tuple[int, int]:
    """(edit distance, number of reference words). A reference with no words is a SpeechCheckError."""
    ref, hyp = words(reference), words(hypothesis)
    if not ref:
        raise SpeechCheckError("the expected text has no words, so there is nothing to compare with")
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1], len(ref)


# ---- one request --------------------------------------------------------------------------------------------------


def _request(host: str, port: int, method: str, path: str, *, body: bytes | None = None, headers: dict | None = None,
             timeout: float, limit: int) -> tuple[int, dict, bytes]:
    """(status, lower-case headers, body). Raises _Failed with a sentence for no connection, a timeout, a redirect or an answer larger than
    `limit` bytes. A redirect is never followed: the target could be a port this command refuses."""
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        resp = conn.getresponse()
        if 300 <= resp.status < 400:
            raise _Failed(f"the server answered {resp.status}, a redirect; redirects are not followed")
        data = resp.read(limit + 1)
        if len(data) > limit:
            raise _Failed(f"the answer is larger than {limit} bytes")
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, data
    except _Failed:
        raise
    except TimeoutError:
        raise _Failed(f"the server timed out after {timeout:g} s") from None
    except (OSError, http.client.HTTPException) as exc:
        raise _Failed(f"{type(exc).__name__}: {exc}") from None
    finally:
        conn.close()


def _json_answer(data: bytes) -> dict | None:
    """The answer as a JSON object, or None when it is not one. Parsed with `safeio.parse_json_bytes`: no NaN or Infinity, no repeated key, no lone
    surrogate, no deep nesting, at most JSON_LIMIT bytes."""
    try:
        return safeio.parse_json_bytes(data, max_bytes=JSON_LIMIT, expect=dict)
    except safeio.SafeReadError:
        return None


def transcript_of(data: bytes) -> str | None:
    """The `text` of a transcription answer, or None when the answer is not a JSON object with a string `text`."""
    doc = _json_answer(data)
    return doc["text"] if doc is not None and isinstance(doc.get("text"), str) else None


def model_ids_of(data: bytes) -> list[str]:
    """The model ids listed by a `/v1/models` answer; empty when the answer is not the expected JSON."""
    doc = _json_answer(data)
    rows = doc.get("data") if doc is not None else None
    return [r["id"] for r in rows if isinstance(r, dict) and isinstance(r.get("id"), str)] if isinstance(rows, list) else []


def _clip(text: str, n: int = 300) -> str:
    return text if len(text) <= n else text[:n] + "..."


def _multipart(fields: dict[str, str], filename: str | None, content: bytes | None) -> tuple[bytes, str]:
    boundary = "speechcheck" + os.urandom(8).hex()
    out = []
    for k, v in fields.items():
        out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    if content is not None:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", filename or "clip")
        out.append(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{safe}"\r\nContent-Type: application/octet-stream\r\n\r\n'.encode()
                   + content + b"\r\n")
    out.append(f"--{boundary}--\r\n".encode())
    return b"".join(out), f"multipart/form-data; boundary={boundary}"


# ---- checks shared by both tasks ------------------------------------------------------------------------------------


def _health(host, port, timeout) -> Result:
    try:
        status, _, _ = _request(host, port, "GET", "/health", timeout=timeout, limit=JSON_LIMIT)
    except _Failed as exc:
        return Result("health", "fail", f"GET /health: {exc}")
    return Result("health", "pass" if status == 200 else "fail", f"GET /health answered {status}", {"status": status})


def _models(host, port, timeout) -> Result:
    try:
        status, _, data = _request(host, port, "GET", "/v1/models", timeout=timeout, limit=JSON_LIMIT)
    except _Failed as exc:
        return Result("models", "fail", f"GET /v1/models: {exc}")
    ids = model_ids_of(data)
    if status != 200 or not ids:
        return Result("models", "fail", f"GET /v1/models answered {status} and listed {len(ids)} model ids", {"status": status, "ids": ids})
    return Result("models", "pass", "GET /v1/models lists " + ", ".join(ids), {"status": status, "ids": ids})


def _expect_4xx(name: str, what: str, call) -> Result:
    """A request the server must refuse: 4xx passes, a 200 or a 5xx fails."""
    try:
        status = call()
    except _Failed as exc:
        return Result(name, "fail", f"{what}: {exc}")
    ok = 400 <= status < 500
    return Result(name, "pass" if ok else "fail", f"{what} answered {status}" + ("" if ok else " (a 4xx was expected)"), {"status": status})


def _check_max_wer(value, default):
    if value is None:
        return default
    if isinstance(value, bool) or not safeio.is_finite_number(value) or not 0 <= value <= 1:
        raise SpeechCheckError("--max-wer must be a number from 0 to 1")
    return float(value)       # finite: ok the comparison above keeps the value between 0 and 1


def _check_timeout(value):
    if isinstance(value, bool) or not safeio.is_finite_number(value) or not 0 < value <= 86400:
        raise SpeechCheckError("--timeout must be a number of seconds above 0 and at most 86400")
    return float(value)       # finite: ok the comparison above keeps the value between 0 and 86400


# ---- speech to text -------------------------------------------------------------------------------------------------


def _transcribe(host, port, content: bytes, filename: str, language, timeout) -> tuple[str | None, str, float]:
    """(transcript or None, a sentence when there is none, seconds). The sentence says what the server answered."""
    import time
    fields = {"response_format": "json"}
    if language:
        fields["language"] = language
    body, ctype = _multipart(fields, filename, content)
    t0 = time.monotonic()
    try:
        status, _, data = _request(host, port, "POST", "/v1/audio/transcriptions", body=body, headers={"Content-Type": ctype}, timeout=timeout, limit=JSON_LIMIT)
    except _Failed as exc:
        return None, str(exc), time.monotonic() - t0
    took = time.monotonic() - t0
    if status != 200:
        return None, f"POST /v1/audio/transcriptions answered {status}", took
    text = transcript_of(data)
    if text is None:
        return None, "the answer is not JSON with a string `text`", took
    return text, "", took


def _read_clip(path) -> bytes:
    try:
        return safeio.read_bytes(path, max_bytes=CLIP_LIMIT)
    except (safeio.SafeReadError, OSError, ValueError) as exc:
        raise SpeechCheckError(f"the clip {path} cannot be read: {exc}") from None


def check_speech_to_text(base_url, clips, *, language=None, max_wer=None, timeout=600.0) -> list[Result]:
    """Run the speech-to-text checks against the server at `base_url`. `clips` is a list of (path, expected text). Everything that can be refused is
    refused before the first request: the address, the options, each clip's file and each expected text."""
    host, port = parse_base_url(base_url)
    max_wer = _check_max_wer(max_wer, DEFAULT_MAX_WER_TEXT)
    timeout = _check_timeout(timeout)
    if not clips:
        raise SpeechCheckError("give at least one clip with its expected text (--clip FILE --expect TEXT)")
    loaded = []
    for path, expected in clips:
        if not isinstance(expected, str) or not words(expected):
            raise SpeechCheckError(f"the expected text for the clip {path} has no words")
        loaded.append((Path(path), _read_clip(path), expected))
    results = [_health(host, port, timeout), _models(host, port, timeout)]
    for path, content, expected in loaded:
        text, why, took = _transcribe(host, port, content, path.name, language, timeout)
        name = f"transcribe {path.name}"
        if text is None:
            results.append(Result(name, "fail", why))
            continue
        errors, n = wer(expected, text)
        rate = errors / n
        ok = rate <= max_wer
        detail = (f"word error rate {rate:.2f} ({errors} of {n} reference words), limit {max_wer:.2f}, {took:.1f} s"
                  + ("" if ok else f". Expected {_clip(expected)!r}, answered {_clip(text)!r}"))
        results.append(Result(name, "pass" if ok else "fail", detail,
                              {"wer": rate, "errors": errors, "reference_words": n, "max_wer": max_wer, "seconds": round(took, 3), "text": _clip(text, 2000)}))

    def no_file():
        body, ctype = _multipart({"response_format": "json"}, None, None)
        return _request(host, port, "POST", "/v1/audio/transcriptions", body=body, headers={"Content-Type": ctype}, timeout=timeout, limit=JSON_LIMIT)[0]

    results.append(_expect_4xx("bad request", "POST /v1/audio/transcriptions without a file", no_file))
    return results


# ---- text to speech -------------------------------------------------------------------------------------------------


def _judge_wav(data: bytes, ctype: str, text: str, sample_rate: int) -> tuple[bool, str, dict]:
    """(passed, a sentence, measured values) for a speech answer. See the module docstring for each rule."""
    if not ctype.lower().startswith("audio/"):
        return False, f"Content-Type is {ctype or 'missing'}, and audio/... was expected", {}
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            channels, width, rate, frames = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
            raw = w.readframes(frames)
    except (wave.Error, EOFError, OSError, ValueError) as exc:
        return False, f"the answer does not open as a WAV file ({exc})", {}
    got = {"sample_rate": rate, "channels": channels, "sample_width": width}
    problems = []
    if width != 2:
        problems.append(f"{width * 8}-bit samples (16-bit expected)")
    if channels != 1:
        problems.append(f"{channels} channels (mono expected)")
    if rate != sample_rate:
        problems.append(f"{rate} Hz (expected {sample_rate})")
    if problems:
        return False, "the WAV has " + ", ".join(problems), got
    seconds = frames / rate if rate else 0.0
    got["seconds"] = seconds
    samples = array.array("h")
    samples.frombytes(raw[: len(raw) - len(raw) % 2])
    if sys.byteorder == "big":
        samples.byteswap()
    peak = max((abs(s) for s in samples), default=0)
    got["peak"] = peak
    if peak < SILENCE_PEAK:
        return False, f"the audio is silent (peak {peak} of 32767, at least {SILENCE_PEAK} expected)", got
    return _judge_duration(seconds, text, got, f"WAV {rate} Hz, 16-bit, mono, peak {peak}")


def _judge_duration(seconds: float, text: str, got: dict, prefix: str) -> tuple[bool, str, dict]:
    per = seconds / max(len(text.strip()), 1)
    got["seconds_per_character"] = per
    lo, hi = SECONDS_PER_CHARACTER
    if not lo <= per <= hi:
        return False, f"{seconds:.2f} s of audio for {len(text.strip())} characters is {per:.3f} seconds per character, and {lo} to {hi} is expected", got
    return True, f"{prefix}, {seconds:.2f} s for {len(text.strip())} characters ({per:.3f} seconds per character)", got


def _speech_request(host, port, payload: dict, timeout, limit=AUDIO_LIMIT):
    return _request(host, port, "POST", "/v1/audio/speech", body=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, timeout=timeout, limit=limit)


def check_text_to_speech(base_url, text, *, voices=(), stream=False, asr_url=None, max_wer=None, sample_rate=DEFAULT_SAMPLE_RATE, timeout=600.0,
                         on_audio=None) -> list[Result]:
    """Run the text-to-speech checks against the server at `base_url`. `asr_url`, when given, is a speech-to-text server that transcribes each answer.
    `on_audio(name, wav_bytes)`, when given, is called for each answer that passed the WAV checks, with the result's name (`speech` or `speech VOICE`);
    the speech route (orchard/speech_route.py) saves the audio there and scores it later on the host."""
    host, port = parse_base_url(base_url)
    asr = parse_base_url(asr_url) if asr_url is not None else None
    max_wer = _check_max_wer(max_wer, DEFAULT_MAX_WER_SPEECH)
    timeout = _check_timeout(timeout)
    if on_audio is not None and not callable(on_audio):
        raise SpeechCheckError("on_audio must be a function of (name, wav bytes)")
    if not isinstance(text, str) or not words(text) or len(text) > TEXT_LIMIT:
        raise SpeechCheckError(f"--text must be text with at least one word and at most {TEXT_LIMIT} characters")
    if isinstance(sample_rate, bool) or not isinstance(sample_rate, int) or not 1000 <= sample_rate <= 192000:
        raise SpeechCheckError("--sample-rate must be a whole number from 1000 to 192000")
    voices = list(voices)
    if not all(isinstance(v, str) and 0 < len(v) <= 100 and v.isprintable() for v in voices):
        raise SpeechCheckError("each --voice must be a printable name of at most 100 characters")
    results = [_health(host, port, timeout), _models(host, port, timeout)]
    for voice in voices or [None]:
        name = "speech" if voice is None else f"speech {voice}"
        payload = {"input": text} | ({"voice": voice} if voice else {})
        try:
            status, headers, data = _speech_request(host, port, payload, timeout)
        except _Failed as exc:
            results.append(Result(name, "fail", f"POST /v1/audio/speech: {exc}"))
            continue
        if status != 200:
            results.append(Result(name, "fail", f"POST /v1/audio/speech answered {status}", {"status": status}))
            continue
        ok, detail, got = _judge_wav(data, headers.get("content-type", ""), text, sample_rate)
        results.append(Result(name, "pass" if ok else "fail", detail, got))
        if ok and on_audio is not None:
            on_audio(name, data)
        if asr is not None and ok:
            results.append(_transcribe_speech(asr, data, text, name, max_wer, timeout))
    results.append(_expect_4xx("bad voice", "POST /v1/audio/speech with a voice the server cannot have",
                               lambda: _speech_request(host, port, {"input": text, "voice": NO_SUCH_VOICE}, timeout, JSON_LIMIT)[0]))
    results.append(_expect_4xx("missing input", "POST /v1/audio/speech without `input`",
                               lambda: _speech_request(host, port, {}, timeout, JSON_LIMIT)[0]))
    if stream:
        results.append(_check_stream(host, port, text, voices[0] if voices else None, sample_rate, timeout))
    return results


def _transcribe_speech(asr, wav: bytes, text: str, speech_name: str, max_wer: float, timeout: float) -> Result:
    name = "transcribe " + speech_name
    heard, why, took = _transcribe(asr[0], asr[1], wav, "speech.wav", None, timeout)
    if heard is None:
        return Result(name, "fail", f"the speech-to-text server gave no transcript: {why}")
    errors, n = wer(text, heard)
    rate = errors / n
    ok = rate <= max_wer
    detail = (f"word error rate {rate:.2f} ({errors} of {n} words of the text), limit {max_wer:.2f}"
              + ("" if ok else f". Text {_clip(text)!r}, heard {_clip(heard)!r}"))
    return Result(name, "pass" if ok else "fail", detail, {"wer": rate, "errors": errors, "reference_words": n, "max_wer": max_wer, "text": _clip(heard, 2000)})


def _check_stream(host, port, text, voice, sample_rate, timeout) -> Result:
    payload = {"input": text, "stream": True, "response_format": "pcm"} | ({"voice": voice} if voice else {})
    try:
        status, _, data = _speech_request(host, port, payload, timeout)
    except _Failed as exc:
        return Result("stream", "fail", f"streamed POST /v1/audio/speech: {exc}")
    if status != 200:
        return Result("stream", "fail", f"streamed POST /v1/audio/speech answered {status}", {"status": status})
    if not data:
        return Result("stream", "fail", "the stream holds no audio")
    if len(data) % 2:
        return Result("stream", "fail", f"the stream is {len(data)} bytes, which is not a whole number of 16-bit samples")
    ok, detail, got = _judge_duration(len(data) / 2 / sample_rate, text, {"bytes": len(data)}, "raw PCM stream")
    return Result("stream", "pass" if ok else "fail", detail, got)


# ---- the command ------------------------------------------------------------------------------------------------------


def parser(add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tt-orchard speech-check", add_help=add_help,
                                description="check a running speech server: transcribe known clips, or request speech and check the audio. "
                                            "The server must be on this machine; the port is required and 8000 and 8100 to 8103 are refused. "
                                            "No device is opened and no lease is needed.")
    p.add_argument("url", help="the server address with its port, for example http://127.0.0.1:20000 (a /v1 at the end is allowed)")
    p.add_argument("--task", required=True, choices=TASKS, help="what the server does")
    p.add_argument("--clip", action="append", default=[], metavar="FILE", help="speech-to-text: an audio file to transcribe (repeat for more); pairs with --expect by position")
    p.add_argument("--expect", action="append", default=[], metavar="TEXT", help="speech-to-text: what the clip says")
    p.add_argument("--language", help="speech-to-text: the language code sent with each clip (omitted by default, so the server detects it)")
    p.add_argument("--text", default=DEFAULT_TEXT, help=f"text-to-speech: the text to speak (default {DEFAULT_TEXT!r})")
    p.add_argument("--voice", action="append", default=[], help="text-to-speech: a voice to try (repeat for more; none sends no voice)")
    p.add_argument("--stream", action="store_true", help="text-to-speech: also check the streamed raw PCM answer")
    p.add_argument("--asr-url", help="text-to-speech: a speech-to-text server on this machine that transcribes the speech, so the words are compared too")
    p.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help=f"text-to-speech: the sample rate expected (default {DEFAULT_SAMPLE_RATE})")
    p.add_argument("--max-wer", type=float, help=f"the largest word error rate that passes (default {DEFAULT_MAX_WER_TEXT} for speech-to-text, {DEFAULT_MAX_WER_SPEECH} for text-to-speech)")
    p.add_argument("--timeout", type=float, default=600.0, help="seconds to wait for each answer (default 600; a first request after a start can be slow)")
    p.add_argument("--json", action="store_true", help="print JSON")
    return p


def _misused(args) -> str | None:
    """Why the options do not fit the task, or None."""
    if args.task == "speech-to-text":
        stray = [n for n, v in (("--text", args.text != DEFAULT_TEXT), ("--voice", args.voice), ("--stream", args.stream), ("--asr-url", args.asr_url),
                                ("--sample-rate", args.sample_rate != DEFAULT_SAMPLE_RATE)) if v]
        if stray:
            return f"{', '.join(stray)} belong to --task text-to-speech"
        if len(args.clip) != len(args.expect):
            return f"give one --expect for each --clip ({len(args.clip)} clips, {len(args.expect)} expected texts)"
        return None
    stray = [n for n, v in (("--clip", args.clip), ("--expect", args.expect), ("--language", args.language)) if v]
    return f"{', '.join(stray)} belong to --task speech-to-text" if stray else None


def run(args, out=None) -> int:
    """Run the command for parsed arguments. Returns 0 when every check passed, 1 when one failed, 2 (one line on stderr, nothing sent) for a refusal."""
    out = out or sys.stdout
    try:
        if (why := _misused(args)) is not None:
            raise SpeechCheckError(why)
        if args.task == "speech-to-text":
            results = check_speech_to_text(args.url, list(zip(args.clip, args.expect)), language=args.language, max_wer=args.max_wer, timeout=args.timeout)
            default = DEFAULT_MAX_WER_TEXT
            not_checked = ["questions about audio through /v1/chat/completions", "languages other than the clips' own", "long audio, concurrency and speed"]
        else:
            results = check_text_to_speech(args.url, args.text, voices=args.voice, stream=args.stream, asr_url=args.asr_url, max_wer=args.max_wer,
                                           sample_rate=args.sample_rate, timeout=args.timeout)
            default = DEFAULT_MAX_WER_SPEECH
            not_checked = ["concurrency, long text and speed", "voices not asked for with --voice"]
            if not args.asr_url:
                not_checked.insert(0, "what the speech says: the words are not checked without --asr-url")
    except SpeechCheckError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    passed = all(r.status == "pass" for r in results)
    rule = {"max_wer": args.max_wer if args.max_wer is not None else default, "wer": WER_RULE}
    if args.task == "text-to-speech":
        rule |= {"seconds_per_character": list(SECONDS_PER_CHARACTER), "sample_rate": args.sample_rate, "silence_peak": SILENCE_PEAK}
    if args.json:
        print(json.dumps({"task": args.task, "url": args.url, "passed": passed, "checks": [r.as_dict() for r in results], "rule": rule,
                          "not_checked": not_checked}, indent=2), file=out)
        return 0 if passed else 1
    print(f"╔══ speech-check: {args.task} at {args.url}", file=out)
    for r in results:
        print(f"║  {r.status:<4}  {r.name}: {r.detail}", file=out)
    print("╠══ not checked", file=out)
    for line in not_checked:
        print(f"║  {line}", file=out)
    n_pass = sum(r.status == "pass" for r in results)
    print(f"╚══ {n_pass} passed, {len(results) - n_pass} failed", file=out)
    return 0 if passed else 1
