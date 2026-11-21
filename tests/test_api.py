"""Self-tests for the pilot-tone speed corrector.

Synthetic tapes are generated with a known speed error, run through the
API, and the corrected output is checked against the known ground truth.
"""
import io
import json
import math

import numpy as np
import pytest
from scipy import signal
from scipy.integrate import cumulative_trapezoid
from scipy.io import wavfile

from fastapi.testclient import TestClient

from app import dsp
from app.main import app

FS = 16000
PROG_HZ = 440.0
client = TestClient(app)


# ---------------------------------------------------------------- helpers
def synth(dur=2.0, speed=1.01, prog_freq=PROG_HZ, gap=None, pilot_amp=0.7,
          noise=0.0, seed=0):
    """Stereo int16 tape: left = program tone, right = 1 kHz pilot.

    speed: constant factor or callable s(t); the digitised pilot then sits
    at 1000*s(t) Hz.  gap: optional (start, end) pilot dropout.
    """
    n = int(round(dur * FS))
    t = np.arange(n) / FS
    tau = cumulative_trapezoid(speed(t), t, initial=0.0) if callable(speed) \
        else speed * t
    program = 0.5 * np.sin(2 * np.pi * prog_freq * tau)
    pilot = pilot_amp * np.sin(2 * np.pi * 1000.0 * tau)
    if noise:
        rng = np.random.default_rng(seed)
        pilot = pilot + noise * rng.standard_normal(n)
    if gap is not None:
        pilot[gap[0]:gap[1]] = 0.0
    stereo = np.stack([program, pilot], axis=1)
    return np.asarray(np.round(stereo * 32767), dtype=np.int16)


def wav_bytes(pcm, rate=FS):
    buf = io.BytesIO()
    wavfile.write(buf, rate, pcm)
    return buf.getvalue()


def post(pcm, gaps=None, rate=FS):
    files = {"file": ("tape.wav", wav_bytes(pcm, rate), "audio/wav")}
    data = {} if gaps is None else {"gaps": json.dumps(gaps)}
    return client.post("/correct", files=files, data=data)


def read_wav(b):
    rate, pcm = wavfile.read(io.BytesIO(b))
    return rate, pcm


def dominant_freq(x):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    k = int(np.argmax(sp))
    if 0 < k < len(sp) - 1:  # parabolic refinement
        a, b, c = sp[k - 1], sp[k], sp[k + 1]
        k = k + 0.5 * (a - c) / (a - 2 * b + c)
    return k * FS / len(x)


# ------------------------------------------------------------------ tests
def test_constant_speed_corrected():
    pcm = synth(dur=2.0, speed=1.015)
    r = post(pcm)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/wav"
    rate, out = read_wav(r.content)
    assert rate == FS and out.ndim == 1 and out.dtype == np.int16
    # timeline stretches by the speed factor
    assert len(out) == pytest.approx(2.0 * 1.015 * FS, rel=0.01)
    # program tone back at its true frequency
    assert dominant_freq(out) == pytest.approx(PROG_HZ, abs=1.0)
    assert float(r.headers["x-mean-speed"]) == pytest.approx(1.015, abs=1e-3)


def test_slow_tape_corrected():
    pcm = synth(dur=2.0, speed=0.985)
    r = post(pcm)
    assert r.status_code == 200, r.text
    _, out = read_wav(r.content)
    assert dominant_freq(out) == pytest.approx(PROG_HZ, abs=1.0)
    assert len(out) == pytest.approx(2.0 * 0.985 * FS, rel=0.01)


def test_time_varying_speed_corrected():
    # wow: pilot sweeps 985..1015 Hz, still inside the 980..1020 contract
    pcm = synth(dur=3.0, speed=lambda t: 1.0 + 0.015 * np.sin(2 * np.pi * 0.5 * t))
    r = post(pcm)
    assert r.status_code == 200, r.text
    _, out = read_wav(r.content)
    assert dominant_freq(out) == pytest.approx(PROG_HZ, abs=1.5)


def test_gap_bridged_unique_cycles():
    gap = (32000, 32160)  # 10 ms pilot dropout at 2.0 s
    pcm = synth(dur=2.5, speed=1.0, gap=gap)
    r = post(pcm, gaps=[list(gap)])
    assert r.status_code == 200, r.text
    assert r.headers["x-gaps-bridged"] == "1"
    _, out = read_wav(r.content)
    # ground truth: program tone on a uniform grid (speed == 1.0)
    n_out = len(out)
    ref = 0.5 * np.sin(2 * np.pi * PROG_HZ * np.arange(n_out) / FS)
    corr = signal.correlate(out.astype(np.float64), ref, mode="full", method="fft")
    lag = int(np.argmax(corr)) - (n_out - 1)
    # one missed pilot cycle would shift the tail by 1 ms = 16 samples
    assert abs(lag) <= 3
    assert corr.max() > 0.9 * 0.125 * n_out  # near-perfect alignment


def test_gap_bridged_with_speed_error_and_noise():
    gap = (24000, 24128)  # 8 ms
    pcm = synth(dur=2.0, speed=1.008, gap=gap, noise=0.02, seed=7)
    r = post(pcm, gaps=[list(gap)])
    assert r.status_code == 200, r.text
    _, out = read_wav(r.content)
    assert dominant_freq(out) == pytest.approx(PROG_HZ, abs=1.5)


def test_gap_too_long_returns_rerecord_intervals():
    gap = (16000, 16800)  # 50 ms -> 40 Hz * 50 ms = 2 cycles of ambiguity
    pcm = synth(dur=2.0, speed=1.01, gap=gap)
    r = post(pcm, gaps=[list(gap)])
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "rerecord_required"
    assert body["rerecord_intervals"] == [
        {"start_sample": gap[0], "end_sample": gap[1],
         "start_sec": gap[0] / FS, "end_sec": gap[1] / FS}]
    assert "audio" not in r.headers.get("content-type", "")


def test_mixed_gaps_reports_only_unresolvable():
    ok_gap = [16000, 16096]    # 6 ms, bridgeable
    bad_gap = [32000, 32800]   # 50 ms, ambiguous
    pcm = synth(dur=3.0, speed=1.0, gap=None)
    pcm[ok_gap[0]:ok_gap[1], 1] = 0
    pcm[bad_gap[0]:bad_gap[1], 1] = 0
    r = post(pcm, gaps=[ok_gap, bad_gap])
    assert r.status_code == 422
    body = r.json()
    assert len(body["rerecord_intervals"]) == 1
    assert body["rerecord_intervals"][0]["start_sample"] == bad_gap[0]
    assert body["rerecord_intervals"][0]["end_sample"] == bad_gap[1]


def test_leading_and_trailing_gaps_are_trimmed():
    n = 2 * FS
    pcm = synth(dur=2.0, speed=1.01)
    pcm[:1600, 1] = 0          # pilot missing over the first/last 100 ms
    pcm[n - 1600:, 1] = 0
    r = post(pcm, gaps=[[0, 1600], [n - 1600, n]])
    assert r.status_code == 200, r.text
    _, out = read_wav(r.content)
    # remaining 1.8 s of pilot, stretched by 1.01
    assert len(out) == pytest.approx(1.8 * 1.01 * FS, rel=0.02)
    assert dominant_freq(out) == pytest.approx(PROG_HZ, abs=1.5)


def test_resolve_cycles_unit():
    # 10 ms at a true 1000 Hz -> exactly 10 cycles, delta 0
    assert dsp.resolve_cycles(0.0, 0.010) == 10
    # 10 ms, half-cycle residual: 9.5 or 10.5 cycles, both outside 9.8..10.2
    assert dsp.resolve_cycles(math.pi, 0.010) is None
    # 40 ms, delta 0: window [39.2, 40.8] is 1.6 cycles wide but happens to
    # hold exactly one integer -> still uniquely resolvable
    assert dsp.resolve_cycles(0.0, 0.040) == 40
    # same span, half-cycle residual: [38.7, 40.3] holds 39 and 40 -> ambiguous
    assert dsp.resolve_cycles(math.pi, 0.040) is None
    # 50 ms: window 2.0 cycles wide -> always ambiguous
    assert dsp.resolve_cycles(0.0, 0.050) is None
    # 25 ms at true 1000 Hz: window exactly 1.0 cycle wide -> still unique
    assert dsp.resolve_cycles(0.0, 0.025) == 25


def test_rejects_mono():
    pcm = synth(dur=1.0)[:, 0]
    r = post(pcm)
    assert r.status_code == 400


def test_rejects_wrong_rate():
    pcm = synth(dur=1.0)
    r = post(pcm, rate=8000)
    assert r.status_code == 400


def test_rejects_too_long():
    pcm = synth(dur=12.5, speed=1.0)
    r = post(pcm)
    assert r.status_code == 400


def test_accepts_exactly_12s():
    pcm = synth(dur=12.0, speed=1.0)
    r = post(pcm)
    assert r.status_code == 200, r.text


def test_rejects_float_wav():
    pcm = synth(dur=1.0).astype(np.float32) / 32768.0
    r = post(pcm)
    assert r.status_code == 400


def test_rejects_bad_gaps():
    pcm = synth(dur=1.0)
    assert post(pcm, gaps=[[500, 100], ]).status_code == 400        # start >= end
    assert post(pcm, gaps=[[0, 99999]]).status_code == 400          # out of range
    assert post(pcm, gaps=[[0, 100], [50, 200]]).status_code == 400  # overlap
    files = {"file": ("t.wav", wav_bytes(pcm), "audio/wav")}
    r = client.post("/correct", files=files, data={"gaps": "not json"})
    assert r.status_code == 400


def test_rejects_missing_pilot():
    pcm = synth(dur=1.0)
    pcm[:, 1] = 0
    r = post(pcm)
    assert r.status_code == 400
    assert "pilot" in r.json()["detail"]


def test_rejects_undeclared_dropout():
    pcm = synth(dur=2.0, gap=(8000, 9600))  # 100 ms dropout, not declared
    r = post(pcm)
    assert r.status_code == 400
    assert "dropout" in r.json()["detail"]


def test_health():
    assert client.get("/health").json() == {"status": "ok"}
