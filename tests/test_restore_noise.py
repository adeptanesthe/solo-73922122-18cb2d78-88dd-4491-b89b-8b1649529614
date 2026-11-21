"""Self-tests for the /restore-noise (noise-floor restoration) API.

Synthetic "gated" recordings are built from coloured room tone with
known silence gaps, run through the API, and checked against the
contract:

* output is a 16 kHz mono 16-bit WAV with the same sample count and
  byte-identical samples outside the gaps;
* the first/last 80 ms of every patch match the same-side reference in
  both the 100-1000 Hz and 1000-4000 Hz bands within 3 dB;
* gaps lacking 80 ms of adjacent original material, or whose reference
  bands fall below -60 dBFS, get a 422 with no WAV.
"""
import io
import json

import numpy as np
import pytest
from scipy import signal
from scipy.io import wavfile

from fastapi.testclient import TestClient

from app import noisefloor as nf
from app.main import app

FS = 16000
client = TestClient(app)
BANDS = ((100.0, 1000.0), (1000.0, 4000.0))


# ---------------------------------------------------------------- helpers
def room_tone(n, level=0.03, seed=0, color_seed=1):
    """Stationary coloured noise with energy in both reference bands."""
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    sos = signal.butter(4, (80.0, 4200.0), btype="bandpass", fs=FS,
                        output="sos")
    tone = signal.sosfiltfilt(sos, white)
    tone /= np.sqrt(np.mean(tone ** 2))
    # jitter the filter seed so two independently generated floors differ
    rng2 = np.random.default_rng(color_seed)
    return tone * level + 1e-4 * rng2.standard_normal(n)


def gated_wav(dur=3.0, gaps=(), level=0.03, tone_hz=None, seed=0):
    """Mono int16 WAV: room tone (or a pure tone) with the gaps silenced."""
    n = int(round(dur * FS))
    if tone_hz is None:
        x = room_tone(n, level=level, seed=seed)
    else:
        t = np.arange(n) / FS
        x = 0.2 * np.sin(2 * np.pi * tone_hz * t)
    x = x.copy()
    for s, e in gaps:
        x[s:e] = 0.0
    return np.asarray(np.rint(x * 32767), dtype=np.int16)


def wav_bytes(pcm, rate=FS):
    buf = io.BytesIO()
    wavfile.write(buf, rate, pcm)
    return buf.getvalue()


def post(pcm, gaps, rate=FS):
    files = {"file": ("restored_in.wav", wav_bytes(pcm, rate), "audio/wav")}
    data = {"gaps": json.dumps(gaps)}
    return client.post("/restore-noise", files=files, data=data)


def read_wav(b):
    rate, pcm = wavfile.read(io.BytesIO(b))
    return rate, pcm


def band_rms_independent(x, f_lo, f_hi):
    """Reference RMS via scipy periodogram (independent code path)."""
    f, pxx = signal.periodogram(x.astype(np.float64), fs=FS,
                                scaling="spectrum")
    mask = (f >= f_lo) & (f <= f_hi)
    return float(np.sqrt(np.sum(pxx[mask])))


def db(x):
    return 20.0 * np.log10(max(float(x), 1e-12))


def edge_margin(out_f, pcm_in, s, e):
    """Worst |patch-edge - same-side reference| mismatch over both bands."""
    worst = 0.0
    details = []
    for f_lo, f_hi in BANDS:
        ref_l = band_rms_independent(pcm_in[max(0, s - nf.REF_SAMPLES):s]
                                     .astype(np.float64) / 32768.0, f_lo, f_hi)
        ref_r = band_rms_independent(pcm_in[e:e + nf.REF_SAMPLES]
                                     .astype(np.float64) / 32768.0, f_lo, f_hi)
        got_l = band_rms_independent(out_f[s:s + nf.EDGE_SAMPLES], f_lo, f_hi)
        got_r = band_rms_independent(out_f[e - nf.EDGE_SAMPLES:e], f_lo, f_hi)
        d_l = abs(db(got_l) - db(ref_l))
        d_r = abs(db(got_r) - db(ref_r))
        worst = max(worst, d_l, d_r)
        details.append((round(db(ref_l), 2), round(db(got_l), 2),
                        round(db(ref_r), 2), round(db(got_r), 2)))
    return worst, details


# ------------------------------------------------------------------ tests
def test_single_gap_patch_matches_references_within_3db():
    gap = (24000, 27200)  # 200 ms
    pcm = gated_wav(dur=3.0, gaps=[gap])
    r = post(pcm, [list(gap)])
    assert r.status_code == 200, r.text
    assert r.headers["content-type"] == "audio/wav"
    rate, out = read_wav(r.content)
    assert rate == FS and out.ndim == 1 and out.dtype == np.int16
    assert len(out) == len(pcm)
    assert r.headers["x-gaps-filled"] == "1"
    assert r.headers["x-output-samples"] == str(len(pcm))

    out_f = out.astype(np.float64) / 32768.0
    # gap is no longer silence
    assert float(np.sqrt(np.mean(out_f[gap[0]:gap[1]] ** 2))) > 0.005
    worst, details = edge_margin(out_f, pcm, *gap)
    assert worst < 3.0, details  # contract: <= 3 dB, implementation aims tighter


def test_samples_outside_gaps_are_identical():
    gaps = [(8000, 11200), (32000, 35200)]
    pcm = gated_wav(dur=4.0, gaps=gaps)
    r = post(pcm, [list(g) for g in gaps])
    assert r.status_code == 200, r.text
    _, out = read_wav(r.content)
    assert len(out) == len(pcm)
    mask = np.ones(len(pcm), dtype=bool)
    for s, e in gaps:
        mask[s:e] = False
    np.testing.assert_array_equal(out[mask], pcm[mask])


def test_four_gaps_and_band_edge_lengths():
    # 1..4 gaps accepted; include the 200 ms minimum and 1000 ms maximum.
    gaps = [(4000, 7200), (12000, 15200),
            (20000, 28000), (36000, 52000)]
    pcm = gated_wav(dur=5.0, gaps=gaps)
    r = post(pcm, [list(g) for g in gaps])
    assert r.status_code == 200, r.text
    _, out = read_wav(r.content)
    assert r.headers["x-gaps-filled"] == "4"
    out_f = out.astype(np.float64) / 32768.0
    for g in gaps:
        worst, _ = edge_margin(out_f, pcm, *g)
        assert worst < 3.0


def test_truncated_reference_window_by_neighbour_gap_still_works():
    # 125 ms between the gaps: each 160 ms window truncates to 125 ms,
    # still >= 80 ms, so the request must succeed.
    gap1, gap2 = (8000, 11200), (13200, 16400)
    pcm = gated_wav(dur=3.0, gaps=[gap1, gap2])
    r = post(pcm, [list(gap1), list(gap2)])
    assert r.status_code == 200, r.text
    _, out = read_wav(r.content)
    out_f = out.astype(np.float64) / 32768.0
    for g in (gap1, gap2):
        worst, _ = edge_margin(out_f, pcm, *g)
        assert worst < 3.0


def test_level_morphs_from_left_to_right_floor():
    # Different stationary floor levels on either side of the gap.
    n = int(3.0 * FS)
    s, e = 24000, 27200
    x = np.concatenate([
        room_tone(s, level=0.012, seed=3),
        np.zeros(e - s),
        room_tone(n - e, level=0.048, seed=4),
    ])
    pcm = np.asarray(np.rint(x * 32767), dtype=np.int16)
    r = post(pcm, [[s, e]])
    assert r.status_code == 200, r.text
    _, out = read_wav(r.content)
    f = out.astype(np.float64) / 32768.0
    lo, hi = BANDS[0]
    left = db(band_rms_independent(f[s:s + nf.EDGE_SAMPLES], lo, hi))
    right = db(band_rms_independent(f[e - nf.EDGE_SAMPLES:e], lo, hi))
    middle = db(band_rms_independent(
        f[(s + e) // 2 - 400:(s + e) // 2 + 400], lo, hi))
    assert right - left > 6.0          # references really differ
    assert left - 1.0 < middle < right + 1.0  # interior sits between them


def test_gap_at_file_start_is_reference_insufficient():
    gap = (0, 4800)
    pcm = gated_wav(dur=2.0, gaps=[gap])
    r = post(pcm, [list(gap)])
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "reference_insufficient"
    items = body["insufficient_references"]
    assert len(items) == 1
    item = items[0]
    assert item["gap_index"] == 0 and item["side"] == "left"
    assert item["reason"] == "too_short"
    assert item["available_samples"] == 0
    assert "audio" not in r.headers.get("content-type", "")


def test_gap_at_file_end_is_reference_insufficient():
    n = int(2.0 * FS)
    gap = (n - 4800, n)
    pcm = gated_wav(dur=2.0, gaps=[gap])
    r = post(pcm, [list(gap)])
    assert r.status_code == 422
    item = r.json()["insufficient_references"][0]
    assert item["side"] == "right" and item["reason"] == "too_short"


def test_neighbour_gap_truncates_below_80ms_and_lists_both_sides():
    # Only 50 ms (800 samples) between two gaps.
    gap1, gap2 = (8000, 11200), (12000, 15200)
    pcm = gated_wav(dur=2.0, gaps=[gap1, gap2])
    r = post(pcm, [list(gap1), list(gap2)])
    assert r.status_code == 422
    items = r.json()["insufficient_references"]
    sides = {(it["gap_index"], it["side"]) for it in items}
    assert (0, "right") in sides and (1, "left") in sides
    assert all(it["available_samples"] == 800 for it in items)


def test_reference_band_below_floor_lists_band():
    # A 500 Hz tone has essentially no energy in the 1000-4000 Hz band.
    gap = (16000, 19200)
    pcm = gated_wav(dur=3.0, gaps=[gap], tone_hz=500.0)
    r = post(pcm, [list(gap)])
    assert r.status_code == 422
    items = r.json()["insufficient_references"]
    assert len(items) == 2  # both sides lack the high band
    for it in items:
        assert it["reason"] == "band_below_floor"
        assert it["bands_below_minus60_dbfs"][0]["band_hz"] == [1000, 4000]
        assert it["bands_below_minus60_dbfs"][0]["rms_dbfs"] < -60.0


def test_validation_errors_are_400():
    pcm = gated_wav(dur=2.0, gaps=[(8000, 9600)])
    files = {"file": ("in.wav", wav_bytes(pcm), "audio/wav")}

    # missing gaps field
    r = client.post("/restore-noise", files=files)
    assert r.status_code == 422  # FastAPI form-validation error
    # empty gap list
    r = client.post("/restore-noise", files=files, data={"gaps": "[]"})
    assert r.status_code == 400
    # five gaps
    five = [[1000 + 4000 * i, 2600 + 4000 * i] for i in range(5)]
    r = client.post("/restore-noise", files=files, data={"gaps": json.dumps(five)})
    assert r.status_code == 400
    # 100 ms gap (< 200 ms minimum)
    r = post(pcm, [[8000, 9600 - 1600]])
    assert r.status_code == 400
    # 1.1 s gap (> 1000 ms maximum)
    r = post(pcm, [[8000, 8000 + 17600]])
    assert r.status_code == 400
    # overlapping gaps
    r = post(pcm, [[8000, 12000], [11000, 14000]])
    assert r.status_code == 400
    # malformed JSON
    r = client.post("/restore-noise", files=files, data={"gaps": "nope"})
    assert r.status_code == 400


def test_rejects_stereo_wrong_rate_float_and_too_long():
    stereo = np.stack([gated_wav(dur=1.0), gated_wav(dur=1.0, seed=9)], axis=1)
    assert post(stereo, [[2000, 3600]]).status_code == 400
    pcm = gated_wav(dur=1.0)
    r = post(pcm, [[2000, 3600]], rate=8000)
    assert r.status_code == 400
    flt = (pcm.astype(np.float32) / 32768.0)
    r = post(flt, [[2000, 3600]])
    assert r.status_code == 400
    long_pcm = gated_wav(dur=13.1)
    r = post(long_pcm, [[2000, 3600]])
    assert r.status_code == 400


def test_accepts_exactly_13_seconds():
    pcm = gated_wav(dur=13.0, gaps=[(100000, 103200)])
    r = post(pcm, [[100000, 103200]])
    assert r.status_code == 200, r.text
    _, out = read_wav(r.content)
    assert len(out) == len(pcm)


def test_output_is_deterministic():
    gap = (20000, 23200)
    pcm = gated_wav(dur=3.0, gaps=[gap])
    r1 = post(pcm, [list(gap)])
    r2 = post(pcm, [list(gap)])
    assert r1.content == r2.content


def test_band_rms_unit_matches_periodogram():
    rng = np.random.default_rng(11)
    x = rng.standard_normal(4096)
    for lo, hi in BANDS:
        a = nf.band_rms(x, lo, hi)
        b = band_rms_independent(x, lo, hi)
        assert a == pytest.approx(b, rel=1e-9)


def test_health_still_ok():
    assert client.get("/health").json() == {"status": "ok"}
