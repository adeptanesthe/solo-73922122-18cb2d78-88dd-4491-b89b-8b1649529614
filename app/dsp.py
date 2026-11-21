"""Pilot-tone tape speed correction (DSP core).

Model
-----
The right track of the digitised tape carries a nominal 1 kHz sine pilot
whose instantaneous frequency is confined to 980..1020 Hz, i.e. the
transport speed error stays within +/-2 %.  If phi(t) is the cumulative
(unwrapped) phase of the digitised pilot, the recording-time coordinate
of the sample digitised at time t is

    tau(t) = phi(t) / (2*pi*1000)

because the pilot completes exactly 1000 cycles per second of *tape*
time.  Resampling the left (program) track from this non-uniform tau
grid onto a uniform 16 kHz grid removes the speed error.

Pilot gaps
----------
Inside a declared gap the pilot phase is unobservable.  The number of
whole pilot cycles crossed during the gap is recovered from the phase
measured at both gap edges plus the 980..1020 Hz constraint: the phase
advance over the gap must equal 2*pi*n + delta with n whole cycles and
must be consistent with the allowed frequency band for the whole span.
The gap is bridged (by linear phase interpolation) only when exactly one
integer n satisfies the constraint; otherwise the gap is reported back
as needing re-recording.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

FS = 16000                 # required sample rate (Hz)
PILOT_HZ = 1000.0          # nominal pilot frequency (Hz)
F_MIN = 980.0              # guaranteed lower bound of instantaneous pilot freq
F_MAX = 1020.0             # guaranteed upper bound of instantaneous pilot freq
MAX_SECONDS = 12.0         # maximum accepted duration

BP_BAND = (900.0, 1100.0)  # band-pass used to isolate the pilot
GUARD = 96                 # samples excluded near gap edges (filter transient), 6 ms
FIT = 192                  # linear phase-fit window, 12 ms
MIN_SEG = GUARD + FIT + 1  # minimum clean pilot needed next to a gap


class DspError(Exception):
    """Input failed validation or processing; maps to HTTP 400."""


class GapUnresolvable(Exception):
    """One or more pilot gaps cannot be bridged unambiguously."""

    def __init__(self, gaps: list[tuple[int, int]]):
        super().__init__(f"unresolvable pilot gap(s): {gaps}")
        self.gaps = gaps  # list of (start_sample, end_sample), end exclusive


def resolve_cycles(delta_mod: float, span_s: float,
                   f_min: float = F_MIN, f_max: float = F_MAX) -> int | None:
    """Unique whole-cycle count crossed during a gap, or None.

    delta_mod : (phase_after - phase_before) mod 2*pi, in radians.
    span_s    : seconds between the two phase anchors.

    The true phase advance is 2*pi*n + delta_mod and must be consistent
    with an instantaneous frequency inside [f_min, f_max] over span_s:

        2*pi*f_min*span <= 2*pi*n + delta_mod <= 2*pi*f_max*span

    Returns the integer n iff exactly one integer satisfies the inequality.
    """
    lo = f_min * span_s - delta_mod / (2.0 * math.pi)
    hi = f_max * span_s - delta_mod / (2.0 * math.pi)
    n_lo = math.ceil(lo - 1e-9)
    n_hi = math.floor(hi + 1e-9)
    if n_lo == n_hi:
        return int(n_lo)
    return None


@dataclass
class Result:
    audio: np.ndarray                       # float64 mono samples, ~[-1, 1]
    n_out: int                              # number of output samples
    mean_speed: float                       # mean tape-time / digitised-time ratio
    bridged_gaps: list[tuple[int, int]] = field(default_factory=list)


def correct(program: np.ndarray, pilot: np.ndarray,
            gaps: list[tuple[int, int]]) -> Result:
    """Rebuild the time axis from the pilot and resample the program track.

    program, pilot : float64 mono signals (full scale ~1.0), len N, FS Hz.
    gaps           : validated [start, end) sample intervals, sorted, disjoint.

    Raises DspError on unusable input and GapUnresolvable when a gap cannot
    be bridged unambiguously.
    """
    n = program.shape[0]
    good = np.ones(n, dtype=bool)
    for s, e in gaps:
        good[s:e] = False
    good_idx = np.flatnonzero(good)
    if len(good_idx) < FS // 2:
        raise DspError("usable pilot outside the declared gaps is shorter than 0.5 s")
    segs = _segments(good_idx)

    pilot_bp = _bandpass(pilot)
    rms = float(np.sqrt(np.mean(pilot_bp[good_idx] ** 2)))
    if rms < 1e-3:
        raise DspError("no pilot tone detected on the right channel")
    _check_dropouts(pilot_bp, segs)

    # Cumulative phase per contiguous good segment (each segment has its own
    # unknown integer-cycle offset until the gaps are bridged).
    phase = np.angle(signal.hilbert(pilot_bp))
    unwrapped = [np.unwrap(phase[s]) for s in segs]

    # --- resolve the whole-cycle count of every gap ----------------------
    # Anchors are linear fits of the unwrapped phase over a clean window
    # GUARD samples away from the gap edge, extrapolated to the edge; this
    # avoids the band-pass/hilbert transient right at the edge.
    anchors = []
    bad: list[tuple[int, int]] = []
    for i in range(len(segs) - 1):
        gap = (int(segs[i][-1]) + 1, int(segs[i + 1][0]))
        left = _edge_phase(segs[i], unwrapped[i], "right")
        right = _edge_phase(segs[i + 1], unwrapped[i + 1], "left")
        if left is None or right is None:
            bad.append(gap)  # too little clean pilot next to this gap
            anchors.append(None)
            continue
        (t1, p1), (t2, q2) = left, right
        span = (t2 - t1) / FS
        delta = float(np.mod(q2 - p1, 2.0 * math.pi))
        ncyc = resolve_cycles(delta, span)
        if ncyc is None:
            bad.append(gap)
            anchors.append(None)
            continue
        anchors.append((p1, q2, ncyc, delta))
        # p1/q2 are in per-segment unwrap frames; the mod-2*pi delta is not.
    if bad:
        raise GapUnresolvable(bad)

    # --- absolute cumulative phase on all good samples -------------------
    offsets = [0.0]
    bridged: list[tuple[int, int]] = []
    for i, a in enumerate(anchors):
        p1, q2, ncyc, delta = a
        offsets.append(p1 + offsets[-1] + 2.0 * math.pi * ncyc + delta - q2)
        bridged.append((int(segs[i][-1]) + 1, int(segs[i + 1][0])))
    abs_phase = np.concatenate([u + off for u, off in zip(unwrapped, offsets)])
    tau_good = abs_phase / (2.0 * math.pi * PILOT_HZ)

    _check_frequency(good_idx, tau_good)

    # --- time axis on every sample, then uniform resampling --------------
    i0, i1 = int(good_idx[0]), int(good_idx[-1])
    pos = np.arange(i0, i1 + 1)
    tau = np.interp(pos, good_idx, tau_good)  # linear phase across gaps
    tau -= tau[0]
    np.maximum.accumulate(tau, out=tau)       # keep the grid non-decreasing

    n_out = int(math.floor(tau[-1] * FS)) + 1
    grid = np.arange(n_out, dtype=np.float64) / FS
    out = np.interp(grid, tau, program[i0:i1 + 1])
    mean_speed = tau[-1] / ((i1 - i0) / FS)
    return Result(audio=out, n_out=n_out, mean_speed=mean_speed,
                  bridged_gaps=bridged)


def _bandpass(x: np.ndarray) -> np.ndarray:
    sos = signal.butter(4, BP_BAND, btype="bandpass", fs=FS, output="sos")
    return signal.sosfiltfilt(sos, x)


def _segments(good_idx: np.ndarray) -> list[np.ndarray]:
    """Split sorted good-sample indices into contiguous runs."""
    cuts = np.flatnonzero(np.diff(good_idx) > 1) + 1
    return np.split(good_idx, cuts)


def _edge_phase(idx: np.ndarray, phase: np.ndarray, side: str):
    """Extrapolated unwrapped phase at a segment edge.

    Fits a line to the unwrapped phase over a FIT-long window that starts
    GUARD samples away from the edge (to skip the filter transient) and
    evaluates it at the edge sample.  Returns (position, phase) or None
    when the segment is too short.
    """
    m = len(idx)
    if side == "right":
        w0, w1 = m - GUARD - FIT, m - GUARD
    else:
        w0, w1 = GUARD, GUARD + FIT
    if w0 < 0 or w1 > m:
        return None
    x = idx[w0:w1].astype(np.float64)
    y = phase[w0:w1]
    xc = float(x.mean())
    a, b = np.polyfit(x - xc, y, 1)
    pos = int(idx[-1]) if side == "right" else int(idx[0])
    return pos, float(a * (pos - xc) + b)


def _check_frequency(good_idx: np.ndarray, tau_good: np.ndarray) -> None:
    """Sanity-check the measured pilot frequency against the contract."""
    same = np.diff(good_idx) == 1
    if not np.any(same):
        raise DspError("not enough contiguous pilot samples")
    f = np.diff(tau_good)[same] * FS * PILOT_HZ
    med = float(np.median(f))
    if not 950.0 <= med <= 1050.0:
        raise DspError(
            f"pilot frequency out of range: median {med:.1f} Hz, "
            "expected a 1 kHz pilot within 980..1020 Hz")
    p1, p99 = (float(v) for v in np.percentile(f, [1, 99]))
    if p1 < 880.0 or p99 > 1120.0:
        raise DspError(
            f"pilot frequency unstable: 1st/99th percentile "
            f"{p1:.1f}/{p99:.1f} Hz, expected within 980..1020 Hz")


def _check_dropouts(pilot_bp: np.ndarray, segs: list[np.ndarray]) -> None:
    """Reject undeclared pilot dropouts (they would corrupt the time axis)."""
    w = 80  # 5 ms RMS window
    env = np.sqrt(np.convolve(pilot_bp ** 2, np.ones(w) / w, mode="same"))
    interiors = [s[GUARD:-GUARD] for s in segs if len(s) > 2 * GUARD]
    if not interiors:
        return
    idx = np.concatenate(interiors)
    med = float(np.median(env[idx]))
    if med <= 0.0:
        return
    if float(np.mean(env[idx] < 0.1 * med)) > 0.01:
        raise DspError(
            "pilot dropout detected outside the declared gaps; "
            "re-submit with the gap sample intervals in the 'gaps' field")
