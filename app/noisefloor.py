"""Noise-floor restoration (DSP core).

A denoising gate has flattened the pauses ("gaps") of an oral-history
recording to digital silence.  This module rebuilds a plausible room-tone
patch for every declared gap.

Reference estimation
--------------------
Next to each gap we take up to 160 ms of *original* samples on each side.
The window starts immediately at the gap edge and is truncated when it
reaches another declared gap or the file boundary.  The spectral shape of
the local noise floor is summarised in two bands, 100-1000 Hz and
1000-4000 Hz, each reduced to its RMS level (periodogram based, so the
same measurement is used on the references and on the synthetic patch).

A side is unusable, and the whole request is rejected without producing a
WAV, when fewer than 80 ms of original samples remain on that side, or
when either of its reference bands sits below -60 dBFS.

Synthesis
---------
For each gap, white Gaussian noise (deterministic seed) is split into the
same two bands with zero-phase Butterworth filters, normalised to unit
RMS, scaled per band by a gain envelope, and summed.  The envelope holds
the left reference level across the first 80 ms of the gap, the right
reference level across the last 80 ms, and follows a smoothstep
transition in between, so the patch characteristics morph from the left
to the right floor entirely *inside* the gap.  A short raised-cosine fade
at both patch ends hides the seam against the untouched neighbours.  A
few fixed-point iterations rescale the envelope plateaus so that the
80 ms patch-edge band RMS matches the same-side reference within a
fraction of a dB (the contract allows 3 dB).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

FS = 16000                  # required sample rate (Hz)
MAX_SECONDS = 13.0          # maximum accepted duration
REF_MS = 160                # reference window length per side (ms)
EDGE_MS = 80                # acceptance / plateau window (ms)
MIN_GAP_MS = 200            # minimum gap length (ms)
MAX_GAP_MS = 1000           # maximum gap length (ms)
MAX_GAPS = 4
BANDS: tuple[tuple[float, float], ...] = ((100.0, 1000.0),
                                          (1000.0, 4000.0))
FLOOR_DBFS = -60.0          # reference band minimum level
SEAM_FADE_MS = 5            # raised-cosine seam fade (ms)

REF_SAMPLES = int(REF_MS * FS / 1000)      # 2560
EDGE_SAMPLES = int(EDGE_MS * FS / 1000)    # 1280
MIN_GAP_SAMPLES = int(MIN_GAP_MS * FS / 1000)   # 3200
MAX_GAP_SAMPLES = int(MAX_GAP_MS * FS / 1000)   # 16000
FLOOR_LEVEL = 10.0 ** (FLOOR_DBFS / 20.0)       # 1e-3 full scale
_FADE = int(SEAM_FADE_MS * FS / 1000)


class NoiseFloorError(Exception):
    """Input failed validation; maps to HTTP 400."""


@dataclass
class SideFailure:
    side: str                          # "left" | "right"
    available_samples: int
    low_bands: list[dict] = field(default_factory=list)


class ReferenceInsufficient(Exception):
    """One or more gaps lack usable reference material; no WAV is made."""

    def __init__(self, failures: list[tuple[int, int, SideFailure]]):
        super().__init__(f"insufficient reference material: {failures}")
        # list of (gap_index, gap_start, gap_end, SideFailure)
        self.failures = failures


# ---------------------------------------------------------------- measurement
def band_rms(x: np.ndarray, f_lo: float, f_hi: float) -> float:
    """RMS of x inside [f_lo, f_hi] Hz, estimated from its periodogram.

    The one-sided periodogram is scaled so that summing every bin equals
    the signal mean square (Parseval); the returned value is the square
    root of the in-band mean square.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n < 16:
        return 0.0
    x = x - x.mean()
    spec = np.fft.rfft(x)
    psd = (spec.real ** 2 + spec.imag ** 2) / (n * n)
    if n > 2:  # interior bins represent both the +k and -k frequency
        psd[1:-1] *= 2.0
    freqs = np.fft.rfftfreq(n, 1.0 / FS)
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    return float(np.sqrt(np.sum(psd[mask])))


def _reference_windows(audio: np.ndarray, gaps: list[tuple[int, int]]
                       ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Up-to-160 ms original-sample window on each side of every gap.

    Windows are truncated by the file edges and by neighbouring gaps.
    """
    n = audio.shape[0]
    windows: list[tuple[np.ndarray, np.ndarray]] = []
    for i, (s, e) in enumerate(gaps):
        l_start = max(0, s - REF_SAMPLES)
        if i > 0:
            l_start = max(l_start, gaps[i - 1][1])
        r_end = min(n, e + REF_SAMPLES)
        if i + 1 < len(gaps):
            r_end = min(r_end, gaps[i + 1][0])
        windows.append((audio[l_start:s], audio[e:r_end]))
    return windows


def _check_references(windows, gaps) -> None:
    """Collect every (gap, side) that is too short or below the floor."""
    failures: list[tuple[int, int, SideFailure]] = []
    for i, ((left, right), (s, e)) in enumerate(zip(windows, gaps)):
        for side, win in (("left", left), ("right", right)):
            avail = int(win.shape[0])
            if avail < EDGE_SAMPLES:
                failures.append((i, s, e, SideFailure(side, avail)))
                continue
            low_bands = []
            for f_lo, f_hi in BANDS:
                rms = band_rms(win, f_lo, f_hi)
                if rms < FLOOR_LEVEL:
                    low_bands.append({
                        "band_hz": [int(f_lo), int(f_hi)],
                        "rms_dbfs": round(20.0 * math.log10(max(rms, 1e-12)), 2),
                    })
            if low_bands:
                failures.append((i, s, e,
                                 SideFailure(side, avail, low_bands)))
    if failures:
        raise ReferenceInsufficient(failures)


# ---------------------------------------------------------------- synthesis
def _band_noise(white: np.ndarray, f_lo: float, f_hi: float) -> np.ndarray:
    """Band-limited unit-RMS noise (zero-phase 4th-order Butterworth)."""
    sos = signal.butter(4, (f_lo, f_hi), btype="bandpass", fs=FS,
                        output="sos")
    y = signal.sosfiltfilt(sos, white)
    rms = float(np.sqrt(np.mean(y ** 2)))
    return y / rms if rms > 0.0 else y


def _shape_envelope(length: int, plateau: int) -> np.ndarray:
    """0 on the left plateau, 1 on the right plateau, smoothstep between."""
    env = np.empty(length, dtype=np.float64)
    env[:plateau] = 0.0
    env[length - plateau:] = 1.0
    mid = length - 2 * plateau
    if mid > 0:
        u = np.arange(mid, dtype=np.float64) / mid
        env[plateau:length - plateau] = 3.0 * u ** 2 - 2.0 * u ** 3
    return env


def _seam_fade(length: int) -> np.ndarray:
    """Raised-cosine fade-in/out of _FADE samples at each patch end."""
    fade = np.ones(length, dtype=np.float64)
    k = min(_FADE, length // 2)
    ramp = 0.5 * (1.0 - np.cos(math.pi * np.arange(k) / k))
    fade[:k] = ramp
    fade[length - k:] = ramp[::-1]
    return fade


def _synth_patch(left: np.ndarray, right: np.ndarray, length: int,
                 start: int, end: int) -> np.ndarray:
    """Noise patch whose edges match the two reference floors."""
    targets = [(band_rms(left, lo, hi), band_rms(right, lo, hi))
               for lo, hi in BANDS]

    seed = (0x5F3759DF ^ (start * 1_000_003) ^ (end * 7919)) & 0xFFFFFFFF
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(length)
    bands = [_band_noise(white, lo, hi) for lo, hi in BANDS]
    shape = _shape_envelope(length, EDGE_SAMPLES)
    fade = _seam_fade(length)

    # Per-band plateau gains, starting at the reference levels.
    levels = [[t[0], t[1]] for t in targets]
    patch = np.zeros(length, dtype=np.float64)
    for _ in range(4):
        patch.fill(0.0)
        for comp, (l_lvl, r_lvl) in zip(bands, levels):
            gain = l_lvl * (1.0 - shape) + r_lvl * shape
            patch += comp * gain
        patch *= fade
        # Re-measure the 80 ms patch edges and rescale each plateau so the
        # measured band RMS converges onto the same-side reference.
        for b, ((t_l, t_r), (lo, hi)) in enumerate(zip(targets, BANDS)):
            m_l = band_rms(patch[:EDGE_SAMPLES], lo, hi)
            m_r = band_rms(patch[length - EDGE_SAMPLES:], lo, hi)
            if m_l > 0.0:
                levels[b][0] *= t_l / m_l
            if m_r > 0.0:
                levels[b][1] *= t_r / m_r
    return patch


def restore(audio: np.ndarray,
            gaps: list[tuple[int, int]]) -> tuple[np.ndarray, dict]:
    """Fill every gap with synthesised room tone.

    audio : float64 mono signal, full scale ~1.0, FS Hz, length <= 13 s.
    gaps  : validated [start, end) sample intervals, sorted, disjoint,
            each 200-1000 ms, at most four.

    Returns (output, report).  Raises ReferenceInsufficient when any gap
    side lacks 80 ms of usable original material or falls below the
    -60 dBFS reference floor; in that case no audio is produced.
    """
    if audio.ndim != 1:
        raise NoiseFloorError("expected a mono WAV (the /correct output)")
    n = audio.shape[0]
    if n == 0:
        raise NoiseFloorError("empty WAV")
    if n > int(MAX_SECONDS * FS):
        raise NoiseFloorError(f"duration exceeds {MAX_SECONDS:.0f} s")
    if not 1 <= len(gaps) <= MAX_GAPS:
        raise NoiseFloorError(f"provide between 1 and {MAX_GAPS} gaps")
    for s, e in gaps:
        if not 0 <= s < e <= n:
            raise NoiseFloorError(f"gap [{s}, {e}] out of range 0..{n}")
        length = e - s
        if not MIN_GAP_SAMPLES <= length <= MAX_GAP_SAMPLES:
            raise NoiseFloorError(
                f"gap [{s}, {e}] is {length} samples; each gap must be "
                f"{MIN_GAP_SAMPLES}..{MAX_GAP_SAMPLES} samples "
                f"({MIN_GAP_MS}..{MAX_GAP_MS} ms)")
    ordered = sorted(gaps)
    for (s0, e0), (s1, e1) in zip(ordered, ordered[1:]):
        if s1 < e0:
            raise NoiseFloorError(
                f"overlapping gaps: [{s0}, {e0}] and [{s1}, {e1}]")

    windows = _reference_windows(audio, ordered)
    _check_references(windows, ordered)

    out = audio.copy()
    edge_checks = []
    for (s, e), (left, right) in zip(ordered, windows):
        length = e - s
        patch = _synth_patch(left, right, length, s, e)
        out[s:e] = patch
        edge_checks.append({
            "start_sample": int(s), "end_sample": int(e),
            "left_db": [round(20 * math.log10(max(
                band_rms(patch[:EDGE_SAMPLES], lo, hi), 1e-12)), 2)
                for lo, hi in BANDS],
            "right_db": [round(20 * math.log10(max(
                band_rms(patch[length - EDGE_SAMPLES:], lo, hi), 1e-12)), 2)
                for lo, hi in BANDS],
        })
    report = {
        "gaps_filled": len(ordered),
        "output_samples": int(n),
        "edge_band_rms_dbfs": edge_checks,
    }
    return out, report
