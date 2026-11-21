#!/usr/bin/env python3
"""Generate the noise-floor restoration demo WAV (deterministic).

gated_silence.wav  6 s mono 16 kHz 16-bit: coloured room tone (~-30 dBFS)
                   with three silenced gaps of 200 ms / 300 ms / 1000 ms,
                   i.e. what a denoising gate leaves behind.
"""
import os

import numpy as np
from scipy import signal
from scipy.io import wavfile

FS = 16000
GAPS = [(16000, 19200), (40000, 44800), (64000, 80000)]  # 200/300/1000 ms


def synth(path):
    n = int(6.0 * FS)
    rng = np.random.default_rng(20261121)
    white = rng.standard_normal(n)
    sos = signal.butter(4, (80.0, 4200.0), btype="bandpass", fs=FS,
                        output="sos")
    tone = signal.sosfiltfilt(sos, white)
    tone /= np.sqrt(np.mean(tone ** 2))
    x = 0.035 * tone
    t = np.arange(n) / FS
    for fc, t0 in ((300.0, 0.8), (1200.0, 3.6), (700.0, 5.2)):
        x += 0.05 * np.sin(2 * np.pi * fc * t) * np.exp(-((t - t0) / 0.15) ** 2)
    for s, e in GAPS:
        x[s:e] = 0.0
    pcm = np.asarray(np.rint(x * 32767), dtype=np.int16)
    wavfile.write(path, FS, pcm)
    print(f"wrote {path}  (gaps {GAPS}, {len(pcm)} samples)")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    synth(os.path.join(here, "gated_silence.wav"))
