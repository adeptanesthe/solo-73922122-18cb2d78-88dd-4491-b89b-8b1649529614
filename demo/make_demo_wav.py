#!/usr/bin/env python3
"""Generate the synthetic demo tapes (deterministic, no noise).

demo_tape.wav          3 s, speed error 1.012 + 0.4 % wow, 8 ms pilot gap
demo_tape_longgap.wav  same but with a 50 ms pilot gap (cannot be bridged)
"""
import os

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.io import wavfile

FS = 16000
GAP_OK = (24000, 24128)    # 8 ms pilot gap at 1.5 s
GAP_BAD = (24000, 24800)   # 50 ms pilot gap at 1.5 s


def synth(path, gap):
    dur = 3.0
    n = int(dur * FS)
    t = np.arange(n) / FS
    speed = 1.012 + 0.004 * np.sin(2 * np.pi * 0.8 * t)  # pilot: 1008..1016 Hz
    tau = cumulative_trapezoid(speed, t, initial=0.0)
    program = (0.45 * np.sin(2 * np.pi * 440.0 * tau)
               + 0.25 * np.sin(2 * np.pi * 659.25 * tau))
    pilot = 0.7 * np.sin(2 * np.pi * 1000.0 * tau)
    pilot[gap[0]:gap[1]] = 0.0
    stereo = np.round(np.stack([program, pilot], axis=1) * 32767).astype(np.int16)
    wavfile.write(path, FS, stereo)
    print(f"wrote {path}  (gap {gap[0]}..{gap[1]} samples)")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    synth(os.path.join(here, "demo_tape.wav"), GAP_OK)
    synth(os.path.join(here, "demo_tape_longgap.wav"), GAP_BAD)
