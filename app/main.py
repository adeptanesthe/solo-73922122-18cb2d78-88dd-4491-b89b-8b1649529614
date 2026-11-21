"""FastAPI service: tape speed correction and noise-floor restoration.

POST /correct accepts a 16 kHz stereo 16-bit PCM WAV (<= 12 s; left =
program, right = pilot) plus an optional JSON form field `gaps` with
declared pilot-gap sample intervals.  On success it returns the
speed-corrected mono program WAV.  When a gap cannot be bridged
unambiguously it answers 422 with the intervals that need re-recording
and no WAV.

POST /restore-noise accepts the mono /correct output format (16 kHz,
16-bit PCM, <= 13 s) plus 1-4 silence-gap sample intervals of 200-1000 ms
each.  It fills the gaps with synthesised room tone matched to the
original samples on both sides.  When a gap side lacks enough reference
material it answers 422 listing the deficient gap/side and returns no WAV.
"""
from __future__ import annotations

import io
import json

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response
from scipy.io import wavfile

from . import dsp, noisefloor as nf

app = FastAPI(title="Pilot-tone tape speed corrector", version="1.0.0")

MAX_BYTES = int(dsp.FS * dsp.MAX_SECONDS) * 2 * 2 + 65536  # payload + header slack
MAX_RESTORE_BYTES = int(nf.FS * nf.MAX_SECONDS) * 2 + 65536


def _fail(status: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"detail": detail})


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def info():
    return {
        "service": "pilot-tone tape speed corrector",
        "usage": [
            "POST /correct with multipart form: file=<16 kHz stereo 16-bit "
            "WAV, <=12 s>, optional gaps='[[start,end],...]' (sample "
            "intervals of pilot dropouts)",
            "POST /restore-noise with multipart form: file=<16 kHz mono "
            "16-bit WAV, <=13 s, /correct output format>, "
            "gaps='[[start,end],...]' (1..4 silence intervals, each "
            "200..1000 ms)",
        ],
    }


@app.post("/correct")
async def correct(file: UploadFile = File(...),
                  gaps: str | None = Form(default=None)):
    raw = await file.read()
    if len(raw) > MAX_BYTES:
        return _fail(400, "file too large: exceeds 12 s of 16 kHz stereo 16-bit PCM")
    try:
        rate, data = wavfile.read(io.BytesIO(raw))
    except Exception as exc:
        return _fail(400, f"not a readable WAV file: {exc}")
    if rate != dsp.FS:
        return _fail(400, f"sample rate must be {dsp.FS} Hz, got {rate}")
    if data.ndim != 2 or data.shape[1] != 2:
        return _fail(400, "expected a stereo WAV: left = program, right = pilot")
    if data.dtype != np.int16:
        return _fail(400, f"expected 16-bit PCM, got dtype {data.dtype}")
    n = int(data.shape[0])
    if n == 0:
        return _fail(400, "empty WAV")
    if n > int(dsp.MAX_SECONDS * dsp.FS):
        return _fail(400, f"duration exceeds {dsp.MAX_SECONDS:.0f} s")

    try:
        gap_list = _parse_gaps(gaps, n)
    except ValueError as exc:
        return _fail(400, str(exc))

    program = data[:, 0].astype(np.float64) / 32768.0
    pilot = data[:, 1].astype(np.float64) / 32768.0
    try:
        res = dsp.correct(program, pilot, gap_list)
    except dsp.GapUnresolvable as exc:
        return JSONResponse(
            status_code=422,
            content={
                "status": "rerecord_required",
                "detail": "pilot gap(s) cannot be bridged unambiguously; "
                          "re-record the listed intervals",
                "rerecord_intervals": [
                    {"start_sample": s, "end_sample": e,
                     "start_sec": round(s / dsp.FS, 6),
                     "end_sec": round(e / dsp.FS, 6)}
                    for s, e in exc.gaps
                ],
            },
        )
    except dsp.DspError as exc:
        return _fail(400, str(exc))

    pcm = np.clip(np.rint(res.audio * 32767.0), -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, dsp.FS, pcm)
    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="corrected.wav"',
            "X-Output-Samples": str(res.n_out),
            "X-Mean-Speed": f"{res.mean_speed:.6f}",
            "X-Gaps-Bridged": str(len(res.bridged_gaps)),
        },
    )


@app.post("/restore-noise")
async def restore_noise(file: UploadFile = File(...),
                        gaps: str = Form(...)):
    """Rebuild the noise floor inside the declared silence gaps."""
    raw = await file.read()
    if len(raw) > MAX_RESTORE_BYTES:
        return _fail(400, "file too large: exceeds 13 s of 16 kHz mono 16-bit PCM")
    try:
        rate, data = wavfile.read(io.BytesIO(raw))
    except Exception as exc:
        return _fail(400, f"not a readable WAV file: {exc}")
    if rate != nf.FS:
        return _fail(400, f"sample rate must be {nf.FS} Hz, got {rate}")
    if data.ndim != 1:
        return _fail(400, "expected a mono WAV in the /correct output format")
    if data.dtype != np.int16:
        return _fail(400, f"expected 16-bit PCM, got dtype {data.dtype}")
    n = int(data.shape[0])
    if n > int(nf.MAX_SECONDS * nf.FS):
        return _fail(400, f"duration exceeds {nf.MAX_SECONDS:.0f} s")

    try:
        gap_list = _parse_noise_gaps(gaps, n)
    except ValueError as exc:
        return _fail(400, str(exc))

    audio = data.astype(np.float64) / 32768.0
    try:
        out, report = nf.restore(audio, gap_list)
    except nf.ReferenceInsufficient as exc:
        return JSONResponse(
            status_code=422,
            content={
                "status": "reference_insufficient",
                "detail": "one or more gap sides lack 80 ms of usable "
                          "original samples or fall below -60 dBFS; "
                          "no WAV is produced for this request",
                "insufficient_references": [
                    {
                        "gap_index": gi,
                        "start_sample": s,
                        "end_sample": e,
                        "start_sec": round(s / nf.FS, 6),
                        "end_sec": round(e / nf.FS, 6),
                        "side": fail.side,
                        "available_samples": fail.available_samples,
                        "reason": ("too_short"
                                   if fail.available_samples < nf.EDGE_SAMPLES
                                   else "band_below_floor"),
                        "bands_below_minus60_dbfs": fail.low_bands,
                    }
                    for gi, s, e, fail in exc.failures
                ],
            },
        )
    except nf.NoiseFloorError as exc:
        return _fail(400, str(exc))

    pcm = np.clip(np.rint(out * 32768.0), -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, nf.FS, pcm)
    return Response(
        content=buf.getvalue(),
        media_type="audio/wav",
        headers={
            "Content-Disposition": 'attachment; filename="restored.wav"',
            "X-Output-Samples": str(report["output_samples"]),
            "X-Gaps-Filled": str(report["gaps_filled"]),
        },
    )


def _parse_gaps(raw: str | None, n: int) -> list[tuple[int, int]]:
    """Validate the optional JSON gaps field: [[start, end), ...] in samples."""
    if raw is None or raw.strip() == "":
        return []
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"gaps must be a JSON array of [start, end] pairs: {exc}")
    if not isinstance(obj, list) or len(obj) > 512:
        raise ValueError("gaps must be a JSON array of at most 512 [start, end] pairs")
    gaps: list[tuple[int, int]] = []
    for item in obj:
        if (not isinstance(item, (list, tuple)) or len(item) != 2
                or not all(isinstance(v, int) and not isinstance(v, bool)
                           for v in item)):
            raise ValueError("each gap must be a [start_sample, end_sample] "
                             "pair of integers")
        s, e = int(item[0]), int(item[1])
        if not 0 <= s < e <= n:
            raise ValueError(f"gap [{s}, {e}] out of range 0..{n}")
        gaps.append((s, e))
    gaps.sort()
    for (s0, e0), (s1, _) in zip(gaps, gaps[1:]):
        if s1 < e0:
            raise ValueError(f"overlapping gaps: [{s0}, {e0}] and [{s1}, ...]")
    return gaps


def _parse_noise_gaps(raw: str, n: int) -> list[tuple[int, int]]:
    """Validate the gaps field for /restore-noise.

    Rules: a JSON array of 1..4 integer [start, end) pairs, each interval
    200..1000 ms long, inside the file and pairwise non-overlapping.
    """
    if raw is None or raw.strip() == "":
        raise ValueError(
            "gaps is required: a JSON array of 1..4 [start_sample, "
            "end_sample] silence intervals, each 200..1000 ms")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"gaps must be a JSON array of [start, end] pairs: {exc}")
    if not isinstance(obj, list) or not 1 <= len(obj) <= nf.MAX_GAPS:
        raise ValueError(f"gaps must contain between 1 and {nf.MAX_GAPS} intervals")
    gaps: list[tuple[int, int]] = []
    for item in obj:
        if (not isinstance(item, (list, tuple)) or len(item) != 2
                or not all(isinstance(v, int) and not isinstance(v, bool)
                           for v in item)):
            raise ValueError("each gap must be a [start_sample, end_sample] "
                             "pair of integers")
        s, e = int(item[0]), int(item[1])
        if not 0 <= s < e <= n:
            raise ValueError(f"gap [{s}, {e}] out of range 0..{n}")
        length = e - s
        if not nf.MIN_GAP_SAMPLES <= length <= nf.MAX_GAP_SAMPLES:
            raise ValueError(
                f"gap [{s}, {e}] is {length} samples; each gap must be "
                f"{nf.MIN_GAP_SAMPLES}..{nf.MAX_GAP_SAMPLES} samples "
                f"({nf.MIN_GAP_MS}..{nf.MAX_GAP_MS} ms)")
        gaps.append((s, e))
    gaps.sort()
    for (s0, e0), (s1, e1) in zip(gaps, gaps[1:]):
        if s1 < e0:
            raise ValueError(
                f"overlapping gaps: [{s0}, {e0}] and [{s1}, {e1}]")
    return gaps
