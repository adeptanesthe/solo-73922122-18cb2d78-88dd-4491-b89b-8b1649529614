#!/usr/bin/env bash
# Demo requests against a running service (default http://localhost:8000).
# Regenerate the WAVs with: python3 make_demo_wav.py
set -euo pipefail
BASE="${BASE:-http://localhost:8000}"
cd "$(dirname "$0")"

echo "== GET /health =="
curl -s "$BASE/health"; echo; echo

echo "== POST /correct: 8 ms pilot gap, declared -> 200 + corrected WAV =="
curl -s -o corrected.wav -w "HTTP %{http_code}, %{size_download} bytes -> corrected.wav\n" \
  -F "file=@demo_tape.wav" \
  -F "gaps=[[24000,24128]]" \
  "$BASE/correct"
echo

echo "== POST /correct: 50 ms pilot gap -> 422 + re-record intervals, no WAV =="
curl -s -w "\nHTTP %{http_code}\n" \
  -F "file=@demo_tape_longgap.wav" \
  -F "gaps=[[24000,24800]]" \
  "$BASE/correct"
echo

echo "== POST /restore-noise: 3 gated gaps (200/300/1000 ms) -> 200 + restored WAV =="
curl -s -o restored.wav -w "HTTP %{http_code}, %{size_download} bytes -> restored.wav\n" \
  -F "file=@gated_silence.wav" \
  -F "gaps=[[16000,19200],[40000,44800],[64000,80000]]" \
  "$BASE/restore-noise"
echo

echo "== POST /restore-noise: gap at sample 0, no left reference -> 422, no WAV =="
curl -s -w "\nHTTP %{http_code}\n" \
  -F "file=@gated_silence.wav" \
  -F "gaps=[[0,3200]]" \
  "$BASE/restore-noise"
