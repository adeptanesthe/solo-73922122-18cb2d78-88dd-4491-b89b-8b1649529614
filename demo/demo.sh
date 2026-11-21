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
