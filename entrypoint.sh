#!/bin/bash

echo "🔹 STEP 1: Running audio-separator test..."
audio-separator /app/AICover-server/input.wav --output_dir /app/AICover-server/output_sep

echo "🔹 STEP 2: Running AICover-server test..."
cd /app/AICover-server
python run.py --rvc_dirname Jimin --song_input input.wav --pitch_change_all 0

echo "✅ All tests complete."
