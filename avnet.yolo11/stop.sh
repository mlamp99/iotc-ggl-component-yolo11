#!/usr/bin/env bash
set -euo pipefail
echo "[avnet.yolo11] stopping..."
# Try a graceful stop by killing the python process
pkill -f "python ./yolo_gglite.py" || true
sleep 1
pkill -9 -f "python ./yolo_gglite.py" || true
echo "[avnet.yolo11] stopped"
