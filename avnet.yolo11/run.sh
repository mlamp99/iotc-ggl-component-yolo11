#!/usr/bin/env bash
set -euo pipefail

# Inherit env from component.yaml
: "${SOURCE:=0}"
: "${MODEL:=yolo11n.pt}"
: "${CONF:=0.25}"
: "${VIEW:=true}"
: "${WIDTH:=640}"
: "${HEIGHT:=480}"
: "${FPS:=30}"
: "${IOTC_SOCKET:=/var/snap/iotconnect/common/iotc.sock}"

echo "[avnet.yolo11] starting with SOURCE=$SOURCE MODEL=$MODEL CONF=$CONF VIEW=$VIEW ${WIDTH}x${HEIGHT}@$FPS"

# Write runtime config for the Python app
cat > ./config.json <<JSON
{
  "source": "${SOURCE}",
  "model": "${MODEL}",
  "conf": ${CONF},
  "view": ${VIEW},
  "width": ${WIDTH},
  "height": ${HEIGHT},
  "fps": ${FPS},
  "iotc_socket": "${IOTC_SOCKET}"
}
JSON

exec ./.venv/bin/python ./yolo_gglite.py --config ./config.json
