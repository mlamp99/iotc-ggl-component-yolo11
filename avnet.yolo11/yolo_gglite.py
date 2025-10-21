#!/usr/bin/env python3
import argparse, json, os, sys, socket, time
from typing import Optional
import cv2
from ultralytics import YOLO

def send_iotc(sock_path: str, payload: dict):
    if not sock_path or not os.path.exists(sock_path):
        return
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(0.5)
        s.connect(sock_path)
        msg = (json.dumps({"t": "telemetry", "d": payload}) + "\n").encode("utf-8")
        s.sendall(msg)
        s.close()
    except Exception:
        # Fail silent – demo should keep running even if socket is missing
        pass

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    return ap.parse_args()

def parse_source(src: str):
    # If it's a plain integer string -> use as camera index; else treat as URL/path
    try:
        return int(src)
    except ValueError:
        return src

def main():
    args = parse_args()
    cfg = json.load(open(args.config))
    source = parse_source(cfg.get("source", "0"))
    model_name = cfg.get("model", "yolo11n.pt")
    conf = float(cfg.get("conf", 0.25))
    view = bool(cfg.get("view", True))
    width = int(cfg.get("width", 640))
    height = int(cfg.get("height", 480))
    fps = int(cfg.get("fps", 30))
    iotc_socket = cfg.get("iotc_socket", "/var/snap/iotconnect/common/iotc.sock")

    model = YOLO(model_name)
    cap = cv2.VideoCapture(source)
    if width > 0 and height > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv2.CAP_PROP_FPS, fps)

    if not cap.isOpened():
        print(f"[avnet.yolo11] ERROR: cannot open source={source}", file=sys.stderr)
        sys.exit(1)

    win = "YOLO11 Preview"
    last_pub = 0.0
    pub_period = 0.5  # seconds, throttle telemetry

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            results = model.predict(frame, conf=conf, verbose=False)
            dets = []
            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None:
                    for b in r.boxes:
                        cls = int(b.cls[0]) if b.cls is not None else -1
                        confv = float(b.conf[0]) if b.conf is not None else 0.0
                        xyxy = b.xyxy[0].tolist()
                        dets.append({"cls": cls, "conf": confv, "xyxy": xyxy})

                # Draw
                if view:
                    annotated = r.plot()  # Ultralytics helper
                else:
                    annotated = frame
            else:
                annotated = frame

            now = time.time()
            if now - last_pub >= pub_period:
                # Minimal telemetry payload – adjust to your template/topic mapping
                payload = {
                    "timestamp": int(now * 1000),
                    "count": len(dets),
                    "detections": dets[:10]  # cap length
                }
                send_iotc(iotc_socket, payload)
                last_pub = now

            if view:
                cv2.imshow(win, annotated)
                # ESC to quit
                if cv2.waitKey(1) & 0xFF == 27:
                    break
    finally:
        cap.release()
        if view:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
