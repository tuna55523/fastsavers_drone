import argparse
import json
import math
import socket
import time


def main():
    parser = argparse.ArgumentParser(description="Send fake tracking telemetry to Android app")
    parser.add_argument("--host", default="192.168.10.2", help="Phone IP in Tello network")
    parser.add_argument("--port", type=int, default=5005)
    parser.add_argument("--hz", type=float, default=10.0)
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    dt = 1.0 / max(1.0, args.hz)
    t = 0.0

    print(f"Sending telemetry to {args.host}:{args.port} @ {args.hz} Hz")
    while True:
        tx = 0.35 * math.sin(t * 0.9)
        ty = 0.20 * math.sin(t * 0.7)
        size = 0.20 + 0.05 * math.sin(t * 0.5)
        conf = 0.85
        payload = {
            "tx": tx,
            "ty": ty,
            "size": max(0.02, min(0.8, size)),
            "conf": conf,
            "id": 1,
            "ts": int(time.time() * 1000),
        }
        sock.sendto(json.dumps(payload).encode("utf-8"), (args.host, args.port))
        time.sleep(dt)
        t += dt


if __name__ == "__main__":
    main()
