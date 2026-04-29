import time

import cv2
from djitellopy import Tello

from config import REAL_STREAM_STALE_SEC, REAL_STREAM_RESET_COOLDOWN_SEC
from system.movement.advanced_flight import AdvancedFlight


class DroneController:

    def __init__(self):
        self.frame_is_rgb = True

        print("[INFO] Connecting Drone")

        self.tello = Tello()
        self.tello.connect()

        print("Battery:", self.tello.get_battery())

        self.tello.streamoff()
        self.tello.streamon()
        time.sleep(0.35)

        self.frame_reader = self.tello.get_frame_read()
        self._last_frame_sig = None
        self._last_frame_change_ts = time.time()
        self._last_stream_reset_ts = 0.0

        self.flight = AdvancedFlight(self.tello)

    def _frame_signature(self, frame):
        try:
            small = cv2.resize(frame, (64, 36))
            return int(small.sum())
        except Exception:
            return None

    def _recover_stream(self, reason=""):
        now = time.time()
        if now - self._last_stream_reset_ts < REAL_STREAM_RESET_COOLDOWN_SEC:
            return False

        self._last_stream_reset_ts = now
        print(f"[DRONE] Stream reset ({reason})")

        try:
            self.flight.hover()
        except Exception:
            pass

        try:
            self.flight.run_sdk(self.tello.streamoff)
            time.sleep(0.20)
        except Exception:
            pass

        try:
            self.flight.run_sdk(self.tello.streamon)
            time.sleep(0.35)
            self.frame_reader = self.flight.run_sdk(self.tello.get_frame_read)
            self._last_frame_sig = None
            self._last_frame_change_ts = time.time()
            return True
        except Exception:
            return False

    def frame(self):
        frame = None
        try:
            if self.frame_reader is not None:
                frame = self.frame_reader.frame
        except Exception:
            frame = None

        if frame is None:
            if self._recover_stream("frame none") and self.frame_reader is not None:
                frame = self.frame_reader.frame

        if frame is None:
            raise RuntimeError("[DRONE] Frame read failed")

        now = time.time()
        sig = self._frame_signature(frame)
        if sig is not None:
            if self._last_frame_sig is None or sig != self._last_frame_sig:
                self._last_frame_sig = sig
                self._last_frame_change_ts = now
            elif now - self._last_frame_change_ts > REAL_STREAM_STALE_SEC:
                if self._recover_stream("stale") and self.frame_reader is not None:
                    refreshed = self.frame_reader.frame
                    if refreshed is not None:
                        frame = refreshed

        return frame

    def get_battery(self):
        return self.flight.run_sdk(self.tello.get_battery)

    def takeoff(self):
        self.flight.takeoff()

    def land(self):
        self.flight.land()

    def manual(self, lr, fb, ud, yaw):
        self.flight.manual(lr, fb, ud, yaw)

    def hover(self):
        self.flight.hover()

    def search_scan(self, yaw_speed):
        self.flight.search_scan(yaw_speed)

    def auto_follow(self, target, w, op_state="TRACK"):
        self.flight.follow(target, w, op_state=op_state)

    def close(self):
        try:
            self.flight.close()
        except Exception:
            pass

        try:
            self.flight.run_sdk(self.tello.streamoff)
        except Exception:
            pass

        try:
            self.tello.end()
        except Exception:
            pass
