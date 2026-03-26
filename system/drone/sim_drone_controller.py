import csv
import os
import time

import cv2

from config import TARGET_AREA


class SimDroneController:

    def __init__(self, video_path="", loop_video=True, command_log_path="logs/sim_commands.csv"):
        self.frame_is_rgb = False
        if isinstance(video_path, str):
            cleaned_path = video_path.strip()
            compact_dot_path = cleaned_path.replace(" .", ".")
            if cleaned_path and not os.path.exists(cleaned_path) and os.path.exists(compact_dot_path):
                cleaned_path = compact_dot_path
            self.video_path = cleaned_path
        else:
            self.video_path = video_path
        self.loop_video = loop_video
        self.command_log_path = command_log_path

        self.airborne = False
        self.battery = 100
        self.frame_index = 0
        self.start_time = time.time()
        self.loop_count = 0
        self.last_frame_ts = 0.0

        self.log_file = None
        self.csv_writer = None

        source = 0 if not self.video_path else self.video_path
        print(f"[SIM] Opening source: {source}")

        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"[SIM] Could not open video source: {source}")

        self.source_fps = self.cap.get(cv2.CAP_PROP_FPS)
        if self.source_fps <= 1e-3:
            self.source_fps = 30.0

        frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if frame_count <= 0:
            frame_count = 0
        self.source_duration = float(frame_count / self.source_fps) if frame_count > 0 else 0.0

        self._open_log_file()

    def _open_log_file(self):
        if not self.command_log_path:
            return

        folder = os.path.dirname(self.command_log_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

        self.log_file = open(self.command_log_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.writer(self.log_file)
        self.csv_writer.writerow([
            "timestamp",
            "frame_idx",
            "source",
            "lr",
            "fb",
            "ud",
            "yaw",
            "target_id",
            "risk",
            "raw_risk",
            "alert_state",
        ])
        self.log_file.flush()

    def _log_command(self, source, lr, fb, ud, yaw, target=None):
        if self.csv_writer is None:
            return

        target_id = -1
        risk = -1.0
        raw_risk = -1.0
        alert_state = "NONE"

        if target is not None:
            target_id = target.get("id", -1)
            risk = float(target.get("risk", -1.0))
            raw_risk = float(target.get("raw_risk", risk))
            alert_state = target.get("alert_state", "NONE")

        self.csv_writer.writerow([
            f"{time.time():.3f}",
            self.frame_index,
            source,
            int(lr),
            int(fb),
            int(ud),
            int(yaw),
            target_id,
            f"{risk:.4f}",
            f"{raw_risk:.4f}",
            alert_state,
        ])
        self.log_file.flush()

    def frame(self):
        ok, frame = self.cap.read()

        if not ok:
            if self.loop_video and self.video_path:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_index = 0
                self.loop_count += 1
                ok, frame = self.cap.read()

            if not ok:
                raise RuntimeError("[SIM] Video ended or frame read failed.")

        self.frame_index += 1
        pos_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        if pos_msec > 0:
            local_ts = pos_msec / 1000.0
        else:
            local_ts = self.frame_index / max(1e-3, self.source_fps)

        if self.source_duration > 0:
            self.last_frame_ts = self.loop_count * self.source_duration + local_ts
        else:
            self.last_frame_ts += 1.0 / max(1e-3, self.source_fps)

        elapsed = time.time() - self.start_time
        self.battery = max(0, 100 - int(elapsed // 120))

        return frame

    def get_frame_timestamp(self):
        return float(self.last_frame_ts)

    def get_battery(self):
        return self.battery

    def takeoff(self):
        self.airborne = True
        print("[SIM] TAKEOFF")
        self._log_command("TAKEOFF", 0, 0, 0, 0)

    def land(self):
        self.airborne = False
        print("[SIM] LAND")
        self._log_command("LAND", 0, 0, 0, 0)

    def manual(self, lr, fb, ud, yaw):
        if not self.airborne:
            lr, fb, ud, yaw = 0, 0, 0, 0
        self._log_command("MANUAL", lr, fb, ud, yaw)

    def hover(self):
        self.manual(0, 0, 0, 0)

    def auto_follow(self, target, frame_w):
        if target is None:
            self.hover()
            return

        center = target.get("center", (frame_w // 2, 0))
        cx, _ = center
        area = target.get("area")

        error = cx - frame_w // 2
        yaw = int(error * 0.25)
        yaw = max(-50, min(50, yaw))

        fb = 0
        if area is not None:
            if area < TARGET_AREA:
                fb = 25
            elif area > TARGET_AREA * 1.3:
                fb = -20

        if not self.airborne:
            fb = 0
            yaw = 0

        self._log_command("AUTO_FOLLOW", 0, fb, 0, yaw, target=target)

    def close(self):
        try:
            self.cap.release()
        except:
            pass

        try:
            if self.log_file is not None:
                self.log_file.close()
        except:
            pass
