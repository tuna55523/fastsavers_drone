import csv
import os
import time

import cv2

from config import (
    TARGET_AREA,
    REAL_RC_FB_SLEW_STEP,
    AUTO_TRACK_YAW_KP,
    AUTO_TRACK_YAW_DEADZONE_PX,
    AUTO_TRACK_YAW_MIN_CMD,
    AUTO_TRACK_YAW_MAX,
    AUTO_TRACK_FB_BAND,
    AUTO_TRACK_FB_MIN_RATIO,
    AUTO_TRACK_FB_MAX_RATIO,
    AUTO_TRACK_FB_MAX_FWD,
    AUTO_TRACK_FB_MAX_BWD,
    AUTO_TRACK_FB_MIN_STEP,
    AUTO_TRACK_WATCH_FB_BOOST,
    AUTO_TRACK_ALERT_FB_BOOST,
    AUTO_TRACK_WATCH_YAW_BOOST,
    AUTO_TRACK_ALERT_YAW_BOOST,
    AUTO_TRACK_WATCH_TARGET_AREA_SCALE,
    AUTO_TRACK_ALERT_TARGET_AREA_SCALE,
    AUTO_TRACK_RISE_RATE_TRIG,
    AUTO_TRACK_RISE_FB_BONUS,
    AUTO_TRACK_RISE_YAW_BONUS,
    AUTO_TRACK_ACUTE_TRIG,
    AUTO_TRACK_ACUTE_FB_BONUS,
)


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
        self._auto_fb_prev = 0

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
            "op_state",
            "lr",
            "fb",
            "ud",
            "yaw",
            "target_id",
            "risk",
            "raw_risk",
            "risk_rise_rate",
            "alert_state",
        ])
        self.log_file.flush()

    def _log_command(self, source, lr, fb, ud, yaw, target=None, op_state="NONE"):
        if self.csv_writer is None:
            return

        target_id = -1
        risk = -1.0
        raw_risk = -1.0
        risk_rise_rate = 0.0
        alert_state = "NONE"

        if target is not None:
            target_id = target.get("id", -1)
            risk = float(target.get("risk", -1.0))
            raw_risk = float(target.get("raw_risk", risk))
            risk_rise_rate = float(target.get("risk_rise_rate", 0.0))
            alert_state = target.get("alert_state", "NONE")

        self.csv_writer.writerow([
            f"{time.time():.3f}",
            self.frame_index,
            source,
            str(op_state),
            int(lr),
            int(fb),
            int(ud),
            int(yaw),
            target_id,
            f"{risk:.4f}",
            f"{raw_risk:.4f}",
            f"{risk_rise_rate:.4f}",
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
        self._auto_fb_prev = 0
        print("[SIM] LAND")
        self._log_command("LAND", 0, 0, 0, 0)

    def manual(self, lr, fb, ud, yaw):
        self._auto_fb_prev = 0
        if not self.airborne:
            lr, fb, ud, yaw = 0, 0, 0, 0
        self._log_command("MANUAL", lr, fb, ud, yaw)

    def hover(self):
        self.manual(0, 0, 0, 0)

    def _slew_axis(self, prev, target, step):
        delta = target - prev
        if delta > step:
            delta = step
        elif delta < -step:
            delta = -step
        return int(prev + delta)

    def _compute_follow_fb(self, area, target_area=None, fb_boost=1.0):
        if area is None:
            target_fb = 0
        else:
            area = float(area)
            target = float(TARGET_AREA if target_area is None else target_area)
            band_lo = target * (1.0 - AUTO_TRACK_FB_BAND)
            band_hi = target * (1.0 + AUTO_TRACK_FB_BAND)
            max_fwd = max(
                int(AUTO_TRACK_FB_MIN_STEP),
                int(round(float(AUTO_TRACK_FB_MAX_FWD) * max(1.0, float(fb_boost)))),
            )

            if area < target * AUTO_TRACK_FB_MIN_RATIO:
                target_fb = int(max_fwd)
            elif area > target * AUTO_TRACK_FB_MAX_RATIO:
                target_fb = -int(AUTO_TRACK_FB_MAX_BWD)
            elif band_lo <= area <= band_hi:
                target_fb = 0
            elif area < band_lo:
                ratio = (band_lo - area) / max(1.0, band_lo)
                target_fb = int(max(AUTO_TRACK_FB_MIN_STEP, min(max_fwd, ratio * max_fwd)))
            else:
                ratio = (area - band_hi) / max(1.0, band_hi)
                target_fb = -int(max(AUTO_TRACK_FB_MIN_STEP, min(AUTO_TRACK_FB_MAX_BWD, ratio * AUTO_TRACK_FB_MAX_BWD)))

        self._auto_fb_prev = self._slew_axis(self._auto_fb_prev, int(target_fb), max(1, int(REAL_RC_FB_SLEW_STEP)))
        return int(self._auto_fb_prev)

    def _follow_response_profile(self, target, op_state="TRACK"):
        state = str(target.get("alert_state", "SAFE"))
        acute = float(target.get("acute_distress", 0.0))
        rise_rate = float(target.get("risk_rise_rate", 0.0))
        op_state = str(op_state or "TRACK").upper()

        fb_boost = 1.0
        yaw_boost = 1.0
        area_scale = 1.0

        if state == "WATCH":
            fb_boost = max(fb_boost, float(AUTO_TRACK_WATCH_FB_BOOST))
            yaw_boost = max(yaw_boost, float(AUTO_TRACK_WATCH_YAW_BOOST))
            area_scale = max(area_scale, float(AUTO_TRACK_WATCH_TARGET_AREA_SCALE))
        elif state == "ALERT":
            fb_boost = max(fb_boost, float(AUTO_TRACK_ALERT_FB_BOOST))
            yaw_boost = max(yaw_boost, float(AUTO_TRACK_ALERT_YAW_BOOST))
            area_scale = max(area_scale, float(AUTO_TRACK_ALERT_TARGET_AREA_SCALE))

        if op_state == "RESCUE":
            fb_boost = max(fb_boost, float(AUTO_TRACK_ALERT_FB_BOOST))
            yaw_boost = max(yaw_boost, float(AUTO_TRACK_ALERT_YAW_BOOST))
            area_scale = max(area_scale, float(AUTO_TRACK_ALERT_TARGET_AREA_SCALE))

        if rise_rate >= float(AUTO_TRACK_RISE_RATE_TRIG):
            fb_boost += float(AUTO_TRACK_RISE_FB_BONUS)
            yaw_boost += float(AUTO_TRACK_RISE_YAW_BONUS)

        if acute >= float(AUTO_TRACK_ACUTE_TRIG):
            fb_boost += float(AUTO_TRACK_ACUTE_FB_BONUS)

        fb_boost = max(1.0, min(1.95, fb_boost))
        yaw_boost = max(1.0, min(1.60, yaw_boost))
        area_scale = max(1.0, min(1.35, area_scale))
        return fb_boost, yaw_boost, area_scale

    def auto_follow(self, target, frame_w, op_state="TRACK"):
        if target is None:
            self._auto_fb_prev = self._slew_axis(self._auto_fb_prev, 0, max(1, int(REAL_RC_FB_SLEW_STEP)))
            self._log_command("AUTO_FOLLOW", 0, self._auto_fb_prev, 0, 0, target=target, op_state=op_state)
            return

        center = target.get("center", (frame_w // 2, 0))
        cx, _ = center
        area = target.get("area")
        fb_boost, yaw_boost, area_scale = self._follow_response_profile(target, op_state=op_state)

        errx = float(cx - frame_w // 2)
        yaw = 0
        deadzone_px = float(AUTO_TRACK_YAW_DEADZONE_PX) / max(1.0, yaw_boost)
        if abs(errx) > deadzone_px:
            yaw = int(max(-AUTO_TRACK_YAW_MAX, min(AUTO_TRACK_YAW_MAX, errx * AUTO_TRACK_YAW_KP * yaw_boost)))
            if 0 < abs(yaw) < int(AUTO_TRACK_YAW_MIN_CMD):
                yaw = int((1 if yaw > 0 else -1) * int(AUTO_TRACK_YAW_MIN_CMD))

        fb = self._compute_follow_fb(
            area,
            target_area=float(TARGET_AREA) * area_scale,
            fb_boost=fb_boost,
        )

        if not self.airborne:
            fb = 0
            yaw = 0

        self._log_command("AUTO_FOLLOW", 0, fb, 0, yaw, target=target, op_state=op_state)

    def search_scan(self, yaw_speed):
        yaw = int(yaw_speed)
        if not self.airborne:
            yaw = 0
        self._auto_fb_prev = self._slew_axis(self._auto_fb_prev, 0, max(1, int(REAL_RC_FB_SLEW_STEP)))
        self._log_command("AUTO_SEARCH", 0, self._auto_fb_prev, 0, yaw)

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
