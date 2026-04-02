import threading
import time

from config import (
    TARGET_AREA,
    REAL_RC_HZ,
    REAL_RC_CMD_TIMEOUT_SEC,
    REAL_RC_SLEW_STEP,
    REAL_RC_FB_SLEW_STEP,
    REAL_LINK_FAIL_STREAK_TRIG,
    REAL_LINK_LOST_SEC_TRIG,
    REAL_LAND_RETRY_COOLDOWN_SEC,
    REAL_FAILSAFE_AUTO_LAND,
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


class AdvancedFlight:

    def __init__(self, tello):
        self.tello = tello

        self._lock = threading.Lock()
        self._sdk_lock = threading.Lock()
        self._command_busy = False
        self._targets = {
            "manual": (0, 0, 0, 0),
            "auto": (0, 0, 0, 0),
        }
        self._active_channel = "manual"
        self._last_command_ts = time.time()

        self._last_sent = (0, 0, 0, 0)
        self._is_airborne = False

        self._auto_fb_prev = 0

        self._link_fail_streak = 0
        self._last_link_ok_ts = time.time()
        self._last_land_attempt_ts = 0.0

        self._running = True
        self._dt = 1.0 / max(1.0, float(REAL_RC_HZ))
        self._rc_thread = threading.Thread(target=self._sender_loop, daemon=True)
        self._rc_thread.start()

    def _clamp(self, value, lo=-100, hi=100):
        return int(max(lo, min(hi, int(value))))

    def _slew_axis(self, prev, target, step):
        delta = target - prev
        if delta > step:
            delta = step
        elif delta < -step:
            delta = -step
        return int(prev + delta)

    def _note_link_ok(self):
        self._link_fail_streak = 0
        self._last_link_ok_ts = time.time()

    def _failsafe_land(self, reason=""):
        if not REAL_FAILSAFE_AUTO_LAND:
            return False

        now = time.time()
        if not self._is_airborne:
            return False

        if now - self._last_land_attempt_ts < REAL_LAND_RETRY_COOLDOWN_SEC:
            return False

        self._last_land_attempt_ts = now
        self._set_target("manual", 0, 0, 0, 0)

        for fn in [self.tello.land, lambda: self.tello.send_command_without_return("land")]:
            try:
                fn()
                self._is_airborne = False
                print(f"[FLIGHT] Failsafe land triggered ({reason})")
                return True
            except Exception:
                continue

        return False

    def _note_link_fail(self, reason=""):
        self._link_fail_streak += 1
        if self._link_fail_streak >= REAL_LINK_FAIL_STREAK_TRIG:
            self._failsafe_land(reason)
            return

        if (time.time() - self._last_link_ok_ts) >= REAL_LINK_LOST_SEC_TRIG:
            self._failsafe_land(reason)

    def _set_target(self, channel, lr, fb, ud, yaw):
        cmd = (
            self._clamp(lr),
            self._clamp(fb),
            self._clamp(ud),
            self._clamp(yaw),
        )
        with self._lock:
            self._targets[channel] = cmd
            self._active_channel = channel
            self._last_command_ts = time.time()

    def run_sdk(self, fn):
        with self._sdk_lock:
            return fn()

    def _sender_loop(self):
        while self._running:
            now = time.time()
            with self._lock:
                channel = self._active_channel
                target = self._targets.get(channel, (0, 0, 0, 0))
                last_command_ts = self._last_command_ts

            if now - last_command_ts > REAL_RC_CMD_TIMEOUT_SEC:
                target = (0, 0, 0, 0)

            if not self._is_airborne:
                self._last_sent = (0, 0, 0, 0)
                time.sleep(self._dt)
                continue

            if self._command_busy:
                time.sleep(self._dt)
                continue

            step = max(1, int(REAL_RC_SLEW_STEP))
            fb_step = max(step, int(REAL_RC_FB_SLEW_STEP))

            plr, pfb, pud, pyaw = self._last_sent
            tlr, tfb, tud, tyaw = target
            next_cmd = (
                self._slew_axis(plr, tlr, step),
                self._slew_axis(pfb, tfb, fb_step),
                self._slew_axis(pud, tud, step),
                self._slew_axis(pyaw, tyaw, step),
            )

            try:
                locked = self._sdk_lock.acquire(blocking=False)
                if not locked:
                    time.sleep(self._dt)
                    continue
                try:
                    self.tello.send_rc_control(*next_cmd)
                finally:
                    self._sdk_lock.release()
                self._last_sent = next_cmd
                self._note_link_ok()
            except Exception:
                self._note_link_fail("rc_sender")

            time.sleep(self._dt)

    def takeoff(self):
        self._set_target("manual", 0, 0, 0, 0)
        self._command_busy = True

        ok = False
        for fn in [self.tello.takeoff, lambda: self.tello.send_command_without_return("takeoff")]:
            try:
                self.run_sdk(fn)
                ok = True
                break
            except Exception:
                continue

        self._command_busy = False

        if not ok:
            raise RuntimeError("Takeoff failed")

        self._is_airborne = True
        self._last_command_ts = time.time()
        time.sleep(0.25)

    def land(self):
        self._set_target("manual", 0, 0, 0, 0)
        self._command_busy = True

        ok = False
        for fn in [self.tello.land, lambda: self.tello.send_command_without_return("land")]:
            try:
                self.run_sdk(fn)
                ok = True
                break
            except Exception:
                continue

        self._command_busy = False

        if not ok:
            raise RuntimeError("Land failed")

        self._is_airborne = False
        self._auto_fb_prev = 0

    def hover(self):
        self._auto_fb_prev = 0
        self._set_target(self._active_channel, 0, 0, 0, 0)

    def manual(self, lr, fb, ud, yaw):
        self._auto_fb_prev = 0
        self._set_target("manual", lr, fb, ud, yaw)

    def search_scan(self, yaw_speed):
        self._auto_fb_prev = self._slew_axis(self._auto_fb_prev, 0, max(1, int(REAL_RC_FB_SLEW_STEP)))
        self._set_target("auto", 0, self._auto_fb_prev, 0, yaw_speed)

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

        self._auto_fb_prev = self._slew_axis(
            self._auto_fb_prev,
            int(target_fb),
            max(1, int(REAL_RC_FB_SLEW_STEP)),
        )
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

    def follow(self, target, frame_w, op_state="TRACK"):
        if target is None:
            self._auto_fb_prev = self._slew_axis(self._auto_fb_prev, 0, max(1, int(REAL_RC_FB_SLEW_STEP)))
            self._set_target("auto", 0, self._auto_fb_prev, 0, 0)
            return

        center = target.get("center", (frame_w // 2, 0))
        cx, _ = center
        area = target.get("area")
        fb_boost, yaw_boost, area_scale = self._follow_response_profile(target, op_state=op_state)

        errx = float(cx - frame_w // 2)
        yaw = 0
        deadzone_px = float(AUTO_TRACK_YAW_DEADZONE_PX) / max(1.0, yaw_boost)
        if abs(errx) > deadzone_px:
            yaw_raw = errx * float(AUTO_TRACK_YAW_KP) * yaw_boost
            yaw = self._clamp(yaw_raw, -int(AUTO_TRACK_YAW_MAX), int(AUTO_TRACK_YAW_MAX))
            if 0 < abs(yaw) < int(AUTO_TRACK_YAW_MIN_CMD):
                yaw = int((1 if yaw > 0 else -1) * int(AUTO_TRACK_YAW_MIN_CMD))

        fb = self._compute_follow_fb(
            area,
            target_area=float(TARGET_AREA) * area_scale,
            fb_boost=fb_boost,
        )
        self._set_target("auto", 0, fb, 0, yaw)

    def close(self):
        self._running = False
        try:
            self._rc_thread.join(timeout=0.4)
        except Exception:
            pass

        try:
            self.tello.send_rc_control(0, 0, 0, 0)
        except Exception:
            pass
