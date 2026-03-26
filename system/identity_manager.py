import math
import time
import cv2

from config import (
    RISK_MAX_MATCH_DIST,
    RISK_WARMUP_FRAMES,
    RISK_WINDOW,
    POSE_MIN_KEYPOINT_CONF,
    POSE_TEMPORAL_SMOOTH,
    POSE_MAX_JUMP_NORM,
    POSE_MAX_BONE_RATIO,
    POSE_LOW_QUALITY_HOLD_FRAMES,
    RISK_WATCH_ENTER,
    RISK_WATCH_EXIT,
    RISK_ALERT_ENTER,
    RISK_ALERT_EXIT,
    RISK_WATCH_ENTER_SECONDS,
    RISK_ALERT_ENTER_SECONDS,
    RISK_EXIT_SECONDS,
    RISK_FAST_WATCH,
    RISK_FAST_ALERT,
    RISK_FAST_WATCH_SECONDS,
    RISK_FAST_ALERT_SECONDS,
    RISK_FAST_WATCH_ACUTE,
    RISK_FAST_ALERT_ACUTE,
    RISK_MIN_RAW_FOR_ALERT,
)


class IdentityManager:
    POSE_SANITY_EDGES = [
        (5, 6), (11, 12),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 11), (6, 12),
    ]

    def __init__(self):
        self.next_id = 0
        self.tracks = {}

    def dist(self, a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _bbox_iou(self, a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, float(ix2 - ix1))
        ih = max(0.0, float(iy2 - iy1))
        inter = iw * ih

        area_a = max(0.0, float(ax2 - ax1)) * max(0.0, float(ay2 - ay1))
        area_b = max(0.0, float(bx2 - bx1)) * max(0.0, float(by2 - by1))
        union = max(1e-6, area_a + area_b - inter)
        return inter / union

    def _mean(self, values):
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _clamp(self, x, lo=0.0, hi=1.0):
        return max(lo, min(hi, x))

    def _norm(self, value, lo, hi):
        if hi <= lo:
            return 0.0
        return self._clamp((value - lo) / (hi - lo))

    def _extract_patch(self, frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox

        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        if x2 - x1 < 12 or y2 - y1 < 12:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.resize(gray, (64, 96), interpolation=cv2.INTER_LINEAR)

    def _patch_motion_scores(self, prev_patch, curr_patch):
        if prev_patch is None or curr_patch is None:
            return 0.0, 0.0

        if prev_patch.shape != curr_patch.shape:
            return 0.0, 0.0

        diff = cv2.absdiff(curr_patch, prev_patch)
        full_motion = float(diff.mean()) / 255.0

        upper = diff[: int(diff.shape[0] * 0.60), :]
        upper_motion = float(upper.mean()) / 255.0

        return full_motion, upper_motion

    def _pose_points_from_detection(self, pose_data):
        if not pose_data:
            return {}

        raw_points = pose_data.get("keypoints")
        if raw_points is None:
            return {}

        points = {}
        for idx, kp in enumerate(raw_points):
            if kp is None:
                continue

            if len(kp) >= 3:
                x, y, conf = float(kp[0]), float(kp[1]), float(kp[2])
            elif len(kp) == 2:
                x, y, conf = float(kp[0]), float(kp[1]), 1.0
            else:
                continue

            if conf < POSE_MIN_KEYPOINT_CONF:
                continue

            points[idx] = (x, y, conf)

        return points

    def _pose_mid(self, p1, p2):
        if p1 is None or p2 is None:
            return None
        return ((p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5)

    def _pose_scale(self, points, bbox):
        x1, y1, x2, y2 = bbox
        bbox_h = max(1.0, float(y2 - y1))
        scales = []

        ls = points.get(5)
        rs = points.get(6)
        lh = points.get(11)
        rh = points.get(12)
        if ls is not None and rs is not None:
            scales.append(math.hypot(ls[0] - rs[0], ls[1] - rs[1]))
        if lh is not None and rh is not None:
            scales.append(math.hypot(lh[0] - rh[0], lh[1] - rh[1]))

        if scales:
            return max(8.0, self._mean(scales))
        return max(8.0, bbox_h * 0.30)

    def _clean_pose_points(self, points, prev_points, bbox):
        if not points:
            return {}

        x1, y1, x2, y2 = bbox
        bw = max(1.0, float(x2 - x1))
        bh = max(1.0, float(y2 - y1))
        margin_x = 0.28 * bw
        margin_y = 0.28 * bh
        scale = self._pose_scale(points, bbox)

        cleaned = {}
        for idx, pt in points.items():
            px, py, pc = pt

            in_window = (
                (x1 - margin_x) <= px <= (x2 + margin_x)
                and (y1 - margin_y) <= py <= (y2 + margin_y)
            )
            if not in_window:
                continue

            if prev_points and idx in prev_points:
                prev = prev_points[idx]
                jump = math.hypot(px - prev[0], py - prev[1]) / max(1.0, scale)
                if jump > POSE_MAX_JUMP_NORM and pc < 0.80:
                    continue

            cleaned[idx] = pt

        if not cleaned:
            return {}

        ref = self._pose_scale(cleaned, bbox)
        for a, b in self.POSE_SANITY_EDGES:
            pa = cleaned.get(a)
            pb = cleaned.get(b)
            if pa is None or pb is None:
                continue
            edge_len = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
            if edge_len > ref * POSE_MAX_BONE_RATIO:
                if pa[2] <= pb[2]:
                    del cleaned[a]
                else:
                    del cleaned[b]

        return cleaned

    def _smooth_pose_points(self, prev_points, curr_points):
        if not curr_points:
            return {}
        if not prev_points:
            return dict(curr_points)

        smoothed = {}
        for idx, curr in curr_points.items():
            if idx not in prev_points:
                smoothed[idx] = curr
                continue

            prev = prev_points[idx]
            alpha = POSE_TEMPORAL_SMOOTH * (0.45 + 0.55 * curr[2])
            alpha = self._clamp(alpha, 0.20, 0.92)
            sx = prev[0] * (1.0 - alpha) + curr[0] * alpha
            sy = prev[1] * (1.0 - alpha) + curr[1] * alpha
            sc = max(curr[2], prev[2] * 0.88)
            smoothed[idx] = (sx, sy, sc)

        return smoothed

    def _pose_track_features(self, person, pose_data, bbox, now, pose_observed=True):
        raw_points = self._pose_points_from_detection(pose_data)
        prev_points = person.get("prev_pose_points")
        prev_ts = person.get("pose_last_ts", now)
        dt = max(1e-3, now - prev_ts)

        if not pose_observed:
            # No pose inference ran for this frame (stride skip). Keep pose
            # features stable instead of treating it as a missing detection.
            if person["last_pose_points"]:
                held = {}
                for idx, p in person["last_pose_points"].items():
                    held[idx] = (p[0], p[1], max(0.0, p[2] * 0.985))
                person["last_pose_points"] = held
            person["pose_last_ts"] = now
            return {
                "pose_flail": person["pose_flail"],
                "pose_verticality": person["pose_verticality"],
                "pose_quality": person["pose_quality"],
                "pose_stability": person["pose_stability"],
            }

        if not raw_points:
            person["pose_missing_frames"] += 1
            person["pose_flail"] *= 0.90
            person["pose_verticality"] *= 0.95
            person["pose_quality"] *= 0.88
            person["pose_stability"] *= 0.92

            if person["pose_missing_frames"] <= POSE_LOW_QUALITY_HOLD_FRAMES and person["last_pose_points"]:
                held = {}
                for idx, p in person["last_pose_points"].items():
                    held[idx] = (p[0], p[1], max(0.0, p[2] * 0.90))
                person["last_pose_points"] = held
            else:
                person["last_pose_points"] = {}
                person["prev_pose_points"] = None

            person["pose_last_ts"] = now
            return {
                "pose_flail": person["pose_flail"],
                "pose_verticality": person["pose_verticality"],
                "pose_quality": person["pose_quality"],
                "pose_stability": person["pose_stability"],
            }

        person["pose_missing_frames"] = 0

        cleaned = self._clean_pose_points(raw_points, prev_points, bbox)
        smoothed = self._smooth_pose_points(prev_points, cleaned)

        upper_ids = [5, 6, 7, 8, 9, 10, 11, 12]
        valid_upper = [smoothed[idx][2] for idx in upper_ids if idx in smoothed]
        if valid_upper:
            pose_quality = self._clamp(sum(valid_upper) / float(len(upper_ids)))
        else:
            pose_quality = 0.0

        left_shoulder = smoothed.get(5)
        right_shoulder = smoothed.get(6)
        left_hip = smoothed.get(11)
        right_hip = smoothed.get(12)
        shoulder_mid = self._pose_mid(left_shoulder, right_shoulder)
        hip_mid = self._pose_mid(left_hip, right_hip)

        if shoulder_mid is not None and hip_mid is not None:
            vx = shoulder_mid[0] - hip_mid[0]
            vy = shoulder_mid[1] - hip_mid[1]
            norm = max(1e-6, math.hypot(vx, vy))
            verticality = abs(vy) / norm
        else:
            verticality = person["pose_verticality"] * 0.95

        bbox_h = max(1.0, float(bbox[3] - bbox[1]))

        def avg_motion(indices):
            vals = []
            if not prev_points:
                return 0.0

            for idx in indices:
                prev = prev_points.get(idx)
                curr = smoothed.get(idx)
                if prev is None or curr is None:
                    continue
                dist = math.hypot(curr[0] - prev[0], curr[1] - prev[1])
                vals.append((dist / dt) / bbox_h)

            return self._mean(vals)

        limb_motion = avg_motion([7, 8, 9, 10])
        torso_motion = avg_motion([5, 6, 11, 12])

        limb_norm = self._norm(limb_motion, 0.7, 3.8)
        torso_norm = self._norm(torso_motion, 0.35, 2.0)
        raw_flail = self._clamp(limb_norm - 0.50 * torso_norm)

        wrists = [smoothed.get(9), smoothed.get(10)]
        wrists = [p for p in wrists if p is not None]
        wrist_spread = 0.0
        if shoulder_mid is not None and wrists:
            spread_vals = [
                math.hypot(p[0] - shoulder_mid[0], p[1] - shoulder_mid[1]) / bbox_h
                for p in wrists
            ]
            wrist_spread = self._norm(self._mean(spread_vals), 0.10, 0.58)

        flail = self._clamp(0.74 * raw_flail + 0.26 * wrist_spread)

        stability = 1.0
        if prev_points:
            jumps = []
            scale = self._pose_scale(smoothed, bbox)
            for idx, curr in smoothed.items():
                prev = prev_points.get(idx)
                if prev is None:
                    continue
                jumps.append(math.hypot(curr[0] - prev[0], curr[1] - prev[1]) / max(1.0, scale))
            if jumps:
                stability = 1.0 - self._norm(self._mean(jumps), 0.08, 0.65)
        stability = self._clamp(stability)

        person["pose_flail"] = 0.66 * person["pose_flail"] + 0.34 * flail
        person["pose_verticality"] = 0.72 * person["pose_verticality"] + 0.28 * verticality
        person["pose_quality"] = 0.60 * person["pose_quality"] + 0.40 * pose_quality
        person["pose_stability"] = 0.62 * person["pose_stability"] + 0.38 * stability
        person["last_pose_points"] = dict(smoothed)
        person["prev_pose_points"] = dict(smoothed)
        person["pose_last_ts"] = now

        return {
            "pose_flail": person["pose_flail"],
            "pose_verticality": person["pose_verticality"],
            "pose_quality": person["pose_quality"],
            "pose_stability": person["pose_stability"],
        }

    def _predict_center(self, track, now):
        dt = max(0.0, min(0.5, now - track["last_seen"]))
        vx, vy = track.get("velocity", (0.0, 0.0))
        return (
            track["center"][0] + vx * dt,
            track["center"][1] + vy * dt,
        )

    def _area_compatible(self, track_area, det_area):
        if track_area <= 0 or det_area <= 0:
            return True
        ratio = det_area / max(1.0, float(track_area))
        return 0.30 <= ratio <= 3.20

    def _build_assignments(self, detections, now):
        if not self.tracks or not detections:
            return {}

        candidates = []
        track_ids = list(self.tracks.keys())

        for didx, det in enumerate(detections):
            center = det["center"]
            area = det["area"]
            det_bbox = det["bbox"]

            for pid in track_ids:
                track = self.tracks[pid]

                if not self._area_compatible(track.get("area", 0), area):
                    continue

                pred_center = self._predict_center(track, now)
                d = self.dist(center, pred_center)

                speed = math.hypot(*track.get("velocity", (0.0, 0.0)))
                gate = RISK_MAX_MATCH_DIST + min(60.0, speed * 0.12)

                if d <= gate:
                    iou = self._bbox_iou(det_bbox, track.get("bbox", det_bbox))
                    dist_cost = d / max(1e-6, gate)
                    iou_cost = 1.0 - iou
                    cost = 0.68 * dist_cost + 0.32 * iou_cost
                    candidates.append((cost, d, didx, pid))

        candidates.sort(key=lambda x: (x[0], x[1]))

        assignments = {}
        used_tracks = set()
        used_detections = set()

        for _, _, didx, pid in candidates:
            if didx in used_detections or pid in used_tracks:
                continue

            assignments[didx] = pid
            used_detections.add(didx)
            used_tracks.add(pid)

        return assignments

    def compute_features(self, history):
        if len(history) < 2:
            return {
                "speed_norm": 0.0,
                "speed_cv": 0.0,
                "vertical_ratio": 0.0,
                "direction_change": 0.0,
                "inactive_seconds": 0.0,
                "progress_ratio": 0.0,
            }

        window = history[-RISK_WINDOW:]

        speeds_norm = []
        dx_values = []
        dy_values = []
        direction_changes = 0
        prev_vec = None
        inactive_seconds = 0.0
        path_px = 0.0

        for i in range(1, len(window)):
            t0, c0, _, b0 = window[i - 1]
            t1, c1, _, _ = window[i]

            dt = max(1e-3, t1 - t0)
            dx = c1[0] - c0[0]
            dy = c1[1] - c0[1]

            step_px = math.hypot(dx, dy)
            path_px += step_px
            speed_px_s = step_px / dt

            bbox_h = max(1.0, float(b0[3] - b0[1]))
            speed_n = speed_px_s / bbox_h

            speeds_norm.append(speed_n)
            dx_values.append(abs(dx))
            dy_values.append(abs(dy))

            if speed_n < 0.35:
                inactive_seconds += dt

            curr_vec = (dx, dy)
            if prev_vec is not None:
                dot = prev_vec[0] * curr_vec[0] + prev_vec[1] * curr_vec[1]
                if dot < 0:
                    direction_changes += 1
            prev_vec = curr_vec

        speed_mean = self._mean(speeds_norm)
        speed_std = math.sqrt(self._mean([(s - speed_mean) ** 2 for s in speeds_norm]))
        speed_cv = speed_std / (speed_mean + 1e-6)

        move_x = self._mean(dx_values)
        move_y = self._mean(dy_values)
        vertical_ratio = move_y / (move_x + move_y + 1e-6)

        direction_change = direction_changes / max(1, len(speeds_norm) - 1)
        net_disp = self.dist(window[0][1], window[-1][1])
        progress_ratio = net_disp / (path_px + 1e-6)

        return {
            "speed_norm": speed_mean,
            "speed_cv": speed_cv,
            "vertical_ratio": vertical_ratio,
            "direction_change": direction_change,
            "inactive_seconds": inactive_seconds,
            "progress_ratio": self._clamp(progress_ratio),
        }

    def compute_risk_v2(
        self,
        history,
        appearance_motion=0.0,
        upper_motion=0.0,
        precomputed_features=None,
        pose_signals=None,
    ):
        if len(history) < RISK_WARMUP_FRAMES:
            early_texture = self._norm(appearance_motion, 0.030, 0.16)
            early_upper = self._norm(upper_motion, 0.030, 0.16)
            early = 0.06 + 0.26 * early_texture + 0.34 * early_upper
            return self._clamp(early, 0.0, 0.62)

        f = precomputed_features if precomputed_features is not None else self.compute_features(history)
        pose_signals = pose_signals or {}

        inactivity = self._norm(f["inactive_seconds"], 0.6, 2.5)

        panic_band = 1.0 - self._clamp(abs(f["speed_norm"] - 1.8) / 1.8)
        panic_unstable = self._norm(f["speed_cv"], 0.45, 1.4)
        panic_vertical = self._norm(f["vertical_ratio"], 0.5, 0.9)
        panic_turns = self._norm(f["direction_change"], 0.2, 0.75)
        panic = panic_band * (
            0.45 * panic_unstable
            + 0.30 * panic_vertical
            + 0.25 * panic_turns
        )

        # Drowning can appear as low-speed vertical struggle. This branch
        # increases sensitivity when movement is erratic but not traveling.
        low_speed = 1.0 - self._norm(f["speed_norm"], 0.7, 1.6)
        panic_low_speed = low_speed * (
            0.55 * panic_vertical
            + 0.45 * panic_unstable
        )

        # Key drowning cue: flailing/struggle motion without forward progress.
        struggle_motion = self._norm(f["speed_norm"], 0.8, 2.4)
        low_progress = 1.0 - self._norm(f["progress_ratio"], 0.25, 0.70)
        struggle_progress = struggle_motion * low_progress * (
            0.50 * panic_vertical
            + 0.30 * panic_unstable
            + 0.20 * panic_turns
        )

        # Visual flutter cue from within person crop (arms/splash turbulence).
        texture_motion = self._norm(appearance_motion, 0.035, 0.20)
        upper_flutter = self._norm(upper_motion, 0.032, 0.18)
        flutter_cue = low_progress * (
            0.65 * upper_flutter
            + 0.35 * texture_motion
        )

        swim_fast = self._norm(f["speed_norm"], 1.2, 3.2)
        swim_stable = 1.0 - self._clamp(f["speed_cv"] / 0.5)
        swim_directional = 1.0 - self._clamp(f["direction_change"] / 0.45)
        swim_progress = self._norm(f["progress_ratio"], 0.35, 0.85)
        swim = swim_fast * swim_progress * (0.45 * swim_stable + 0.55 * swim_directional)
        calm_swim = swim_progress * (1.0 - upper_flutter)

        pose_quality = self._clamp(pose_signals.get("pose_quality", 0.0))
        pose_flail = self._clamp(pose_signals.get("pose_flail", 0.0))
        pose_verticality = self._clamp(pose_signals.get("pose_verticality", 0.0))
        pose_stability = self._clamp(pose_signals.get("pose_stability", 0.0))
        pose_weight = pose_quality * (0.55 + 0.45 * pose_stability)
        pose_struggle = pose_weight * low_progress * (
            0.62 * pose_flail
            + 0.38 * (pose_verticality * low_progress)
        )
        pose_swim = pose_weight * (1.0 - pose_verticality) * swim_progress

        risk = 0.05
        risk += 0.42 * inactivity
        risk += 0.54 * panic
        risk += 0.40 * panic_low_speed
        risk += 0.48 * struggle_progress
        risk += 0.58 * flutter_cue
        risk += 0.34 * pose_struggle
        risk -= 0.28 * swim
        risk -= 0.18 * calm_swim
        risk -= 0.16 * pose_swim

        return self._clamp(risk, 0.0, 1.0)

    def compute_acute_distress(self, features, appearance_motion, upper_motion, pose_signals=None):
        pose_signals = pose_signals or {}
        low_progress = 1.0 - self._norm(features.get("progress_ratio", 0.0), 0.25, 0.70)
        vertical = self._norm(features.get("vertical_ratio", 0.0), 0.5, 0.9)
        unstable = self._norm(features.get("speed_cv", 0.0), 0.45, 1.4)
        upper_flutter = self._norm(upper_motion, 0.032, 0.18)
        texture_motion = self._norm(appearance_motion, 0.035, 0.20)
        mid_speed_band = 1.0 - self._clamp(abs(features.get("speed_norm", 0.0) - 1.7) / 1.7)
        pose_flail = self._clamp(pose_signals.get("pose_flail", 0.0))
        pose_verticality = self._clamp(pose_signals.get("pose_verticality", 0.0))
        pose_quality = self._clamp(pose_signals.get("pose_quality", 0.0))
        pose_stability = self._clamp(pose_signals.get("pose_stability", 0.0))
        pose_weight = pose_quality * (0.55 + 0.45 * pose_stability)
        pose_distress = pose_weight * low_progress * (
            0.55 * pose_flail
            + 0.45 * (pose_verticality * low_progress)
        )

        acute = low_progress * (
            0.30 * vertical
            + 0.22 * unstable
            + 0.30 * upper_flutter
            + 0.12 * texture_motion
            + 0.06 * mid_speed_band
        )
        acute += 0.10 * pose_distress
        return self._clamp(acute)

    def apply_risk_persistence(self, person, raw_risk, acute_distress, now):
        state = person["alert_state"]
        progress_ratio = person.get("features", {}).get("progress_ratio", 1.0)
        upper_motion = person.get("features", {}).get("upper_motion", 0.0)
        low_progress = 1.0 - self._norm(progress_ratio, 0.25, 0.70)
        upper_flutter = self._norm(upper_motion, 0.032, 0.18)

        distress_shape = low_progress * upper_flutter
        watch_enter = self._clamp(
            RISK_WATCH_ENTER - 0.12 * low_progress - 0.08 * distress_shape,
            0.30,
            0.95,
        )
        alert_enter = self._clamp(
            RISK_ALERT_ENTER - 0.10 * low_progress - 0.10 * distress_shape,
            0.46,
            0.98,
        )
        watch_enter_seconds = max(
            0.15,
            RISK_WATCH_ENTER_SECONDS * (1.0 - 0.62 * low_progress - 0.25 * distress_shape),
        )
        alert_enter_seconds = max(
            0.30,
            RISK_ALERT_ENTER_SECONDS * (1.0 - 0.45 * low_progress - 0.35 * distress_shape),
        )

        fast_watch_thr = max(0.36, RISK_FAST_WATCH - 0.14 * low_progress)
        fast_alert_thr = max(0.46, RISK_FAST_ALERT - 0.20 * low_progress)
        fast_alert_gate = max(0.62, fast_alert_thr)
        fast_watch_seconds = max(0.16, RISK_FAST_WATCH_SECONDS * (1.0 - 0.45 * low_progress))
        fast_alert_seconds = max(0.16, RISK_FAST_ALERT_SECONDS * (1.0 - 0.45 * low_progress))

        prev_raw_risk = float(person.get("prev_raw_risk", raw_risk))
        risk_rise = raw_risk - prev_raw_risk
        if risk_rise >= 0.03:
            watch_enter_seconds *= 0.78
            alert_enter_seconds *= 0.84
            fast_alert_seconds *= 0.75

        # Fast path: if cues are strongly drowning-like, escalate early.
        fast_watch_condition = (
            raw_risk >= fast_watch_thr
            and (
                (acute_distress >= RISK_FAST_WATCH_ACUTE and upper_flutter >= 0.22)
                or (low_progress >= 0.72 and upper_flutter >= 0.26)
            )
        )
        if fast_watch_condition:
            if person["fast_watch_since"] is None:
                person["fast_watch_since"] = now
        else:
            person["fast_watch_since"] = None

        fast_alert_condition = (
            raw_risk >= fast_alert_gate
            and (
                (acute_distress >= RISK_FAST_ALERT_ACUTE and upper_flutter >= 0.30)
                or (low_progress >= 0.78 and upper_flutter >= 0.28)
                or (raw_risk >= 0.60 and low_progress >= 0.72 and upper_flutter >= 0.38)
            )
        )
        if fast_alert_condition:
            if person["fast_alert_since"] is None:
                person["fast_alert_since"] = now
        else:
            person["fast_alert_since"] = None

        if person["fast_watch_since"] is not None:
            if now - person["fast_watch_since"] >= fast_watch_seconds:
                if state == "SAFE":
                    state = "WATCH"
                person["watch_since"] = None

        if person["fast_alert_since"] is not None:
            if now - person["fast_alert_since"] >= fast_alert_seconds:
                state = "ALERT"
                person["alert_since"] = None

        if state == "SAFE":
            person["watch_to_alert_since"] = None
            if raw_risk >= watch_enter:
                if person["watch_since"] is None:
                    person["watch_since"] = now
                if now - person["watch_since"] >= watch_enter_seconds:
                    state = "WATCH"
                    person["watch_since"] = None
            else:
                person["watch_since"] = None

        elif state == "WATCH":
            # Sustained struggle in WATCH should escalate even if raw score
            # is noisy around threshold.
            if raw_risk >= RISK_MIN_RAW_FOR_ALERT and low_progress >= 0.70 and upper_flutter >= 0.32:
                if person["watch_to_alert_since"] is None:
                    person["watch_to_alert_since"] = now
                if now - person["watch_to_alert_since"] >= 0.38:
                    state = "ALERT"
                    person["watch_to_alert_since"] = None
            else:
                person["watch_to_alert_since"] = None

            if raw_risk >= max(alert_enter, RISK_MIN_RAW_FOR_ALERT):
                if person["alert_since"] is None:
                    person["alert_since"] = now
                if now - person["alert_since"] >= alert_enter_seconds:
                    state = "ALERT"
                    person["alert_since"] = None
            else:
                person["alert_since"] = None

            if raw_risk < RISK_WATCH_EXIT:
                if person["deescalate_since"] is None:
                    person["deescalate_since"] = now
                if now - person["deescalate_since"] >= RISK_EXIT_SECONDS:
                    state = "SAFE"
                    person["deescalate_since"] = None
            else:
                person["deescalate_since"] = None

        else:  # ALERT
            person["watch_to_alert_since"] = None
            if raw_risk < RISK_ALERT_EXIT:
                if person["deescalate_since"] is None:
                    person["deescalate_since"] = now
                if now - person["deescalate_since"] >= RISK_EXIT_SECONDS:
                    state = "WATCH"
                    person["deescalate_since"] = None
            else:
                person["deescalate_since"] = None

        person["alert_state"] = state
        person["prev_raw_risk"] = raw_risk

        if state == "ALERT":
            stable_risk = max(raw_risk, 0.80)
        elif state == "WATCH":
            stable_risk = max(raw_risk, 0.46)
        else:
            stable_risk = min(raw_risk, 0.31)

        return self._clamp(stable_risk)

    def update(self, frame, detections, frame_ts=None):

        now = time.time() if frame_ts is None else float(frame_ts)
        persons = []
        assignments = self._build_assignments(detections, now)

        for didx, det in enumerate(detections):

            center = det["center"]
            area = det["area"]

            matched_id = assignments.get(didx)

            if matched_id is None:

                matched_id = self.next_id
                self.next_id += 1

                self.tracks[matched_id] = {
                    "id": matched_id,
                    "center": center,
                    "bbox": det["bbox"],
                    "area": area,
                    "history": [],
                    "features": {},
                    "appearance_motion": 0.0,
                    "upper_motion": 0.0,
                    "pose_flail": 0.0,
                    "pose_verticality": 0.0,
                    "pose_quality": 0.0,
                    "pose_stability": 0.0,
                    "pose_missing_frames": 0,
                    "last_pose_points": {},
                    "prev_pose_points": None,
                    "pose_last_ts": now,
                    "prev_patch": None,
                    "velocity": (0.0, 0.0),
                    "alert_state": "SAFE",
                    "watch_since": None,
                    "alert_since": None,
                    "deescalate_since": None,
                    "watch_to_alert_since": None,
                    "fast_watch_since": None,
                    "fast_alert_since": None,
                    "risk_memory": 0.0,
                    "risk": 0.0,
                    "raw_risk": 0.0,
                    "acute_distress": 0.0,
                    "last_seen": now,
                }

            person = self.tracks[matched_id]
            prev_center = person["center"]
            prev_seen = person["last_seen"]

            person["center"] = center
            person["bbox"] = det["bbox"]
            person["area"] = area
            person["last_seen"] = now

            dt_track = max(1e-3, now - prev_seen)
            inst_vx = (center[0] - prev_center[0]) / dt_track
            inst_vy = (center[1] - prev_center[1]) / dt_track
            old_vx, old_vy = person.get("velocity", (0.0, 0.0))
            person["velocity"] = (
                0.75 * old_vx + 0.25 * inst_vx,
                0.75 * old_vy + 0.25 * inst_vy,
            )

            person["history"].append((now, center, area, det["bbox"]))

            if len(person["history"]) > 60:
                person["history"].pop(0)

            curr_patch = self._extract_patch(frame, det["bbox"])
            raw_app, raw_upper = self._patch_motion_scores(person["prev_patch"], curr_patch)
            person["prev_patch"] = curr_patch

            person["appearance_motion"] = 0.70 * person["appearance_motion"] + 0.30 * raw_app
            person["upper_motion"] = 0.65 * person["upper_motion"] + 0.35 * raw_upper

            pose_signals = self._pose_track_features(
                person,
                det.get("pose"),
                det["bbox"],
                now,
                pose_observed=det.get("pose_observed", True),
            )

            person["features"] = self.compute_features(person["history"])
            person["features"]["appearance_motion"] = person["appearance_motion"]
            person["features"]["upper_motion"] = person["upper_motion"]
            person["features"]["pose_flail"] = pose_signals["pose_flail"]
            person["features"]["pose_verticality"] = pose_signals["pose_verticality"]
            person["features"]["pose_quality"] = pose_signals["pose_quality"]
            person["features"]["pose_stability"] = pose_signals["pose_stability"]

            risk = self.compute_risk_v2(
                person["history"],
                appearance_motion=person["appearance_motion"],
                upper_motion=person["upper_motion"],
                precomputed_features=person["features"],
                pose_signals=pose_signals,
            )
            person["acute_distress"] = self.compute_acute_distress(
                person["features"],
                person["appearance_motion"],
                person["upper_motion"],
                pose_signals=pose_signals,
            )
            person["features"]["acute_distress"] = person["acute_distress"]

            # Early boost to reduce delayed suspicion on strong distress cues.
            upper_flutter = self._norm(person["features"].get("upper_motion", 0.0), 0.032, 0.18)
            pose_flutter = self._norm(person["features"].get("pose_flail", 0.0), 0.18, 0.72)
            risk = self._clamp(
                risk
                + 0.20 * person["acute_distress"]
                + 0.24 * person["acute_distress"] * (0.82 * upper_flutter + 0.18 * pose_flutter)
            )

            alpha = 0.38 if risk > person["risk_memory"] else 0.06
            distress_combo = person["acute_distress"] * (0.82 * upper_flutter + 0.18 * pose_flutter)
            if distress_combo >= 0.08 and risk > person["risk_memory"]:
                alpha = max(alpha, 0.55)
            if distress_combo >= 0.13 and risk > person["risk_memory"]:
                alpha = max(alpha, 0.72)
            person["risk_memory"] = (
                person["risk_memory"] * (1.0 - alpha)
                + risk * alpha
            )
            person["raw_risk"] = self._clamp(person["risk_memory"])

            person["risk"] = self.apply_risk_persistence(
                person, person["raw_risk"], person["acute_distress"], now
            )

            persons.append({
                "id": matched_id,
                "bbox": det["bbox"],
                "center": center,
                "area": area,
                "risk": person["risk"],
                "raw_risk": person["raw_risk"],
                "alert_state": person["alert_state"],
                "acute_distress": person["acute_distress"],
                "features": person["features"],
                "pose_points": person.get("last_pose_points", {}),
            })

        remove = []

        for pid, p in self.tracks.items():
            if now - p["last_seen"] > 4:
                remove.append(pid)

        for r in remove:
            del self.tracks[r]

        return persons
