import os
import math
import cv2
from ultralytics import YOLO
from system.identity_manager import IdentityManager
from config import (
    MODEL_PATH,
    POSE_MODEL_PATH,
    POSE_MODEL_ID,
    POSE_AUTO_DOWNLOAD,
    POSE_FALLBACK_MODEL_PATH,
    CONF_THRESHOLD,
    POSE_ENABLED,
    POSE_CONF_THRESHOLD,
    POSE_MIN_KEYPOINT_CONF,
    POSE_INFER_EVERY_N_FRAMES,
    POSE_DRAW_OVERLAY,
    POSE_RUN_ON_CROPS,
    POSE_BATCH_INFERENCE,
    POSE_CROP_EXPAND,
    POSE_CROP_MIN_SIDE,
    IGNORE_CLASSES,
    WATER_FILTER_ENABLED,
    WATER_LINE_RATIO,
)


class DetectTrackSystem:
    COCO_SKELETON_EDGES = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 6),
        (5, 7), (7, 9),
        (6, 8), (8, 10),
        (5, 11), (6, 12), (11, 12),
        (11, 13), (13, 15),
        (12, 14), (14, 16),
    ]

    def __init__(self):
        print("[VISION] Loading model...")
        self.model = YOLO(MODEL_PATH)
        self.frame_idx = 0
        self.person_class_ids = self._resolve_person_class_ids()

        self.pose_model = self._load_pose_model()

        self.identity = IdentityManager()

    def _resolve_person_class_ids(self):
        names = self.model.names
        ids = []

        if isinstance(names, dict):
            for cid, cname in names.items():
                if str(cname).lower() == "person":
                    ids.append(int(cid))
        elif isinstance(names, (list, tuple)):
            for cid, cname in enumerate(names):
                if str(cname).lower() == "person":
                    ids.append(int(cid))

        return ids if ids else None

    def _load_pose_model(self):
        if not POSE_ENABLED:
            print("[VISION] Pose disabled by config.")
            return None

        local_candidates = []
        if POSE_MODEL_PATH:
            local_candidates.append(POSE_MODEL_PATH)
        if POSE_FALLBACK_MODEL_PATH and POSE_FALLBACK_MODEL_PATH not in local_candidates:
            local_candidates.append(POSE_FALLBACK_MODEL_PATH)
        if POSE_MODEL_ID and POSE_MODEL_ID not in local_candidates:
            local_candidates.append(POSE_MODEL_ID)

        for candidate in local_candidates:
            if candidate and os.path.exists(candidate):
                try:
                    print(f"[VISION] Loading pose model from local path: {candidate}")
                    model = YOLO(candidate)
                    print("[VISION] Pose model ready.")
                    return model
                except Exception as e:
                    print(f"[VISION] Local pose load failed for {candidate}: {e}")

        if POSE_AUTO_DOWNLOAD and POSE_MODEL_ID:
            try:
                print(f"[VISION] Auto-downloading pose model: {POSE_MODEL_ID}")
                model = YOLO(POSE_MODEL_ID)
                print("[VISION] Pose model ready (auto-download).")
                return model
            except Exception as e:
                print(f"[VISION] Pose auto-download failed: {e}")

        print("[VISION] Pose disabled, no usable model available.")
        return None

    def reset_tracking(self):
        self.identity = IdentityManager()
        self.frame_idx = 0

    # ------------------------------------------------
    # HELPERS
    # ------------------------------------------------
    def ignore_object(self, name):
        return name in IGNORE_CLASSES

    def _class_name(self, cls_id):
        names = self.model.names
        if isinstance(names, dict):
            return str(names.get(int(cls_id), cls_id))
        if isinstance(names, (list, tuple)) and 0 <= int(cls_id) < len(names):
            return str(names[int(cls_id)])
        return str(cls_id)

    def _expand_bbox(self, bbox, shape):
        x1, y1, x2, y2 = bbox
        h, w = shape[:2]

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        pad_w = int(bw * POSE_CROP_EXPAND)
        pad_h = int(bh * POSE_CROP_EXPAND)

        side = max(POSE_CROP_MIN_SIDE, bw, bh)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        half = side // 2

        cx1 = max(0, cx - half - pad_w)
        cy1 = max(0, cy - half - pad_h)
        cx2 = min(w, cx + half + pad_w)
        cy2 = min(h, cy + half + pad_h)

        if cx2 - cx1 < 16 or cy2 - cy1 < 16:
            return None

        return (cx1, cy1, cx2, cy2)

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

    def _pose_candidates_from_result(self, result, origin=(0, 0)):
        if result.boxes is None or result.keypoints is None:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        kp_data = result.keypoints.data.cpu().numpy()

        ox, oy = origin
        candidates = []

        for idx, (conf, cls) in enumerate(zip(confs, classes)):
            if int(cls) != 0:
                continue
            if float(conf) < POSE_CONF_THRESHOLD:
                continue

            px1, py1, px2, py2 = map(int, boxes[idx])
            gx1, gy1 = ox + px1, oy + py1
            gx2, gy2 = ox + px2, oy + py2
            gcx = (gx1 + gx2) // 2
            gcy = (gy1 + gy2) // 2

            global_kps = []
            for kp in kp_data[idx]:
                if len(kp) >= 3:
                    kx, ky, kc = float(kp[0]), float(kp[1]), float(kp[2])
                elif len(kp) == 2:
                    kx, ky, kc = float(kp[0]), float(kp[1]), float(conf)
                else:
                    continue
                global_kps.append([kx + ox, ky + oy, kc])

            candidates.append({
                "bbox": (gx1, gy1, gx2, gy2),
                "center": (gcx, gcy),
                "conf": float(conf),
                "keypoints": global_kps,
            })

        return candidates

    def _pick_best_pose_candidate(self, candidates, ref_bbox, used_indices=None):
        if not candidates:
            return None, None

        if used_indices is None:
            used_indices = set()

        rx1, ry1, rx2, ry2 = ref_bbox
        rcx = 0.5 * (rx1 + rx2)
        rcy = 0.5 * (ry1 + ry2)
        rdiag = max(1.0, math.hypot(rx2 - rx1, ry2 - ry1))

        best_idx = -1
        best_score = -1.0

        for idx, cand in enumerate(candidates):
            if idx in used_indices:
                continue

            iou = self._bbox_iou(cand["bbox"], ref_bbox)
            ccx, ccy = cand["center"]
            dist = math.hypot(ccx - rcx, ccy - rcy)
            dist_score = max(0.0, 1.0 - (dist / (1.5 * rdiag)))

            score = 0.55 * iou + 0.30 * cand["conf"] + 0.15 * dist_score
            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx < 0:
            return None, None

        return best_idx, candidates[best_idx]

    def _infer_pose_on_crops(self, frame, bboxes):
        if self.pose_model is None or not bboxes:
            return {}

        crop_infos = []
        crops = []
        for det_idx, bbox in enumerate(bboxes):
            crop_bbox = self._expand_bbox(bbox, frame.shape)
            if crop_bbox is None:
                continue

            x1, y1, x2, y2 = crop_bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_infos.append({"det_idx": det_idx, "bbox": bbox, "crop_bbox": crop_bbox})
            crops.append(crop)

        if not crops:
            return {}

        try:
            if POSE_BATCH_INFERENCE:
                results = self.pose_model(crops, verbose=False)
            else:
                results = [self.pose_model(crop, verbose=False)[0] for crop in crops]
        except Exception as e:
            print(f"[VISION] Pose crop inference failed: {e}")
            return {}

        pose_by_idx = {}
        for info, result in zip(crop_infos, results):
            cx1, cy1, _, _ = info["crop_bbox"]
            candidates = self._pose_candidates_from_result(result, origin=(cx1, cy1))
            _, best = self._pick_best_pose_candidate(candidates, info["bbox"])
            if best is not None:
                pose_by_idx[info["det_idx"]] = best

        return pose_by_idx

    def _infer_pose_fullframe(self, frame, bboxes):
        if self.pose_model is None or not bboxes:
            return {}

        try:
            result = self.pose_model(frame, verbose=False)[0]
        except Exception as e:
            print(f"[VISION] Pose full-frame inference failed: {e}")
            return {}

        candidates = self._pose_candidates_from_result(result)
        pose_by_idx = {}
        used = set()

        for det_idx, bbox in enumerate(bboxes):
            cand_idx, best = self._pick_best_pose_candidate(candidates, bbox, used_indices=used)
            if cand_idx is None or best is None:
                continue
            used.add(cand_idx)
            pose_by_idx[det_idx] = best

        return pose_by_idx

    def _draw_pose(self, frame, pose_points, color):
        if not pose_points:
            return

        for a, b in self.COCO_SKELETON_EDGES:
            pa = pose_points.get(a)
            pb = pose_points.get(b)
            if pa is None or pb is None:
                continue
            if pa[2] < POSE_MIN_KEYPOINT_CONF or pb[2] < POSE_MIN_KEYPOINT_CONF:
                continue
            cv2.line(
                frame,
                (int(pa[0]), int(pa[1])),
                (int(pb[0]), int(pb[1])),
                color,
                2,
            )

        for kp in pose_points.values():
            if kp[2] < POSE_MIN_KEYPOINT_CONF:
                continue
            radius = 2 if kp[2] < 0.45 else 3
            cv2.circle(frame, (int(kp[0]), int(kp[1])), radius, color, -1)

    # ------------------------------------------------
    # MAIN RUN
    # ------------------------------------------------
    def run(self, frame, frame_ts=None):
        infer_kwargs = {
            "verbose": False,
            "conf": CONF_THRESHOLD,
        }
        if self.person_class_ids:
            infer_kwargs["classes"] = self.person_class_ids

        results = self.model(frame, **infer_kwargs)[0]
        use_pose_this_frame = (
            self.pose_model is not None
            and self.frame_idx % max(1, POSE_INFER_EVERY_N_FRAMES) == 0
        )

        raw_detections = []
        water_line = int(frame.shape[0] * WATER_LINE_RATIO) if WATER_FILTER_ENABLED else -1

        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confs):
                name = self._class_name(int(cls)).lower()

                if conf < CONF_THRESHOLD:
                    continue

                if name != "person":
                    continue

                if self.ignore_object(name):
                    continue

                x1, y1, x2, y2 = map(int, box)

                if WATER_FILTER_ENABLED and y2 <= water_line:
                    continue

                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue

                cx = x1 + w // 2
                cy = y1 + h // 2

                raw_detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "center": (cx, cy),
                    "area": w * h,
                })

        pose_by_idx = {}
        if use_pose_this_frame and raw_detections:
            bboxes = [d["bbox"] for d in raw_detections]
            if POSE_RUN_ON_CROPS:
                pose_by_idx = self._infer_pose_on_crops(frame, bboxes)
            else:
                pose_by_idx = self._infer_pose_fullframe(frame, bboxes)

        detections = []
        for idx, det in enumerate(raw_detections):
            detections.append({
                "bbox": det["bbox"],
                "center": det["center"],
                "area": det["area"],
                "pose": pose_by_idx.get(idx),
                "pose_observed": use_pose_this_frame,
            })

        persons = self.identity.update(frame, detections, frame_ts=frame_ts)
        self.frame_idx += 1

        target = None
        if persons:
            rank = {"SAFE": 0, "WATCH": 1, "ALERT": 2}
            target = max(
                persons,
                key=lambda p: (
                    rank.get(p.get("alert_state", "SAFE"), 0),
                    p.get("acute_distress", 0.0),
                    p.get("risk_rise_rate", 0.0),
                    p.get("risk", 0.0),
                ),
            )

        for p in persons:
            x1, y1, x2, y2 = p["bbox"]
            risk = p["risk"]
            raw_risk = p.get("raw_risk", risk)
            alert_state = p.get("alert_state", "SAFE")
            acute = p.get("acute_distress", 0.0)
            rise = p.get("risk_rise_rate", 0.0)
            pose_q = p.get("features", {}).get("pose_quality", 0.0)
            pose_f = p.get("features", {}).get("pose_flail", 0.0)
            pose_s = p.get("features", {}).get("pose_stability", 0.0)

            color = (0, 255, 0)
            if alert_state == "ALERT":
                color = (0, 0, 255)
            elif alert_state == "WATCH":
                color = (0, 165, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if POSE_DRAW_OVERLAY:
                self._draw_pose(frame, p.get("pose_points", {}), color)

            cv2.putText(
                frame,
                f"ID:{p['id']} R:{risk:.2f} RAW:{raw_risk:.2f} RR:{rise:.2f}/s AD:{acute:.2f} PQ:{pose_q:.2f} PF:{pose_f:.2f} PS:{pose_s:.2f} {alert_state}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return frame, target, persons
