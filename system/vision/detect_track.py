import os
import math
import cv2
import numpy as np
_YOLO_CONFIG_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".yolo_config"))
os.makedirs(_YOLO_CONFIG_BASE, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", _YOLO_CONFIG_BASE)
try:
    from ultralytics import YOLO
    _YOLO_IMPORT_ERROR = None
except Exception as yolo_import_exc:
    YOLO = None  # type: ignore[assignment]
    _YOLO_IMPORT_ERROR = yolo_import_exc
from system.identity_manager import IdentityManager
from config import (
    MODEL_PATH,
    OBJECT_MODEL_PATH,
    POSE_MODEL_PATH,
    POSE_MODEL_ID,
    POSE_AUTO_DOWNLOAD,
    POSE_FALLBACK_MODEL_PATH,
    CONF_THRESHOLD,
    OBJECT_DETECTION_ENABLED,
    OBJECT_CONF_THRESHOLD,
    OBJECT_DISPLAY_CLASSES,
    OBJECT_MAX_RESULTS,
    OBJECT_MIN_AREA_RATIO,
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
        self.model = None
        if YOLO is not None:
            try:
                self.model = YOLO(MODEL_PATH)
            except Exception as e:
                print(f"[VISION] Detection model load failed: {e}")
        else:
            print(f"[VISION] Ultralytics yok, face fallback modu aktif: {_YOLO_IMPORT_ERROR}")
        self.frame_idx = 0
        self.person_class_ids = self._resolve_person_class_ids()
        self.object_model = self._load_object_model()
        self.object_class_ids = self._resolve_named_class_ids(self.object_model, OBJECT_DISPLAY_CLASSES)
        self.face_cascade = self._load_face_cascade()

        self.pose_model = self._load_pose_model()

        self.identity = IdentityManager()

    def _resolve_person_class_ids(self):
        return self._resolve_named_class_ids(self.model, ["person"])

    def _resolve_named_class_ids(self, model, names_to_find):
        if model is None:
            return None

        wanted = {str(name).lower() for name in names_to_find if str(name).strip()}
        if not wanted:
            return None

        names = model.names
        ids = []

        if isinstance(names, dict):
            for cid, cname in names.items():
                if str(cname).lower() in wanted:
                    ids.append(int(cid))
        elif isinstance(names, (list, tuple)):
            for cid, cname in enumerate(names):
                if str(cname).lower() in wanted:
                    ids.append(int(cid))

        return ids if ids else None

    def _load_object_model(self):
        if YOLO is None:
            return None
        if not OBJECT_DETECTION_ENABLED:
            return None

        candidates = []
        if OBJECT_MODEL_PATH:
            candidates.append(OBJECT_MODEL_PATH)
            basename = os.path.basename(OBJECT_MODEL_PATH)
            if basename and basename not in candidates:
                candidates.append(basename)

        for candidate in candidates:
            if not candidate:
                continue
            if not os.path.exists(candidate):
                continue
            try:
                print(f"[VISION] Loading object model from local path: {candidate}")
                model = YOLO(candidate)
                print("[VISION] Object model ready.")
                return model
            except Exception as e:
                print(f"[VISION] Object model load failed for {candidate}: {e}")

        print("[VISION] Object detection disabled, no usable model available.")
        return None

    def _load_pose_model(self):
        if YOLO is None or self.model is None:
            print("[VISION] Pose disabled, YOLO algilama modeli hazir degil.")
            return None

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

    def _load_face_cascade(self):
        cascade_dir = getattr(cv2.data, "haarcascades", "")
        if not cascade_dir:
            return None

        cascade_path = os.path.join(cascade_dir, "haarcascade_frontalface_default.xml")
        if not os.path.exists(cascade_path):
            return None

        cascade = cv2.CascadeClassifier(cascade_path)
        if cascade.empty():
            return None
        return cascade

    def reset_tracking(self):
        self.identity = IdentityManager()
        self.frame_idx = 0

    # ------------------------------------------------
    # HELPERS
    # ------------------------------------------------
    def ignore_object(self, name):
        return name in IGNORE_CLASSES

    def _model_class_name(self, model, cls_id):
        if model is None:
            return str(cls_id)
        names = model.names
        if isinstance(names, dict):
            return str(names.get(int(cls_id), cls_id))
        if isinstance(names, (list, tuple)) and 0 <= int(cls_id) < len(names):
            return str(names[int(cls_id)])
        return str(cls_id)

    def _class_name(self, cls_id):
        return self._model_class_name(self.model, cls_id)

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

    def _clamp01(self, value):
        return max(0.0, min(1.0, float(value)))

    def _masked_ratio(self, mask, valid_mask):
        total = int(valid_mask.sum())
        if total <= 0:
            return 0.0
        return float((mask & valid_mask).sum()) / float(total)

    def _estimate_water_context(self, frame, bbox, det_source="person", water_line=-1):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        if det_source == "face":
            rx1 = max(0, x1 - int(bw * 0.70))
            rx2 = min(w, x2 + int(bw * 0.70))
            ry1 = max(0, y1 + int(bh * 0.10))
            ry2 = min(h, y2 + int(bh * 3.40))
        else:
            rx1 = max(0, x1 - int(bw * 0.28))
            rx2 = min(w, x2 + int(bw * 0.28))
            ry1 = max(0, y1 + int(bh * 0.18))
            ry2 = min(h, y2 + int(bh * 0.32))

        if rx2 - rx1 < 12 or ry2 - ry1 < 12:
            return 0.0, False

        roi = frame[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            return 0.0, False

        context_mask = np.ones(roi.shape[:2], dtype=bool)
        inner_x1 = max(0, x1 - rx1)
        inner_y1 = max(0, y1 - ry1)
        inner_x2 = min(roi.shape[1], x2 - rx1)
        inner_y2 = min(roi.shape[0], y2 - ry1)
        if inner_x2 > inner_x1 and inner_y2 > inner_y1:
            context_mask[inner_y1:inner_y2, inner_x1:inner_x2] = False

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        blue_mask = (hue >= 78) & (hue <= 138) & (sat >= 24) & (val >= 28)
        teal_mask = (hue >= 58) & (hue <= 104) & (sat >= 16) & (val >= 24)
        water_mask = (blue_mask | teal_mask) & context_mask

        half = water_mask.shape[0] // 2
        lower_water = water_mask[half:, :]
        lower_valid = context_mask[half:, :]

        overall_ratio = self._masked_ratio(water_mask, context_mask)
        lower_ratio = self._masked_ratio(lower_water, lower_valid)
        lower_coverage = float(lower_water.any(axis=0).mean()) if lower_water.size else 0.0

        mean_b, _, mean_r = [float(v) for v in cv2.mean(roi)[:3]]
        blue_bias = self._clamp01((mean_b - mean_r + 20.0) / 90.0)

        if water_line >= 0:
            region_contact = self._clamp01((ry2 - max(ry1, water_line)) / max(1.0, float(ry2 - ry1)))
            bbox_contact = self._clamp01((y2 - water_line) / max(1.0, float(bh)))
            line_contact = max(region_contact, bbox_contact)
        else:
            line_contact = 0.0

        water_score = (
            0.34 * lower_ratio
            + 0.24 * overall_ratio
            + 0.24 * lower_coverage
            + 0.18 * line_contact
        )
        if blue_bias < 0.34:
            water_score *= 0.55
        else:
            water_score = min(1.0, water_score + 0.10 * blue_bias)

        in_water = (
            (line_contact >= 0.16 and lower_ratio >= 0.10)
            or (water_score >= 0.36 and lower_coverage >= 0.22)
        )
        return water_score, in_water

    def _detect_faces(self, frame):
        if self.face_cascade is None:
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        try:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(28, 28),
            )
        except Exception as e:
            print(f"[VISION] Face detect failed: {e}")
            return []

        detections = []
        h, w = frame.shape[:2]
        for x, y, fw, fh in faces:
            pad_x = int(fw * 0.35)
            pad_top = int(fh * 0.45)
            pad_bottom = int(fh * 0.80)

            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_top)
            x2 = min(w - 1, x + fw + pad_x)
            y2 = min(h - 1, y + fh + pad_bottom)
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                "area": max(1, (x2 - x1) * (y2 - y1)),
                "det_source": "face",
            })

        return detections

    def _collect_aux_objects(self, frame):
        if self.object_model is None:
            return []

        infer_kwargs = {
            "verbose": False,
            "conf": OBJECT_CONF_THRESHOLD,
        }
        if self.object_class_ids:
            infer_kwargs["classes"] = self.object_class_ids

        try:
            results = self.object_model(frame, **infer_kwargs)[0]
        except Exception as e:
            print(f"[VISION] Object inference failed: {e}")
            return []

        if results.boxes is None:
            return []

        boxes = results.boxes.xyxy.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()

        allowed = {str(name).lower() for name in OBJECT_DISPLAY_CLASSES if str(name).strip()}
        objects = []
        min_area = max(36.0, float(frame.shape[0] * frame.shape[1]) * float(OBJECT_MIN_AREA_RATIO))
        for box, cls, conf in zip(boxes, classes, confs):
            label = self._model_class_name(self.object_model, int(cls)).lower()
            if label == "person":
                continue
            if allowed and label not in allowed:
                continue
            if float(conf) < OBJECT_CONF_THRESHOLD:
                continue

            x1, y1, x2, y2 = map(int, box)
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            area = float(w * h)
            if area < min_area:
                continue

            objects.append({
                "label": label,
                "confidence": float(conf),
                "bbox": (x1, y1, x2, y2),
                "center": (x1 + w // 2, y1 + h // 2),
                "area": area,
            })

        objects.sort(key=lambda obj: (obj["confidence"], obj["area"]), reverse=True)
        deduped = []
        iou_thr = 0.45
        for obj in objects:
            keep = True
            for prev in deduped:
                if prev["label"] != obj["label"]:
                    continue
                if self._bbox_iou(prev["bbox"], obj["bbox"]) >= iou_thr:
                    keep = False
                    break
            if keep:
                deduped.append(obj)
        return deduped[:max(0, int(OBJECT_MAX_RESULTS))]

    def _activity_label(self, person):
        features = person.get("features", {})
        risk = float(person.get("risk", 0.0))
        raw_risk = float(person.get("raw_risk", risk))
        acute = float(person.get("acute_distress", 0.0))

        if person.get("source") == "face":
            face_signal = max(raw_risk, 0.72 * risk + 0.28 * acute)
            face_safe_score = max(0.0, min(1.0, 1.0 - face_signal))
            label = "FACE STABLE" if face_safe_score >= 0.50 else "FACE DISTRESS"
            return label, face_safe_score

        progress_ratio = max(0.0, min(1.0, float(features.get("progress_ratio", 0.0))))
        speed_norm = max(0.0, min(1.0, float(features.get("speed_norm", 0.0)) / 2.4))
        direction_change = max(0.0, min(1.0, float(features.get("direction_change", 0.0)) / 0.55))
        upper_motion = max(0.0, min(1.0, float(features.get("upper_motion", 0.0)) / 0.18))
        pose_quality = max(0.0, min(1.0, float(features.get("pose_quality", 0.0))))
        pose_verticality = max(0.0, min(1.0, float(features.get("pose_verticality", 0.0))))

        swim_score = 0.0
        swim_score += 0.34 * progress_ratio
        swim_score += 0.24 * speed_norm
        swim_score += 0.18 * (1.0 - direction_change)
        swim_score += 0.14 * (1.0 - upper_motion)
        swim_score += 0.10 * (pose_quality * (1.0 - pose_verticality))
        swim_score -= 0.38 * raw_risk
        swim_score = max(0.0, min(1.0, swim_score))

        label = "SWIMMING" if swim_score >= 0.42 and person.get("alert_state", "SAFE") == "SAFE" else "NOT SWIMMING"
        return label, swim_score

    def _danger_status(self, person):
        in_water = bool(person.get("in_water", False))
        alert_state = str(person.get("alert_state", "SAFE")).upper()
        risk = float(person.get("risk", 0.0))
        raw_risk = float(person.get("raw_risk", risk))
        acute = float(person.get("acute_distress", 0.0))
        swim_score = float(person.get("swim_score", 0.0))
        water_score = float(person.get("water_score", 0.0))

        if WATER_FILTER_ENABLED and (not in_water or water_score < 0.30):
            return "SAFE", False

        if person.get("source") == "face":
            return "SAFE", False

        is_danger = (
            alert_state in {"WATCH", "ALERT"}
            or raw_risk >= 0.52
            or acute >= 0.58
            or (risk >= 0.46 and swim_score < 0.30)
        )
        return ("DROWNING RISK", True) if is_danger else ("SAFE", False)

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

        results = self.model(frame, **infer_kwargs)[0] if self.model is not None else None
        aux_objects = self._collect_aux_objects(frame)
        use_pose_this_frame = (
            self.pose_model is not None
            and self.frame_idx % max(1, POSE_INFER_EVERY_N_FRAMES) == 0
        )

        raw_detections = []
        water_line = int(frame.shape[0] * WATER_LINE_RATIO) if WATER_FILTER_ENABLED else -1

        if results is not None and results.boxes is not None:
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
                    "det_source": "person",
                })

        if not raw_detections:
            raw_detections = self._detect_faces(frame)

        pose_by_idx = {}
        if use_pose_this_frame and raw_detections:
            person_indexed_bboxes = [
                (idx, d["bbox"])
                for idx, d in enumerate(raw_detections)
                if d.get("det_source", "person") == "person"
            ]
            if person_indexed_bboxes:
                person_bboxes = [bbox for _, bbox in person_indexed_bboxes]
                if POSE_RUN_ON_CROPS:
                    pose_local = self._infer_pose_on_crops(frame, person_bboxes)
                else:
                    pose_local = self._infer_pose_fullframe(frame, person_bboxes)
                for local_idx, pose in pose_local.items():
                    if local_idx >= len(person_indexed_bboxes):
                        continue
                    det_idx, _ = person_indexed_bboxes[local_idx]
                    pose_by_idx[det_idx] = pose

        detections = []
        for idx, det in enumerate(raw_detections):
            water_score, in_water = self._estimate_water_context(
                frame,
                det["bbox"],
                det_source=det.get("det_source", "person"),
                water_line=water_line,
            )
            detections.append({
                "bbox": det["bbox"],
                "center": det["center"],
                "area": det["area"],
                "pose": pose_by_idx.get(idx),
                "pose_observed": use_pose_this_frame,
                "det_source": det.get("det_source", "person"),
                "water_score": water_score,
                "in_water": in_water,
            })

        persons = self.identity.update(frame, detections, frame_ts=frame_ts)
        self.frame_idx += 1

        for person, det in zip(persons, detections):
            person["water_score"] = float(det.get("water_score", 0.0))
            person["in_water"] = bool(det.get("in_water", False))

        for p in persons:
            activity_label, swim_score = self._activity_label(p)
            p["activity_label"] = activity_label
            p["swim_score"] = swim_score
            status_label, is_danger = self._danger_status(p)
            p["status_label"] = status_label
            p["is_danger"] = is_danger

        target = None
        if persons:
            target = max(
                persons,
                key=lambda p: (
                    1 if p.get("is_danger", False) else 0,
                    p.get("water_score", 0.0),
                    p.get("risk", 0.0),
                ),
            )

        for p in persons:
            x1, y1, x2, y2 = p["bbox"]
            risk = p["risk"]
            status_label = p.get("status_label", "SAFE")
            is_danger = bool(p.get("is_danger", False))

            color = (0, 255, 0)
            if is_danger:
                color = (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if POSE_DRAW_OVERLAY:
                self._draw_pose(frame, p.get("pose_points", {}), color)

            cv2.putText(
                frame,
                f"ID:{p['id']} {status_label} R:{risk:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        return frame, target, persons, aux_objects
