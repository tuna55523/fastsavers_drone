# pyright: reportGeneralTypeIssues=false
# pylint: disable=no-member
import os
import sys
import time
import threading
import importlib
import argparse
from typing import Any, Callable

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
from djitellopy import Tello  # pyright: ignore[reportMissingImports]

DetectTrackSystem = None  # type: ignore[assignment]
_VISION_IMPORT_ERROR = None


def parse_runtime_args():
    parser = argparse.ArgumentParser(description="Tello paneli veya video test modu")
    parser.add_argument("--video", type=str, default="", help="Drone yerine oynatilacak video dosyasi")
    parser.add_argument("--loop-video", action="store_true", help="Video bitince basa sar")
    return parser.parse_args()

# =========================================================
# AYARLAR
# =========================================================
SAVE_DIR = os.path.join(os.path.expanduser("~"), "Pictures", "Saved Pictures")
os.makedirs(SAVE_DIR, exist_ok=True)

DISPLAY_W = 1280 
DISPLAY_H = 720
TARGET_FPS = 45
FRAME_TIME = 1.0 / TARGET_FPS

MAX_SPEED = 72  # manuel rc_control icin (batarya dostu)
FLIP_COOLDOWN_SEC = 1.5
FLIP_MIN_BATTERY = 20

# =========================================================
# OTONOM HIZ AYARLARI (rc_control tabanli)
# =========================================================
# Drone cm/s olarak hareket eder, biz sure ile mesafeyi kontrol ederiz
# move_forward(200) yerine: rc ile ~2.0 sn boyunca 60 cm/s ileri
AUTO_FWD_SPEED = 52   # cm/s ileri hiz (rc_control scale: 0-100)
AUTO_YAW_SPEED = 40   # donme hizi (rc_control scale)
AUTO_HOME_CORR_SPEED = 35
AUTO_DOCK_SPEED = 12
USE_MISSION_PAD_DOCKING = False

# Mesafe/sure hesabi:
# 100 cm / 60 cm/s = ~1.67 sn
# 200 cm / 60 cm/s = ~3.33 sn
# 300 cm / 60 cm/s = ~5.00 sn
# 90 derece / 40 yaw = ~2.25 sn (deneysel, ayarlanabilir)
DIST_100_SEC = 1.7    # 1m icin sure (sn) - gerekte ayarla
DIST_200_SEC = 3.4    # 2m icin sure (sn) - gerekte ayarla
DIST_300_SEC = 5.1    # 3m icin sure (sn) - gerekte ayarla
ROT_90_SEC   = 2.3    # 90 derece donme suresi (sn) - gerekte ayarla
AUTO_STEP_HOVER_SEC = 0.6
AUTO_FINAL_HOVER_SEC = 1.4
AUTO_HOME_TOL_CM = 35.0
AUTO_RETURN_MAX_PASSES = 3
AUTO_ALT_HOLD_TOL_CM = 12.0
AUTO_ALT_CORR_SPEED = 20
AUTO_PAD_SCAN_SEC = 2.5
AUTO_DOCK_APPROACH_Z_CM = 45
AUTO_DOCK_FINAL_Z_CM = 22
TAKEOFF_READY_HEIGHT_CM = 28
TAKEOFF_WAIT_SEC = 6.0
MANUAL_TAKEOFF_HOVER_CM = 28
MANUAL_TAKEOFF_TOL_CM = 7
MANUAL_TAKEOFF_MAX_DESCEND_CM = 70
AUTO_MIN_BATTERY = 25
BATTERY_POLL_SEC = 5.0
MANUAL_SPEED_LIMIT_CM_S = 55

# Manuel modda havada beklerken irtifa tutma
HOVER_HOLD_ENABLED = True
HOVER_HOLD_IDLE_DELAY_SEC = 0.28
HOVER_HOLD_SAMPLE_SEC = 0.20
HOVER_HOLD_DEADBAND_CM = 7.0
HOVER_HOLD_GAIN = 1.7
HOVER_HOLD_MIN_UD = 10
HOVER_HOLD_MAX_UD = 22

# Takla sonrasi toparlama
FLIP_RECOVER_SEC = 1.5
FLIP_RECOVER_DEADBAND_CM = 5.0
FLIP_RECOVER_GAIN = 2.1
FLIP_RECOVER_MAX_UD = 26

# Pil tasarrufu (dinamik)
POWER_SAVE_ENABLED = True
POWER_SAVE_BAT_MID = 35
POWER_SAVE_BAT_LOW = 25
POWER_SAVE_BAT_CRIT = 17
POWER_SAVE_SCALE_MID = 0.92
POWER_SAVE_SCALE_LOW = 0.84
POWER_SAVE_SCALE_CRIT = 0.74
BATTERY_OPT_ENABLE = True
ECO_BATTERY_LOW = 35
ECO_BATTERY_CRIT = 25
ECO_SPEED_SCALE_LOW = 0.78
ECO_SPEED_SCALE_CRIT = 0.62

# =========================================================
# GORUNTU ISLEME
# =========================================================
VISION_AUTO_ENABLE = True
VISION_UPDATE_INTERVAL_SEC = 0.12
VISION_UPDATE_INTERVAL_LOW = 0.20
VISION_UPDATE_INTERVAL_CRIT = 0.28
VISION_RESULT_TTL_SEC = 1.2
VISION_TRACK_LOST_SEC = 5.0
VISION_MIN_TRACK_BOX_W = 28
VISION_MIN_TRACK_BOX_H = 38
VISION_REACQ_MAX_CENTER_DIST_PX = 190.0
VISION_TARGET_SWITCH_COOLDOWN_SEC = 0.8
VISION_TRACK_ERROR_COOLDOWN_SEC = 1.2
TRACK_ONLY_SAFE_PERSON = True
DANGER_HOLD_ON_TRACK = True
DANGER_AUTO_SNAPSHOT = True
DANGER_SNAPSHOT_COOLDOWN_SEC = 2.4
DANGER_TOAST_COOLDOWN_SEC = 1.5
VISION_STICKY_LOCK = True
VISION_LOCK_PERSIST_ON_OCCLUSION = True
VISION_REACQ_KEEP_SEC = 14.0

AUTO_ROUTE_STEPS = [
    ("forward",   "1m ileri", AUTO_FWD_SPEED, DIST_100_SEC),
    ("yaw_right", "Saga don", AUTO_YAW_SPEED, ROT_90_SEC),
    ("forward",   "1m ileri", AUTO_FWD_SPEED, DIST_100_SEC),
    ("yaw_right", "Saga don", AUTO_YAW_SPEED, ROT_90_SEC),
    ("forward",   "1m ileri", AUTO_FWD_SPEED, DIST_100_SEC),
    ("yaw_right", "Saga don", AUTO_YAW_SPEED, ROT_90_SEC),
    ("forward",   "1m ileri", AUTO_FWD_SPEED, DIST_100_SEC),
]

# =========================================================
# TAKİP PARAMETRELERİ
# =========================================================
TRACK_SPEED   = 100
DEAD_ZONE     = 10
TRACK_GAIN_YAW = 0.26
TRACK_GAIN_UD  = 0.22
MIN_CMD_YAW   = 10
MIN_CMD_UD    = 10
MAX_YAW       = 55
MAX_UD        = 55
COMBO_FLIP_STEP_DELAY_SEC = 0.35
COMBO_FLIP_COOLDOWN_SEC = 4.0
SHOW_SEGMENT_TICK_SEC = 0.05
SHOW_SQUARE_FB_SPEED = 46
SHOW_SQUARE_YAW_SPEED = 42
SHOW_SQUARE_EDGE_SEC = 1.15
SHOW_SQUARE_TURN_SEC = 1.05
SHOW_EIGHT_FB_SPEED = 42
SHOW_EIGHT_YAW_SPEED = 40
SHOW_EIGHT_LOOP_SEC = 1.85
MEVLANA_YAW_SPEED = 58

# =========================================================
# İLERİ/GERİ (FB) - MESAFE KORUMA
# =========================================================
FB_SLEW     = 12
fb_prev_cmd = 0
FB_BAND     = 0.10
FB_MIN_W    = 0.72
FB_MAX_W    = 1.32
FB_MAX_FWD  = 32
FB_MAX_BWD  = 32
FB_MIN_STEP = 8
fb_active      = False
fb_enable_time = 0.0
dist_target_w  = None

# =========================================================
# RE-ACQUIRE
# =========================================================
TM_THRESH                = 0.56
REACQUIRE_SEARCH_SCALE   = 3.2
TEMPLATE_UPDATE_INTERVAL = 25
LOST_GRACE_SEC           = 10.0
SEARCH_YAW               = 18
SEARCH_TOGGLE_EVERY      = 12

# =========================================================
# HARD LOCK
# =========================================================
lock_enabled     = False
LOCK_FUSE_STRICT = 0.60
LOCK_HIST_STRICT = 0.50
LOCK_EDGE_STRICT = 0.32
REACQ_CONFIRM_N  = 2
CENTER_MAX_JUMP  = 0.65
ASPECT_TOL       = 0.70
SIZE_TOL         = 0.75

# =========================================================
# STREAM WATCHDOG
# =========================================================
WATCHDOG_STALE_SEC      = 0.50
WATCHDOG_RESET_COOLDOWN = 2.0

# =========================================================
# BAĞLANTI FAILSAFE
# =========================================================
LINK_FAIL_STREAK_TRIG = 10
LINK_LOST_SEC_TRIG    = 1.20
LAND_RETRY_COOLDOWN   = 1.0

# =========================================================
# KLAVYE
# =========================================================
FAST_KEYS  = True
pressed    = set()
key_events = []
_listener  = None
CANCEL_KEYS = {"o", "\u00f6"}

try:
    from pynput import keyboard  # pyright: ignore[reportMissingModuleSource]
except Exception:
    print("pynput yok. Kur: pip install pynput")
    raise


def normalize_key_token(token):
    if token is None:
        return None
    text = str(token).strip().lower()
    if not text:
        return None

    aliases = {
        "return": "enter",
        "esc": "escape",
        "spacebar": "space",
        "\u00f6": "\u00f6",
        "\u00d6".lower(): "\u00f6",
        "\u00c3\u00b6": "\u00f6",
        "\u00c3\u0192\u00c2\u00b6": "\u00f6",
        "\u00c3\u00a3\u00c2\u00b6": "\u00f6",
    }
    return aliases.get(text, text)


def _key_to_str(k):
    try:
        if hasattr(k, "char") and k.char:
            return normalize_key_token(k.char)
    except Exception:
        pass
    if k == keyboard.Key.space:  return "space"
    if k == keyboard.Key.enter:  return "enter"
    if k == keyboard.Key.tab:    return "tab"
    return None


def _on_press(k):
    s = _key_to_str(k)
    if not s: return
    pressed.add(s)
    if FAST_KEYS:
        key_events.append(s)


def _on_release(k):
    s = _key_to_str(k)
    if not s: return
    pressed.discard(s)


def start_keyboard_listener():
    global _listener
    if _listener is None:
        _listener = keyboard.Listener(on_press=_on_press, on_release=_on_release)
        _listener.daemon = True
        _listener.start()


def pop_key_presses():
    global key_events
    ev = key_events[:]
    key_events = []
    return ev


def dedupe_key_events(events):
    seen = set()
    unique_events = []
    for event in events:
        if event in seen:
            continue
        seen.add(event)
        unique_events.append(event)
    return unique_events


def key_down(k: str) -> bool:
    if auto_running or emergency or takeoff_busy:
        return False
    return k in pressed


# =========================================================
# UI HELPERS
# =========================================================
def clamp(v, mn, mx):
    return max(mn, min(mx, v))


def fit_text_width(text, max_width, scale=0.55, thickness=2):
    text = "" if text is None else str(text)
    max_width = int(max_width)
    if max_width <= 0:
        return ""

    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    if text_width <= max_width:
        return text

    ellipsis = "..."
    ellipsis_width = cv2.getTextSize(ellipsis, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    if ellipsis_width > max_width:
        return ""

    lo = 0
    hi = len(text)
    best = ellipsis
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip()
        candidate = f"{candidate}{ellipsis}" if candidate else ellipsis
        candidate_width = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
        if candidate_width <= max_width:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def format_battery_text(value):
    try:
        battery = int(value)
    except Exception:
        return "N/A"
    return f"{battery}%" if battery >= 0 else "N/A"


def get_battery_eco_profile(battery_value):
    if not BATTERY_OPT_ENABLE:
        return ("OFF", 1.0, VISION_UPDATE_INTERVAL_SEC)
    try:
        battery_int = int(battery_value)
    except Exception:
        return ("N/A", 1.0, VISION_UPDATE_INTERVAL_SEC)
    if battery_int < 0:
        return ("N/A", 1.0, VISION_UPDATE_INTERVAL_SEC)
    if battery_int <= ECO_BATTERY_CRIT:
        return ("CRIT", ECO_SPEED_SCALE_CRIT, VISION_UPDATE_INTERVAL_CRIT)
    if battery_int <= ECO_BATTERY_LOW:
        return ("LOW", ECO_SPEED_SCALE_LOW, VISION_UPDATE_INTERVAL_LOW)
    return ("NORMAL", 1.0, VISION_UPDATE_INTERVAL_SEC)


def get_runtime_speed_scale():
    _, scale, _ = get_battery_eco_profile(battery_level)
    return float(scale)


def get_runtime_vision_interval():
    _, _, interval = get_battery_eco_profile(battery_level)
    return float(interval)


def clamp_bbox_to_frame(bbox, fw, fh):
    x, y, w, h = map(int, bbox)
    x = clamp(x, 0, fw - 1)
    y = clamp(y, 0, fh - 1)
    w = clamp(w, 20, fw - x)
    h = clamp(h, 20, fh - y)
    return (x, y, w, h)


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def normalize_xyxy_bbox(bbox, fw, fh, min_w=10, min_h=10):
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in bbox[:4]]
    except Exception:
        return None

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 = clamp(x1, 0, max(0, fw - 1))
    y1 = clamp(y1, 0, max(0, fh - 1))
    x2 = clamp(x2, x1 + 1, max(x1 + 1, fw - 1))
    y2 = clamp(y2, y1 + 1, max(y1 + 1, fh - 1))

    if (x2 - x1) < int(min_w) or (y2 - y1) < int(min_h):
        return None
    return (x1, y1, x2, y2)


def bbox_xyxy_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def bbox_xyxy_iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = float((ix2 - ix1) * (iy2 - iy1))
    area_a = float(max(1, (ax2 - ax1) * (ay2 - ay1)))
    area_b = float(max(1, (bx2 - bx1) * (by2 - by1)))
    union = max(1.0, area_a + area_b - inter)
    return inter / union


def sanitize_person_detections(persons, fw, fh):
    clean = []
    if not isinstance(persons, (list, tuple)):
        return clean

    anon_idx = 0
    for person in persons:
        if not isinstance(person, dict):
            continue
        bbox = normalize_xyxy_bbox(
            person.get("bbox"),
            fw,
            fh,
            min_w=VISION_MIN_TRACK_BOX_W,
            min_h=VISION_MIN_TRACK_BOX_H,
        )
        if bbox is None:
            continue

        normalized = dict(person)
        person_id = normalized.get("id")
        if person_id is None:
            person_id = f"anon_{anon_idx}"
            anon_idx += 1
        normalized["id"] = person_id
        normalized["bbox"] = bbox
        normalized["risk"] = clamp(safe_float(normalized.get("risk", 0.0)), 0.0, 1.0)
        normalized["confidence"] = clamp(
            safe_float(normalized.get("confidence", normalized.get("conf", 0.0))),
            0.0,
            1.0,
        )
        clean.append(normalized)
    return clean


def sanitize_object_detections(objects, fw, fh):
    clean = []
    if not isinstance(objects, (list, tuple)):
        return clean
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        bbox = normalize_xyxy_bbox(obj.get("bbox"), fw, fh, min_w=12, min_h=12)
        if bbox is None:
            continue
        normalized = dict(obj)
        normalized["bbox"] = bbox
        normalized["confidence"] = clamp(
            safe_float(normalized.get("confidence", normalized.get("conf", 0.0))),
            0.0,
            1.0,
        )
        clean.append(normalized)
    return clean


def choose_tracked_person(persons, target_id, last_bbox):
    if not persons:
        return None, target_id

    for person in persons:
        if person.get("id") == target_id:
            return person, target_id

    best_person = None
    best_score = -1e9
    for person in persons:
        bbox = person.get("bbox")
        if bbox is None:
            continue
        score = 0.0
        score += float(person.get("risk", 0.0)) * 0.60
        score += float(person.get("confidence", 0.0)) * 0.28
        is_danger = bool(person.get("is_danger", False))
        if is_danger:
            score += 0.10
        if TRACK_ONLY_SAFE_PERSON:
            score += -0.45 if is_danger else 0.22
        if last_bbox is not None:
            iou = bbox_xyxy_iou(bbox, last_bbox)
            score += iou * 1.05
            cx, cy = bbox_xyxy_center(bbox)
            lx, ly = bbox_xyxy_center(last_bbox)
            center_dist = float(((cx - lx) ** 2 + (cy - ly) ** 2) ** 0.5)
            dist_norm = clamp(center_dist / VISION_REACQ_MAX_CENTER_DIST_PX, 0.0, 1.0)
            score += (1.0 - dist_norm) * 0.55
            if center_dist > VISION_REACQ_MAX_CENTER_DIST_PX:
                score -= 0.55
        if score > best_score:
            best_score = score
            best_person = person

    if best_person is None:
        return None, target_id
    return best_person, best_person.get("id")


def find_first_danger_person(persons, ignore_id=None):
    if not persons:
        return None
    for person in persons:
        if bool(person.get("is_danger", False)):
            if ignore_id is None or person.get("id") != ignore_id:
                return person
    return None


def blend_rect(img, x1, y1, x2, y2, color, alpha=0.35, border=None, thickness=1):
    x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
    x1 = clamp(x1, 0, img.shape[1] - 1)
    y1 = clamp(y1, 0, img.shape[0] - 1)
    x2 = clamp(x2, x1 + 1, img.shape[1] - 1)
    y2 = clamp(y2, y1 + 1, img.shape[0] - 1)
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), border, thickness)


def draw_cinematic_overlay(frame_bgr):
    h, w = frame_bgr.shape[:2]
    top_h = max(44, h // 14)
    bottom_h = max(56, h // 10)
    blend_rect(frame_bgr, 0, 0, w - 1, top_h, (10, 14, 18), alpha=0.24)
    blend_rect(frame_bgr, 0, h - bottom_h, w - 1, h - 1, (8, 10, 14), alpha=0.28)
    cv2.line(frame_bgr, (0, top_h), (w, top_h), (58, 86, 102), 1)
    cv2.line(frame_bgr, (0, h - bottom_h), (w, h - bottom_h), (58, 86, 102), 1)


def make_right_panel_overlay_modern(frame_bgr, lines, alpha=0.18, pad=16, panel_w=300):
    h, w = frame_bgr.shape[:2]
    x0 = max(0, w - panel_w)
    panel = frame_bgr[:, x0:w].copy()
    overlay = panel.copy()
    overlay[:] = (14, 17, 22)
    cv2.addWeighted(overlay, 0.78, panel, 0.22, 0, panel)
    cv2.line(panel, (0, 0), (0, h - 1), (0, 214, 255), 3)
    cv2.rectangle(panel, (0, 0), (panel_w - 2, h - 2), (52, 72, 82), 1)
    cv2.putText(panel, "TELLO OPS", (pad, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.74, (240, 248, 252), 2, cv2.LINE_AA)
    cv2.putText(panel, "manual / track / vision", (pad, 54),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (146, 172, 184), 1, cv2.LINE_AA)
    cv2.line(panel, (pad, 66), (panel_w - pad, 66), (58, 86, 102), 1)
    y = 88
    text_width = max(40, panel_w - pad * 2)
    for t in lines:
        if not t:
            y += 10
            continue
        color = (235, 242, 245)
        scale = 0.50
        if t.isupper() or t.endswith(":"):
            color = (0, 214, 255)
            scale = 0.46
        text = fit_text_width(t, text_width, scale=scale, thickness=1)
        cv2.putText(panel, text, (pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += 21
    frame_bgr[:, x0:w] = panel


def draw_badge(img, text, x, y, bg=(0,0,0), fg=(255,255,255), scale=0.55, pad=8):
    margin = 8
    x = clamp(int(x), margin, max(margin, img.shape[1] - margin))
    text = fit_text_width(text, img.shape[1] - x - pad * 2 - margin, scale=scale, thickness=2)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    blend_rect(img, x, y-th-pad, x+tw+pad*2, y+pad, bg, alpha=0.82, border=(255,255,255), thickness=1)
    cv2.putText(img, text, (x+pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 2, cv2.LINE_AA)


def draw_center_badge(img, text, center_x, y, bg=(0,0,0), fg=(255,255,255), scale=0.55, pad=8):
    text = fit_text_width(text, img.shape[1] - 24, scale=scale, thickness=2)
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = int(center_x - (tw + pad * 2) / 2)
    draw_badge(img, text, x, y, bg=bg, fg=fg, scale=scale, pad=pad)


def summarize_aux_objects(objects, max_items=3):
    if not objects:
        return "NONE"

    counts = {}
    for obj in objects:
        label = str(obj.get("label", "object")).upper()
        counts[label] = counts.get(label, 0) + 1

    parts = []
    for label, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:max_items]:
        parts.append(f"{label}x{count}" if count > 1 else label)
    return ", ".join(parts)


def draw_reticle_modern(img, cx, cy, color=(230, 236, 242), accent=(0, 214, 255)):
    cv2.circle(img, (cx, cy), 26, color, 1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), 6, accent, 1, cv2.LINE_AA)
    cv2.line(img, (cx - 34, cy), (cx - 12, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx + 12, cy), (cx + 34, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 34), (cx, cy - 12), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy + 12), (cx, cy + 34), color, 1, cv2.LINE_AA)


def compute_route_preview_points(route_steps):
    x = 0.0
    y = 0.0
    yaw = 0.0
    pts = [(x, y)]
    for action, _, speed, duration_sec in route_steps:
        distance_cm = float(abs(speed) * duration_sec)
        yaw_rad = np.deg2rad(yaw)
        fwd_x = np.cos(yaw_rad)
        fwd_y = np.sin(yaw_rad)
        right_x = -np.sin(yaw_rad)
        right_y = np.cos(yaw_rad)
        if action == "forward":
            x += fwd_x * distance_cm
            y += fwd_y * distance_cm
        elif action == "backward":
            x -= fwd_x * distance_cm
            y -= fwd_y * distance_cm
        elif action == "right":
            x += right_x * distance_cm
            y += right_y * distance_cm
        elif action == "left":
            x -= right_x * distance_cm
            y -= right_y * distance_cm
        elif action == "yaw_right":
            yaw = (yaw + 90.0) % 360.0
        elif action == "yaw_left":
            yaw = (yaw - 90.0) % 360.0
        pts.append((x, y))
    return pts


def draw_route_minimap(frame_bgr, route_steps, pose_state, x, y, size=190):
    panel_h = size + 34
    blend_rect(frame_bgr, x, y, x + size, y + panel_h, (10, 14, 18), alpha=0.62, border=(58, 86, 102))
    cv2.putText(frame_bgr, "AUTO MAP", (x + 14, y + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 214, 255), 1, cv2.LINE_AA)

    pts = compute_route_preview_points(route_steps)
    current_x = float(pose_state.get("x", 0.0))
    current_y = float(pose_state.get("y", 0.0))
    all_pts = pts + [(current_x, current_y)]
    max_extent = max(max(abs(px), abs(py)) for px, py in all_pts) if all_pts else 1.0
    max_extent = max(80.0, max_extent)
    pad = 22
    map_size = size - pad * 2
    center_x = x + size // 2
    center_y = y + 28 + map_size // 2 + 6
    scale = map_size / (max_extent * 2.2)

    cv2.rectangle(frame_bgr, (x + pad, y + 36), (x + size - pad, y + panel_h - 16), (38, 52, 60), 1)
    cv2.line(frame_bgr, (x + pad, center_y), (x + size - pad, center_y), (34, 42, 50), 1)
    cv2.line(frame_bgr, (center_x, y + 36), (center_x, y + panel_h - 16), (34, 42, 50), 1)

    map_pts = []
    for px, py in pts:
        sx = int(center_x + px * scale)
        sy = int(center_y - py * scale)
        map_pts.append((sx, sy))

    if len(map_pts) >= 2:
        cv2.polylines(frame_bgr, [np.array(map_pts, dtype=np.int32)], False, (0, 184, 255), 2, cv2.LINE_AA)

    home_pt = map_pts[0]
    cv2.circle(frame_bgr, home_pt, 5, (0, 196, 255), -1, cv2.LINE_AA)

    if pose_state.get("active", False):
        cur_pt = (int(center_x + current_x * scale), int(center_y - current_y * scale))
        cv2.circle(frame_bgr, cur_pt, 6, (105, 255, 164), -1, cv2.LINE_AA)
        yaw_rad = np.deg2rad(float(pose_state.get("yaw", 0.0)))
        tip = (int(cur_pt[0] + np.cos(yaw_rad) * 16), int(cur_pt[1] - np.sin(yaw_rad) * 16))
        cv2.line(frame_bgr, cur_pt, tip, (105, 255, 164), 2, cv2.LINE_AA)


def watchdog_frame_signature(rgb_frame):
    try:
        small = cv2.resize(rgb_frame, (64, 36))
        return int(small.sum())
    except Exception:
        return None


# =========================================================
# TRACKER HELPERS
# =========================================================
def _resolve_tracker_factory() -> Callable[[], Any] | None:
    tracker_paths = (
        ("TrackerCSRT_create",),
        ("legacy", "TrackerCSRT_create"),
        ("TrackerKCF_create",),
        ("legacy", "TrackerKCF_create"),
    )
    for path in tracker_paths:
        candidate: Any = cv2
        for attr in path:
            candidate = getattr(candidate, attr, None)
            if candidate is None:
                break
        if callable(candidate):
            return candidate
    return None


def create_tracker():
    tracker_factory = _resolve_tracker_factory()
    if tracker_factory is not None:
        return tracker_factory()
    raise RuntimeError("Tracker yok. opencv-contrib-python kur.")


def get_template(frame_bgr, bbox):
    x, y, w, h = map(int, bbox)
    patch = frame_bgr[y:y+h, x:x+w]
    if patch.size == 0: return None
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(g, (3, 3), 0)


def reacquire_by_template(frame_bgr, template_gray, last_bbox):
    if template_gray is None or last_bbox is None: return None, 0.0
    hh, ww = frame_bgr.shape[:2]
    lx, ly, lw, lh = map(int, last_bbox)
    cx, cy = lx + lw//2, ly + lh//2
    sw, sh = int(lw*REACQUIRE_SEARCH_SCALE), int(lh*REACQUIRE_SEARCH_SCALE)
    sx1 = clamp(cx-sw//2, 0, ww-1); sy1 = clamp(cy-sh//2, 0, hh-1)
    sx2 = clamp(cx+sw//2, 0, ww-1); sy2 = clamp(cy+sh//2, 0, hh-1)
    roi = frame_bgr[sy1:sy2, sx1:sx2]
    if roi.size == 0: return None, 0.0
    t = template_gray
    roi_g = cv2.GaussianBlur(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (3,3), 0)
    if roi_g.shape[0] < t.shape[0] or roi_g.shape[1] < t.shape[1]: return None, 0.0
    try:
        res = cv2.matchTemplate(roi_g, t, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
    except Exception: return None, 0.0
    if max_val < TM_THRESH: return None, float(max_val)
    tx, ty = max_loc
    th, tw = t.shape[:2]
    return clamp_bbox_to_frame((sx1+tx, sy1+ty, tw, th), ww, hh), float(max_val)


def get_hsv_hist(frame_bgr, bbox):
    x, y, w, h = map(int, bbox)
    patch = frame_bgr[y:y+h, x:x+w]
    if patch.size == 0: return None
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0,1], None, [30,32], [0,180,0,256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def compare_hist_corr(h1, h2):
    if h1 is None or h2 is None: return -1.0
    try: return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
    except: return -1.0


def get_edge_signature(frame_bgr, bbox):
    x, y, w, h = map(int, bbox)
    patch = frame_bgr[y:y+h, x:x+w]
    if patch.size == 0: return None
    g = cv2.GaussianBlur(cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (3,3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    edge_mask = mag > (np.mean(mag) + np.std(mag)*0.5)
    if int(edge_mask.sum()) < 40: return None
    hist, _ = np.histogram(ang[edge_mask], bins=16, range=(0,360))
    hist = hist.astype(np.float32)
    s = float(hist.sum())
    if s <= 0: return None
    hist /= (s + 1e-6)
    return hist


def compare_edge_sig(e1, e2):
    if e1 is None or e2 is None: return -1.0
    a = e1 / (np.linalg.norm(e1)+1e-6)
    b = e2 / (np.linalg.norm(e2)+1e-6)
    return float(np.dot(a, b))


def bbox_center(b):
    x, y, w, h = b
    return (x+w*0.5, y+h*0.5)


def continuity_ok(candidate_bbox, ref_bbox, ww, hh):
    if candidate_bbox is None or ref_bbox is None: return True
    cx, cy = bbox_center(candidate_bbox)
    rx, ry = bbox_center(ref_bbox)
    if (abs(cx-rx)/max(1.,ww) + abs(cy-ry)/max(1.,hh)) > CENTER_MAX_JUMP: return False
    _, _, cw, ch = candidate_bbox
    _, _, rw, rh = ref_bbox
    if rw<=0 or rh<=0 or cw<=0 or ch<=0: return False
    if abs(cw/ch - rw/rh) / max(1e-6, rw/rh) > ASPECT_TOL: return False
    if abs(cw-rw)/max(1.,rw) > SIZE_TOL: return False
    return True


def fuse_score(hist_corr, edge_corr, tm_score):
    hc = (float(hist_corr)+1.)*0.5
    ec = (float(edge_corr)+1.)*0.5
    ts = float(np.clip(tm_score, 0., 1.))
    return 0.45*hc + 0.35*ec + 0.20*ts


# =========================================================
# TELLO BAĞLANTISI
# =========================================================
tello = None
tello_connected = False
video_mode = False
video_path = ""
video_loop = False
video_cap = None
video_frame_is_rgb = False
sim_is_flying = False
sim_battery = 100


def toast(msg, sec=1.5):
    global toast_text, toast_until
    toast_text  = msg
    toast_until = time.time() + sec
    print(msg)


def save_photo(frame_bgr):
    if frame_bgr is None or getattr(frame_bgr, "size", 0) <= 0:
        toast("FOTO HATA: frame yok", 1.8)
        return False

    stamp = time.strftime("%Y%m%d_%H%M%S")
    millis = int((time.time() % 1.0) * 1000)
    filename = f"tello_{stamp}_{millis:03d}.jpg"
    path = os.path.join(SAVE_DIR, filename)

    try:
        ok = cv2.imwrite(path, frame_bgr)
        if not ok:
            raise RuntimeError("dosya yazilamadi")
        toast(f"FOTO: {filename}", 2.0)
        print(f"[PHOTO] Saved: {path}")
        return True
    except Exception as e:
        toast(f"FOTO HATA: {e}", 2.2)
        return False


def safe_is_flying():
    if video_mode:
        return bool(sim_is_flying)
    tello_ref = tello
    if tello_ref is None:
        return False
    try: return bool(tello_ref.is_flying)
    except: return False


def safe_get_height_cm(default=0.0):
    if video_mode:
        return float(MANUAL_TAKEOFF_HOVER_CM if sim_is_flying else 0.0)
    tello_ref = tello
    if tello_ref is None:
        return float(default)
    try: return float(run_sdk_command(tello_ref.get_height))
    except: return float(default)


def safe_get_battery(default=-1):
    if video_mode:
        return int(sim_battery)
    tello_ref = tello
    if tello_ref is None:
        return int(default)
    try: return int(run_sdk_command(tello_ref.get_battery))
    except: return int(default)


def power_save_motion_scale(bat_value):
    if not POWER_SAVE_ENABLED:
        return 1.0
    try:
        bat = int(bat_value)
    except Exception:
        return 1.0
    if bat < 0:
        return 1.0
    if bat <= POWER_SAVE_BAT_CRIT:
        return POWER_SAVE_SCALE_CRIT
    if bat <= POWER_SAVE_BAT_LOW:
        return POWER_SAVE_SCALE_LOW
    if bat <= POWER_SAVE_BAT_MID:
        return POWER_SAVE_SCALE_MID
    return 1.0


def scale_axis_cmd(cmd, scale, min_abs=0):
    if cmd == 0 or scale >= 0.999:
        return int(cmd)
    scaled = int(round(float(cmd) * float(scale)))
    if scaled == 0:
        scaled = 1 if cmd > 0 else -1
    if min_abs > 0 and abs(scaled) < min_abs:
        scaled = int(np.sign(scaled) * min_abs)
    return int(clamp(scaled, -100, 100))


def manual_axis_speed_from_battery(bat_value):
    scale = power_save_motion_scale(bat_value)
    return max(45, int(round(MAX_SPEED * scale)))


def stop_and_hover():
    send_rc_control_safe(0, 0, 0, 0, reason="hover_stop", report_fail=False, report_success=False)


sdk_command_lock = threading.RLock()


def run_sdk_command(fn):
    with sdk_command_lock:
        return fn()


def connect_tello():
    global tello, tello_connected
    if video_mode:
        tello_connected = True
        return True
    if tello is None:
        tello = Tello()
    if tello_connected:
        return True
    run_sdk_command(tello.connect)
    tello_connected = True
    return True


def send_rc_control_safe(lr=0, fb=0, ud=0, yv=0, *, reason="rc", report_fail=True, report_success=False):
    if video_mode:
        if report_success:
            note_link_ok()
        return True
    tello_ref = tello
    if tello_ref is None:
        if report_fail:
            note_link_fail(f"{reason}:no_tello")
        return False
    try:
        with sdk_command_lock:
            tello_ref.send_rc_control(int(lr), int(fb), int(ud), int(yv))
        if report_success:
            note_link_ok()
        return True
    except Exception:
        if report_fail:
            note_link_fail(reason)
        return False


def fast_takeoff(wait_sec=TAKEOFF_WAIT_SEC, min_height_cm=TAKEOFF_READY_HEIGHT_CM):
    global sim_is_flying
    if video_mode:
        sim_is_flying = True
        return True
    stop_and_hover()
    try:
        run_sdk_command(lambda: tello.send_command_without_return("takeoff"))
    except Exception:
        run_sdk_command(tello.takeoff)
        return True

    t0 = time.time()
    while time.time() - t0 < wait_sec:
        if emergency:
            return False
        h = safe_get_height_cm(0.0)
        if h >= min_height_cm or safe_is_flying():
            try:
                tello.is_flying = True
            except Exception:
                pass
            return True
        time.sleep(0.08)
    return safe_is_flying() or safe_get_height_cm(0.0) >= max(18.0, min_height_cm * 0.65)


def settle_manual_takeoff_height(
    target_height_cm=MANUAL_TAKEOFF_HOVER_CM,
    tolerance_cm=MANUAL_TAKEOFF_TOL_CM,
):
    if not safe_is_flying():
        return False

    time.sleep(0.35)
    current_h = safe_get_height_cm(0.0)
    if current_h <= 0:
        return False

    delta_cm = float(current_h - target_height_cm)
    if delta_cm <= tolerance_cm:
        return True

    if delta_cm >= 22.0:
        descend_cm = int(min(MANUAL_TAKEOFF_MAX_DESCEND_CM, max(20, round(delta_cm - tolerance_cm * 0.5))))
        try:
            run_sdk_command(lambda: tello.move_down(descend_cm))
            time.sleep(0.2)
            stop_and_hover()
            current_h = safe_get_height_cm(target_height_cm)
            delta_cm = float(current_h - target_height_cm)
            if delta_cm <= tolerance_cm:
                return True
        except Exception:
            pass

    descend_speed = -18 if delta_cm < 18 else -22
    descend_sec = min(1.0, max(0.18, max(0.0, delta_cm) / 55.0))
    t0 = time.time()
    while time.time() - t0 < descend_sec:
        if emergency or not safe_is_flying():
            break
        if not send_rc_control_safe(0, 0, descend_speed, 0, reason="manual_descend"):
            break
        time.sleep(0.06)
    stop_and_hover()
    current_h = safe_get_height_cm(target_height_cm)
    return current_h <= (target_height_cm + tolerance_cm + 4.0)


def reset_hover_hold_target(preferred_height_cm=MANUAL_TAKEOFF_HOVER_CM):
    global hover_target_height_cm, hover_idle_since_t, hover_last_sample_t
    if safe_is_flying():
        measured = safe_get_height_cm(float(preferred_height_cm))
        if measured > 0:
            hover_target_height_cm = float(measured)
        else:
            hover_target_height_cm = float(preferred_height_cm)
    else:
        hover_target_height_cm = None
    hover_idle_since_t = time.time()
    hover_last_sample_t = 0.0


def compute_manual_hover_ud(now, manual_input_active):
    global hover_target_height_cm, hover_idle_since_t, hover_last_sample_t

    if (
        not HOVER_HOLD_ENABLED
        or emergency
        or auto_running
        or takeoff_busy
        or mode != 0
        or (not safe_is_flying())
    ):
        hover_target_height_cm = None
        hover_idle_since_t = now
        return 0

    sample_due = (now - hover_last_sample_t) >= HOVER_HOLD_SAMPLE_SEC
    if manual_input_active:
        hover_idle_since_t = now
        if sample_due:
            hover_last_sample_t = now
            current_h = safe_get_height_cm(0.0)
            if current_h > 0:
                if hover_target_height_cm is None:
                    hover_target_height_cm = float(current_h)
                else:
                    hover_target_height_cm = (hover_target_height_cm * 0.86) + (float(current_h) * 0.14)
        return 0

    if now - hover_idle_since_t < HOVER_HOLD_IDLE_DELAY_SEC:
        return 0
    if not sample_due:
        return 0

    hover_last_sample_t = now
    default_h = MANUAL_TAKEOFF_HOVER_CM if hover_target_height_cm is None else hover_target_height_cm
    current_h = safe_get_height_cm(default_h)
    if current_h <= 0:
        return 0

    if hover_target_height_cm is None:
        hover_target_height_cm = float(current_h)
        return 0

    delta_h = float(hover_target_height_cm - current_h)
    if abs(delta_h) <= HOVER_HOLD_DEADBAND_CM:
        return 0

    ud_cmd = int(clamp(delta_h * HOVER_HOLD_GAIN, -HOVER_HOLD_MAX_UD, HOVER_HOLD_MAX_UD))
    if 0 < abs(ud_cmd) < HOVER_HOLD_MIN_UD:
        ud_cmd = int(np.sign(ud_cmd) * HOVER_HOLD_MIN_UD)
    return int(ud_cmd)


def stabilize_after_flip(reference_height_cm):
    if not safe_is_flying():
        return

    try:
        ref_height = float(reference_height_cm)
    except Exception:
        ref_height = 0.0
    if ref_height <= 0:
        ref_height = safe_get_height_cm(MANUAL_TAKEOFF_HOVER_CM)

    deadline = time.time() + FLIP_RECOVER_SEC
    while time.time() < deadline:
        if emergency or auto_running or not safe_is_flying():
            break
        current_h = safe_get_height_cm(ref_height)
        delta_h = ref_height - current_h
        ud_cmd = 0
        if abs(delta_h) > FLIP_RECOVER_DEADBAND_CM:
            ud_cmd = int(clamp(delta_h * FLIP_RECOVER_GAIN, -FLIP_RECOVER_MAX_UD, FLIP_RECOVER_MAX_UD))
            if 0 < abs(ud_cmd) < HOVER_HOLD_MIN_UD:
                ud_cmd = int(np.sign(ud_cmd) * HOVER_HOLD_MIN_UD)
        send_rc_control_safe(0, 0, ud_cmd, 0, reason="flip_recover", report_fail=False, report_success=False)
        time.sleep(0.07)
    stop_and_hover()
    reset_hover_hold_target(ref_height)


def force_sdk_speed_max():
    if video_mode:
        return True
    if (
        globals().get("auto_running", False)
        or globals().get("takeoff_busy", False)
        or globals().get("emergency", False)
        or globals().get("mode", 0) != 0
    ):
        return False
    for fn in [
        lambda: tello.set_speed(MANUAL_SPEED_LIMIT_CM_S),
        lambda: tello.send_command_without_return(f"speed {MANUAL_SPEED_LIMIT_CM_S}"),
        lambda: tello.send_command_with_return(f"speed {MANUAL_SPEED_LIMIT_CM_S}", timeout=2),
    ]:
        try: run_sdk_command(fn); return True
        except: pass
    return False

battery_level = -1
last_battery_poll_t = 0.0

# =========================================================
# STATE
# =========================================================
mode        = 0
tracker     = None
tracker_on  = False

template_gray           = None
last_good_bbox          = None
tm_conf                 = 0.0
successful_track_frames = 0

lock_hist   = None
lock_edge   = None
hist_conf   = -1.0
edge_conf   = -1.0
fused_conf  = 0.0
reacq_confirm = 0
lost_since  = None

# --- VISION TRACKING STATE ---
vision_track_active    = False
vision_track_target_id = None
vision_track_lost_t    = None
vision_track_bbox      = None
vision_track_last_switch_t = 0.0
vision_track_last_error_t = 0.0
vision_track_wait_toast_t = 0.0
danger_snapshot_last_t = 0.0
danger_toast_last_t = 0.0

frame_read            = None
last_frame_sig        = None
last_frame_change_t   = time.time()
last_watchdog_reset_t = 0.0
link_fail_streak      = 0
last_link_ok_t        = time.time()
last_land_attempt_t   = 0.0

emergency    = False
toast_text   = ""
toast_until  = 0.0

auto_running    = False
auto_cancel     = False
auto_step_label = ""   # Hangi adımda olduğunu gösterir
show_running    = False
show_cancel     = False
show_mode       = ""
show_step_label = ""
mevlana_mode    = False

fps_smooth = 0.0
FPS_ALPHA  = 0.08
auto_pose_hud = {"active": False, "x": 0.0, "y": 0.0, "yaw": 0.0}
takeoff_busy = False
last_flip_t = 0.0
hover_target_height_cm = None
hover_idle_since_t = time.time()
hover_last_sample_t = 0.0

roi_pending       = False
roi_pending_bbox  = None
roi_pending_frame = None

vision_system = None
vision_enabled = False
vision_last_target = None
vision_last_persons = []
vision_last_objects = []
vision_last_run_t = 0.0
vision_last_ok_t = 0.0
vision_last_error = ""
vision_last_infer_ms = 0.0
vision_init_thread = None
vision_init_running = False
ui_panel_visible = False

# =========================================================
# RC THREAD
# =========================================================
rc_lr = 0; rc_fb = 0; rc_ud = 0; rc_yv = 0
rc_lock    = threading.Lock()
rc_running = True
RC_HZ = 45.0
RC_IDLE_HZ = 20.0
RC_LOW_BAT_HZ = 32.0
RC_DT = 1.0 / RC_HZ
RC_IDLE_DT = 1.0 / RC_IDLE_HZ
RC_LOW_BAT_DT = 1.0 / RC_LOW_BAT_HZ


def note_link_ok():
    global link_fail_streak, last_link_ok_t
    link_fail_streak = 0
    last_link_ok_t   = time.time()


def connection_failsafe_land(reason=""):
    global emergency, auto_cancel, auto_running, mode, last_land_attempt_t
    global rc_lr, rc_fb, rc_ud, rc_yv
    if not safe_is_flying(): return False
    now = time.time()
    if now - last_land_attempt_t < LAND_RETRY_COOLDOWN: return False
    last_land_attempt_t = now
    emergency = True; auto_cancel = True; auto_running = False; mode = 0
    reset_tracking()
    with rc_lock: rc_lr = rc_fb = rc_ud = rc_yv = 0
    stop_and_hover()
    toast(f"BAGLANTI KOPTU ({reason}) -> INIS", 2.0)
    for fn in [lambda: run_sdk_command(tello.land), lambda: run_sdk_command(lambda: tello.send_command_without_return("land"))]:
        try: fn(); return True
        except: pass
    return False


def note_link_fail(reason=""):
    global link_fail_streak
    link_fail_streak += 1
    if link_fail_streak >= LINK_FAIL_STREAK_TRIG:
        connection_failsafe_land(reason); return
    if (time.time() - last_link_ok_t) >= LINK_LOST_SEC_TRIG:
        connection_failsafe_land(reason)


def rc_sender_loop():
    while rc_running:
        # Otonom veya acil durumda sadece bekle, rc degerlerine dokunma
        if emergency or auto_running or show_running or mode != 0 or takeoff_busy:
            time.sleep(RC_DT)
            continue
        # Manuel mod: ana dongunun yazdigi degerleri gonder
        with rc_lock:
            lr = int(rc_lr); fb = int(rc_fb)
            ud = int(rc_ud); yv = int(rc_yv)
        send_rc_control_safe(lr, fb, ud, yv, reason="rc_sender")
        idle_rc = (lr == 0 and fb == 0 and ud == 0 and yv == 0)
        if idle_rc:
            sleep_dt = RC_IDLE_DT
        else:
            bat_now = battery_level
            low_bat = POWER_SAVE_ENABLED and bat_now >= 0 and bat_now <= POWER_SAVE_BAT_LOW
            sleep_dt = RC_LOW_BAT_DT if low_bat else RC_DT
        time.sleep(sleep_dt)


def recover_stream(reason=""):
    global frame_read, last_watchdog_reset_t, last_frame_sig, last_frame_change_t
    now = time.time()
    if now - last_watchdog_reset_t < WATCHDOG_RESET_COOLDOWN: return False
    last_watchdog_reset_t = now
    toast(f"STREAM RESET ({reason})", 1.0)
    stop_and_hover()
    if video_mode:
        try:
            if video_cap is not None:
                video_cap.release()
        except Exception:
            pass
        return init_stream()
    try: run_sdk_command(tello.streamoff); time.sleep(0.2)
    except: pass
    try:
        run_sdk_command(tello.streamon); time.sleep(0.35)
        frame_read = tello.get_frame_read()
        last_frame_sig = None; last_frame_change_t = time.time()
        note_link_ok(); return True
    except:
        note_link_fail(f"stream:{reason}"); return False


def init_stream():
    global frame_read, last_frame_sig, last_frame_change_t, video_cap
    if video_mode:
        try:
            if video_cap is not None:
                video_cap.release()
        except Exception:
            pass
        video_cap = cv2.VideoCapture(video_path)
        if not video_cap.isOpened():
            return False
        frame_read = None
        last_frame_sig = None
        last_frame_change_t = time.time()
        note_link_ok()
        return True
    try:
        try: run_sdk_command(tello.streamoff)
        except: pass
        time.sleep(0.25)
        run_sdk_command(tello.streamon); time.sleep(0.35)
        frame_read = tello.get_frame_read()
        last_frame_sig = None; last_frame_change_t = time.time()
        note_link_ok(); return True
    except:
        note_link_fail("stream:init"); return False


def reset_tracking():
    global tracker, tracker_on, template_gray, last_good_bbox, tm_conf, successful_track_frames
    global fb_active, fb_prev_cmd, fb_enable_time, dist_target_w
    global lock_enabled, lock_hist, lock_edge, hist_conf, edge_conf, fused_conf, reacq_confirm
    global lost_since, roi_pending, roi_pending_bbox, roi_pending_frame
    global vision_track_active, vision_track_target_id, vision_track_lost_t, vision_track_bbox
    global vision_track_last_switch_t, vision_track_last_error_t, vision_track_wait_toast_t
    global danger_snapshot_last_t, danger_toast_last_t

    tracker = None; tracker_on = False
    template_gray = None; last_good_bbox = None; tm_conf = 0.0
    successful_track_frames = 0
    fb_active = False; fb_prev_cmd = 0; fb_enable_time = 0.0; dist_target_w = None
    lock_enabled = False; lock_hist = None; lock_edge = None
    hist_conf = -1.0; edge_conf = -1.0; fused_conf = 0.0; reacq_confirm = 0
    lost_since = None
    roi_pending = False; roi_pending_bbox = None; roi_pending_frame = None
    vision_track_active = False; vision_track_target_id = None
    vision_track_lost_t = None; vision_track_bbox = None
    vision_track_last_switch_t = 0.0
    vision_track_last_error_t = 0.0
    vision_track_wait_toast_t = 0.0
    danger_snapshot_last_t = 0.0
    danger_toast_last_t = 0.0


def set_auto_pose_hud(active=False, pose=None):
    global auto_pose_hud
    if not active or pose is None:
        auto_pose_hud = {"active": False, "x": 0.0, "y": 0.0, "yaw": 0.0}
        return
    auto_pose_hud = {
        "active": True,
        "x": float(pose.get("x", 0.0)),
        "y": float(pose.get("y", 0.0)),
        "yaw": float(pose.get("yaw", 0.0)),
    }


def init_vision_system():
    global DetectTrackSystem, _VISION_IMPORT_ERROR
    global vision_system, vision_enabled, vision_last_error
    global vision_last_target, vision_last_persons, vision_last_objects
    global vision_last_run_t, vision_last_ok_t, vision_last_infer_ms

    if vision_system is not None:
        vision_enabled = True
        vision_last_error = ""
        vision_last_run_t = 0.0
        return True

    try:
        vision_module = importlib.import_module("system.vision.detect_track")
        vision_module = importlib.reload(vision_module)
        DetectTrackSystem = getattr(vision_module, "DetectTrackSystem", None)
        _VISION_IMPORT_ERROR = None
    except Exception as exc:
        DetectTrackSystem = None
        _VISION_IMPORT_ERROR = exc

    if DetectTrackSystem is None:
        vision_system = None
        vision_enabled = False
        vision_last_error = str(_VISION_IMPORT_ERROR) if _VISION_IMPORT_ERROR is not None else "DetectTrackSystem import edilemedi"
        return False

    try:
        vision_system = DetectTrackSystem()
        vision_enabled = True
        vision_last_error = ""
        vision_last_target = None
        vision_last_persons = []
        vision_last_objects = []
        vision_last_run_t = 0.0
        vision_last_ok_t = 0.0
        vision_last_infer_ms = 0.0
        return True
    except Exception as exc:
        vision_system = None
        vision_enabled = False
        vision_last_error = str(exc)
        print(f"[VISION] Baslatma hatasi: {exc}")
        return False


def start_vision_init_async(label="VISION"):
    global vision_enabled, vision_init_thread, vision_init_running

    if vision_system is not None:
        vision_enabled = True
        start_vision_thread()
        toast(f"{label}: ACIK", 1.2)
        return True
    if vision_init_running:
        toast(f"{label}: yukleniyor", 1.2)
        return False

    vision_init_running = True

    def _worker():
        global vision_init_running
        try:
            ok = init_vision_system()
            if ok:
                start_vision_thread()
                toast(f"{label}: hazir (threaded)", 1.8)
            else:
                print(f"[VISION] Devre disi: {vision_last_error}")
        finally:
            vision_init_running = False

    vision_init_thread = threading.Thread(target=_worker, daemon=True)
    vision_init_thread.start()
    toast(f"{label}: yukleniyor", 1.2)
    return False


def short_vision_error(max_len=42):
    msg = (vision_last_error or "").strip()
    if not msg:
        return "bilinmeyen hata"
    msg = " ".join(msg.split())
    if len(msg) <= max_len:
        return msg
    return f"{msg[:max_len - 3]}..."


# --- VISION THREAD ---
vision_lock = threading.Lock()
vision_thread_running = False
vision_thread_frame = None
vision_thread_frame_ts = 0.0


def update_vision_cache_threaded(frame_bgr, frame_ts):
    """Ana donguden cagirilir - sadece frame'i kuyruğa koyar, BLOKLAMAZ."""
    global vision_thread_frame, vision_thread_frame_ts
    if not vision_enabled or vision_system is None:
        return
    vision_thread_frame = frame_bgr
    vision_thread_frame_ts = frame_ts


def _vision_worker_loop():
    """Arka plan thread'i - YOLO inference burada calisir."""
    global vision_last_target, vision_last_persons, vision_last_objects
    global vision_last_run_t, vision_last_ok_t
    global vision_last_error, vision_last_infer_ms

    while vision_thread_running:
        if not vision_enabled or vision_system is None:
            time.sleep(0.05)
            continue

        frame_bgr = vision_thread_frame
        frame_ts = vision_thread_frame_ts

        if frame_bgr is None:
            time.sleep(0.02)
            continue

        now = time.time()
        if now - vision_last_run_t < get_runtime_vision_interval():
            time.sleep(0.01)
            continue

        vision_last_run_t = now
        t0 = time.time()
        try:
            run_result = vision_system.run(frame_bgr.copy(), frame_ts=frame_ts)
            if isinstance(run_result, (tuple, list)) and len(run_result) >= 4:
                _, target, persons, objects = run_result[:4]
            else:
                _, target, persons = run_result
                objects = []

            fh, fw = frame_bgr.shape[:2]
            persons = sanitize_person_detections(persons, fw, fh)
            objects = sanitize_object_detections(objects, fw, fh)
            if isinstance(target, dict):
                target = dict(target)
                target_bbox = normalize_xyxy_bbox(
                    target.get("bbox"),
                    fw,
                    fh,
                    min_w=VISION_MIN_TRACK_BOX_W,
                    min_h=VISION_MIN_TRACK_BOX_H,
                )
                if target_bbox is not None:
                    target["bbox"] = target_bbox
                elif target.get("id") is None:
                    target = None

            with vision_lock:
                vision_last_target = target
                vision_last_persons = persons
                vision_last_objects = objects
                vision_last_ok_t = frame_ts
                vision_last_error = ""
                vision_last_infer_ms = (time.time() - t0) * 1000.0
        except Exception as exc:
            with vision_lock:
                vision_last_error = str(exc)
                vision_last_target = None
                vision_last_persons = []
                vision_last_objects = []
            print(f"[VISION] Kare isleme hatasi: {exc}")

        time.sleep(0.005)


def start_vision_thread():
    global vision_thread_running
    if vision_thread_running:
        return
    vision_thread_running = True
    t = threading.Thread(target=_vision_worker_loop, daemon=True)
    t.start()
    print("[VISION] Background thread baslatildi")


def stop_vision_thread():
    global vision_thread_running
    vision_thread_running = False


def draw_vision_overlay(frame_bgr, now_ts):
    if not vision_enabled:
        return
    if now_ts - vision_last_ok_t > VISION_RESULT_TTL_SEC:
        return

    with vision_lock:
        target_snapshot = dict(vision_last_target) if isinstance(vision_last_target, dict) else None
        persons_snapshot = list(vision_last_persons) if vision_last_persons else []
        objects_snapshot = list(vision_last_objects) if vision_last_objects else []

    target_id = None if target_snapshot is None else target_snapshot.get("id")
    fh, fw = frame_bgr.shape[:2]

    for obj in sanitize_object_detections(objects_snapshot, fw, fh):
        x1, y1, x2, y2 = obj.get("bbox", (0, 0, 0, 0))
        label = str(obj.get("label", "object")).upper()
        conf = float(obj.get("confidence", 0.0))
        color = (0, 214, 255)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 1)
        cv2.putText(
            frame_bgr,
            f"{label} {conf:.2f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    for person in sanitize_person_detections(persons_snapshot, fw, fh):
        x1, y1, x2, y2 = person.get("bbox", (0, 0, 0, 0))
        is_target = person.get("id") == target_id
        is_danger = bool(person.get("is_danger", False))
        status_label = str(person.get("status_label", "SAFE")).upper()
        activity_label = str(person.get("activity_label", "NOT SWIMMING")).upper()

        color = (0, 255, 0)
        if is_danger:
            color = (0, 0, 255)
        elif status_label == "WATCH":
            color = (0, 165, 255)

        box_thickness = 3 if is_target else 1
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, box_thickness)
        if vision_system is not None and hasattr(vision_system, "_draw_pose") and (is_target or is_danger):
            try:
                vision_system._draw_pose(frame_bgr, person.get("pose_points", {}), color)
            except Exception:
                pass

        label = f"{status_label} | {activity_label}"
        cv2.putText(
            frame_bgr,
            label,
            (x1, max(22, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
            cv2.LINE_AA,
        )

def do_flip(direction):
    global last_flip_t
    now = time.time()
    if now - last_flip_t < FLIP_COOLDOWN_SEC:
        return
    if video_mode:
        last_flip_t = now
        names = {'l': 'SOL', 'r': 'SAG', 'f': 'ILERI', 'b': 'GERI'}
        toast(f"SIM TAKLA: {names.get(direction, direction)}")
        return
    if not safe_is_flying(): toast("Takla icin once havalanin!"); return
    bat_now = battery_level if battery_level >= 0 else safe_get_battery(-1)
    if bat_now != -1 and bat_now < FLIP_MIN_BATTERY:
        toast(f"Takla iptal: pil dusuk ({bat_now}%)")
        return
    pre_flip_height = safe_get_height_cm(MANUAL_TAKEOFF_HOVER_CM)
    last_flip_t = now
    try:
        stop_and_hover()
        time.sleep(0.15)
        run_sdk_command(lambda: tello.flip(direction))
        stabilize_after_flip(pre_flip_height)
        names = {'l': 'SOL', 'r': 'SAG', 'f': 'ILERI', 'b': 'GERI'}
        toast(f"TAKLA: {names.get(direction, direction)} | DENGELEME")
    except Exception as e:
        last_flip_t = 0.0
        toast(f"Takla hata: {e}")


def do_double_flip(direction='f'):
    global last_flip_t
    now = time.time()
    if now - last_flip_t < max(FLIP_COOLDOWN_SEC, COMBO_FLIP_COOLDOWN_SEC):
        return
    if video_mode:
        last_flip_t = now
        toast("SIM KOMBO TAKLA 1/4 ILERI", 0.7)
        return
    if not safe_is_flying():
        toast("Double takla icin once havalanin!")
        return
    battery_now = safe_get_battery(-1)
    if battery_now != -1 and battery_now < AUTO_MIN_BATTERY:
        toast(f"Double takla iptal: Batarya dusuk ({battery_now}%)")
        return

    last_flip_t = now
    names = {'l': 'SOL', 'r': 'SAG', 'f': 'ILERI', 'b': 'GERI'}
    combo = ['f', 'l', 'b', 'r']  # ileri -> sol -> geri -> sag
    try:
        for idx, step_dir in enumerate(combo, start=1):
            if not safe_is_flying():
                toast("Double takla iptal: ucus durumu degisti")
                return
            battery_now = safe_get_battery(-1)
            if battery_now != -1 and battery_now < AUTO_MIN_BATTERY:
                toast(f"Double takla iptal: Batarya dusuk ({battery_now}%)")
                return
            stop_and_hover()
            time.sleep(0.15)
            run_sdk_command(lambda d=step_dir: tello.flip(d))
            toast(f"KOMBO TAKLA {idx}/4: {names.get(step_dir, step_dir)}", 0.75)
            if idx < len(combo):
                time.sleep(COMBO_FLIP_STEP_DELAY_SEC)
    except Exception as e:
        last_flip_t = 0.0
        toast(f"Double takla hata: {e}")


def _run_show_segment(*, lr=0, fb=0, ud=0, yv=0, duration_sec=0.5, label=""):
    global show_step_label
    show_step_label = label
    started = time.time()
    while (time.time() - started) < duration_sec:
        if show_cancel or emergency or auto_cancel:
            send_rc_control_safe(0, 0, 0, 0, reason="show_abort", report_fail=False, report_success=False)
            return False
        send_rc_control_safe(int(lr), int(fb), int(ud), int(yv), reason="show_rc", report_fail=False, report_success=False)
        time.sleep(SHOW_SEGMENT_TICK_SEC)
    send_rc_control_safe(0, 0, 0, 0, reason="show_hover", report_fail=False, report_success=False)
    time.sleep(0.12)
    return True


def _show_worker(kind):
    global show_running, show_cancel, show_mode, show_step_label, mode
    try:
        mode = 0
        reset_tracking()
        scale = get_runtime_speed_scale()
        sq_fb = max(24, int(SHOW_SQUARE_FB_SPEED * scale))
        sq_yaw = max(20, int(SHOW_SQUARE_YAW_SPEED * scale))
        eight_fb = max(22, int(SHOW_EIGHT_FB_SPEED * scale))
        eight_yaw = max(20, int(SHOW_EIGHT_YAW_SPEED * scale))

        if kind == "square":
            toast("GOSTERI: Kare basladi", 1.2)
            for i in range(1, 5):
                if not _run_show_segment(fb=sq_fb, duration_sec=SHOW_SQUARE_EDGE_SEC, label=f"Kare kenar {i}/4"):
                    return
                if not _run_show_segment(yv=sq_yaw, duration_sec=SHOW_SQUARE_TURN_SEC, label=f"Kare donus {i}/4"):
                    return
            toast("GOSTERI: Kare tamamlandi", 1.0)
        elif kind == "eight":
            toast("GOSTERI: 8 basladi", 1.2)
            if not _run_show_segment(fb=eight_fb, yv=+eight_yaw, duration_sec=SHOW_EIGHT_LOOP_SEC, label="8 halka 1/2"):
                return
            if not _run_show_segment(fb=eight_fb, yv=-eight_yaw, duration_sec=SHOW_EIGHT_LOOP_SEC, label="8 halka 2/2"):
                return
            toast("GOSTERI: 8 tamamlandi", 1.0)
    finally:
        send_rc_control_safe(0, 0, 0, 0, reason="show_end", report_fail=False, report_success=False)
        show_running = False
        show_cancel = False
        show_mode = ""
        show_step_label = ""


def start_show_mode(kind):
    global show_running, show_cancel, show_mode, mevlana_mode, mode
    if show_running:
        toast("GOSTERI zaten calisiyor")
        return
    if auto_running or takeoff_busy or emergency:
        toast("GOSTERI icin once otonomu/acil modu kapat")
        return
    if not safe_is_flying():
        toast("GOSTERI icin drone havada olmali")
        return
    battery_now = safe_get_battery(-1)
    if battery_now != -1 and battery_now < AUTO_MIN_BATTERY:
        toast(f"GOSTERI iptal: Batarya dusuk ({battery_now}%)")
        return
    mode = 0
    mevlana_mode = False
    show_cancel = False
    show_running = True
    show_mode = kind
    stop_and_hover()
    threading.Thread(target=_show_worker, args=(kind,), daemon=True).start()


def toggle_mevlana_mode():
    global mevlana_mode, show_running, show_cancel, show_mode
    if auto_running or emergency:
        toast("Mevlana modu su an acilamaz")
        return
    if not safe_is_flying():
        toast("Mevlana modu icin once havalanin")
        return
    if show_running:
        show_cancel = True
        show_running = False
        show_mode = ""
    mevlana_mode = not mevlana_mode
    if mevlana_mode:
        toast("MEVLANA: ACIK", 1.0)
    else:
        stop_and_hover()
        toast("MEVLANA: KAPALI", 1.0)


# =========================================================
# OTONOM WORKER — rc_control tabanlı, NON-BLOCKING
# =========================================================
def start_manual_takeoff():
    global takeoff_busy
    if takeoff_busy:
        toast("Kalkis zaten baslatildi")
        return
    if safe_is_flying():
        toast("Drone zaten havada")
        return
    takeoff_busy = True

    def _worker():
        global takeoff_busy
        try:
            toast("KALKIS BASLADI", 1.2)
            ok = fast_takeoff()
            if ok:
                settle_manual_takeoff_height()
                h_now = int(round(safe_get_height_cm(MANUAL_TAKEOFF_HOVER_CM)))
                reset_hover_hold_target(h_now)
                toast(f"KALKIS {h_now}cm", 1.4)
            else:
                toast("Kalkis zaman asimi")
        except Exception as e:
            toast(f"Kalkis hata: {e}")
        finally:
            takeoff_busy = False

    try:
        threading.Thread(target=_worker, daemon=True).start()
    except Exception:
        takeoff_busy = False
        raise


def autonomous_worker():
    global auto_running, auto_cancel, mode, auto_step_label
    global rc_lr, rc_fb, rc_ud, rc_yv

    def should_abort():
        return auto_cancel or emergency

    def send_rc(lr=0, fb=0, ud=0, yv=0):
        return send_rc_control_safe(lr, fb, ud, yv, reason="auto_rc", report_success=True)

    def move_timed(lr_spd=0, fb_spd=0, yv_spd=0, ud_spd=0, duration_sec=1.0, label=""):
        global auto_step_label
        tick_sec = 0.05
        commanded_sec = 0.0
        deadline = time.time() + max(duration_sec * 2.4, duration_sec + 2.0)
        auto_step_label = label
        toast(f"OTONOM: {label}", duration_sec + 0.5)
        while commanded_sec < duration_sec:
            if should_abort():
                send_rc(0, 0, 0, 0)
                return False
            if time.time() > deadline:
                toast(f"OTONOM: {label} zaman asimi", 1.5)
                send_rc(0, 0, 0, 0)
                return False
            if send_rc(lr_spd, fb_spd, ud_spd, yv_spd):
                commanded_sec += tick_sec
            elif should_abort():
                send_rc(0, 0, 0, 0)
                return False
            time.sleep(tick_sec)
        send_rc(0, 0, 0, 0)
        time.sleep(0.3)
        return True

    def normalize_heading(deg):
        return float(deg % 360.0)

    def signed_heading_delta(current_deg, target_deg):
        return ((target_deg - current_deg + 180.0) % 360.0) - 180.0

    def safe_get_height_cm(default=0.0):
        try:
            return float(run_sdk_command(tello.get_height))
        except Exception:
            return float(default)

    def safe_get_pad_id():
        try:
            return int(run_sdk_command(tello.get_mission_pad_id))
        except Exception:
            return -1

    def cm_to_duration(distance_cm, speed_cmd):
        return max(0.35, float(abs(distance_cm)) / max(1.0, float(abs(speed_cmd))))

    def rotate_precise(label, clockwise=True, degrees=90):
        global auto_step_label
        if should_abort():
            return False
        auto_step_label = label
        toast(f"OTONOM: {label}", 1.4)
        send_rc(0, 0, 0, 0)
        try:
            if clockwise:
                run_sdk_command(lambda: tello.rotate_clockwise(int(degrees)))
            else:
                run_sdk_command(lambda: tello.rotate_counter_clockwise(int(degrees)))
        except Exception:
            yaw_cmd = AUTO_YAW_SPEED if clockwise else -AUTO_YAW_SPEED
            return move_timed(yv_spd=yaw_cmd, duration_sec=ROT_90_SEC, label=label)
        send_rc(0, 0, 0, 0)
        time.sleep(0.25)
        return not should_abort()

    def update_pose_estimate(pose, action, speed, duration_sec):
        distance_cm = float(abs(speed) * duration_sec)
        yaw_rad = np.deg2rad(pose["yaw"])
        fwd_x = np.cos(yaw_rad)
        fwd_y = np.sin(yaw_rad)
        right_x = -np.sin(yaw_rad)
        right_y = np.cos(yaw_rad)
        if action == "forward":
            pose["x"] += fwd_x * distance_cm
            pose["y"] += fwd_y * distance_cm
        elif action == "backward":
            pose["x"] -= fwd_x * distance_cm
            pose["y"] -= fwd_y * distance_cm
        elif action == "right":
            pose["x"] += right_x * distance_cm
            pose["y"] += right_y * distance_cm
        elif action == "left":
            pose["x"] -= right_x * distance_cm
            pose["y"] -= right_y * distance_cm
        elif action == "yaw_right":
            pose["yaw"] = normalize_heading(pose["yaw"] + 90.0)
        elif action == "yaw_left":
            pose["yaw"] = normalize_heading(pose["yaw"] - 90.0)

    def execute_route_step(pose, action, label, speed, duration_sec):
        if action == "forward":
            ok = move_timed(fb_spd=speed, duration_sec=duration_sec, label=label)
        elif action == "backward":
            ok = move_timed(fb_spd=-speed, duration_sec=duration_sec, label=label)
        elif action == "right":
            ok = move_timed(lr_spd=speed, duration_sec=duration_sec, label=label)
        elif action == "left":
            ok = move_timed(lr_spd=-speed, duration_sec=duration_sec, label=label)
        elif action == "yaw_right":
            ok = rotate_precise(label, clockwise=True, degrees=90)
        elif action == "yaw_left":
            ok = rotate_precise(label, clockwise=False, degrees=90)
        else:
            raise ValueError(f"Bilinmeyen rota adimi: {action}")
        if ok:
            update_pose_estimate(pose, action, speed, duration_sec)
            pose["z"] = safe_get_height_cm(pose.get("z", 0.0))
            set_auto_pose_hud(True, pose)
        return ok

    def hover_wait(sec=0.5):
        t0 = time.time()
        while time.time() - t0 < sec:
            if should_abort(): return False
            send_rc(0, 0, 0, 0)
            time.sleep(0.05)
        return True

    def stabilize_altitude(pose, target_height_cm, label_prefix="Irtifa duzeltme", max_passes=2):
        if target_height_cm <= 0:
            return True

        target_height_cm = float(target_height_cm)
        for _ in range(max_passes):
            current_h = safe_get_height_cm(target_height_cm)
            pose["z"] = current_h
            set_auto_pose_hud(True, pose)
            delta = target_height_cm - current_h
            if abs(delta) <= AUTO_ALT_HOLD_TOL_CM:
                return True

            move_sec = min(1.8, cm_to_duration(delta, AUTO_ALT_CORR_SPEED))
            label = f"{label_prefix} {'yuksel' if delta > 0 else 'alcal'} {int(abs(delta))}cm"
            if not move_timed(
                ud_spd=AUTO_ALT_CORR_SPEED if delta > 0 else -AUTO_ALT_CORR_SPEED,
                duration_sec=move_sec,
                label=label,
            ):
                return False
            if not hover_wait(0.35):
                return False

        current_h = safe_get_height_cm(target_height_cm)
        pose["z"] = current_h
        set_auto_pose_hud(True, pose)
        toast(f"OTONOM: Irtifa sapmasi {int(abs(target_height_cm - current_h))}cm", 1.1)
        return abs(target_height_cm - current_h) <= AUTO_ALT_HOLD_TOL_CM * 1.5

    def capture_home_anchor():
        anchor = {"pad_enabled": False, "pad_id": -1}
        if not USE_MISSION_PAD_DOCKING:
            return anchor
        try:
            run_sdk_command(tello.enable_mission_pads)
            run_sdk_command(lambda: tello.set_mission_pad_detection_direction(2))
            anchor["pad_enabled"] = True
        except Exception:
            return anchor

        scan_deadline = time.time() + AUTO_PAD_SCAN_SEC
        while time.time() < scan_deadline:
            if should_abort():
                return anchor
            pad_id = safe_get_pad_id()
            if pad_id != -1:
                anchor["pad_id"] = pad_id
                toast(f"OTONOM: Ev padi m{pad_id} kaydedildi", 1.5)
                return anchor
            send_rc(0, 0, 0, 0)
            time.sleep(0.1)

        toast("OTONOM: Mission pad yok, yerel koordinat ile donecek", 1.5)
        return anchor

    def align_to_home_heading(pose, home_pose):
        delta = signed_heading_delta(pose["yaw"], home_pose["yaw"])
        if abs(delta) < 45.0:
            return True

        turn_action = "yaw_right" if delta > 0 else "yaw_left"
        turns = int(round(abs(delta) / 90.0))
        for idx in range(turns):
            label = f"Eve hizalanma {idx + 1}/{turns}"
            if not execute_route_step(pose, turn_action, label, AUTO_YAW_SPEED, ROT_90_SEC):
                return False
            if not hover_wait(0.25):
                return False
        return True

    def return_to_home_local(pose, home_pose):
        for pass_idx in range(AUTO_RETURN_MAX_PASSES):
            dx = float(home_pose["x"] - pose["x"])
            dy = float(home_pose["y"] - pose["y"])
            home_dist = float(np.hypot(dx, dy))
            if home_dist <= AUTO_HOME_TOL_CM:
                toast("OTONOM: Ev koordinatina geri gelindi", 1.0)
                return True

            yaw_rad = np.deg2rad(pose["yaw"])
            body_right = (-np.sin(yaw_rad) * dx) + (np.cos(yaw_rad) * dy)
            body_forward = (np.cos(yaw_rad) * dx) + (np.sin(yaw_rad) * dy)
            corrections = []

            if abs(body_right) > AUTO_HOME_TOL_CM:
                side_action = "right" if body_right >= 0 else "left"
                corrections.append((
                    side_action,
                    f"Ev hizalama yanal {int(abs(body_right))}cm",
                    AUTO_HOME_CORR_SPEED,
                    cm_to_duration(body_right, AUTO_HOME_CORR_SPEED),
                ))
            if abs(body_forward) > AUTO_HOME_TOL_CM:
                move_action = "forward" if body_forward >= 0 else "backward"
                corrections.append((
                    move_action,
                    f"Ev hizalama ileri-geri {int(abs(body_forward))}cm",
                    AUTO_HOME_CORR_SPEED,
                    cm_to_duration(body_forward, AUTO_HOME_CORR_SPEED),
                ))

            if not corrections:
                break

            toast(f"OTONOM: Eve donus duzeltme {pass_idx + 1}/{AUTO_RETURN_MAX_PASSES}", 1.2)
            for action, label, speed, duration_sec in corrections:
                if not execute_route_step(pose, action, label, speed, duration_sec):
                    return False
                if not hover_wait(0.35):
                    return False

            if not stabilize_altitude(pose, home_pose["z"], label_prefix="Eve donus irtifa", max_passes=1):
                return False

        home_dist = float(np.hypot(pose["x"] - home_pose["x"], pose["y"] - home_pose["y"]))
        toast(f"OTONOM: Eve donus sapmasi {int(home_dist)}cm", 1.2)
        return home_dist <= AUTO_HOME_TOL_CM * 1.4

    def dock_to_home_anchor(home_anchor):
        global auto_step_label
        auto_step_label = "Sarj istasyonu yaklasmasi"

        if home_anchor["pad_id"] != -1:
            try:
                toast(f"OTONOM: Sarj padi m{home_anchor['pad_id']} hizalaniyor", 1.6)
                run_sdk_command(lambda: tello.go_xyz_speed_mid(0, 0, AUTO_DOCK_APPROACH_Z_CM, AUTO_DOCK_SPEED, home_anchor["pad_id"]))
                if not hover_wait(1.0):
                    return False
                run_sdk_command(lambda: tello.go_xyz_speed_mid(0, 0, AUTO_DOCK_FINAL_Z_CM, AUTO_DOCK_SPEED, home_anchor["pad_id"]))
                if not hover_wait(1.0):
                    return False
                return True
            except Exception as exc:
                toast(f"OTONOM: Pad docking fallback ({exc})", 1.6)

        current_h = safe_get_height_cm(AUTO_DOCK_APPROACH_Z_CM)
        if current_h > AUTO_DOCK_APPROACH_Z_CM + 8.0:
            descend_cm = current_h - AUTO_DOCK_APPROACH_Z_CM
            descend_sec = cm_to_duration(descend_cm, AUTO_DOCK_SPEED)
            if not move_timed(ud_spd=-AUTO_DOCK_SPEED, duration_sec=descend_sec, label="Sarj istasyonu ilk alcalma"):
                return False
            if not hover_wait(0.9):
                return False

        current_h = safe_get_height_cm(AUTO_DOCK_APPROACH_Z_CM)
        if current_h > AUTO_DOCK_FINAL_Z_CM + 6.0:
            descend_cm = current_h - AUTO_DOCK_FINAL_Z_CM
            descend_sec = cm_to_duration(descend_cm, AUTO_DOCK_SPEED)
            if not move_timed(ud_spd=-AUTO_DOCK_SPEED, duration_sec=descend_sec, label="Sarj istasyonu son yaklasma"):
                return False
        return hover_wait(0.8)

    try:
        home_pose = {"x": 0.0, "y": 0.0, "z": 0.0, "yaw": 0.0}
        pose = home_pose.copy()
        home_anchor = {"pad_enabled": False, "pad_id": -1}

        auto_cancel = False
        mode        = 0
        pressed.clear(); key_events.clear()
        with rc_lock: rc_lr = rc_fb = rc_ud = rc_yv = 0
        reset_tracking(); send_rc(0,0,0,0); time.sleep(0.2)

        if emergency: return
        battery_now = safe_get_battery(-1)
        if battery_now != -1 and battery_now < AUTO_MIN_BATTERY:
            raise RuntimeError(f"Batarya dusuk ({battery_now}%)")

        # Kalkış ve kalkış noktasını kaydet
        if not safe_is_flying():
            auto_step_label = "Kalkis"
            toast("OTONOM: Kalkis...")
            if not fast_takeoff():
                raise RuntimeError("Kalkis tamamlanamadi")
            if not hover_wait(1.5): return
        else:
            toast("OTONOM: Zaten havada")
            if not hover_wait(0.5): return

        home_pose["z"] = safe_get_height_cm(0.0)
        home_anchor = capture_home_anchor()
        pose = home_pose.copy()
        set_auto_pose_hud(True, pose)
        toast("OTONOM: Ev koordinati kaydedildi", 1.2)

        # Rota
        for action, label, speed, duration_sec in AUTO_ROUTE_STEPS:
            if not execute_route_step(pose, action, label, speed, duration_sec): return
            if not stabilize_altitude(pose, home_pose["z"], label_prefix="Rota irtifa", max_passes=1): return
            if not hover_wait(AUTO_STEP_HOVER_SEC): return

        # Kalkış noktasına dönüş
        if not return_to_home_local(pose, home_pose): return
        if not align_to_home_heading(pose, home_pose): return
        if not stabilize_altitude(pose, home_pose["z"], label_prefix="Eve varis irtifa", max_passes=2): return
        if not dock_to_home_anchor(home_anchor): return
        home_dist = float(np.hypot(pose["x"] - home_pose["x"], pose["y"] - home_pose["y"]))

        if home_anchor["pad_id"] != -1:
            auto_step_label = "Sarj padi ustu inis"
            toast(f"OTONOM: Ev padi m{home_anchor['pad_id']} uzerinde inis", 2.5)
        elif home_dist <= AUTO_HOME_TOL_CM:
            auto_step_label = "Sarj noktasina inis"
            toast("OTONOM: Ev koordinatinda kontrollu inis", 2.5)
        else:
            auto_step_label = "Inis"
            toast(f"OTONOM: Eve yakin ({int(home_dist)}cm) -> kontrollu inis", 2.5)

        if not hover_wait(AUTO_FINAL_HOVER_SEC): return
        run_sdk_command(tello.land)

    except Exception as e:
        toast(f"Otonom hata: {e}", 2.0)

    finally:
        send_rc(0,0,0,0)
        set_auto_pose_hud(False)
        try:
            run_sdk_command(tello.disable_mission_pads)
        except Exception:
            pass
        pressed.clear(); key_events.clear()
        with rc_lock:
            rc_lr = rc_fb = rc_ud = rc_yv = 0
        auto_running    = False
        auto_step_label = ""
        mode            = 0

# =========================================================
# BAŞLAT
# =========================================================
if __name__ == "__main__":
    runtime_args = parse_runtime_args()
    if runtime_args.video:
        candidate_video_path = runtime_args.video.strip()
        if not os.path.isabs(candidate_video_path):
            candidate_video_path = os.path.abspath(os.path.join(PROJECT_ROOT, candidate_video_path))
        video_mode = True
        video_path = candidate_video_path
        video_loop = bool(runtime_args.loop_video)
        print(f"[VIDEO TEST] Kaynak: {video_path}")
        print(f"[VIDEO TEST] Loop: {'ACIK' if video_loop else 'KAPALI'}")

    try:
        connect_tello()
    except Exception as exc:
        print(f"TELLO baglanti hatasi: {exc}")
        raise SystemExit(1)

    force_sdk_speed_max()
    try:    bat0 = safe_get_battery(-1)
    except: bat0 = -1
    print("BAT:", bat0, "%")
    battery_level = int(bat0)
    last_battery_poll_t = time.time()

    start_keyboard_listener()
    cv2.namedWindow("TELLO UI", cv2.WINDOW_NORMAL)
    startup_frame = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
    cv2.putText(startup_frame, "TELLO UI BASLATILIYOR...", (35, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.imshow("TELLO UI", startup_frame)
    cv2.waitKey(1)

    ok = init_stream()
    if not ok: recover_stream("init fail")

    t_rc = threading.Thread(target=rc_sender_loop, daemon=True)
    t_rc.start()

    if VISION_AUTO_ENABLE:
        start_vision_init_async("VISION")

    last_speed_force_t = time.time()
    SPEED_FORCE_EVERY  = 10.0

    if video_mode:
        toast("VIDEO TEST | P vision | G takip | TAB panel | --video modu", 3.0)
    else:
        toast("Hazir | F foto | P vision | G takip | TAB panel | V kalkis | R otonom", 3.0)

    man_lr = man_fb = man_ud = man_yv = 0
    
    try:
        while True:
            loop_start = time.time()
            now        = time.time()

            if mode == 0 and (not takeoff_busy) and (now - last_speed_force_t > SPEED_FORCE_EVERY):
                last_speed_force_t = now
                force_sdk_speed_max()

            if video_mode:
                raw = None
                if video_cap is not None:
                    ok_frame, raw = video_cap.read()
                    if not ok_frame:
                        if video_loop:
                            video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ok_frame, raw = video_cap.read()
                        if not ok_frame:
                            raw = None
            else:
                raw = frame_read.frame if frame_read is not None else None
            if raw is None:
                ok_rec = recover_stream("frame none")
                if not ok_rec: note_link_fail("frame_none")
                dummy = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
                frame_msg = "VIDEO BITTI / FRAME YOK..." if video_mode else "FRAME YOK..."
                cv2.putText(dummy, frame_msg, (35, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
                cv2.imshow("TELLO UI", dummy)
                cv2.waitKey(1); time.sleep(0.03)
                continue
    
            sig = watchdog_frame_signature(raw)
            if sig is not None:
                if last_frame_sig is None or sig != last_frame_sig:
                    last_frame_sig = sig; last_frame_change_t = now; note_link_ok()
                elif (now - last_frame_change_t) > WATCHDOG_STALE_SEC:
                    ok_rec = recover_stream("stale")
                    if not ok_rec: note_link_fail("stale")
                    last_frame_change_t = now
    
            if video_mode or not video_frame_is_rgb:
                frame_disp = raw.copy()
            else:
                frame_disp = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
            frame_disp = cv2.resize(frame_disp, (DISPLAY_W, DISPLAY_H))
            # Vision frame'i thread'e gonder (BLOKLAMAZ)
            update_vision_cache_threaded(frame_disp.copy(), now)
            hh, ww = frame_disp.shape[:2]
    
            # --- Tuş okuma ---
            pressed_once = pop_key_presses()
            kcv = cv2.waitKey(1) & 0xFF
            if kcv != 255:
                try:
                    if kcv in (13, 10): pressed_once.append("enter")
                    elif kcv == 9: pressed_once.append("tab")
                    else:
                        ch = normalize_key_token(chr(kcv))
                        if ch: pressed_once.append(ch)
                except: pass
    
            photo_requested = 'f' in pressed_once
    
            # Otonom/gosteri çalışırken sadece iptal/acil tuşları
            if auto_running or show_running:
                pressed_once = [k for k in pressed_once if k in ('space', 'm', 'tab') or k in CANCEL_KEYS]
    
            # --- Tuş işleme ---
            if auto_running and photo_requested and 'f' not in pressed_once:
                pressed_once.append('f')
            for k in dedupe_key_events(pressed_once):
    
                if k == 'tab':
                    ui_panel_visible = not ui_panel_visible
                    toast("PANEL: ACIK" if ui_panel_visible else "PANEL: KAPALI", 1.2)
    
                elif k == 'i' and not auto_running: do_flip('f')
                elif k == 'k' and not auto_running: do_flip('b')
                elif k == 'j' and not auto_running: do_flip('l')
                elif k == 'l' and not auto_running: do_flip('r')
                elif k == 'u' and not auto_running:
                    threading.Thread(target=do_double_flip, args=('f',), daemon=True).start()
                elif k == '8' and not auto_running:
                    start_show_mode("eight")
                elif k == '9' and not auto_running:
                    start_show_mode("square")
                elif k == '7' and not auto_running:
                    toggle_mevlana_mode()
    
                elif k == 'v' and not auto_running:
                    start_manual_takeoff()
    
                elif k == 'f':
                    save_photo(frame_disp.copy())
    
                elif k == 'n' and not auto_running:
                    if video_mode:
                        sim_is_flying = False
                        toast("SIM INIS")
                    else:
                        try: run_sdk_command(tello.land); toast("INIS")
                        except Exception as e: toast(f"Inis hata: {e}")
    
                elif k == 'h' and not auto_running:
                    stop_and_hover()
                    reset_hover_hold_target()
                    toast("HOVER")
    
                elif k == 'p' and not auto_running:
                    if vision_enabled:
                        vision_enabled = False
                        stop_vision_thread()
                        vision_last_target = None
                        vision_last_persons = []
                        vision_last_objects = []
                        toast("VISION: KAPALI")
                    else:
                        start_vision_init_async("VISION")
    
                elif k == 'm':
                    emergency = False; auto_cancel = True; auto_running = False
                    mode = 0; reset_tracking(); stop_and_hover()
                    with rc_lock: rc_lr = rc_fb = rc_ud = rc_yv = 0
                    pressed.clear(); key_events.clear()
                    toast("MOD: MANUEL")
    
                elif k == 'g' and not auto_running:
                    # VISION TABANLI INSAN TAKIBI BASLAT
                    if not vision_enabled:
                        start_vision_init_async("VISION")
                        continue
                    with vision_lock:
                        _target = vision_last_target
                        _persons = list(vision_last_persons) if vision_last_persons else []
                    _persons = sanitize_person_detections(_persons, ww, hh)
                    if _target is not None and _target.get("id") is not None:
                        vision_track_active = True
                        vision_track_target_id = _target.get("id")
                        vision_track_lost_t = None
                        vision_track_last_switch_t = time.time()
                        mode = 1
                        fb_active = False; fb_prev_cmd = 0; dist_target_w = None
                        toast(f"VISION TAKIP: ID {vision_track_target_id} | Y: mesafe", 2.0)
                    elif _persons:
                        best = max(_persons, key=lambda p: float(p.get("risk", 0.0)))
                        vision_track_active = True
                        vision_track_target_id = best.get("id")
                        vision_track_lost_t = None
                        vision_track_last_switch_t = time.time()
                        mode = 1
                        fb_active = False; fb_prev_cmd = 0; dist_target_w = None
                        toast(f"VISION TAKIP: ID {vision_track_target_id} | Y: mesafe", 2.0)
                    else:
                        # Henuz kisi yok ama takip modunu baslat, bulunca kilitlenir
                        vision_track_active = True
                        vision_track_target_id = None
                        vision_track_lost_t = None
                        mode = 1
                        fb_active = False; fb_prev_cmd = 0; dist_target_w = None
                        toast("VISION TAKIP: Hedef aranıyor...", 2.0)
    
                elif k == 'enter' and not auto_running:
                    # ENTER de ayni sekilde vision takip baslatir
                    if not vision_enabled:
                        start_vision_init_async("VISION")
                        continue
                    with vision_lock:
                        _target = vision_last_target
                        _persons = list(vision_last_persons) if vision_last_persons else []
                    _persons = sanitize_person_detections(_persons, ww, hh)
                    if _target is not None and _target.get("id") is not None:
                        vision_track_active = True
                        vision_track_target_id = _target.get("id")
                        vision_track_lost_t = None
                        vision_track_last_switch_t = time.time()
                        mode = 1
                        fb_active = False; fb_prev_cmd = 0; dist_target_w = None
                        toast(f"VISION TAKIP: ID {vision_track_target_id}", 2.0)
                    elif _persons:
                        best = max(_persons, key=lambda p: float(p.get("risk", 0.0)))
                        vision_track_active = True
                        vision_track_target_id = best.get("id")
                        vision_track_lost_t = None
                        vision_track_last_switch_t = time.time()
                        mode = 1
                        fb_active = False; fb_prev_cmd = 0; dist_target_w = None
                        toast(f"VISION TAKIP: ID {vision_track_target_id}", 2.0)
                    else:
                        vision_track_active = True
                        vision_track_target_id = None
                        vision_track_lost_t = None
                        mode = 1
                        fb_active = False; fb_prev_cmd = 0; dist_target_w = None
                        toast("VISION TAKIP: Hedef aranıyor...", 2.0)
    
                elif k == 'y' and not auto_running:
                    if mode == 1 and vision_track_active and vision_track_bbox is not None:
                        bw = int(vision_track_bbox[2] - vision_track_bbox[0])
                        if bw > 25:
                            dist_target_w = float(bw); fb_active = True
                            fb_enable_time = time.time()+0.5; fb_prev_cmd = 0
                            toast(f"MESAFE KORUMA: ON (W={int(dist_target_w)})")
                        else: toast("Y: hedef cok kucuk")
                    else: toast("Y: once G ile takip baslat")
    
                elif k == 't' and not auto_running:
                    toast("KILIT: Vision takipte otomatik")

                elif k == 'b' and not auto_running:
                    # HEDEF DEGISTIR: Bir sonraki kisiye gec
                    with vision_lock:
                        _persons = list(vision_last_persons) if vision_last_persons else []
                    _persons = sanitize_person_detections(_persons, ww, hh)
                    if mode == 1 and vision_track_active and _persons:
                        ids = [p.get("id") for p in _persons if p.get("id") is not None]
                        ids = list(dict.fromkeys(ids))
                        if not ids:
                            toast("B: Gecerli hedef yok")
                            continue
                        if vision_track_target_id in ids:
                            idx = ids.index(vision_track_target_id)
                            next_idx = (idx + 1) % len(ids)
                        else:
                            next_idx = 0
                        vision_track_target_id = ids[next_idx]
                        vision_track_lost_t = None
                        vision_track_last_switch_t = time.time()
                        toast(f"HEDEF: ID {vision_track_target_id} ({next_idx+1}/{len(ids)})", 1.5)
                    else:
                        toast("B: Once G ile takip baslat")
    
                elif k == 'r':
                    if takeoff_busy:
                        toast("Kalkis suruyor, bekleyin")
                    elif not auto_running and not emergency:
                        auto_cancel = False; auto_running = True
                        try:
                            threading.Thread(target=autonomous_worker, daemon=True).start()
                            toast("OTONOM: BASLADI | O/Ö ile iptal edilir")
                        except Exception as e:
                            auto_running = False; toast(f"Thread hata: {e}")
                    elif auto_running:
                        toast("Otonom zaten calisiyor")
    
                elif k in CANCEL_KEYS:
                    auto_cancel = True
                    show_cancel = True
                    mevlana_mode = False
                    roi_pending = False; roi_pending_bbox = None; roi_pending_frame = None
                    stop_and_hover()
                    with rc_lock: rc_lr = rc_fb = rc_ud = rc_yv = 0
                    pressed.clear(); key_events.clear()
                    toast("OTONOM/GOSTERI: IPTAL")
    
                elif k == 'space':
                    emergency = True; auto_cancel = True
                    show_cancel = True
                    mevlana_mode = False
                    roi_pending = False; roi_pending_bbox = None; roi_pending_frame = None
                    reset_tracking(); stop_and_hover()
                    with rc_lock: rc_lr = rc_fb = rc_ud = rc_yv = 0
                    pressed.clear(); key_events.clear()
                    toast("!!! ACIL DURDURMA !!!")
    
                elif k == 'c':
                    raise SystemExit
    
            # =========================================================
            # RC HESAPLA
            # =========================================================
            lr, fb, ud, yv = 0, 0, 0, 0
    
            # --- VISION TABANLI TAKİP ---
            if mode == 1 and not emergency and not auto_running:
                motion_scale = power_save_motion_scale(battery_level)
                if vision_track_active and vision_enabled:
                    try:
                        # Vision sonuclarini thread-safe oku
                        with vision_lock:
                            _persons = list(vision_last_persons) if vision_last_persons else []
                        _persons = sanitize_person_detections(_persons, ww, hh)

                        tracked_person = None
                        next_target_id = vision_track_target_id
                        danger_cancelled = False

                        if _persons:
                            # Kilitliyken hedef ID kesinlikle sabit kalsin.
                            if VISION_STICKY_LOCK and vision_track_target_id is not None:
                                tracked_person = next(
                                    (p for p in _persons if p.get("id") == vision_track_target_id),
                                    None,
                                )
                                next_target_id = vision_track_target_id
                            else:
                                tracked_person, next_target_id = choose_tracked_person(
                                    _persons,
                                    vision_track_target_id,
                                    vision_track_bbox,
                                )

                            # Kilitli kisiyi takip ederken diger kisilerde danger cikarsa kilidi hemen iptal et.
                            danger_person = find_first_danger_person(_persons, ignore_id=vision_track_target_id)
                            if danger_person is not None:
                                now_danger = time.time()
                                if DANGER_AUTO_SNAPSHOT and (now_danger - danger_snapshot_last_t) >= DANGER_SNAPSHOT_COOLDOWN_SEC:
                                    save_photo(frame_disp.copy())
                                    danger_snapshot_last_t = now_danger
                                if (now_danger - danger_toast_last_t) >= DANGER_TOAST_COOLDOWN_SEC:
                                    d_id = danger_person.get("id", "?")
                                    toast(f"DANGER ID:{d_id} -> KILIT IPTAL", 1.5)
                                    danger_toast_last_t = now_danger
                                danger_cancelled = True
                                mode = 0
                                reset_tracking()
                                stop_and_hover()
                                lr = fb = ud = yv = 0

                        if (not danger_cancelled) and tracked_person is not None and next_target_id != vision_track_target_id:
                            now_switch = time.time()
                            can_switch = (
                                vision_track_target_id is None
                                or (now_switch - vision_track_last_switch_t) >= VISION_TARGET_SWITCH_COOLDOWN_SEC
                            )
                            if can_switch:
                                vision_track_target_id = next_target_id
                                vision_track_last_switch_t = now_switch
                                if vision_track_target_id is not None:
                                    risk_txt = float(tracked_person.get("risk", 0.0))
                                    toast(f"KILITLENDI: ID {vision_track_target_id} R:{risk_txt:.2f}", 1.2)
                            else:
                                tracked_person = None

                    except Exception as track_exc:
                        tracked_person = None
                        danger_cancelled = False
                        now_err = time.time()
                        if (now_err - vision_track_last_error_t) >= VISION_TRACK_ERROR_COOLDOWN_SEC:
                            vision_track_last_error_t = now_err
                            toast(f"TAKIP HATA: {str(track_exc)[:48]}", 1.0)

                    if danger_cancelled:
                        tracked_person = None
                    elif tracked_person is not None:
                        # HEDEF BULUNDU
                        x1, y1, x2, y2 = tracked_person.get("bbox", (0, 0, 0, 0))
                        bw = x2 - x1
                        bh = y2 - y1
                        cx = x1 + bw // 2
                        cy = y1 + bh // 2
                        vision_track_bbox = (x1, y1, x2, y2)
                        vision_track_lost_t = None

                        # Hedef kutusunu ciz
                        is_danger = bool(tracked_person.get("is_danger", False))
                        box_color = (0, 0, 255) if is_danger else (0, 255, 0)
                        cv2.rectangle(frame_disp, (x1, y1), (x2, y2), box_color, 3)
                        cv2.circle(frame_disp, (cx, cy), 5, box_color, -1)
                        status_lbl = str(tracked_person.get("status_label", "SAFE"))
                        activity_lbl = str(tracked_person.get("activity_label", "NOT SWIMMING"))
                        risk_val = float(tracked_person.get("risk", 0.0))
                        tid = tracked_person.get("id", "?")
                        cv2.putText(frame_disp,
                                    f"TAKIP ID:{tid} {status_lbl} | {activity_lbl} R:{risk_val:.2f}",
                                    (x1, max(22, y1 - 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, box_color, 2, cv2.LINE_AA)

                        status_upper = status_lbl.upper()
                        # Etiket metni değişken olabileceği için ana karar kırmızı (danger) bayrağından verilir.
                        track_allowed = (not is_danger) if TRACK_ONLY_SAFE_PERSON else True

                        if track_allowed:
                            # YAW komutu (yatay takip)
                            errx = cx - ww // 2
                            if abs(errx) > DEAD_ZONE:
                                yv_raw = errx * TRACK_GAIN_YAW * (TRACK_SPEED / 100.)
                                yv = int(clamp(yv_raw, -MAX_YAW, MAX_YAW))
                                if 0 < abs(yv) < MIN_CMD_YAW:
                                    yv = int(np.sign(yv) * MIN_CMD_YAW)

                            # UD komutu (dikey takip)
                            erry = cy - hh // 2
                            if abs(erry) > DEAD_ZONE:
                                ud_raw = -erry * TRACK_GAIN_UD * (TRACK_SPEED / 100.)
                                ud = int(clamp(ud_raw, -MAX_UD, MAX_UD))
                                if 0 < abs(ud) < MIN_CMD_UD:
                                    ud = int(np.sign(ud) * MIN_CMD_UD)

                            # FB komutu (mesafe koruma)
                            if fb_active and time.time() > fb_enable_time and dist_target_w is not None:
                                w_now = max(1., float(bw))
                                tw = float(dist_target_w)
                                band_lo = tw * (1. - FB_BAND)
                                band_hi = tw * (1. + FB_BAND)
                                if w_now < tw * FB_MIN_W:
                                    target_fb = +FB_MAX_FWD
                                elif w_now > tw * FB_MAX_W:
                                    target_fb = -FB_MAX_BWD
                                elif band_lo <= w_now <= band_hi:
                                    target_fb = 0
                                elif w_now < band_lo:
                                    target_fb = int(clamp((band_lo - w_now) / band_lo * FB_MAX_FWD, FB_MIN_STEP, FB_MAX_FWD))
                                else:
                                    target_fb = -int(clamp((w_now - band_hi) / band_hi * FB_MAX_BWD, FB_MIN_STEP, FB_MAX_BWD))
                                fb_prev_cmd += int(clamp(target_fb - fb_prev_cmd, -FB_SLEW, FB_SLEW))
                                fb = int(fb_prev_cmd)
                            else:
                                fb = 0; fb_prev_cmd = 0

                            if motion_scale < 0.999:
                                yv = scale_axis_cmd(yv, motion_scale)
                                ud = scale_axis_cmd(ud, motion_scale)
                                fb = scale_axis_cmd(fb, motion_scale)
                        else:
                            if DANGER_HOLD_ON_TRACK:
                                lr = fb = ud = yv = 0
                                fb_prev_cmd = 0
                            now_danger = time.time()
                            if DANGER_AUTO_SNAPSHOT and (now_danger - danger_snapshot_last_t) >= DANGER_SNAPSHOT_COOLDOWN_SEC:
                                save_photo(frame_disp.copy())
                                danger_snapshot_last_t = now_danger
                            if (now_danger - danger_toast_last_t) >= DANGER_TOAST_COOLDOWN_SEC:
                                toast("KIRMIZI HEDEF: takip beklemede", 1.0)
                                danger_toast_last_t = now_danger

                    else:
                        # HEDEF KAYIP
                        stop_and_hover(); lr = fb = ud = yv = 0; fb_prev_cmd = 0
                        if vision_track_lost_t is None:
                            vision_track_lost_t = time.time()
                        elapsed = time.time() - vision_track_lost_t
                        if elapsed < VISION_TRACK_LOST_SEC and safe_is_flying():
                            # Arama: saga sola yaw ile tara
                            direction = 1 if (int(elapsed * 10) // SEARCH_TOGGLE_EVERY) % 2 == 0 else -1
                            yv = int(direction * SEARCH_YAW)
                        elif elapsed >= VISION_TRACK_LOST_SEC:
                            # Kilitli hedef gecici kayipta mode dusmesin, ayni ID beklenmeye devam etsin.
                            if VISION_STICKY_LOCK and VISION_LOCK_PERSIST_ON_OCCLUSION and vision_track_target_id is not None:
                                if elapsed < VISION_REACQ_KEEP_SEC and safe_is_flying():
                                    direction = 1 if (int(elapsed * 6) // SEARCH_TOGGLE_EVERY) % 2 == 0 else -1
                                    yv = int(direction * max(8, int(SEARCH_YAW * 0.75)))
                                else:
                                    lr = fb = ud = yv = 0
                                if (time.time() - vision_track_wait_toast_t) >= 2.0:
                                    toast(f"HEDEF BEKLENIYOR ID:{vision_track_target_id}", 1.0)
                                    vision_track_wait_toast_t = time.time()
                            else:
                                toast(f"{int(VISION_TRACK_LOST_SEC)}sn gecti -> MANUEL", 2.0)
                                mode = 0; reset_tracking(); stop_and_hover()
                else:
                    mode = 0; reset_tracking()
    
            # --- MANUEL ---
            elif mode == 0 and not emergency and not auto_running and not show_running:
                axis_speed = manual_axis_speed_from_battery(battery_level)
                if key_down('w'): fb =  axis_speed
                if key_down('s'): fb = -axis_speed
                if key_down('a'): lr = -axis_speed
                if key_down('d'): lr =  axis_speed
                if key_down('q'): yv = -axis_speed
                if key_down('e'): yv =  axis_speed
                if key_down('z'): ud =  axis_speed
                if key_down('x'): ud = -axis_speed
                if mevlana_mode:
                    yv = max(24, int(MEVLANA_YAW_SPEED * get_runtime_speed_scale()))

                manual_input_active = any((lr, fb, ud, yv))
                hover_ud = compute_manual_hover_ud(now, manual_input_active)
                if (not manual_input_active) and hover_ud != 0:
                    ud = int(hover_ud)
    
            # --- RC GONDER ---
            if emergency or auto_running or show_running:
                with rc_lock: rc_lr = rc_fb = rc_ud = rc_yv = 0
            elif mode == 1:
                send_rc_control_safe(int(lr), int(fb), int(ud), int(yv), reason="track_rc")
                with rc_lock: rc_lr = rc_fb = rc_ud = rc_yv = 0
            else:
                with rc_lock:
                    rc_lr = int(lr); rc_fb = int(fb); rc_ud = int(ud); rc_yv = int(yv)
    
            # =========================================================
            # FPS
            # =========================================================
            dt_loop    = max(1e-6, time.time()-loop_start)
            fps_smooth = (1.-FPS_ALPHA)*fps_smooth + FPS_ALPHA*(1./dt_loop)
    
            # =========================================================
            # HUD
            # =========================================================
            if (now - last_battery_poll_t) >= BATTERY_POLL_SEC or battery_level < 0:
                bat_sample = safe_get_battery(-1)
                last_battery_poll_t = now
                if bat_sample != -1:
                    battery_level = bat_sample
                    note_link_ok()
                else:
                    note_link_fail("battery")
            bat = battery_level
    
            draw_cinematic_overlay(frame_disp)
            draw_vision_overlay(frame_disp, now)
    
            state_txt = "ACIL" if emergency else ("KALKIS" if takeoff_busy else ("OTONOM" if auto_running else ("GOSTERI" if show_running else "NORMAL")))
            mode_txt  = "OTONOM" if auto_running else ("GOSTERI" if show_running else ("MEVLANA" if mevlana_mode else ("VISION TAKIP" if mode==1 and vision_track_active else ("TAKIP" if mode==1 else "MANUEL"))))
            lock_txt  = f"HEDEF:ID{vision_track_target_id}" if vision_track_active else ("KILIT:ON" if lock_enabled else "KILIT:OFF")
            fb_txt    = "FB:ON"    if fb_active     else "FB:OFF"
            battery_txt = format_battery_text(bat)
            eco_label, _, eco_vision_int = get_battery_eco_profile(bat)
            stream_alive = (now - last_frame_change_t) <= (WATCHDOG_STALE_SEC + 0.15)
            video_status_txt = "VIDEO LIVE" if stream_alive else "VIDEO WAIT"
            video_status_fg = (174, 242, 255) if stream_alive else (255, 220, 140)
            if vision_enabled:
                if vision_last_target is not None:
                    target_status = str(vision_last_target.get("status_label", "SAFE")).upper()
                    target_activity = str(vision_last_target.get("activity_label", "NOT SWIMMING")).upper()
                    vision_state_txt = f"VISION:{target_status} {target_activity} K:{len(vision_last_persons)} O:{len(vision_last_objects)}"
                    vision_bg = (0, 0, 255) if bool(vision_last_target.get("is_danger", False)) else (18, 52, 24)
                else:
                    vision_state_txt = f"VISION:ON K:{len(vision_last_persons)} O:{len(vision_last_objects)}"
                    vision_bg = (20, 27, 34)
            else:
                vision_state_txt = "VISION:OFF"
                vision_bg = (34, 28, 20)
            state_color = (0, 0, 255) if emergency else (0, 175, 255) if takeoff_busy else (0, 214, 255) if auto_running else (0, 180, 118)
            accent_color = (0, 214, 255) if auto_running or takeoff_busy else (162, 235, 81)
    
            draw_reticle_modern(frame_disp, ww//2, hh//2, accent=accent_color)
            if vision_enabled and vision_last_target is not None:
                reticle_status = str(vision_last_target.get("status_label", "SAFE")).upper()
                reticle_activity = str(vision_last_target.get("activity_label", "NOT SWIMMING")).upper()
                reticle_danger = bool(vision_last_target.get("is_danger", False))
                reticle_bg = (0, 0, 180) if reticle_danger else (18, 92, 42)
                reticle_fg = (255, 246, 246) if reticle_danger else (222, 255, 232)
                draw_center_badge(
                    frame_disp,
                    f"{reticle_status} | {reticle_activity}",
                    ww // 2,
                    hh // 2 + 64,
                    bg=reticle_bg,
                    fg=reticle_fg,
                    scale=0.56,
                    pad=8,
                )
    
            draw_badge(frame_disp, state_txt, 20, 35,
                       bg=state_color,
                       scale=0.62)
            draw_badge(frame_disp, f"MOD:{mode_txt}", 20, 70, bg=(20, 27, 34))
            draw_badge(frame_disp, f"BAT:{battery_txt}", 20, 105, bg=(20, 27, 34))
            draw_badge(frame_disp, f"ECO:{eco_label} V:{eco_vision_int:.2f}s", 20, 140, bg=(20, 27, 34))
            if mode == 1 or lock_enabled or fb_active:
                draw_badge(frame_disp, f"{lock_txt} {fb_txt}", 20, 175, bg=(20, 27, 34))
            draw_badge(frame_disp, vision_state_txt, 20, 175 if not (mode == 1 or lock_enabled or fb_active) else 210, bg=vision_bg)
            draw_badge(frame_disp, f"FPS:{fps_smooth:4.1f}", ww-185, 35, bg=(20, 27, 34))
            draw_badge(frame_disp, video_status_txt, ww-205, 70, bg=(20, 27, 34), fg=video_status_fg)
    
            # Otonom adım göstergesi
            if auto_running and auto_step_label:
                draw_badge(frame_disp, f"ADIM: {auto_step_label}",
                           ww//2-145, 35, bg=(0,214,255), fg=(8,14,18), scale=0.60)
            elif show_running and show_step_label:
                draw_badge(frame_disp, f"GOSTERI: {show_step_label}",
                           ww//2-175, 35, bg=(0,214,255), fg=(8,14,18), scale=0.56)
    
            if roi_pending:
                draw_badge(frame_disp, "ROI ONAY BEKL. -> ENTER",
                           ww//2-180, hh//2-20, bg=(0,165,255), fg=(0,0,0), scale=0.65)
                if roi_pending_bbox is not None:
                    rx,ry,rw,rh = map(int, roi_pending_bbox)
                    cv2.rectangle(frame_disp,(rx,ry),(rx+rw,ry+rh),(0,165,255),2)
    
            # Vision takipte kayip gosterimi
            if vision_track_lost_t is not None and mode == 1:
                left = max(0, int(VISION_TRACK_LOST_SEC - (time.time() - vision_track_lost_t)))
                draw_badge(frame_disp, f"KAYIP: {left}s", ww-190, 105, bg=(0,255,255), fg=(0,0,0))
    
            if auto_running or auto_pose_hud.get("active", False):
                map_x = max(20, ww - 505)
                draw_route_minimap(frame_disp, AUTO_ROUTE_STEPS, auto_pose_hud, map_x, 88, size=160)
    
            rc_text = f"LR:{lr:+4d}   FB:{fb:+4d}   UD:{ud:+4d}   YV:{yv:+4d}"
            blend_rect(frame_disp, 18, hh-48, 360, hh-12, (12, 18, 24), alpha=0.58, border=(58, 86, 102))
            cv2.putText(frame_disp, rc_text, (30, hh-22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.60, (236,242,245), 2, cv2.LINE_AA)
    
            if time.time() < toast_until:
                render_toast = fit_text_width(toast_text, ww - 64, scale=0.68, thickness=2)
                tw = cv2.getTextSize(render_toast, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)[0][0]
                toast_x = 20
                toast_y = hh - 84
                blend_rect(frame_disp, toast_x, toast_y - 24, toast_x + tw + 26, toast_y + 10,
                           (6, 12, 18), alpha=0.72, border=(0, 214, 255))
                cv2.putText(frame_disp, render_toast, (toast_x + 12, toast_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0,236,255), 2, cv2.LINE_AA)

            # =========================================================
            # EKRANDA SUREKLI GORUNEN KONTROL LISTESI (sol alt)
            # =========================================================
            ctrl_lines = [
                "V:Kalkis  N:Inis  H:Hover",
                "WASD:Hareket  QE:Don  ZX:Yuksel/Alcal",
                "G/ENTER:Takip  B:Hedef Degistir  Y:Mesafe",
                "R:Otonom  O:Iptal  M:Manuel  P:Vision",
                "F:Foto  SPACE:Acil  C:Cikis  TAB:Panel",
            ]
            ctrl_y_start = hh - 56 - len(ctrl_lines) * 16
            ctrl_panel_h = len(ctrl_lines) * 16 + 12
            blend_rect(frame_disp, ww - 380, ctrl_y_start - 6, ww - 8, ctrl_y_start + ctrl_panel_h,
                       (10, 14, 18), alpha=0.55, border=(48, 68, 82))
            for ci, cline in enumerate(ctrl_lines):
                cy_pos = ctrl_y_start + 10 + ci * 16
                cv2.putText(frame_disp, cline, (ww - 374, cy_pos),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 200, 210), 1, cv2.LINE_AA)

            if vision_enabled and vision_last_target is not None:
                obj_summary = summarize_aux_objects(vision_last_objects)
                vision_info = f"TGT:{vision_last_target.get('id', '?')} {vision_last_target.get('status_label', 'SAFE')} {vision_last_target.get('activity_label', 'NOT SWIMMING')} | OBJ:{obj_summary}"
            elif vision_enabled:
                obj_summary = summarize_aux_objects(vision_last_objects)
                vision_info = f"VISION | K:{len(vision_last_persons)} O:{len(vision_last_objects)} | {vision_last_infer_ms:.0f}ms"
            elif vision_last_error:
                vision_info = f"VISION HATA: {short_vision_error(28)}"
            else:
                vision_info = "VISION OFF"
    
            ui_lines = [
                "KONTROLLER:",
                "V kalkis | N inis | H hover",
                "F foto | U kombo takla | B hedef degistir",
                "WASD QE ZX manuel",
                "7 mevlana | 8 sekiz | 9 kare",
                "G insan takip | ENTER takip",
                "Y mesafe koruma",
                "R otonom | O/O iptal",
                "P vision | M manuel",
                "SPACE acil | C cikis",
                "",
                "STATUS:",
                f"MODE {mode_txt} | BAT {battery_txt}",
                f"ECO {eco_label} | VINT {eco_vision_int:.2f}s",
                vision_info,
                f"KISI:{len(vision_last_persons)} | FWD:{AUTO_FWD_SPEED} YAW:{AUTO_YAW_SPEED}",
            ]
            if ui_panel_visible:
                make_right_panel_overlay_modern(frame_disp, ui_lines, alpha=0.18, panel_w=300)
            cv2.imshow("TELLO UI", frame_disp)
    
            dt = time.time() - loop_start
            if dt < FRAME_TIME:
                time.sleep(FRAME_TIME - dt)
    
    except SystemExit:
        pass
    except Exception as e:
        print(f"Ana dongu hatasi: {e}")
        import traceback; traceback.print_exc()
    finally:
        try: stop_vision_thread()
        except: pass
        try: rc_running = False; time.sleep(0.1)
        except: pass
        try: auto_cancel = True; stop_and_hover()
        except: pass
        try:
            if safe_is_flying() and not video_mode: run_sdk_command(tello.land)
        except: pass
        try:
            if video_mode and video_cap is not None:
                video_cap.release()
            elif not video_mode:
                run_sdk_command(tello.streamoff)
        except:
            pass
        cv2.destroyAllWindows()

