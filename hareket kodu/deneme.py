# pyright: reportGeneralTypeIssues=false
# pylint: disable=no-member
import os
import time
import threading
from typing import Any, Callable

import cv2  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
from djitellopy import Tello  # pyright: ignore[reportMissingImports]

# =========================================================
# AYARLAR
# =========================================================
SAVE_DIR = os.path.join(os.path.expanduser("~"), "Pictures", "Saved Pictures")
os.makedirs(SAVE_DIR, exist_ok=True)

DISPLAY_W = 1280
DISPLAY_H = 720
TARGET_FPS = 60
FRAME_TIME = 1.0 / TARGET_FPS

MAX_SPEED = 100  # manuel rc_control icin

# =========================================================
# OTONOM HIZ AYARLARI (rc_control tabanli)
# =========================================================
# Drone cm/s olarak hareket eder, biz sure ile mesafeyi kontrol ederiz
# move_forward(200) yerine: rc ile ~2.0 sn boyunca 60 cm/s ileri
AUTO_FWD_SPEED = 60   # cm/s ileri hiz (rc_control scale: 0-100)
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
AUTO_PAD_SCAN_SEC = 2.5
AUTO_DOCK_APPROACH_Z_CM = 45
AUTO_DOCK_FINAL_Z_CM = 22
TAKEOFF_READY_HEIGHT_CM = 28
TAKEOFF_WAIT_SEC = 6.0

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

try:
    from pynput import keyboard  # pyright: ignore[reportMissingModuleSource]
except Exception:
    print("pynput yok. Kur: pip install pynput")
    raise


def _key_to_str(k):
    try:
        if hasattr(k, "char") and k.char:
            return k.char.lower()
    except Exception:
        pass
    if k == keyboard.Key.space:  return "space"
    if k == keyboard.Key.enter:  return "enter"
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


def key_down(k: str) -> bool:
    if auto_running or emergency or takeoff_busy:
        return False
    return k in pressed


# =========================================================
# UI HELPERS
# =========================================================
def clamp(v, mn, mx):
    return max(mn, min(mx, v))


def clamp_bbox_to_frame(bbox, fw, fh):
    x, y, w, h = map(int, bbox)
    x = clamp(x, 0, fw - 1)
    y = clamp(y, 0, fh - 1)
    w = clamp(w, 20, fw - x)
    h = clamp(h, 20, fh - y)
    return (x, y, w, h)


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


def make_right_panel_overlay_modern(frame_bgr, lines, alpha=0.18, pad=16, panel_w=390):
    h, w = frame_bgr.shape[:2]
    x0 = max(0, w - panel_w)
    panel = frame_bgr[:, x0:w].copy()
    overlay = panel.copy()
    overlay[:] = (14, 17, 22)
    cv2.addWeighted(overlay, 0.78, panel, 0.22, 0, panel)
    cv2.line(panel, (0, 0), (0, h - 1), (0, 214, 255), 3)
    cv2.rectangle(panel, (0, 0), (panel_w - 2, h - 2), (52, 72, 82), 1)
    cv2.putText(panel, "TELLO OPS", (pad, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 248, 252), 2, cv2.LINE_AA)
    cv2.putText(panel, "flight / vision / autonomous", (pad, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (146, 172, 184), 1, cv2.LINE_AA)
    cv2.line(panel, (pad, 72), (panel_w - pad, 72), (58, 86, 102), 1)
    y = 100
    for t in lines:
        if not t:
            y += 12
            continue
        color = (235, 242, 245)
        scale = 0.58
        if t.isupper() or t.endswith(":"):
            color = (0, 214, 255)
            scale = 0.52
        cv2.putText(panel, t, (pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)
        y += 24
    frame_bgr[:, x0:w] = panel


def draw_badge(img, text, x, y, bg=(0,0,0), fg=(255,255,255), scale=0.55, pad=8):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    blend_rect(img, x, y-th-pad, x+tw+pad*2, y+pad, bg, alpha=0.82, border=(255,255,255), thickness=1)
    cv2.putText(img, text, (x+pad, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 2, cv2.LINE_AA)


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
    m = mag > (np.mean(mag) + np.std(mag)*0.5)
    if int(m.sum()) < 40: return None
    hist, _ = np.histogram(ang[m], bins=16, range=(0,360))
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
tello = Tello()
tello.connect()


def toast(msg, sec=1.5):
    global toast_text, toast_until
    toast_text  = msg
    toast_until = time.time() + sec
    print(msg)


def safe_is_flying():
    try: return bool(tello.is_flying)
    except: return False


def safe_get_height_cm(default=0.0):
    try: return float(tello.get_height())
    except: return float(default)


def stop_and_hover():
    try: tello.send_rc_control(0, 0, 0, 0)
    except: pass


sdk_command_lock = threading.Lock()


def run_sdk_command(fn):
    with sdk_command_lock:
        return fn()


def fast_takeoff(wait_sec=TAKEOFF_WAIT_SEC, min_height_cm=TAKEOFF_READY_HEIGHT_CM):
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


def force_sdk_speed_max():
    if globals().get("auto_running", False) or globals().get("takeoff_busy", False) or globals().get("emergency", False):
        return False
    for fn in [
        lambda: tello.set_speed(100),
        lambda: tello.send_command_without_return("speed 100"),
        lambda: tello.send_command_with_return("speed 100", timeout=2),
    ]:
        try: run_sdk_command(fn); return True
        except: pass
    return False


force_sdk_speed_max()
try:    bat0 = tello.get_battery()
except: bat0 = -1
print("BAT:", bat0, "%")

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

fps_smooth = 0.0
FPS_ALPHA  = 0.08
auto_pose_hud = {"active": False, "x": 0.0, "y": 0.0, "yaw": 0.0}
takeoff_busy = False

roi_pending       = False
roi_pending_bbox  = None
roi_pending_frame = None

# =========================================================
# RC THREAD
# =========================================================
rc_lr = 0; rc_fb = 0; rc_ud = 0; rc_yv = 0
rc_lock    = threading.Lock()
rc_running = True
RC_HZ = 80.0
RC_DT = 1.0 / RC_HZ


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
    for fn in [lambda: tello.land(), lambda: tello.send_command_without_return("land")]:
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
        if emergency or auto_running or mode != 0 or takeoff_busy:
            time.sleep(RC_DT)
            continue
        # Manuel mod: ana dongunun yazdigi degerleri gonder
        with rc_lock:
            lr = int(rc_lr); fb = int(rc_fb)
            ud = int(rc_ud); yv = int(rc_yv)
        try: tello.send_rc_control(lr, fb, ud, yv)
        except: note_link_fail("rc_sender")
        time.sleep(RC_DT)


def recover_stream(reason=""):
    global frame_read, last_watchdog_reset_t, last_frame_sig, last_frame_change_t
    now = time.time()
    if now - last_watchdog_reset_t < WATCHDOG_RESET_COOLDOWN: return False
    last_watchdog_reset_t = now
    toast(f"STREAM RESET ({reason})", 1.0)
    stop_and_hover()
    try: tello.streamoff(); time.sleep(0.2)
    except: pass
    try:
        tello.streamon(); time.sleep(0.35)
        frame_read = tello.get_frame_read()
        last_frame_sig = None; last_frame_change_t = time.time()
        note_link_ok(); return True
    except:
        note_link_fail(f"stream:{reason}"); return False


def init_stream():
    global frame_read, last_frame_sig, last_frame_change_t
    try:
        try: tello.streamoff()
        except: pass
        time.sleep(0.25)
        tello.streamon(); time.sleep(0.35)
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

    tracker = None; tracker_on = False
    template_gray = None; last_good_bbox = None; tm_conf = 0.0
    successful_track_frames = 0
    fb_active = False; fb_prev_cmd = 0; fb_enable_time = 0.0; dist_target_w = None
    lock_enabled = False; lock_hist = None; lock_edge = None
    hist_conf = -1.0; edge_conf = -1.0; fused_conf = 0.0; reacq_confirm = 0
    lost_since = None
    roi_pending = False; roi_pending_bbox = None; roi_pending_frame = None


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


def do_flip(direction):
    if not safe_is_flying(): toast("Takla icin once havalanin!"); return
    try:
        tello.flip(direction)
        names = {'l': 'SOL', 'r': 'SAG', 'f': 'ILERI', 'b': 'GERI'}
        toast(f"TAKLA: {names.get(direction, direction)}")
    except Exception as e: toast(f"Takla hata: {e}")


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

    def _worker():
        global takeoff_busy
        takeoff_busy = True
        try:
            toast("KALKIS BASLADI", 1.2)
            ok = fast_takeoff()
            if ok:
                toast("KALKIS")
            else:
                toast("Kalkis zaman asimi")
        except Exception as e:
            toast(f"Kalkis hata: {e}")
        finally:
            takeoff_busy = False

    threading.Thread(target=_worker, daemon=True).start()


def autonomous_worker():
    global auto_running, auto_cancel, mode, auto_step_label
    global rc_lr, rc_fb, rc_ud, rc_yv

    def should_abort():
        return auto_cancel or emergency

    def send_rc(lr=0, fb=0, ud=0, yv=0):
        try:
            tello.send_rc_control(int(lr), int(fb), int(ud), int(yv))
        except Exception:
            pass

    def move_timed(lr_spd=0, fb_spd=0, yv_spd=0, ud_spd=0, duration_sec=1.0, label=""):
        global auto_step_label
        auto_step_label = label
        toast(f"OTONOM: {label}", duration_sec + 0.5)
        t0 = time.time()
        while time.time() - t0 < duration_sec:
            if should_abort():
                send_rc(0, 0, 0, 0)
                return False
            send_rc(lr_spd, fb_spd, ud_spd, yv_spd)
            time.sleep(0.05)
        send_rc(0, 0, 0, 0)
        time.sleep(0.3)
        return True

    def normalize_heading(deg):
        return float(deg % 360.0)

    def signed_heading_delta(current_deg, target_deg):
        return ((target_deg - current_deg + 180.0) % 360.0) - 180.0

    def safe_get_height_cm(default=0.0):
        try:
            return float(tello.get_height())
        except Exception:
            return float(default)

    def safe_get_pad_id():
        try:
            return int(tello.get_mission_pad_id())
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
            set_auto_pose_hud(True, pose)
        return ok

    def hover_wait(sec=0.5):
        t0 = time.time()
        while time.time() - t0 < sec:
            if should_abort(): return False
            send_rc(0, 0, 0, 0)
            time.sleep(0.05)
        return True

    def capture_home_anchor():
        anchor = {"pad_enabled": False, "pad_id": -1}
        if not USE_MISSION_PAD_DOCKING:
            return anchor
        try:
            tello.enable_mission_pads()
            tello.set_mission_pad_detection_direction(2)
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

        for action, label, speed, duration_sec in corrections:
            if not execute_route_step(pose, action, label, speed, duration_sec):
                return False
            if not hover_wait(0.35):
                return False

        home_dist = float(np.hypot(pose["x"] - home_pose["x"], pose["y"] - home_pose["y"]))
        toast(f"OTONOM: Eve donus sapmasi {int(home_dist)}cm", 1.2)
        return home_dist <= AUTO_HOME_TOL_CM * 1.6

    def dock_to_home_anchor(home_anchor):
        global auto_step_label
        auto_step_label = "Sarj istasyonu yaklasmasi"

        if home_anchor["pad_id"] != -1:
            try:
                toast(f"OTONOM: Sarj padi m{home_anchor['pad_id']} hizalaniyor", 1.6)
                tello.go_xyz_speed_mid(0, 0, AUTO_DOCK_APPROACH_Z_CM, AUTO_DOCK_SPEED, home_anchor["pad_id"])
                if not hover_wait(1.0):
                    return False
                tello.go_xyz_speed_mid(0, 0, AUTO_DOCK_FINAL_Z_CM, AUTO_DOCK_SPEED, home_anchor["pad_id"])
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
            if not hover_wait(AUTO_STEP_HOVER_SEC): return

        # Kalkış noktasına dönüş
        if not return_to_home_local(pose, home_pose): return
        if not align_to_home_heading(pose, home_pose): return
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
        tello.land()

    except Exception as e:
        toast(f"Otonom hata: {e}", 2.0)

    finally:
        send_rc(0,0,0,0)
        set_auto_pose_hud(False)
        try:
            tello.disable_mission_pads()
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
start_keyboard_listener()
cv2.namedWindow("TELLO UI", cv2.WINDOW_NORMAL)

ok = init_stream()
if not ok: recover_stream("init fail")

t_rc = threading.Thread(target=rc_sender_loop, daemon=True)
t_rc.start()

last_speed_force_t = 0.0
SPEED_FORCE_EVERY  = 3.0

toast("Hazir | V kalkis | R otonom | G ROI | ENTER onayla | O/O iptal | SPACE acil", 3.0)

try:
    while True:
        loop_start = time.time()
        now        = time.time()

        if (not auto_running) and (not takeoff_busy) and (now - last_speed_force_t > SPEED_FORCE_EVERY):
            last_speed_force_t = now
            force_sdk_speed_max()

        raw = frame_read.frame if frame_read is not None else None
        if raw is None:
            ok_rec = recover_stream("frame none")
            if not ok_rec: note_link_fail("frame_none")
            dummy = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
            cv2.putText(dummy, "FRAME YOK...", (35, 70),
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

        frame_disp = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)
        frame_disp = cv2.resize(frame_disp, (DISPLAY_W, DISPLAY_H))
        hh, ww = frame_disp.shape[:2]

        # --- Tuş okuma ---
        pressed_once = pop_key_presses()
        kcv = cv2.waitKey(1) & 0xFF
        if kcv != 255:
            try:
                if kcv in (13, 10): pressed_once.append("enter")
                else:
                    ch = chr(kcv).lower()
                    if ch: pressed_once.append(ch)
            except: pass

        # Otonom çalışırken sadece iptal/acil tuşları
        if auto_running:
            pressed_once = [k for k in pressed_once if k in ('space','o','ö','m')]

        # --- Tuş işleme ---
        for k in pressed_once:

            if k == 'i' and not auto_running: do_flip('f')
            elif k == 'k' and not auto_running: do_flip('b')
            elif k == 'j' and not auto_running: do_flip('l')
            elif k == 'l' and not auto_running: do_flip('r')

            elif k == 'v' and not auto_running:
                start_manual_takeoff()

            elif k == 'n' and not auto_running:
                try: tello.land(); toast("INIS")
                except Exception as e: toast(f"Inis hata: {e}")

            elif k == 'h' and not auto_running:
                stop_and_hover(); toast("HOVER")

            elif k == 'm':
                emergency = False; auto_cancel = True; auto_running = False
                mode = 0; reset_tracking(); stop_and_hover()
                with rc_lock: rc_lr = rc_fb = rc_ud = rc_yv = 0
                pressed.clear(); key_events.clear()
                toast("MOD: MANUEL")

            elif k == 'g' and not auto_running:
                stop_and_hover()
                bbox = cv2.selectROI("TELLO UI", frame_disp, False)
                cv2.waitKey(1)
                if bbox[2] > 20 and bbox[3] > 20:
                    bbox = clamp_bbox_to_frame(bbox, ww, hh)
                    roi_pending = True; roi_pending_bbox = bbox
                    roi_pending_frame = frame_disp.copy()
                    toast("ROI secildi -> ENTER ile onayla", 2.5)
                else:
                    roi_pending = False; roi_pending_bbox = None; roi_pending_frame = None
                    toast("ROI kucuk, tekrar dene")

            elif k == 'enter' and not auto_running:
                if roi_pending and roi_pending_bbox is not None and roi_pending_frame is not None:
                    try:
                        tr = create_tracker()
                        tr.init(roi_pending_frame, roi_pending_bbox)
                        tracker = tr; tracker_on = True; mode = 1
                        template_gray = get_template(roi_pending_frame, roi_pending_bbox)
                        last_good_bbox = roi_pending_bbox
                        tm_conf = 0.0; successful_track_frames = 0
                        lock_enabled = False; lock_hist = None; lock_edge = None
                        reacq_confirm = 0; fb_active = False
                        dist_target_w = None; fb_prev_cmd = 0; lost_since = None
                        roi_pending = False; roi_pending_bbox = None; roi_pending_frame = None
                        toast("TAKIP BASLADI | Y: mesafe | T: kilit")
                    except Exception as e:
                        roi_pending = False; roi_pending_bbox = None; roi_pending_frame = None
                        toast(f"Tracker hata: {e}")
                else:
                    toast("ENTER: Once G ile ROI secin")

            elif k == 'y' and not auto_running:
                if tracker_on and last_good_bbox is not None:
                    _, _, w, _ = last_good_bbox
                    if w > 25:
                        dist_target_w = float(w); fb_active = True
                        fb_enable_time = time.time()+0.5; fb_prev_cmd = 0
                        toast(f"MESAFE: Ayarlandi (W={int(dist_target_w)})")
                    else: toast("Y: bbox cok kucuk")
                else: toast("Y: once G ile ROI sec")

            elif k == 't' and not auto_running:
                if tracker_on and last_good_bbox is not None:
                    lock_hist = get_hsv_hist(frame_disp, last_good_bbox)
                    lock_edge = get_edge_signature(frame_disp, last_good_bbox)
                    lock_enabled = True; reacq_confirm = 0
                    toast("KILIT: ACIK")
                else: toast("T: once G ile ROI sec")

            elif k == 'r':
                if takeoff_busy:
                    toast("Kalkis suruyor, bekleyin")
                elif not auto_running and not emergency:
                    auto_cancel = False; auto_running = True
                    try:
                        threading.Thread(target=autonomous_worker, daemon=True).start()
                        toast("OTONOM: BASLADI | O/O ile iptal edilir")
                    except Exception as e:
                        auto_running = False; toast(f"Thread hata: {e}")
                elif auto_running:
                    toast("Otonom zaten calisiyor")

            elif k in ('o', 'ö'):
                auto_cancel = True
                roi_pending = False; roi_pending_bbox = None; roi_pending_frame = None
                stop_and_hover()
                with rc_lock: rc_lr = rc_fb = rc_ud = rc_yv = 0
                pressed.clear(); key_events.clear()
                toast("OTONOM: IPTAL")

            elif k == 'space':
                emergency = True; auto_cancel = True
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

        # --- TAKİP ---
        if mode == 1 and not emergency and not auto_running:
            if tracker_on and tracker is not None:
                ok_t, bbox = False, None
                try:
                    ok_t, rb = tracker.update(frame_disp)
                    if ok_t: bbox = clamp_bbox_to_frame(rb, ww, hh)
                except: ok_t, bbox = False, None

                if ok_t and bbox is not None:
                    x, y, w, h = bbox
                    cx, cy = x+w//2, y+h//2

                    if lock_enabled and last_good_bbox is not None:
                        cur_hist  = get_hsv_hist(frame_disp, bbox)
                        cur_edge  = get_edge_signature(frame_disp, bbox)
                        hist_conf = compare_hist_corr(lock_hist, cur_hist)  if lock_hist is not None else 1.0
                        edge_conf = compare_edge_sig(lock_edge, cur_edge)   if lock_edge is not None else 1.0
                        cont_ok   = continuity_ok(bbox, last_good_bbox, ww, hh)
                        fused_conf = fuse_score(hist_conf, edge_conf, 1.0)
                        lock_bad  = (hist_conf < LOCK_HIST_STRICT or edge_conf < LOCK_EDGE_STRICT
                                     or not cont_ok or fused_conf < LOCK_FUSE_STRICT)
                        if lock_bad:
                            stop_and_hover()
                            if lost_since is None: lost_since = time.time()
                            lr = fb = ud = yv = 0; ok_t = False; bbox = None
                        else:
                            last_good_bbox = bbox; lost_since = None
                    else:
                        last_good_bbox = bbox; lost_since = None

                    if ok_t and bbox is not None:
                        cv2.rectangle(frame_disp, (x,y), (x+w,y+h), (0,255,0), 2)
                        cv2.circle(frame_disp, (cx,cy), 4, (0,255,0), -1)
                        successful_track_frames += 1
                        if successful_track_frames % TEMPLATE_UPDATE_INTERVAL == 0:
                            tpl = get_template(frame_disp, bbox)
                            if tpl is not None: template_gray = tpl

                        errx = cx - ww//2
                        if abs(errx) > DEAD_ZONE:
                            yv_raw = errx * TRACK_GAIN_YAW * (TRACK_SPEED/100.)
                            yv = int(clamp(yv_raw, -MAX_YAW, MAX_YAW))
                            if 0 < abs(yv) < MIN_CMD_YAW: yv = int(np.sign(yv)*MIN_CMD_YAW)

                        erry = cy - hh//2
                        if abs(erry) > DEAD_ZONE:
                            ud_raw = -erry * TRACK_GAIN_UD * (TRACK_SPEED/100.)
                            ud = int(clamp(ud_raw, -MAX_UD, MAX_UD))
                            if 0 < abs(ud) < MIN_CMD_UD: ud = int(np.sign(ud)*MIN_CMD_UD)

                        if fb_active and time.time() > fb_enable_time and dist_target_w is not None:
                            w_now = max(1., float(w)); tw = float(dist_target_w)
                            band_lo = tw*(1.-FB_BAND); band_hi = tw*(1.+FB_BAND)
                            if   w_now < tw*FB_MIN_W:     target_fb = +FB_MAX_FWD
                            elif w_now > tw*FB_MAX_W:     target_fb = -FB_MAX_BWD
                            elif band_lo <= w_now <= band_hi: target_fb = 0
                            elif w_now < band_lo:
                                target_fb = int(clamp((band_lo-w_now)/band_lo*FB_MAX_FWD, FB_MIN_STEP, FB_MAX_FWD))
                            else:
                                target_fb = -int(clamp((w_now-band_hi)/band_hi*FB_MAX_BWD, FB_MIN_STEP, FB_MAX_BWD))
                            fb_prev_cmd += int(clamp(target_fb-fb_prev_cmd, -FB_SLEW, FB_SLEW))
                            fb = int(fb_prev_cmd)
                        else:
                            fb = 0; fb_prev_cmd = 0

                if (not ok_t) or (bbox is None):
                    stop_and_hover(); lr = fb = ud = yv = 0
                    if lost_since is None: lost_since = time.time()
                    new_bbox, tm_conf = reacquire_by_template(frame_disp, template_gray, last_good_bbox)
                    if new_bbox is not None:
                        if not continuity_ok(new_bbox, last_good_bbox, ww, hh):
                            reacq_confirm = 0
                        else:
                            if lock_enabled:
                                ch_ = get_hsv_hist(frame_disp, new_bbox)
                                ce_ = get_edge_signature(frame_disp, new_bbox)
                                hc_ = compare_hist_corr(lock_hist, ch_) if lock_hist is not None else 1.
                                ec_ = compare_edge_sig(lock_edge, ce_)  if lock_edge is not None else 1.
                                ok_lock = hc_ >= LOCK_HIST_STRICT and ec_ >= LOCK_EDGE_STRICT and fuse_score(hc_,ec_,tm_conf) >= LOCK_FUSE_STRICT
                                reacq_confirm = (reacq_confirm+1) if ok_lock else 0
                            else:
                                reacq_confirm = min(REACQ_CONFIRM_N, reacq_confirm+1)
                        if reacq_confirm >= REACQ_CONFIRM_N:
                            try:
                                tr = create_tracker(); tr.init(frame_disp, new_bbox)
                                tracker = tr; tracker_on = True
                                last_good_bbox = new_bbox; lost_since = None; reacq_confirm = 0
                                toast("NESNE GERI BULUNDU", 1.0)
                            except: pass

                    if lost_since is not None:
                        elapsed = time.time() - lost_since
                        if elapsed < LOST_GRACE_SEC and safe_is_flying():
                            direction = 1 if (int(elapsed*10)//SEARCH_TOGGLE_EVERY)%2==0 else -1
                            yv = int(direction*SEARCH_YAW)
                        elif elapsed >= LOST_GRACE_SEC:
                            toast("10sn gecti -> MANUEL", 2.0)
                            mode = 0; reset_tracking(); stop_and_hover()
            else:
                mode = 0; reset_tracking()

        # --- MANUEL ---
        elif mode == 0 and not emergency and not auto_running:
            if key_down('w'): fb =  MAX_SPEED
            if key_down('s'): fb = -MAX_SPEED
            if key_down('a'): lr = -MAX_SPEED
            if key_down('d'): lr =  MAX_SPEED
            if key_down('q'): yv = -MAX_SPEED
            if key_down('e'): yv =  MAX_SPEED
            if key_down('z'): ud =  MAX_SPEED
            if key_down('x'): ud = -MAX_SPEED

        # --- RC GONDER ---
        if emergency or auto_running:
            with rc_lock: rc_lr = rc_fb = rc_ud = rc_yv = 0
        elif mode == 1:
            try: tello.send_rc_control(int(lr), int(fb), int(ud), int(yv))
            except: note_link_fail("track_rc")
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
        try:    bat = tello.get_battery(); note_link_ok()
        except: bat = -1; note_link_fail("battery")

        draw_cinematic_overlay(frame_disp)

        state_txt = "ACIL" if emergency else ("KALKIS" if takeoff_busy else ("OTONOM" if auto_running else "NORMAL"))
        mode_txt  = "TAKIP" if mode==1 else "MANUEL"
        lock_txt  = "KILIT:ON" if lock_enabled else "KILIT:OFF"
        fb_txt    = "FB:ON"    if fb_active     else "FB:OFF"
        state_color = (0, 0, 255) if emergency else (0, 175, 255) if takeoff_busy else (0, 214, 255) if auto_running else (0, 180, 118)
        accent_color = (0, 214, 255) if auto_running or takeoff_busy else (162, 235, 81)

        draw_reticle_modern(frame_disp, ww//2, hh//2, accent=accent_color)

        draw_badge(frame_disp, state_txt, 20, 35,
                   bg=state_color,
                   scale=0.62)
        draw_badge(frame_disp, f"MOD:{mode_txt}", 20, 70, bg=(20, 27, 34))
        draw_badge(frame_disp, f"BAT:{bat}%", 20, 105, bg=(20, 27, 34))
        draw_badge(frame_disp, f"{lock_txt} {fb_txt}", 20, 140, bg=(20, 27, 34))
        draw_badge(frame_disp, f"FPS:{fps_smooth:4.1f}", ww-185, 35, bg=(20, 27, 34))
        draw_badge(frame_disp, "VIDEO LIVE", ww-205, 70, bg=(20, 27, 34), fg=(174, 242, 255))

        # Otonom adım göstergesi
        if auto_running and auto_step_label:
            draw_badge(frame_disp, f"ADIM: {auto_step_label}",
                       ww//2-145, 35, bg=(0,214,255), fg=(8,14,18), scale=0.60)

        if roi_pending:
            draw_badge(frame_disp, "ROI ONAY BEKL. -> ENTER",
                       ww//2-180, hh//2-20, bg=(0,165,255), fg=(0,0,0), scale=0.65)
            if roi_pending_bbox is not None:
                rx,ry,rw,rh = map(int, roi_pending_bbox)
                cv2.rectangle(frame_disp,(rx,ry),(rx+rw,ry+rh),(0,165,255),2)

        if lost_since is not None and mode==1:
            left = max(0, int(LOST_GRACE_SEC-(time.time()-lost_since)))
            draw_badge(frame_disp, f"LOST: {left}s", ww-180, 70, bg=(0,255,255), fg=(0,0,0))

        map_x = max(20, ww - 610)
        draw_route_minimap(frame_disp, AUTO_ROUTE_STEPS, auto_pose_hud, map_x, 88, size=190)

        rc_text = f"LR:{lr:+4d}   FB:{fb:+4d}   UD:{ud:+4d}   YV:{yv:+4d}"
        blend_rect(frame_disp, 18, hh-48, 360, hh-12, (12, 18, 24), alpha=0.58, border=(58, 86, 102))
        cv2.putText(frame_disp, rc_text, (30, hh-22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (236,242,245), 2, cv2.LINE_AA)

        if time.time() < toast_until:
            tw = cv2.getTextSize(toast_text, cv2.FONT_HERSHEY_SIMPLEX, 0.68, 2)[0][0]
            toast_x = 20
            toast_y = hh - 84
            blend_rect(frame_disp, toast_x, toast_y - 24, toast_x + tw + 26, toast_y + 10,
                       (6, 12, 18), alpha=0.72, border=(0, 214, 255))
            cv2.putText(frame_disp, toast_text, (toast_x + 12, toast_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0,236,255), 2, cv2.LINE_AA)

        ui_lines = [
            "KONTROLLER:",
            "V: Normal kalkis   N: Inis   H: Hover",
            "M: Manuel moda don",
            "WASD/QE/ZX: Manuel kontrol",
            "G: ROI sec",
            "ENTER: ROI onayla (takip baslar)",
            "Y: Mesafe hedefi (FB)",
            "T: Kilit (baska nesneye gecmez)",
            "R: Otonom kalkis + rota + inis",
            "O/O: Otonom iptal",
            "SPACE: Acil dur | C: Cikis",
            "",
            "OTONOM ROTA:",
            "1m > sag 90 > 1m",
            "sag 90 > 1m > sag 90",
            "1m > eve don > yavas inis",
            "",
            f"FWD:{AUTO_FWD_SPEED} YAW:{AUTO_YAW_SPEED} DOCK:{AUTO_DOCK_SPEED}",
        ]
        make_right_panel_overlay_modern(frame_disp, ui_lines, alpha=0.18, panel_w=390)
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
    try: rc_running = False; time.sleep(0.1)
    except: pass
    try: auto_cancel = True; stop_and_hover()
    except: pass
    try:
        if safe_is_flying(): tello.land()
    except: pass
    try: tello.streamoff()
    except: pass
    cv2.destroyAllWindows()
