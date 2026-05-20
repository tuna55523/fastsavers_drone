import cv2
import os
import sys
import time
import threading
from pynput import keyboard

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from system.drone.drone_controller import DroneController
from system.drone.sim_drone_controller import SimDroneController
from system.vision.detect_track import DetectTrackSystem
from system.intelligence.state_machine import RescueStateMachine
from system.intelligence.search_behavior import SearchBehavior
from system.safety.safety_manager import SafetyManager
from config import (
    RUN_MODE,
    SIM_VIDEO_PATH,
    SIM_LOOP_VIDEO,
    SIM_COMMAND_LOG_PATH,
    FRAME_WIDTH,
    FRAME_HEIGHT,
    MANUAL_SPEED,
    ACTION_SAFE,
    ACTION_WATCH,
    ACTION_ALERT,
    RISK_WATCH_ENTER,
    RISK_ALERT_ENTER,
    RISK_WATCH_ENTER_SECONDS,
    RISK_ALERT_ENTER_SECONDS,
    RISK_FAST_WATCH,
    RISK_FAST_ALERT,
    RISK_FAST_WATCH_SECONDS,
    RISK_FAST_ALERT_SECONDS,
    BATTERY_POLL_INTERVAL_SEC,
)


# ======================================================
# INIT
# ======================================================

if RUN_MODE.lower() == "sim":
    print("[MODE] SIMULATION")
    drone = SimDroneController(
        video_path=SIM_VIDEO_PATH,
        loop_video=SIM_LOOP_VIDEO,
        command_log_path=SIM_COMMAND_LOG_PATH
    )
else:
    print("[MODE] REAL DRONE")
    drone = DroneController()

vision = DetectTrackSystem()

sm = RescueStateMachine()
search = SearchBehavior()
safety = SafetyManager()

keys = set()

AUTO_MODE = False
TARGET_COUNT = 0
TAKEOFF_BUSY = False
LAND_BUSY = False


# ======================================================
# KEYBOARD
# ======================================================

def on_press(key):
    try:
        keys.add(key.char.lower())
    except:
        pass


def on_release(key):
    try:
        keys.remove(key.char.lower())
    except:
        pass


keyboard.Listener(on_press=on_press, on_release=on_release).start()


# ======================================================
# FPS
# ======================================================

prev_time = time.time()
fps = 0
CURRENT_OP_STATE = "SEARCH"
battery = -1
last_battery_poll = 0.0


# ======================================================
# HUD
# ======================================================

def _blend_rect(img, x1, y1, x2, y2, color=(18, 22, 28), alpha=0.35, border=(80, 100, 115)):
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(img.shape[1] - 1, int(x2))
    y2 = min(img.shape[0] - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), border, 1)


def _draw_text_lines(frame, x, y, lines, title_color=(0, 214, 255)):
    line_h = 19
    for idx, line in enumerate(lines):
        if not line:
            y += 8
            continue
        if idx == 0:
            color = title_color
            scale = 0.57
            thick = 2
        else:
            color = (235, 240, 245)
            scale = 0.46
            thick = 1
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        y += line_h


def draw_hud(frame, battery, auto_mode, op_state, target, fps, takeoff_busy, land_busy):

    h, w = frame.shape[:2]
    left_w = min(320, max(240, int(w * 0.28)))
    right_w = min(360, max(250, int(w * 0.31)))
    panel_h = min(245, max(185, int(h * 0.36)))

    _blend_rect(frame, 10, 10, 10 + left_w, 10 + panel_h, alpha=0.33)
    _blend_rect(frame, w - right_w - 10, 10, w - 10, 10 + panel_h, alpha=0.33)

    mode_text = "AUTO" if auto_mode else "MANUAL"
    flight_text = "TAKEOFF" if takeoff_busy else ("LANDING" if land_busy else "READY")
    people_text = f"PEOPLE: {TARGET_COUNT}"

    if target:
        alert_state = target.get("alert_state", "SAFE")
        raw_risk = target.get("raw_risk", target.get("risk", 0.0))
        risk_rise = target.get("risk_rise_rate", 0.0)
        action = ACTION_SAFE
        if alert_state == "ALERT":
            action = ACTION_ALERT
        elif alert_state == "WATCH":
            action = ACTION_WATCH
        target_line_1 = f"TARGET ID:{target['id']} R:{target['risk']:.2f} RAW:{raw_risk:.2f} RR:{risk_rise:.2f}/s"
        target_line_2 = f"ALERT:{alert_state} ACTION:{action}"
    else:
        target_line_1 = "TARGET: NONE"
        target_line_2 = "ACTION: SCAN AREA"

    left_lines = [
        "VISION / MISSION",
        f"MODE: {mode_text}",
        f"OP: {op_state}",
        f"FLIGHT: {flight_text}",
        f"BATTERY: {battery}%",
        f"FPS: {fps:.1f}",
        people_text,
        "",
        target_line_1,
        target_line_2,
        "",
        f"W>={RISK_WATCH_ENTER:.2f}/{RISK_WATCH_ENTER_SECONDS:.1f}s  A>={RISK_ALERT_ENTER:.2f}/{RISK_ALERT_ENTER_SECONDS:.1f}s",
        f"FAST W>={RISK_FAST_WATCH:.2f}/{RISK_FAST_WATCH_SECONDS:.2f}s  A>={RISK_FAST_ALERT:.2f}/{RISK_FAST_ALERT_SECONDS:.2f}s",
    ]
    _draw_text_lines(frame, 22, 34, left_lines)

    right_lines = [
        "MOTION / CONTROLS",
        "U: AUTO / MANUAL",
        "T: TAKEOFF",
        "L: LAND",
        "",
        "W/S: FORWARD/BACK",
        "A/D: LEFT/RIGHT",
        "Q/E: YAW LEFT/RIGHT",
        "C: UP",
        "Z or X: DOWN",
        "",
        "AUTO SEARCH: smooth yaw sweep",
        "AUTO TRACK: smoothed follow",
        "ESC: EXIT",
    ]
    _draw_text_lines(frame, w - right_w + 6, 34, right_lines)


def _start_takeoff_async(drone):
    if TAKEOFF_BUSY or LAND_BUSY:
        return

    def _worker():
        global TAKEOFF_BUSY
        TAKEOFF_BUSY = True
        try:
            drone.takeoff()
            print("[FLIGHT] TAKEOFF OK")
        except Exception as e:
            print(f"[FLIGHT] TAKEOFF ERROR: {e}")
        finally:
            TAKEOFF_BUSY = False

    threading.Thread(target=_worker, daemon=True).start()


def _start_land_async(drone):
    if TAKEOFF_BUSY or LAND_BUSY:
        return

    def _worker():
        global LAND_BUSY
        LAND_BUSY = True
        try:
            drone.land()
            print("[FLIGHT] LAND OK")
        except Exception as e:
            print(f"[FLIGHT] LAND ERROR: {e}")
        finally:
            LAND_BUSY = False

    threading.Thread(target=_worker, daemon=True).start()


# ======================================================
# MAIN LOOP
# ======================================================

while True:

    try:
        frame = drone.frame()
    except Exception as e:
        print(f"[INFO] Stopping: {e}")
        break

    if getattr(drone, "frame_is_rgb", False):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if RUN_MODE.lower() == "sim":
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_LINEAR)

    frame_ts = time.time()
    if RUN_MODE.lower() == "sim" and hasattr(drone, "get_frame_timestamp"):
        frame_ts = drone.get_frame_timestamp()
    vision_result = vision.run(frame, frame_ts=frame_ts)
    if isinstance(vision_result, (tuple, list)) and len(vision_result) >= 4:
        frame, target, persons, _ = vision_result[:4]
    else:
        frame, target, persons = vision_result

    TARGET_COUNT = len(persons)

    h, w = frame.shape[:2]

    # =========================
    # TOGGLES
    # =========================

    if 'u' in keys:
        AUTO_MODE = not AUTO_MODE
        print("AUTO MODE:", AUTO_MODE)
        if not AUTO_MODE:
            try:
                drone.hover()
            except:
                pass
        keys.discard('u')

    if 't' in keys:
        _start_takeoff_async(drone)
        keys.discard('t')

    if 'l' in keys:
        _start_land_async(drone)
        keys.discard('l')

    # =========================
    # AUTO MODE
    # =========================

    if TAKEOFF_BUSY:
        CURRENT_OP_STATE = "TAKEOFF"
        try:
            drone.hover()
        except:
            pass

    elif LAND_BUSY:
        CURRENT_OP_STATE = "LANDING"
        try:
            drone.hover()
        except:
            pass

    elif AUTO_MODE:

        state = sm.update(target)
        CURRENT_OP_STATE = state

        if state == "SEARCH":
            search.run(drone)

        elif state in ["TRACK", "RESCUE"]:
            drone.auto_follow(target, w, op_state=state)

    # =========================
    # MANUAL MODE
    # =========================

    else:
        CURRENT_OP_STATE = "MANUAL"

        lr = fb = ud = yaw = 0
        speed = MANUAL_SPEED

        if 'w' in keys: fb = speed
        if 's' in keys: fb = -speed
        if 'a' in keys: lr = -speed
        if 'd' in keys: lr = speed
        if 'q' in keys: yaw = -speed
        if 'e' in keys: yaw = speed
        if 'c' in keys: ud = speed
        if 'z' in keys: ud = -speed
        if 'x' in keys: ud = -speed

        drone.manual(lr, fb, ud, yaw)

    # =========================
    # FPS
    # =========================

    now = time.time()
    fps = 1 / (now - prev_time)
    prev_time = now

    if now - last_battery_poll >= BATTERY_POLL_INTERVAL_SEC:
        try:
            battery = drone.get_battery()
        except:
            battery = -1
        last_battery_poll = now

    safety.check(drone, battery_hint=battery)

    draw_hud(frame, battery, AUTO_MODE, CURRENT_OP_STATE, target, fps, TAKEOFF_BUSY, LAND_BUSY)

    cv2.imshow("RESCUE DRONE", frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

try:
    drone.close()
except:
    pass
