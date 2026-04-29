import cv2
import os
import sys
import time
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

# FAST SPEED
MANUAL_SPEED = 75
AUTO_SPEED_BOOST = True


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

def draw_hud(frame, battery, auto_mode, op_state, target, fps):

    h, w = frame.shape[:2]

    cv2.rectangle(frame, (0, 0), (560, 245), (20, 20, 20), -1)

    mode_text = "AUTO" if auto_mode else "MANUAL"

    cv2.putText(frame, f"MODE: {mode_text}", (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    cv2.putText(frame, f"OP STATE: {op_state}", (180, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)

    cv2.putText(frame, f"BATTERY: {battery}%", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    cv2.putText(frame, f"FPS: {fps:.1f}", (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.putText(frame, f"PEOPLE: {TARGET_COUNT}", (15, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

    if target:
        alert_state = target.get("alert_state", "SAFE")
        raw_risk = target.get("raw_risk", target.get("risk", 0.0))

        if alert_state == "ALERT":
            action = ACTION_ALERT
            action_color = (0, 0, 255)
        elif alert_state == "WATCH":
            action = ACTION_WATCH
            action_color = (0, 165, 255)
        else:
            action = ACTION_SAFE
            action_color = (0, 255, 0)

        cv2.putText(frame,
                    f"TARGET ID:{target['id']} RISK:{target['risk']:.2f} RAW:{raw_risk:.2f}",
                    (15, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, action_color, 2)
        cv2.putText(frame,
                    f"ALERT:{alert_state}  ACTION:{action}",
                    (15, 180),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, action_color, 2)
    else:
        cv2.putText(frame, "TARGET: NONE  ACTION: SCAN AREA", (15, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255,255,255), 2)

    cv2.putText(
        frame,
        f"POLICY W>={RISK_WATCH_ENTER:.2f}/{RISK_WATCH_ENTER_SECONDS:.1f}s  A>={RISK_ALERT_ENTER:.2f}/{RISK_ALERT_ENTER_SECONDS:.1f}s",
        (15, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
    )
    cv2.putText(
        frame,
        f"FAST W>={RISK_FAST_WATCH:.2f}/{RISK_FAST_WATCH_SECONDS:.2f}s  A>={RISK_FAST_ALERT:.2f}/{RISK_FAST_ALERT_SECONDS:.2f}s",
        (15, 225),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
    )


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
        keys.discard('u')

    if 't' in keys:
        drone.takeoff()
        keys.discard('t')

    if 'l' in keys:
        drone.land()
        keys.discard('l')

    safety.check(drone)

    # =========================
    # AUTO MODE
    # =========================

    if AUTO_MODE:

        state = sm.update(target)
        CURRENT_OP_STATE = state

        if state == "SEARCH":
            search.run(drone)

        elif state in ["TRACK", "RESCUE"]:
            drone.auto_follow(target, w)

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

    draw_hud(frame, battery, AUTO_MODE, CURRENT_OP_STATE, target, fps)

    cv2.imshow("RESCUE DRONE", frame)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()

try:
    drone.close()
except:
    pass
