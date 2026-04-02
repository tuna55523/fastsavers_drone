import time

from config import SAFETY_BATTERY_CHECK_INTERVAL_SEC


class SafetyManager:

    def __init__(self):
        self.low_battery_landed = False
        self.last_check_ts = 0.0

    def check(self, drone, battery_hint=None):
        now = time.time()
        if now - self.last_check_ts < SAFETY_BATTERY_CHECK_INTERVAL_SEC:
            return
        self.last_check_ts = now

        try:
            if battery_hint is not None:
                battery = float(battery_hint)
            elif hasattr(drone, "get_battery"):
                battery = drone.get_battery()
            else:
                battery = drone.tello.get_battery()

            if battery < 15 and not self.low_battery_landed:
                print("[SAFETY] LOW BATTERY LAND")
                drone.land()
                self.low_battery_landed = True
            elif battery >= 20:
                # Rearm when battery is healthy again (sim restart / reconnect case).
                self.low_battery_landed = False

        except:
            pass
