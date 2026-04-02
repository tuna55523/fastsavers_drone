import time

from config import SEARCH_SWEEP_YAW, SEARCH_SWEEP_TOGGLE_SEC


class SearchBehavior:

    def __init__(self):
        self.direction = 1
        self.last_toggle_ts = time.time()

    def run(self, drone):
        now = time.time()
        if now - self.last_toggle_ts >= SEARCH_SWEEP_TOGGLE_SEC:
            self.direction *= -1
            self.last_toggle_ts = now

        yaw = int(SEARCH_SWEEP_YAW * self.direction)
        if hasattr(drone, "search_scan"):
            drone.search_scan(yaw)
        else:
            drone.manual(0, 0, 0, yaw)
