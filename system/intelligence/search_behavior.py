import time


class SearchBehavior:

    def __init__(self, yaw_speed=25, switch_interval_sec=0.8):
        self.direction = 1
        self.yaw_speed = int(yaw_speed)
        self.switch_interval_sec = float(switch_interval_sec)
        self.last_switch_time = time.time()

    def run(self, drone):
        now = time.time()
        if now - self.last_switch_time >= self.switch_interval_sec:
            self.direction *= -1
            self.last_switch_time = now
        drone.manual(0, 0, 0, self.yaw_speed * self.direction)
