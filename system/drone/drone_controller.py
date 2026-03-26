from djitellopy import Tello
from system.movement.advanced_flight import AdvancedFlight


class DroneController:

    def __init__(self):
        self.frame_is_rgb = True

        print("[INFO] Connecting Drone")

        self.tello = Tello()
        self.tello.connect()

        print("Battery:", self.tello.get_battery())

        self.tello.streamoff()
        self.tello.streamon()

        self.flight = AdvancedFlight(self.tello)

    def frame(self):
        return self.tello.get_frame_read().frame

    def get_battery(self):
        return self.tello.get_battery()

    def takeoff(self):
        self.flight.takeoff()

    def land(self):
        self.flight.land()

    def manual(self, lr, fb, ud, yaw):
        self.flight.manual(lr, fb, ud, yaw)

    def hover(self):
        self.flight.hover()

    def auto_follow(self, target, w):
        self.flight.follow(target, w)

    def close(self):
        try:
            self.tello.send_rc_control(0, 0, 0, 0)
        except:
            pass
