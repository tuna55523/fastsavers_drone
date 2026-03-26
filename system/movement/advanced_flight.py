class AdvancedFlight:

    def __init__(self, tello):
        self.tello = tello

    def takeoff(self):
        self.tello.takeoff()

    def land(self):
        self.tello.land()

    def hover(self):
        self.tello.send_rc_control(0,0,0,0)

    def manual(self, lr, fb, ud, yaw):
        self.tello.send_rc_control(lr, fb, ud, yaw)

    def follow(self, target, frame_w):

        if target is None:
            self.hover()
            return

        center = target.get("center", (frame_w // 2, 0))
        cx, _ = center
        area = target.get("area")

        error = cx - frame_w//2

        yaw = int(error * 0.25)
        yaw = max(-50,min(50,yaw))

        fb = 0

        from config import TARGET_AREA

        if area is not None:
            if area < TARGET_AREA:
                fb = 25
            elif area > TARGET_AREA * 1.3:
                fb = -20

        self.tello.send_rc_control(0, fb, 0, yaw)
