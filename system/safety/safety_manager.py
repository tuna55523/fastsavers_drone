class SafetyManager:

    def check(self, drone):

        try:
            if hasattr(drone, "get_battery"):
                battery = drone.get_battery()
            else:
                battery = drone.tello.get_battery()

            if battery < 15:
                print("[SAFETY] LOW BATTERY LAND")
                drone.land()

        except:
            pass
