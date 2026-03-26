from config import RISK_ENTER_RESCUE, RISK_EXIT_RESCUE


class RescueStateMachine:

    def __init__(self):
        self.state = "SEARCH"

    def update(self, target):

        if target is None:
            self.state = "SEARCH"

        else:
            risk = target["risk"]

            if self.state == "RESCUE":
                if risk < RISK_EXIT_RESCUE:
                    self.state = "TRACK"
            else:
                if risk >= RISK_ENTER_RESCUE:
                    self.state = "RESCUE"
                else:
                    self.state = "TRACK"

        return self.state
