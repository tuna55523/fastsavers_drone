from config import (
    RISK_ENTER_RESCUE,
    RISK_EXIT_RESCUE,
    RISK_RESCUE_ALERT_ENTER,
    RISK_RESCUE_WATCH_ENTER,
    RISK_RESCUE_RISE_RATE_TRIG,
    RISK_RESCUE_RISE_DROP,
    RISK_RESCUE_ACUTE_TRIG,
    RISK_RESCUE_ACUTE_DROP,
)


class RescueStateMachine:

    def __init__(self):
        self.state = "SEARCH"

    def update(self, target):

        if target is None:
            self.state = "SEARCH"

        else:
            risk = float(target.get("risk", 0.0))
            alert_state = str(target.get("alert_state", "SAFE"))
            acute_distress = float(target.get("acute_distress", 0.0))
            risk_rise_rate = float(target.get("risk_rise_rate", 0.0))

            rescue_enter = float(RISK_ENTER_RESCUE)

            if alert_state == "ALERT":
                rescue_enter = min(rescue_enter, float(RISK_RESCUE_ALERT_ENTER))
            elif alert_state == "WATCH":
                rescue_enter = min(rescue_enter, float(RISK_RESCUE_WATCH_ENTER))

            if risk_rise_rate >= float(RISK_RESCUE_RISE_RATE_TRIG):
                rescue_enter -= float(RISK_RESCUE_RISE_DROP)

            if acute_distress >= float(RISK_RESCUE_ACUTE_TRIG):
                rescue_enter -= float(RISK_RESCUE_ACUTE_DROP)

            rescue_enter = max(float(RISK_EXIT_RESCUE) + 0.03, rescue_enter)

            if self.state == "RESCUE":
                if risk < float(RISK_EXIT_RESCUE) and alert_state != "ALERT":
                    self.state = "TRACK"
            else:
                if alert_state == "ALERT" or risk >= rescue_enter:
                    self.state = "RESCUE"
                else:
                    self.state = "TRACK"

        return self.state
