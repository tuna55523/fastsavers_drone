import time


class RiskEngine:

    def __init__(self):
        self.memory = {}

    def compute(self, person):

        pid = person["id"]
        now = time.time()

        if pid not in self.memory:
            self.memory[pid] = {
                "last_seen": now,
                "risk": 0.1
            }

        mem = self.memory[pid]

        dt = now - mem["last_seen"]
        mem["last_seen"] = now

        # hareketsizlik risk arttırır
        mem["risk"] += dt * 0.05

        mem["risk"] = min(1.0, mem["risk"])

        return mem["risk"]