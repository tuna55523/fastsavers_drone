import time


class PersonMemory:

    def __init__(self):
        self.memory = {}

    def update(self, persons):

        now = time.time()

        for p in persons:

            pid = p["id"]

            if pid not in self.memory:
                self.memory[pid] = {
                    "first_seen": now,
                    "last_seen": now,
                    "positions": []
                }

            self.memory[pid]["last_seen"] = now
            self.memory[pid]["positions"].append(p["center"])

        self.cleanup()

        return self.memory

    def cleanup(self, timeout=3):

        now = time.time()

        remove_ids = []

        for pid, data in self.memory.items():
            if now - data["last_seen"] > timeout:
                remove_ids.append(pid)

        for pid in remove_ids:
            del self.memory[pid]