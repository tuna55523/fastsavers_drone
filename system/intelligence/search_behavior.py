class SearchBehavior:

    def __init__(self):
        self.direction = 1

    def run(self, drone):
        drone.manual(0, 0, 0, 25 * self.direction)
        self.direction *= -1