class WaterFilter:

    def __init__(self, water_line_ratio=0.55):
        # ekranın alt %55'i su kabul edilir
        self.water_line_ratio = water_line_ratio

    def filter(self, persons, frame_shape):

        height = frame_shape[0]
        water_line = int(height * self.water_line_ratio)

        water_persons = []

        for p in persons:

            _, _, _, y2 = p["bbox"]

            # bbox alt noktası suyun içinde mi?
            if y2 > water_line:
                water_persons.append(p)

        return water_persons