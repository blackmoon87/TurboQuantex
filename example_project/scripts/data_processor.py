import json
import os

class DataProcessor:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_metrics(self):
        if not os.path.exists(self.data_path):
            return {"status": "error", "message": "File not found"}
            
        with open(self.data_path, 'r') as f:
            return json.load(f)

    def calculate_average(self, values):
        if not values:
            return 0.0
        return sum(values) / len(values)
