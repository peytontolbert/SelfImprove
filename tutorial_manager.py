import os
import json

class TutorialManager:
    def __init__(self, tutorial_directory="tutorials"):
        self.tutorial_directory = tutorial_directory
        if not os.path.exists(self.tutorial_directory):
            os.makedirs(self.tutorial_directory)

    def save_tutorial(self, tutorial_name, content):
        file_path = os.path.join(self.tutorial_directory, f"{tutorial_name}.json")
        with open(file_path, 'w') as file:
            json.dump(content, file)

    def load_tutorial(self, tutorial_name):
        file_path = os.path.join(self.tutorial_directory, f"{tutorial_name}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return json.load(file)
        return None

    def list_tutorials(self):
        return [f.split('.')[0] for f in os.listdir(self.tutorial_directory) if f.endswith('.json')]
