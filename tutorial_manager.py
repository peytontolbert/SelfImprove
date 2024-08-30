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

    def update_tutorial(self, tutorial_name, new_content):
        """Update an existing tutorial."""
        file_path = os.path.join(self.tutorial_directory, f"{tutorial_name}.json")
        if os.path.exists(file_path):
            with open(file_path, 'w') as file:
                json.dump(new_content, file)
            return True
        return False

    def search_tutorials(self, keyword):
        """Search for tutorials containing a specific keyword."""
        matching_tutorials = []
        for tutorial_name in self.list_tutorials():
            tutorial_content = self.load_tutorial(tutorial_name)
            if keyword.lower() in json.dumps(tutorial_content).lower():
                matching_tutorials.append(tutorial_name)
        return matching_tutorials
        return [f.split('.')[0] for f in os.listdir(self.tutorial_directory) if f.endswith('.json')]
