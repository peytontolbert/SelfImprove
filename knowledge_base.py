import os
import json

class KnowledgeBase:
    def __init__(self, base_directory="knowledge_base"):
        self.base_directory = base_directory
        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory)

    def add_entry(self, entry_name, data):
        file_path = os.path.join(self.base_directory, f"{entry_name}.json")
        with open(file_path, 'w') as file:
            json.dump(data, file)

    def get_entry(self, entry_name):
        file_path = os.path.join(self.base_directory, f"{entry_name}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return json.load(file)
        else:
            return None

    def update_entry(self, entry_name, data):
        self.add_entry(entry_name, data)

    def list_entries(self):
        return [f.split('.')[0] for f in os.listdir(self.base_directory) if f.endswith('.json')]
