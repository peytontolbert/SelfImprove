import json
import os

class PromptManager:
    def __init__(self, prompt_directory="prompts/versions"):
        self.prompt_directory = prompt_directory
        if not os.path.exists(self.prompt_directory):
            os.makedirs(self.prompt_directory)

    def save_prompt(self, prompt_name, prompt_content):
        version = self.get_next_version(prompt_name)
        file_path = os.path.join(self.prompt_directory, f"{prompt_name}_v{version}.json")
        with open(file_path, 'w') as file:
            json.dump({"version": version, "content": prompt_content}, file)

    def get_next_version(self, prompt_name):
        existing_versions = [
            int(f.split('_v')[-1].split('.json')[0])
            for f in os.listdir(self.prompt_directory)
            if f.startswith(prompt_name)
        ]
        return max(existing_versions, default=0) + 1

    def load_prompt(self, prompt_name, version=None):
        if version is None:
            version = self.get_next_version(prompt_name) - 1
        file_path = os.path.join(self.prompt_directory, f"{prompt_name}_v{version}.json")
        try:
            with open(file_path, 'r') as file:
                return json.load(file)["content"]
        except FileNotFoundError:
            # Handle the case where the prompt file does not exist
            return "Default prompt content"
