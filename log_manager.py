import json
import os
import logging

class LogManager:
    def __init__(self, log_directory="logs"):
        self.log_directory = log_directory
        if not os.path.exists(self.log_directory):
            os.makedirs(self.log_directory)
        self.logger = logging.getLogger(__name__)

    def save_log(self, log_name, log_data):
        file_path = os.path.join(self.log_directory, f"{log_name}.json")
        try:
            with open(file_path, 'w') as file:
                json.dump(log_data, file, indent=2)
            self.logger.info(f"Log saved: {file_path}")
        except Exception as e:
            self.logger.error(f"Failed to save log {log_name}: {str(e)}")
