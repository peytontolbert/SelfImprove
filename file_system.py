import os

class FileSystem:
    def __init__(self, base_directory="system_data"):
        self.base_directory = base_directory
        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory)

    def create_file(self, filename, data=""):
        """Create a new file with optional initial data."""
        file_path = os.path.join(self.base_directory, filename)
        with open(file_path, 'w') as file:
            file.write(data)
        return file_path

    def write_to_file(self, filename, data):
        file_path = os.path.join(self.base_directory, filename)
        with open(file_path, 'w') as file:
            file.write(data)

    def read_from_file(self, filename):
        file_path = os.path.join(self.base_directory, filename)
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                return file.read()
        else:
            return None

    def edit_file(self, filename, new_data):
        """Edit an existing file with new data."""
        file_path = os.path.join(self.base_directory, filename)
        if os.path.exists(file_path):
            with open(file_path, 'w') as file:
                file.write(new_data)
            return file_path
        else:
            raise FileNotFoundError(f"File {filename} does not exist.")

    def delete_file(self, filename):
        """Delete a file."""
        file_path = os.path.join(self.base_directory, filename)
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        else:
            raise FileNotFoundError(f"File {filename} does not exist.")

    def list_files(self):
        return os.listdir(self.base_directory)

    def list_directories(self):
        return [d for d in os.listdir(self.base_directory) if os.path.isdir(os.path.join(self.base_directory, d))]
