import logging

class ErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.ERROR)

    def log_error(self, error):
        self.logger.error(f"Error occurred: {str(error)}")

    async def handle_error(self, ollama_interface, error):
        self.log_error(error)
        recovery_suggestion = await ollama_interface.handle_error(error)
        return recovery_suggestion
