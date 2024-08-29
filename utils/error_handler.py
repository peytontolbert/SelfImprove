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
        
        if recovery_suggestion.get('decompose_task', False):
            subtasks = await self.decompose_task(ollama_interface, recovery_suggestion.get('original_task'))
            recovery_suggestion['subtasks'] = subtasks
        
        self.logger.info(f"Recovery suggestion: {recovery_suggestion}")
        return recovery_suggestion

    async def decompose_task(self, ollama_interface, task):
        decomposition_prompt = f"Decompose this task into smaller, manageable subtasks: {task}"
        decomposition_result = await ollama_interface.query_ollama("task_decomposition", decomposition_prompt)
        subtasks = decomposition_result.get('subtasks', [])
        self.logger.info(f"Decomposed task into subtasks: {subtasks}")
        return subtasks
