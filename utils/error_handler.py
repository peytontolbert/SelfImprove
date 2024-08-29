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
        
        if recovery_suggestion:
            if recovery_suggestion.get('decompose_task', False):
                subtasks = await self.decompose_task(ollama_interface, recovery_suggestion.get('original_task'))
                recovery_suggestion['subtasks'] = subtasks
            if recovery_suggestion.get('retry', False):
                self.logger.info("Retrying the operation as suggested by Ollama.")
            if recovery_suggestion.get('alternate_approach', False):
                self.logger.info("Considering an alternate approach as suggested by Ollama.")
        else:
            self.logger.error("No valid recovery suggestion received from Ollama.")
            recovery_suggestion = {"error": "No valid recovery suggestion"}
        
        self.logger.info(f"Recovery suggestion: {recovery_suggestion}")
        return recovery_suggestion

    async def decompose_task(self, ollama_interface, task):
        decomposition_prompt = f"Decompose this task into smaller, manageable subtasks: {task}"
        context = {
            "task": task,
            "error_details": "Detailed error information",
            "previous_attempts": "Number of previous attempts"
        }
        decomposition_result = await ollama_interface.query_ollama("task_decomposition", decomposition_prompt, context=context)
        subtasks = decomposition_result.get('subtasks', [])
        all_subtasks = subtasks.copy()
        
        # Continue decomposing each subtask until no further subtasks are suggested
        for subtask in subtasks:
            context = {"subtask": subtask}
            further_decomposition = await ollama_interface.query_ollama("task_decomposition", f"Further decompose this subtask: {subtask}", context=context)
            further_subtasks = further_decomposition.get('subtasks', [])
            all_subtasks.extend(further_subtasks)
            self.logger.info(f"Further decomposed subtask '{subtask}' into: {further_subtasks}")
        
        self.logger.info(f"Maximally decomposed task into subtasks: {all_subtasks}")
        return all_subtasks
