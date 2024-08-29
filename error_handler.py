import logging

class ErrorHandler:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)

    def log_error(self, error):
        self.logger.error(f"Error occurred: {str(error)}")

    def classify_errors(self, error):
        """Classify errors into categories."""
        # Enhanced error classification logic
        error_types = {
            "ValueError": "value_error",
            "TypeError": "type_error",
            "KeyError": "key_error",
            "AttributeError": "attribute_error",
            "IndexError": "index_error",
            "IOError": "io_error",
            "OSError": "os_error",
            "TimeoutError": "timeout_error",
            "ConnectionError": "connection_error",
            "generic": "generic_error"
        }
        self.logger.info(f"Classified error '{str(error)}' as '{error_types.get(type(error).__name__, 'generic_error')}'")
        error_types = {
            "ValueError": "value_error",
            "TypeError": "type_error",
            "KeyError": "key_error",
            "AttributeError": "attribute_error",
            "generic": "generic_error"
        }
        return {"error_type": error_types.get(type(error).__name__, "generic_error")}

    async def handle_error(self, ollama_interface, error):
        self.logger.info(f"Initiating error handling for: {str(error)}")
        self.log_error(error)
        self.logger.info(f"Error logged successfully: {str(error)}")
        error_type = type(error).__name__
        self.logger.info(f"Handling error of type: {error_type}")
        recovery_suggestion = await ollama_interface.handle_error(error)
        if not recovery_suggestion:
            self.logger.error("No recovery suggestion received from Ollama.")
            recovery_suggestion = self.suggest_recovery_strategy(error_type)
        
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
        
        self.logger.info(f"Final recovery suggestion: {recovery_suggestion}")
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
    def suggest_recovery_strategy(self, error_type):
        """Suggest recovery strategies based on error type."""
        self.logger.info(f"Suggesting recovery strategy for error type: {error_type}")
        strategies = {
            "value_error": "Check the input values for correctness.",
            "type_error": "Ensure the data types are compatible.",
            "key_error": "Verify the existence of the key in the data structure.",
            "attribute_error": "Check if the object has the required attribute.",
            "index_error": "Ensure the index is within the valid range.",
            "io_error": "Check file paths and permissions.",
            "os_error": "Verify system resources and permissions.",
            "timeout_error": "Consider increasing the timeout duration.",
            "connection_error": "Check network connectivity and server status.",
            "generic_error": "Review the error details and context."
        }
        strategy = strategies.get(error_type, "No specific strategy available.")
        self.logger.info(f"Suggested strategy for '{error_type}': {strategy}")
        return {"recovery_strategy": strategy}
