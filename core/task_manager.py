import logging
from core.ollama_interface import OllamaInterface

logger = logging.getLogger(__name__)

class TaskQueue:
    def __init__(self, ollama: OllamaInterface):
        self.tasks = []
        self.ollama = ollama

    async def create_task(self, task_details):
        # Use Ollama to decide on task creation and management
        context = {"task_details": task_details}
        decision = await self.ollama.query_ollama(self.ollama.system_prompt, f"Should I create this task: {task_details}", task="task_management", context=context)
        if decision.get('create_task', False):
            # Decompose the task into subtasks
            context = {"task_details": task_details}
            subtasks = await self.ollama.query_ollama(self.ollama.system_prompt, f"Decompose this task into subtasks: {task_details}", task="task_decomposition", context=context)
            self.tasks.extend(subtasks.get('subtasks', [task_details]))
            logger.info(f"Task created with subtasks: {subtasks.get('subtasks', [task_details])}")
        else:
            logger.info(f"Task creation declined: {task_details}")

    async def manage_orchestration(self):
        if self.tasks:
            context = {"tasks": self.tasks}
            orchestration_decision = await self.ollama.query_ollama(self.ollama.system_prompt, f"How should I orchestrate these tasks: {self.tasks}", task="task_orchestration", context=context)
            # Implement orchestration logic based on Ollama's decision
            logger.info(f"Task orchestration: {orchestration_decision}")

    def is_task_completed(self, task_details):
        # Simplified check, replace with actual logic
        return task_details in self.tasks

    def remove_task(self, task_details):
        self.tasks.remove(task_details)

    def get_task_count(self):
        return len(self.tasks)
