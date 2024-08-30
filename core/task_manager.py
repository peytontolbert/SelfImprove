import logging
from core.ollama_interface import OllamaInterface

logger = logging.getLogger(__name__)

from datetime import datetime
from typing import List, Dict, Any

class TaskQueue:
    def __init__(self, ollama: OllamaInterface):
        self.tasks: List[Dict[str, Any]] = []
        self.ollama = ollama

    def add_task(self, name: str, priority: int = 1, deadline: datetime = None, status: str = "Pending"):
        """Add a new task to the queue."""
        task = {
            "name": name,
            "priority": priority,
            "deadline": deadline,
            "status": status
        }
        self.tasks.append(task)
        self.tasks.sort(key=lambda x: (x['priority'], x['deadline'] or datetime.max))
        print(f"Task added: {task}")

    def update_task_status(self, name: str, status: str):
        """Update the status of a task."""
        for task in self.tasks:
            if task['name'] == name:
                task['status'] = status
                print(f"Task status updated: {task}")
                break

    def get_pending_tasks(self):
        """Get all pending tasks."""
        return [task for task in self.tasks if task['status'] == "Pending"]

    def get_overdue_tasks(self):
        """Get all overdue tasks."""
        now = datetime.now()
        return [task for task in self.tasks if task['deadline'] and task['deadline'] < now and task['status'] == "Pending"]

    async def create_project_plan(self, project_requirements):
        context = {"requirements": project_requirements}
        project_plan = await self.ollama.query_ollama("project_planning", f"Create a project plan for these requirements: {project_requirements}", context=context)
        logger.info(f"Project plan: {project_plan}")
        return project_plan

    async def assign_tasks(self, project_plan):
        for task in project_plan['tasks']:
            self.add_task(task['name'], task.get('priority', 1), task.get('deadline'), task.get('status', 'Pending'))
        logger.info("Tasks assigned to queue")

    async def generate_progress_report(self):
        completed_tasks = [task for task in self.tasks if task['status'] == "Completed"]
        pending_tasks = [task for task in self.tasks if task['status'] == "Pending"]
        context = {"completed": completed_tasks, "pending": pending_tasks}
        report = await self.ollama.query_ollama("progress_report", "Generate a progress report", context=context)
        logger.info(f"Progress report: {report}")
        return report
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
