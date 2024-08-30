import logging
from core.ollama_interface import OllamaInterface

logger = logging.getLogger(__name__)

import asyncio
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

    async def evaluate_tasks(self):
        """Evaluate tasks based on historical data and feedback."""
        for task in self.tasks:
            feedback = await self.ollama.query_ollama("task_evaluation", f"Evaluate the task: {task['name']}", context={"task": task})
            task["evaluation"] = feedback.get("evaluation", "No evaluation available")
            logger.info(f"Task evaluation for {task['name']}: {task['evaluation']}")

    async def refine_tasks(self):
        """Refine tasks based on feedback loops."""
        for task in self.tasks:
            refinement = await self.ollama.query_ollama("task_refinement", f"Refine the task: {task['name']}", context={"task": task})
            task["refinement"] = refinement.get("refinement", "No refinement available")
            logger.info(f"Task refinement for {task['name']}: {task['refinement']}")

    async def adaptive_task_learning(self, project_requirements):
        """Implement adaptive learning to adjust task strategies."""
        for task in self.tasks:
            learning = await self.ollama.query_ollama("adaptive_task_learning", f"Adapt task strategy for: {task['name']}", context={"task": task})
            task["strategy"] = learning.get("strategy", "No strategy available")
            logger.info(f"Adaptive learning for task {task['name']}: {task['strategy']}")
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
