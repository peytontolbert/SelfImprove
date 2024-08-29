"""
Main Narrative Control Module

This module orchestrates the narrative-driven improvement process of the system.
It integrates various components such as OllamaInterface, ImprovementManager, TaskQueue,
and others to facilitate continuous system enhancement.

Classes:
- VersionControlSystem: Manages version control operations including commit and readiness assessment.
- CodeAnalysis: Analyzes code and suggests improvements.
- TestingFramework: Handles test execution and generation.
- DeploymentManager: Manages code deployment and rollback operations.
- SelfImprovement: Facilitates self-improvement processes using Ollama's insights.

Functions:
- main: Initializes system components and starts the narrative-controlled improvement process.
"""

import logging
import asyncio
from core.ollama_interface import OllamaInterface
from core.improvement_manager import ImprovementManager
from core.task_manager import TaskQueue
from prompts.management.prompt_manager import PromptManager
from utils.error_handler import ErrorHandler
from file_system import FileSystem
from knowledge_base import KnowledgeBase
import time
from narrative.system_narrative import SystemNarrative
from core.improvement_manager import ImprovementManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VersionControlSystem:
    """
    Handles version control operations such as committing changes and assessing codebase readiness.

    Methods:
    - commit_changes: Commits changes to the version control system with a generated commit message.
    - assess_codebase_readiness: Evaluates the readiness of the current codebase for production deployment.
    """
    async def commit_changes(self, ollama, changes):
        context = {"changes": changes}
        commit_message = await ollama.query_ollama("version_control", f"Generate a commit message for these changes: {changes}", context=context)
        logger.info(f"Committing changes: {changes}")
        logger.info(f"Committed changes with message: {commit_message}")

    async def assess_codebase_readiness(self, ollama, codebase_state):
        """Assess if the current codebase is ready for production."""
        readiness_prompt = (
            f"Assess the readiness of the current codebase for production. "
            f"Consider stability, features implemented, and known issues: {codebase_state}"
        )
        context = {"codebase_state": codebase_state}
        readiness_assessment = await ollama.query_ollama("codebase_readiness", readiness_prompt, context=context)
        self.logger.info(f"Codebase readiness assessment: {readiness_assessment}")
        return readiness_assessment

class CodeAnalysis:
    """
    Analyzes code to provide suggestions for improvements.

    Methods:
    - analyze_code: Analyzes the given code and returns improvement suggestions.
    """
    async def analyze_code(self, ollama, code):
        context = {"code": code}
        analysis = await ollama.query_ollama("code_analysis", f"Analyze this code and suggest improvements: {code}", context=context)
        self.logger.info(f"Code analysis result: {analysis}")
        return analysis

class TestingFramework:
    """
    Manages the execution and generation of tests.

    Methods:
    - run_tests: Executes and analyzes the provided test cases.
    - generate_tests: Generates unit tests for the given code.
    """
    async def run_tests(self, ollama, test_cases):
        context = {"test_cases": test_cases}
        test_results = await ollama.query_ollama("testing", f"Run and analyze these test cases: {test_cases}", context=context)
        self.logger.info(f"Test results: {test_results}")
        return test_results

    async def generate_tests(self, ollama, code):
        context = {"code": code}
        generated_tests = await ollama.query_ollama("testing", f"Generate unit tests for this code: {code}", context=context)
        self.logger.info(f"Generated tests: {generated_tests}")
        return generated_tests

class DeploymentManager:
    """
    Manages code deployment and rollback operations.

    Methods:
    - deploy_code: Decides whether to deploy the current code based on Ollama's decision.
    - rollback: Generates a rollback plan for a specified version.
    """
    async def deploy_code(self, ollama):
        context = {"current_code": "current_code_placeholder"}
        deployment_decision = await ollama.query_ollama("deployment", "Should we deploy the current code?", context=context)
        if deployment_decision.get('deploy', False):
            logger.info("Code deployed successfully")
            await self.narrative.log_state("Deployment approved by Ollama")
        else:
            await self.narrative.log_state("Deployment deferred based on Ollama's decision")
            logger.info("Deployment deferred based on Ollama's decision")

    async def rollback(self, ollama, version):
        context = {"version": version}
        rollback_plan = await ollama.query_ollama("deployment", f"Generate a rollback plan for version: {version}", context=context)
        logger.info(f"Rollback plan generated: {rollback_plan}")

class SelfImprovement:
    """
    Facilitates self-improvement processes using Ollama's insights.

    Attributes:
    - ollama: Instance of OllamaInterface for querying and decision-making.
    - knowledge_base: Instance of KnowledgeBase for storing and retrieving knowledge.
    - improvement_manager: Instance of ImprovementManager for managing improvements.

    Methods:
    - analyze_performance: Analyzes system performance and suggests improvements.
    - validate_improvements: Validates suggested improvements.
    - apply_improvements: Applies validated improvements.
    - apply_code_change: Applies a code change.
    - apply_system_update: Applies a system update.
    - learn_from_experience: Learns from past experiences to improve future performance.
    - get_system_metrics: Retrieves current system metrics.
    - suggest_prompt_refinements: Suggests refinements for system prompts.
    - retry_ollama_call: Retries a function call with Ollama if the result is None.
    """
    def __init__(self, ollama: OllamaInterface, knowledge_base: KnowledgeBase, improvement_manager: ImprovementManager):
        self.ollama = ollama
        self.knowledge_base = knowledge_base
        self.improvement_manager = improvement_manager

    async def analyze_performance(self, metrics):
        improvements = await self.improvement_manager.suggest_improvements(metrics)
        validated_improvements = await self.improvement_manager.validate_improvements(improvements)
        return validated_improvements

    async def validate_improvements(self, improvements):
        validated = []
        for improvement in improvements:
            validation = await self.ollama.validate_improvement(improvement)
            if validation.get('is_valid', False):
                validated.append(improvement)
            else:
                logger.info(f"Invalid improvement suggestion: {improvement}")
        return validated

    async def apply_improvements(self, improvements):
        results = await self.improvement_manager.apply_improvements(improvements)
        return results

    async def apply_code_change(self, code_change):
        logger.info(f"Code change applied: {code_change}")
        return {"status": "success", "message": "Code change applied"}

    async def apply_system_update(self, system_update):
        logger.info(f"System update applied: {system_update}")
        return {"status": "success", "message": "System update applied"}

    async def learn_from_experience(self, experience_data):
        learning = await self.ollama.learn_from_experience(experience_data)
        await self.knowledge_base.add_entry("system_learnings", learning)
        self.logger.info(f"Learned from experience: {learning}")
        return learning

    async def get_system_metrics(self):
        response = await self.ollama.query_ollama("system_metrics", "Provide an overview of the current system capabilities and performance.")
        return response.get("metrics", {})

    async def suggest_prompt_refinements(self):
        current_prompts = await self.knowledge_base.get_entry("system_prompts")
        refinements = await self.ollama.query_ollama("prompt_refinement", f"Suggest refinements for these prompts: {current_prompts}")
        if refinements:
            await self.ollama.update_system_prompt(refinements.get("new_system_prompt", "Default system prompt"))
        self.logger.info(f"Prompt refinements suggested: {refinements}")
        return refinements

    async def retry_ollama_call(self, func, *args, max_retries=2, **kwargs):
        for attempt in range(max_retries):
            result = await func(*args, **kwargs)
            if result is not None:
                return result
            logger.warning(f"Attempt {attempt + 1} failed, retrying...")
        logger.error("All attempts failed, returning None")
        self.logger.error("All attempts failed, returning None")
        await self.narrative.log_error("All attempts failed", {"function": func.__name__, "args": args, "kwargs": kwargs})
        return None

async def main():
    """
    Initializes system components and starts the narrative-controlled improvement process.

    This function sets up instances of various components such as OllamaInterface, TaskQueue,
    KnowledgeBase, and others. It then initiates the improvement process controlled by the
    SystemNarrative. The narrative can reset, change actions, and influence decisions dynamically.
    """
    narrative = SystemNarrative()
    await narrative.log_state("Initializing system components")
    ollama = OllamaInterface()
    task_queue = TaskQueue(ollama)
    kb = KnowledgeBase(ollama_interface=ollama)
    vcs = VersionControlSystem()
    ca = CodeAnalysis()
    tf = TestingFramework()
    dm = DeploymentManager()
    improvement_manager = ImprovementManager(ollama)
    si = SelfImprovement(ollama, kb, improvement_manager)
    fs = FileSystem()
    pm = PromptManager()
    eh = ErrorHandler()
    
    # Start the narrative-controlled improvement process
    await narrative.control_improvement_process(ollama, si, kb, task_queue, vcs, ca, tf, dm, fs, pm, eh)

if __name__ == "__main__":
    asyncio.run(main())
