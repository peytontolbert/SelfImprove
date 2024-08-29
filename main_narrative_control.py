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
    async def commit_changes(self, ollama, changes):
        commit_message = await ollama.query_ollama("version_control", f"Generate a commit message for these changes: {changes}")
        logger.info(f"Committing changes: {changes}")
        logger.info(f"Committed changes with message: {commit_message}")

    async def assess_codebase_readiness(self, ollama, codebase_state):
        """Assess if the current codebase is ready for production."""
        readiness_prompt = (
            f"Assess the readiness of the current codebase for production. "
            f"Consider stability, features implemented, and known issues: {codebase_state}"
        )
        readiness_assessment = await ollama.query_ollama("codebase_readiness", readiness_prompt)
        return readiness_assessment
        strategy = await ollama.query_ollama("version_control", f"Suggest a branching strategy based on the current state: {current_state}")
        return strategy

class CodeAnalysis:
    async def analyze_code(self, ollama, code):
        analysis = await ollama.query_ollama("code_analysis", f"Analyze this code and suggest improvements: {code}")
        return analysis

class TestingFramework:
    async def run_tests(self, ollama, test_cases):
        test_results = await ollama.query_ollama("testing", f"Run and analyze these test cases: {test_cases}")
        return test_results

    async def generate_tests(self, ollama, code):
        generated_tests = await ollama.query_ollama("testing", f"Generate unit tests for this code: {code}")
        return generated_tests

class DeploymentManager:
    async def deploy_code(self, ollama):
        deployment_decision = await ollama.query_ollama("deployment", "Should we deploy the current code?")
        if deployment_decision.get('deploy', False):
            logger.info("Code deployed successfully")
        else:
            logger.info("Deployment deferred based on Ollama's decision")

    async def rollback(self, ollama, version):
        rollback_plan = await ollama.query_ollama("deployment", f"Generate a rollback plan for version: {version}")
        logger.info(f"Rollback plan generated: {rollback_plan}")

class SelfImprovement:
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
        return learning

    async def get_system_metrics(self):
        response = await self.ollama.query_ollama("system_metrics", "Provide an overview of the current system capabilities and performance.")
        return response.get("metrics", {})

    async def suggest_prompt_refinements(self):
        current_prompts = await self.knowledge_base.get_entry("system_prompts")
        refinements = await self.ollama.query_ollama("prompt_refinement", f"Suggest refinements for these prompts: {current_prompts}")
        if refinements:
            await self.ollama.update_system_prompt(refinements.get("new_system_prompt", "Default system prompt"))
        return refinements

    async def retry_ollama_call(self, func, *args, max_retries=2, **kwargs):
        for attempt in range(max_retries):
            result = await func(*args, **kwargs)
            if result is not None:
                return result
            logger.warning(f"Attempt {attempt + 1} failed, retrying...")
        logger.error("All attempts failed, returning None")
        return None

async def main():
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
