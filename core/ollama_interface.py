import asyncio
import aiohttp
import json
from typing import Dict, Any, List
import logging
from chat_with_ollama import ChatGPT
class OllamaInterface:
    def __init__(self, max_retries: int = 3):
        self.gpt = ChatGPT()
        self.max_retries = max_retries
        self.session = None
        self.conversation_history = []
        self.logger = logging.getLogger(__name__)
        self.system_prompt = "Default system prompt"  # Define a default system prompt

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()

    async def query_ollama(self, system_prompt: str, prompt: str) -> Dict[str, Any]:
        for attempt in range(self.max_retries):
            try:
                result = self.gpt.chat_with_ollama(system_prompt, prompt)
                if isinstance(result, str):
                    try:
                        return json.loads(result)
                    except json.JSONDecodeError:
                        return {"response": result}
                elif isinstance(result, dict):
                    return result
                else:
                    return {"response": str(result)}
            except Exception as e:
                self.logger.error(f"Error querying Ollama (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    raise

    async def analyze_code(self, code: str) -> Dict[str, Any]:
        prompt = f"Analyze the following code and provide suggestions for improvement:\n\n{code}"
        return await self.query_ollama(prompt, {"task": "code_analysis"})

    async def generate_code(self, spec: str) -> str:
        prompt = f"Generate code based on the following specification:\n\n{spec}"
        response = await self.query_ollama(prompt, {"task": "code_generation"})
        return response.get("code", "")

    async def handle_error(self, error: Exception) -> Dict[str, Any]:
        prompt = f"An error occurred: {str(error)}. Suggest a recovery strategy."
        return await self.query_ollama(prompt, {"task": "error_handling"})

    async def improve_system(self, performance_metrics: Dict[str, Any]) -> List[str]:
        prompt = f"Analyze these performance metrics and suggest improvements:\n\n{json.dumps(performance_metrics, indent=2)}"
        response = await self.query_ollama(prompt, {"task": "system_improvement"})
        return response.get("suggestions", [])

    async def implement_improvement(self, improvement: str) -> Dict[str, Any]:
        prompt = f"Implement this improvement: {improvement}. Consider the system's current architecture and capabilities. Provide a detailed plan for implementation."
        return await self.query_ollama(prompt, {"task": "improvement_implementation"})

    async def validate_improvement(self, improvement: str) -> Dict[str, Any]:
        prompt = f"Validate the following improvement suggestion: {improvement}. Consider potential risks, conflicts with existing system architecture, and alignment with project goals."
        return await self.query_ollama(prompt, {"task": "improvement_validation"})

    async def learn_from_experience(self, experience_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"Analyze this experience data and extract learnings: {json.dumps(experience_data)}. Focus on patterns, successful strategies, and areas for improvement."
        return await self.query_ollama(prompt, {"task": "experience_learning"})

    async def refine_prompt(self, prompt: str, task: str) -> str:
        refinement_prompt = f"Refine the following prompt for the task of {task}:\n\n{prompt}"
        response = await self.query_ollama(refinement_prompt, {"task": "prompt_refinement"})
        return response.get("refined_prompt", prompt)

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        return self.conversation_history

    async def evaluate_system_state(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"Evaluate the current system state: {json.dumps(system_state)}. Identify potential issues, bottlenecks, and areas for optimization."
        return await self.query_ollama(prompt, {"task": "system_evaluation"})

