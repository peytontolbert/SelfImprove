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
                result = self.gpt.chat_with_ollama(f"{system_prompt}\n\n{prompt}")
                return result
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

    async def refine_prompt(self, prompt: str, task: str) -> str:
        refinement_prompt = f"Refine the following prompt for the task of {task}:\n\n{prompt}"
        response = await self.query_ollama(refinement_prompt, {"task": "prompt_refinement"})
        return response.get("refined_prompt", prompt)

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        return self.conversation_history

