import asyncio
import aiohttp
import json
from typing import Dict, Any, List
import logging
from chat_with_ollama import ChatGPT
from functools import lru_cache
import time

class OllamaInterface:
    def __init__(self, max_retries: int = 3):
        self.gpt = ChatGPT()
        self.max_retries = max_retries
        self.session = None
        self.conversation_history = []
        self.logger = logging.getLogger(__name__)
        self.system_prompt = "Default system prompt"
        self.prompt_cache = {}
        self.prompt_templates = {}
        self.conversation_contexts = {}

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.session.close()

    async def query_ollama(self, system_prompt: str, prompt: str, task: str = "general", context: Dict[str, Any] = None, refine: bool = True) -> Dict[str, Any]:
        if refine and task not in ["logging", "categorization"]:
            refined_prompt = await self.refine_prompt(prompt, task)
            if refined_prompt and isinstance(refined_prompt, str):
                prompt = refined_prompt.strip()
        if context:
            context_str = json.dumps(context, indent=2)
            prompt = f"Context: {context_str}\n\n{prompt}"
        else:
            context = {"default": "No specific context provided"}
            context_str = json.dumps(context, indent=2)
            prompt = f"Context: {context_str}\n\n{prompt}"
            self.logger.warning("No specific context provided. Using default context.")
        for attempt in range(self.max_retries):
            try:
                result = self.gpt.chat_with_ollama(system_prompt, prompt)
                if isinstance(result, str):
                    try:
                        response_data = json.loads(result)
                        if not response_data:
                            self.logger.error("Empty response data received from Ollama.")
                            return {"error": "Empty response data"}
                        return response_data
                    except json.JSONDecodeError:
                        self.logger.error("Failed to decode JSON response from Ollama.")
                        self.logger.error(f"Invalid JSON response received: {result}")
                        return {"error": "Invalid JSON response", "response": result}
                elif isinstance(result, dict):
                    if not result:
                        self.logger.error("Empty response data received from Ollama.")
                        self.logger.error("Received empty response data from Ollama.")
                        return {"error": "Empty response data"}
                    self.logger.info(f"Received response from Ollama: {result}")
                    return result
                else:
                    self.logger.error("Unexpected response type from Ollama.")
                    return {"error": "Unexpected response type", "response": str(result)}
            except Exception as e:
                self.logger.error(f"Error querying Ollama (attempt {attempt + 1}/{self.max_retries}): {str(e)}")
                if attempt == self.max_retries - 1:
                    recovery_suggestion = await self.suggest_error_recovery(e)
                    raise Exception(f"Max retries reached. Error: {str(e)}. Recovery suggestion: {recovery_suggestion}")

    async def refine_prompt(self, prompt: str, task: str) -> str:
        if task == "general":
            refinement_prompt = (
                f"Refine the following prompt for assessing alignment implications:\n\n"
                f"Assess the alignment implications of recent system changes. "
                f"Consider user behavior nuances and organizational goals."
            )
        else:
            refinement_prompt = f"Refine the following prompt for the task of {task}:\n\n{prompt}"
        context = {"task": task}
        response = await self.query_ollama("prompt_refinement", refinement_prompt, context=context, refine=False)
        return response.get("refined_prompt", prompt).strip()

    async def analyze_code(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        prompt = f"Analyze the following code and provide suggestions for improvement:\n\n{code}"
        if context is None:
            context = {}
        context.update({"code": code})
        response = await self.query_ollama("code_analysis", prompt, context=context)
        if response:
            return response
        else:
            self.logger.error("No response from Ollama")
            return {"error": "No response from Ollama"}

    async def generate_code(self, spec: str, context: Dict[str, Any] = None) -> str:
        prompt = f"Generate code based on the following specification:\n\n{spec}"
        if context is None:
            context = {}
        context.update({"spec": spec})
        response = await self.query_ollama("code_generation", prompt, context=context)
        return response.get("code", "") if response else ""

    async def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        prompt = f"An error occurred: {str(error)}. Suggest a recovery strategy."
        if context is None:
            context = {}
        context.update({"error": str(error)})
        response = await self.query_ollama("error_handling", prompt, context=context)
        return response if response else {"error": "No response from Ollama"}

    async def improve_system(self, performance_metrics: Dict[str, Any], context: Dict[str, Any] = None) -> List[str]:
        prompt = f"Analyze these performance metrics and suggest improvements:\n\n{json.dumps(performance_metrics, indent=2)}"
        if context is None:
            context = {}
        context.update({"performance_metrics": performance_metrics})
        response = await self.query_ollama("system_improvement", prompt, context=context)
        # Suggest resource allocation and scaling strategies
        context = {"performance_metrics": performance_metrics}
        scaling_suggestions = await self.query_ollama("resource_optimization", f"Suggest resource allocation and scaling strategies based on these metrics: {performance_metrics}", context=context)
        return response.get("suggestions", []) + scaling_suggestions.get("scaling_suggestions", [])

    async def implement_improvement(self, improvement: str) -> Dict[str, Any]:
        prompt = f"Implement this improvement: {improvement}. Consider the system's current architecture and capabilities. Provide a detailed plan for implementation."
        context = {"improvement": improvement}
        return await self.query_ollama(self.system_prompt, prompt, task="improvement_implementation", context=context)

    async def validate_improvement(self, improvement: str) -> Dict[str, Any]:
        prompt = f"Validate the following improvement suggestion: {improvement}. Consider potential risks, conflicts with existing system architecture, and alignment with project goals."
        context = {"improvement": improvement}
        return await self.query_ollama(self.system_prompt, prompt, task="improvement_validation", context=context)

    async def learn_from_experience(self, experience_data: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"Analyze this experience data and extract learnings: {json.dumps(experience_data)}. Focus on patterns, successful strategies, and areas for improvement."
        context = {"experience_data": experience_data}
        return await self.query_ollama(self.system_prompt, prompt, task="experience_learning", context=context)

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        return self.conversation_history

    async def evaluate_system_state(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"Evaluate the current system state: {json.dumps(system_state)}. Identify potential issues, bottlenecks, and areas for optimization."
        context = {"system_state": system_state}
        return await self.query_ollama(prompt, {"task": "system_evaluation"}, context=context)

    async def update_system_prompt(self, new_prompt: str) -> None:
        """Update the system prompt dynamically."""
        self.system_prompt = new_prompt
        self.logger.info(f"System prompt updated: {new_prompt}")
        # Save the updated prompt to the knowledge base
        await self.knowledge_base.add_entry("system_prompt", {"prompt": new_prompt})

    async def cached_query(self, prompt: str, task: str) -> Dict[str, Any]:
        """Cache frequently used prompts to reduce API calls."""
        cache_key = f"{task}:{prompt}"
        if cache_key in self.prompt_cache:
            cached_result, timestamp = self.prompt_cache[cache_key]
            if time.time() - timestamp < 3600:  # Cache valid for 1 hour
                return cached_result
        
        result = await self.query_ollama(self.system_prompt, prompt)
        self.prompt_cache[cache_key] = (result, time.time())
        return result

    async def generate_context_aware_prompt(self, task: str, context: Dict[str, Any]) -> str:
        """Generate a prompt based on the current context and task."""
        context_str = json.dumps(context)
        prompt = f"Generate a prompt for the task '{task}' with the following context: {context_str}"
        response = await self.query_ollama(self.system_prompt, prompt)
        # Refine the prompt for better user understanding
        refined_prompt = await self.refine_prompt(response.get("generated_prompt", ""), task)
        return refined_prompt

    async def handle_multi_step_task(self, task: str, steps: List[str]) -> List[Dict[str, Any]]:
        """Handle a multi-step task by breaking it down and processing each step."""
        results = []
        for step in steps:
            step_prompt = f"Complete the following step for task '{task}': {step}"
            step_result = await self.query_ollama(self.system_prompt, step_prompt)
            results.append(step_result)
        return results

    async def suggest_error_recovery(self, error: Exception) -> str:
        """Suggest a recovery strategy for a given error."""
        error_prompt = f"Suggest a recovery strategy for the following error: {str(error)}"
        context = {"error": str(error)}
        recovery_suggestion = await self.query_ollama(self.system_prompt, error_prompt, context=context)
        return recovery_suggestion.get("recovery_strategy", "No recovery strategy suggested.")

    async def manage_prompt_template(self, template_name: str, template: str) -> None:
        """Manage prompt templates dynamically."""
        self.prompt_templates[template_name] = template
        self.logger.info(f"Prompt template '{template_name}' added/updated")
        # Generate documentation for the prompt template
        context = {"template_name": template_name}
        documentation = await self.query_ollama("documentation_generation", f"Generate documentation for the prompt template: {template_name}", context=context)
        self.logger.info(f"Documentation generated for '{template_name}': {documentation}")

    async def use_prompt_template(self, template_name: str, **kwargs) -> str:
        """Use a prompt template with given parameters."""
        if template_name not in self.prompt_templates:
            raise ValueError(f"Prompt template '{template_name}' not found")
        return self.prompt_templates[template_name].format(**kwargs)

    async def manage_conversation_context(self, context_id: str, context: Dict[str, Any]) -> None:
        """Manage conversation contexts for different tasks or users."""
        self.conversation_contexts[context_id] = context
        self.logger.info(f"Conversation context '{context_id}' added/updated")

    async def query_with_context(self, context_id: str, prompt: str) -> Dict[str, Any]:
        """Query Ollama with a specific conversation context."""
        if context_id not in self.conversation_contexts:
            raise ValueError(f"Conversation context '{context_id}' not found")
        context = self.conversation_contexts[context_id]
        context_aware_prompt = f"Given the context: {json.dumps(context)}\n\nRespond to: {prompt}"
        context = {"context_id": context_id}
        return await self.query_ollama(self.system_prompt, context_aware_prompt, context=context)

    async def handle_parallel_tasks(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Handle multiple tasks in parallel."""
        async def process_task(task):
            prompt = task.get('prompt', '')
            task_type = task.get('type', 'general')
            return await self.query_ollama(self.system_prompt, prompt)

        results = await asyncio.gather(*[process_task(task) for task in tasks])
        return results

    async def adaptive_error_handling(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Handle errors adaptively based on the error type and context."""
        error_type = type(error).__name__
        error_prompt = f"An error of type {error_type} occurred: {str(error)}. Context: {json.dumps(context)}. Suggest an adaptive recovery strategy."
        context = {"error": str(error), "context": context}
        recovery_strategy = await self.query_ollama(self.system_prompt, error_prompt, context=context)
        
        if 'retry' in recovery_strategy:
            # Implement retry logic
            self.logger.info("Retrying the operation as suggested by Ollama.")
            # Example retry logic: Call the function again
            return await self.query_ollama(self.system_prompt, error_prompt, context=context)
        elif 'alternate_approach' in recovery_strategy:
            # Implement alternate approach
            self.logger.info("Considering an alternate approach as suggested by Ollama.")
            # Example alternate approach logic: Modify the context or prompt
            context['alternate'] = True
            return await self.query_ollama(self.system_prompt, error_prompt, context=context)
        elif 'human_intervention' in recovery_strategy:
            # Request human intervention
            self.logger.info("Requesting human intervention as suggested by Ollama.")
            # Example human intervention logic: Log the error and notify a human
            self.logger.error(f"Human intervention required for error: {str(error)}")
            return {"status": "human_intervention_required"}
        
        return recovery_strategy

