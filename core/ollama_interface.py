import asyncio
import aiohttp
import json
from typing import Dict, Any, List
import logging
from chat_with_ollama import ChatGPT
from knowledge_base import KnowledgeBase
from functools import lru_cache
import time
from tutorial_manager import TutorialManager
from log_manager import LogManager

class OllamaInterface:
    def __init__(self, max_retries: int = 3, knowledge_base: KnowledgeBase = None):
        self.gpt = ChatGPT()
        self.max_retries = max_retries
        self.session = None
        self.knowledge_base = knowledge_base or KnowledgeBase(ollama_interface=self)
        self.first_run = True
        self.tutorial_manager = TutorialManager()
        self.conversation_history = []
        self.logger = logging.getLogger(__name__)
        self.system_prompt = "Default software assistant prompt"
        self.prompt_cache = {}
        self.prompt_templates = {}
        self.conversation_contexts = {}
        self.log_manager = LogManager()
        self.log_manager = LogManager()

    async def __aenter__(self):
        if not self.session:
            self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session:
            await self.session.close()
            self.session = None
            self.logger.info("Client session closed successfully.")
        else:
            self.logger.warning("Attempted to close a non-existent session.")

    def simplify_context_memory(self, context_memory, max_depth=3, current_depth=0):
        """Simplify the context memory structure to avoid excessive nesting."""
        if current_depth >= max_depth:
            return "..."
        if isinstance(context_memory, dict):
            return {k: self.simplify_context_memory(v, max_depth, current_depth + 1) for k, v in context_memory.items() if v}
        return context_memory
    async def query_ollama(self, system_prompt: str, prompt: str, task: str = "general", context: Dict[str, Any] = None, refine: bool = True, use_contextual_memory: bool = True) -> Dict[str, Any]:
        context = context or {}
        if self.first_run:
            tutorial = self.tutorial_manager.load_tutorial("getting_started")
            context = {}
            if tutorial:
                self.logger.info(f"Loaded tutorial: {tutorial}")
                context.update({"tutorial": tutorial})
            self.first_run = False
        # Clarify the meaning of "system" in the context
        context.update({"system_definition": "The term 'system' refers to the project and its capabilities for complex software development assistant tasks."})
        
        if use_contextual_memory:
            if "longterm_memory" not in context:
                context_memory = await self.knowledge_base.get_longterm_memory()
                # Summarize context memory to fit within context limits
                summarized_memory = self.knowledge_base.summarize_memory(context_memory)
                context.update({"context_memory": summarized_memory})
        if refine and task not in ["logging", "categorization"]:
            refined_prompt = await self.refine_prompt(prompt, task)
            if refined_prompt and isinstance(refined_prompt, str):
                prompt = refined_prompt.strip()
        if not context:
            self.logger.warning("No specific context provided. Using default context.")
        self.logger.info(f"Querying Ollama with context: {json.dumps(context, indent=2)}")
        # Monitor software assistantperformance and log decisions
        context.update({"timestamp": time.time()})
        context_str = json.dumps(context, indent=2)
        prompt = f"Context: {context_str}\n\n{prompt}"

        async def attempt_query():
            try:
                return await self.gpt.chat_with_ollama(system_prompt, prompt)
            except Exception as e:
                self.logger.error(f"Error querying Ollama: {str(e)}")
                return None

        result = await self.retry_with_backoff(attempt_query)
        if result is None:
            self.log_interaction(system_prompt, prompt, result)
            self.log_interaction(system_prompt, prompt, result)
            self.logger.error("No response received from Ollama after retries.")
            # Implement a fallback strategy
            fallback_response = {
                "error": "No response from Ollama",
                "suggestion": "Consider checking network connectivity or contacting support."
            }
            self.logger.info(f"Fallback response: {fallback_response}")
            return fallback_response

        self.logger.debug(f"Request payload: {prompt}")

        self.log_interaction(system_prompt, prompt, result)

        self.log_interaction(system_prompt, prompt, result)

        if isinstance(result, str):
            try:
                response_data = json.loads(result)
                if not response_data:
                    self.logger.error("Empty response data received from Ollama.")
                    self.logger.debug(f"Raw response: {result}")
                    return {"error": "Empty response data"}
                return response_data
            except json.JSONDecodeError:
                self.logger.error("Failed to decode JSON response from Ollama.")
                self.logger.error(f"Invalid JSON response received: {result}")
                return {"error": "Invalid JSON response", "response": result}
        elif isinstance(result, dict):
            if not result:
                self.logger.error("Empty response data received from Ollama. This may indicate a network issue or a problem with the Ollama service.")
                self.logger.debug(f"Raw response: {result}")
                return {"error": "Empty response data"}
            self.logger.info(f"Received response from Ollama: {result}")
            return result
        else:
            self.logger.error("Unexpected response type from Ollama.")
            self.logger.debug(f"Raw response: {result}")
            return {"error": "Unexpected response type", "response": str(result)}

    def log_interaction(self, system_prompt, prompt, response):
        """Log the interaction with Ollama."""
        log_data = {
            "system_prompt": system_prompt,
            "prompt": prompt,
            "response": response,
            "timestamp": time.time()
        }
        log_name = f"ollama_interaction_{int(time.time())}"
        self.log_manager.save_log(log_name, log_data)
    def log_interaction(self, system_prompt, prompt, response):
        """Log the interaction with Ollama."""
        log_data = {
            "system_prompt": system_prompt,
            "prompt": prompt,
            "response": response,
            "timestamp": time.time()
        }
        log_name = f"ollama_interaction_{int(time.time())}"
        self.log_manager.save_log(log_name, log_data)

    async def refine_prompt(self, prompt: str, task: str) -> str:
        if task == "general":
            refinement_prompt = (
                f"Refine the following prompt for assessing alignment implications:\n\n"
                f"Assess the alignment implications of recent software assistantchanges. "
                f"Consider user behavior nuances and organizational goals."
            )
        else:
            refinement_prompt = f"Refine the following prompt for the task of {task}:\n\n{prompt}"
        # Include task-specific details in the context
        context = {"task": task, "prompt_length": len(prompt)}
        response = await self.query_ollama("prompt_refinement", refinement_prompt, context=context, refine=False)
        refined_prompt = response.get("refined_prompt", prompt)
        if isinstance(refined_prompt, str):
            return refined_prompt.strip()
        else:
            self.logger.error("Refined prompt is not a string.")
            return prompt

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
        # Include spec length for context-aware generation
        context.update({"spec": spec, "spec_length": len(spec)})
        response = await self.query_ollama("code_generation", prompt, context=context)
        return response.get("code", "") if response else ""

    async def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        prompt = f"An error occurred: {str(error)}. Suggest a recovery strategy."
        if context is None:
            context = {}
        context.update({"error": str(error)})
        response = await self.query_ollama("error_handling", prompt, context=context)
        return response if response else {"error": "No response from Ollama"}

    async def get_reinforcement_feedback(self, metrics: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        prompt = f"Provide reinforcement learning feedback based on these metrics: {json.dumps(metrics, indent=2)}"
        if context is None:
            context = {}
        context.update({"metrics": metrics})
        response = await self.query_ollama("reinforcement_learning", prompt, context=context)
        return response if response else {"feedback": []}
        prompt = f"Analyze these performance metrics and suggest improvements:\n\n{json.dumps(metrics, indent=2)}"
        if context is None:
            context = {}
        context.update({"performance_metrics": metrics})
        response = await self.query_ollama("system_improvement", prompt, context=context)
        # Suggest resource allocation and scaling strategies
        context = {"performance_metrics": metrics}
        scaling_suggestions = await self.query_ollama("resource_optimization", f"Suggest resource allocation and scaling strategies based on these metrics: {metrics}", context=context)
        return response.get("suggestions", []) + scaling_suggestions.get("scaling_suggestions", [])

    async def implement_improvement(self, improvement: str) -> Dict[str, Any]:
        prompt = f"Implement this improvement: {improvement}. Consider the software assistants current architecture and capabilities. Provide a detailed plan for implementation."
        context = {"improvement": improvement}
        return await self.query_ollama(self.system_prompt, prompt, task="improvement_implementation", context=context)

    async def validate_improvement(self, improvement: str) -> Dict[str, Any]:
        prompt = f"Validate the following improvement suggestion: {improvement}. Consider potential risks, conflicts with existing software assistantarchitecture, and alignment with project goals."
        context = {"improvement": improvement}
        return await self.query_ollama(self.system_prompt, prompt, task="improvement_validation", context=context)

    async def learn_from_experience(self, experience_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process experience data and extract learnings."""
        prompt = f"Analyze this experience data and extract learnings: {json.dumps(experience_data)}. Focus on patterns, successful strategies, and areas for improvement."
        context = {"experience_data": experience_data}
        result = await self.query_ollama(self.system_prompt, prompt, task="experience_learning", context=context)
        self.logger.info(f"Learned from experience: {result}")
        return result

    async def collaborative_learning(self, other_ai_systems: List[str], knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Learn collaboratively with other AI systems."""
        prompt = f"Collaborate with these AI systems: {other_ai_systems} to enhance learning and share knowledge: {json.dumps(knowledge)}"
        context = {"other_ai_systems": other_ai_systems, "knowledge": knowledge}
        collaboration_result = await self.query_ollama("collaborative_learning", prompt, context=context)
        self.logger.info(f"Collaborative learning result: {collaboration_result}")
        return collaboration_result

    def get_conversation_history(self) -> List[Dict[str, Any]]:
        return self.conversation_history

    async def evaluate_system_state(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        prompt = f"Evaluate the current software assistantstate: {json.dumps(system_state)}. Identify potential issues, bottlenecks, and areas for optimization."
        context = {
            "system_state": system_state,
            "recent_changes": "recent_system_changes_placeholder",
            "longterm_memory": await self.knowledge_base.get_longterm_memory()
        }
        return await self.query_ollama(prompt, {"task": "system_evaluation"}, context=context)

    async def update_system_prompt(self, new_prompt: str) -> None:
        """Update the software assistantprompt dynamically."""
        self.system_prompt = new_prompt
        if new_prompt:
            self.system_prompt = new_prompt
            self.logger.info(f"software assistantprompt updated: {new_prompt}")
            # Save the updated prompt to the knowledge base
            await self.knowledge_base.add_entry("system_prompt", {"prompt": new_prompt})
        else:
            self.logger.warning("Received empty prompt. Retaining the default software assistantprompt.")

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
        # Add error type to context for more precise handling
        context = {"error": str(error), "error_type": type(error).__name__}
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
        if 'dynamic_recovery' in recovery_strategy:
            self.logger.info("Implementing dynamic recovery strategy as suggested by Ollama.")
            # Example dynamic recovery logic: Adjust software assistantparameters or restart services
            await self.dynamic_recovery(recovery_strategy['dynamic_recovery'], error)
        
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

