import os
import asyncio
import aiohttp
import json
from typing import Dict, Any, List, Callable, Coroutine
import logging
from chat_with_ollama import ChatGPT
from knowledge_base import KnowledgeBase
from functools import lru_cache
import time
import subprocess
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
            try:
                await self.session.close()
                self.logger.info("Client session closed successfully.")
            except Exception as e:
                self.logger.error(f"Error closing client session: {e}")
            finally:
                self.session = None
                self.logger.info("Client session set to None.")
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
        # Simplify context to include only essential elements
        essential_context = {
            "task": context.get("task"),
            "prompt_length": len(prompt),
            "system_definition": "The term 'system' refers to the project and its capabilities for complex software development assistant tasks."
        }
        
        if use_contextual_memory and "longterm_memory" not in context:
            context_memory = await self.knowledge_base.get_longterm_memory()
            summarized_memory = self.knowledge_base.summarize_memory(context_memory)
            essential_context.update({"context_memory": summarized_memory})
        
        if refine and task not in ["logging", "categorization"]:
            # Incorporate historical feedback for adaptive refinement
            historical_feedback = await self.knowledge_base.get_entry("historical_feedback") or {}
            refined_prompt = await self.adaptive_refine_prompt(prompt, task, feedback={**context.get("feedback", {}), **historical_feedback})
            if refined_prompt and isinstance(refined_prompt, str):
                prompt = refined_prompt.strip()
        
        essential_context.update({"timestamp": time.time()})
        context_str = json.dumps(essential_context, indent=2)
        prompt = f"Context: {context_str}\n\n{prompt}"
        self.logger.info(f"Querying Ollama with prompt: {prompt}")

        async def attempt_query():
            try:
                return await self.gpt.chat_with_ollama(system_prompt, prompt)
            except Exception as e:
                self.logger.error(f"Error querying Ollama: {str(e)}")
                return None

        try:
            result = await self.retry_with_backoff(attempt_query)
            if result is None:
                self.logger.error("No response received from Ollama after retries.")
                # Implement a fallback strategy
                fallback_response = {
                    "error": "No response from Ollama",
                    "suggestion": "Consider checking network connectivity or contacting support."
                }
                self.logger.info(f"Fallback response: {fallback_response}")
                return fallback_response
        except Exception as e:
            self.logger.error(f"Exception during Ollama interaction: {str(e)}")
            return {"error": "Exception during Ollama interaction", "details": str(e)}

        self.logger.debug(f"Request payload: {prompt}")

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
                
                # Attempt to clean and parse the response
                cleaned_result = result.replace('\n', '').replace('\r', '')
                try:
                    response_data = json.loads(cleaned_result)
                    self.logger.info("Successfully parsed cleaned JSON response.")
                    return response_data
                except json.JSONDecodeError:
                    self.logger.error("Failed to decode cleaned JSON response.")
                    return {"error": "Invalid JSON response after cleaning", "response": result}
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
        """Log the interaction with Ollama and save to dataset."""
        log_data = {
            "system_prompt": system_prompt,
            "prompt": prompt,
            "response": response,
            "timestamp": time.time()
        }
        log_name = f"ollama_interaction_{int(time.time())}"
        self.log_manager.save_log(log_name, log_data)

        # Append interaction to dataset file
        dataset_file = os.path.join("datasets", "ollama_interactions.jsonl")
        os.makedirs(os.path.dirname(dataset_file), exist_ok=True)
        with open(dataset_file, 'a') as file:
            json.dump(log_data, file)
            file.write('\n')

    async def retry_with_backoff(self, func: Callable[[], Coroutine], retries: int = 3, delay: int = 1) -> Any:
        """Retry a coroutine with exponential backoff."""
        for attempt in range(retries):
            try:
                return await func()
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {delay} seconds...")
                await asyncio.sleep(delay)
                delay *= 2
        self.logger.error("All retry attempts failed.")
        return None

    async def adaptive_refine_prompt(self, prompt: str, task: str, feedback: Dict[str, Any] = None) -> str:
        """Adaptively refine a given prompt based on the task and feedback."""
        refinement_prompt = f"Refine the following prompt for the task of {task}:\n\n{prompt}"
        context = {"task": task, "prompt_length": len(prompt), "feedback": feedback or {}}
        response = await self.query_ollama("adaptive_prompt_refinement", refinement_prompt, context=context, refine=False)
        refined_prompt = response.get("refined_prompt", prompt)
        if isinstance(refined_prompt, str):
            return refined_prompt.strip()
        else:
            self.logger.error("Refined prompt is not a string. Returning original prompt.")
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
        if context is None:
            context = {}
        context.update({"metrics": metrics})

        # Get reinforcement learning feedback
        prompt = f"Provide reinforcement learning feedback based on these metrics: {json.dumps(metrics, indent=2)}"
        feedback_response = await self.query_ollama("reinforcement_learning", prompt, context=context)
        feedback = feedback_response.get("feedback", [])

        # Analyze performance metrics and suggest improvements
        prompt = f"Analyze these performance metrics and suggest improvements:\n\n{json.dumps(metrics, indent=2)}"
        improvement_response = await self.query_ollama("system_improvement", prompt, context=context)
        suggestions = improvement_response.get("suggestions", [])

        # Suggest resource allocation and scaling strategies
        scaling_suggestions_response = await self.query_ollama("resource_optimization", f"Suggest resource allocation and scaling strategies based on these metrics: {metrics}", context=context)
        scaling_suggestions = scaling_suggestions_response.get("scaling_suggestions", [])

        # Combine all feedback and suggestions
        return {"feedback": feedback, "suggestions": suggestions + scaling_suggestions}

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
            # Example dynamic recovery logic: Adjust software assistant parameters or restart services
            await self.dynamic_recovery(recovery_strategy['dynamic_recovery'], error)

    async def dynamic_recovery(self, recovery_actions: Dict[str, Any], error: Exception):
        """Implement dynamic recovery actions based on the provided strategy."""
        try:
            for action in recovery_actions.get('actions', []):
                action_type = action.get('type')
                details = action.get('details', {})
                if action_type == "adjust_parameters":
                    self.logger.info(f"Adjusting parameters: {details}")
                    await self.adjust_parameters(details)
                elif action_type == "restart_service":
                    self.logger.info(f"Restarting service: {details}")
                    await self.restart_service(details)
                else:
                    self.logger.warning(f"Unknown recovery action type: {action_type}")
            self.logger.info("Dynamic recovery actions completed successfully.")
        except Exception as e:
            self.logger.error(f"Error during dynamic recovery: {str(e)}")

        recovery_strategy = recovery_actions.get('strategy', {})
        error_prompt = f"An error occurred: {str(error)}. Suggest a recovery strategy."
        context = {"error": str(error)}

        if 'retry' in recovery_strategy:
            self.logger.info("Retrying the operation as suggested by Ollama.")
            return await self.query_ollama(self.system_prompt, error_prompt, context=context)
        elif 'alternate_approach' in recovery_strategy:
            self.logger.info("Considering an alternate approach as suggested by Ollama.")
            context['alternate'] = True
            return await self.query_ollama(self.system_prompt, error_prompt, context=context)
        elif 'human_intervention' in recovery_strategy:
            self.logger.info("Requesting human intervention as suggested by Ollama.")
            self.logger.error(f"Human intervention required for error: {str(error)}")
            return {"status": "human_intervention_required"}

        return recovery_strategy

    async def adjust_parameters(self, details: Dict[str, Any]):
        """Adjust system parameters based on the provided details."""
        try:
            for param, value in details.items():
                self.logger.info(f"Setting parameter {param} to {value}")
                # Example logic to update in-memory settings
                if hasattr(self, param):
                    setattr(self, param, value)
                else:
                    self.logger.warning(f"Parameter {param} not found in the current configuration.")
        except Exception as e:
            self.logger.error(f"Error adjusting parameters: {str(e)}")

    async def restart_service(self, details: Dict[str, Any]):
        """Restart a service based on the provided details."""
        try:
            service_name = details.get("service_name")
            self.logger.info(f"Restarting service: {service_name}")
            # Implement actual service restart logic here
            # Example: Use subprocess to restart a service
            subprocess.run(["systemctl", "restart", service_name], check=True)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to restart service {service_name}: {e}")
        except Exception as e:
            self.logger.error(f"Error restarting service: {str(e)}")

