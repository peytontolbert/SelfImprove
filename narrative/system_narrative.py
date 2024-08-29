import logging
from knowledge_base import KnowledgeBase
from core.ollama_interface import OllamaInterface
import asyncio
import time
import subprocess

class SystemNarrative:
    def __init__(self, ollama_interface=None, knowledge_base=None):
        self.logger = logging.getLogger("SystemNarrative")
        self.ollama = ollama_interface or OllamaInterface()
        self.knowledge_base = knowledge_base or KnowledgeBase()
        logging.basicConfig(level=logging.INFO)

    async def generate_thoughts(self, context=None):
        """Generate detailed thoughts or insights about the current state and tasks."""
        longterm_memory = await self.knowledge_base.get_longterm_memory()
        prompt = "Generate detailed thoughts about the current system state, tasks, and potential improvements."
        if context:
            prompt += f" | Context: {context}"
        if longterm_memory:
            prompt += f" | Long-term Memory: {longterm_memory}"
        ollama_response = await self.ollama.query_ollama("thought_generation", prompt)
        thoughts = ollama_response.get('thoughts', 'No thoughts generated')
        self.logger.info(f"Ollama Detailed Thoughts: {thoughts}")
        return thoughts

    async def log_state(self, message, context=None):
        if context:
            self.logger.info(f"System State: {message} | Context: {context}")
        else:
            self.logger.info(f"System State: {message}")
        await self.log_with_ollama(message, context)
        # Generate and log thoughts about the current state
        await self.generate_thoughts(context)

    async def log_decision(self, decision, rationale=None):
        """Log decisions with detailed rationale."""
        if rationale:
            self.logger.info(f"System Decision: {decision} | Detailed Rationale: {rationale}")
        else:
            self.logger.info(f"System Decision: {decision}")
        await self.log_with_ollama(decision, rationale)
        # Generate and log thoughts about the decision
        await self.generate_thoughts({"decision": decision, "rationale": rationale})

    async def log_error(self, error, context=None):
        """Log errors with context and recovery strategies."""
        if context:
            self.logger.error(f"System Error: {error} | Context: {context}")
        else:
            self.logger.error(f"System Error: {error}")
        await self.log_with_ollama(error, context)
        # Suggest and log recovery strategies
        recovery_strategy = await self.ollama.suggest_error_recovery(error)
        self.logger.info(f"Recovery Strategy: {recovery_strategy}")
        await self.log_with_ollama(f"Recovery Strategy: {recovery_strategy}", context)
        recovery_strategy = await self.ollama.suggest_error_recovery(error)
        self.logger.info(f"Recovery Strategy: {recovery_strategy}")
        await self.log_with_ollama(f"Recovery Strategy: {recovery_strategy}", context)

    async def log_with_ollama(self, message, context=None):
        """Log messages with Ollama's assistance."""
        prompt = f"Log this message: {message}"
        if context:
            prompt += f" | Context: {context}"
        await self.ollama.query_ollama("logging", prompt, refine=False)

    async def log_recovery(self, recovery_action, success=True):
        status = "successful" if success else "failed"
        self.logger.info(f"Recovery Action: {recovery_action} | Status: {status}")
        await self.log_with_ollama(recovery_action, {"success": success})

    async def control_improvement_process(self, ollama, si, kb, task_queue, vcs, ca, tf, dm, fs, pm, eh):
        while True:
            try:
                await self.log_state("Analyzing current system state")
                system_state = await ollama.evaluate_system_state({"metrics": await si.get_system_metrics()})

                await self.log_state("Generating improvement suggestions")
                improvements = await si.analyze_performance(system_state)

                if improvements:
                    for improvement in improvements:
                        validation = await si.validate_improvements([improvement])
                        if validation:
                            await self.log_decision(f"Applying improvement: {improvement}")
                            result = await si.apply_improvements([improvement])

                            experience_data = {
                                "improvement": improvement,
                                "result": result,
                                "system_state": system_state
                            }
                            learning = await si.learn_from_experience(experience_data)

                            await kb.add_entry(f"improvement_{int(time.time())}", {
                                "improvement": improvement,
                                "result": result,
                                "learning": learning
                            })

                            await self.log_state("Learning from experience", experience_data)

                await self.log_state("Performing additional system improvement tasks")
                await task_queue.manage_orchestration()
                code_analysis = await ca.analyze_code(ollama, "current_system_code")
                if code_analysis.get('improvements'):
                    for code_improvement in code_analysis['improvements']:
                        await si.apply_code_change(code_improvement)

                test_results = await tf.run_tests(ollama, "current_test_suite")
                if test_results.get('failed_tests'):
                    for failed_test in test_results['failed_tests']:
                        fix = await ollama.query_ollama("test_fixing", f"Fix this failed test: {failed_test}")
                        await si.apply_code_change(fix['code_change'])

                deployment_decision = await dm.deploy_code(ollama)
                if deployment_decision and deployment_decision.get('deploy', False):
                    await self.log_state("Deployment approved by Ollama")
                else:
                    await self.log_state("Deployment deferred based on Ollama's decision")

                await self.log_state("Performing version control operations")
                changes = "Recent system changes"
                await vcs.commit_changes(ollama, changes)

                fs.write_to_file("system_state.log", str(system_state))

                await self.log_state("Managing prompts")
                new_prompts = await pm.generate_new_prompts(ollama)
                for prompt_name, prompt_content in new_prompts.items():
                    pm.save_prompt(prompt_name, prompt_content)

                await self.log_state("Checking for system errors")
                system_errors = await eh.check_for_errors(ollama)
                if system_errors:
                    for error in system_errors:
                        await eh.handle_error(ollama, error)

                await self.log_state("Completed improvement cycle")

            except Exception as e:
                await self.log_error(f"Error in control_improvement_process: {str(e)}")
                recovery_suggestion = await eh.handle_error(ollama, e)
                if recovery_suggestion and recovery_suggestion.get('decompose_task', False):
                    subtasks = await eh.decompose_task(ollama, recovery_suggestion.get('original_task'))
                    await self.log_state("Decomposed task into subtasks", {"subtasks": subtasks})
                else:
                    await self.log_error("No valid recovery suggestion received from Ollama.", {"error": str(e)})

            # Check for reset command
            reset_command = await ollama.query_ollama("system_control", "Check if a reset is needed")
            if reset_command.get('reset', False):
                await self.log_state("Resetting system state as per command")
                # Example reset logic
                # This is a placeholder for actual reset logic
                # Assuming a simple reset command or script
                try:
                    subprocess.run(["./reset.sh"], check=True)
                    self.logger.info("System reset executed successfully.")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"System reset failed: {e}")
                continue

            await asyncio.sleep(3600)  # Wait for an hour
