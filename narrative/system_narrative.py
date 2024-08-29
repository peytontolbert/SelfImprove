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
        context = {
            "longterm_memory": longterm_memory,
            "current_tasks": "List of current tasks",
            "system_status": "Current system status"
        }
        ollama_response = await self.ollama.query_ollama(self.ollama.system_prompt, prompt, task="thought_generation", context=context)
        thoughts = ollama_response.get('thoughts', 'No thoughts generated')
        self.logger.info(f"Ollama Detailed Thoughts: {thoughts}")
        await self.knowledge_base.save_longterm_memory(longterm_memory)
        return thoughts

    async def log_state(self, message, context=None):
        if context:
            self.logger.info(f"System State: {message} | Context: {context}")
        else:
            self.logger.info(f"System State: {message}")
        await self.log_with_ollama(message, context)
        # Generate and log thoughts about the current state
        await self.generate_thoughts(context)
        # Analyze feedback and suggest improvements
        context = {"current_state": message}
        feedback = await self.ollama.query_ollama(self.ollama.system_prompt, f"Analyze feedback for the current state: {message}", task="feedback_analysis", context=context)
        self.logger.info(f"Feedback analysis: {feedback}")

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
        if context is None:
            context = {}
        context.update({"error": str(error)})
        recovery_strategy = await self.ollama.query_ollama(self.ollama.system_prompt, f"Suggest a recovery strategy for the following error: {str(error)}", task="error_recovery", context=context)
        self.logger.info(f"Recovery Strategy: {recovery_strategy}")
        await self.log_with_ollama(f"Recovery Strategy: {recovery_strategy}", context)
        context = {"error": str(error)}
        recovery_strategy = await self.ollama.query_ollama(self.ollama.system_prompt, f"Suggest a recovery strategy for the following error: {str(error)}", task="error_recovery", context=context)
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

    def calculate_improvement_cycle_frequency(self, system_state):
        """Calculate the sleep duration between improvement cycles based on system performance."""
        # Example logic: Adjust sleep duration based on a simple metric
        # This can be replaced with more sophisticated logic as needed
        performance_metric = system_state.get("performance_metric", 1)
        if performance_metric > 0.8:
            return 1800  # 30 minutes
        elif performance_metric > 0.5:
            return 3600  # 1 hour
        else:
            return 7200  # 2 hours
    async def control_improvement_process(self, ollama, si, kb, task_queue, vcs, ca, tf, dm, fs, pm, eh):
        system_state = {}
        while True:
            try:
                async with asyncio.timeout(300):  # 5-minute timeout for the entire cycle
                    # Log the start of system state analysis
                    await self.log_state("Analyzing current system state")
                    self.logger.info("System state analysis started.")
                    system_state = await ollama.evaluate_system_state({"metrics": await si.get_system_metrics()})

                    # Log the start of generating improvement suggestions
                    await self.log_state("Generating improvement suggestions")
                    self.logger.info("Improvement suggestions generation started.")
                    improvements = await si.retry_ollama_call(si.analyze_performance, system_state)

                    if improvements:
                        for improvement in improvements:
                            validation = await si.validate_improvements([improvement])
                            if validation:
                                await self.log_decision(f"Applying improvement: {improvement}")
                                result = await si.retry_ollama_call(si.apply_improvements, [improvement])

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
                                }, narrative_context={"system_state": system_state})

                                await self.log_state("Learning from experience", experience_data)
                                self.logger.info(f"Improvement result: {result}")
                                new_metrics = await si.get_system_metrics()
                                self.logger.info(f"System metrics before improvement: {system_state.get('metrics', {})}")
                                self.logger.info(f"System metrics after improvement: {new_metrics}")
                                await kb.add_entry(f"metrics_{int(time.time())}", {
                                    "before": system_state.get('metrics', {}),
                                    "after": new_metrics
                                })

                    # Perform additional system improvement tasks
                    await self.log_state("Performing additional system improvement tasks")
                    self.logger.info("Additional system improvement tasks started.")
                    await task_queue.manage_orchestration()
                    
                    # Analyze code for potential improvements
                    code_analysis = await si.retry_ollama_call(ca.analyze_code, ollama, "current_system_code")
                    if code_analysis.get('improvements'):
                        for code_improvement in code_analysis['improvements']:
                            await si.apply_code_change(code_improvement)

                    # Run and fix tests
                    test_results = await tf.run_tests(ollama, "current_test_suite")
                    if test_results.get('failed_tests'):
                        for failed_test in test_results['failed_tests']:
                            fix = await ollama.query_ollama("test_fixing", f"Fix this failed test: {failed_test}")
                            await si.apply_code_change(fix['code_change'])

                    # Decide on deployment
                    deployment_decision = await si.retry_ollama_call(dm.deploy_code, ollama)
                    if deployment_decision and deployment_decision.get('deploy', False):
                        await self.log_state("Deployment approved by Ollama")
                    else:
                        await self.log_state("Deployment deferred based on Ollama's decision")

                    # Perform version control operations
                    await self.log_state("Performing version control operations with Ollama's guidance")
                    self.logger.info("Version control operations started.")
                    changes = "Recent system changes"
                    await vcs.commit_changes(ollama, changes)

                    fs.write_to_file("system_state.log", str(system_state))

                    # Manage prompts
                    await self.log_state("Managing prompts")
                    new_prompts = await pm.generate_new_prompts(ollama)
                    for prompt_name, prompt_content in new_prompts.items():
                        pm.save_prompt(prompt_name, prompt_content)

                    # Check for system errors
                    await self.log_state("Checking for system errors")
                    system_errors = await eh.check_for_errors(ollama)
                    if system_errors:
                        for error in system_errors:
                            await eh.handle_error(ollama, error)

                    await self.log_state("Completed improvement cycle")

                    # Assess alignment implications
                    context = {"recent_changes": "recent_system_changes_placeholder"}
                    alignment_considerations = await ollama.query_ollama(
                        "alignment_consideration",
                        "Assess the alignment implications of recent system changes. Consider user behavior nuances and organizational goals.",
                        context=context
                    )
                    if not alignment_considerations or not any(alignment_considerations.values()):
                        self.logger.warning("Alignment considerations are missing or incomplete. Initiating detailed analysis.")
                        alignment_considerations = await ollama.query_ollama(
                            "alignment_consideration",
                            "Provide a detailed analysis of alignment implications considering user behavior nuances and organizational goals.",
                            context=context
                        )
                    self.logger.info(f"Alignment considerations: {alignment_considerations}")
                    for implication in alignment_considerations.get('assessed_implications', []):
                        category = implication.get('category', 'unknown')
                        description = implication.get('description', 'No description provided.')
                        urgency = implication.get('level_of_urgency', 'unknown')
                        
                        self.logger.info(f"Implication Category: {category} | Description: {description} | Urgency: {urgency}")
                        
                        if urgency == 'high':
                            self.logger.warning(f"High urgency implication detected in category: {category}. Immediate action required.")
                            # Implement logic to handle high urgency implications
                        elif urgency == 'medium-high':
                            self.logger.info(f"Medium-high urgency implication detected in category: {category}. Prioritize for review.")
                            # Implement logic to handle medium-high urgency implications
                        elif urgency == 'low-medium':
                            self.logger.info(f"Low-medium urgency implication detected in category: {category}. Monitor and review as needed.")
                            # Implement logic to handle low-medium urgency implications
                        else:
                            self.logger.error(f"Unknown urgency level: {urgency}.")

            except asyncio.TimeoutError:
                await self.log_error("Timeout occurred in control_improvement_process")
                await self.handle_timeout()
            except Exception as e:
                await self.log_error(f"Error in control_improvement_process: {str(e)}")
                recovery_suggestion = await eh.handle_error(ollama, e)
                if recovery_suggestion and recovery_suggestion.get('decompose_task', False):
                    subtasks = await eh.decompose_task(ollama, recovery_suggestion.get('original_task'))
                    await self.log_state("Decomposed task into subtasks", {"subtasks": subtasks})
                else:
                    await self.log_error("No valid recovery suggestion received from Ollama.", {"error": str(e)})

            # Check for reset command
            context = {"system_state": "current_system_state_placeholder"}
            reset_command = await ollama.query_ollama("system_control", "Check if a reset is needed", context=context)
            if reset_command.get('reset', False):
                await self.log_state("Resetting system state as per command")
                try:
                    subprocess.run(["./reset.sh"], check=True)
                    self.logger.info("System reset executed successfully.")
                except subprocess.CalledProcessError as e:
                    self.logger.error(f"System reset failed: {e}")
                continue

            # Dynamically adjust the sleep duration based on system performance
            sleep_duration = self.calculate_improvement_cycle_frequency(system_state)
            await asyncio.sleep(sleep_duration)

    async def handle_timeout(self):
        self.logger.warning("Timeout occurred in the improvement cycle. Initiating recovery process.")
        await self.log_state("Timeout recovery initiated")

        # 1. Save the current state
        current_state = await self.ollama.evaluate_system_state({})
        await self.knowledge_base.add_entry("timeout_state", current_state)

        # 2. Query Ollama for recovery actions
        recovery_actions = await self.ollama.query_ollama("timeout_recovery", "Suggest detailed recovery actions for a timeout in the improvement cycle, including component restarts and resource adjustments.")

        # 3. Log recovery actions
        self.logger.info(f"Suggested recovery actions: {recovery_actions}")

        # 4. Implement recovery actions
        for action in recovery_actions.get('actions', []):
            if action.get('type') == 'restart_component':
                component = action.get('component')
                self.logger.info(f"Restarting component: {component}")
                # Implement restart logic here
                # For example: await self.restart_component(component)
            elif action.get('type') == 'adjust_resource':
                resource = action.get('resource')
                new_value = action.get('new_value')
                self.logger.info(f"Adjusting resource: {resource} to {new_value}")
                # Implement resource adjustment logic here
                # For example: await self.adjust_resource(resource, new_value)

        # 5. Notify administrators
        admin_notification = f"Timeout occurred in improvement cycle. Recovery actions taken: {recovery_actions}"
        self.logger.critical(admin_notification)
        # Implement admin notification logic here
        # For example: await self.notify_admin(admin_notification)

        # 6. Adjust future timeout duration
        new_timeout = recovery_actions.get('new_timeout', 300)  # Default to 5 minutes if not specified
        self.logger.info(f"Adjusting future timeout duration to {new_timeout} seconds")
        # Implement timeout adjustment logic here
        # For example: self.timeout_duration = new_timeout

        await self.log_state("Timeout recovery completed")
        return recovery_actions
