import logging
from knowledge_base import KnowledgeBase
from core.ollama_interface import OllamaInterface
import asyncio
import time
import subprocess
import json
from reinforcement_learning_module import ReinforcementLearningModule
from spreadsheet_manager import SpreadsheetManager
class SystemNarrative:
    def __init__(self, ollama_interface=None, knowledge_base=None):
        self.request_log = []  # Initialize a log to track requests and expected responses
        self.logger = logging.getLogger("SystemNarrative")
        self.ollama = ollama_interface or OllamaInterface()
        self.knowledge_base = knowledge_base or KnowledgeBase()
        self.spreadsheet_manager = SpreadsheetManager("narrative_data.xlsx")
        self.rl_module = ReinforcementLearningModule(ollama_interface)
        logging.basicConfig(level=logging.INFO)

    async def generate_thoughts(self, context=None):
        """Generate detailed thoughts or insights about the current state and tasks."""
        longterm_memory = await self.knowledge_base.get_longterm_memory()
        prompt = "Generate detailed thoughts about the current system state, tasks, and potential improvements."
        if context:
            prompt += f" | Context: {context}"
        if longterm_memory:
            prompt += f" | Long-term Memory: {longterm_memory}"
        context = context or {}
        context.update({
            "longterm_memory": longterm_memory,
            "current_tasks": "List of current tasks",
            "system_status": "Current system status"
        })
        self.logger.info(f"Generated thoughts with context: {json.dumps(context, indent=2)}")
        self.knowledge_base.log_interaction("SystemNarrative", "generate_thoughts", {"context": context}, improvement="Generated thoughts")
        self.track_request("thought_generation", prompt, "thoughts")
        ollama_response = await self.ollama.query_ollama(self.ollama.system_prompt, prompt, task="thought_generation", context=context)
        thoughts = ollama_response.get('thoughts', 'No thoughts generated')
        self.logger.info(f"Ollama Detailed Thoughts: {thoughts}", extra={"thoughts": thoughts})
        await self.knowledge_base.save_longterm_memory(longterm_memory)
        # Log thoughts to spreadsheet
        self.spreadsheet_manager.write_data((1, 1), [["Thoughts"], [thoughts]])
        return thoughts

    def track_request(self, task, prompt, expected_response):
        """Track requests made to Ollama and the expected responses."""
        self.request_log.append({
            "task": task,
            "prompt": prompt,
            "expected_response": expected_response,
            "timestamp": time.time()
        })
        self.logger.info(f"Tracked request for task '{task}' with expected response: {expected_response}")

    async def log_state(self, message, context=None):
        if context is None:
            context = {}
        context.update({
            "system_status": "Current system status",
            "recent_changes": "Recent changes in the system",
            "longterm_memory": await self.knowledge_base.get_longterm_memory()
        })
        self.logger.info(f"System State: {message} | Context: {json.dumps(context, indent=2)}")
        self.spreadsheet_manager.write_data((5, 1), [["State"], [message]])
        self.spreadsheet_manager.write_data((5, 1), [["State"], [message]])
        await self.log_with_ollama(message, context)
        # Log state to spreadsheet
        self.spreadsheet_manager.write_data((5, 1), [["State"], [message]])
        # Generate and log thoughts about the current state
        await self.generate_thoughts(context)
        # Analyze feedback and suggest improvements
        self.track_request("feedback_analysis", f"Analyze feedback for the current state: {message}", "feedback")
        feedback = await self.ollama.query_ollama(self.ollama.system_prompt, f"Analyze feedback for the current state: {message}", task="feedback_analysis", context=context)
        self.logger.info(f"Feedback analysis: {feedback}")

    async def log_decision(self, decision, rationale=None):
        """Log decisions with detailed rationale."""
        if rationale:
            self.logger.info(f"System Decision: {decision} | Detailed Rationale: {rationale}")
        else:
            self.logger.info(f"System Decision: {decision}")
        await self.log_with_ollama(decision, rationale)
        # Log decision to spreadsheet
        self.spreadsheet_manager.write_data((10, 1), [["Decision", "Rationale"], [decision, rationale or ""]])
        # Generate and log thoughts about the decision
        await self.generate_thoughts({"decision": decision, "rationale": rationale})

    async def log_error(self, error, context=None):
        """Log errors with context and recovery strategies."""
        if context:
            self.logger.error(f"System Error: {error} | Context: {context}", extra={"error": str(error), "context": context})
        else:
            self.logger.error(f"System Error: {error} | Context: {json.dumps(context or {})}")
        await self.log_with_ollama(error, context)
        # Log error to spreadsheet
        self.spreadsheet_manager.write_data((15, 1), [["Error", "Context"], [str(error), json.dumps(context or {})]])
        # Suggest and log recovery strategies
        if context is None:
            context = {}
        context.update({"error": str(error)})
        recovery_strategy = await self.ollama.query_ollama(self.ollama.system_prompt, f"Suggest a recovery strategy for the following error: {str(error)}", task="error_recovery", context=context)
        self.logger.info(f"Recovery Strategy: {recovery_strategy}", extra={"recovery_strategy": recovery_strategy})
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
        improvement_cycle_count = 0
        while True:
            improvement_cycle_count += 1
            try:
                await asyncio.wait_for(self.improvement_cycle(ollama, si, kb, task_queue, vcs, ca, tf, dm, fs, pm, eh, improvement_cycle_count), timeout=300)  # 5-minute timeout for the entire cycle
            except asyncio.TimeoutError:
                await self.handle_timeout_error()
            except Exception as e:
                await self.handle_general_error(e, eh, ollama)

            # Check for reset command and adjust sleep duration
            await self.check_reset_and_sleep(ollama, system_state)

    async def improvement_cycle(self, ollama, si, kb, task_queue, vcs, ca, tf, dm, fs, pm, eh, improvement_cycle_count):
        await self.log_state(f"Starting improvement cycle {improvement_cycle_count}")
        
        # System state analysis
        await self.log_state("Analyzing current system state")
        system_state = await ollama.evaluate_system_state({"metrics": await si.get_system_metrics()})
        self.logger.info(f"System state: {json.dumps(system_state, indent=2)}")

        # Generate and apply improvements in parallel
        await self.log_state("Generating improvement suggestions")
        improvements = await si.retry_ollama_call(si.analyze_performance, system_state)
        
        # Validate and apply improvements in parallel
        if improvements:
            tasks = [self.apply_and_log_improvement(si, kb, improvement, system_state) for improvement in improvements]
            await asyncio.gather(*tasks)

        # Perform additional tasks in parallel
        await asyncio.gather(
            self.perform_additional_tasks(task_queue, ca, tf, dm, vcs, ollama, si),
            self.manage_prompts_and_errors(pm, eh, ollama),
            self.assess_alignment_implications(ollama)
        )

        # Manage prompts and check for errors
        await self.manage_prompts_and_errors(pm, eh, ollama)

        # Assess alignment implications
        await self.assess_alignment_implications(ollama)

        # Use reinforcement learning feedback and predictive analysis
        rl_feedback = await self.rl_module.get_feedback(system_state)
        self.logger.info(f"Reinforcement learning feedback: {rl_feedback}")
        await self.knowledge_base.add_entry("rl_feedback", {"feedback": rl_feedback})
        self.spreadsheet_manager.write_data((25, 1), [["Reinforcement Learning Feedback"], [rl_feedback]])

        # Integrate predictive analysis
        # Enhance predictive analysis with historical data
        historical_data = await self.knowledge_base.get_entry("historical_metrics")
        predictive_context = {**system_state, "historical_data": historical_data}
        predictive_insights = await self.ollama.query_ollama("predictive_analysis", "Provide predictive insights based on current and historical system metrics.", context=predictive_context)
        self.logger.info(f"Enhanced Predictive insights: {predictive_insights}")
        await self.knowledge_base.add_entry("predictive_insights", {"insights": predictive_insights})
        self.spreadsheet_manager.write_data((30, 1), [["Enhanced Predictive Insights"], [predictive_insights]])

        # Evolve feedback loop for long-term evolution
        await self.evolve_feedback_loop(rl_feedback, predictive_insights)
        await self.log_state(f"Completed improvement cycle {improvement_cycle_count}")

    async def evolve_feedback_loop(self, rl_feedback, predictive_insights):
        """Evolve the feedback loop by integrating reinforcement learning feedback and predictive insights."""
        # Refine feedback loop with adaptive mechanisms
        historical_feedback = await self.knowledge_base.get_entry("historical_feedback")
        combined_feedback = rl_feedback + predictive_insights.get("suggestions", []) + historical_feedback.get("feedback", [])
        self.logger.info(f"Refining feedback loop with adaptive feedback: {combined_feedback}")
        await self.knowledge_base.add_entry("refined_feedback", {"combined_feedback": combined_feedback})
        self.spreadsheet_manager.write_data((35, 1), [["Refined Feedback"], [combined_feedback]])
        # Implement further logic to utilize combined feedback for long-term evolution

    async def apply_and_log_improvement(self, si, kb, improvement, system_state):
        await self.log_decision(f"Applying improvement: {improvement}")
        result = await si.retry_ollama_call(si.apply_improvements, [improvement])
        experience_data = {"improvement": improvement, "result": result, "system_state": system_state}
        kb.log_interaction("SelfImprovement", "apply_and_log_improvement", {"improvement": improvement, "result": result})
        learning = await si.learn_from_experience(experience_data)
        await kb.add_entry(f"improvement_{int(time.time())}", {
            "improvement": improvement,
            "result": result,
            "learning": learning
        }, narrative_context={"system_state": system_state})
        await self.log_state("Learning from experience", context=experience_data)
        self.logger.info(f"Improvement result: {result}")
        new_metrics = await si.get_system_metrics()
        self.logger.info(f"Metrics before: {system_state.get('metrics', {})}")
        self.logger.info(f"Metrics after: {new_metrics}")
        await kb.add_entry(f"metrics_{int(time.time())}", {
            "before": system_state.get('metrics', {}),
            "after": new_metrics
        })

    async def perform_additional_tasks(self, task_queue, ca, tf, dm, vcs, ollama, si):
        await self.log_state("Performing additional system improvement tasks")
        await task_queue.manage_orchestration()
        
        code_analysis = await si.retry_ollama_call(ca.analyze_code, ollama, "current_system_code")
        if code_analysis.get('improvements'):
            for code_improvement in code_analysis['improvements']:
                await si.apply_code_change(code_improvement)

        test_results = await tf.run_tests(ollama, "current_test_suite")
        if test_results.get('failed_tests'):
            for failed_test in test_results['failed_tests']:
                fix = await ollama.query_ollama("test_fixing", f"Fix this failed test: {failed_test}")
                await si.apply_code_change(fix['code_change'])

        deployment_decision = await si.retry_ollama_call(dm.deploy_code, ollama)
        if deployment_decision and deployment_decision.get('deploy', False):
            await self.log_state("Deployment approved by Ollama")
        else:
            await self.log_state("Deployment deferred based on Ollama's decision")

        await self.log_state("Performing version control operations")
        changes = "Recent system changes"
        await vcs.commit_changes(ollama, changes)

    async def manage_prompts_and_errors(self, pm, eh, ollama):
        await self.log_state("Managing prompts")
        new_prompts = await pm.generate_new_prompts(ollama)
        for prompt_name, prompt_content in new_prompts.items():
            pm.save_prompt(prompt_name, prompt_content)

        await self.log_state("Checking for system errors")
        system_errors = await eh.check_for_errors(ollama)
        if system_errors:
            for error in system_errors:
                await eh.handle_error(ollama, error)

    async def assess_alignment_implications(self, ollama):
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
        await self.process_alignment_implications(alignment_considerations)

    async def process_alignment_implications(self, alignment_considerations):
        for implication in alignment_considerations.get('assessed_implications', []):
            category = implication.get('category', 'unknown')
            description = implication.get('description', 'No description provided.')
            urgency = implication.get('level_of_urgency', 'unknown')
            
            self.logger.info(f"Implication Category: {category} | Description: {description} | Urgency: {urgency}")
            
            if urgency == 'high':
                await self.handle_high_urgency_implication(category, description)
            elif urgency == 'medium-high':
                await self.handle_medium_high_urgency_implication(category, description)
            elif urgency == 'low-medium':
                await self.handle_low_medium_urgency_implication(category, description)
            else:
                self.logger.error(f"Unknown urgency level: {urgency}.")

    async def handle_high_urgency_implication(self, category, description):
        self.logger.warning(f"High urgency implication detected in category: {category}. Immediate action required.")
        # Implement logic to handle high urgency implications
        # For example, trigger an immediate review or alert the system administrators
        await self.log_state(f"High urgency implication in {category}: {description}")
        # You might want to add a method to alert administrators or trigger an immediate response

    async def handle_medium_high_urgency_implication(self, category, description):
        self.logger.info(f"Medium-high urgency implication detected in category: {category}. Prioritize for review.")
        # Implement logic to handle medium-high urgency implications
        # For example, add to a priority queue for review
        await self.log_state(f"Medium-high urgency implication in {category}: {description}")
        # You might want to add a method to schedule a review or add to a priority task list

    async def handle_low_medium_urgency_implication(self, category, description):
        self.logger.info(f"Low-medium urgency implication detected in category: {category}. Monitor and review as needed.")
        # Implement logic to handle low-medium urgency implications
        # For example, add to a monitoring list
        await self.log_state(f"Low-medium urgency implication in {category}: {description}")
        # You might want to add a method to add this to a monitoring list or schedule a future review

    async def handle_timeout_error(self):
        await self.log_error("Timeout occurred in control_improvement_process")
        await self.handle_timeout()

    async def handle_general_error(self, e, eh, ollama):
        await self.log_error(f"Error in control_improvement_process: {str(e)}")
        await self.process_recovery_suggestion(eh, ollama, e)

    async def process_recovery_suggestion(self, eh, ollama, e):
        recovery_suggestion = await eh.handle_error(ollama, e)
        if recovery_suggestion and recovery_suggestion.get('decompose_task', False):
            subtasks = await eh.decompose_task(ollama, recovery_suggestion.get('original_task'))
            await self.log_state("Decomposed task into subtasks", {"subtasks": subtasks})
        else:
            await self.log_error("No valid recovery suggestion received from Ollama.", {"error": str(e)})

    async def check_reset_and_sleep(self, ollama, system_state):
        context = {"system_state": "current_system_state_placeholder"}
        reset_command = await ollama.query_ollama("system_control", "Check if a reset is needed", context=context)
        if reset_command.get('reset', False):
            await self.log_state("Resetting system state as per command")
            try:
                subprocess.run(["./reset.sh"], check=True)
                self.logger.info("System reset executed successfully.")
            except subprocess.CalledProcessError as e:
                self.logger.error(f"System reset failed: {e}")
            return


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
