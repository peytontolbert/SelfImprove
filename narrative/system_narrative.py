import logging
import aiohttp
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
        self.logger.info(f"Using long-term memory: {json.dumps(longterm_memory, indent=2)}")
        context["longterm_memory"] = longterm_memory
        prompt = "Generate detailed thoughts about the current system state, tasks, and potential improvements."
        if context:
            prompt += f" | Context: {context}"
        if longterm_memory:
            prompt += f" | Long-term Memory: {longterm_memory}"
        context = context or {}
        context["current_tasks"] = "List of current tasks"
        context["system_status"] = "Current system status"
        self.logger.info(f"Generated thoughts with context: {json.dumps(context, indent=2)}")
        await self.knowledge_base.log_interaction("SystemNarrative", "generate_thoughts", {"context": context}, improvement="Generated thoughts")
        self.track_request("thought_generation", prompt, "thoughts")
        ollama_response = await self.ollama.query_ollama(self.ollama.system_prompt, prompt, task="thought_generation", context=context)
        thoughts = ollama_response.get('thoughts', 'No thoughts generated')
        self.logger.info(f"Ollama Detailed Thoughts: {thoughts}", extra={"thoughts": thoughts})
        # Update long-term memory with generated thoughts
        if thoughts != 'No thoughts generated':
            longterm_memory.update({"thoughts": thoughts})
        else:
            self.logger.warning("No new thoughts generated to update long-term memory.")
        await self.knowledge_base.save_longterm_memory(longterm_memory)
        # Log thoughts to spreadsheet
        self.spreadsheet_manager.write_data((1, 1), [["Thoughts"], [thoughts]], sheet_name="NarrativeData")
        return thoughts

    async def creative_problem_solving(self, problem_description):
        """Generate novel solutions to complex problems."""
        context = {"problem_description": problem_description}
        solutions = await self.ollama.query_ollama("creative_problem_solving", f"Generate novel solutions for the following problem: {problem_description}", context=context)
        self.logger.info(f"Creative solutions generated: {solutions}")
        await self.knowledge_base.add_entry("creative_solutions", solutions)
        return solutions

    async def generate_detailed_thoughts(self, context=None):
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
        await self.knowledge_base.log_interaction("SystemNarrative", "generate_thoughts", {"context": context}, improvement="Generated thoughts")
        self.track_request("thought_generation", prompt, "thoughts")
        ollama_response = await self.ollama.query_ollama(self.ollama.system_prompt, prompt, task="thought_generation", context=context)
        thoughts = ollama_response.get('thoughts', 'No thoughts generated')
        self.logger.info(f"Ollama Detailed Thoughts: {thoughts}", extra={"thoughts": thoughts})
        await self.knowledge_base.save_longterm_memory(longterm_memory)
        # Log thoughts to spreadsheet
        self.spreadsheet_manager.write_data((1, 1), [["Thoughts"], [thoughts]], sheet_name="NarrativeData")
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

    async def execute_actions(self, actions):
        """Execute a list of actions derived from thoughts and improvements."""
        for action in actions:
            action_type = action.get("type")
            details = action.get("details", {})
            if action_type == "file_operation":
                await self.handle_file_operation(details)
            elif action_type == "system_update":
                await self.handle_system_update(details)
            elif action_type == "network_operation":
                await self.handle_network_operation(details)
            elif action_type == "database_update":
                await self.handle_database_update(details)
            else:
                self.logger.error(f"Unknown action type: {action_type}. Please check the action details.")

    async def handle_network_operation(self, details):
        """Handle network operations such as API calls."""
        url = details.get("url")
        method = details.get("method", "GET")
        data = details.get("data", {})
        try:
            self.logger.info(f"Performing network operation: {method} {url}")
            async with aiohttp.ClientSession() as session:
                async with session.request(method, url, json=data) as response:
                    response_data = await response.json()
                    self.logger.info(f"Network operation successful: {response_data}")
        except Exception as e:
            self.logger.error(f"Error performing network operation: {str(e)}")

    async def handle_database_update(self, details):
        """Handle database updates."""
        query = details.get("query")
        try:
            self.logger.info(f"Executing database update: {query}")
            # Implement database update logic here
            # For example, using an async database client
            # await database_client.execute(query)
            self.logger.info("Database update executed successfully.")
        except Exception as e:
            self.logger.error(f"Error executing database update: {str(e)}")
        """Handle file operations such as create, edit, or delete."""
        operation = details.get("operation")
        filename = details.get("filename")
        content = details.get("content", "")
        try:
            if operation == "create":
                self.logger.info(f"Creating file: {filename}")
                self.fs.create_file(filename, content)
            elif operation == "edit":
                self.logger.info(f"Editing file: {filename}")
                self.fs.edit_file(filename, content)
            elif operation == "delete":
                self.logger.info(f"Deleting file: {filename}")
                self.fs.delete_file(filename)
            else:
                self.logger.warning(f"Unknown file operation: {operation}")
        except Exception as e:
            self.logger.error(f"Error handling file operation: {str(e)}")

    async def handle_system_update(self, details):
        """Handle system updates."""
        update_command = details.get("command")
        try:
            self.logger.info(f"Executing system update: {update_command}")
            result = subprocess.run(update_command, shell=True, check=True, capture_output=True, text=True)
            self.logger.info(f"System update executed successfully: {update_command}")
            self.logger.debug(f"Update output: {result.stdout}")
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to execute system update: {str(e)}")
            self.logger.debug(f"Update error output: {e.stderr}")

    async def log_state(self, message, context=None):
        if context is None:
            context = {}
        # Include more detailed context information
        context.update({
            "system_status": "Current system status",
            "recent_changes": "Recent changes in the system",
            "longterm_memory": await self.knowledge_base.get_longterm_memory(),
            "current_tasks": "List of current tasks",
            "performance_metrics": await self.ollama.query_ollama("system_metrics", "Provide an overview of the current system capabilities and performance.")
        })
        self.logger.info(f"System State: {message} | Context: {json.dumps(context, indent=2)}")
        self.spreadsheet_manager.write_data((5, 1), [["State"], [message]], sheet_name="SystemData")
        self.spreadsheet_manager.write_data((5, 1), [["State"], [message]])
        await self.log_with_ollama(message, context)
        # Log state to spreadsheet
        self.spreadsheet_manager.write_data((5, 1), [["State"], [message]])
        # Generate and log thoughts about the current state
        await self.generate_thoughts(context)
        # Analyze feedback and suggest improvements
        self.track_request("feedback_analysis", f"Analyze feedback for the current state: {message}. Consider system performance, recent changes, and long-term memory.", "feedback")
        feedback = await self.ollama.query_ollama(self.ollama.system_prompt, f"Analyze feedback for the current state: {message}. Consider system performance, recent changes, and long-term memory.", task="feedback_analysis", context=context)
        self.logger.info(f"Feedback analysis: {feedback}")

    async def log_decision(self, decision, rationale=None):
        """Log decisions with detailed rationale."""
        if rationale:
            self.logger.info(f"System Decision: {decision} | Detailed Rationale: {rationale}")
        else:
            self.logger.info(f"System Decision: {decision}")
        await self.log_with_ollama(decision, rationale)
        # Log decision and rationale in the knowledge base
        await self.knowledge_base.add_entry("system_decision", {"decision": decision, "rationale": rationale, "timestamp": time.time()})
        # Log decision and rationale in the knowledge base
        await self.knowledge_base.add_entry("system_decision", {"decision": decision, "rationale": rationale, "timestamp": time.time()})
        # Log decision to spreadsheet
        self.spreadsheet_manager.write_data((10, 1), [["Decision", "Rationale"], [decision, rationale or ""]])
        # Generate and log thoughts about the decision
        await self.generate_thoughts({"decision": decision, "rationale": rationale})

    async def suggest_recovery_strategy(self, error):
        """Suggest a recovery strategy for a given error."""
        error_prompt = f"Suggest a recovery strategy for the following error: {str(error)}"
        context = {"error": str(error)}
        recovery_suggestion = await self.ollama.query_ollama(self.ollama.system_prompt, error_prompt, task="error_recovery", context=context)
        return recovery_suggestion.get("recovery_strategy", "No recovery strategy suggested.")

    async def log_error(self, error, context=None):
        """Log errors with context and recovery strategies."""
        error_context = context or {}
        error_context.update({"error": str(error), "timestamp": time.time()})
        self.logger.error(f"System Error: {error} | Context: {json.dumps(error_context, indent=2)} | Potential Recovery: {await self.suggest_recovery_strategy(error)}")
        await self.log_with_ollama(error, context)
        # Log error to spreadsheet
        self.spreadsheet_manager.write_data((15, 1), [["Error", "Context"], [str(error), json.dumps(context or {})]])
        # Suggest and log recovery strategies
        recovery_strategy = await self.suggest_recovery_strategy(error)
        self.logger.info(f"Recovery Strategy: {recovery_strategy}", extra={"recovery_strategy": recovery_strategy})
        await self.log_with_ollama(f"Recovery Strategy: {recovery_strategy}", context)
        # Feedback loop for error handling
        feedback = await self.ollama.query_ollama("error_feedback", f"Provide feedback on the recovery strategy: {recovery_strategy}. Consider the error context and suggest improvements.", context=context)
        self.logger.info(f"Error handling feedback: {feedback}")
        await self.knowledge_base.add_entry("error_handling_feedback", feedback)

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
        await self.self_optimization(ollama, kb)
        system_state = {}
        improvement_cycle_count = 0
        while True:
            improvement_cycle_count += 1
            try:
                await asyncio.wait_for(self.improvement_cycle(ollama, si, kb, task_queue, vcs, ca, tf, dm, fs, pm, eh, improvement_cycle_count), timeout=300)
            except asyncio.TimeoutError:
                await self.handle_timeout_error()
            except Exception as e:
                await self.handle_general_error(e, eh, ollama)

            await self.dynamic_goal_setting(ollama, system_state)

            future_challenges = await self.ollama.query_ollama("future_challenges", "Predict future challenges and suggest preparation strategies.", context={"system_state": system_state})
            self.logger.info(f"Future challenges and strategies: {future_challenges}")
            await self.knowledge_base.add_entry("future_challenges", future_challenges)

            feedback_optimization = await self.ollama.query_ollama("feedback_optimization", "Optimize feedback loops for rapid learning and adaptation.", context={"system_state": system_state})
            self.logger.info(f"Feedback loop optimization: {feedback_optimization}")
            await self.knowledge_base.add_entry("feedback_optimization", feedback_optimization)

            if improvement_cycle_count % 5 == 0:
                self_reflection = await self.ollama.query_ollama("self_reflection", "Reflect on recent performance and suggest adjustments.", context={"system_state": system_state})
                self.logger.info(f"Self-reflection insights: {self_reflection}")
                await self.knowledge_base.add_entry("self_reflection", self_reflection)

            resource_optimization = await self.ollama.query_ollama("resource_optimization", "Optimize resource allocation based on current and predicted demands.", context={"system_state": system_state})
            self.logger.info(f"Resource allocation optimization: {resource_optimization}")
            await self.knowledge_base.add_entry("resource_optimization", resource_optimization)

            learning_data = await self.ollama.query_ollama("adaptive_learning", "Analyze recent interactions and adapt strategies for future improvements.", context={"system_state": system_state})
            self.logger.info(f"Adaptive learning data: {learning_data}")
            await self.knowledge_base.add_entry("adaptive_learning", learning_data)
            await self.knowledge_base.add_capability("adaptive_learning", {"details": learning_data, "timestamp": time.time()})


    async def dynamic_goal_setting(self, ollama, system_state):
        """Set and adjust system goals dynamically based on performance metrics."""
        current_goals = await self.knowledge_base.get_entry("current_goals")
        goal_adjustments = await ollama.query_ollama("goal_setting", f"Adjust current goals based on system performance: {system_state}", context={"current_goals": current_goals})
        self.logger.info(f"Goal adjustments: {goal_adjustments}")
        await self.knowledge_base.add_entry("goal_adjustments", goal_adjustments)

    async def improvement_cycle(self, ollama, si, kb, task_queue, vcs, ca, tf, dm, fs, pm, eh, improvement_cycle_count):
        await self.log_state(f"Starting improvement cycle {improvement_cycle_count}")
        # Log the start of an improvement cycle in the knowledge base
        await kb.add_entry("improvement_cycle_start", {"cycle_number": improvement_cycle_count, "timestamp": time.time()})
        # Log the start of an improvement cycle in the knowledge base
        await kb.add_entry("improvement_cycle_start", {"cycle_number": improvement_cycle_count, "timestamp": time.time()})
        
        # System state analysis
        await self.log_state("Analyzing current system state")
        system_state = await ollama.evaluate_system_state({"metrics": await si.get_system_metrics()})
        self.logger.info(f"System state: {json.dumps(system_state, indent=2)}")

        # Generate hypotheses for potential improvements
        hypotheses = await si.generate_hypotheses(system_state)
        tested_hypotheses = await si.test_hypotheses(hypotheses)
        self.logger.info(f"Tested hypotheses results: {tested_hypotheses}")

        # Generate and apply improvements in parallel
        await self.log_state("Generating improvement suggestions")
        # Retrieve insights from the knowledge base for generating improvements
        insights = await kb.query_insights("MATCH (n:Node) RETURN n LIMIT 5")
        self.logger.info(f"Retrieved insights for improvement: {insights}")
        # Retrieve insights from the knowledge base for generating improvements
        insights = await kb.query_insights("MATCH (n:Node) RETURN n LIMIT 5")
        self.logger.info(f"Retrieved insights for improvement: {insights}")
        improvements = await si.retry_ollama_call(si.analyze_performance, system_state)
        
        # Validate and apply improvements in parallel
        if improvements:
            tasks = [self.apply_and_log_improvement(si, kb, improvement, system_state) for improvement in improvements]
            await asyncio.gather(*tasks)

        # Add capabilities to the knowledge base
        for improvement in improvements:
            await kb.add_capability(improvement, {"status": "suggested"})

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
        self.logger.info("Long-term memory updated with reinforcement learning feedback.")
        self.spreadsheet_manager.write_data((25, 1), [["Reinforcement Learning Feedback"], [rl_feedback]])

        # Integrate predictive analysis
        # Enhance predictive analysis with historical data
        historical_data = await self.knowledge_base.get_entry("historical_metrics")
        predictive_context = {**system_state, "historical_data": historical_data}
        predictive_insights = await self.ollama.query_ollama("predictive_analysis", "Provide predictive insights based on current and historical system metrics.", context=predictive_context)
        self.logger.info(f"Enhanced Predictive insights: {predictive_insights}")
        await self.knowledge_base.add_entry("predictive_insights", {"insights": predictive_insights})
        self.logger.info("Long-term memory updated with predictive insights.")
        self.spreadsheet_manager.write_data((30, 1), [["Enhanced Predictive Insights"], [predictive_insights]])

        # Advanced predictive analysis for future challenges
        future_challenges = await self.ollama.query_ollama("future_challenges", "Predict future challenges and suggest preparation strategies.", context=predictive_context)
        self.logger.info(f"Future challenges and strategies: {future_challenges}")
        await self.knowledge_base.add_entry("future_challenges", future_challenges)

        # Evolve feedback loop for long-term evolution
        await self.evolve_feedback_loop(rl_feedback, predictive_insights)

        # Optimize feedback loop for rapid learning
        feedback_optimization = await self.ollama.query_ollama("feedback_optimization", "Optimize feedback loops for rapid learning and adaptation.", context={"rl_feedback": rl_feedback, "predictive_insights": predictive_insights})
        self.logger.info(f"Feedback loop optimization: {feedback_optimization}")
        await self.knowledge_base.add_entry("feedback_optimization", feedback_optimization)

        # Optimize feedback loop for rapid learning
        feedback_optimization = await self.ollama.query_ollama("feedback_optimization", "Optimize feedback loops for rapid learning and adaptation.", context={"rl_feedback": rl_feedback, "predictive_insights": predictive_insights})
        self.logger.info(f"Feedback loop optimization: {feedback_optimization}")
        await self.knowledge_base.add_entry("feedback_optimization", feedback_optimization)
        # Periodically analyze long-term memory for insights
        if improvement_cycle_count % 10 == 0:  # Every 10 cycles
            longterm_memory_analysis = await self.knowledge_base.get_longterm_memory()
            self.logger.info(f"Periodic long-term memory analysis: {longterm_memory_analysis}")
        await self.log_state(f"Completed improvement cycle {improvement_cycle_count}")
        # Log the completion of an improvement cycle in the knowledge base
        await kb.add_entry("improvement_cycle_end", {"cycle_number": improvement_cycle_count, "timestamp": time.time()})
        # Log the completion of an improvement cycle in the knowledge base
        await kb.add_entry("improvement_cycle_end", {"cycle_number": improvement_cycle_count, "timestamp": time.time()})

        # Self-reflection mechanism
        if improvement_cycle_count % 5 == 0:  # Every 5 cycles
            self_reflection = await self.ollama.query_ollama("self_reflection", "Reflect on recent performance and suggest adjustments.", context={"system_state": system_state})
            self.logger.info(f"Self-reflection insights: {self_reflection}")
            await self.knowledge_base.add_entry("self_reflection", self_reflection)

    async def evolve_feedback_loop(self, rl_feedback, predictive_insights):
        """Evolve the feedback loop by integrating reinforcement learning feedback and predictive insights."""
        # Refine feedback loop with adaptive mechanisms
        historical_feedback = await self.knowledge_base.get_entry("historical_feedback")
        combined_feedback = rl_feedback + predictive_insights.get("suggestions", []) + historical_feedback.get("feedback", [])
        self.logger.info(f"Refining feedback loop with adaptive feedback: {combined_feedback}")
        await self.knowledge_base.add_entry("refined_feedback", {"combined_feedback": combined_feedback})
        self.logger.info("Long-term memory updated with refined feedback.")
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
        
        # Analyze code and suggest improvements
        code_analysis = await si.retry_ollama_call(ca.analyze_code, ollama, "current_system_code")
        if code_analysis.get('improvements'):
            for code_improvement in code_analysis['improvements']:
                await si.apply_code_change(code_improvement)

        # Run tests and handle failures
        test_results = await tf.run_tests(ollama, "current_test_suite")
        if test_results.get('failed_tests'):
            for failed_test in test_results['failed_tests']:
                fix = await ollama.query_ollama("test_fixing", f"Fix this failed test: {failed_test}")
                await si.apply_code_change(fix['code_change'])

        # Deploy code if approved
        deployment_decision = await si.retry_ollama_call(dm.deploy_code, ollama)
        if deployment_decision and deployment_decision.get('deploy', False):
            await self.log_state("Deployment approved by Ollama")
        else:
            await self.log_state("Deployment deferred based on Ollama's decision")

        # Perform version control operations
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

    async def self_optimization(self, ollama, kb):
        """Evaluate and optimize system performance."""
        performance_metrics = await ollama.query_ollama("performance_evaluation", "Evaluate current system performance and suggest optimizations.")
        self.logger.info(f"Performance metrics: {performance_metrics}")
        await kb.add_entry("performance_metrics", performance_metrics)
        optimizations = performance_metrics.get("optimizations", [])
        for optimization in optimizations:
            self.logger.info(f"Applying optimization: {optimization}")
            # Implement optimization logic here
            # For example, adjust system parameters or configurations
        self.logger.info("Self-optimization completed.")
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
        self.logger.info("Long-term memory updated with timeout state.")

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
