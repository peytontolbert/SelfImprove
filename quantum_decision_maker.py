import logging
from typing import Dict, Any, List
from main_narrative_control import SimpleNN
import torch
from core.ollama_interface import OllamaInterface

class QuantumDecisionMaker:
    def __init__(self, ollama_interface: OllamaInterface):
        self.logger = logging.getLogger(__name__)
        self.ollama = ollama_interface

    async def evaluate_possibilities(self, action, system_state, feedback) -> List[Dict[str, Any]]:
        """
        Evaluate multiple possibilities for a given action using quantum-inspired logic.

        Parameters:
        - action: The action to evaluate.
        - system_state: Current state of the system.
        - feedback: Feedback data to consider.

        Returns:
        - A list of possible outcomes with their scores.
        """
        try:
            # Prepare input data for the model
            input_data = torch.tensor([system_state.get('metric', 0), feedback.get('metric', 0)], dtype=torch.float32)
            with torch.no_grad():
                # Use the model to predict a score
                predicted_score = self.model(input_data).item()

            possible_outcomes = [
                {"action": action, "score": await self.calculate_score(action, system_state, feedback, variation) + predicted_score}
                for variation in range(5)
            ]
            self.logger.info(f"Evaluated possibilities for action '{action}': {possible_outcomes}")
            return possible_outcomes
        except Exception as e:
            self.logger.error(f"Error evaluating possibilities for action '{action}': {e}")
            return []

    async def quantum_decision_tree(self, decision_space: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use a quantum-inspired decision tree to make complex decisions.

        Parameters:
        - decision_space: A dictionary containing possible decisions and their contexts.

        Returns:
        - The optimal decision based on quantum evaluation.
        """
        self.logger.info("Building quantum decision tree.")
        # Example logic: Evaluate decisions based on a combination of factors
        optimal_decision = max(decision_space, key=lambda decision: decision.get("score", 0))
        self.logger.info(f"Optimal decision made: {optimal_decision}")
        return optimal_decision

    async def calculate_score(self, action, system_state, feedback, variation) -> int:
        """
        Calculate a score for a given action variation.

        Parameters:
        - action: The action to evaluate.
        - system_state: Current state of the system.
        - feedback: Feedback data to consider.
        - variation: The variation of the action to evaluate.

        Returns:
        - An integer score representing the evaluation.
        """
        # Example scoring logic based on variation
        base_score = feedback.get(action, {}).get("base_score", 1)
        # Integrate insights from OllamaInterface
        insights = await self.ollama.query_ollama("decision_insights", f"Provide insights for action: {action}", context={"system_state": system_state})
        insight_score = insights.get("insight_score", 0)
        score = base_score + variation + insight_score  # Enhanced scoring logic
        self.logger.debug(f"Calculated score for action '{action}' variation {variation}: {score}")
        return score
