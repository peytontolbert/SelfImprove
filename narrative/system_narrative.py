import logging
from core.ollama_interface import OllamaInterface
import asyncio

class SystemNarrative:
    def __init__(self, ollama_interface=None):
        self.logger = logging.getLogger("SystemNarrative")
        self.ollama = ollama_interface or OllamaInterface()
        logging.basicConfig(level=logging.INFO)

    async def generate_thoughts(self, context=None):
        """Generate detailed thoughts or insights about the current state and tasks."""
        prompt = "Generate detailed thoughts about the current system state, tasks, and potential improvements."
        if context:
            prompt += f" | Context: {context}"
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

    async def log_recovery(self, recovery_action, success=True):
        status = "successful" if success else "failed"
        self.logger.info(f"Recovery Action: {recovery_action} | Status: {status}")
        await self.log_with_ollama(recovery_action, {"success": success})
