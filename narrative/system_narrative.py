import logging
from core.ollama_interface import OllamaInterface
import asyncio

class SystemNarrative:
    def __init__(self, ollama_interface=None):
        self.logger = logging.getLogger("SystemNarrative")
        self.ollama = ollama_interface or OllamaInterface()
        logging.basicConfig(level=logging.INFO)

    async def generate_thoughts(self, context=None):
        """Generate thoughts or insights about the current state and tasks."""
        prompt = "Generate thoughts about the current system state and tasks."
        if context:
            prompt += f" | Context: {context}"
        ollama_response = await self.ollama.query_ollama("thought_generation", prompt)
        thoughts = ollama_response.get('thoughts', 'No thoughts generated')
        self.logger.info(f"Ollama Thoughts: {thoughts}")
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
        if rationale:
            self.logger.info(f"System Decision: {decision} | Rationale: {rationale}")
        else:
            self.logger.info(f"System Decision: {decision}")
        await self.log_with_ollama(decision, rationale)

    async def log_error(self, error, context=None):
        if context:
            self.logger.error(f"System Error: {error} | Context: {context}")
        else:
            self.logger.error(f"System Error: {error}")
        await self.log_with_ollama(error, context)

    async def log_recovery(self, recovery_action, success=True):
        status = "successful" if success else "failed"
        self.logger.info(f"Recovery Action: {recovery_action} | Status: {status}")
        await self.log_with_ollama(recovery_action, {"success": success})
