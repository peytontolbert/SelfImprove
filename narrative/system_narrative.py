import logging
from core.ollama_interface import OllamaInterface
import asyncio

class SystemNarrative:
    def __init__(self, ollama_interface=None):
        self.logger = logging.getLogger("SystemNarrative")
        self.ollama = ollama_interface or OllamaInterface()
        logging.basicConfig(level=logging.INFO)

    async def log_with_ollama(self, message, context=None):
        """Log a message with insights from Ollama."""
        prompt = f"Log this message with context: {message}"
        if context:
            prompt += f" | Context: {context}"
        ollama_response = await self.ollama.query_ollama("logging", prompt)
        self.logger.info(f"Ollama Insight: {ollama_response.get('insight', 'No insight provided')}")

    def log_state(self, message, context=None):
        if context:
            self.logger.info(f"System State: {message} | Context: {context}")
        else:
            self.logger.info(f"System State: {message}")
        asyncio.run(self.log_with_ollama(message, context))

    def log_decision(self, decision, rationale=None):
        if rationale:
            self.logger.info(f"System Decision: {decision} | Rationale: {rationale}")
        else:
            self.logger.info(f"System Decision: {decision}")
        asyncio.run(self.log_with_ollama(decision, rationale))

    def log_error(self, error, context=None):
        if context:
            self.logger.error(f"System Error: {error} | Context: {context}")
        else:
            self.logger.error(f"System Error: {error}")
        asyncio.run(self.log_with_ollama(error, context))

    def log_recovery(self, recovery_action, success=True):
        status = "successful" if success else "failed"
        self.logger.info(f"Recovery Action: {recovery_action} | Status: {status}")
        asyncio.run(self.log_with_ollama(recovery_action, {"success": success}))
