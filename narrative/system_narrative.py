import logging

class SystemNarrative:
    def __init__(self):
        self.logger = logging.getLogger("SystemNarrative")
        logging.basicConfig(level=logging.INFO)

    def log_state(self, message, context=None):
        if context:
            self.logger.info(f"System State: {message} | Context: {context}")
        else:
            self.logger.info(f"System State: {message}")
        self.logger.info(f"System State: {message}")

    def log_decision(self, decision, rationale=None):
        if rationale:
            self.logger.info(f"System Decision: {decision} | Rationale: {rationale}")
        else:
            self.logger.info(f"System Decision: {decision}")
        self.logger.info(f"System Decision: {decision}")

    def log_error(self, error, context=None):
        if context:
            self.logger.error(f"System Error: {error} | Context: {context}")
        else:
            self.logger.error(f"System Error: {error}")
        self.logger.error(f"System Error: {error}")

    def log_recovery(self, recovery_action, success=True):
        status = "successful" if success else "failed"
        self.logger.info(f"Recovery Action: {recovery_action} | Status: {status}")
        self.logger.info(f"Recovery Action: {recovery_action}")
