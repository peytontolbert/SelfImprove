import logging

class SystemNarrative:
    def __init__(self):
        self.logger = logging.getLogger("SystemNarrative")
        logging.basicConfig(level=logging.INFO)

    def log_state(self, message):
        self.logger.info(f"System State: {message}")

    def log_decision(self, decision):
        self.logger.info(f"System Decision: {decision}")

    def log_error(self, error):
        self.logger.error(f"System Error: {error}")

    def log_recovery(self, recovery_action):
        self.logger.info(f"Recovery Action: {recovery_action}")
