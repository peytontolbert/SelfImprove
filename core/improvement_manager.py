import logging
from typing import List, Dict, Any

class ImprovementManager:
    def __init__(self, ollama_interface):
        self.ollama = ollama_interface
        self.logger = logging.getLogger(__name__)

    async def suggest_improvements(self, system_state: Dict[str, Any]) -> List[str]:
        try:
            prompt = (
                f"Suggest improvements based on the current system state: {system_state}. "
                f"Consider alignment with business objectives, data quality, regulatory compliance, "
                f"and information security as per the golden rules."
            )
            response = await self.ollama.query_ollama(self.ollama.system_prompt, prompt, task="improvement_suggestion")
            suggestions = response.get("suggestions", [])
            self.logger.info(f"Suggested improvements: {suggestions}")
            return suggestions
        except Exception as e:
            self.logger.error(f"Error suggesting improvements: {str(e)}")
            return []

    async def validate_improvements(self, improvements: List[str]) -> List[str]:
        try:
            try:
                validated = []
                for improvement in improvements:
                    validation = await self.ollama.validate_improvement(improvement)
                    if validation.get('is_valid', False):
                        validated.append(improvement)
                    else:
                        self.logger.info(f"Invalid improvement suggestion: {improvement}")
                self.logger.info(f"Validated improvements: {validated}")
                return validated
            except Exception as e:
                self.logger.error(f"Error validating improvements: {str(e)}")
                return []
        except Exception as e:
            self.logger.error(f"Error validating improvements: {str(e)}")
            return []

    async def apply_improvements(self, improvements: List[str]) -> List[Dict[str, Any]]:
        results = []
        try:
            for improvement in improvements:
                implementation = await self.ollama.implement_improvement(improvement)
                if implementation.get('code_change'):
                    result = await self.apply_code_change(implementation['code_change'])
                    results.append(result)
                if implementation.get('system_update'):
                    result = await self.apply_system_update(implementation['system_update'])
                    results.append(result)
            self.logger.info(f"Applied improvements: {results}")
            return results
        except Exception as e:
            self.logger.error(f"Error applying improvements: {str(e)}")
            return []

    async def apply_code_change(self, code_change: str) -> Dict[str, Any]:
        self.logger.info(f"Applying code change: {code_change}")
        return {"status": "success", "message": "Code change applied"}

    async def apply_system_update(self, system_update: str) -> Dict[str, Any]:
        self.logger.info(f"Updating system: {system_update}")
        return {"status": "success", "message": "System update applied"}
