import logging
from typing import Dict, Any
from core.ollama_interface import OllamaInterface
from knowledge_base import KnowledgeBase

class MetaLearner:
    def __init__(self, ollama: OllamaInterface, knowledge_base: KnowledgeBase):
        self.ollama = ollama
        self.knowledge_base = knowledge_base
        self.logger = logging.getLogger(__name__)

    async def optimize_learning_strategies(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Optimizing learning strategies based on performance data.")
        strategies = await self.ollama.query_ollama(
            "meta_learning",
            "Optimize learning strategies based on performance data",
            context={"performance_data": performance_data}
        )
        self.logger.info(f"Optimized strategies: {strategies}")
        await self.knowledge_base.add_entry("optimized_strategies", strategies)
        return strategies

    async def evolve_longterm_strategies(self, historical_data: Dict[str, Any]) -> Dict[str, Any]:
        self.logger.info("Evolving long-term strategies using historical data.")
        evolution_suggestions = await self.ollama.query_ollama(
            "longterm_evolution",
            "Suggest long-term evolution strategies based on historical data",
            context={"historical_data": historical_data}
        )
        self.logger.info(f"Evolution suggestions: {evolution_suggestions}")
        await self.knowledge_base.add_entry("longterm_evolution_suggestions", evolution_suggestions)
        return evolution_suggestions

    async def integrate_with_self_improvement(self, self_improvement):
        self.logger.info("Integrating MetaLearner with SelfImprovement module.")
        performance_data = await self_improvement.get_system_metrics()
        strategies = await self.optimize_learning_strategies(performance_data)
        await self_improvement.apply_improvements(strategies)
        self.logger.info("Integration with SelfImprovement completed.")
