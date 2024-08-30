class MetaLearner:
    async def optimize_learning_strategies(self, ollama, performance_data):
        strategies = await ollama.query_ollama(
            "meta_learning",
            "Optimize learning strategies based on performance data",
            context={"performance_data": performance_data}
        )
        return strategies
