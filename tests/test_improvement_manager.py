import unittest
from unittest.mock import AsyncMock
from core.improvement_manager import ImprovementManager
from core.ollama_interface import OllamaInterface

class TestImprovementManager(unittest.TestCase):
    def setUp(self):
        ollama = OllamaInterface()
        ollama.query_ollama = AsyncMock(return_value={"suggestions": ["mocked improvement"]})
        ollama.validate_improvement = AsyncMock(return_value={"is_valid": True})
        ollama.implement_improvement = AsyncMock(return_value={"code_change": "mocked code change"})
        self.improvement_manager = ImprovementManager(ollama)

    async def test_suggest_improvements(self):
        improvements = await self.improvement_manager.suggest_improvements({"state": "test"})
        self.assertEqual(improvements, ["mocked improvement"])

    async def test_apply_improvements(self):
        results = await self.improvement_manager.apply_improvements(["test improvement"])
        self.assertEqual(results, [{"status": "success", "message": "Code change applied"}])

if __name__ == '__main__':
    unittest.main()
