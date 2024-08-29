import unittest
from core.improvement_manager import ImprovementManager
from core.ollama_interface import OllamaInterface

class TestImprovementManager(unittest.TestCase):
    def setUp(self):
        ollama = OllamaInterface()
        self.improvement_manager = ImprovementManager(ollama)

    async def test_suggest_improvements(self):
        improvements = await self.improvement_manager.suggest_improvements({"state": "test"})
        self.assertIsInstance(improvements, list)

    async def test_apply_improvements(self):
        results = await self.improvement_manager.apply_improvements(["test improvement"])
        self.assertIsInstance(results, list)

if __name__ == '__main__':
    unittest.main()
