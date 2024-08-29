import unittest
from core.ollama_interface import OllamaInterface

class TestOllamaInterface(unittest.TestCase):
    def setUp(self):
        self.ollama = OllamaInterface()

    async def test_query_ollama(self):
        response = await self.ollama.query_ollama("test_prompt", "Test query")
        self.assertIsInstance(response, dict)

    async def test_refine_prompt(self):
        refined_prompt = await self.ollama.refine_prompt("Test prompt", "Test task")
        self.assertIsInstance(refined_prompt, str)

if __name__ == '__main__':
    unittest.main()
