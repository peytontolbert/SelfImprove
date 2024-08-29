import unittest
from unittest.mock import AsyncMock
from core.ollama_interface import OllamaInterface

class TestOllamaInterface(unittest.TestCase):
    def setUp(self):
        self.ollama = OllamaInterface()

    def setUp(self):
        self.ollama = OllamaInterface()
        self.ollama.query_ollama = AsyncMock(return_value={"response": "mocked response"})

    async def test_query_ollama(self):
        response = await self.ollama.query_ollama("test_prompt", "Test query")
        self.ollama.query_ollama.assert_called_once_with("test_prompt", "Test query", None, True)
        self.assertEqual(response, {"response": "mocked response"})

    async def test_refine_prompt(self):
        refined_prompt = await self.ollama.refine_prompt("Test prompt", "Test task")
        self.assertIsInstance(refined_prompt, str)

if __name__ == '__main__':
    unittest.main()
