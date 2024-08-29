import os
import json
import asyncio
from core.ollama_interface import OllamaInterface

class KnowledgeBase:
    def __init__(self, base_directory="knowledge_base", ollama_interface=None):
        self.base_directory = base_directory
        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory)
        self.ollama = ollama_interface or OllamaInterface()

    async def add_entry(self, entry_name, data):
        decision = await self.ollama.query_ollama("knowledge_base", f"Should I add this entry: {entry_name} with data: {data}")
        if decision.get('add_entry', False):
            file_path = os.path.join(self.base_directory, f"{entry_name}.json")
            with open(file_path, 'w') as file:
                json.dump(data, file)
            return True
        return False

    async def get_entry(self, entry_name):
        file_path = os.path.join(self.base_directory, f"{entry_name}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
            interpretation = await self.ollama.query_ollama("knowledge_base", f"Interpret this data: {data}")
            return interpretation.get('interpretation', data)
        else:
            return None

    async def update_entry(self, entry_name, data):
        decision = await self.ollama.query_ollama("knowledge_base", f"Should I update this entry: {entry_name} with data: {data}")
        if decision.get('update_entry', False):
            return await self.add_entry(entry_name, data)
        return False

    async def list_entries(self):
        entries = [f.split('.')[0] for f in os.listdir(self.base_directory) if f.endswith('.json')]
        categorization = await self.ollama.query_ollama("knowledge_base", f"Categorize these entries: {entries}")
        return categorization.get('categorized_entries', entries)

    async def analyze_knowledge_base(self):
        entries = await self.list_entries()
        analysis = await self.ollama.query_ollama("knowledge_base", f"Analyze the current state of the knowledge base with these entries: {entries}")
        return analysis.get('analysis', "No analysis available")

    async def suggest_improvements(self):
        analysis = await self.analyze_knowledge_base()
        suggestions = await self.ollama.query_ollama("knowledge_base", f"Suggest improvements based on this analysis: {analysis}")
        return suggestions.get('improvements', [])

    async def apply_improvement(self, improvement):
        implementation = await self.ollama.implement_improvement(improvement)
        if implementation.get('knowledge_base_update'):
            await self.add_entry(f"improvement_{len(self.list_entries()) + 1}", implementation['knowledge_base_update'])
            return f"Applied improvement: {improvement}"
        return f"No knowledge base update for improvement: {improvement}"
