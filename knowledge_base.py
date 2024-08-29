import logging
import os
import json
import asyncio
from core.ollama_interface import OllamaInterface

class KnowledgeBase:
    def __init__(self, base_directory="knowledge_base", ollama_interface=None):
        self.base_directory = base_directory
        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory)
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.ollama = ollama_interface or OllamaInterface()

    async def add_entry(self, entry_name, data, version=None):
        decision = await self.ollama.query_ollama("knowledge_base", f"Should I add this entry: {entry_name} with data: {data}")
        if decision.get('add_entry', False):
            version = version or self.get_next_version(entry_name)
            file_path = os.path.join(self.base_directory, f"{entry_name}_v{version}.json")
            with open(file_path, 'w') as file:
                json.dump(data, file)
            self.logger.info(f"Entry added: {entry_name} | Version: {version}")
            return True
        self.logger.info(f"Entry addition declined: {entry_name}")
        return False

    async def get_entry(self, entry_name):
        version = version or self.get_latest_version(entry_name)
        file_path = os.path.join(self.base_directory, f"{entry_name}_v{version}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                data = json.load(file)
            interpretation = await self.ollama.query_ollama("knowledge_base", f"Interpret this data: {data}")
            interpretation_result = interpretation.get('interpretation', data)
            self.logger.info(f"Entry retrieved: {entry_name} | Interpretation: {interpretation_result}")
            return interpretation_result
        else:
            return None

    async def update_entry(self, entry_name, data):
        decision = await self.ollama.query_ollama("knowledge_base", f"Should I update this entry: {entry_name} with data: {data}")
        if decision.get('update_entry', False):
            update_result = await self.add_entry(entry_name, data)
            self.logger.info(f"Entry updated: {entry_name}")
            return update_result
        return False

    async def list_entries(self, include_versions=False):
        entries = [f.split('_v')[0] for f in os.listdir(self.base_directory) if f.endswith('.json')]
        if include_versions:
            entries = [f.replace('.json', '') for f in os.listdir(self.base_directory) if f.endswith('.json')]
        categorization = await self.ollama.query_ollama("knowledge_base", f"Categorize these entries: {entries}")
        categorized_entries = categorization.get('categorized_entries', entries)
        self.logger.info(f"Entries listed: {categorized_entries}")
        return categorized_entries

    async def search_entries(self, keyword):
        """Search for entries containing the given keyword."""
        matching_entries = []
        for entry in os.listdir(self.base_directory):
            if entry.endswith('.json'):
                with open(os.path.join(self.base_directory, entry), 'r') as file:
                    data = json.load(file)
                    if keyword.lower() in json.dumps(data).lower():
                        matching_entries.append(entry)
        self.logger.info(f"Entries matching '{keyword}': {matching_entries}")
        return matching_entries
        entries = await self.list_entries()
        analysis = await self.ollama.query_ollama("knowledge_base", f"Analyze the current state of the knowledge base with these entries: {entries}")
        analysis_result = analysis.get('analysis', "No analysis available")
        self.logger.info(f"Knowledge base analysis: {analysis_result}")
        return analysis_result

    async def get_next_version(self, entry_name):
        """Get the next version number for an entry."""
        existing_versions = [
            int(f.split('_v')[-1].split('.json')[0])
            for f in os.listdir(self.base_directory)
            if f.startswith(entry_name)
        ]
        return max(existing_versions, default=0) + 1

    async def get_latest_version(self, entry_name):
        """Get the latest version number for an entry."""
        existing_versions = [
            int(f.split('_v')[-1].split('.json')[0])
            for f in os.listdir(self.base_directory)
            if f.startswith(entry_name)
        ]
        return max(existing_versions, default=0)
        analysis = await self.analyze_knowledge_base()
        suggestions = await self.ollama.query_ollama("knowledge_base", f"Suggest improvements based on this analysis: {analysis}")
        improvement_suggestions = suggestions.get('improvements', [])
        self.logger.info(f"Improvement suggestions: {improvement_suggestions}")
        return improvement_suggestions

    async def apply_improvement(self, improvement):
        implementation = await self.ollama.implement_improvement(improvement)
        if implementation.get('knowledge_base_update'):
            await self.add_entry(f"improvement_{len(self.list_entries()) + 1}", implementation['knowledge_base_update'])
            self.logger.info(f"Improvement applied: {improvement}")
            return f"Applied improvement: {improvement}"
        self.logger.info(f"No knowledge base update for improvement: {improvement}")
        return f"No knowledge base update for improvement: {improvement}"
