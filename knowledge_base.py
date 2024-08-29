import logging
import os
import json
import asyncio
import time
class KnowledgeBase:
    def __init__(self, base_directory="knowledge_base", ollama_interface=None):
        self.base_directory = base_directory
        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory)
        self.longterm_memory = {}
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.ollama = ollama_interface

    async def add_entry(self, entry_name, data, metadata=None, narrative_context=None):
        decision = await self.ollama.query_ollama(self.ollama.system_prompt, f"Should I add this entry: {entry_name} with data: {data}", task="knowledge_base")
        if decision.get('add_entry', False):
            entry_data = {
                "data": data,
                "metadata": metadata or {},
                "narrative_context": narrative_context or {},
                "timestamp": time.time()
            }
            file_path = os.path.join(self.base_directory, f"{entry_name}.json")
            with open(file_path, 'w') as file:
                json.dump(entry_data, file)
            self.logger.info(f"Entry added: {entry_name} with metadata: {metadata} and narrative context: {narrative_context}")
            await self.save_longterm_memory({entry_name: entry_data})
            return True
        self.logger.info(f"Entry addition declined: {entry_name}")
        return False

    async def get_entry(self, entry_name, include_metadata=False):
        file_path = os.path.join(self.base_directory, f"{entry_name}.json")
        if os.path.exists(file_path):
            with open(file_path, 'r') as file:
                entry_data = json.load(file)
            data = entry_data.get("data")
            metadata = entry_data.get("metadata", {})
            interpretation = await self.ollama.query_ollama(self.ollama.system_prompt, f"Interpret this data: {data}", task="knowledge_base")
            interpretation_result = interpretation.get('interpretation', data)
            self.logger.info(f"Entry retrieved: {entry_name} | Interpretation: {interpretation_result}")
            if include_metadata:
                return {"data": interpretation_result, "metadata": metadata}
            return interpretation_result
        else:
            return None

    async def update_entry(self, entry_name, data):
        decision = await self.ollama.query_ollama(self.ollama.system_prompt, f"Should I update this entry: {entry_name} with data: {data}", task="knowledge_base")
        if decision.get('update_entry', False):
            update_result = await self.add_entry(entry_name, data)
            self.logger.info(f"Entry updated: {entry_name}")
            if update_result:
                await self.save_longterm_memory({entry_name: data})
            return update_result
        return False

    async def list_entries(self):
        entries = [f.split('.')[0] for f in os.listdir(self.base_directory) if f.endswith('.json')]
        categorization = await self.ollama.query_ollama(self.ollama.system_prompt, f"Categorize these entries: {entries}", task="knowledge_base")
        categorized_entries = categorization.get('categorized_entries', entries)
        self.logger.info(f"Entries listed: {categorized_entries}")
        return categorized_entries

    async def get_longterm_memory(self):
        """Retrieve long-term memory entries."""
        if not self.longterm_memory:
            entries = await self.list_entries()
            for entry in entries:
                if isinstance(entry, dict):
                    entry_name = entry.get('entry', str(entry))
                else:
                    entry_name = str(entry)
                data = await self.get_entry(entry_name)
                self.longterm_memory[entry_name] = data
            self.logger.info(f"Retrieved long-term memory: {self.longterm_memory}")
            await self.save_longterm_memory(self.longterm_memory)
        return self.longterm_memory

    async def save_longterm_memory(self, longterm_memory):
        """Save long-term memory to a file."""
        self.longterm_memory.update({str(k): v for k, v in longterm_memory.items()})
        file_path = os.path.join(self.base_directory, "longterm_memory.json")
        with open(file_path, 'w') as file:
            json.dump(self.longterm_memory, file)
        self.logger.info("Long-term memory updated and saved to file.")
        entries = await self.list_entries()
        analysis = await self.ollama.query_ollama(self.ollama.system_prompt, f"Analyze the current state of the knowledge base with these entries: {entries}", task="knowledge_base")
        analysis_result = analysis.get('analysis', "No analysis available")
        self.logger.info(f"Knowledge base analysis: {analysis_result}")
        return analysis_result

    async def suggest_improvements(self):
        analysis = await self.analyze_knowledge_base()
        suggestions = await self.ollama.query_ollama(self.ollama.system_prompt, f"Suggest improvements based on this analysis: {analysis}", task="knowledge_base")
        improvement_suggestions = suggestions.get('improvements', [])
        self.logger.info(f"Improvement suggestions: {improvement_suggestions}")
        # Automatically apply suggested improvements
        for improvement in improvement_suggestions:
            await self.apply_improvement(improvement)
        return improvement_suggestions

    async def apply_improvement(self, improvement):
        implementation = await self.ollama.query_ollama(self.ollama.system_prompt, f"Implement this improvement: {improvement}", task="improvement_implementation")
        if implementation.get('knowledge_base_update'):
            await self.add_entry(f"improvement_{len(self.list_entries()) + 1}", implementation['knowledge_base_update'])
            self.logger.info(f"Improvement applied: {improvement}")
            return f"Applied improvement: {improvement}"
        self.logger.info(f"No knowledge base update for improvement: {improvement}")
        return f"No knowledge base update for improvement: {improvement}"
