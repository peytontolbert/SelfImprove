import logging
import os
import json
import asyncio
import time
from neo4j import GraphDatabase

class KnowledgeBase:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="password", ollama_interface=None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.ollama = ollama_interface
        self.longterm_memory = {}

    def close(self):
        self.driver.close()

    def add_node(self, label, properties):
        with self.driver.session() as session:
            session.write_transaction(self._create_node, label, properties)

    @staticmethod
    def _create_node(tx, label, properties):
        query = f"CREATE (n:{label} {{properties}})"
        tx.run(query, properties=properties)

    def add_relationship(self, from_node, to_node, relationship_type, properties=None):
        with self.driver.session() as session:
            session.write_transaction(self._create_relationship, from_node, to_node, relationship_type, properties)

    @staticmethod
    def _create_relationship(tx, from_node, to_node, relationship_type, properties):
        query = (
            f"MATCH (a), (b) WHERE a.name = $from_node AND b.name = $to_node "
            f"CREATE (a)-[r:{relationship_type} {{properties}}]->(b)"
        )
        tx.run(query, from_node=from_node, to_node=to_node, properties=properties or {})

    def add_capability(self, capability_name, properties):
        """Add a new capability node to the graph."""
        with self.driver.session() as session:
            session.write_transaction(self._create_node, "Capability", {"name": capability_name, **properties})

    def add_capability_relationship(self, from_capability, to_capability, relationship_type, properties=None):
        """Add a relationship between two capabilities."""
        with self.driver.session() as session:
            session.write_transaction(self._create_relationship, from_capability, to_capability, relationship_type, properties)

    def get_capability_evolution(self, capability_name):
        """Retrieve the evolution of a specific capability."""
        with self.driver.session() as session:
            result = session.read_transaction(self._find_capability_evolution, capability_name)
            return result

    @staticmethod
    def _find_capability_evolution(tx, capability_name):
        query = (
            "MATCH (c:Capability {name: $capability_name})-[r]->(next:Capability) "
            "RETURN c.name AS current, r, next.name AS next"
        )
        result = tx.run(query, capability_name=capability_name)
        return [{"current": record["current"], "relationship": record["r"], "next": record["next"]} for record in result]
    async def add_entry(self, entry_name, data, metadata=None, narrative_context=None):
        decision = await self.ollama.query_ollama(self.ollama.system_prompt, f"Should I add this entry: {entry_name} with data: {data}", task="knowledge_base")
        if decision.get('add_entry', False):
            properties = {
                "data": data,
                "metadata": metadata or {},
                "narrative_context": narrative_context or {},
                "timestamp": time.time()
            }
            self.add_node(entry_name, properties)
            self.logger.info(f"Entry added: {entry_name} with metadata: {metadata} and narrative context: {narrative_context}")
            return True
        self.logger.info(f"Entry addition declined: {entry_name}")
        return False

    async def get_entry(self, entry_name, include_metadata=False):
        with self.driver.session() as session:
            result = session.read_transaction(self._find_node, entry_name)
            if result:
                data = result.get("data")
                metadata = result.get("metadata", {})
                interpretation = await self.ollama.query_ollama(self.ollama.system_prompt, f"Interpret this data: {data}", task="knowledge_base")
                interpretation_result = interpretation.get('interpretation', data)
                self.logger.info(f"Entry retrieved: {entry_name} | Interpretation: {interpretation_result}")
                if include_metadata:
                    return {"data": interpretation_result, "metadata": metadata}
                return interpretation_result
            else:
                return None

    @staticmethod
    def _find_node(tx, entry_name):
        query = f"MATCH (n) WHERE n.name = $entry_name RETURN n"
        result = tx.run(query, entry_name=entry_name)
        return result.single()[0] if result.single() else None

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

    async def analyze_capability_evolution(self):
        """Analyze the evolution of system capabilities."""
        capabilities = await self.list_entries()
        analysis = []
        for capability in capabilities:
            evolution = self.get_capability_evolution(capability)
            analysis.append({"capability": capability, "evolution": evolution})
        self.logger.info(f"Capability evolution analysis: {analysis}")
        return analysis
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
