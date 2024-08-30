import logging
from typing import List, Dict, Any
import os
import json
import asyncio
import time
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

class KnowledgeBase:
    def __init__(self, uri=None, user=None, password=None, ollama_interface=None):
        uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        user = user or os.getenv("NEO4J_USER", "neo4j")
        password = password or os.getenv("NEO4J_PASSWORD", "12345678")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        self.check_connection()
        self.initialize_database()
        self.initialize_database()
        self.ollama = ollama_interface
        self.longterm_memory = {}
        self.base_directory = "knowledge_base_data"
        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory)

    def check_connection(self):
        """Check if the connection to the Neo4j database can be established."""
        try:
            with self.driver.session() as session:
                session.run("RETURN 1")
            self.logger.info("Successfully connected to the Neo4j database.")
        except Exception as e:
            self.logger.error(f"Failed to connect to the Neo4j database: {str(e)}")
            raise ConnectionError("Could not connect to the Neo4j database. Please ensure it is running and accessible.")
        """Initialize the database with necessary nodes and relationships."""
        self.logger.info("Starting database initialization.")
        try:
            with self.driver.session() as session:
                session.write_transaction(self._create_initial_nodes)
            self.logger.info("Database initialized successfully.")
            self.logger.info("Initial nodes and constraints created.")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            self.logger.info("Attempting to create a new database.")
            self.create_database()
            self.logger.info("New database creation attempted.")

    def initialize_database(self):
        """Initialize the database with necessary nodes and relationships."""
        self.logger.info("Starting database initialization.")
        try:
            with self.driver.session() as session:
                session.write_transaction(self._create_initial_nodes)
            self.logger.info("Database initialized successfully.")
            self.logger.info("Initial nodes and constraints created.")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {str(e)}")
            self.logger.info("Attempting to create a new database.")
            self.create_database()
            self.logger.info("New database creation attempted.")

    def create_database(self):
        """Create a new database if it doesn't exist."""
        with self.driver.session() as session:
            session.write_transaction(self._create_initial_nodes)
        self.logger.info("New database created and initialized successfully.")

    @staticmethod
    def _create_initial_nodes(tx):
        # Create initial nodes or constraints if needed
        tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Node) REQUIRE n.name IS UNIQUE")

    def add_node(self, label, properties):
        """Add a node with detailed properties to the graph."""
        with self.driver.session() as session:
            session.write_transaction(self._create_node, label, properties)

    @staticmethod
    def _create_node(tx, label, properties):
        # Construct the query with explicit property assignments
        property_assignments = ", ".join([f"n.{k} = ${k}" for k in properties.keys()])
        query = f"MERGE (n:{label}) SET {property_assignments}"
        sanitized_properties = {k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in properties.items()}
        tx.run(query, **sanitized_properties)

    def add_relationship(self, from_node, to_node, relationship_type, properties=None):
        """Add or update a relationship with properties between nodes."""
        with self.driver.session() as session:
            session.write_transaction(self._create_relationship, from_node, to_node, relationship_type, properties)

    @staticmethod
    def _create_relationship(tx, from_node, to_node, relationship_type, properties):
        query = (
            f"MATCH (a), (b) WHERE a.name = $from_node AND b.name = $to_node "
            f"MERGE (a)-[r:{relationship_type}]->(b) "
            f"ON CREATE SET r = $properties "
            f"ON MATCH SET r += $properties"
        )
        tx.run(query, from_node=from_node, to_node=to_node, properties=properties or {})

    async def add_capability(self, capability_name, properties):
        """Add a new capability node to the graph."""
        with self.driver.session() as session:
            session.write_transaction(self._create_node, "Capability", {"name": capability_name, **properties})

    def add_capability_relationship(self, from_capability, to_capability, relationship_type, properties=None):
        """Add a relationship between two capabilities."""
        with self.driver.session() as session:
            session.write_transaction(self._create_relationship, from_capability, to_capability, relationship_type, properties)

    def query_insights(self, query):
        """Execute a custom query to retrieve insights from the graph database."""
        with self.driver.session() as session:
            result = session.run(query)
            return [record.data() for record in result]

    def get_capability_evolution(self, capability_name):
        """Retrieve the evolution of a specific capability."""
        query = (
            "MATCH (c:Capability {name: $capability_name})-[r]->(next:Capability) "
            "RETURN c.name AS current, r, next.name AS next"
        )
        return self.query_insights(query)
    async def add_entry(self, entry_name, data, metadata=None, narrative_context=None, context=None):
        if context:
            data.update({"context": context})

        # Evaluate the relevance of the data
        relevance_decision = await self.evaluate_relevance(entry_name, data)
        if not relevance_decision.get('is_relevant', False):
            self.logger.info(f"Entry deemed irrelevant and not added: {entry_name}")
            return False

        decision = await self.ollama.query_ollama(
            self.ollama.system_prompt,
            f"Should I add this entry: {entry_name} with data: {data}",
            task="knowledge_base"
        )

        if decision.get('add_entry', False):
            properties = {
                "data": data,
                "metadata": metadata or {},
                "narrative_context": narrative_context or {},
                "timestamp": time.time()
            }
            self.add_node(entry_name, properties)
            self.logger.info(f"Entry added: {entry_name} with metadata: {metadata} and narrative context: {narrative_context}")
            # Log the addition of new capabilities for future analysis
            await self.log_capability_evolution(entry_name, properties)
            return True

        self.logger.info(f"Entry addition declined: {entry_name}")
        return False

    async def evaluate_relevance(self, entry_name, data):
        """Evaluate the relevance of the data before adding it to long-term memory."""
        prompt = f"Evaluate the relevance of this entry: {entry_name} with data: {data}"
        context = {"entry_name": entry_name, "data": data}
        relevance_decision = await self.ollama.query_ollama(self.ollama.system_prompt, prompt, task="relevance_evaluation", context=context)
        self.logger.info(f"Relevance evaluation for {entry_name}: {relevance_decision}")
        return relevance_decision

    async def integrate_cross_disciplinary_knowledge(self, disciplines: List[str], knowledge: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate knowledge across multiple disciplines."""
        prompt = f"Integrate knowledge from these disciplines: {disciplines} with the following data: {json.dumps(knowledge)}"
        context = {"disciplines": disciplines, "knowledge": knowledge}
        integration_result = await self.ollama.query_ollama("cross_disciplinary_integration", prompt, context=context)
        self.logger.info(f"Cross-disciplinary integration result: {integration_result}")
        return integration_result

    async def get_entry(self, entry_name, include_metadata=False, context=None):
        if context:
            self.logger.info(f"Retrieving entry with context: {context}")
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
        single_result = result.single()
        return single_result[0] if single_result else None

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

    def summarize_memory(self, memory, max_length=10000):
        """Summarize or trim the memory to fit within the context limits."""
        def extract_key_insights(data):
            """Extract key insights from the data."""
            if isinstance(data, dict):
                return {k: extract_key_insights(v) for k, v in data.items() if v}
            elif isinstance(data, list):
                return [extract_key_insights(item) for item in data]
            else:
                return str(data)[:5000]  # Limit individual entries to 5000 characters

        memory_str = json.dumps(extract_key_insights(memory))
        if len(memory_str) > max_length:
            # Implement a more sophisticated summarization strategy
            return memory_str[:max_length] + "..."
        return memory_str

    async def get_longterm_memory(self):
        """Retrieve long-term memory entries."""
        if not self.longterm_memory:
            entries = [f.split('.')[0] for f in os.listdir(self.base_directory) if f.endswith('.json')]
            for entry_name in entries:
                file_path = os.path.join(self.base_directory, f"{entry_name}.json")
                with open(file_path, 'r') as file:
                    data = json.load(file)
                self.longterm_memory[entry_name] = data
            if self.longterm_memory:
                self.logger.info(f"Retrieved long-term memory: {json.dumps(self.longterm_memory, indent=2)}")
            else:
                self.logger.warning("No long-term memory entries found.")
        return self.longterm_memory

    async def save_longterm_memory(self, longterm_memory):
        """Save long-term memory to a file."""
        # Ensure long-term memory is updated with new data
        self.longterm_memory.update({str(k): v for k, v in longterm_memory.items() if v})
        for entry_name, data in longterm_memory.items():
            self.add_node("LongTermMemory", {"name": entry_name, "data": data})
        file_path = os.path.join(self.base_directory, "longterm_memory.json")
        with open(file_path, 'w') as file:
            json.dump(self.longterm_memory, file)
        self.logger.info("Long-term memory updated and saved to file.")
        entries = await self.list_entries()
        analysis = await self.ollama.query_ollama(self.ollama.system_prompt, f"Analyze the current state of the knowledge base with these entries: {entries}", task="knowledge_base")
        analysis_result = analysis.get('analysis', "No analysis available")
        self.logger.info(f"Knowledge base analysis: {analysis_result}")
        return analysis_result


    async def log_interaction(self, source, action, details, improvement):
        """Log interactions with the knowledge base."""
        self.logger.info(f"Interaction logged from {source}: {action} with details: {details}")
        implementation = await self.ollama.query_ollama(self.ollama.system_prompt, f"Implement this improvement: {improvement}", task="improvement_implementation")
        if implementation.get('knowledge_base_update'):
            await self.add_entry(f"improvement_{len(self.list_entries()) + 1}", implementation['knowledge_base_update'])
            self.logger.info(f"Improvement applied: {improvement}")
            return f"Applied improvement: {improvement}"
        self.logger.info(f"No knowledge base update for improvement: {improvement}")
        return f"No knowledge base update for improvement: {improvement}"
