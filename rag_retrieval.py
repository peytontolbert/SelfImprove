import logging
from typing import List, Dict, Any
from knowledge_base import KnowledgeBase

class RAGRetrieval:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def __init__(self, knowledge_base: KnowledgeBase):
        self.logger = logging.getLogger(__name__)
        self.knowledge_base = knowledge_base

    async def retrieve_documents(self, query: str, include_longterm_memory: bool = True) -> List[Dict[str, Any]]:
        """Retrieve relevant documents based on the query using the graph database, optionally including long-term memory insights."""
        self.logger.info(f"Retrieving documents for query: {query}")
        try:
            # Query the graph database for relevant documents
            documents = await self.knowledge_base.query_insights(query)
            if include_longterm_memory:
                longterm_memory = await self.knowledge_base.get_longterm_memory()
                documents.extend(longterm_memory.get("insights", []))
            self.logger.debug(f"Retrieved documents: {documents}")
            return documents
        except Exception as e:
            self.logger.error(f"Error during document retrieval: {str(e)}")
            return []

    async def augment_prompt_with_retrieval(self, prompt: str, task: str, historical_context: bool = True) -> str:
        """Augment the prompt with retrieved documents and historical context."""
        documents = await self.retrieve_documents(prompt)
        augmented_prompt = prompt + "\n\n### Retrieved Documents:\n"
        for doc in documents:
            augmented_prompt += f"- **{doc['title']}**: {doc['content']}\n"
        
        if historical_context:
            historical_data = await self.knowledge_base.get_entry("historical_context")
            augmented_prompt += "\n\n### Historical Context:\n"
            for entry in historical_data:
                augmented_prompt += f"- **{entry['title']}**: {entry['content']}\n"

        # Log the augmented prompt for debugging
        self.logger.debug(f"Augmented prompt: {augmented_prompt}")
        self.logger.info(f"Augmented prompt for task '{task}': {augmented_prompt}")
        return augmented_prompt
