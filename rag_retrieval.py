import logging
from typing import List, Dict, Any
from knowledge_base import KnowledgeBase

class RAGRetrieval:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def __init__(self, knowledge_base: KnowledgeBase):
        self.logger = logging.getLogger(__name__)
        self.knowledge_base = knowledge_base

    async def retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve relevant documents based on the query using the graph database."""
        self.logger.info(f"Retrieving documents for query: {query}")
        try:
            # Query the graph database for relevant documents
            documents = await self.knowledge_base.query_insights(query)
            self.logger.debug(f"Retrieved documents: {documents}")
            return documents
        except Exception as e:
            self.logger.error(f"Error during document retrieval: {str(e)}")
            return []

    async def augment_prompt_with_retrieval(self, prompt: str, task: str) -> str:
        """Augment the prompt with retrieved documents."""
        documents = await self.retrieve_documents(prompt)
        augmented_prompt = prompt + "\n\n### Retrieved Documents:\n"
        for doc in documents:
            augmented_prompt += f"- **{doc['title']}**: {doc['content']}\n"
        # Log the augmented prompt for debugging
        self.logger.debug(f"Augmented prompt: {augmented_prompt}")
        self.logger.info(f"Augmented prompt for task '{task}': {augmented_prompt}")
        return augmented_prompt
