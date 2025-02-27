"""LangChain agent for Qdrant vector database."""

from typing import List, Dict, Any, Optional, Union
import json
import logging

from langchain.agents import Tool
from langchain.tools import tool
from langchain.agents import AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory

from qdrant_agent.config import config
from qdrant_agent.client import qdrant_client

# Configure logging
logging.basicConfig(level=logging.INFO if config.verbose else logging.WARNING)
logger = logging.getLogger(__name__)


class QdrantAgent:
    """Agent for interacting with Qdrant vector database using LangChain."""
    
    def __init__(self, 
                openai_api_key: Optional[str] = None,
                model_name: str = "gpt-3.5-turbo"):
        """Initialize the Qdrant agent.
        
        Args:
            openai_api_key: OpenAI API key. Defaults to config value.
            model_name: LLM model to use. Defaults to "gpt-3.5-turbo".
        """
        self.openai_api_key = openai_api_key or config.openai_api_key
        self.model_name = model_name
        
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        
        self.llm = ChatOpenAI(
            model=self.model_name,
            openai_api_key=self.openai_api_key,
            temperature=0
        )
        
        self._setup_tools()
        self._setup_memory()
        self._setup_prompt()
        self._setup_agent()
    
    def _setup_tools(self):
        """Set up the agent's tools."""
        
        @tool
        def list_collections() -> List[str]:
            """List all collections in the Qdrant database."""
            return qdrant_client.list_collections()
        
        @tool
        def create_collection(collection_name: str, dimension: Optional[int] = None) -> bool:
            """Create a new collection in the Qdrant database.
            
            Args:
                collection_name: Name of the collection to create.
                dimension: Dimension of the embedding vectors. Defaults to config value.
            
            Returns:
                bool: True if the collection was created successfully, False otherwise.
            """
            dimension = dimension or config.embedding_dimension
            return qdrant_client.create_collection(collection_name, dimension)
        
        @tool
        def delete_collection(collection_name: str) -> bool:
            """Delete a collection from the Qdrant database.
            
            Args:
                collection_name: Name of the collection to delete.
            
            Returns:
                bool: True if the collection was deleted successfully, False otherwise.
            """
            return qdrant_client.delete_collection(collection_name)
        
        @tool
        def get_collection_info(collection_name: str) -> Dict[str, Any]:
            """Get information about a collection.
            
            Args:
                collection_name: Name of the collection to get info for.
            
            Returns:
                Dict[str, Any]: Information about the collection.
            """
            return qdrant_client.get_collection_info(collection_name)
        
        @tool
        def add_documents(collection_name: str, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
            """Add documents to a collection.
            
            Args:
                collection_name: Name of the collection to add documents to.
                texts: List of document texts to add.
                metadatas: List of metadata dicts for each document.
            
            Returns:
                List[str]: List of IDs of the added documents.
            """
            return qdrant_client.add_documents(collection_name, texts, metadatas)
        
        @tool
        def similarity_search(collection_name: str, query: str, k: int = 5) -> List[Dict[str, Any]]:
            """Perform a similarity search on a collection.
            
            Args:
                collection_name: Name of the collection to search.
                query: Query text.
                k: Number of results to return.
            
            Returns:
                List[Dict[str, Any]]: List of search results with document text and metadata.
            """
            return qdrant_client.similarity_search(collection_name, query, k)
        
        self.tools = [
            list_collections,
            create_collection,
            delete_collection,
            get_collection_info,
            add_documents,
            similarity_search
        ]
    
    def _setup_memory(self):
        """Set up the agent's memory."""
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
    
    def _setup_prompt(self):
        """Set up the agent's prompt."""
        system_message = """You are an expert assistant for working with Qdrant vector database.
You help users manage collections, add documents, and perform similarity searches.

Follow these guidelines:
1. For collection operations, always check if the collection exists first.
2. When adding documents, ensure the collection exists or suggest creating it.
3. For similarity searches, provide clear explanations of the results.
4. Always format your responses in a clear, readable way.
5. If you encounter errors, explain them in simple terms and suggest solutions.

Available tools:
- list_collections: List all collections in the database
- create_collection: Create a new collection
- delete_collection: Delete a collection
- get_collection_info: Get information about a collection
- add_documents: Add documents to a collection with automatic embedding
- similarity_search: Search for similar documents in a collection

If users ask about vector embeddings, explain that they are automatically handled by the agent using OpenAI's embedding models.
"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
    
    def _setup_agent(self):
        """Set up the agent executor."""
        self.agent = (
            {
                "input": lambda x: x["input"],
                "chat_history": lambda x: x["chat_history"],
                "agent_scratchpad": lambda x: format_to_openai_function_messages(
                    x["intermediate_steps"]
                )
            }
            | self.prompt
            | self.llm.bind(functions=[t.to_openai_function() for t in self.tools])
            | OpenAIFunctionsAgentOutputParser()
        )
        
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=config.verbose,
            handle_parsing_errors=True
        )
    
    def run(self, query: str) -> str:
        """Run the agent on a query.
        
        Args:
            query: Query from the user.
        
        Returns:
            str: Agent's response.
        """
        try:
            result = self.agent_executor.invoke({"input": query})
            return result["output"]
        except Exception as e:
            logger.error(f"Error running agent: {e}")
            return f"Error: {str(e)}"
    
    def reset(self):
        """Reset the agent's memory."""
        self.memory.clear()
