"""Qdrant client implementation for the agent."""

from typing import Optional, List, Dict, Any, Union
import logging

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.http.exceptions import UnexpectedResponse
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant

from qdrant_agent.config import config

# Configure logging
logging.basicConfig(level=logging.INFO if config.verbose else logging.WARNING)
logger = logging.getLogger(__name__)


class QdrantAgentClient:
    """Client for interacting with Qdrant."""
    
    def __init__(self, 
                qdrant_url: Optional[str] = None, 
                qdrant_api_key: Optional[str] = None,
                openai_api_key: Optional[str] = None,
                embedding_model: Optional[str] = None):
        """Initialize the Qdrant client.
        
        Args:
            qdrant_url: URL of the Qdrant server. Defaults to config value.
            qdrant_api_key: API key for Qdrant Cloud. Defaults to config value.
            openai_api_key: OpenAI API key. Defaults to config value.
            embedding_model: Embedding model to use. Defaults to config value.
        """
        self.qdrant_url = qdrant_url or config.qdrant_url
        self.qdrant_api_key = qdrant_api_key or config.qdrant_api_key
        self.openai_api_key = openai_api_key or config.openai_api_key
        self.embedding_model = embedding_model or config.embedding_model
        
        # Initialize Qdrant client
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key
        )
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            openai_api_key=self.openai_api_key
        )
        
        logger.info(f"Connected to Qdrant at {self.qdrant_url}")
    
    def list_collections(self) -> List[str]:
        """List all collections in Qdrant.
        
        Returns:
            List[str]: List of collection names.
        """
        try:
            collections = self.client.get_collections().collections
            return [collection.name for collection in collections]
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            raise
    
    def collection_exists(self, collection_name: str) -> bool:
        """Check if a collection exists.
        
        Args:
            collection_name: Name of the collection to check.
            
        Returns:
            bool: True if the collection exists, False otherwise.
        """
        try:
            collections = self.list_collections()
            return collection_name in collections
        except Exception as e:
            logger.error(f"Error checking if collection exists: {e}")
            raise
    
    def create_collection(self, 
                         collection_name: str, 
                         dimension: Optional[int] = None,
                         distance: str = "Cosine") -> bool:
        """Create a new collection in Qdrant.
        
        Args:
            collection_name: Name of the collection to create.
            dimension: Dimension of the embedding vectors. Defaults to config value.
            distance: Distance function to use. Defaults to "Cosine".
            
        Returns:
            bool: True if the collection was created successfully, False otherwise.
        """
        if self.collection_exists(collection_name):
            logger.warning(f"Collection {collection_name} already exists")
            return False
        
        dimension = dimension or config.embedding_dimension
        
        try:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=rest.VectorParams(
                    size=dimension,
                    distance=rest.Distance[distance]
                )
            )
            logger.info(f"Created collection {collection_name} with dimension {dimension}")
            return True
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def delete_collection(self, collection_name: str) -> bool:
        """Delete a collection from Qdrant.
        
        Args:
            collection_name: Name of the collection to delete.
            
        Returns:
            bool: True if the collection was deleted successfully, False otherwise.
        """
        if not self.collection_exists(collection_name):
            logger.warning(f"Collection {collection_name} does not exist")
            return False
        
        try:
            self.client.delete_collection(collection_name=collection_name)
            logger.info(f"Deleted collection {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """Get information about a collection.
        
        Args:
            collection_name: Name of the collection to get info for.
            
        Returns:
            Dict[str, Any]: Information about the collection.
        """
        if not self.collection_exists(collection_name):
            logger.warning(f"Collection {collection_name} does not exist")
            return {}
        
        try:
            info = self.client.get_collection(collection_name=collection_name)
            return info.dict()
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            raise
    
    def add_documents(self, 
                    collection_name: str, 
                    texts: List[str], 
                    metadatas: Optional[List[Dict[str, Any]]] = None) -> List[str]:
        """Add documents to a collection.
        
        Args:
            collection_name: Name of the collection to add documents to.
            texts: List of document texts to add.
            metadatas: List of metadata dicts for each document.
            
        Returns:
            List[str]: List of IDs of the added documents.
        """
        if not self.collection_exists(collection_name):
            logger.warning(f"Collection {collection_name} does not exist, creating it")
            self.create_collection(collection_name)
        
        try:
            langchain_qdrant = Qdrant(
                client=self.client,
                collection_name=collection_name,
                embeddings=self.embeddings
            )
            
            ids = langchain_qdrant.add_texts(texts=texts, metadatas=metadatas)
            logger.info(f"Added {len(texts)} documents to collection {collection_name}")
            return ids
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def similarity_search(self, 
                         collection_name: str, 
                         query: str, 
                         k: int = 5) -> List[Dict[str, Any]]:
        """Perform a similarity search on a collection.
        
        Args:
            collection_name: Name of the collection to search.
            query: Query text.
            k: Number of results to return.
            
        Returns:
            List[Dict[str, Any]]: List of search results with document text and metadata.
        """
        if not self.collection_exists(collection_name):
            logger.error(f"Collection {collection_name} does not exist")
            raise ValueError(f"Collection {collection_name} does not exist")
        
        try:
            langchain_qdrant = Qdrant(
                client=self.client,
                collection_name=collection_name,
                embeddings=self.embeddings
            )
            
            results = langchain_qdrant.similarity_search_with_score(query, k=k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Error performing similarity search: {e}")
            raise


# Create a global client instance
qdrant_client = QdrantAgentClient()
