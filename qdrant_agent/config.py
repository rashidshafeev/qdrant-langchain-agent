"""Configuration handler for the Qdrant agent."""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class QdrantAgentConfig(BaseModel):
    """Configuration for the Qdrant agent."""
    
    # Qdrant configuration
    qdrant_url: str = Field(
        default=os.getenv("QDRANT_URL", "http://localhost:6333"),
        description="URL of the Qdrant server"
    )
    qdrant_api_key: Optional[str] = Field(
        default=os.getenv("QDRANT_API_KEY"),
        description="API key for Qdrant Cloud (optional, for cloud deployments)"
    )
    
    # OpenAI configuration
    openai_api_key: str = Field(
        default=os.getenv("OPENAI_API_KEY", ""),
        description="OpenAI API key for embeddings"
    )
    
    # Embedding configuration
    embedding_model: str = Field(
        default=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        description="Embedding model to use"
    )
    embedding_dimension: int = Field(
        default=int(os.getenv("EMBEDDING_DIMENSION", "384")),
        description="Dimension of the embedding vectors"
    )
    
    # Agent configuration
    verbose: bool = Field(
        default=os.getenv("VERBOSE", "True").lower() in ("true", "1", "t"),
        description="Whether to show verbose output"
    )
    
    # Advanced settings
    batch_size: int = Field(
        default=int(os.getenv("BATCH_SIZE", "100")),
        description="Number of documents to process in a batch"
    )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary."""
        return self.model_dump()
    
    def validate_config(self) -> bool:
        """Validate the configuration.
        
        Returns:
            bool: True if the configuration is valid, False otherwise.
        """
        if not self.openai_api_key and self.embedding_model.startswith("text-embedding"):
            return False
            
        return True


# Create a global configuration instance
config = QdrantAgentConfig()
