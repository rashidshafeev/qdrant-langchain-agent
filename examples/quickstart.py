#!/usr/bin/env python3
"""Quickstart example for the Qdrant LangChain Agent."""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check for API keys
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please set it in your .env file or as an environment variable.")
    sys.exit(1)

# Import after environment checks
from qdrant_agent.client import QdrantAgentClient
from qdrant_agent.agent import QdrantAgent


def main():
    """Run the quickstart example."""
    print("Qdrant LangChain Agent - Quickstart Example")
    print("===========================================\n")
    
    # Initialize client
    print("Connecting to Qdrant...")
    client = QdrantAgentClient()
    
    # List collections
    print("\nListing collections:")
    collections = client.list_collections()
    print(f"Found {len(collections)} collections: {', '.join(collections) if collections else 'None'}")
    
    # Create a test collection
    collection_name = "quickstart_test"
    print(f"\nCreating collection: {collection_name}")
    client.create_collection(collection_name)
    
    # Add some sample documents
    print("\nAdding sample documents...")
    sample_texts = [
        "Qdrant is a vector similarity search engine. It provides a production-ready service with a convenient API to store, search, and manage vectors with an additional payload.",
        "LangChain is a framework for developing applications powered by language models. It enables applications that are data-aware, agentic, and connect model outputs to real-world actions.",
        "Vector embeddings are numerical representations of data that capture semantic meaning, allowing machines to understand similarities between concepts.",
        "RAG (Retrieval Augmented Generation) is a technique that enhances language models by retrieving relevant information from external sources before generating responses.",
        "Semantic search uses natural language processing to understand the intent and contextual meaning of search queries, going beyond keyword matching."
    ]
    
    sample_metadata = [
        {"source": "qdrant_docs", "category": "vector_database"},
        {"source": "langchain_docs", "category": "framework"},
        {"source": "ml_glossary", "category": "concept"},
        {"source": "ai_techniques", "category": "methodology"},
        {"source": "search_glossary", "category": "concept"}
    ]
    
    ids = client.add_documents(collection_name, sample_texts, sample_metadata)
    print(f"Added {len(ids)} documents")
    
    # Perform a search
    query = "How does semantic search work?"
    print(f"\nPerforming similarity search for: '{query}'")
    results = client.similarity_search(collection_name, query)
    
    print("\nSearch results:")
    for i, result in enumerate(results):
        print(f"\n--- Result {i+1} (Score: {result['score']:.4f}) ---")
        print(f"Text: {result['text']}")
        print(f"Metadata: {result['metadata']}")
    
    # Initialize the agent
    print("\nInitializing the LangChain agent...")
    agent = QdrantAgent()
    
    # Ask the agent a question
    question = "Tell me about the collections in my Qdrant database"
    print(f"\nAsking the agent: '{question}'")
    
    response = agent.run(question)
    print("\nAgent response:")
    print(response)
    
    # Clean up
    print("\nCleaning up (deleting test collection)...")
    client.delete_collection(collection_name)
    print("Done!")


if __name__ == "__main__":
    main()
