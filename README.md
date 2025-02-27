# Qdrant LangChain Agent

A LangChain-based agent for interacting with Qdrant vector database - manage collections, create documents, and perform queries.

## Features

- Create and manage Qdrant collections
- List available collections
- Add documents to collections with automatic embedding
- Perform semantic searches using natural language queries
- Automated setup with virtual environment creation

## Quick Start

### Automated Setup

The easiest way to get started is to use the automated setup script:

```bash
./setup.sh
```

This will:
1. Create a Python virtual environment
2. Install all required dependencies
3. Configure environment variables

### Manual Setup

If you prefer manual setup:

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure environment variables:
   - Copy `.env.example` to `.env`
   - Edit `.env` to include your API keys and configuration

## Usage

```bash
# Activate the virtual environment if not already activated
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run the Qdrant agent
python -m qdrant_agent [command] [options]
```

### Example Commands

```bash
# List all collections
python -m qdrant_agent list_collections

# Create a new collection
python -m qdrant_agent create_collection --name my_collection --dimension 384

# Add documents to a collection
python -m qdrant_agent add_documents --collection my_collection --source documents.json

# Query a collection
python -m qdrant_agent query --collection my_collection --text "Find similar documents about machine learning"
```

## Configuration

Configuration options can be set via:
1. Environment variables
2. `.env` file
3. Command-line arguments

Priority is: command-line arguments > environment variables > `.env` file > defaults.

### Key Configuration Parameters

- `QDRANT_URL`: URL of your Qdrant server (default: http://localhost:6333)
- `QDRANT_API_KEY`: API key for Qdrant Cloud (optional, for cloud deployments)
- `OPENAI_API_KEY`: OpenAI API key for embeddings (required unless using alternative embeddings)
- `EMBEDDING_MODEL`: Embedding model to use (default: OpenAI's text-embedding-ada-002)

## Requirements

- Python 3.8+
- Qdrant server (local or cloud)
- OpenAI API key (for default embeddings)

## License

MIT
