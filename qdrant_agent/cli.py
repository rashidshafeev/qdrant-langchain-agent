"""Command-line interface for the Qdrant agent."""

import sys
import json
import logging
from typing import List, Dict, Any, Optional
import os

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn

from qdrant_agent.config import config
from qdrant_agent.client import qdrant_client
from qdrant_agent.agent import QdrantAgent

# Configure logging
logging.basicConfig(level=logging.INFO if config.verbose else logging.WARNING)
logger = logging.getLogger(__name__)

# Initialize rich console
console = Console()


@click.group()
@click.version_option()
def cli():
    """Qdrant LangChain Agent - Interact with Qdrant vector database."""
    # Check if OpenAI API key is set
    if not config.openai_api_key:
        console.print(
            "[bold red]Error:[/bold red] OpenAI API key not found. "
            "Please set the OPENAI_API_KEY environment variable or add it to your .env file."
        )
        sys.exit(1)


@cli.command("list_collections")
def list_collections_cmd():
    """List all collections in the Qdrant database."""
    with console.status("[bold blue]Connecting to Qdrant...[/bold blue]"):
        try:
            collections = qdrant_client.list_collections()
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            sys.exit(1)
    
    if not collections:
        console.print("[yellow]No collections found[/yellow]")
        return
    
    # Create a table
    table = Table(title="Qdrant Collections")
    table.add_column("Collection Name", style="cyan")
    
    for collection in collections:
        table.add_row(collection)
    
    console.print(table)


@cli.command("create_collection")
@click.option("--name", "-n", required=True, help="Name of the collection to create")
@click.option("--dimension", "-d", type=int, default=None, help="Dimension of the vectors")
def create_collection_cmd(name: str, dimension: Optional[int]):
    """Create a new collection in the Qdrant database."""
    dimension = dimension or config.embedding_dimension
    
    with console.status(f"[bold blue]Creating collection {name}...[/bold blue]"):
        try:
            created = qdrant_client.create_collection(name, dimension)
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            sys.exit(1)
    
    if created:
        console.print(f"[bold green]Collection {name} created successfully[/bold green]")
    else:
        console.print(f"[yellow]Collection {name} already exists[/yellow]")


@cli.command("delete_collection")
@click.option("--name", "-n", required=True, help="Name of the collection to delete")
@click.option("--force", "-f", is_flag=True, help="Force deletion without confirmation")
def delete_collection_cmd(name: str, force: bool):
    """Delete a collection from the Qdrant database."""
    if not force:
        if not click.confirm(f"Are you sure you want to delete collection '{name}'?"):
            console.print("[yellow]Operation cancelled[/yellow]")
            return
    
    with console.status(f"[bold blue]Deleting collection {name}...[/bold blue]"):
        try:
            deleted = qdrant_client.delete_collection(name)
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            sys.exit(1)
    
    if deleted:
        console.print(f"[bold green]Collection {name} deleted successfully[/bold green]")
    else:
        console.print(f"[yellow]Collection {name} not found[/yellow]")


@cli.command("collection_info")
@click.option("--name", "-n", required=True, help="Name of the collection")
def collection_info_cmd(name: str):
    """Get information about a collection."""
    with console.status(f"[bold blue]Getting info for collection {name}...[/bold blue]"):
        try:
            info = qdrant_client.get_collection_info(name)
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            sys.exit(1)
    
    if not info:
        console.print(f"[yellow]Collection {name} not found[/yellow]")
        return
    
    # Format and display the info
    syntax = Syntax(
        json.dumps(info, indent=2),
        "json",
        theme="monokai",
        line_numbers=True
    )
    console.print(Panel(
        syntax,
        title=f"Collection: {name}",
        border_style="blue"
    ))


@cli.command("add_documents")
@click.option("--collection", "-c", required=True, help="Name of the collection")
@click.option("--source", "-s", required=True, help="Source file (JSON or text)")
@click.option("--field", "-f", default="text", help="Field name for text content if JSON")
@click.option("--batch-size", "-b", type=int, default=None, help="Batch size for processing")
def add_documents_cmd(collection: str, source: str, field: str, batch_size: Optional[int]):
    """Add documents to a collection."""
    batch_size = batch_size or config.batch_size
    
    if not os.path.exists(source):
        console.print(f"[bold red]Error:[/bold red] File {source} not found")
        sys.exit(1)
    
    # Read the source file
    try:
        with open(source, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        console.print(f"[bold red]Error reading file:[/bold red] {str(e)}")
        sys.exit(1)
    
    # Parse content based on file extension
    try:
        if source.endswith(".json"):
            # Parse as JSON
            data = json.loads(content)
            
            # Handle different JSON formats
            if isinstance(data, list):
                if all(isinstance(item, str) for item in data):
                    # List of strings
                    texts = data
                    metadatas = None
                elif all(isinstance(item, dict) for item in data):
                    # List of dictionaries
                    if field not in data[0]:
                        console.print(
                            f"[bold red]Error:[/bold red] Field '{field}' not found in JSON data"
                        )
                        sys.exit(1)
                    
                    texts = [item[field] for item in data]
                    # Remove the text field from metadata
                    metadatas = [{k: v for k, v in item.items() if k != field} for item in data]
                else:
                    console.print(
                        "[bold red]Error:[/bold red] Unsupported JSON format"
                    )
                    sys.exit(1)
            else:
                console.print(
                    "[bold red]Error:[/bold red] JSON must be a list of strings or objects"
                )
                sys.exit(1)
        else:
            # Treat as plain text, split by lines
            texts = [line.strip() for line in content.split("\n") if line.strip()]
            metadatas = None
    except Exception as e:
        console.print(f"[bold red]Error parsing file:[/bold red] {str(e)}")
        sys.exit(1)
    
    # Process documents in batches
    total_count = len(texts)
    added_count = 0
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]Processing documents... {task.completed}/{task.total}"),
        console=console
    ) as progress:
        task = progress.add_task("Adding", total=total_count)
        
        for i in range(0, total_count, batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size] if metadatas else None
            
            try:
                ids = qdrant_client.add_documents(collection, batch_texts, batch_metadatas)
                added_count += len(ids)
                progress.update(task, advance=len(batch_texts))
            except Exception as e:
                console.print(f"[bold red]Error adding batch:[/bold red] {str(e)}")
                sys.exit(1)
    
    console.print(f"[bold green]Added {added_count} documents to collection {collection}[/bold green]")


@cli.command("query")
@click.option("--collection", "-c", required=True, help="Name of the collection")
@click.option("--text", "-t", required=True, help="Query text")
@click.option("--k", "-k", type=int, default=5, help="Number of results to return")
def query_cmd(collection: str, text: str, k: int):
    """Query a collection with similarity search."""
    with console.status(f"[bold blue]Searching collection {collection}...[/bold blue]"):
        try:
            results = qdrant_client.similarity_search(collection, text, k)
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
            sys.exit(1)
    
    if not results:
        console.print("[yellow]No results found[/yellow]")
        return
    
    # Display results
    console.print(f"[bold]Query:[/bold] {text}\n")
    
    for i, result in enumerate(results):
        console.print(f"[bold cyan]Result {i+1}[/bold cyan] [dim](Score: {result['score']:.4f})[/dim]")
        
        # Print metadata if available
        if result.get("metadata"):
            metadata_str = json.dumps(result["metadata"], indent=2)
            console.print(Panel(
                Syntax(metadata_str, "json", theme="monokai"),
                title="Metadata",
                border_style="blue",
                expand=False
            ))
        
        # Print text content
        console.print(Panel(
            result["text"],
            title="Content",
            border_style="green",
            expand=False
        ))
        
        console.print("")  # Add a blank line between results


@cli.command("interactive")
@click.option("--model", "-m", default="gpt-3.5-turbo", help="OpenAI model to use")
def interactive_cmd(model: str):
    """Start an interactive session with the Qdrant agent."""
    console.print(
        Panel(
            "[bold]Qdrant Agent Interactive Mode[/bold]\n"
            "Type your questions about Qdrant or commands for managing collections.\n"
            "Type [bold blue]'exit'[/bold blue] to quit.",
            title="Qdrant Agent",
            border_style="blue"
        )
    )
    
    try:
        agent = QdrantAgent(model_name=model)
    except Exception as e:
        console.print(f"[bold red]Error initializing agent:[/bold red] {str(e)}")
        sys.exit(1)
    
    while True:
        try:
            # Get user input
            query = click.prompt("\n[bold]You[/bold]", prompt_suffix="")
            
            if query.lower() in ("exit", "quit", "q"):
                break
            
            # Process with agent
            with console.status("[bold blue]Thinking...[/bold blue]"):
                response = agent.run(query)
            
            # Display response
            console.print("\n[bold]Agent[/bold]")
            console.print(Markdown(response))
            
        except KeyboardInterrupt:
            console.print("\n[yellow]Session terminated by user[/yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
    
    console.print("[bold blue]Goodbye![/bold blue]")


if __name__ == "__main__":
    cli()
