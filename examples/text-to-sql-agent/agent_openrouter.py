"""Text-to-SQL Deep Agent using OpenRouter (free tier)."""

import argparse
import os
import sys
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openrouter import ChatOpenRouter
from rich.console import Console
from rich.panel import Panel

BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from the repo-root .env
load_dotenv(dotenv_path=BASE_DIR.parent.parent / ".env")

console = Console()


def create_sql_deep_agent():
    """Create and return a text-to-SQL Deep Agent using OpenRouter.

    Returns:
        A compiled Deep Agent wired to the Chinook SQLite database.
    """
    # Connect to Chinook database
    db = SQLDatabase.from_uri(
        f"sqlite:///{BASE_DIR / 'chinook.db'}", sample_rows_in_table_info=3
    )

    # Initialize OpenRouter free model. The model can be overridden via the
    # OPENROUTER_MODEL env var. `gpt-oss-20b:free` is a currently available
    # free-tier model that supports tool calling.
    # NOTE: ChatOpenRouter's `timeout` is in milliseconds, not seconds.
    model = ChatOpenRouter(
        model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-20b:free"),
        temperature=0,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        timeout=120_000,
    )

    # Create SQL toolkit and get tools
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    sql_tools = toolkit.get_tools()

    # Create the Deep Agent with all parameters
    return create_deep_agent(
        model=model,
        memory=["./AGENTS.md"],
        skills=["./skills/"],
        tools=sql_tools,
        subagents=[],
        backend=FilesystemBackend(root_dir=str(BASE_DIR)),
    )


def main() -> None:
    """Run the SQL Deep Agent CLI with a natural-language question."""
    parser = argparse.ArgumentParser(
        description="Text-to-SQL Deep Agent powered by LangChain Deep Agents (OpenRouter free tier)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent_openrouter.py "What are the top 5 best-selling artists?"
  python agent_openrouter.py "How many customers are from Canada?"
        """,
    )
    parser.add_argument(
        "query", type=str, help="Natural language question about the database"
    )
    args = parser.parse_args()

    console.print(
        Panel(f"[bold cyan]Query:[/bold cyan] {args.query}", title="Text-to-SQL Agent")
    )

    agent = create_sql_deep_agent()

    console.print("\n[yellow]Agent is thinking...[/yellow]\n")

    try:
        result = agent.invoke({"messages": [{"role": "user", "content": args.query}]})
    except Exception as e:  # noqa: BLE001  # CLI entry point: surface any agent failure to the user
        console.print(Panel(f"[bold red]Error:[/bold red]\n\n{e}", border_style="red"))
        sys.exit(1)

    # Print the final response
    final_message = result["messages"][-1]
    console.print(
        Panel(
            final_message.content,
            title="[bold green]Answer[/bold green]",
            border_style="green",
        )
    )


if __name__ == "__main__":
    main()
