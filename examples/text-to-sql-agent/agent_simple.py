"""Simple Text-to-SQL agent using OpenRouter free tier (minimal version)."""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openrouter import ChatOpenRouter

BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from the repo-root .env
load_dotenv(dotenv_path=BASE_DIR.parent.parent / ".env")


def create_sql_agent():
    """Create and return a minimal text-to-SQL agent using OpenRouter.

    Uses `langchain.agents.create_agent` instead of the full Deep Agents
    harness, which is lighter-weight for free-tier rate limits.

    Returns:
        A compiled agent wired to the Chinook SQLite database.
    """
    # Connect to database
    db = SQLDatabase.from_uri(
        f"sqlite:///{BASE_DIR / 'chinook.db'}", sample_rows_in_table_info=2
    )

    # Initialize model with longer timeout. The model can be overridden via the
    # OPENROUTER_MODEL env var. `gpt-oss-20b:free` is a currently available
    # free-tier model that supports tool calling.
    # NOTE: ChatOpenRouter's `timeout` is in milliseconds, not seconds.
    model = ChatOpenRouter(
        model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-oss-20b:free"),
        temperature=0,
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        timeout=180_000,
        max_retries=2,
    )

    # Get SQL tools
    toolkit = SQLDatabaseToolkit(db=db, llm=model)
    tools = toolkit.get_tools()

    # Create simple agent (not deep agent - too heavy for free tier)
    return create_agent(model, tools)


def main() -> None:
    """Run the simple SQL agent with a natural-language question."""
    parser = argparse.ArgumentParser(
        description="Minimal text-to-SQL agent using OpenRouter free tier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python agent_simple.py "How many customers are from Canada?"
  python agent_simple.py "What are the top 5 best-selling artists?"
        """,
    )
    parser.add_argument(
        "query", type=str, help="Natural language question about the database"
    )
    args = parser.parse_args()

    print(f"Query: {args.query}\n")

    agent = create_sql_agent()

    try:
        result = agent.invoke({"messages": [{"role": "user", "content": args.query}]})
    except Exception as e:  # noqa: BLE001  # CLI entry point: surface any agent failure to the user
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(result["messages"][-1].content)


if __name__ == "__main__":
    main()
