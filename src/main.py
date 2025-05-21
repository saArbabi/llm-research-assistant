import asyncio

# from coordinator import ResearchCoordinator
from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

from agent import Agent

load_dotenv()

console = Console()


async def main() -> None:
    console.print("[bold cyan]Research Tool[/bold cyan]")
    console.print(
        "This tool generates research ideas, assesses novelty of ideas w.r.t literature, and suggests "
        "experiments to validate a research idea, performs research on any topic using AI agents."
    )

    # get the users query
    query = Prompt.ask("\n[bold]What would you like to research?[/bold]")

    if not query.strip():
        console.print("[bold red]Error:[/bold red] Please provide a valid query.")
        return

    llm_agent = Agent(query)
    ans = await llm_agent.run()


if __name__ == "__main__":
    asyncio.run(main())
