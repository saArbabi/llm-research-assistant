from rich.console import Console
from rich.panel import Panel

from tasks.generate_query import QueryResponse, generate_search_query, mock_search_query

console = Console()


class Agent:
    def __init__(self, query: str):
        self.query = query

    async def run(self) -> str:
        query_response = await self.generate_queries()
        return query_response

    async def generate_queries(self) -> QueryResponse:
        with console.status("[bold cyan]Analyzing query...[/bold cyan]") as status:

            # result = await mock_search_query()
            result = await generate_search_query(self.query)

            # Display the results
            console.print(Panel("[bold cyan]Query Generation[/bold cyan]"))
            console.print(f"[yellow]Thoughts:[/yellow] {result.thoughts}")
            console.print("\n[yellow]Generated Search Queries:[/yellow]")
            for i, query in enumerate(result.queries, 1):
                console.print(f"  {i}. {query}")

            return result
