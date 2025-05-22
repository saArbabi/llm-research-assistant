from loguru import logger
from rich.console import Console
from rich.panel import Panel

from tasks.generate_ideas import LLMSolutionResponse, generate_search_ideas
from tasks.generate_query import QueryResponse, generate_search_query
from tasks.literature_search import perform_literature_search
from tasks.similarity_search import search

logger.remove()
logger.add(
    sink=lambda message: print(message), format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

console = Console()


class Agent:
    def __init__(self, query: str):
        self.query = query

    async def run(self) -> str:
        query_response = await self.generate_queries()
        search_results = await self.perform_literature_search(query_response.queries)
        await self.perform_similarity_search(search_results)
        await self.generate_solutions()

        return query_response

    async def generate_queries(self) -> QueryResponse:
        with console.status("[bold cyan]Analyzing query...[/bold cyan]") as status:

            result = await generate_search_query(self.query)

            # Display the results
            console.print(Panel("[bold cyan]Query Generation[/bold cyan]"))
            console.print(f"[yellow]Thoughts:[/yellow] {result.thoughts}")
            console.print("\n[yellow]Generated Search Queries:[/yellow]")
            for i, query in enumerate(result.queries, 1):
                console.print(f"  {i}. {query}")

            return result

    async def perform_literature_search(self, queries: list[str]) -> None:
        search_results = perform_literature_search(queries)
        if search_results:
            logger.success(f"Scraped a total of {len(search_results)} abstracts.")
        else:
            raise ValueError("No abstracts were scraped, try again.")

        return search_results

    async def perform_similarity_search(
        self, llm_search_results: list[str], llm_ideas: list[str] = [], top_k=5
    ) -> None:
        """Compares every llm generated idea with the searched literature.
        Returns a list of closest abstract matches.
        """
        llm_ideas = llm_search_results[:3]
        # for llm_idea in llm_ideas:
        search(llm_ideas[0]["abstract"], llm_search_results, top_k=10)

    async def generate_solutions(self) -> LLMSolutionResponse:
        result = await generate_search_ideas(self.query)
        # Display the results
        console.print(Panel("[bold cyan]Solution Generation[/bold cyan]"))
        console.print(f"[yellow]Proposed solution:[/yellow] {result.idea}")
        console.print(f"[yellow]Abstract:[/yellow] {result.abstract}")
