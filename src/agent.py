from loguru import logger
from rich.console import Console
from rich.panel import Panel

from tasks.generate_ideas import LLMSolutionResponse, generate_search_ideas
from tasks.generate_query import QueryResponse, generate_search_query
from tasks.literature_search import perform_literature_search
from tasks.similarity_search import SimilarityResult, search

logger.remove()
logger.add(
    sink=lambda message: print(message), format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

console = Console()


class Agent:
    def __init__(self, query: str):
        self.query = query

    async def run(self) -> str:
        search_queries = await self.generate_search_queries()
        search_results = await self.perform_literature_search(search_queries.queries)
        llm_solution = await self.generate_solutions()
        most_similar = await self.perform_similarity_search(search_results, llm_solution.abstract)
        return search_queries

    async def generate_search_queries(self) -> QueryResponse:
        with console.status("[bold cyan]Analyzing query...[/bold cyan]") as status:

            result = await generate_search_query(self.query)

            # Display the results
            console.print(Panel("[bold cyan]Query Generation[/bold cyan]"))
            console.print(f"[yellow]Thoughts:[/yellow] {result.thoughts}")
            console.print("\n[yellow]Generated Search Queries:[/yellow]")
            for i, query in enumerate(result.queries, 1):
                console.print(f"  {i}. {query}")

            return result

    async def perform_literature_search(self, queries: list[str]) -> list[dict[str, str]]:
        search_results = perform_literature_search(queries)
        if search_results:
            logger.success(f"Scraped a total of {len(search_results)} abstracts.")
        else:
            raise ValueError("No abstracts were scraped, try again.")

        return search_results

    async def generate_solutions(self) -> LLMSolutionResponse:
        result = await generate_search_ideas(self.query)
        console.print(Panel("[bold cyan]Solution Generation[/bold cyan]"))
        console.print(f"[yellow]Proposed solution:[/yellow] {result.idea}")
        console.print(f"[yellow]Abstract:[/yellow] {result.abstract}")
        return result

    async def perform_similarity_search(
        self, search_results: list[dict[str, str]], llm_proposed_abstract: str, top_k: int = 5
    ) -> list[SimilarityResult]:
        results = search(llm_proposed_abstract, search_results, top_k)
        console.print(Panel("[bold cyan]Similarity Search[/bold cyan]"))
        for i in range(min(3, len(results))):
            console.print(
                f"[yellow]Most similar {i + 1})[/yellow] {results[i].abstract[:200]} ... "
                f"- L2 {results[i].distance}")
        return results
