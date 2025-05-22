from loguru import logger
from rich.console import Console
from rich.panel import Panel

from tasks.generate_query import QueryResponse, generate_search_query, mock_search_query
from tasks.literature_search import save_papers_to_jsonl, search_semantic_scholar
from tasks.similarity_search import get_query_embedding, search

logger.remove()
logger.add(
    sink=lambda message: print(message), format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

console = Console()


def load_jsonl(filename="papers.jsonl"):
    import json

    data = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            data.append(d)
    return data


class Agent:
    def __init__(self, query: str):
        self.query = query

    async def run(self) -> str:
        query_response = await self.generate_queries()
        await self.perform_literature_search(query_response.queries)

        return query_response

    async def generate_queries(self) -> QueryResponse:
        with console.status("[bold cyan]Analyzing query...[/bold cyan]") as status:

            result = await mock_search_query()
            # result = await generate_search_query(self.query)

            # Display the results
            console.print(Panel("[bold cyan]Query Generation[/bold cyan]"))
            console.print(f"[yellow]Thoughts:[/yellow] {result.thoughts}")
            console.print("\n[yellow]Generated Search Queries:[/yellow]")
            for i, query in enumerate(result.queries, 1):
                console.print(f"  {i}. {query}")

            return result

    async def perform_literature_search(self, queries: list[str]) -> None:
        raw_search_results = []
        search_results = []
        # for query in queries:
        #     try:
        # console.print(f"\n[bold cyan]Searching for:[/bold cyan] {query}")
        #         search_results.extend(search_semantic_scholar(query))
        #     except Exception as ex:
        #         console.print(f"[bold red]Search error for query '{query}':[/bold red] {str(ex)}")
        raw_search_results = load_jsonl("data/testing_search.jsonl")
        search_results = []
        for res in raw_search_results:
            if not res["abstract"]:
                continue
            search_results.append(res)

        if search_results:
            logger.success(f"Scraped a total of {len(search_results)} abstracts!")
        else:
            raise ValueError(
                "No abstracts were scraped, try again."
            )

        return search_results

    async def perform_similarity_search(
        self, llm_search_results: list[str], llm_ideas: list[str] = [], top_k=5
    ) -> None:
        """Compares every llm generated idea with the searched literature.
        Returns a list of closest abstract matches.
        """
        llm_ideas = llm_search_results[:3]
        for llm_idea in llm_ideas:
            search(llm_idea["abstract"], index_path="faiss.index", metadata_path="metadata.pkl", top_k=3):
